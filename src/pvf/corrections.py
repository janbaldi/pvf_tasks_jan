"""Known data corrections, as a table rather than as code.

A correction is a business fact someone established against a source system — a
clump count that is really a date Excel mangled, a clinical site recorded under
the wrong country, a site that stores cell counts without the 1e6 divisor. Those
facts change when the sources change, so they belong in the workspace, next to
the data, in ``corrections.yaml``, where a data steward can add one without
touching the code. Each rule names its site, its column, what it does and why:

.. code-block:: yaml

    corrections:
      - name: clump_count_date_serial
        site: Raritan
        column: "Post-Mixing Clumps: # of Clumps"
        when: {equals: "45691"}              # compared as text
        set: null                            # → missing
        reason: a clump count Excel turned into a date serial
      - name: israel_country
        site: Raritan
        column: Country
        when: {column: Clinical Site, equals: "107306"}
        set: Israel
      - site: Raritan
        column: OOS Type
        replace_text: [[VIability, Viability], [Low Dose, Dose]]   # substrings, in order
      - site: Ghent
        column: Mycoplasma
        replace_values: {mycoplasma not detected: not detected}    # whole values
      - site: Raritan
        column: Total Viable Cells/bag
        divide_by: 1.0e6                     # applied after numeric coercion

A rule without ``when`` applies to every row. ``cleaning.disable`` in the config
switches named rules off for a workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .logger import log

MODULE = "clean"

#: The rules applied before values are coerced to their types, and after.
INTEGRITY, SCALE = "integrity", "scale"
_ACTIONS = ("set", "replace_text", "replace_values", "divide_by")
_KEYS = ("name", "site", "column", "when", "reason", "enabled", *_ACTIONS)


class CorrectionsError(ValueError):
    """A corrections table that cannot be applied, naming the rule at fault."""


@dataclass(frozen=True)
class Rule:
    site: str
    column: str
    action: str
    value: Any
    name: str = ""
    when_column: str = ""
    when_equals: Any = None
    reason: str = ""
    enabled: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def stage(self) -> str:
        return SCALE if self.action == "divide_by" else INTEGRITY

    @property
    def label(self) -> str:
        return self.name or f"{self.site}/{self.column}/{self.action}"


def default_path() -> Path:
    """The corrections shipped with the package, used when a config names none."""
    return Path(str(resources.files("pvf") / "templates" / "corrections.yaml"))


def load(path: str | Path | None) -> list[Rule]:
    """Read and validate a corrections table."""
    path = Path(path) if path else default_path()
    if not path.is_file():
        raise CorrectionsError(f"cleaning.corrections: {path} does not exist")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("corrections") if isinstance(raw, dict) else raw
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise CorrectionsError(f"{path}: expected a 'corrections:' list of rules")

    rules: list[Rule] = []
    errors: list[str] = []
    for index, entry in enumerate(entries):
        where = f"{path.name} corrections[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{where}: expected a mapping")
            continue
        unknown = [key for key in entry if key not in _KEYS]
        if unknown:
            errors.append(f"{where}: unknown keys {unknown} (allowed: {', '.join(_KEYS)})")
        actions = [key for key in _ACTIONS if key in entry]
        if len(actions) != 1:
            errors.append(f"{where}: give exactly one of {', '.join(_ACTIONS)}")
            continue
        if not entry.get("site") or not entry.get("column"):
            errors.append(f"{where}: site and column are required")
            continue
        action = actions[0]
        value = entry[action]
        if action == "replace_text" and not (
            isinstance(value, list) and all(isinstance(p, list) and len(p) == 2 for p in value)
        ):
            errors.append(f"{where}.replace_text: expected a list of [old, new] pairs")
            continue
        if action == "replace_values" and not isinstance(value, dict):
            errors.append(f"{where}.replace_values: expected a mapping of old: new")
            continue
        if action == "divide_by":
            # YAML reads 1.0e6 as text (it wants 1.0e+6), so a numeric string counts.
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            if isinstance(entry[action], bool) or value == 0:
                errors.append(f"{where}.divide_by: expected a non-zero number")
                continue
        when = entry.get("when") or {}
        if when and (not isinstance(when, dict) or "equals" not in when):
            errors.append(
                f"{where}.when: expected {{equals: value}} or {{column: ..., equals: ...}}"
            )
            continue
        rules.append(
            Rule(
                site=str(entry["site"]),
                column=str(entry["column"]),
                action=action,
                value=value,
                name=str(entry.get("name") or ""),
                when_column=str(when.get("column") or ""),
                when_equals=when.get("equals"),
                reason=str(entry.get("reason") or ""),
                enabled=bool(entry.get("enabled", True)),
            )
        )
    if errors:
        raise CorrectionsError(
            "The corrections table cannot be used:\n  - " + "\n  - ".join(errors)
        )
    return rules


def apply(
    phf: dict[str, pd.DataFrame],
    rules: list[Rule],
    stage: str,
    disabled: set[str] | None = None,
    reasons: dict[str, str] | None = None,
) -> list[dict]:
    """Apply one stage's rules to the site frames, in place, and say what each did."""
    disabled = disabled or set()
    reasons = reasons or {}
    changes: list[dict] = []
    for rule in (r for r in rules if r.stage == stage):
        if not rule.enabled or rule.name in disabled:
            log.info(
                MODULE, f"[{rule.site}] {rule.column}: correction '{rule.label}' is switched off"
            )
            continue
        frame = phf.get(rule.site)
        if frame is None or rule.column not in frame.columns:
            log.info(
                MODULE,
                f"[{rule.site}] {rule.column}: not in this run's data — correction "
                f"'{rule.label}' had nothing to do",
            )
            continue
        if rule.when_column:
            if rule.when_column not in frame.columns:
                log.info(
                    MODULE, f"[{rule.site}] {rule.when_column}: absent — '{rule.label}' skipped"
                )
                continue
            mask = frame[rule.when_column].astype(str).str.strip() == str(rule.when_equals)
        elif rule.when_equals is not None:
            mask = frame[rule.column].astype(str).str.strip() == str(rule.when_equals)
        else:
            mask = pd.Series(True, index=frame.index)

        description, rows = _act(frame, rule, mask)
        reason = reasons.get(rule.name) or rule.reason
        changes.append(
            {
                "site": rule.site,
                "column": rule.column,
                "correction": description,
                "rows": rows,
                "why": reason,
                "rule": rule.label,
            }
        )
        log.info(MODULE, f"[{rule.site}] {rule.column}: {description} ({rows} rows)")
    return changes


def _act(frame: pd.DataFrame, rule: Rule, mask: pd.Series) -> tuple[str, int]:
    column = rule.column
    if rule.action == "set":
        value = np.nan if rule.value is None else rule.value
        differs = (frame[column].astype("string") != str(rule.value)).fillna(True)
        rows = int((mask & differs).sum())
        if mask.all():
            frame[column] = value
        else:
            frame.loc[mask, column] = value
        target = "missing" if rule.value is None else f"'{rule.value}'"
        condition = (
            f" where {rule.when_column or column} is '{rule.when_equals}'"
            if rule.when_column or rule.when_equals is not None
            else " on every row"
        )
        return f"set to {target}{condition}", rows
    if rule.action == "replace_text":
        before = frame[column].copy()
        text = frame[column]
        for old, new in rule.value:
            text = text.where(
                ~mask, text.astype("string").str.replace(str(old), str(new), regex=False)
            )
        frame[column] = text.where(text.notna(), before)
        rows = int((before.astype("string") != frame[column].astype("string")).fillna(False).sum())
        return f"{len(rule.value)} text replacements", rows
    if rule.action == "replace_values":
        before = frame[column].copy()
        mapping = {str(k): v for k, v in rule.value.items()}
        replaced = frame[column].map(lambda v: mapping.get(str(v), v) if pd.notna(v) else v)
        frame[column] = replaced.where(mask, before)
        rows = int((before.astype("string") != frame[column].astype("string")).fillna(False).sum())
        pairs = ", ".join(f"'{k}' → '{v}'" for k, v in rule.value.items())
        return f"replaced {pairs}", rows
    # divide_by
    numbers = pd.to_numeric(frame[column], errors="coerce")
    frame[column] = numbers.where(~mask, numbers / float(rule.value))
    return f"divided by {float(rule.value):.0e}", int((mask & numbers.notna()).sum())
