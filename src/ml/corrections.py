"""Known data corrections, as a table rather than as code.

A correction is a business fact someone established against a source system — a
count that is really a date Excel mangled, a clinical site recorded under the
wrong country, a site that stores cell counts without the 1e6 divisor. Those
facts change when the sources change, so they belong with the data: the
``Corrections`` sheet of the vocabulary workbook on SharePoint, where a data
steward can add one without touching the code. One row per rule, or per pair
for the rules that replace text:

============  ===============================================================
Name          the rule's name; rows sharing a name form one rule
Site          the site's name, as the vocabulary names it
Column        the column it corrects
Action        set | replace_text | replace_values | divide_by
Old           replace_text / replace_values: what is replaced
New           set: the new value (empty → missing) · replace_*: the replacement ·
              divide_by: the divisor
When column   optional: only rows where this column…
When equals   …equals this, compared as text (without When column: the column itself)
Reason        why
Enabled       optional: no / false switches it off
============  ===============================================================

``cleaning.disable`` in config.yaml switches named rules off for a workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .logger import log

MODULE = "clean"

#: The rules applied before values are coerced to their types, and after.
INTEGRITY, SCALE = "integrity", "scale"
_ACTIONS = ("set", "replace_text", "replace_values", "divide_by")
#: The columns of the Corrections sheet.
HEADERS = [
    "Name",
    "Site",
    "Column",
    "Action",
    "Old",
    "New",
    "When column",
    "When equals",
    "Reason",
    "Enabled",
]


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


def _cell(value: Any) -> Any:
    """A cell as the rule means it: blank is None, and 12345.0 is the text "12345"."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value)
    return text if text.strip() else None


def from_frame(frame: pd.DataFrame, source: str = "the corrections table") -> list[Rule]:
    """Read and validate a corrections table: one row per rule, or per replaced pair."""
    headers = {str(c).strip().lower(): c for c in frame.columns}
    absent = [c for c in ("site", "column", "action") if c not in headers]
    if absent:
        raise CorrectionsError(f"{source}: no {', '.join(absent)} column")

    def get(record: dict, column: str) -> Any:
        return _cell(record.get(headers[column])) if column in headers else None

    groups: dict[str, dict] = {}
    order: list[str] = []
    errors: list[str] = []
    for number, row in enumerate(frame.itertuples(index=False), start=2):
        record = dict(zip(frame.columns, row))
        site, column, action = get(record, "site"), get(record, "column"), get(record, "action")
        if not (site or column or action):
            continue
        where = f"{source} row {number}"
        if not site or not column:
            errors.append(f"{where}: Site and Column are required")
            continue
        action = str(action or "").strip().lower()
        if action not in _ACTIONS:
            errors.append(f"{where}: Action '{action}' is not one of {', '.join(_ACTIONS)}")
            continue
        name = get(record, "name") or f"row {number}"
        group = groups.get(name)
        if group is None:
            group = groups[name] = {
                "name": get(record, "name") or "",
                "site": site,
                "column": column,
                "action": action,
                "pairs": [],
                "new": get(record, "new"),
                "when_column": get(record, "when column") or "",
                "when_equals": get(record, "when equals"),
                "reason": get(record, "reason") or "",
                "enabled": str(get(record, "enabled") or "yes").strip().lower()
                not in ("no", "false", "0", "off"),
                "where": where,
            }
            order.append(name)
        elif (group["site"], group["column"], group["action"]) != (site, column, action):
            errors.append(f"{where}: rule '{name}' already has another site, column or action")
            continue
        if action in ("replace_text", "replace_values"):
            old, new = get(record, "old"), get(record, "new")
            if old is None:
                errors.append(f"{where}: {action} needs Old")
                continue
            group["pairs"].append((old, "" if new is None else new))

    rules: list[Rule] = []
    for name in order:
        group = groups[name]
        action, value = group["action"], group["new"]
        if action == "replace_text":
            value = [list(pair) for pair in group["pairs"]]
        elif action == "replace_values":
            value = dict(group["pairs"])
        elif action == "divide_by":
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            if not value:
                errors.append(f"{group['where']}: divide_by needs a non-zero number in New")
                continue
        rules.append(
            Rule(
                site=group["site"],
                column=group["column"],
                action=action,
                value=value,
                name=group["name"],
                when_column=group["when_column"],
                when_equals=group["when_equals"],
                reason=group["reason"],
                enabled=group["enabled"],
            )
        )
    if errors:
        raise CorrectionsError(
            "The corrections table cannot be used:\n  - " + "\n  - ".join(errors)
        )
    return rules


def to_frame(rules: list[Rule]) -> pd.DataFrame:
    """The table for a list of rules, one row per rule or per replaced pair."""
    rows = []
    for rule in rules:
        base = {
            "Name": rule.name,
            "Site": rule.site,
            "Column": rule.column,
            "Action": rule.action,
            "Old": None,
            "New": None,
            "When column": rule.when_column or None,
            "When equals": rule.when_equals,
            "Reason": rule.reason,
            "Enabled": None if rule.enabled else "no",
        }
        if rule.action == "replace_text":
            rows += [dict(base, Old=old, New=new) for old, new in rule.value]
        elif rule.action == "replace_values":
            rows += [dict(base, Old=old, New=new) for old, new in rule.value.items()]
        else:
            rows.append(dict(base, New=rule.value))
    return pd.DataFrame(rows, columns=HEADERS)


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
