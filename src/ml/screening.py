"""The screening register: parameters left out of every task, and why.

The build's missingness, correlation and shifts reports show how each parameter
goes missing, which rank the batches alike, and which move in the same level
groups. Deciding that a parameter (or a group of them) is not fit to predict
with, or is represented by another, is a judgement, and
this module keeps those judgements in one place, the register: a workbook on
SharePoint next to the catalog (``sharepoint.screening`` in config.yaml).

The register is a log, not a list. Each row is one decision — exclude or
reinstate one parameter at one site (or ``all``) — with who made it, when, why,
and the evidence it was made on. An exclusion can name the parameter that
represents it: a member of a correlated group is left out because another
member, which stays usable, carries its information. Rows are only ever added, by ``ml screening
add``; a change of mind is a new row. The state today is the log replayed in
order: per parameter and site, the last decision stands.

Every task applies that state when it picks its predictors, records which
decisions it applied, and keeps a copy of the register in its run folder, so a
rerun applies exactly the same ones.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from . import missingness, share
from .logger import log

MODULE = "screening"

#: The register's sheets: the log, and the digest ``ml screening`` last wrote it with.
SHEET = "Decisions"
CHECK_SHEET = "Check"
#: The build's evidence workbook: every parameter's missingness per site, and the build.
EVIDENCE_FILE = "missingness.xlsx"
EVIDENCE_SHEET = "Parameters"
#: The evidence column naming a parameter's group in the correlation report.
CORRELATION_GROUP = "Correlation group"
#: The evidence column naming the member the correlation report suggests for its group.
SUGGESTED = "Suggested representative"
#: The same two for the shifts report.
SHIFT_GROUP = "Shift group"
SHIFT_SUGGESTED = "Suggested shift representative"
ABOUT_SHEET = "About"

#: The register's columns, in order.
COLUMNS = (
    "ID",
    "Date",
    "Author",
    "Action",
    "Parameter",
    "Site",
    "Group",
    "Reason",
    "Rationale",
    "Missing %",
    "Pattern",
    "Last measured",
    "Batch table",
    "Review by",
    "Represented by",
)
#: Columns a register written before they existed may lack.
OPTIONAL = ("Represented by",)
#: What a proposal holds; ``ml screening add`` fills in the rest.
PROPOSAL_COLUMNS = (
    "Action",
    "Parameter",
    "Group",
    "Site",
    "Reason",
    "Rationale",
    "Author",
    "Represented by",
)

EXCLUDE, REINSTATE = "exclude", "reinstate"
ACTIONS = (EXCLUDE, REINSTATE)
ALL = "all"
#: The reason of a correlated group's members, each represented by another member.
CORRELATED = "correlated with group"
#: The reason of the members of a group that shifts value together, likewise.
SHIFTED = "shifts with group"
#: The reasons whose exclusions need a representative: its evidence group and suggestion columns.
REPRESENTED = {CORRELATED: (CORRELATION_GROUP, SUGGESTED), SHIFTED: (SHIFT_GROUP, SHIFT_SUGGESTED)}
#: Why a parameter is left out. Fixed, so decisions can be counted and compared.
REASONS = (
    "never recorded",
    "retired",
    "introduced late",
    "too sparse",
    "irregular",
    "missing with group",
    CORRELATED,
    SHIFTED,
    "data quality",
    "other",
)
#: A decision is looked at again after this long.
REVIEW_AFTER = timedelta(days=91)
#: The register column behind each Decision field, in order.
FIELDS = dict(
    zip(
        (
            "id",
            "date",
            "author",
            "action",
            "parameter",
            "site",
            "group",
            "reason",
            "rationale",
            "missing",
            "pattern",
            "last_measured",
            "batch_table",
            "review_by",
            "represented_by",
        ),
        COLUMNS,
    )
)
#: A change in missing % this large since the decision is worth a look.
DRIFT_POINTS = 20.0
#: The missingness column that says whether a parameter is measured lately.
RECENT = f"Measured in last {missingness.RECENT} %"


class ScreeningError(ValueError):
    """A register or a proposal that cannot be used, with every problem named."""


@dataclass(frozen=True)
class Decision:
    id: str
    date: str
    author: str
    action: str
    parameter: str
    site: str
    group: str
    reason: str
    rationale: str
    missing: str
    pattern: str
    last_measured: str
    batch_table: str
    review_by: str
    represented_by: str = ""

    @property
    def label(self) -> str:
        """How a task names the decision it applied: ``S-0003: retired (Ghent)``."""
        instead = f", represented by '{self.represented_by}'" if self.represented_by else ""
        return f"{self.id}: {self.reason} ({self.site}){instead}"


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


# ---------------------------------------------------------------------------
# Reading the register, and what it says today
# ---------------------------------------------------------------------------
def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(COLUMNS))


def from_frame(frame: pd.DataFrame, sites: tuple[str, ...], source: str = "") -> list[Decision]:
    """The register's rows as decisions, in log order; refuses a register with errors."""
    missing = [c for c in COLUMNS if c not in frame.columns and c not in OPTIONAL]
    if missing:
        raise ScreeningError(f"{source}: the register has no {', '.join(missing)} column")
    errors: list[str] = []
    decisions = []
    for number, row in enumerate(frame.to_dict("records"), start=2):
        fields = {c: _text(row.get(c)) for c in COLUMNS}
        if not any(fields.values()):
            continue
        where = f"row {number}"
        for column in ("ID", "Date", "Author", "Action", "Parameter", "Site", "Reason"):
            if not fields[column]:
                errors.append(f"{where}: {column} is empty")
        if fields["Action"] and fields["Action"] not in ACTIONS:
            errors.append(f"{where}: Action '{fields['Action']}' is not {' or '.join(ACTIONS)}")
        if fields["Site"] and fields["Site"] not in (ALL, *sites):
            errors.append(f"{where}: Site '{fields['Site']}' is not {ALL} or a site name")
        if fields["Reason"] and fields["Reason"] not in REASONS:
            errors.append(f"{where}: Reason '{fields['Reason']}' is not one of the list")
        decisions.append(Decision(**{f: fields[c] for f, c in FIELDS.items()}))
    if errors:
        raise ScreeningError(f"{source} cannot be used:\n  - " + "\n  - ".join(errors))
    return decisions


def to_frame(decisions: list[Decision]) -> pd.DataFrame:
    rows = [{c: getattr(d, f) for f, c in FIELDS.items()} for d in decisions]
    return pd.DataFrame(rows, columns=list(COLUMNS))


def _canonical(value: str) -> str:
    """A cell as its digest sees it: Excel keeps ``0.0`` as text or as a number, either way."""
    try:
        return _text(float(value))
    except ValueError:
        return value


def digest(decisions: list[Decision]) -> str:
    """A digest of the register's content, so an edit outside ``ml screening`` shows."""
    # An empty optional field is left out, so a register from before it keeps its digest.
    kept = [(f, c not in OPTIONAL) for f, c in FIELDS.items()]
    text = "\n".join(
        "\t".join(_canonical(getattr(d, f)) for f, always in kept if always or getattr(d, f))
        for d in decisions
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sites(decision: Decision, sites: tuple[str, ...]) -> tuple[str, ...]:
    return sites if decision.site == ALL else (decision.site,)


def current(decisions: list[Decision], sites: tuple[str, ...]) -> dict[tuple[str, str], Decision]:
    """The decision that stands per (parameter, site): the last one that covers it."""
    state: dict[tuple[str, str], Decision] = {}
    for decision in decisions:
        for site in _sites(decision, sites):
            state[(decision.parameter, site)] = decision
    return state


def excluded(
    decisions: list[Decision], sites: tuple[str, ...], cohort: tuple[str, ...] = ()
) -> dict[str, Decision]:
    """Parameter → the decision that excludes it, for a task over ``cohort``'s sites.

    A task over both sites leaves a parameter out when either site excludes it:
    a predictor that is unusable at one site would teach the model the site.
    """
    out: dict[str, Decision] = {}
    for (parameter, site), decision in current(decisions, sites).items():
        if decision.action == EXCLUDE and (not cohort or site in cohort):
            out.setdefault(parameter, decision)
    return out


# ---------------------------------------------------------------------------
# Adding decisions
# ---------------------------------------------------------------------------
def evidence_frame(missing_by_site: dict[str, list[dict]]) -> pd.DataFrame:
    """The build's per-site missingness rows as one table: what a decision cites."""
    rows = [
        {"Site": site, **row} for site, site_rows in missing_by_site.items() for row in site_rows
    ]
    return pd.DataFrame(rows)


def _evidence(evidence: pd.DataFrame, parameter: str, sites: tuple[str, ...]) -> dict[str, str]:
    """Missing %, pattern and last measurement of one parameter at ``sites``."""
    found = evidence[(evidence["Parameter"] == parameter) & evidence["Site"].isin(sites)]
    many = len(sites) > 1

    def joined(column: str) -> str:
        return "; ".join(
            f"{row['Site']} {_text(row[column])}" if many else _text(row[column])
            for _, row in found.iterrows()
        )

    return {
        "Missing %": joined("Missing %"),
        "Pattern": joined("Pattern"),
        "Last measured": joined("Last measured"),
    }


def add(
    decisions: list[Decision],
    proposal: pd.DataFrame,
    evidence: pd.DataFrame,
    batch_table: str,
    author: str,
    sites: tuple[str, ...],
    today: date | None = None,
) -> list[Decision]:
    """The decisions a proposal makes, checked against the latest build's evidence.

    One proposal row is one decision ID. A row naming a ``Group`` and no
    ``Parameter`` becomes one decision row per member of that group at that
    site, as the build that wrote ``evidence`` grouped them; the group's label
    is kept as context, since a group's number can change with the next build.
    Rows naming a ``Parameter`` and a ``Group`` (as the missingness report writes
    a group) are one decision when everything else about them is the same.

    A ``Represented by`` names the parameter that stands in for the excluded ones:
    it is never excluded itself, and must be usable at the site. An exclusion for
    a reason in :data:`REPRESENTED` needs one; left empty, it is the build's
    suggestion for the members' group in that report.

    The author is the proposal's ``Author`` column, or ``author`` where it is empty.
    """
    today = today or date.today()
    missing = [c for c in ("Action", "Site", "Reason") if c not in proposal.columns]
    if missing:
        raise ScreeningError(f"The proposal has no {', '.join(missing)} column")
    numbers = [int(d.id.split("-")[-1]) for d in decisions if d.id.split("-")[-1].isdigit()]
    next_number = max(numbers, default=0) + 1
    state = current(decisions, sites)
    errors: list[str] = []
    added: list[Decision] = []
    shared: dict[tuple, str] = {}
    for number, row in enumerate(proposal.to_dict("records"), start=2):
        fields = {c: _text(row.get(c)) for c in PROPOSAL_COLUMNS}
        if not any(fields.values()):
            continue
        where = f"proposal row {number}"
        action, site, reason = fields["Action"], fields["Site"], fields["Reason"]
        problems = []
        if action not in ACTIONS:
            problems.append(f"Action '{action}' is not {' or '.join(ACTIONS)}")
        if site not in (ALL, *sites):
            problems.append(f"Site '{site}' is not {ALL} or one of {', '.join(sites)}")
        if reason not in REASONS:
            problems.append(f"Reason '{reason}' is not one of: {', '.join(REASONS)}")
        if not fields["Parameter"] and not fields["Group"]:
            problems.append("give a Parameter, a Group, or both")
        who = fields["Author"] or author.strip()
        if not who:
            problems.append("say who decides: fill in Author, or pass --author")
        if fields["Group"] and not fields["Parameter"] and site == ALL:
            problems.append("a Group is numbered per site: name the site")
        if problems:
            errors += [f"{where}: {p}" for p in problems]
            continue
        covered = sites if site == ALL else (site,)
        known = set(evidence.loc[evidence["Site"].isin(covered), "Parameter"])
        if fields["Group"] and not fields["Parameter"]:
            at_site = evidence[evidence["Site"] == site]
            named = at_site[
                [c for c in ("Group", CORRELATION_GROUP, SHIFT_GROUP) if c in at_site.columns]
            ]
            # A cell may name several groups (one per plant), joined by "; ".
            hit = named.map(lambda cell: fields["Group"] in _text(cell).split("; ")).any(axis=1)
            members = at_site[hit]["Parameter"].tolist()
            if not members:
                errors.append(
                    f"{where}: the latest build has no group '{fields['Group']}' at {site}"
                )
                continue
        else:
            members = [fields["Parameter"]]
            if fields["Parameter"] not in known:
                errors.append(
                    f"{where}: '{fields['Parameter']}' is not a column of the latest batch table"
                )
                continue
        if action == REINSTATE:
            idle = [
                m
                for m in members
                if not any((state.get((m, s)) and state[(m, s)].action == EXCLUDE) for s in covered)
            ]
            if idle:
                errors.append(f"{where}: nothing to reinstate, not excluded: {', '.join(idle)}")
                continue
        if action == EXCLUDE and reason in REPRESENTED and not fields["Represented by"]:
            found = evidence[evidence["Parameter"].isin(members) & evidence["Site"].isin(covered)]
            group_column, column = REPRESENTED[reason]
            suggested = set()
            for _, row in found.iterrows():
                groups = _text(row.get(group_column)).split("; ")
                for group, suggestion in zip(groups, _text(row.get(column)).split("; ")):
                    if not fields["Group"] or group == fields["Group"]:
                        suggested.add(suggestion)
            if len(suggested - {""}) != 1:
                errors.append(f"{where}: name the parameter that represents it in Represented by")
                continue
            fields["Represented by"] = (suggested - {""}).pop()
        instead = fields["Represented by"]
        if instead:
            members = [m for m in members if m != instead]
            problem = (
                f"'{instead}' is not a column of the latest batch table"
                if instead not in known
                else "it is the only parameter in it: nothing is left to exclude"
                if not members
                else next(
                    (
                        f"'{instead}' is screened out at {s}, so it cannot represent others"
                        for s in covered
                        if state.get((instead, s)) and state[(instead, s)].action == EXCLUDE
                    ),
                    "",
                )
            )
            if problem:
                errors.append(f"{where}: Represented by: {problem}")
                continue
        together = tuple(fields[c] for c in PROPOSAL_COLUMNS if c != "Parameter")
        if fields["Group"] and together in shared:
            identifier = shared[together]
        else:
            identifier = f"S-{next_number:04d}"
            next_number += 1
            shared[together] = identifier
        review = (today + REVIEW_AFTER).isoformat()
        for member in members:
            added.append(
                Decision(
                    id=identifier,
                    date=today.isoformat(),
                    author=who,
                    action=action,
                    parameter=member,
                    site=site,
                    group=fields["Group"],
                    reason=reason,
                    rationale=fields["Rationale"],
                    **dict(
                        zip(
                            ("missing", "pattern", "last_measured"),
                            _evidence(evidence, member, covered).values(),
                        )
                    ),
                    batch_table=batch_table,
                    review_by=review if action == EXCLUDE else "",
                    represented_by=instead if action == EXCLUDE else "",
                )
            )
    out = {(d.parameter, s) for d in added if d.action == EXCLUDE for s in _sites(d, sites)}
    errors += [
        f"{d.id}: '{d.parameter}' is represented by '{d.represented_by}', which the proposal "
        "excludes too"
        for d in added
        if any((d.represented_by, s) in out for s in _sites(d, sites))
    ]
    if errors:
        raise ScreeningError("The proposal was not added:\n  - " + "\n  - ".join(errors))
    if not added:
        raise ScreeningError("The proposal holds no decisions")
    return added


#: Who the register names for the decisions the build makes on its own.
BUILD_AUTHOR = "ml build"


def never_recorded(
    decisions: list[Decision],
    missing_by_site: dict[str, list[dict]],
    batch_table: str,
    sites: tuple[str, ...],
    today: date | None = None,
) -> list[Decision]:
    """The exclusions the build adds on its own: parameters missing in every batch of a site.

    A parameter that is never recorded at a site has nothing to predict with
    there, so it needs no judgement. One decision ID per site covers them all.
    A parameter the register already decides on at that site is left alone, so
    a reinstatement stands until someone excludes it again.
    """
    today = today or date.today()
    state = current(decisions, sites)
    numbers = [int(d.id.split("-")[-1]) for d in decisions if d.id.split("-")[-1].isdigit()]
    next_number = max(numbers, default=0) + 1
    added: list[Decision] = []
    for site, rows in missing_by_site.items():
        found = [
            row
            for row in rows
            if pd.to_numeric(row.get("Missing %"), errors="coerce") == 100
            and (row["Parameter"], site) not in state
        ]
        if not found:
            continue
        identifier = f"S-{next_number:04d}"
        next_number += 1
        for row in found:
            added.append(
                Decision(
                    id=identifier,
                    date=today.isoformat(),
                    author=BUILD_AUTHOR,
                    action=EXCLUDE,
                    parameter=row["Parameter"],
                    site=site,
                    group="",
                    reason="never recorded",
                    rationale=f"missing in every batch at {site}; excluded by the build",
                    missing=_text(row.get("Missing %")),
                    pattern=_text(row.get("Pattern")),
                    last_measured=_text(row.get("Last measured")),
                    batch_table=batch_table,
                    review_by=(today + REVIEW_AFTER).isoformat(),
                )
            )
    return added


# ---------------------------------------------------------------------------
# Looking again
# ---------------------------------------------------------------------------
def review(
    decisions: list[Decision],
    evidence: pd.DataFrame,
    sites: tuple[str, ...],
    today: date | None = None,
) -> list[dict]:
    """Standing exclusions the latest build gives a reason to look at again.

    Nothing here changes a decision: it says which ones to reconsider, and why.
    """
    today = today or date.today()
    findings = []
    state = current(decisions, sites)
    for (parameter, site), decision in sorted(state.items()):
        if decision.action != EXCLUDE:
            continue
        reasons = []
        instead = decision.represented_by
        if instead and state.get((instead, site)) and state[(instead, site)].action == EXCLUDE:
            reasons.append(f"its representative '{instead}' is screened out too")
        if decision.review_by and decision.review_by <= today.isoformat():
            reasons.append(f"review date {decision.review_by} has passed")
        now = evidence[(evidence["Parameter"] == parameter) & (evidence["Site"] == site)]
        if now.empty:
            reasons.append("no longer a column of the batch table")
        else:
            row = now.iloc[0]
            recent = pd.to_numeric(pd.Series([row.get(RECENT)]), errors="coerce").iloc[0]
            if decision.reason in ("never recorded", "retired") and recent and recent > 0:
                reasons.append(f"measured again: {recent:.0f}% of the latest batches")
            then = _then(decision.missing, site)
            rate = pd.to_numeric(pd.Series([row.get("Missing %")]), errors="coerce").iloc[0]
            if then is not None and pd.notna(rate) and abs(rate - then) >= DRIFT_POINTS:
                reasons.append(f"missing {then:.0f}% then, {rate:.0f}% now")
        if reasons:
            findings.append(
                {
                    "Decision": decision.id,
                    "Parameter": parameter,
                    "Site": site,
                    "Why look again": "; ".join(reasons),
                }
            )
    return findings


def _then(recorded: str, site: str) -> float | None:
    """The missing % a decision recorded for one site (``Ghent 80.0; Raritan 12.5``)."""
    for part in recorded.split(";"):
        part = part.strip()
        value = part.rsplit(" ", 1)[-1] if " " in part else part
        if not part or (" " in part and not part.startswith(site)):
            continue
        try:
            return float(value)
        except ValueError:
            return None
    return None


def summary(decisions: list[Decision], sites: tuple[str, ...]) -> pd.DataFrame:
    """The standing exclusions, one row per parameter and site."""
    rows = [
        {
            "Parameter": parameter,
            "Site": site,
            "Decision": d.id,
            "Date": d.date,
            "Reason": d.reason,
            "Rationale": d.rationale,
            "Review by": d.review_by,
            "Represented by": d.represented_by,
        }
        for (parameter, site), d in sorted(current(decisions, sites).items())
        if d.action == EXCLUDE
    ]
    return pd.DataFrame(rows)


def task_catalog(
    catalog: pd.DataFrame,
    columns,
    state: dict[tuple[str, str], Decision],
    sites: tuple[str, ...],
) -> pd.DataFrame:
    """The catalog rows a task may still pick: in the batch table, and usable at some site.

    ``Usable at`` names the sites that do not exclude the parameter (a task over
    both sites needs both); ``Represents`` the parameters screened out because it
    carries their information, per site.
    """
    excluded = {key for key, d in state.items() if d.action == EXCLUDE}
    represents: dict[str, dict[str, list[str]]] = {}
    for (parameter, site), d in sorted(state.items()):
        if (parameter, site) in excluded and d.represented_by:
            represents.setdefault(d.represented_by, {}).setdefault(site, []).append(parameter)
    rows = catalog[catalog["Parameter"].isin(set(columns))].copy()
    usable = rows["Parameter"].map(lambda p: [s for s in sites if (p, s) not in excluded])
    rows.insert(1, "Usable at", usable.map(", ".join))
    rows.insert(
        2,
        "Represents",
        rows["Parameter"].map(
            lambda p: "; ".join(f"{s}: {', '.join(ps)}" for s, ps in represents.get(p, {}).items())
        ),
    )
    return rows[usable.map(bool)].reset_index(drop=True)


def status_line(decisions: list[Decision], sites: tuple[str, ...]) -> str:
    standing = summary(decisions, sites)
    count = standing["Parameter"].nunique() if not standing.empty else 0
    last = decisions[-1].id if decisions else "none"
    noun = "parameter" if count == 1 else "parameters"
    return f"{count} {noun} excluded at some site; {len(decisions)} rows, last decision {last}"


def note(decision: Decision | None) -> str:
    """The missingness report's Screening cell for one parameter at one site."""
    if decision is None:
        return ""
    return f"{decision.action}d {decision.id}" if decision.action == EXCLUDE else ""


def log_added(added: list[Decision]) -> None:
    for identifier in dict.fromkeys(d.id for d in added):
        rows = [d for d in added if d.id == identifier]
        first = rows[0]
        log.success(
            MODULE,
            f"{identifier}: {first.action} {len(rows)} "
            f"parameter{'' if len(rows) == 1 else 's'} at {first.site}"
            f" — {first.reason}",
            ", ".join(d.parameter for d in rows[:10]),
        )


# ---------------------------------------------------------------------------
# The register on SharePoint, and a run's copy of it
# ---------------------------------------------------------------------------
@dataclass
class Register:
    decisions: list[Decision]
    where: str
    #: False when the content does not match the digest ``ml screening`` wrote it with.
    intact: bool = True


def read(location, sites: tuple[str, ...]) -> Register:
    """The register at a SharePoint location; refuses one that does not parse."""
    frame = share.read_excel(location.with_sheet(SHEET))
    decisions = from_frame(frame, sites, str(location))
    try:
        check = share.read_excel(location.with_sheet(CHECK_SHEET))
        written = _text(check.iloc[0]["Digest"]) if len(check) else ""
    except Exception:  # a register written by hand has no check sheet
        written = ""
    register = Register(decisions, str(location), intact=written == digest(decisions))
    if not register.intact:
        log.warn(
            MODULE,
            "The screening register was changed outside `ml screening`: its rows no longer "
            "match the digest it was last written with",
            str(location),
        )
    return register


def read_copy(path, sites: tuple[str, ...]) -> Register:
    """A run's own copy of the register, which a rerun applies."""
    frame = pd.read_excel(path, sheet_name=SHEET)
    return Register(from_frame(frame, sites, str(path)), str(path))


def write(decisions: list[Decision], path) -> None:
    """The register as a workbook: the log, and the digest it was written with."""
    with pd.ExcelWriter(path) as workbook:
        to_frame(decisions).to_excel(workbook, sheet_name=SHEET, index=False)
        pd.DataFrame(
            [
                {
                    "Rows": len(decisions),
                    "Last decision": decisions[-1].id if decisions else "",
                    "Written": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "Digest": digest(decisions),
                }
            ]
        ).to_excel(workbook, sheet_name=CHECK_SHEET, index=False)
