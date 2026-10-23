"""The screening register: parameters left out of every task, and why.

The build's missingness and correlation reports show how each parameter goes
missing, and which rank the batches alike. Deciding that a parameter (or a
group of them) is not fit to predict with, or is represented by another, is a
judgement, and this module keeps those judgements in one place, the register: a
workbook on SharePoint next to the catalog (``sharepoint.screening`` in
config.yaml). The shifts report has no part in it.

The register is a log, not a list. Each row is one decision — exclude or
reinstate one parameter at one site (or ``all``) — with who made it, when, why,
and the evidence it was made on, and the report it was made from: its track
(:data:`TRACKS`). An exclusion can name the parameter that represents it: a
member of a correlated group is left out because another member, which stays
usable, carries its information. Rows are only ever added, by ``ml screening
add``; a change of mind is a new row. The state today is the log replayed in
order: per parameter, site and track, the last decision stands, so a
missingness exclusion and a correlation exclusion stand side by side, and
reinstating one leaves the other. A parameter is excluded when any track
excludes it.

A representative that is itself excluded leaves its group with nothing: that
is a conflict (:func:`conflicts`). ``ml screening add`` refuses to make one;
the build's own exclusions can, and every report, ``ml screening status`` and
every task names it.

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
    "Report",
)
#: Columns a register written before they existed may lack.
OPTIONAL = ("Represented by", "Report")
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
    "Report",
)

EXCLUDE, REINSTATE = "exclude", "reinstate"
ACTIONS = (EXCLUDE, REINSTATE)
ALL = "all"
#: The reason of a correlated group's members, each represented by another member.
CORRELATED = "correlated with group"
#: The reasons whose exclusions need a representative: its evidence group and suggestion columns.
REPRESENTED = {CORRELATED: (CORRELATION_GROUP, SUGGESTED)}
#: The reports a decision is made from. Each keeps its own state, and shows the other's.
MISSINGNESS, CORRELATION = "missingness", "correlation"
TRACKS = (MISSINGNESS, CORRELATION)
#: Why a parameter is left out. Fixed, so decisions can be counted and compared.
REASONS = (
    "never recorded",
    "retired",
    "introduced late",
    "too sparse",
    "irregular",
    "missing with group",
    CORRELATED,
    "data quality",
    "other",
)
#: The reasons each report decides on.
TRACK_REASONS = {
    MISSINGNESS: tuple(r for r in REASONS if r != CORRELATED),
    CORRELATION: (CORRELATED,),
}
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
            "report",
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
    report: str = ""

    @property
    def track(self) -> str:
        """The report the decision belongs to; a row without one, by its reason."""
        return self.report or track_of(self.reason)

    @property
    def label(self) -> str:
        """How a task names the decision it applied: ``S-0003: retired (Ghent)``."""
        instead = f", represented by '{self.represented_by}'" if self.represented_by else ""
        return f"{self.id}: {self.reason} ({self.site}){instead}"


def track_of(reason: str) -> str:
    """The report a reason is decided in."""
    return CORRELATION if reason in TRACK_REASONS[CORRELATION] else MISSINGNESS


def labels(decisions: list[Decision]) -> str:
    """How a task names the decisions that leave one parameter out."""
    return "; ".join(d.label for d in decisions)


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
        elif fields["Report"] and fields["Report"] not in TRACKS:
            errors.append(f"{where}: Report '{fields['Report']}' is not {' or '.join(TRACKS)}")
        elif fields["Report"] and fields["Reason"] not in TRACK_REASONS[fields["Report"]]:
            errors.append(
                f"{where}: Reason '{fields['Reason']}' is not decided in the "
                f"{fields['Report']} report"
            )
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


def current(
    decisions: list[Decision], sites: tuple[str, ...]
) -> dict[tuple[str, str, str], Decision]:
    """The decision that stands per (parameter, site, track): the last one that covers it."""
    state: dict[tuple[str, str, str], Decision] = {}
    for decision in decisions:
        for site in _sites(decision, sites):
            state[(decision.parameter, site, decision.track)] = decision
    return state


def standing(
    decisions: list[Decision], sites: tuple[str, ...]
) -> dict[tuple[str, str], list[Decision]]:
    """The exclusions that stand per (parameter, site), one per track that excludes it."""
    out: dict[tuple[str, str], list[Decision]] = {}
    state = current(decisions, sites)
    for (parameter, site, track), decision in sorted(
        state.items(), key=lambda item: (item[0][0], item[0][1], TRACKS.index(item[0][2]))
    ):
        if decision.action == EXCLUDE:
            out.setdefault((parameter, site), []).append(decision)
    return out


def represented(
    decisions: list[Decision], sites: tuple[str, ...]
) -> dict[tuple[str, str], list[Decision]]:
    """Per (representative, site), the standing exclusions it stands in for."""
    out: dict[tuple[str, str], list[Decision]] = {}
    for (_, site), found in standing(decisions, sites).items():
        for decision in found:
            if decision.represented_by:
                out.setdefault((decision.represented_by, site), []).append(decision)
    return out


def conflicts(decisions: list[Decision], sites: tuple[str, ...]) -> list[dict]:
    """Representatives that are excluded themselves, so their group is left out whole."""
    out = standing(decisions, sites)
    return [
        {
            "Site": site,
            "Representative": parameter,
            "Excluded by": labels(out[(parameter, site)]),
            "Represents": ", ".join(d.parameter for d in members),
            "Decisions": ", ".join(dict.fromkeys(d.id for d in members)),
        }
        for (parameter, site), members in sorted(represented(decisions, sites).items())
        if (parameter, site) in out
    ]


def conflict_line(conflict: dict) -> str:
    """One conflict, in words."""
    return (
        f"'{conflict['Representative']}' at {conflict['Site']} represents "
        f"{conflict['Represents']} ({conflict['Decisions']}) but is excluded itself "
        f"({conflict['Excluded by']}): all of them are left out"
    )


def excluded(
    decisions: list[Decision], sites: tuple[str, ...], cohort: tuple[str, ...] = ()
) -> dict[str, list[Decision]]:
    """Parameter → the decisions that exclude it, for a task over ``cohort``'s sites.

    A task over both sites leaves a parameter out when either site excludes it:
    a predictor that is unusable at one site would teach the model the site.
    """
    out: dict[str, list[Decision]] = {}
    for (parameter, site), found in standing(decisions, sites).items():
        if not cohort or site in cohort:
            out.setdefault(parameter, []).extend(found)
    return out


def as_row(decision: Decision) -> dict[str, str]:
    """A standing exclusion as a report's tables show it."""
    return {
        "Decision": decision.id,
        "Date": decision.date,
        "Report": decision.track,
        "Reason": decision.reason,
        "Rationale": decision.rationale,
        "Represented by": decision.represented_by,
    }


def at_site(
    decisions: list[Decision], sites: tuple[str, ...], site: str, track: str
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Parameter → its standing exclusion at ``site``: the ones ``track``'s report made,
    and the ones the other report made, which it shows but does not change."""
    own: dict[str, dict] = {}
    other: dict[str, dict] = {}
    for (parameter, where), found in standing(decisions, sites).items():
        if where != site:
            continue
        for decision in found:
            (own if decision.track == track else other).setdefault(parameter, as_row(decision))
    return own, other


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

    A row's ``Report`` is the track it decides in; left empty, the report its
    reason belongs to. A reinstatement only undoes an exclusion of its own track.
    Neither order of a conflict is accepted: a representative that is excluded,
    or an exclusion of a parameter that represents others.

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
        track = fields["Report"] or track_of(reason)
        if reason not in REASONS:
            problems.append(f"Reason '{reason}' is not one of: {', '.join(REASONS)}")
        elif track not in TRACKS:
            problems.append(f"Report '{track}' is not {' or '.join(TRACKS)}")
        elif reason not in TRACK_REASONS[track]:
            problems.append(f"Reason '{reason}' is not decided in the {track} report")
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
            here = evidence[evidence["Site"] == site]
            named = here[[c for c in ("Group", CORRELATION_GROUP) if c in here.columns]]
            hit = named.map(lambda cell: fields["Group"] in _text(cell).split("; ")).any(axis=1)
            members = here[hit]["Parameter"].tolist()
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
                if not any(
                    state.get((m, s, track)) and state[(m, s, track)].action == EXCLUDE
                    for s in covered
                )
            ]
            if idle:
                errors.append(
                    f"{where}: nothing to reinstate, not excluded in the {track} report: "
                    f"{', '.join(idle)}"
                )
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
                else ""
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
                    report=track,
                )
            )
    # Checked on the register as the proposal leaves it, so one proposal can move a group
    # to another representative: reinstate it, and exclude the others represented by it.
    before = {(c["Representative"], c["Site"]) for c in conflicts(decisions, sites)}
    errors += [
        f"{conflict_line(c)} — pick another representative, or leave it in"
        for c in conflicts(decisions + added, sites)
        if (c["Representative"], c["Site"]) not in before
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
    A parameter the missingness report already decides on at that site is left
    alone, so a reinstatement stands until someone excludes it again.
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
            and (row["Parameter"], site, MISSINGNESS) not in state
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
                    report=MISSINGNESS,
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
    out = standing(decisions, sites)
    for (parameter, site), decision in (
        (key, d) for key, found in sorted(out.items()) for d in found
    ):
        reasons = []
        instead = decision.represented_by
        if instead and (instead, site) in out:
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
    """The standing exclusions, one row per parameter, site and report."""
    rows = [
        {
            "Parameter": parameter,
            "Site": site,
            "Report": d.track,
            "Decision": d.id,
            "Date": d.date,
            "Reason": d.reason,
            "Rationale": d.rationale,
            "Review by": d.review_by,
            "Represented by": d.represented_by,
        }
        for (parameter, site), found in sorted(standing(decisions, sites).items())
        for d in found
    ]
    return pd.DataFrame(rows)


def task_catalog(
    catalog: pd.DataFrame,
    columns,
    decisions: list[Decision],
    sites: tuple[str, ...],
) -> pd.DataFrame:
    """The catalog rows a task may still pick: in the batch table, and usable at some site.

    ``Usable at`` names the sites that do not exclude the parameter (a task over
    both sites needs both); ``Represents`` the parameters screened out because it
    carries their information, per site.
    """
    excluded = set(standing(decisions, sites))
    represents: dict[str, dict[str, list[str]]] = {}
    for (parameter, site), members in sorted(represented(decisions, sites).items()):
        represents.setdefault(parameter, {})[site] = [d.parameter for d in members]
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


def note(
    parameter: str,
    site: str,
    out: dict[tuple[str, str], list[Decision]],
    stands_for: dict[tuple[str, str], list[Decision]],
) -> str:
    """What the register holds on one parameter at one site, every report's decisions:
    ``excluded S-0004 (missingness: retired); represents A, B (S-0007)``."""
    notes = [
        f"excluded {d.id} ({d.track}: {d.reason}"
        + (f", represented by {d.represented_by}" if d.represented_by else "")
        + ")"
        for d in out.get((parameter, site), [])
    ]
    members = stands_for.get((parameter, site), [])
    if members:
        notes.append(
            f"represents {', '.join(d.parameter for d in members)} "
            f"({', '.join(dict.fromkeys(d.id for d in members))})"
        )
    if members and (parameter, site) in out:
        notes.append("CONFLICT: a representative that is excluded")
    return "; ".join(notes)


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
