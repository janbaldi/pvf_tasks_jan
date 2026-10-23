"""Assembling a run into the report payload.

One report per stage, built the same way. The payload is an ordered list of
sections, each an ordered list of blocks. This module writes no markup and draws
no figures: :mod:`ml.blocks` says what a block is, :mod:`ml.plots` builds the
figures, :mod:`ml.streamlit_app` renders the payload in the browser and
:mod:`ml.stlite` wraps that page into one file. Nothing imports back the other
way.

Each stage answers the question it is for. The catalog report: what do the sources
record that the schema has never been told about. The build report: what came
out, what the data needed doing to it, where every parameter came from, and which
catalog parameters the batch table still does not have. The build's missingness,
correlation and shifts reports: which parameters go missing together, rank the
batches alike, or run in groups over time, and what to screen.
The tasks report: which batches are in the cohort, what each parameter became,
and what every column of the dataset means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import correlation, decorrelate, missingness, plots, screening, shifts, stlite
from .blocks import (
    Block,
    bold,
    chooser,
    clean,
    code,
    collapsed,
    csv_text,
    deferred,
    download,
    esc,
    md,
    missing_viewer,
    num,
    sortable,
    table,
)
from .blocks import (
    frame as frame_block,
)
from .features import BLOCKED, CREATED, NOT_IN_CATALOG, FeatureReport, Params
from .names import site as site_name

DERIVED = "Derived"


@dataclass
class BuildRun:
    """Everything the build's report needs, collected as the run goes."""

    result: pd.DataFrame
    phf: dict[str, pd.DataFrame]
    site_shapes: dict[str, tuple[int, int]]
    catalog_cols: list[str]
    catalog_mapping: dict[str, str]
    cleaning: Any
    features: dict[str, FeatureReport]
    merge: Any
    events: list
    params: Params
    provenance: dict
    origins: dict[str, list[str]] = field(default_factory=dict)
    lv_coa_coverage: dict[str, dict] = field(default_factory=dict)
    raw_material_columns: list[str] = field(default_factory=list)
    unmapped_sites: list[str] = field(default_factory=list)
    source_rows: list[list[str]] = field(default_factory=list)
    run_start: datetime = field(default_factory=datetime.now)
    #: site → :func:`ml.missingness.for_build` over its batches: one row per column, the groups
    missingness: dict[str, tuple[list[dict], Any]] = field(default_factory=dict)
    #: site → :func:`ml.correlation.for_build` over the same batches
    correlation: dict[str, tuple[list[dict], Any]] = field(default_factory=dict)
    #: site → :func:`ml.shifts.for_build` over the same batches
    shifts: dict[str, tuple[list[dict], Any]] = field(default_factory=dict)
    #: Which screening register the missingness and correlation reports show, or why none.
    register_note: str = ""


def _section(section_id: str, title: str, blocks: list[Block | None]) -> dict | None:
    kept = clean(blocks)
    return {"id": section_id, "title": title, "blocks": kept} if kept else None


def _entries(run: BuildRun) -> list[dict]:
    """Every registry entry from every site, in one list."""
    return [
        dict(entry, site=site) for site, report in run.features.items() for entry in report.entries
    ]


def _status_of(run: BuildRun, name: str) -> dict[str, dict]:
    """What each site's registry did with one feature name."""
    return {
        site: entry
        for site, report in run.features.items()
        for entry in report.entries
        if entry["name"] == name
    }


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _overview(run: BuildRun) -> dict | None:
    """The scoreboard, and nothing else. No prose above the numbers."""
    levels = [event.level for event in run.events]
    created = sum(report.count(CREATED) for report in run.features.values())
    blocked = sum(report.count(BLOCKED) for report in run.features.values())
    absent = sum(report.count(NOT_IN_CATALOG) for report in run.features.values())

    rows = [
        ["Batches in the batch table", f"{run.merge.total_rows:,}"],
        ["Parameters in the batch table", f"{run.merge.total_columns:,}"],
        ["Catalog parameters", f"{len(run.catalog_cols):,}"],
        ["Catalog parameters the batch table does not have", f"{len(run.merge.catalog_missing):,}"],
        ["Features computed", f"{created:,}"],
        ["Features blocked by a missing input", f"{blocked:,}"],
        ["Features waiting on the catalog", f"{absent:,}"],
        ["Warnings", f"{levels.count('WARN'):,}"],
        ["Errors", f"{levels.count('ERROR'):,}"],
    ]
    return _section(
        "overview",
        "Overview",
        [table(["", ""], [[bold(label), value] for label, value in rows])],
    )


def _sources(run: BuildRun) -> dict | None:
    rows = [[esc(a), b, c, d] for a, b, c, d in run.source_rows]
    for site, (n_rows, n_columns) in run.site_shapes.items():
        rows.append(
            [f"{esc(site)} batch records, as loaded", f"{n_rows:,}", f"{n_columns:,}", "loaded"]
        )
    merged = [[esc(site), f"{count:,}"] for site, count in run.merge.site_rows.items()] + [
        [bold("Merged batch table"), bold(f"{run.merge.total_rows:,}")]
    ]

    return _section(
        "sources",
        "Sources",
        [
            table(["Source", "Rows", "Columns", "Status"], rows),
            table(
                ["Site", "Batches in the batch table"],
                merged,
                note=f"{run.merge.total_columns:,} parameters wide after the merge.",
            ),
        ],
    )


def _cleaning(run: BuildRun) -> dict | None:
    report = run.cleaning
    modified = report.string_fixes.get("cells_modified", 0)
    blocks: list[Block | None] = [
        md(
            f"Whitespace and non-breaking spaces were stripped from "
            f"**{modified:,}** cells, and empty strings were turned into missing "
            "values rather than left as text."
        )
    ]

    changes = list(report.integrity_changes) + list(getattr(report, "scale_changes", []))
    source = report.settings.get("corrections", "")
    if changes:
        blocks.append(
            collapsed(
                f"Data corrections ({len(changes)})",
                [
                    table(
                        ["Site", "Parameter", "Correction", "Rows", "Why"],
                        [
                            [
                                esc(c["site"]),
                                esc(c["column"]),
                                esc(c["correction"]),
                                esc(c.get("rows", "")),
                                esc(c.get("why", "")),
                            ]
                            for c in changes
                        ],
                        note="Established corrections against the source systems, not general "
                        f"truths. They are the rules in {source or 'the corrections table'}; "
                        "cleaning.disable in the config switches named rules off"
                        + (
                            " (off in this run: "
                            + ", ".join(report.settings["disabled_corrections"])
                            + ")"
                            if report.settings.get("disabled_corrections")
                            else ""
                        )
                        + ".",
                    )
                ],
            )
        )

    if report.censored:
        blocks.append(
            table(
                ["Site", "Parameter", "Censored values", "What they became", "Examples"],
                [
                    [
                        esc(c["site"]),
                        esc(c["column"]),
                        f"{c['values']:,}",
                        esc(c["meaning"]),
                        esc(", ".join(map(str, c["examples"]))),
                    ]
                    for c in report.censored
                ],
                note="A value written '<0.5' is below the assay's limit. That is not the same "
                "as 0.5 and not the same as missing, so the policy is recorded with it."
                + (
                    f" Each such column also has a '<name> censored' indicator "
                    f"({len(report.censored_flags)} added), as cleaning.censored_flags asks."
                    if getattr(report, "censored_flags", None)
                    else ""
                ),
            )
        )

    notable = [
        c
        for c in report.numeric_coercions
        if "stripped" in c.get("action", "") or "NaN" in c.get("action", "")
    ]
    if notable:
        blocks.append(
            collapsed(
                f"Numeric columns that lost something in coercion ({len(notable)})",
                [
                    table(
                        ["Site", "Parameter", "What happened"],
                        [[esc(c["site"]), esc(c["column"]), esc(c["action"])] for c in notable],
                    )
                ],
            )
        )
    else:
        blocks.append(
            md("Every numeric column coerced cleanly: no inequality operators and no lost values.")
        )

    for label, failures, columns in (
        ("Dates that would not parse", report.datetime_failures, ("count", "sample")),
        ("Durations that would not parse", report.duration_failures, ("count", "sample")),
        ("Percentages outside [0, 100]", report.pct_out_of_range, ("count", "values")),
    ):
        if failures:
            blocks.append(
                collapsed(
                    f"{label} ({len(failures)} parameters)",
                    [
                        table(
                            ["Site", "Parameter", "Rows", "Values"],
                            [
                                [
                                    esc(f["site"]),
                                    esc(f["column"]),
                                    f"{f[columns[0]]:,}",
                                    esc(str(f[columns[1]])[:100]),
                                ]
                                for f in failures
                            ],
                        )
                    ],
                )
            )

    if report.yesno_imputations:
        blocks.append(
            collapsed(
                f"Yes/no blanks filled, where config said what one means "
                f"({len(report.yesno_imputations)})",
                [
                    table(
                        ["Site", "Parameter", "Rows filled", "Value used"],
                        [
                            [esc(c["site"]), esc(c["column"]), f"{c['imputed']:,}", esc(c["value"])]
                            for c in report.yesno_imputations
                        ],
                        note="Only the columns cleaning.yes_no_defaults names are filled. "
                        "Elsewhere a blank stays unknown, because it is.",
                    )
                ],
            )
        )

    if report.categorical_diffs:
        blocks.append(
            collapsed(
                f"Categories one site uses and the other does not "
                f"({len(report.categorical_diffs)} parameters)",
                [
                    table(
                        ["Parameter", f"{site_name('a')} only", f"{site_name('b')} only"],
                        [
                            [
                                esc(c["column"]),
                                esc(", ".join(c["site_a_only"][:5]) or "—"),
                                esc(", ".join(c["site_b_only"][:5]) or "—"),
                            ]
                            for c in report.categorical_diffs
                        ],
                    )
                ],
            )
        )
    else:
        blocks.append(md("The shared categorical parameters use the same levels at both sites."))

    return _section("cleaning", "Cleaning", blocks)


def _origin_index(run: BuildRun) -> dict[str, str]:
    """Where each column in the batch table came from.

    Derived features name their group, so a calculation is told from a
    measurement at a glance.
    """
    origins: dict[str, str] = {}
    for label, columns in run.origins.items():
        for column in columns:
            origins[column] = label
    for report in run.features.values():
        for entry in report.entries:
            if entry["status"] == CREATED:
                origins[entry["name"]] = f"{DERIVED} · {entry['group']}"
    return origins


#: Display formats of the parameter table's numeric columns.
_MISSING_FORMATS = {
    "Missing %": "%.1f",
    "Gaps": "%d",
    "Random would give": "%.1f",
    "Runs test z": "%.2f",
    "Longest gap": "%d",
    "Missing after missing %": "%.1f",
    f"Measured in last {missingness.RECENT} %": "%.0f",
}


def _missing_viewer(run: BuildRun) -> dict | None:
    """The pattern viewer, with the group and parameter tables to pick what it draws."""
    sites = []
    for site, (rows, analysis) in run.missingness.items():
        # A parameter recorded in every batch has no pattern to show.
        rows = [row for row in rows if row["Parameter"] in analysis.missing]
        if not rows:
            continue
        headers = list(rows[0])
        sites.append(
            {
                "label": site,
                "batches": analysis.batches,
                "dates": analysis.dates,
                "patterns": analysis.missing,
                "in_statistics": analysis.in_statistics,
                **_register_view(analysis),
                "groups": {g.name: g.members for g in analysis.groups},
                "group_headers": list(_GROUP_FORMATS),
                "group_rows": [
                    [
                        g.name,
                        len(g.members),
                        round(100 * g.group_rate, 1),
                        g.kind,
                        *_excluded_counts(g.members, analysis),
                    ]
                    for g in analysis.groups
                ],
                "group_formats": list(_GROUP_FORMATS.values()),
                "headers": headers,
                # The parameter table: those in a group are in the group table already.
                "rows": [[row[h] for h in headers] for row in rows if not row["Group"]],
                "formats": [_MISSING_FORMATS.get(h) for h in headers],
            }
        )
    block = missing_viewer(sites)
    if block is not None:
        # What the proposal form offers, so the page writes what `ml screening add` reads.
        block["proposal"] = {
            "columns": list(screening.PROPOSAL_COLUMNS),
            "actions": list(screening.ACTIONS),
            "reasons": list(screening.TRACK_REASONS[screening.MISSINGNESS]),
            "report": screening.MISSINGNESS,
        }
    return _section("viewer", "Missingness pattern", [block])


def _register_view(analysis: Any) -> dict:
    """What a report's viewer shows of the register at a site: its own exclusions (which it
    can reinstate), the other report's, and the conflicts."""
    return {
        "screened": analysis.screened,
        "elsewhere": analysis.elsewhere,
        "conflicts": [screening.conflict_line(c) for c in analysis.conflicts],
    }


def _excluded_counts(members: list[str], analysis: Any) -> list[int]:
    """How many of a group's members this report excludes, and how many the other one does."""
    return [
        sum(m in analysis.screened for m in members),
        sum(m in analysis.elsewhere and m not in analysis.screened for m in members),
    ]


#: The group table's columns, with their display formats.
_GROUP_FORMATS = {
    "Group": None,
    "Members": "%d",
    "Batches missing ≥1 member %": "%.1f",
    "Pattern": None,
    "Excluded here": "%d",
    "Excluded elsewhere": "%d",
}


def _missingness_help(run: BuildRun) -> dict | None:
    """How every number in the report was decided: the help the sidebar opens."""
    if not run.missingness:
        return None
    settings = next(iter(run.missingness.values()))[1].settings
    return {
        "title": "Method",
        "blocks": [
            md(
                "#### Reading the report\n\n"
                "Pick a site. Click a row of a table to draw it: a group draws all its "
                "parameters. Tick the box at the start of rows, in any table, to propose a "
                "screening decision about them. Each row of the plot is a "
                "parameter, each cell a batch in date order, newest on the right: scroll left "
                "to go back in time, hover a cell for its batch id and date. Batches with no "
                "date come last.\n\n#### How the numbers are decided"
            ),
            md(
                "Each site is analysed on its own batches, in date order. This is description "
                "only: nothing here changes the batch table. Batches drawn with green gaps are "
                "not in any statistic or group (About this report, in the sidebar, says "
                "which ones).\n\n"
                "- **Pattern.** A gap is a run of consecutive missing batches. The runs test "
                "compares the number of gaps with what chance gives for that many missing batches "
                f"(*Random would give*); |z| > {missingness.RUNS_Z} (p < 0.01, two-sided) is not "
                "random. Clumped patterns are named after where at least "
                f"{missingness.DOMINANT:.0%} of the missing batches sit.\n"
                "- **Missing after missing** is the share of batches missing when the batch "
                "before it is missing: near the missing rate is random, near 100% is long gaps, "
                "near 0% is regular skipping.\n"
                "- A parameter this site never records is *Never recorded* here."
            ),
            md(
                "#### Groups of parameters missing together\n\n"
                f"Parameters missing on {settings['min_rate']:.0%} to "
                f"{settings['max_rate']:.0%} of a site's batches take part. Two parameters group "
                "when the share of batches missing either that miss both (Jaccard similarity) "
                f"reaches {settings['threshold']}, by average linkage; a group has at least "
                f"{settings['min_size']} members. A group's pattern is that of "
                "the batches missing at least one member."
            ),
            md("#### Patterns"),
            table(
                ["Pattern", "Meaning"],
                [[esc(kind), meaning] for kind, meaning in missingness.PATTERNS.items()],
            ),
        ],
    }


#: The correlation and shifts reports' parameter tables, with their display formats.
_GROUPED_FORMATS = {
    "Batches with a value": "%d",
    "Distinct values": "%d",
    "Groups": "%d",
    "Days per group": "%.1f",
    "Permutation p": "%.3f",
    "|ρ| with it": "%.3f",
    "Shared shifts with it": "%.2f",
    "Batches in common": "%d",
    "q": "%.2g",
}


def _usable_representative(group: Any, elsewhere: dict) -> str:
    """The best-scoring member the other report does not exclude: ``= suggested`` when that
    is the suggestion, empty when there is none."""
    usable = [m for m in group.members if m not in elsewhere]
    if not usable:
        return ""
    best = max(usable, key=lambda m: group.scores.get(m, 0.0))
    return "= suggested" if best == group.representative else best


def _grouped_viewer(
    results: dict[str, tuple[list[dict], Any]],
    kind: str,
    title: str,
    measure: str,
    reason: str,
) -> dict | None:
    """The plot, with the group and parameter tables to pick what it draws.

    For the correlation report (``rank_viewer``): ``measure`` names the similarity the
    groups are made of, as in *Mean |ρ|*.
    """
    group_formats = {
        "Group": None,
        "Members": "%d",
        "Suggested representative": None,
        "Representative score": "%.3f",
        f"Mean {measure}": "%.3f",
        f"Weakest {measure}": "%.3f",
        "Fewest batches in common": "%d",
        "Weakest pair q": "%.2g",
        "Best usable representative": None,
        "Excluded here": "%d",
        "Excluded elsewhere": "%d",
    }
    sites = []
    for site, (rows, analysis) in results.items():
        if not rows:
            continue
        rows = [
            {"Parameter": row["Parameter"], "Register": row.get("Register", "")} | row
            for row in rows
        ]
        headers = list(rows[0])
        sites.append(
            {
                "label": site,
                "batches": analysis.batches,
                "dates": analysis.dates,
                # Drawn as scatterplots of the values.
                "values": analysis.values,
                **_register_view(analysis),
                # Every catalog parameter not compared, with why: the report lists them all.
                "not_analysed": [[p, why] for p, why in sorted(analysis.not_analysed.items())],
                "groups": {g.name: g.members for g in analysis.groups},
                "group_headers": list(group_formats),
                "group_rows": [
                    [
                        g.name,
                        len(g.members),
                        g.representative,
                        round(g.score, 3),
                        round(g.mean, 3),
                        round(g.weakest, 3),
                        g.min_batches,
                        float(f"{g.max_q:.3g}"),
                        _usable_representative(g, analysis.elsewhere),
                        *_excluded_counts(g.members, analysis),
                    ]
                    for g in analysis.groups
                ],
                "group_formats": list(group_formats.values()),
                "headers": headers,
                # The parameter table holds those in no group.
                "rows": [[row[h] for h in headers] for row in rows if not row["Group"]],
                "formats": [_GROUPED_FORMATS.get(h) for h in headers],
            }
        )
    block = missing_viewer(sites, kind=kind)
    if block is not None:
        block["site_label"] = "Site"
        block["proposal"] = {
            "columns": list(screening.PROPOSAL_COLUMNS),
            "actions": list(screening.ACTIONS),
            "reasons": list(screening.TRACK_REASONS[screening.CORRELATION]),
            "report": screening.CORRELATION,
            # A group is represented by one member rather than excluded whole.
            "reason": reason,
            "represent": True,
        }
    return _section("viewer", title, [block])


def _correlation_help(run: BuildRun) -> dict | None:
    if not run.correlation:
        return None
    threshold = next(iter(run.correlation.values()))[1].threshold
    return {
        "title": "Method",
        "blocks": [
            md(
                "#### Reading the report\n\n"
                "Pick a site. Click a group to draw its parameters against each other, one dot "
                "per batch with a value for both. The parameter table lists the parameters in no "
                "group; click one to draw it against its closest parameter. Tick parameters and "
                "a proposal excludes each one's closest, represented by it. Its *Register* "
                "column says what the register holds already. Up to four "
                "parameters show every pair; more show the parameter picked in *Against* "
                "(the representative by default) against each of the others. Parameters "
                "that rank the batches alike lie along a rising or falling curve. ⊘ marks a "
                "parameter this report already excludes here (as "
                f"*{correlation.REASON}*), ◇ one the missingness report excludes. "
                "*Excluded here* and *Excluded elsewhere* count them per group.\n\n"
                "To keep only a group's representative, tick the group (or several) and open "
                "*Propose a screening decision*: exclude, with the representative chosen under "
                "*Represented by*. Download the proposal and run `ml screening add`. Every "
                "new task file then lists the other members under `exclude`, with the "
                "decision as the reason. A group you do not propose stays whole.\n\n"
                "#### Why rank correlation\n\n"
                "A decision tree splits on the order of a parameter's values, not on the values: "
                "two parameters that order the batches alike offer it the same splits, and it "
                "picks one by chance, which splits the importance between them. Rank correlation "
                "(Spearman's ρ) measures exactly that, unchanged by any monotone transform, and "
                "a negative ρ is as redundant to a tree as a positive one, so groups use |ρ|."
            ),
            md(
                "#### How the numbers are decided\n\n"
                "Each site is analysed on its own batches: the batches the missingness report "
                "counts in its statistics. Batches it only draws are left out here.\n\n"
                "- **Parameters.** Every parameter the catalog types as "
                f"{', '.join(correlation.NUMERIC_TYPES)}, with at least "
                f"{correlation.MIN_BATCHES} values and two distinct ones, whatever the "
                "screening register says of it: the groups and suggestions depend on the data "
                "alone. *Not compared* lists the other catalog parameters, and why.\n"
                "- **ρ** of a pair is computed over the batches where both have a value, and "
                f"needs {correlation.MIN_BATCHES} of them (*Batches in common*).\n"
                "- **q** is the p-value of ρ (t-test on n-2 degrees of freedom), "
                "Benjamini-Hochberg adjusted over every pair compared at the site.\n"
                "- **Groups.** Complete linkage on 1-|ρ|: two parameters or groups join only "
                f"when every pair between them has |ρ| of at least {threshold} "
                "(`build.correlation_threshold` in config.yaml), so *Weakest |ρ|* never falls "
                "below it. A pair with too few batches in common counts as unrelated. *Weakest "
                "|ρ|* and *Weakest pair q* are the group's least related pair.\n"
                "- **Closest parameter** has the highest |ρ| to it, in a group or not.\n"
                "- **Suggested representative** is the member that carries most of the others' "
                "information: the highest *representative score*, the mean over the other "
                "members of |ρ| × the share of that member's batches the candidate has a value "
                "on too, since it stands in for another only where it was measured. Ties go to "
                "the most batches with a value. It is drawn first."
            ),
            md(
                "#### Screening\n\n"
                "To represent a group by one parameter, tick the group and open *Propose a "
                "screening decision*: its *Represented by* starts on the suggested representative. "
                "The proposal excludes every other member with the reason "
                f"*{correlation.REASON}* and records the representative in the register, which "
                "stays usable in every task. The build's *task_catalog.xlsx* lists what a task "
                "can still use, and what each representative stands for.\n\n"
                "A representative the missingness report excludes would leave its whole group "
                "out: a **conflict**. `ml screening add` refuses to make one; the build's own "
                "never-recorded exclusions can, and then a ⚠ banner names it here, in the "
                "missingness report, in `ml screening status` and in every task. *Best usable "
                "representative* is the best-scoring member the missingness report leaves in: "
                "pick it under *Represented by* to resolve one. To move a group to another "
                "representative, tick the group and pick it: the proposal reinstates it and "
                "excludes the others, represented by it."
            ),
        ],
    }


def _shift_viewer(run: BuildRun) -> dict | None:
    """Per plant: the groups over time of the parameters picked, and the tables to pick them."""
    sites = []
    for plant, (rows, analysis) in run.shifts.items():
        if not rows:
            continue
        headers = list(rows[0])
        sites.append(
            {
                "label": plant,
                "segments": analysis.segments,
                "groups": {g.name: g.members for g in analysis.groups},
                "group_headers": [
                    "Group",
                    "Members",
                    "Mean shared shifts",
                    "Weakest shared shifts",
                ],
                "group_rows": [
                    [g.name, len(g.members), round(g.mean, 2), round(g.weakest, 2)]
                    for g in analysis.groups
                ],
                "group_formats": [None, "%d", "%.2f", "%.2f"],
                "headers": headers,
                # The parameter table holds those in no group.
                "rows": [[row[h] for h in headers] for row in rows if not row["Shift group"]],
                "formats": [_GROUPED_FORMATS.get(h) for h in headers],
            }
        )
    block = {"kind": "shift_viewer", "sites": sites, "site_label": "Manufacturing site"}
    return _section("viewer", "Groups over time", [block if sites else None])


def _shifts_help(run: BuildRun) -> dict | None:
    if not run.shifts:
        return None
    return {
        "title": "Method",
        "blocks": [
            md(
                "#### Reading the report\n\n"
                "Pick a manufacturing site. Click a group to draw its parameters, or a "
                "parameter to draw it. Each row of the plot is a parameter; each bar is one "
                "of its groups: one value, from the first day it was used to the last, with "
                "the value written in it where it fits (hover for it, and the batches). The "
                "bars of a row alternate dark and light only so that neighbours are told "
                "apart. A blank stretch has no value. Thin grey lines mark the days the first "
                "row changes value: the rows of a group change on those days too.\n\n"
                "#### Leaving them out of a task\n\n"
                "`ml task new <folder> --exclude-shifting` lists every parameter in the "
                "group and parameter tables, at any manufacturing site, under "
                "`predictors.exclude` in the new task file, each with why: how many groups it "
                "runs in, where, and the group of parameters it changes with. It reads the "
                "latest build's `shifts.xlsx`, which holds them all.\n\n"
                "#### What runs in groups\n\n"
                "A raw material or consumable lot, its certificate values, a vector lot and "
                "its titre each keep one value for as long as a lot lasts, then change to "
                "another. A model can learn such a parameter as a stand-in for time."
            ),
            md(
                "#### How the numbers are decided\n\n"
                "Each manufacturing site is analysed on its own batches: time, and the lots, "
                "run per plant rather than per data source. They are the batches the "
                "missingness report counts in its statistics, split by the *Manufacturing "
                "Site* column; batches without a date are left out. Dates count by the day: "
                "a day's batches have no order among themselves, and on the day a lot runs out "
                "some of them can have the old lot and some the new one.\n\n"
                f"- **Parameters.** Any parameter with at least {shifts.MIN_BATCHES} values and "
                "two distinct ones is tested, unless the screening register already "
                "excludes it at the plant's data source.\n"
                "- **Groups.** Over the days the parameter has a value (blanks skipped), a "
                "group is one value kept from one day to the next; values are compared "
                "exactly. One group carries on from a day to the next, and only when both "
                "days hold its value; any other value on a day starts a group. So the day a "
                "lot changes ends the old lot's group and starts the new one's, while two "
                "values used side by side start a group every day.\n"
                "- A parameter **runs in groups** when its groups last at least "
                f"{shifts.MIN_DAYS:g} of its days on average (*Days per group*: its days with "
                "a value over its groups), so a value set per day, such as a date, does not; "
                f"and it has fewer groups than chance: in at most {shifts.P_MAX:g} of "
                f"{shifts.PERMUTATIONS} shuffles of its values over its batches do as few "
                "come out (*Permutation p*), so a rare value scattered at random does not.\n"
                "- **Shared shifts** of a pair: its *shift days* are the days a group starts, "
                "but its first. Over the days both have a value (at least "
                f"{shifts.MIN_BATCHES} batches), each shift day counts on the first of those "
                "days on or after it, and shared shifts is the share of their shift days the "
                "two have in common: 1 when they change on the same days.\n"
                "- **Groups of parameters.** Complete linkage on 1 - shared shifts: every "
                f"pair in a group shares at least {shifts.THRESHOLD:g} of its shift days. A "
                "group's name ends with the plant."
            ),
        ],
    }


def _missing_reason(run: BuildRun, parameter: str) -> str:
    """Why a catalog parameter is not in the batch table, in the pipeline's own terms."""
    statuses = _status_of(run, parameter)
    if not statuses:
        return "no source supplies it"
    notes = {
        f"{site}: {entry['status']} — {entry['note']}"
        if entry["note"]
        else f"{site}: {entry['status']}"
        for site, entry in statuses.items()
    }
    return "; ".join(sorted(notes))


def _catalog_coverage(run: BuildRun) -> dict | None:
    catalog = set(run.catalog_cols)
    missing = run.merge.catalog_missing
    extra = [c for c in run.result.columns if c not in catalog]

    blocks: list[Block | None] = []
    if missing:
        blocks.append(
            sortable(
                ["Catalog parameter", "Value type", "Why it is not in the batch table"],
                [
                    [str(p), str(run.catalog_mapping.get(p, "—")), _missing_reason(run, p)]
                    for p in missing
                ],
                filter_column="Catalog parameter",
                note=f"{len(missing):,} of {len(run.catalog_cols):,} catalog parameters are "
                "absent from the merged batch table. A parameter no source supplies has to come "
                "from the source system; a blocked feature needs the inputs it names.",
            )
        )
    else:
        blocks.append(md("Every catalog parameter is present in the merged batch table."))

    if extra:
        blocks.append(
            collapsed(
                f"Batch table parameters the catalog does not list ({len(extra)})",
                [
                    sortable(
                        ["Parameter", "Origin"],
                        [[str(c), _origin_index(run).get(c, "batch records")] for c in extra],
                        filter_column="Parameter",
                        note="These reach the batch table because a source carries them. The "
                        "catalog decides whether they belong there.",
                    )
                ],
            )
        )

    only = [[f"{site_name('a')} only", c] for c in run.merge.site_a_only_cols] + [
        [f"{site_name('b')} only", c] for c in run.merge.site_b_only_cols
    ]
    if only:
        blocks.append(
            collapsed(
                f"Parameters only one site has ({len(only)})",
                [
                    sortable(
                        ["Site", "Parameter"],
                        [[site, str(column)] for site, column in only],
                        filter_column="Parameter",
                        note="Outside the catalog name mapping, so the other site's rows are "
                        "empty for these.",
                    )
                ],
            )
        )

    return _section("catalog-coverage", "Catalog coverage", blocks)


def _enrichment(run: BuildRun) -> dict | None:
    blocks: list[Block | None] = []

    if run.lv_coa_coverage:
        blocks.append(
            table(
                ["Site", "Batches with a vector certificate", "Batches", "Coverage"],
                [
                    [esc(site), f"{cov['matched']:,}", f"{cov['total']:,}", f"{cov['pct']:.1f}%"]
                    for site, cov in run.lv_coa_coverage.items()
                ],
                note="A batch whose vector lot has no certificate keeps its row and "
                "gets empty CoA parameters.",
            )
        )

    if run.raw_material_columns:
        blocks.append(
            collapsed(
                f"Raw material and consumable identifiers joined ({len(run.raw_material_columns)})",
                [table(["Parameter"], [[esc(c)] for c in run.raw_material_columns])],
            )
        )

    if run.unmapped_sites:
        blocks.append(
            collapsed(
                f"{site_name('b')} clinical site acronyms with no institution "
                f"({len(run.unmapped_sites)})",
                [
                    table(
                        ["Acronym"],
                        [[esc(a)] for a in run.unmapped_sites],
                        note="These stay as acronyms in the batch table. Adding them to a mapping "
                        "file in mappings/ resolves them.",
                    )
                ],
            )
        )

    return _section("enrichment", "Joined sources", blocks)


def _log(run) -> dict | None:
    """The run's own log, counted in its label and drawn only on request.

    A few hundred events is a data grid Streamlit would otherwise build on every
    load, open or not, for a table most readers never look at.
    """
    if not run.events:
        return None
    levels = [event.level for event in run.events]
    label = (
        f"Event log ({len(run.events):,} events, {levels.count('WARN'):,} warnings, "
        f"{levels.count('ERROR'):,} errors)"
    )
    return _section(
        "log",
        "Run log",
        [
            deferred(
                label,
                [
                    sortable(
                        ["Time", "Level", "Module", "Message", "Detail"],
                        [
                            [
                                event.ts.strftime("%H:%M:%S"),
                                event.level,
                                event.module,
                                event.message,
                                event.detail or "",
                            ]
                            for event in run.events
                        ],
                        filter_column="Level",
                    )
                ],
                hint="Show the log",
            )
        ],
    )


def _provenance(run, parameters: list[list[str]] | None = None) -> dict | None:
    """What produced this file. Anything undeterminable is said out loud."""
    block = run.provenance
    rows = [
        ["Run date", block.get("run_date") or "unknown"],
        ["Git commit", block.get("git_commit") or "unknown (not a git checkout)"],
        ["Config", block.get("config") or "unknown"],
        ["Seed", str(block.get("seed")) if block.get("seed") is not None else "not set"],
        ["Duration", str(datetime.now() - run.run_start).split(".")[0]],
        ["Effective settings digest", (block.get("config_digest") or "unknown")[:16]],
        ["stlite", stlite.STLITE_VERSION],
        ["plotly", stlite.PLOTLY_VERSION],
    ]
    if parameters is None:
        # Whatever the stage was tuned by, as config had it for this run.
        parameters = [
            [name.replace("_", " "), num(value)]
            for name, value in vars(run.params or object()).items()
        ]
    inputs = [
        [
            source["label"],
            source.get("kind", "local file"),
            source.get("location", ""),
            source.get("digest") or "could not be read",
        ]
        for source in (block.get("sources") or [])
    ]
    environment = [[name, value] for name, value in (block.get("environment") or {}).items()]

    return _section(
        "provenance",
        "Provenance",
        [
            collapsed(
                "How this report was produced",
                [
                    table(["", ""], [[bold(label), esc(value)] for label, value in rows]),
                    table(
                        ["Input", "Kind", "Where it was read from", "Digest"],
                        [[esc(c) for c in row] for row in inputs],
                    )
                    if inputs
                    else None,
                    table(
                        ["Parameter", "Value"],
                        [[esc(name), esc(value)] for name, value in parameters],
                        note="The setpoints and thresholds the features were computed "
                        "against, as config had them for this run.",
                    ),
                    table(
                        ["Package", "Version"],
                        [[esc(name), esc(value)] for name, value in environment],
                    )
                    if environment
                    else None,
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------
def build_batch_table_payload(run: BuildRun) -> dict[str, Any]:
    """The build's report: a title, a lead, and the sections that had something."""
    sections = [
        _overview(run),
        _sources(run),
        _cleaning(run),
        _catalog_coverage(run),
        _enrichment(run),
        _log(run),
        _provenance(run),
    ]
    stamp = run.run_start.strftime("%Y-%m-%d %H:%M")
    return {
        "title": "Batch table build",
        "lead": f"{run.merge.total_rows:,} batches and "
        f"{run.merge.total_columns:,} parameters from {site_name('a')} and {site_name('b')}, built "
        f"against {len(run.catalog_cols):,} catalog parameters.",
        "subtitle": f"CAR-T manufacturing data · run {stamp}",
        "sections": [section for section in sections if section],
    }


def build_missingness_payload(run: BuildRun) -> dict[str, Any]:
    """The build's missingness report: the viewer, its provenance, and the method as help."""
    sections = [_missing_viewer(run), _provenance(run)]
    stamp = run.run_start.strftime("%Y-%m-%d %H:%M")
    cohorts = [
        f"{site}: {analysis.cohort}"
        for site, (_, analysis) in run.missingness.items()
        if analysis.cohort
    ]
    return {
        "title": "Batch table missingness",
        "lead": f"How each of the {run.merge.total_columns:,} parameters goes missing, per site, "
        "batch by batch: its pattern over time and the parameters missing with it. A parameter "
        "recorded in every batch of a site is left out; one missing in every batch is excluded "
        "in the screening register by the build. "
        + run.register_note
        + (" Batches described — " + "; ".join(cohorts) + "." if cohorts else ""),
        "subtitle": f"CAR-T manufacturing data · run {stamp}",
        "sections": [section for section in sections if section],
        "help": _missingness_help(run),
        # Opens on the plot, with the site and this lead in the sidebar.
        "compact": True,
    }


def _grouped_payload(
    run: BuildRun,
    results: dict[str, tuple[list[dict], Any]],
    viewer: dict | None,
    title: str,
    lead: str,
    help_: dict | None,
) -> dict[str, Any]:
    """The correlation or shifts report, laid out as the missingness report is."""
    sections = [viewer, _provenance(run)]
    cohorts = [f"{site}: {a.cohort}" for site, (_, a) in results.items() if a.cohort]
    return {
        "title": title,
        "lead": lead + (" Batches compared — " + "; ".join(cohorts) + "." if cohorts else ""),
        "subtitle": f"CAR-T manufacturing data · run {run.run_start.strftime('%Y-%m-%d %H:%M')}",
        "sections": [section for section in sections if section],
        "help": help_,
        "compact": True,
    }


def build_correlation_payload(run: BuildRun) -> dict[str, Any]:
    """The build's correlation report."""
    viewer = _grouped_viewer(
        run.correlation, "rank_viewer", "Rank correlation", "|ρ|", correlation.REASON
    )
    return _grouped_payload(
        run,
        run.correlation,
        viewer,
        "Batch table rank correlation",
        "Which numerical parameters put the batches in the same order, per site: "
        "redundant to a decision tree. " + run.register_note,
        _correlation_help(run),
    )


def build_shifts_payload(run: BuildRun) -> dict[str, Any]:
    """The build's shifts report."""
    tested = ", ".join(
        f"{site} {len(a.analysed):,} of {a.tested:,}" for site, (_, a) in run.shifts.items()
    )
    return _grouped_payload(
        run,
        run.shifts,
        _shift_viewer(run),
        "Batch table shifts",
        "Which parameters keep one value for a stretch of days and then change to another, as "
        "a lot does, and which change on the same days, per manufacturing site. "
        "`ml task new --exclude-shifting` leaves them all out of a task. Parameters that run "
        f"in groups — {tested}.",
        _shifts_help(run),
    )


def write(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Render a payload into one self-contained HTML file."""
    return stlite.write(payload, output_path)


# ---------------------------------------------------------------------------
# Stage 1: the catalog report
# ---------------------------------------------------------------------------
@dataclass
class CatalogRun:
    """What the catalog stage found, and what produced it."""

    catalog: Any
    events: list
    provenance: dict
    params: Any = None
    run_start: datetime = field(default_factory=datetime.now)


def _catalog_overview(run: CatalogRun) -> dict | None:
    report = run.catalog
    rows = [
        ["Parameters in the catalog", f"{report.catalog_parameters:,}"],
        ["Sources compared", f"{len(report.sources_read):,}"],
        ["Source columns the catalog does not list", f"{report.total:,}"],
        ["Distinct names among them", f"{len(report.new_parameters):,}"],
        ["Sources that could not be read", f"{len(report.sources_absent):,}"],
    ]
    if report.written_to:
        rows.append(["Written to", esc(report.written_to)])
    return _section(
        "overview",
        "Overview",
        [table(["", ""], [[bold(label), value] for label, value in rows])],
    )


def _catalog_sources(run: CatalogRun) -> dict | None:
    report = run.catalog
    rows = [
        [esc(label), f"{columns:,}", f"{len(report.by_source.get(label, [])):,}", "read"]
        for label, columns in report.sources_read.items()
    ]
    rows += [[esc(label), "—", "—", "not available"] for label in report.sources_absent]
    return _section(
        "sources",
        "Sources",
        [table(["Source", "Columns", "Not in the catalog", "Status"], rows)],
    )


def _catalog_new(run: CatalogRun) -> dict | None:
    report = run.catalog
    if not report.new_parameters:
        return _section(
            "new",
            "Parameters new to the catalog",
            [md("Every column in every source is already a catalog parameter.")],
        )

    source_of: dict[str, list[str]] = {}
    for label, names in report.by_source.items():
        for name in names:
            source_of.setdefault(name, []).append(label)

    return _section(
        "new",
        "Parameters new to the catalog",
        [
            sortable(
                ["Parameter", "Recorded by"],
                [[str(name), ", ".join(source_of.get(name, []))] for name in report.new_parameters],
                filter_column="Parameter",
                note=f"{len(report.new_parameters):,} distinct names the sources record and the "
                "catalog does not list. Until the catalog names one, the build cannot carry it "
                "into the batch table — so this is the list to work through, not a list of errors.",
            ),
            collapsed(
                f"The same thing, source by source ({report.total})",
                [
                    table(
                        ["Source", "Parameter"],
                        [
                            [esc(label), esc(name)]
                            for label, names in report.by_source.items()
                            for name in names
                        ],
                        note="A name recorded by two sources appears twice here and once above.",
                    )
                ],
            ),
        ],
    )


def build_catalog_payload(run: CatalogRun) -> dict[str, Any]:
    """The catalog stage's report."""
    sections = [
        _catalog_overview(run),
        _catalog_sources(run),
        _catalog_new(run),
        _log(run),
        _provenance(run, parameters=[]),
    ]
    report = run.catalog
    return {
        "title": "Catalog coverage",
        "lead": f"{len(report.new_parameters):,} parameters the sources record and the catalog "
        f"does not list, against {report.catalog_parameters:,} it does.",
        "subtitle": f"CAR-T manufacturing data · run {run.run_start.strftime('%Y-%m-%d %H:%M')}",
        "sections": [section for section in sections if section],
    }


# ---------------------------------------------------------------------------
# Stage 3: one task
# ---------------------------------------------------------------------------
@dataclass
class TaskRun:
    """What one task produced, and what produced it."""

    dataset: pd.DataFrame
    result: Any
    spec: Any
    events: list
    provenance: dict
    output_path: str = ""
    run_start: datetime = field(default_factory=datetime.now)

    @property
    def params(self):  # the provenance panel lists whatever the stage was tuned by
        return None


def _issues(run: TaskRun) -> list[str]:
    """The handful of things a reader has to know before using this dataset."""
    result = run.result
    issues = list(result.warnings)
    for reason, count in result.cohort.rejected.items():
        issues.append(f"{count} batches left out: {reason}")
    if result.dropped_constant:
        issues.append(
            f"{len(result.dropped_constant)} columns were dropped as constant on the fitted rows"
        )
    if result.dropped_sparse:
        issues.append(
            f"{len(result.dropped_sparse)} columns were dropped for having fewer than "
            f"{run.spec.params.min_non_missing} values"
        )
    if result.clusters.unsupported_pairs:
        issues.append(
            f"{result.clusters.unsupported_pairs} parameter pairs share fewer than "
            f"{result.clusters.min_overlap} batches, so their relationship is unknown"
        )
    failed_fits = [
        fit for cluster in result.clusters.clusters for fit in cluster.fits if not fit.residualised
    ]
    if failed_fits:
        issues.append(
            f"{len(failed_fits)} cluster members had too little data to residualise and were "
            "kept as recorded"
        )
    warnings = [event for event in run.events if event.level in ("WARN", "ERROR")]
    if warnings:
        issues.append(f"{len(warnings)} warnings in this run's log")
    return issues


def _task_overview(run: TaskRun) -> dict | None:
    result, spec = run.result, run.spec
    unit = _unit_of(spec.target.column)
    groups = (
        int(run.dataset[spec.roles.group].nunique())
        if spec.roles.group and spec.roles.group in run.dataset.columns
        else 0
    )
    rows = [
        ["Task", esc(spec.name)],
        ["Question", esc(spec.question or "not stated in the task file")],
        [
            "Target",
            esc(_target_text(spec, result, unit)),
        ],
        ["Cohort", esc(", ".join(spec.cohort.sites) or "every site")],
        ["Batches in the dataset", f"{result.rows:,}"],
        ["Predictor columns", f"{result.feature_columns:,}"],
    ]
    if groups:
        rows.append([esc(f"Distinct {spec.roles.group} values"), f"{groups:,}"])
    rows += [
        ["Correlated groups found", f"{len(result.clusters.clusters):,}"],
        ["Clustering action", esc(result.clusters.action)],
        ["What the table is", esc(result.dataset_role)],
    ]

    blocks: list[Block | None] = [
        table(["", ""], [[bold(label), value] for label, value in rows]),
        md(
            "This is a description of how the dataset was built. No model was fitted here "
            "and no predictive performance is claimed by any number in this report."
        ),
    ]
    issues = _issues(run)
    if issues:
        blocks.append(
            table(
                ["Worth knowing before using this dataset"],
                [[esc(issue)] for issue in issues],
                note="The package was generated successfully. Whether the data in it is good "
                "enough for a given question is the separate judgement these lines are for.",
            )
        )
    else:
        blocks.append(md("Nothing in this run needed a warning."))
    return _section("overview", "Overview", blocks)


def _target_text(spec, result, unit: str) -> str:
    """The target in a line: its column and kind, or the two periods it compares."""
    if spec.target.type == "periods":
        first, second = spec.target.periods
        return (
            f"{spec.target.column}: {second.name} ({second.start} – {second.stop}) = 1, "
            f"{first.name} ({first.start} – {first.stop}) = 0"
        )
    return f"{spec.target.column}{f' ({unit})' if unit else ''} — {result.target_kind}"


def _unit_of(name: str) -> str:
    import re as _re

    match = _re.search(r"\(([^()]{1,24})\)\s*$", str(name))
    return match.group(1) if match else ""


def _task_cohort(run: TaskRun) -> dict | None:
    result = run.result
    funnel = [
        [
            bold("Every batch in the batch table"),
            f"{result.cohort.raw_rows:,}",
            "",
            f"{result.cohort.raw_rows:,}",
        ]
    ]
    funnel += [
        [
            esc(step["Filter"]),
            f"{step['Batches before']:,}",
            f"{step['Removed']:,}",
            f"{step['Batches left']:,}",
        ]
        for step in result.cohort.funnel
    ]
    blocks: list[Block | None] = [
        table(
            ["Filter", "Batches before", "Removed", "Batches left"],
            funnel,
            note=f"Applied in this order, including the step for batches whose target was "
            f"never measured. The last line is the {result.rows:,} rows of the dataset.",
        )
    ]

    if result.cohort.rejected:
        blocks.append(
            table(
                ["Why a batch is not in the dataset", "Batches"],
                [[esc(reason), f"{count:,}"] for reason, count in result.cohort.rejected.items()],
                note="Counts, separately from the identifiers below: a count is a fact about "
                "the cohort, a list of batch numbers is a lead to follow.",
            )
        )
    for reason, examples in result.cohort.examples.items():
        if examples:
            blocks.append(
                collapsed(
                    f"Batches left out — {reason} ({len(examples)} named)",
                    [
                        sortable(
                            ["Batch"],
                            [[batch] for batch in examples],
                            filter_column="Batch",
                            note="They stay in the batch table. metadata/batches.csv in the "
                            "package has every batch with the filter that excluded it.",
                        )
                    ],
                )
            )

    excluded = [row for row in result.cohort.exclusions if not row["Included"]]
    blocks.append(
        download(
            f"Download batches.csv ({len(result.cohort.exclusions):,} batches, "
            f"{len(excluded):,} excluded)",
            "batches.csv",
            csv_text(result.cohort.exclusions),
            note="One row per batch in the batch table: whether it is in the dataset, "
            "and the first "
            "filter that ruled it out.",
        )
    )
    return _section("cohort", "Cohort", blocks)


def _quality_rows(run: TaskRun) -> list[dict]:
    frame = run.dataset
    rows = []
    for column in frame.columns:
        present = int(frame[column].notna().sum())
        rows.append(
            {
                "column": str(column),
                "present": present,
                "missing_pct": round(100 * (len(frame) - present) / len(frame), 1)
                if len(frame)
                else 0.0,
                "distinct": int(frame[column].nunique(dropna=True)),
            }
        )
    return rows


def _task_quality(run: TaskRun) -> dict | None:
    result = run.result
    rows = _quality_rows(run)
    constants = [row for row in rows if row["distinct"] <= 1]
    blocks: list[Block | None] = [
        md(
            f"Every column of the exported table, and how much of it there is. "
            f"{len(constants)} column(s) hold one value or none across the "
            f"{result.rows:,} batches."
        ),
        sortable(
            ["Column", "Batches with a value", "% missing", "Distinct values"],
            [[row["column"], row["present"], row["missing_pct"], row["distinct"]] for row in rows],
            formats=[None, "%d", "%.1f", "%d"],
            filter_column="Column",
            flag={"column": "Batches with a value", "below": run.spec.params.min_non_missing},
            note="A count below the task's minimum is marked. Columns that fell under it "
            "before export are in the decisions table, not here.",
        ),
        plots.missingness(rows),
    ]

    if result.numeric_notes:
        blocks.append(
            table(
                ["Parameter", "Values that are not numbers", "Infinite values"],
                [
                    [esc(note["Parameter"]), f"{note['Unreadable']:,}", f"{note['Infinite']:,}"]
                    for note in result.numeric_notes
                ],
                note="Both became missing. A value that could not be read is a parse failure; "
                "an infinite one is a division that should have been missing.",
            )
        )
    if result.nonpositive_durations:
        blocks.append(
            table(
                ["Duration parameter", "Values at or below zero"],
                [
                    [esc(column), f"{count:,}"]
                    for column, count in result.nonpositive_durations.items()
                ],
                note="Durations are in minutes. Zero or less is a recording error, so it is "
                "treated as missing.",
            )
        )
    if result.ordinal_unexpected:
        blocks.append(
            collapsed(
                f"Values not in the catalog's category list ({len(result.ordinal_unexpected)})",
                [
                    table(
                        ["Parameter", "Value", "Batches"],
                        [
                            [esc(row["Parameter"]), esc(row["Value"]), str(row["Batches"])]
                            for row in result.ordinal_unexpected
                        ],
                        note="Left missing rather than ranked: where they belong in the order "
                        "is exactly what is unknown.",
                    )
                ],
            )
        )
    if result.missing_from_batch_table:
        blocks.append(
            collapsed(
                "Catalog parameters the batch table does not have "
                f"({len(result.missing_from_batch_table)})",
                [
                    sortable(
                        ["Parameter"],
                        [[str(p)] for p in result.missing_from_batch_table],
                        filter_column="Parameter",
                        note="They could not become predictors. The build's own report says "
                        "why each one is absent.",
                    )
                ],
            )
        )
    if result.duplicate_catalog_parameters:
        blocks.append(
            table(
                ["Catalog parameter listed more than once"],
                [[esc(name)] for name in result.duplicate_catalog_parameters],
                note="Each was used once.",
            )
        )
    return _section("quality", "Data quality", blocks)


def _task_parameters(run: TaskRun) -> dict | None:
    result = run.result
    decisions = [d.row() for d in result.decisions]
    blocks: list[Block | None] = [
        md(
            f"**{len(result.candidates):,}** of the batch table's parameters were allowed to be "
            f"predictors, and they produced **{result.feature_columns:,}** columns. "
            "Everything else was excluded for a reason, and every reason is below."
        ),
        sortable(
            ["Column", "From", "Encoding", "Meaning"],
            [[r["output"], r["source"], r["strategy"], r["detail"]] for r in result.dictionary],
            filter_column="Column",
            note="Every predictor column of the dataset. This is how a name like `…_resid`, "
            "`…_target_enc` or `…_hash_07` is read back to the parameter it came from.",
        ),
        download(
            f"Download dataset_columns.csv ({len(result.columns):,} columns)",
            "dataset_columns.csv",
            csv_text(result.columns),
            note="One row per exported column, in output order, including the identifier, the "
            "grouping column and the target.",
        ),
        sortable(
            ["Parameter", "Decision", "Reason", "Stage"],
            [[row["Parameter"], row["Decision"], row["Reason"], row["Stage"]] for row in decisions],
            filter_column="Parameter",
            note="Included, excluded or changed, and why. Descent from the target is in here "
            "with the chain that produced it.",
        ),
        download(
            f"Download parameters.csv ({len(decisions):,} decisions)",
            "parameters.csv",
            csv_text(decisions),
        ),
        download(
            f"Download encoded_columns.csv ({len(result.dictionary):,} features)",
            "encoded_columns.csv",
            csv_text(
                [
                    {
                        "Feature": r["output"],
                        "Parameter": r["source"],
                        "Encoding": r["strategy"],
                        "Meaning": r["detail"],
                    }
                    for r in result.dictionary
                ]
            ),
            note="The feature dictionary: every column the recipe produces, and its parameter.",
        ),
    ]
    if result.proxies:
        blocks.append(
            table(
                ["Parameter", "Column", "r with the target", "Shared batches"],
                [
                    [
                        esc(p["Parameter"]),
                        esc(p["Feature"]),
                        f"{p['r']:+.3f}",
                        str(p["Shared batches"]),
                    ]
                    for p in result.proxies
                ],
                note="Measured on the fitted rows only. A predictor this close to the target is "
                "often the target measured another way — a leak — and is worth a second look "
                "before anyone trusts a model that leans on it.",
            )
        )

    if result.routes:
        blocks.append(plots.encoder_split(result.routes))
        blocks.append(
            sortable(
                ["Parameter", "Categories", "Categories per batch", "Encoder", "Why"],
                [
                    [r.column, r.categories, round(r.ratio, 3), r.strategy, r.reason]
                    for r in result.routes
                ],
                formats=[None, "%d", "%.3f", None, None],
                filter_column="Parameter",
                note="Cardinality decides the encoder. Identifiers and grouping columns were "
                "removed by role before this ran, so nothing here is one of those.",
            )
        )

    info = result.target_encoding
    if info.get("columns"):
        strategy = info.get("strategy", "shuffled")
        blocks.append(
            md(
                f"Target encoding ran over **{info.get('batches', 0):,}** labelled batches with "
                f"{strategy} folds (K={info.get('folds')}), smoothing m="
                f"{info.get('smoothing'):g}. Each fold's prior and category means come from "
                "that fold's training rows alone, and rows outside the fitted population are "
                f"encoded with the frozen training means (prior {num(info.get('prior'))}). "
                + (f"Note: {info.get('note')}. " if info.get("note") else "")
                + "It uses the target, so refit it inside any outer fold you evaluate in."
            )
        )
        starved = {
            name: count
            for name, count in (info.get("unseen") or {}).items()
            if count >= 0.5 * max(info.get("batches", 1), 1)
        }
        if starved:
            blocks.append(
                table(
                    ["Column", "Rows encoded with the fold prior", "What that means"],
                    [
                        [
                            esc(name),
                            f"{count:,} of {info.get('batches', 0):,}",
                            "the category was not in that fold's training rows, so the column "
                            "carries little or nothing here",
                        ]
                        for name, count in starved.items()
                    ],
                    note="Usually this means the category is confounded with whatever the folds "
                    "respect — a shift team that only ever runs one vector lot, say.",
                )
            )
    if result.one_hot_tail:
        blocks.append(
            collapsed(
                f"What the one-hot tail swept up ({len(result.one_hot_tail)})",
                [
                    table(
                        ["Parameter", "Categories kept", "Batches in Other", "Other %"],
                        [
                            [
                                esc(row["Parameter"]),
                                str(row["Categories kept"]),
                                f"{row['Batches in Other']:,}",
                                f"{row['Other %']:.1f}%",
                            ]
                            for row in result.one_hot_tail
                        ],
                        note="A category first seen outside the fitted rows lands here too.",
                    )
                ],
            )
        )
    if result.hashing_summary:
        blocks.append(
            collapsed(
                f"Hashed parameters ({len(result.hashing_summary)})",
                [
                    table(
                        ["Parameter", "Categories", "Buckets"],
                        [
                            [esc(row["Parameter"]), f"{row['Categories']:,}", str(row["Buckets"])]
                            for row in result.hashing_summary
                        ],
                        note="A deterministic MD5 hash, so the same category lands in the same "
                        "bucket in every run. Two categories may share one.",
                    )
                ],
            )
        )
    if result.duplicate_decisions:
        blocks.append(
            collapsed(
                f"Parameters dropped as near-duplicates ({len(result.duplicate_decisions)})",
                [
                    sortable(
                        ["Dropped", "Kept instead", "R²", "Shared batches"],
                        [
                            [
                                row["Dropped"],
                                row["Kept"],
                                row["R2"],
                                row["Shared batches"],
                            ]
                            for row in result.duplicate_decisions
                        ],
                        formats=[None, None, "%.4f", "%d"],
                        filter_column="Dropped",
                        note="Each dropped parameter names one that is still in the dataset, "
                        "measured on the batches the two of them share.",
                    )
                ],
            )
        )
    if result.dropped_sparse:
        blocks.append(
            collapsed(
                f"Columns dropped for having too few values ({len(result.dropped_sparse)})",
                [
                    sortable(
                        ["Column", "Batches with a value", "Produced by"],
                        [
                            [row["Feature"], row["Batches with a value"], row["Produced by"]]
                            for row in result.dropped_sparse
                        ],
                        formats=[None, "%d", None],
                        filter_column="Column",
                    )
                ],
            )
        )
    if result.dropped_constant:
        blocks.append(
            collapsed(
                f"Columns dropped as constant on the fitted rows ({len(result.dropped_constant)})",
                [
                    sortable(
                        ["Column", "Produced by", "Value"],
                        [
                            [row["Feature"], row["Produced by"], row["Value"]]
                            for row in result.dropped_constant
                        ],
                        filter_column="Column",
                        note="Constant to measurement precision: an empty one-hot tail, a hash "
                        "bucket nobody landed in, or a value that only differs in its last "
                        "digits. Scaled to unit variance, the last kind would dominate a model.",
                    )
                ],
            )
        )
    return _section("parameters", "Parameters and encodings", blocks)


def _task_missingness(run: TaskRun) -> dict | None:
    """Parameters that go missing on the same batches, and what explains it."""
    analysis = run.result.missingness
    if analysis is None or not analysis.enabled:
        return None
    blocks: list[Block | None] = [
        md(
            f"Columns whose values go missing on the same batches, over the {analysis.rows:,} "
            f"batches of the cohort. {len(analysis.analysed):,} of {analysis.columns:,} columns "
            f"have a missing rate between {analysis.settings['min_rate']} and "
            f"{analysis.settings['max_rate']} and take part: {analysis.too_complete:,} are too "
            f"complete and {analysis.too_empty:,} too empty to have a pattern. Two columns group "
            f"when their Jaccard similarity reaches {analysis.settings['threshold']}. "
            "This is description only: nothing here changes the dataset."
        )
    ]
    if not analysis.groups:
        blocks.append(md("No group of parameters goes missing together."))
        return _section("missingness", "Parameters missing together", blocks)
    blocks.append(md(f"#### Summary of all {len(analysis.groups)} groups"))
    blocks.append(
        table(
            [
                "Group",
                "Members",
                "Batches missing ≥1 member (%)",
                "Mean Jaccard",
                "Best explainer",
                "Score",
                "Explainer blanks track the group",
            ],
            [
                [
                    esc(g.name),
                    str(len(g.members)),
                    f"{100 * g.group_rate:.0f}",
                    f"{g.cohesion:.2f}",
                    esc(g.explainer.column if g.explainer else "—"),
                    f"{g.explainer.score:.2f}" if g.explainer else "—",
                    "yes" if g.explainer and g.explainer.self_correlated else "no",
                ]
                for g in analysis.groups
            ],
            note="The score is the mutual information between the explainer's categories and "
            "the group's missing pattern over the pattern's entropy: 0 says nothing, 1 says the "
            "category alone decides whether the group is recorded.",
        )
    )
    blocks.append(md("#### Group details"))
    blocks.append(
        chooser(
            "Group",
            [
                {
                    "label": g.name,
                    "blocks": [
                        md(f"#### {g.name}"),
                        table(
                            ["Parameter", "Missing rate"],
                            [[esc(m), f"{100 * g.member_rates[m]:.0f}%"] for m in g.members],
                        ),
                        plots.missing_pattern(g),
                        plots.missing_by_category(g),
                        table(
                            ["Runner-up explainer", "Score"],
                            [[esc(name), f"{score:.2f}"] for name, score in g.alternatives],
                        )
                        if g.alternatives
                        else None,
                    ],
                }
                for g in analysis.groups
            ],
        )
    )
    blocks.append(
        download(
            "Download missingness.csv",
            "missingness.csv",
            csv_text(missingness.rows(analysis)),
        )
    )
    return _section("missingness", "Parameters missing together", blocks)


def _cluster_detail(cluster) -> list[Block | None]:
    def defined(value) -> str:
        value = _number_or_none(value)
        return "undefined" if value is None else f"{value:.2f}"

    members = [
        [
            member,
            "representative" if member == cluster.representative else "member",
            (
                cluster.why_representative
                if member == cluster.representative
                else next(
                    (
                        f"{fit.model}, {fit.rows} batches"
                        + (f" — {fit.fallback}" if fit.fallback else "")
                        for fit in cluster.fits
                        if fit.column == member
                    ),
                    "dropped" if member in cluster.dropped else "unchanged",
                )
            ),
            next(
                (
                    ", ".join(fit.regressors)
                    for fit in cluster.fits
                    if fit.column == member and fit.residualised
                ),
                "",
            ),
            next(
                (round(fit.adj_r2, 3) for fit in cluster.fits if fit.column == member),
                None,
            ),
        ]
        for member in cluster.members
    ]
    return [
        md(f"#### {esc(cluster.name)}"),
        table(
            ["", ""],
            [
                [bold("Members"), f"{len(cluster.members):,}"],
                [bold("Representative"), esc(cluster.representative)],
                [bold("Why it represents the group"), esc(cluster.why_representative)],
                [bold("Action"), esc(cluster.strategy)],
                [bold("Smallest shared batches"), f"{cluster.min_overlap:,}"],
                [
                    bold("Max |r| before / after"),
                    f"{defined(cluster.corr_before)} / {defined(cluster.corr_after)}",
                ],
                [
                    bold("Max VIF before / after"),
                    f"{defined(cluster.vif_before)} / {defined(cluster.vif_after)}",
                ],
                [bold("Rows the VIF was computed on"), f"{cluster.vif_rows_before}"],
            ],
        ),
        sortable(
            ["Parameter", "Role", "What happened to it", "Fitted on", "Adjusted R²"],
            members,
            formats=[None, None, None, None, "%.3f"],
            filter_column="Parameter",
            note="Adjusted R² is how much of the member its regressors accounted for on the "
            "rows where all of them are present. It is a description of these batches, not "
            "evidence of anything out of sample.",
        ),
        plots.cluster_correlation(cluster),
        plots.cluster_fits(cluster),
        table(
            ["Code", "Before", "After"],
            [
                [
                    str(i + 1),
                    esc(cluster.members[i]) if i < len(cluster.members) else "—",
                    esc(cluster.labels_after[i]) if i < len(cluster.labels_after) else "—",
                ]
                for i in range(max(len(cluster.members), len(cluster.labels_after)))
            ],
            note="The codes on the heatmap axes.",
        ),
    ]


def _task_clusters(run: TaskRun, membership: list[dict]) -> dict | None:
    report = run.result.clusters
    if report.action == "off":
        return _section(
            "clusters",
            "Correlated groups",
            [
                md(
                    "Clustering is off in the task file, so correlated parameters were neither "
                    "looked for nor changed."
                )
            ],
        )

    blocks: list[Block | None] = [
        md(
            f"Parameters that move together, found in the data: {report.method} correlation, "
            f"complete linkage, cut at |r| ≥ {report.threshold}, and a pair needs at least "
            f"{report.min_overlap} shared batches before its correlation counts as evidence. "
            f"Action: **{report.action}**. "
            + {
                "describe": "The groups are described here and every parameter went into "
                "the dataset unchanged.",
                "keep_one": "One member of each group was kept and the others dropped.",
                "residualise": "One member of each group was kept and the others replaced by what "
                "is left of them once the retained ones are accounted for.",
                "summarise": "Each group was replaced by one score: the mean of its z-scored "
                "members, signs aligned to the representative.",
            }.get(report.action, "")
            + f" Each group was redrawn on {decorrelate.STABILITY_RESAMPLES} subsamples of "
            f"{decorrelate.STABILITY_FRACTION:.0%} of the batches; only groups found again in "
            f"at least {run.spec.clustering.min_stability:.0%} of them were acted on, and "
            "only those are explanation groups in metadata/groups.json."
            + (
                " Correlations were computed within site — each site's mean removed first — "
                "so a group means the parameters move together, not that the sites differ."
                if report.within_site
                else ""
            )
        )
    ]
    if report.action == "residualise":
        blocks.append(
            md(
                "Residualising does not preserve everything the group held, and it does not "
                "make the columns independent: each regression is fitted on the rows where "
                "all its inputs are present, so columns with different gaps are only "
                "orthogonal there, and least-squares orthogonality is not zero rank "
                "correlation."
            )
        )

    if not report.clusters:
        blocks.append(
            md(
                f"No group of {run.spec.clustering.min_size} or more of the "
                f"{len(report.usable):,} eligible parameters correlates that strongly on "
                "enough shared batches."
            )
        )
    else:
        blocks.append(md(f"#### Summary of all {len(report.clusters)} clusters"))
        blocks.append(
            sortable(
                [
                    "Cluster",
                    "Members",
                    "Representative",
                    "Stability",
                    "Action",
                    "Smallest shared batches",
                    "Max |r| before",
                    "Max |r| after",
                    "Max VIF before",
                    "Max VIF after",
                    "VIF rows",
                ],
                [
                    [
                        cluster.name,
                        len(cluster.members),
                        cluster.representative,
                        _number_or_none(cluster.stability),
                        cluster.action,
                        cluster.min_overlap,
                        _number_or_none(cluster.corr_before),
                        _number_or_none(cluster.corr_after),
                        _number_or_none(cluster.vif_before),
                        _number_or_none(cluster.vif_after),
                        cluster.vif_rows_before,
                    ]
                    for cluster in report.clusters
                ],
                formats=[
                    None,
                    "%d",
                    None,
                    "%.2f",
                    None,
                    "%d",
                    "%.2f",
                    "%.2f",
                    "%.1f",
                    "%.1f",
                    "%d",
                ],
                filter_column="Representative",
                note="An empty correlation or VIF cell is undefined — too few complete rows to "
                "compute it — which is not the same as zero or as infinite. VIF is computed on "
                "the rows where every member is present, and that count is the last column.",
            )
        )
        grouped = [row for row in membership if row["Cluster"]]
        blocks.append(md(f"#### Every member of every cluster ({len(grouped):,})"))
        blocks.append(
            sortable(
                [
                    "Cluster",
                    "Parameter",
                    "Representative",
                    "Outcome",
                    "Output column",
                    "Rows fitted",
                    "Adjusted R²",
                ],
                [
                    [
                        row["Cluster"],
                        row["Parameter"],
                        row["Representative"],
                        row["Outcome"],
                        row["Output column"],
                        _number_or_none(row["Rows fitted"]),
                        _number_or_none(row["Adjusted R2"]),
                    ]
                    for row in grouped
                ],
                formats=[None, None, None, None, None, "%d", "%.3f"],
                filter_column="Parameter",
                note="Type a parameter name to find its cluster. The same rows as clusters.csv.",
            )
        )
        blocks.append(
            md(
                "#### Cluster details\n\nPick a cluster in the dropdown to see its numbers, "
                "its members, the correlation heatmap before and after, and the fits. One "
                "cluster is drawn at a time."
            )
        )
        blocks.append(
            chooser(
                f"Cluster ({len(report.clusters)})",
                [
                    {
                        "label": f"{cluster.name} — {cluster.representative} "
                        f"({len(cluster.members)} members)",
                        "blocks": _cluster_detail(cluster),
                    }
                    for cluster in report.clusters
                ],
            )
        )

    if report.skipped:
        blocks.append(
            collapsed(
                f"Parameters that could not take part ({len(report.skipped)})",
                [
                    sortable(
                        ["Parameter", "Usable values", "Why"],
                        [
                            [row["Parameter"], row["Values"], row["Reason"]]
                            for row in report.skipped
                        ],
                        formats=[None, "%d", None],
                        filter_column="Parameter",
                    )
                ],
            )
        )
    if report.singletons:
        blocks.append(
            collapsed(
                f"Parameters that correlate with nothing above the threshold "
                f"({len(report.singletons)})",
                [
                    sortable(
                        ["Parameter"],
                        [[name] for name in report.singletons],
                        filter_column="Parameter",
                    )
                ],
            )
        )
    blocks.append(
        download(
            f"Download clusters.csv ({len(membership):,} rows)",
            "clusters.csv",
            csv_text(membership),
            note="Members, singletons and skipped parameters alike, with what happened to each.",
        )
    )
    return _section("clusters", "Correlated groups", blocks)


def _number_or_none(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and value not in (float("inf"), float("-inf")) else None


def _task_dataset(run: TaskRun) -> dict | None:
    result, spec = run.result, run.spec
    frame = run.dataset
    rows = [
        ["Shape", f"{result.rows:,} batches × {frame.shape[1]:,} columns"],
        ["What this table is", esc(result.dataset_role)],
        ["Preprocessing was fitted on", esc(result.fitted_on)],
        ["Target column", esc(f"{result.target_column} (last)")],
    ]
    if spec.roles.id:
        rows.append(["Identifier", esc(f"{spec.roles.id} (first)")])
    if spec.roles.group:
        rows.append(["Grouping column", esc(spec.roles.group)])
    rows.append(
        [
            "Transformed view",
            esc(
                "data/dataset_encoded.csv: the recipe fitted on the whole cohort (target "
                "encoding cross-fitted); refit encoding.json inside each fold you evaluate in"
            ),
        ]
    )
    rows.append(["Written to", esc(run.output_path)])

    blocks: list[Block | None] = [
        table(["", ""], [[bold(label), value] for label, value in rows]),
        plots.target_distribution(
            frame[result.target_column].tolist(),
            spec.target.column,
            unit=_unit_of(spec.target.column),
            classes=result.target_classes,
        ),
    ]

    if spec.roles.group and spec.roles.group in frame.columns:
        counts = frame[spec.roles.group].astype(str).value_counts()
        blocks.append(
            collapsed(
                f"Batches per {spec.roles.group} ({len(counts)} values)",
                [
                    sortable(
                        [spec.roles.group, "Batches"],
                        [[str(name), int(count)] for name, count in counts.items()],
                        formats=[None, "%d"],
                        filter_column=spec.roles.group,
                        note="Grouped cross-validation needs enough of these to split on.",
                    )
                ],
            )
        )

    preview_rows = min(spec.report.preview_rows, len(frame))
    if preview_rows:
        preview = frame.head(preview_rows)
        blocks.append(
            collapsed(
                f"The first {preview_rows} rows",
                [frame_block(preview.iloc[:, : min(12, preview.shape[1])])],
            )
        )
    text = frame.to_csv(index=False)
    if len(text) <= 2_000_000:
        blocks.append(
            download(
                f"Download dataset.csv ({result.rows:,} × {frame.shape[1]:,})",
                "dataset.csv",
                text,
                note="The same bytes as the file in the package.",
            )
        )
    else:
        blocks.append(
            md(
                f"The dataset is {len(text) / 1e6:.1f} MB, too large to carry inside this "
                "page. It is `data/dataset.csv` in the task folder."
            )
        )
    return _section("dataset", "The dataset", blocks)


def _task_provenance(run: TaskRun) -> dict | None:
    block = run.provenance
    spec = run.spec
    rows = [
        ["Run date", block.get("run_date") or "unknown"],
        ["Git commit", block.get("git_commit") or "unknown (not a git checkout)"],
        ["Task file", block.get("config") or "unknown"],
        ["Effective settings digest", (block.get("config_digest") or "unknown")[:16]],
        ["Seed", str(block.get("seed")) if block.get("seed") is not None else "not set"],
        ["Duration", str(datetime.now() - run.run_start).split(".")[0]],
    ]
    sources = [
        [
            esc(source["label"]),
            esc(source["kind"]),
            esc(source["location"]),
            esc(source.get("digest") or "could not be read"),
            esc(source.get("digest_of") or ""),
        ]
        for source in block.get("sources") or []
    ]
    environment = [[name, value] for name, value in (block.get("environment") or {}).items()]

    limitations = [
        "Every number here was computed before the file was written; the page recomputes nothing.",
        "A fresh report fetches its Python runtime from a CDN, so opening one needs network "
        "access once. It is not an offline file.",
    ]
    limitations.append(
        "There is no train/validation split: the preprocessing was fitted over the whole "
        "cohort. Cross-validate in the tool that fits the models, refitting the recipe in "
        "each fold."
    )

    return _section(
        "provenance",
        "Provenance and run log",
        [
            table(["", ""], [[bold(label), esc(value)] for label, value in rows]),
            table(
                ["Input", "Kind", "Where it was read from", "Digest", "Digest of"],
                sources,
            )
            if sources
            else None,
            table(["What this report does and does not do"], [[esc(line)] for line in limitations]),
            collapsed(
                "Effective task settings, defaults included",
                [
                    code(
                        yaml.safe_dump(spec.as_dict(), sort_keys=False, allow_unicode=True),
                        label="task.yaml",
                        language="yaml",
                    )
                ],
            ),
            collapsed(
                f"Library versions ({len(environment)})",
                [table(["Package", "Version"], [[esc(k), esc(v)] for k, v in environment])],
            ),
            _log(run)["blocks"][0] if _log(run) else None,
        ],
    )


def build_task_payload(run: TaskRun) -> dict[str, Any]:
    """One task's report, in the order a reader needs it."""
    retained = set(run.result.recipe.get("output_columns") or [])
    membership = decorrelate.rows(run.result.clusters, retained)
    sections = [
        _task_overview(run),
        _task_cohort(run),
        _task_quality(run),
        _task_parameters(run),
        _task_clusters(run, membership),
        _task_missingness(run),
        _task_dataset(run),
        _task_provenance(run),
    ]
    result = run.result
    return {
        "title": f"Task · {run.spec.name}",
        "lead": run.spec.question
        or f"{result.rows:,} batches and {result.feature_columns:,} predictor columns, "
        f"predicting {run.spec.target.column}.",
        "subtitle": f"CAR-T manufacturing data · run {run.run_start.strftime('%Y-%m-%d %H:%M')}",
        "sections": [section for section in sections if section],
    }
