"""Assembling a run into the report payload.

One report per stage, built the same way. The payload is an ordered list of
sections, each an ordered list of blocks. This module writes no markup and draws
no figures: :mod:`pvf.blocks` says what a block is, :mod:`pvf.plots` builds the
figures, :mod:`pvf.streamlit_app` renders the payload in the browser and
:mod:`pvf.stlite` wraps that page into one file. Nothing imports back the other
way.

Each stage answers the question it is for. The PTF report: what do the sources
record that the schema has never been told about. The build report: what came
out, what the data needed doing to it, where every parameter came from, and which
PTF parameters the PVF still does not have. The tasks report: which batches are
in the cohort, what each parameter became, and what every column of the dataset
means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import plots, stlite
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
    num,
    sortable,
    table,
)
from .blocks import (
    frame as frame_block,
)
from .features import BLOCKED, CREATED, ERRORED, NOT_IN_PTF, PRESENT, FeatureReport, Params

DERIVED = "Derived"
GROWTH_ORDER = ("undergrowth", "normal", "overgrowth")
CD4_ORDER = ("low CD4", "medium CD4", "high CD4")
#: Below this share of batches, a column is drawn as a warning in the inventory.
THIN_COVERAGE = 5.0


@dataclass
class BuildRun:
    """Everything the build's report needs, collected as the run goes."""

    result: pd.DataFrame
    phf: dict[str, pd.DataFrame]
    site_shapes: dict[str, tuple[int, int]]
    ptf_cols: list[str]
    ptf_mapping: dict[str, str]
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
    absent = sum(report.count(NOT_IN_PTF) for report in run.features.values())

    rows = [
        ["Batches in the PVF", f"{run.merge.total_rows:,}"],
        ["Parameters in the PVF", f"{run.merge.total_columns:,}"],
        ["PTF parameters", f"{len(run.ptf_cols):,}"],
        ["PTF parameters the PVF does not have", f"{len(run.merge.ptf_missing):,}"],
        ["Features computed", f"{created:,}"],
        ["Features blocked by a missing input", f"{blocked:,}"],
        ["Features waiting on the PTF", f"{absent:,}"],
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
        rows.append([f"{esc(site)} PHF, as loaded", f"{n_rows:,}", f"{n_columns:,}", "loaded"])
    merged = [[esc(site), f"{count:,}"] for site, count in run.merge.site_rows.items()] + [
        [bold("Merged PVF"), bold(f"{run.merge.total_rows:,}")]
    ]

    return _section(
        "sources",
        "Sources",
        [
            table(["Source", "Rows", "Columns", "Status"], rows),
            table(
                ["Site", "Batches in the PVF"],
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

    if report.integrity_changes:
        blocks.append(
            collapsed(
                f"Integrity corrections ({len(report.integrity_changes)})",
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
                            for c in report.integrity_changes
                        ],
                        note="Established corrections against the source systems, not general "
                        "truths. Which of them run is a config decision.",
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
                "as 0.5 and not the same as missing, so the policy is recorded with it.",
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
                        ["Parameter", "Ghent only", "Raritan only"],
                        [
                            [
                                esc(c["column"]),
                                esc(", ".join(c["ghent_only"][:5]) or "—"),
                                esc(", ".join(c["raritan_only"][:5]) or "—"),
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
    """Where each column in the PVF came from.

    Derived features name their group, so the inventory distinguishes a
    calculation from a measurement at a glance.
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


def _inventory(run: BuildRun) -> dict | None:
    """One row per PVF parameter: where it came from and how much of it there is."""
    origins = _origin_index(run)
    sites = list(run.phf)
    rows = []
    for column in run.result.columns:
        coverage = 100 * run.result[column].notna().mean() if len(run.result) else 0.0
        rows.append(
            [
                str(column),
                origins.get(column, "PHF"),
                str(run.ptf_mapping.get(column, "—")),
                "yes" if column in set(run.ptf_cols) else "no",
                *["yes" if column in run.phf[site].columns else "—" for site in sites],
                round(float(coverage), 1),
            ]
        )
    if not rows:
        return None

    derived = sum(1 for row in rows if row[1].startswith(DERIVED))
    return _section(
        "inventory",
        "Parameter inventory",
        [
            sortable(
                ["Parameter", "Origin", "Value type", "In PTF", *sites, "Coverage %"],
                rows,
                formats=[None, None, None, None, *[None] * len(sites), "%.1f"],
                flag={"column": "Coverage %", "below": THIN_COVERAGE},
                filter_column="Parameter",
                note=f"{len(rows):,} parameters, of which {derived:,} are calculated "
                "from other parameters. Coverage is the share of merged batches that "
                "have a value; anything under "
                f"{THIN_COVERAGE:.0f}% is marked. Click a column header to sort.",
            )
        ],
    )


def _missing_reason(run: BuildRun, parameter: str) -> str:
    """Why a PTF parameter is not in the PVF, in the pipeline's own terms."""
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


def _ptf_coverage(run: BuildRun) -> dict | None:
    ptf = set(run.ptf_cols)
    missing = run.merge.ptf_missing
    extra = [c for c in run.result.columns if c not in ptf]

    blocks: list[Block | None] = []
    if missing:
        blocks.append(
            sortable(
                ["PTF parameter", "Value type", "Why it is not in the PVF"],
                [
                    [str(p), str(run.ptf_mapping.get(p, "—")), _missing_reason(run, p)]
                    for p in missing
                ],
                filter_column="PTF parameter",
                note=f"{len(missing):,} of {len(run.ptf_cols):,} PTF parameters are "
                "absent from the merged PVF. A parameter no source supplies has to come "
                "from the source system; a blocked feature needs the inputs it names.",
            )
        )
    else:
        blocks.append(md("Every PTF parameter is present in the merged PVF."))

    if extra:
        blocks.append(
            collapsed(
                f"PVF parameters the PTF does not list ({len(extra)})",
                [
                    sortable(
                        ["Parameter", "Origin"],
                        [[str(c), _origin_index(run).get(c, "PHF")] for c in extra],
                        filter_column="Parameter",
                        note="These reach the PVF because a source carries them. The "
                        "PTF decides whether they belong there.",
                    )
                ],
            )
        )

    only = [["Ghent only", c] for c in run.merge.ghent_only_cols] + [
        ["Raritan only", c] for c in run.merge.raritan_only_cols
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
                        note="Outside the PTF name mapping, so the other site's rows are "
                        "empty for these.",
                    )
                ],
            )
        )

    return _section("ptf-coverage", "PTF coverage", blocks)


def _feature_status_table(entries: list[dict], status: str, note: str) -> Block | None:
    rows = [
        [entry["site"], entry["name"], entry["group"], entry["process_day"], entry["note"]]
        for entry in entries
        if entry["status"] == status
    ]
    if not rows:
        return None
    return sortable(
        ["Site", "Feature", "Group", "Day", "Detail"],
        rows,
        filter_column="Feature",
        note=note,
    )


def _profile_table(
    profiles: dict[str, dict[str, dict[str, int]]], column: str, order: tuple[str, ...]
) -> Block | None:
    """Batches per profile class and their share, per site."""
    rows = []
    for site, by_column in profiles.items():
        counts = by_column.get(column) or {}
        total = sum(counts.values())
        classes = list(order) + sorted(c for c in counts if c not in order)
        for name in classes:
            if name in counts:
                rows.append(
                    [
                        esc(site),
                        esc(name),
                        f"{counts[name]:,}",
                        f"{100 * counts[name] / total:.1f}%" if total else "—",
                    ]
                )
    if not rows:
        return None
    return table(
        ["Site", esc(column), "Batches", "Share"],
        rows,
        note="Batches with no reading get no profile, so the shares are of the "
        "batches that have one.",
    )


def _features(run: BuildRun) -> dict | None:
    entries = _entries(run)
    if not entries:
        return None

    blocks: list[Block | None] = [
        plots.registry_outcome(list(run.features.values())),
        sortable(
            ["Site", "Feature", "Group", "Day", "Type", "Coverage %"],
            [
                [
                    entry["site"],
                    entry["name"],
                    entry["group"],
                    entry["process_day"],
                    entry["value_type"],
                    round(100 * entry["non_null"] / run.features[entry["site"]].rows, 1)
                    if run.features[entry["site"]].rows
                    else 0.0,
                ]
                for entry in entries
                if entry["status"] == CREATED
            ],
            formats=[None, None, None, None, None, "%.1f"],
            flag={"column": "Coverage %", "below": THIN_COVERAGE},
            filter_column="Feature",
            note="Features the run computed, with the share of that site's batches "
            "that ended up with a value.",
        ),
        _feature_status_table(
            entries,
            BLOCKED,
            "Computed as soon as the inputs named here reach the PVF. Nothing else is needed.",
        ),
        _feature_status_table(
            entries,
            NOT_IN_PTF,
            "The calculation is written and its inputs may well be there, but the PTF "
            "does not list the parameter, so the run does not create it. Add the row to "
            "the PTF and the next run picks it up.",
        ),
        _feature_status_table(entries, ERRORED, "The calculation raised. These need looking at."),
        _feature_status_table(
            entries, PRESENT, "A source already supplies these, so the registry stood aside."
        ),
    ]

    catalogue = collapsed(
        f"What each feature is for ({len({e['name'] for e in entries})})",
        [
            sortable(
                ["Feature", "Group", "Day", "Type", "Inputs", "Why"],
                [
                    [
                        entry["name"],
                        entry["group"],
                        entry["process_day"],
                        entry["value_type"],
                        entry["inputs"],
                        entry["rationale"],
                    ]
                    for entry in {e["name"]: e for e in entries}.values()
                ],
                filter_column="Feature",
                note="The registry as it stands, whatever each site managed to compute. "
                "These columns are what the PTF needs to accept a new parameter.",
            )
        ],
    )
    blocks.append(catalogue)

    profiles = {site: report.profiles for site, report in run.features.items()}
    for column, order in (("growth profile", GROWTH_ORDER), ("CD4 profile", CD4_ORDER)):
        blocks.append(_profile_table(profiles, column, order))
        blocks.append(plots.profile_distribution(profiles, column, order))
    for report in run.features.values():
        blocks.append(plots.coverage_bars(report))

    return _section("features", "Derived features", blocks)


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
                f"Raritan clinical site acronyms with no institution ({len(run.unmapped_sites)})",
                [
                    table(
                        ["Acronym"],
                        [[esc(a)] for a in run.unmapped_sites],
                        note="These stay as acronyms in the PVF. Adding them to a mapping "
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
def build_pvf_payload(run: BuildRun) -> dict[str, Any]:
    """The build's report: a title, a lead, and the sections that had something."""
    sections = [
        _overview(run),
        _sources(run),
        _cleaning(run),
        _inventory(run),
        _ptf_coverage(run),
        _features(run),
        _enrichment(run),
        _log(run),
        _provenance(run),
    ]
    stamp = run.run_start.strftime("%Y-%m-%d %H:%M")
    return {
        "title": "PVF build",
        "lead": f"{run.merge.total_rows:,} batches and "
        f"{run.merge.total_columns:,} parameters from Ghent and Raritan, built "
        f"against {len(run.ptf_cols):,} PTF parameters.",
        "subtitle": f"CAR-T manufacturing data · run {stamp}",
        "sections": [section for section in sections if section],
    }


def write(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Render a payload into one self-contained HTML file."""
    return stlite.write(payload, output_path)


# ---------------------------------------------------------------------------
# Stage 1: the PTF report
# ---------------------------------------------------------------------------
@dataclass
class PtfRun:
    """What the PTF stage found, and what produced it."""

    ptf: Any
    events: list
    provenance: dict
    params: Any = None
    run_start: datetime = field(default_factory=datetime.now)


def _ptf_overview(run: PtfRun) -> dict | None:
    report = run.ptf
    rows = [
        ["Parameters in the PTF", f"{report.ptf_parameters:,}"],
        ["Sources compared", f"{len(report.sources_read):,}"],
        ["Source columns the PTF does not list", f"{report.total:,}"],
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


def _ptf_sources(run: PtfRun) -> dict | None:
    report = run.ptf
    rows = [
        [esc(label), f"{columns:,}", f"{len(report.by_source.get(label, [])):,}", "read"]
        for label, columns in report.sources_read.items()
    ]
    rows += [[esc(label), "—", "—", "not available"] for label in report.sources_absent]
    return _section(
        "sources",
        "Sources",
        [table(["Source", "Columns", "Not in the PTF", "Status"], rows)],
    )


def _ptf_new(run: PtfRun) -> dict | None:
    report = run.ptf
    if not report.new_parameters:
        return _section(
            "new",
            "Parameters new to the PTF",
            [md("Every column in every source is already a PTF parameter.")],
        )

    source_of: dict[str, list[str]] = {}
    for label, names in report.by_source.items():
        for name in names:
            source_of.setdefault(name, []).append(label)

    return _section(
        "new",
        "Parameters new to the PTF",
        [
            sortable(
                ["Parameter", "Recorded by"],
                [[str(name), ", ".join(source_of.get(name, []))] for name in report.new_parameters],
                filter_column="Parameter",
                note=f"{len(report.new_parameters):,} distinct names the sources record and the "
                "PTF does not list. Until the PTF names one, the build cannot carry it into the "
                "PVF — so this is the list to work through, not a list of errors.",
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


def build_ptf_payload(run: PtfRun) -> dict[str, Any]:
    """The PTF stage's report."""
    sections = [
        _ptf_overview(run),
        _ptf_sources(run),
        _ptf_new(run),
        _log(run),
        _provenance(run, parameters=[]),
    ]
    report = run.ptf
    return {
        "title": "PTF coverage",
        "lead": f"{len(report.new_parameters):,} parameters the sources record and the PTF "
        f"does not list, against {report.ptf_parameters:,} it does.",
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
    report_only: bool = False
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
        ["Intended use", esc(spec.purpose)],
        [
            "Target",
            esc(f"{spec.target.column}{f' ({unit})' if unit else ''} — {result.target_kind}"),
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
        [
            "This run",
            "report only — no dataset was written"
            if run.report_only
            else "complete task package written",
        ],
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


def _unit_of(name: str) -> str:
    import re as _re

    match = _re.search(r"\(([^()]{1,24})\)\s*$", str(name))
    return match.group(1) if match else ""


def _task_cohort(run: TaskRun) -> dict | None:
    result = run.result
    funnel = [
        [
            bold("Every batch in the PVF"),
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
                            note="They stay in the PVF. metadata/cohort.csv in the package "
                            "has every batch with the filter that excluded it.",
                        )
                    ],
                )
            )

    excluded = [row for row in result.cohort.exclusions if not row["Included"]]
    blocks.append(
        download(
            f"Download cohort.csv ({len(result.cohort.exclusions):,} batches, "
            f"{len(excluded):,} excluded)",
            "cohort.csv",
            csv_text(result.cohort.exclusions),
            note="One row per batch in the PVF: whether it is in the dataset, and the first "
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
                f"Values not in the PTF's category list ({len(result.ordinal_unexpected)})",
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
    if result.missing_from_pvf:
        blocks.append(
            collapsed(
                f"PTF parameters the PVF does not have ({len(result.missing_from_pvf)})",
                [
                    sortable(
                        ["Parameter"],
                        [[str(p)] for p in result.missing_from_pvf],
                        filter_column="Parameter",
                        note="They could not become predictors. The build's own report says "
                        "why each one is absent.",
                    )
                ],
            )
        )
    if result.duplicate_ptf_parameters:
        blocks.append(
            table(
                ["PTF parameter listed more than once"],
                [[esc(name)] for name in result.duplicate_ptf_parameters],
                note="Each was used once.",
            )
        )
    return _section("quality", "Data quality", blocks)


def _task_parameters(run: TaskRun) -> dict | None:
    result = run.result
    decisions = [d.row() for d in result.decisions]
    blocks: list[Block | None] = [
        md(
            f"**{len(result.candidates):,}** of the PVF's parameters were allowed to be "
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
            f"Download columns.csv ({len(result.columns):,} columns)",
            "columns.csv",
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
            f"Download decisions.csv ({len(decisions):,} decisions)",
            "decisions.csv",
            csv_text(decisions),
        ),
    ]

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
    return _section("parameters", "Parameters and encodings", blocks)


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
                "report_only": "The groups are described here and every parameter went into "
                "the dataset unchanged.",
                "representative": "One member of each group was kept and the others dropped.",
                "linear": "One member of each group was kept and the others replaced by what "
                "is left of them once the retained ones are accounted for.",
                "nonlinear": "One member was kept and the others residualised against it by "
                "whichever curve fitted — an experimental option.",
            }.get(report.action, "")
        )
    ]
    if report.transforms:
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
                        cluster.min_overlap,
                        _number_or_none(cluster.corr_before),
                        _number_or_none(cluster.corr_after),
                        _number_or_none(cluster.vif_before),
                        _number_or_none(cluster.vif_after),
                        cluster.vif_rows_before,
                    ]
                    for cluster in report.clusters
                ],
                formats=[None, "%d", None, "%d", "%.2f", "%.2f", "%.1f", "%.1f", "%d"],
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
    if result.split:
        rows.append(
            [
                "Split",
                esc(
                    f"{result.split['strategy']}: {result.split['training']} training, "
                    f"{result.split['validation']} validation — {result.split['detail']}"
                ),
            ]
        )
    rows.append(["Written to", esc(run.output_path or "nothing — this is a report-only run")])

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
    if not run.result.split:
        limitations.append(
            "No train/validation split was requested, so the preprocessing was fitted over the "
            "whole cohort. Treat the transformed view as exploratory."
        )
    if run.report_only:
        limitations.append(
            "This run wrote a report and its metadata only. No dataset was written, nothing "
            "was uploaded, and the analysis behind the report still ran in full."
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
    from .cli import _cluster_rows

    membership = _cluster_rows(run.result)
    sections = [
        _task_overview(run),
        _task_cohort(run),
        _task_quality(run),
        _task_parameters(run),
        _task_clusters(run, membership),
        _task_dataset(run),
        _task_provenance(run),
    ]
    result = run.result
    return {
        "title": f"Task · {run.spec.name}",
        "lead": run.spec.question
        or f"{result.rows:,} batches and {result.feature_columns:,} predictor columns, "
        f"predicting {run.spec.target.column}.",
        "subtitle": f"CAR-T manufacturing data · run {run.run_start.strftime('%Y-%m-%d %H:%M')}"
        + (" · report only" if run.report_only else ""),
        "sections": [section for section in sections if section],
    }
