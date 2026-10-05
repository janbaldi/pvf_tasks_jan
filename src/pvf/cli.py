"""The three stages, and the one place they are run from.

    pvf ptf      what the sources record that the PTF does not list
    pvf build    the PVF: every batch, every parameter, cleaned and derived
    pvf tasks    one task, as a folder: dataset, dictionary, decisions, report
    pvf all      the three in that order

They are in that order because each depends on the one before it. The PTF is the
schema, so it comes first; the build fills it in; a task narrows what the build
produced to one question.

Two things are decided here rather than by what happens to be installed. Which
sources a run reads is a setting (``sources.location``), not a consequence of an
internal package being importable, and uploading is a separate setting again —
reading from the share and writing to it are different decisions.

Paths in a config file are relative to that file's own folder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from . import (
    clean,
    dataset,
    enrich,
    features,
    io,
    merge,
    package,
    provenance,
    ptf,
    report,
    taskconfig,
)
from .decorrelate import RESIDUAL_SUFFIX
from .logger import log
from .taskconfig import TaskConfigError, TaskSpec

load_dotenv(override=True)

MODULE = "cli"
CONFIG_PATH = Path("config/config.yaml")

#: Config keys that name a file or a list of files, resolved against the config.
PATH_KEYS = (
    "ptf",
    "phf_ghent",
    "phf_raritan",
    "param_mapping",
    "remfg",
    "lv_coa",
    "raw_materials",
    "investigations",
    "site_maps",
    "pvf",
    "new_parameters",
    "reports",
    "tasks",
)

#: Where each artefact sits in a task's run folder. ``manifest.json`` is at the top.
TASK_LAYOUT = {
    "task.yaml": "config/task.yaml",
    "dataset.csv": "data/dataset.csv",
    "raw_features.csv": "data/raw_features.csv",
    "splits.csv": "data/splits.csv",
    "columns.csv": "metadata/columns.csv",
    "decisions.csv": "metadata/decisions.csv",
    "cohort.csv": "metadata/cohort.csv",
    "clusters.csv": "metadata/clusters.csv",
    "recipe.json": "metadata/recipe.json",
    "report.html": "report/report.html",
    "report_payload.json": "report/report_payload.json",
    "ptf_manifest.json": "provenance/ptf_manifest.json",
    "pvf_manifest.json": "provenance/pvf_manifest.json",
    "run.log": "logs/run.log",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the config once, with its paths resolved against its own folder."""
    path = Path(path).expanduser().resolve()
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} is not a mapping")
    missing = [key for key in ("paths", "features") if key not in config]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}")

    base = path.parent
    paths = config["paths"]
    for key in PATH_KEYS:
        value = paths.get(key)
        if isinstance(value, str):
            paths[key] = str(base / value) if not Path(value).is_absolute() else value
        elif isinstance(value, list):
            paths[key] = [
                str(base / item) if not Path(str(item)).is_absolute() else str(item)
                for item in value
            ]
    config["__path__"] = str(path)
    return config


def _sources(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("sources") or {}


def _remote_reader(config: dict[str, Any], name: str):
    """A SharePoint reader, but only where the config asked for one.

    Installing the internal package no longer changes what a run reads. A task
    that says ``location: sharepoint`` and cannot reach SharePoint fails, unless
    it also declared a fallback — in which case the fallback is recorded.
    """
    sources = _sources(config)
    if str(sources.get("location", "local")).lower() != "sharepoint":
        return None
    try:
        import io_sharepoint
    except ImportError as exc:
        if sources.get("fallback_to_local"):
            log.warn(
                MODULE,
                f"io_sharepoint is not installed — {name} falls back to the local copy, "
                "as sources.fallback_to_local allows",
            )
            return None
        raise RuntimeError(
            "sources.location is 'sharepoint' but io_sharepoint is not installed. "
            "Set sources.location: local, or sources.fallback_to_local: true"
        ) from exc
    return getattr(io_sharepoint, name)


def _report_path(config: dict[str, Any], stage: str) -> Path:
    directory = Path(config["paths"].get("reports", "outputs/reports"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stage}_report.html"


def _source(config: dict[str, Any], label: str, frame: Any, remote: bool) -> provenance.Source:
    """One input as this stage actually read it.

    Called right after the read, before anything cleans the table in place. A
    SharePoint read has no local bytes, so it is recorded by its location and a
    digest of the table that came back. Hashing the local path instead would
    either fail, because there is no copy, or record a copy the run never used.
    """
    path = str(config["paths"].get(label) or "")
    if frame is None:
        return provenance.missing(label, path, "could not be read; the stage went on without it")
    if remote and label in io.SHAREPOINT:
        return provenance.remote(label, io.sharepoint_location(label), frame)
    return provenance.local(label, path)


def _provenance(
    config: dict[str, Any], config_path: str | Path, sources: list[provenance.Source]
) -> dict:
    return provenance.capture(
        sources,
        seed=config.get("seed"),
        config_path=str(config_path),
        config={k: v for k, v in config.items() if k != "__path__"},
    )


def manifest_path(pvf: str | Path, stage: str) -> Path:
    """Where a stage's manifest lives: beside the PVF, which is where a task looks."""
    return Path(pvf).parent / f"{stage}_manifest.json"


def _outputs(files: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"what": what, "path": str(path), "sha256": provenance.file_sha256(str(path))}
        for what, path in files.items()
        if path
    ]


def _write_manifest(
    config: dict[str, Any],
    stage: str,
    run_start: datetime,
    mark: int,
    prov: dict,
    body: dict[str, Any],
) -> Path:
    """What a PTF or build run read, found and wrote, for a task to carry along."""
    events = log.since(mark)
    path = manifest_path(config["paths"]["pvf"], stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "status": "complete",
        "started": run_start.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        **body,
        "warnings": [e.message for e in events if e.level == "WARN"],
        "errors": [e.message for e in events if e.level == "ERROR"],
        "provenance": prov,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Stage 1: the PTF
# ---------------------------------------------------------------------------
def run_ptf(config: dict[str, Any], config_path: str | Path = CONFIG_PATH) -> Path:
    """Compare what the sources record against what the PTF lists."""
    run_start = datetime.now()
    mark = log.mark()
    paths = config["paths"]
    log.stage("PTF")

    read_excel = _remote_reader(config, "load_excel_from_sharepoint")
    remote = read_excel is not None

    ptf_cols, _ = io.load_ptf(paths["ptf"],sharepoint_loader=read_excel)
    ghent = io.load_phf_ghent(paths["phf_ghent"], sharepoint_loader=read_excel)
    raritan = io.load_phf_raritan(paths["phf_raritan"], sharepoint_loader=read_excel)
    mapping = io.load_site_mapping(paths["param_mapping"], sharepoint_loader=read_excel)
    investigations = io.load_investigations(
        paths.get("investigations", ""), sharepoint_loader=read_excel
    )
    sources = [
        provenance.local("ptf", paths["ptf"]),
        _source(config, "phf_ghent", ghent, remote),
        _source(config, "phf_raritan", raritan, remote),
        _source(config, "param_mapping", _mapping_table(mapping), remote),
        _source(config, "investigations", investigations, remote),
    ]

    found = ptf.find_new_parameters(
        ptf_cols,
        {
            "Ghent PHF": ghent,
            "Raritan batch data": raritan,
            "Investigations Power Query": investigations,
        },
        raritan_mapping=mapping,
    )

    if (config.get("ptf") or {}).get("write_new_parameters") and found.new_parameters:
        ptf.write_new_parameters(found, paths["new_parameters"])

    prov = _provenance(config, config_path, sources)
    path = report.write(
        report.build_ptf_payload(
            report.PtfRun(ptf=found, events=log.since(mark), provenance=prov, run_start=run_start)
        ),
        _report_path(config, "ptf"),
    )
    log.success(MODULE, f"Report written to {path}")
    manifest = _write_manifest(
        config,
        "ptf",
        run_start,
        mark,
        prov,
        {
            "ptf": {"path": paths["ptf"], "parameters": found.ptf_parameters},
            "sources_compared": found.sources_read,
            "sources_not_read": found.sources_absent,
            "not_in_ptf": found.by_source,
            "new_parameters": found.new_parameters,
            "outputs": _outputs({"report": path, "new parameters": found.written_to}),
        },
    )

    for label, names in found.by_source.items():
        if names:
            log.names(f"{label}: columns the PTF does not list", names)
        else:
            log.names(f"{label}: every column is in the PTF", [], level="SUCCESS")
    for label in found.sources_absent:
        log.names(f"{label}: not read, so not compared", [], level="ERROR")
    log.facts(
        "Written",
        [
            ("Report", str(path)),
            ("Manifest", str(manifest)),
            (
                "New parameters",
                found.written_to or "not written (ptf.write_new_parameters is off)",
            ),
        ],
    )
    log.summary(mark)
    return path


def _mapping_table(mapping: dict[str, str]) -> pd.DataFrame:
    """The name mapping as the table it was read into, for its provenance digest."""
    return pd.DataFrame(list(mapping.items()), columns=["Raritan", "Ghent"])


# ---------------------------------------------------------------------------
# Stage 2: the PVF
# ---------------------------------------------------------------------------
def run_build(
    config: dict[str, Any],
    mode: str = "dev",
    skip_write: bool = False,
    config_path: str | Path = CONFIG_PATH,
) -> Path:
    """Build the PVF from both sites and write its report."""
    run_start = datetime.now()
    mark = log.mark()
    paths = config["paths"]
    params = features.Params(**config["features"])
    log.stage("PVF build")
    log.info(MODULE, f"Mode: {mode.upper()}")

    read_excel = _remote_reader(config, "load_excel_from_sharepoint")
    remote = read_excel is not None

    # ── Load ───────────────────────────────────────────────────────────────
    log.section("01 · Loading")
    ptf_cols, ptf_mapping = io.load_ptf(paths["ptf"], sharepoint_loader=read_excel)

    df_ghent = io.load_phf_ghent(paths["phf_ghent"], sharepoint_loader=read_excel)
    raritan_raw = io.load_phf_raritan(paths["phf_raritan"], sharepoint_loader=read_excel)
    mapping = io.load_site_mapping(paths["param_mapping"], sharepoint_loader=read_excel)
    df_raritan = io.apply_column_mapping(raritan_raw, mapping)

    df_remfg = io.load_remanufacturing_supplement(paths["remfg"], sharepoint_loader=read_excel)
    df_raritan = io.attach_supplement(
        df_raritan, df_remfg, "Patient Lot/Batch #", ["Number of Prior Lines of Therapy"]
    )

    lv_coa = config.get("lv_coa") or {}
    df_lv_coa = io.load_lv_coa(
        paths["lv_coa"],
        ptf_cols,
        ph_nominal=lv_coa.get("ph_nominal"),
        osmo_nominal=lv_coa.get("osmo_nominal"),
        sharepoint_loader=read_excel
    )
    df_raw = io.load_raw_materials(paths["raw_materials"], ptf_cols, sharepoint_loader=read_excel)

    # Recorded now, before cleaning changes the site tables in place.
    sources = [
        provenance.local("ptf", paths["ptf"]),
        _source(config, "phf_ghent", df_ghent, remote),
        _source(config, "phf_raritan", raritan_raw, remote),
        _source(config, "param_mapping", _mapping_table(mapping), remote),
        _source(config, "remfg", df_remfg, remote),
        provenance.local("lv_coa", paths["lv_coa"]),
        provenance.local("raw_materials", paths["raw_materials"]),
    ] + [
        provenance.local(f"site_map_{index}", site_map)
        for index, site_map in enumerate(paths.get("site_maps") or [], start=1)
    ]

    phf = {"Ghent": df_ghent, "Raritan": df_raritan}
    site_shapes = {site: df.shape for site, df in phf.items()}

    # Where every column came from, collected as the columns arrive.
    ghent_cols, raritan_cols = set(df_ghent.columns), set(df_raritan.columns)
    origins: dict[str, list[str]] = {
        "Both sites' PHF": sorted(ghent_cols & raritan_cols),
        "Ghent PHF only": sorted(ghent_cols - raritan_cols),
        "Raritan PHF only": sorted(raritan_cols - ghent_cols),
        "Re-manufacturing supplement": ["Number of Prior Lines of Therapy"],
    }
    source_rows = [
        ["PTF", f"{len(ptf_cols):,}", "2", "loaded"],
        [
            "Vector certificates of analysis",
            f"{len(df_lv_coa):,}",
            f"{df_lv_coa.shape[1]:,}",
            "loaded",
        ],
        ["Raw materials and consumables", f"{len(df_raw):,}", f"{df_raw.shape[1]:,}", "loaded"],
        ["Re-manufacturing supplement", f"{len(df_remfg):,}", f"{df_remfg.shape[1]:,}", "loaded"],
    ]

    # ── Clean ──────────────────────────────────────────────────────────────
    log.section("02 · Cleaning")
    cleaning_report = clean.run_all(phf, ptf_cols, ptf_mapping, config.get("cleaning") or {})

    # ── Enrich: original parameters from other systems ────────────────────
    log.section("03 · Joining sources")
    origins["Databricks (Acridine Orange)"] = enrich.add_viability_parameters(
        phf, ptf_cols, enabled=bool(_sources(config).get("databricks"))
    )
    lv_coa_coverage, coa_columns = enrich.merge_lv_coa(phf, df_lv_coa)
    origins["Vector certificate of analysis"] = coa_columns
    origins["Raw materials and consumables"] = enrich.merge_raw_materials(phf, df_raw)
    unmapped_sites, site_columns = enrich.enrich_clinical_sites(phf, paths["site_maps"], sharepoint_loader=read_excel)
    origins["Clinical site mapping"] = site_columns
    enrich.harmonise_countries(phf)

    # ── Features: calculated from the parameters above ────────────────────
    log.section("04 · Derived features")
    feature_reports = {
        site: features.apply(df, ptf_cols, params, site=site) for site, df in phf.items()
    }

    # ── Merge ──────────────────────────────────────────────────────────────
    log.section("05 · Merge and checks")
    result = merge.merge_sites(phf)
    merge_report = merge.run_sanity_checks(result, phf, ptf_cols, ptf_mapping)
    origins["Merge"] = ["Site Merged"]

    # ── Write ──────────────────────────────────────────────────────────────
    log.section("06 · Output")
    out_path = Path(paths["pvf"])
    uploaded = None
    if not skip_write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_excel(out_path, index=False)
        log.success(MODULE, f"PVF written to {out_path}")
        uploaded = _upload(config, mode, result)
    elif mode.upper() == "PRD":
        log.info(MODULE, "--report-only: nothing was written and nothing was uploaded")

    prov = _provenance(config, config_path, sources)
    path = report.write(
        report.build_pvf_payload(
            report.BuildRun(
                result=result,
                phf=phf,
                site_shapes=site_shapes,
                ptf_cols=ptf_cols,
                ptf_mapping=ptf_mapping,
                cleaning=cleaning_report,
                features=feature_reports,
                merge=merge_report,
                events=log.since(mark),
                params=params,
                provenance=prov,
                origins=origins,
                lv_coa_coverage=lv_coa_coverage,
                raw_material_columns=origins["Raw materials and consumables"],
                unmapped_sites=unmapped_sites,
                source_rows=source_rows,
                run_start=run_start,
            )
        ),
        _report_path(config, "build"),
    )
    log.success(MODULE, f"Report written to {path}")

    written = [
        ("PVF", str(out_path) if not skip_write else "not written (--report-only)"),
        ("Report", str(path)),
    ]
    if skip_write:
        # The manifest describes the PVF on disk, which this run did not write.
        written.append(("Manifest", "not written (--report-only)"))
    else:
        manifest = _write_manifest(
            config,
            "pvf",
            run_start,
            mark,
            prov,
            {
                "mode": mode.upper(),
                "pvf": {
                    "path": str(out_path),
                    "sha256": provenance.file_sha256(str(out_path)),
                    "rows": int(result.shape[0]),
                    "columns": int(result.shape[1]),
                    "site_rows": merge_report.site_rows,
                },
                "ptf": {"path": paths["ptf"], "parameters": len(ptf_cols)},
                "ptf_parameters_not_in_pvf": merge_report.ptf_missing,
                "pvf_parameters_not_in_ptf": [
                    str(c) for c in result.columns if c not in set(ptf_cols)
                ],
                "uploaded_to": uploaded,
                "outputs": _outputs({"PVF": out_path, "report": path}),
            },
        )
        written.append(("Manifest", str(manifest)))
    if uploaded:
        written.append(("Uploaded to", uploaded))

    log.names("PTF parameters that are not in the PVF", merge_report.ptf_missing)
    log.facts(f"Written — {result.shape[0]:,} batches × {result.shape[1]:,} columns", written)
    log.summary(mark)
    return path


def _upload(config: dict[str, Any], mode: str, result: pd.DataFrame) -> str | None:
    """Put the PVF on the share, when a run is explicitly asked to.

    Two switches, both deliberate: the upload has to be enabled in config and the
    run has to be a PRD one. Reading from the share does not imply writing to it,
    and a test or a local run cannot upload by accident.
    """
    upload_config = config.get("upload") or {}
    if not upload_config.get("enabled"):
        return None
    if mode.upper() != "PRD":
        log.info(MODULE, f"Upload is enabled but this is a {mode.upper()} run — nothing uploaded")
        return None

    try:
        import io_sharepoint
    except ImportError:
        log.error(MODULE, "upload.enabled is set but io_sharepoint is not installed")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        staged = Path(tmpdir) / "PVF.xlsx"
        result.to_excel(staged, index=False)
        io_sharepoint.upload_file_to_sharepoint(
            local_path=staged,
            sharepoint_target_path="MS%26T%20MSAT%20Data%20Team//Reports/Adv%20Analystics%20%26%20AI/DATA/PVF/",
            drive_id="DRIVE_ID_GHENT",
        )
    log.success(MODULE, f"PVF uploaded to SharePoint: MS%26T%20MSAT%20Data%20Team//Reports/Adv%20Analystics%20%26%20AI/Data/PVF/PVF.xlsx")
    return "MS%26T%20MSAT%20Data%20Team//Reports/Adv%20Analystics%20%26%20AI/Data/PVF/PVF.xlsx"


# ---------------------------------------------------------------------------
# Stage 3: one task
# ---------------------------------------------------------------------------
#: What a finished task folder has to contain before it is called finished.
REQUIRED_ARTIFACTS = tuple(
    TASK_LAYOUT[name]
    for name in (
        "task.yaml",
        "dataset.csv",
        "columns.csv",
        "decisions.csv",
        "cohort.csv",
        "clusters.csv",
        "report.html",
        "report_payload.json",
        "run.log",
    )
)
REPORT_ONLY_ARTIFACTS = tuple(
    TASK_LAYOUT[name] for name in ("task.yaml", "report.html", "report_payload.json", "run.log")
)


def _cluster_rows(result: dataset.TaskResult) -> list[dict]:
    """Cluster membership, singletons and skips — the whole population, not part."""
    rows: list[dict] = []
    retained = set(result.recipe.get("output_columns") or [])
    for cluster in result.clusters.clusters:
        for member in cluster.members:
            fit = next((f for f in cluster.fits if f.column == member), None)
            representative = member == cluster.representative
            rows.append(
                {
                    "Cluster": cluster.name,
                    "Parameter": member,
                    "Representative": cluster.representative,
                    "Is representative": representative,
                    "Action": cluster.action,
                    "Outcome": (
                        "kept as the representative"
                        if representative
                        else "dropped"
                        if member in cluster.dropped
                        else fit.model
                        if fit
                        else "unchanged"
                    ),
                    "Output column": (
                        member
                        if representative or not fit or fit.fallback
                        else f"{member}{RESIDUAL_SUFFIX}"
                    ),
                    "Retained in dataset": (
                        member in retained
                        or f"{member}{RESIDUAL_SUFFIX}" in retained
                        or result.clusters.action == "report_only"
                    ),
                    "Rows fitted": fit.rows if fit else cluster.representative_rows,
                    "Adjusted R2": fit.adj_r2 if fit else "",
                    "Regressors": ", ".join(fit.regressors) if fit else "",
                    "Smallest shared batches in cluster": cluster.min_overlap,
                }
            )
    for column in result.clusters.singletons:
        rows.append(
            {
                "Cluster": "",
                "Parameter": column,
                "Representative": column,
                "Is representative": True,
                "Action": "none",
                "Outcome": "correlates with nothing above the threshold",
                "Output column": column,
                "Retained in dataset": column in retained,
                "Rows fitted": "",
                "Adjusted R2": "",
                "Regressors": "",
                "Smallest shared batches in cluster": "",
            }
        )
    for skipped in result.clusters.skipped:
        rows.append(
            {
                "Cluster": "",
                "Parameter": skipped["Parameter"],
                "Representative": "",
                "Is representative": False,
                "Action": "skipped",
                "Outcome": skipped["Reason"],
                "Output column": "",
                "Retained in dataset": skipped["Parameter"] in retained,
                "Rows fitted": skipped["Values"],
                "Adjusted R2": "",
                "Regressors": "",
                "Smallest shared batches in cluster": "",
            }
        )
    return rows


def _check(result: dataset.TaskResult, final: pd.DataFrame) -> None:
    """The counts in the package have to be the same counts. Then it is complete."""
    problems = []
    if len(final) != result.rows:
        problems.append(f"the dataset has {len(final)} rows, the run recorded {result.rows}")
    if len(result.columns) != final.shape[1]:
        problems.append(
            f"columns.csv has {len(result.columns)} rows for {final.shape[1]} dataset columns"
        )
    exported = {row["Column"] for row in result.columns}
    if exported != set(final.columns):
        problems.append("columns.csv and the dataset do not name the same columns")
    if result.cohort.funnel:
        left = result.cohort.funnel[-1]["Batches left"]
        if left != result.rows:
            problems.append(
                f"the cohort funnel ends at {left} batches but the dataset has {result.rows}"
            )
    if problems:
        raise ValueError("The task package is inconsistent: " + "; ".join(problems))


def _carry_upstream(pkg: package.TaskPackage, spec: TaskSpec, pvf_digest: str | None) -> dict:
    """Copy the PTF and build manifests into the run, and check they fit its PVF.

    They are what says how the PVF this task read was made. A PVF manifest whose
    digest is not this PVF's describes some other build, and is said to.
    """
    upstream: dict[str, str] = {}
    for stage in ("ptf", "pvf"):
        source = manifest_path(spec.pvf, stage)
        target = TASK_LAYOUT[f"{stage}_manifest.json"]
        if not source.exists():
            message = (
                f"No {stage.upper()} manifest beside the PVF, so the task folder cannot say "
                f"how its input was made — rerun `pvf {'ptf' if stage == 'ptf' else 'build'}`"
            )
            log.warn(MODULE, message, str(source))
            pkg.warnings.append(message)
            upstream[stage] = f"not found at {source}"
            continue
        text = source.read_text(encoding="utf-8")
        pkg.write_text(target, text)
        upstream[stage] = target
        if stage == "pvf":
            recorded = (json.loads(text).get("pvf") or {}).get("sha256")
            if recorded and pvf_digest and recorded != pvf_digest:
                message = (
                    "The PVF manifest describes a different file than the PVF this task read: "
                    "the PVF changed after that build, or the manifest is stale"
                )
                log.warn(MODULE, message, f"manifest {recorded[:12]}, PVF {pvf_digest[:12]}")
                pkg.warnings.append(message)
    return upstream


def run_task(config: dict[str, Any], spec: TaskSpec, report_only: bool = False) -> Path:
    """Run one task and leave a complete folder behind, or a failed one.

    ``--report-only`` writes the report and its supporting metadata into a fresh
    folder of its own. It still runs the whole analysis — there is no other way
    to have anything to report — but it writes no dataset and uploads nothing,
    and its manifest says the package is a report, not a dataset.
    """
    run_start = datetime.now()
    mark = log.mark()
    log.stage(f"Task '{spec.name}'")
    pkg = package.open_package(spec.output_root, spec.name)
    manifest: dict[str, Any] = {"mode": "report only" if report_only else "full package"}

    read_excel = _remote_reader(config, "load_excel_from_sharepoint")
    remote = read_excel is not None

    try:
        pkg.write_text(
            TASK_LAYOUT["task.yaml"],
            "# The effective settings of this run, defaults included.\n"
            + yaml.safe_dump(spec.as_dict(), sort_keys=False, allow_unicode=True),
        )
        pvf = io.load_pvf(spec.pvf, sharepoint_loader=read_excel)
        ptf_frame = io.read_ptf(spec.ptf, sharepoint_loader=read_excel)
        final, raw_table, result = dataset.build(pvf, ptf_frame, spec)
        _check(result, final)

        prov = provenance.capture(
            [provenance.local("pvf", spec.pvf), provenance.local("ptf", spec.ptf)],
            seed=spec.seed,
            config_path=spec.source,
            config=spec.as_dict(),
        )
        upstream = _carry_upstream(pkg, spec, prov["inputs"].get("pvf"))

        if not report_only:
            pkg.write_csv(TASK_LAYOUT["dataset.csv"], final)
            if not spec.predictive:
                # The transformed table is dataset.csv here, so the untransformed
                # predictors go beside it. For a predictive task dataset.csv is
                # already that table and a second copy would say nothing new.
                pkg.write_csv(TASK_LAYOUT["raw_features.csv"], raw_table)
            pkg.write_csv(TASK_LAYOUT["columns.csv"], result.columns)
            pkg.write_csv(TASK_LAYOUT["decisions.csv"], [d.row() for d in result.decisions])
            pkg.write_csv(TASK_LAYOUT["cohort.csv"], result.cohort.exclusions)
            pkg.write_csv(TASK_LAYOUT["clusters.csv"], _cluster_rows(result))
            pkg.write_json(TASK_LAYOUT["recipe.json"], result.recipe)
            if result.split:
                pkg.write_csv(
                    TASK_LAYOUT["splits.csv"],
                    [
                        {
                            "Batch": str(final.at[index, spec.roles.id])
                            if spec.roles.id in final.columns
                            else f"row {index}",
                            "Split": final.at[index, "split"],
                        }
                        for index in final.index
                    ],
                )

        payload = report.build_task_payload(
            report.TaskRun(
                dataset=final,
                result=result,
                spec=spec,
                events=log.since(mark),
                provenance=prov,
                output_path=""
                if report_only
                else str(pkg.destination / TASK_LAYOUT["dataset.csv"]),
                report_only=report_only,
                run_start=run_start,
            )
        )
        pkg.write_json(TASK_LAYOUT["report_payload.json"], payload)
        report.write(payload, pkg.path_for(TASK_LAYOUT["report.html"]))

        pkg.warnings.extend(result.warnings)
        pkg.write_text(TASK_LAYOUT["run.log"], _log_text(log.since(mark)))
        destination = pkg.complete(
            {
                **manifest,
                "question": spec.question,
                "purpose": spec.purpose,
                "dataset_role": "not written (report only)" if report_only else result.dataset_role,
                "provenance": prov,
                "upstream_manifests": upstream,
                "config_digest": prov.get("config_digest"),
                "environment": prov.get("environment"),
                "counts": {
                    "pvf_rows": result.cohort.raw_rows,
                    "cohort_rows": result.cohort.rows,
                    "dataset_rows": result.rows,
                    "dataset_columns": int(final.shape[1]),
                    "predictor_columns": result.feature_columns,
                    "candidates": len(result.candidates),
                    "clusters": len(result.clusters.clusters),
                    "decisions": len(result.decisions),
                },
            },
            required=REPORT_ONLY_ARTIFACTS if report_only else REQUIRED_ARTIFACTS,
        )
    except Exception as exc:
        pkg.write_text(TASK_LAYOUT["run.log"], _log_text(log.since(mark)))
        pkg.fail(f"{type(exc).__name__}: {exc}", manifest)
        raise

    log.facts(
        f"Written — {result.rows:,} batches × {final.shape[1]:,} columns",
        [
            ("Task folder", str(destination)),
            ("Report", str(destination / TASK_LAYOUT["report.html"])),
            (
                "Dataset",
                "not written (--report-only)"
                if report_only
                else str(destination / TASK_LAYOUT["dataset.csv"]),
            ),
            ("Manifest", str(destination / package.MANIFEST)),
        ],
    )
    log.summary(mark)
    return destination


def _log_text(events) -> str:
    return "\n".join(
        " ".join(
            filter(
                None,
                [
                    event.ts.strftime("%Y-%m-%d %H:%M:%S"),
                    event.level,
                    event.module,
                    event.message,
                    f"| {event.detail}" if event.detail else "",
                ],
            )
        )
        for event in events
    )


def load_task(args, config: dict[str, Any] | None = None) -> TaskSpec:
    """The task to run: a task YAML if one was given, else the old config block."""
    if getattr(args, "task", None):
        return taskconfig.load(args.task)
    if config is None:
        config = load_config(args.config)
    if not config.get("tasks"):
        raise TaskConfigError(
            f"{args.config} has no tasks: section and no --task file was given. "
            "Write a task YAML (see README) or pass --task"
        )
    return taskconfig.from_legacy(config, config["__path__"])


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pvf", description="CAR-T manufacturing data: the PTF, the PVF, and tasks from it"
    )
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Config file to run from")
    subcommands = parser.add_subparsers(dest="stage", required=True)

    subcommands.add_parser("ptf", help="what the sources record that the PTF does not list")
    build = subcommands.add_parser("build", help="build the PVF from both sites")
    build.add_argument("--mode", default=os.getenv("MODE", "dev"), help="PRD or DEV")
    build.add_argument(
        "--report-only", action="store_true", help="build the report without writing the PVF"
    )
    tasks = subcommands.add_parser("tasks", help="run one task and write its folder")
    tasks.add_argument("--task", help="a task YAML; without it, the config's tasks: section")
    tasks.add_argument(
        "--report-only",
        action="store_true",
        help="write only the report and its metadata, into a folder of their own",
    )
    everything = subcommands.add_parser("all", help="run the three stages in order")
    everything.add_argument("--mode", default=os.getenv("MODE", "dev"), help="PRD or DEV")
    everything.add_argument("--task", help="a task YAML for the third stage")

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.stage in ("ptf", "all"):
        run_ptf(config, config_path=config["__path__"])
    if args.stage in ("build", "all"):
        run_build(
            config,
            mode=getattr(args, "mode", "dev"),
            skip_write=getattr(args, "report_only", False),
            config_path=config["__path__"],
        )
    if args.stage in ("tasks", "all"):
        run_task(config, load_task(args, config), report_only=getattr(args, "report_only", False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
