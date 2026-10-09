"""The commands, and the one place the stages are run from.

    pvf init <folder> [--demo]   a new workspace: config, corrections, folders
    pvf new-task <name|path>     a task file, inside the workspace or anywhere else
    pvf check [--task ...]       validate the config and tasks; run nothing
    pvf ptf                      what the sources record that the PTF does not list
    pvf build                    the PVF: every batch, every parameter, cleaned and derived
    pvf tasks [--task ...]       each task, as a folder: dataset, dictionary, decisions, report
    pvf all                      ptf, build and tasks, in that order
    pvf apply --run <folder>     a task's fitted recipe, applied to new batches

The stages are in that order because each depends on the one before it. The PTF
is the schema, so it comes first; the build fills it in; a task narrows what the
build produced to one question.

Code, configuration and data are kept apart: this package is the code, and a
*workspace* (any folder with a ``pvf.yaml``) holds the configuration and the
data. Commands find the workspace config by ``--config``, ``$PVF_CONFIG``, or the
nearest ``pvf.yaml`` above the working directory or the task file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv

from . import (
    clean,
    corrections,
    dataset,
    enrich,
    features,
    io,
    manifest,
    merge,
    missingness,
    package,
    provenance,
    ptf,
    recipe,
    report,
    scaffold,
    taskconfig,
)
from . import config as workspace
from .config import ConfigError
from .decorrelate import RESIDUAL_SUFFIX
from .logger import log
from .taskconfig import TaskConfigError, TaskSpec

# A .env in the working directory (or above it); the workspace's own .env is
# loaded with its config, see pvf.config.load_env.
load_dotenv(find_dotenv(usecwd=True))

MODULE = "cli"
#: Kept for callers of the old module-level name; the config is now found by
#: :func:`pvf.config.find`.
CONFIG_PATH = workspace.LEGACY_CONFIG

#: Where each artefact sits in a task's run folder. ``manifest.json`` is at the top.
TASK_LAYOUT = {
    "task.yaml": "config/task.yaml",
    "dataset.csv": "data/dataset.csv",
    "raw_features.csv": "data/raw_features.csv",
    "transformed.csv": "data/transformed.csv",
    "splits.csv": "data/splits.csv",
    "columns.csv": "metadata/columns.csv",
    "features.csv": "metadata/features.csv",
    "decisions.csv": "metadata/decisions.csv",
    "cohort.csv": "metadata/cohort.csv",
    "clusters.csv": "metadata/clusters.csv",
    "missingness.csv": "metadata/missingness.csv",
    "recipe.json": "metadata/recipe.json",
    "report.html": "report/report.html",
    "report_payload.json": "report/report_payload.json",
    "ptf_manifest.json": "provenance/ptf_manifest.json",
    "pvf_manifest.json": "provenance/pvf_manifest.json",
    "run.log": "logs/run.log",
}

#: Errors that are the user's to fix, shown as a message rather than a traceback.
USER_ERRORS = (
    ConfigError,
    TaskConfigError,
    dataset.TaskRefused,
    corrections.CorrectionsError,
    FileNotFoundError,
    FileExistsError,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a workspace config, validated, with its paths resolved."""
    return workspace.load(path)


def manifest_path(pvf: str | Path, stage: str) -> Path:
    """Where a stage's manifest lives: beside the PVF, which is where a task looks."""
    return Path(pvf).parent / f"{stage}_manifest.json"


def _report_path(config: dict[str, Any], stage: str) -> Path:
    directory = Path(
        config["paths"].get("reports") or Path(config["__dir__"]) / "outputs" / "reports"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stage}_report.html"


def _provenance(config: dict[str, Any], sources: list[provenance.Source]) -> dict:
    return provenance.capture(
        sources,
        seed=config.get("seed"),
        config_path=config.get("__path__"),
        config=workspace.settings(config),
    )


def _events(mark: int) -> tuple[list[str], list[str]]:
    events = log.since(mark)
    return (
        [e.message for e in events if e.level == "WARN"],
        [e.message for e in events if e.level == "ERROR"],
    )


def _mapping_table(mapping: dict[str, str]) -> pd.DataFrame:
    """The name mapping as the table it was read into, for its provenance digest."""
    return pd.DataFrame(list(mapping.items()), columns=["Raritan", "Ghent"])


# ---------------------------------------------------------------------------
# Stage 1: the PTF
# ---------------------------------------------------------------------------
def run_ptf(config: dict[str, Any], config_path: str | Path | None = None) -> Path:
    """Compare what the sources record against what the PTF lists."""
    run_start = datetime.now()
    mark = log.mark()
    workspace.require_paths(config, ("ptf", "phf_ghent", "phf_raritan", "param_mapping", "pvf"))
    paths = config["paths"]
    log.stage("PTF")

    reader = workspace.remote_reader(config)
    src = {
        label: workspace.source(config, label, reader)
        for label in ("ptf", "phf_ghent", "phf_raritan", "param_mapping", "investigations")
    }
    ptf_frame = io.read_ptf(src["ptf"])
    ptf_cols = list(ptf_frame["Parameter"])
    ghent = io.load_phf_ghent(src["phf_ghent"])
    raritan = io.load_phf_raritan(src["phf_raritan"])
    mapping = io.load_site_mapping(src["param_mapping"], exclude=_mapping_exclude(config))
    reachable = src["investigations"].path or src["investigations"].is_remote
    investigations = io.load_investigations(src["investigations"]) if reachable else None
    sources = [
        src["ptf"].record(ptf_frame),
        src["phf_ghent"].record(ghent),
        src["phf_raritan"].record(raritan),
        src["param_mapping"].record(_mapping_table(mapping)),
        src["investigations"].record(investigations),
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

    prov = _provenance(config, sources)
    path = report.write(
        report.build_ptf_payload(
            report.PtfRun(ptf=found, events=log.since(mark), provenance=prov, run_start=run_start)
        ),
        _report_path(config, "ptf"),
    )
    log.success(MODULE, f"Report written to {path}")

    target = manifest_path(paths["pvf"], "ptf")
    warnings, errors = _events(mark)
    manifest.write(
        target,
        manifest.envelope(
            "ptf",
            run_id=package.run_id(run_start),
            started=run_start,
            config_path=str(config.get("__path__") or config_path or ""),
            prov=prov,
            sources=sources,
            written=manifest.outputs(
                target.parent, {"report": path, "new parameters": found.written_to}
            ),
            warnings=warnings,
            errors=errors,
            ptf={"location": src["ptf"].location, "parameters": found.ptf_parameters},
            sources_compared=found.sources_read,
            sources_not_read=found.sources_absent,
            not_in_ptf=found.by_source,
            new_parameters=found.new_parameters,
        ),
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
            ("Manifest", str(target)),
            (
                "New parameters",
                found.written_to or "not written (ptf.write_new_parameters is off)",
            ),
        ],
    )
    log.summary(mark)
    return path


def _mapping_exclude(config: dict[str, Any]) -> list[str]:
    mapping = config.get("mapping")
    if mapping is None:  # configs from before the setting existed
        return ["Non-Conformance Type Calc."]
    return [str(name) for name in mapping.get("exclude") or []]


# ---------------------------------------------------------------------------
# Stage 2: the PVF
# ---------------------------------------------------------------------------
def run_build(
    config: dict[str, Any],
    mode: str = "dev",
    skip_write: bool = False,
    config_path: str | Path | None = None,
) -> Path:
    """Build the PVF from both sites and write its report."""
    run_start = datetime.now()
    mark = log.mark()
    workspace.require_paths(
        config,
        (
            "ptf",
            "phf_ghent",
            "phf_raritan",
            "param_mapping",
            "remfg",
            "lv_coa",
            "raw_materials",
            "pvf",
        ),
    )
    paths = config["paths"]
    params = features.Params(**(config.get("features") or {}))
    log.stage("PVF build")
    log.info(MODULE, f"Mode: {mode.upper()}")

    reader = workspace.remote_reader(config)
    src = {
        label: workspace.source(config, label, reader)
        for label in (
            "ptf",
            "phf_ghent",
            "phf_raritan",
            "param_mapping",
            "remfg",
            "lv_coa",
            "raw_materials",
        )
    }
    site_maps = workspace.site_map_sources(config, reader)

    # ── Load ───────────────────────────────────────────────────────────────
    log.section("01 · Loading")
    ptf_frame = io.read_ptf(src["ptf"])
    ptf_cols = list(ptf_frame["Parameter"])
    ptf_mapping = dict(zip(ptf_frame["Parameter"], ptf_frame["Value Type"]))

    df_ghent = io.load_phf_ghent(src["phf_ghent"])
    raritan_raw = io.load_phf_raritan(src["phf_raritan"])
    mapping = io.load_site_mapping(src["param_mapping"], exclude=_mapping_exclude(config))
    df_raritan = io.apply_column_mapping(raritan_raw, mapping)

    df_remfg = io.load_remanufacturing_supplement(src["remfg"])
    df_raritan = io.attach_supplement(
        df_raritan, df_remfg, "Patient Lot/Batch #", ["Number of Prior Lines of Therapy"]
    )

    lv_coa = config.get("lv_coa") or {}
    df_lv_coa = io.load_lv_coa(
        src["lv_coa"],
        ptf_cols,
        ph_nominal=lv_coa.get("ph_nominal"),
        osmo_nominal=lv_coa.get("osmo_nominal"),
    )
    df_raw = io.load_raw_materials(src["raw_materials"], ptf_cols)

    # Recorded now, as each source was actually read and before cleaning
    # changes the site tables in place.
    sources = [
        src["ptf"].record(ptf_frame),
        src["phf_ghent"].record(df_ghent),
        src["phf_raritan"].record(raritan_raw),
        src["param_mapping"].record(_mapping_table(mapping)),
        src["remfg"].record(df_remfg),
        src["lv_coa"].record(df_lv_coa),
        src["raw_materials"].record(df_raw),
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
        ["PTF", f"{len(ptf_cols):,}", f"{ptf_frame.shape[1]}", "loaded"],
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
    if cleaning_report.censored_flags:
        origins["Cleaning (censoring indicators)"] = list(cleaning_report.censored_flags)

    # ── Enrich: original parameters from other systems ────────────────────
    log.section("03 · Joining sources")
    origins["Databricks (Acridine Orange)"] = enrich.add_viability_parameters(
        phf, ptf_cols, enabled=bool((config.get("sources") or {}).get("databricks"))
    )
    lv_coa_coverage, coa_columns = enrich.merge_lv_coa(phf, df_lv_coa)
    origins["Vector certificate of analysis"] = coa_columns
    origins["Raw materials and consumables"] = enrich.merge_raw_materials(phf, df_raw)
    unmapped_sites, site_columns = enrich.enrich_clinical_sites(phf, site_maps)
    origins["Clinical site mapping"] = site_columns
    enrich.harmonise_countries(phf)
    sources += [_site_map_record(site_map) for site_map in site_maps]

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
    parquet_path = None
    content_digest = None
    if not skip_write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_excel(out_path, index=False)
        log.success(MODULE, f"PVF written to {out_path}")
        # The digest of the table as a task will read it back, which is what a
        # task compares against when it reads the PVF from SharePoint.
        content_digest = provenance.frame_digest(io.read_table(out_path))
        if (config.get("build") or {}).get("write_parquet", True):
            parquet_path = _write_parquet(result, out_path.with_suffix(".parquet"))
        uploaded = _upload(config, mode, out_path)
    elif mode.upper() == "PRD":
        log.info(MODULE, "--report-only: nothing was written and nothing was uploaded")

    prov = _provenance(config, sources)
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
    if parquet_path:
        written.insert(1, ("PVF (Parquet)", str(parquet_path)))
    if skip_write:
        # The manifest describes the PVF on disk, which this run did not write.
        written.append(("Manifest", "not written (--report-only)"))
    else:
        target = manifest_path(out_path, "pvf")
        warnings, errors = _events(mark)
        manifest.write(
            target,
            manifest.envelope(
                "pvf",
                run_id=package.run_id(run_start),
                started=run_start,
                config_path=str(config.get("__path__") or config_path or ""),
                prov=prov,
                sources=sources,
                written=manifest.outputs(
                    target.parent,
                    {"PVF": out_path, "PVF (Parquet)": parquet_path, "report": path},
                ),
                warnings=warnings,
                errors=errors,
                mode=mode.upper(),
                pvf={
                    "path": out_path.name,
                    "sha256": provenance.file_sha256(str(out_path)),
                    "content_sha256": content_digest,
                    "parquet": parquet_path.name if parquet_path else None,
                    "rows": int(result.shape[0]),
                    "columns": int(result.shape[1]),
                    "site_rows": merge_report.site_rows,
                },
                ptf={"location": src["ptf"].location, "parameters": len(ptf_cols)},
                ptf_parameters_not_in_pvf=merge_report.ptf_missing,
                pvf_parameters_not_in_ptf=[
                    str(c) for c in result.columns if c not in set(ptf_cols)
                ],
                duplicate_batches=merge_report.duplicate_batches,
                uploaded_to=uploaded,
            ),
        )
        written.append(("Manifest", str(target)))
    if uploaded:
        written.append(("Uploaded to", uploaded))

    log.names("PTF parameters that are not in the PVF", merge_report.ptf_missing)
    log.facts(f"Written — {result.shape[0]:,} batches × {result.shape[1]:,} columns", written)
    log.summary(mark)
    return path


def _site_map_record(site_map: io.Source) -> provenance.Source:
    """A clinical-site workbook as provenance records it, without reading it twice."""
    if site_map.is_remote:
        return provenance.remote(site_map.label, site_map.location)
    if Path(site_map.path).is_file():
        return provenance.local(site_map.label, site_map.path)
    return provenance.missing(site_map.label, site_map.path, "could not be read")


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path | None:
    """The PVF with its column types intact, for tasks to read quickly and exactly.

    Parquet holds one type per column, so a text column that also holds the odd
    number is written as text throughout — which is what Excel shows anyway.
    """
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            kinds = {type(v) for v in out[column].dropna()}
            if len(kinds) > 1:
                out[column] = out[column].map(lambda v: v if pd.isna(v) else str(v))
    try:
        out.to_parquet(path, index=False)
    except Exception as exc:  # the Excel PVF is the record; the Parquet copy is a convenience
        log.warn(MODULE, f"The Parquet copy of the PVF was not written: {exc}")
        return None
    log.success(MODULE, f"PVF (Parquet) written to {path}")
    return path


def _upload(config: dict[str, Any], mode: str, written: Path) -> str | None:
    """Put the PVF on the share, when a run is explicitly asked to.

    Two switches, both deliberate: the upload has to be enabled in config and the
    run has to be a PRD one. Reading from the share does not imply writing to it,
    and a test or a local run cannot upload by accident. Where it goes is
    ``upload.path``/``upload.drive_id`` in the config (or ``$DATA_LINK``), and that
    is what the manifest records.
    """
    upload_config = config.get("upload") or {}
    if not upload_config.get("enabled"):
        return None
    if mode.upper() != "PRD":
        log.info(MODULE, f"Upload is enabled but this is a {mode.upper()} run — nothing uploaded")
        return None
    target = workspace.upload_target(config)
    try:
        import io_sharepoint
    except ImportError:
        log.error(MODULE, "upload.enabled is set but io_sharepoint is not installed")
        return None

    staged = written
    if written.name != target["file_name"]:
        staged = written.with_name(target["file_name"])
        staged.write_bytes(written.read_bytes())
    io_sharepoint.upload_file_to_sharepoint(
        local_path=staged,
        sharepoint_target_path=target["folder"],
        drive_id=target["drive_id"],
    )
    if staged != written:
        staged.unlink()
    log.success(MODULE, f"PVF uploaded to {target['location']}")
    return target["location"]


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
        "features.csv",
        "decisions.csv",
        "cohort.csv",
        "clusters.csv",
        "recipe.json",
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


def _carry_upstream(
    pkg: package.TaskPackage, spec: TaskSpec, read: provenance.Source, content: str | None
) -> dict:
    """Copy the PTF and build manifests into the run, and check they fit its PVF.

    They are what says how the PVF this task read was made. A PVF manifest that
    describes a different file — by its bytes when the PVF was a local file, by
    its content when it came from SharePoint — belongs to some other build, and
    is said to.
    """
    upstream: dict[str, str] = {}
    pvf_path = Path(spec.pvf)
    if pvf_path.suffix == ".parquet":
        pvf_path = pvf_path.with_suffix(".xlsx")
    for stage in ("ptf", "pvf"):
        source = manifest_path(pvf_path, stage)
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
            recorded = json.loads(text).get("pvf") or {}
            if read.kind == "local file" and Path(spec.pvf).suffix != ".parquet":
                same = recorded.get("sha256") in (None, read.digest)
                mine, theirs = read.digest, recorded.get("sha256")
            else:
                same = recorded.get("content_sha256") in (None, content)
                mine, theirs = content, recorded.get("content_sha256")
            if not same:
                message = (
                    "The PVF manifest describes a different file than the PVF this task read: "
                    "the PVF changed after that build, or the manifest is stale"
                )
                log.warn(MODULE, message, f"manifest {str(theirs)[:12]}, PVF {str(mine)[:12]}")
                pkg.warnings.append(message)
    return upstream


def _task_source(config: dict[str, Any] | None, label: str, path: Path, reader) -> io.Source:
    """The task's PVF or PTF: the workspace's SharePoint copy when it is the workspace's file."""
    if config and reader is not None:
        configured = (config.get("paths") or {}).get(label)
        if configured and Path(configured).resolve() == Path(path).resolve():
            return workspace.source(config, label, reader)
    return io.as_source(path, label)


def run_task(config: dict[str, Any] | None, spec: TaskSpec, report_only: bool = False) -> Path:
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
    fields: dict[str, Any] = {
        "mode": "report only" if report_only else "full package",
        "config_path": spec.source,
        "question": spec.question,
        "purpose": spec.purpose,
        "workspace": spec.workspace,
    }

    try:
        reader = workspace.remote_reader(config) if config else None
        pkg.write_text(
            TASK_LAYOUT["task.yaml"],
            "# The effective settings of this run, defaults included.\n"
            + yaml.safe_dump(spec.as_dict(), sort_keys=False, allow_unicode=True),
        )
        pvf_source = _task_source(config, "pvf", spec.pvf, reader)
        ptf_source = _task_source(config, "ptf", spec.ptf, reader)
        pvf = io.load_pvf(pvf_source)
        ptf_frame = io.read_ptf(ptf_source)
        sources = [pvf_source.record(pvf), ptf_source.record(ptf_frame)]
        prov = provenance.capture(
            sources, seed=spec.seed, config_path=spec.source, config=spec.as_dict()
        )
        fields.update({"prov": prov, "sources": sources})

        final, raw_table, result = dataset.build(pvf, ptf_frame, spec)
        _check(result, final)
        upstream = _carry_upstream(pkg, spec, sources[0], provenance.frame_digest(pvf))

        if not report_only:
            _write_package(pkg, spec, result, final, raw_table)

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
        warnings, errors = _events(mark)
        destination = pkg.complete(
            {
                **fields,
                "errors": errors,
                "warnings": warnings,
                "dataset_role": "not written (report only)" if report_only else result.dataset_role,
                "upstream_manifests": upstream,
                "counts": {
                    "pvf_rows": result.cohort.raw_rows,
                    "cohort_rows": result.cohort.rows,
                    "dataset_rows": result.rows,
                    "dataset_columns": int(final.shape[1]),
                    "predictor_columns": result.feature_columns,
                    "transformed_columns": result.transformed_columns,
                    "candidates": len(result.candidates),
                    "clusters": len(result.clusters.clusters),
                    "decisions": len(result.decisions),
                    "training_rows": result.split.get("training", result.rows),
                    "validation_rows": result.split.get("validation", 0),
                    "folds": result.split.get("folds", 0),
                },
                "split": result.split,
            },
            required=REPORT_ONLY_ARTIFACTS if report_only else REQUIRED_ARTIFACTS,
        )
    except Exception as exc:
        pkg.write_text(TASK_LAYOUT["run.log"], _log_text(log.since(mark)))
        pkg.fail(f"{type(exc).__name__}: {exc}", fields)
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


def _write_package(
    pkg: package.TaskPackage,
    spec: TaskSpec,
    result: dataset.TaskResult,
    final: pd.DataFrame,
    raw_table: pd.DataFrame,
) -> None:
    pkg.write_csv(TASK_LAYOUT["dataset.csv"], final)
    if spec.predictive:
        # dataset.csv is the raw modelling input; the transformed view — fitted
        # on the training rows, applied to the validation rows — goes beside it.
        pkg.write_csv(TASK_LAYOUT["transformed.csv"], result.transformed)
    else:
        # The transformed table is dataset.csv here, so the untransformed
        # predictors go beside it.
        pkg.write_csv(TASK_LAYOUT["raw_features.csv"], raw_table)
    pkg.write_csv(TASK_LAYOUT["columns.csv"], result.columns)
    pkg.write_csv(
        TASK_LAYOUT["features.csv"],
        pd.DataFrame(result.dictionary, columns=["output", "source", "strategy", "detail"]).rename(
            columns={
                "output": "Feature",
                "source": "Parameter",
                "strategy": "Encoding",
                "detail": "Meaning",
            }
        ),
    )
    pkg.write_csv(TASK_LAYOUT["decisions.csv"], [d.row() for d in result.decisions])
    pkg.write_csv(TASK_LAYOUT["cohort.csv"], result.cohort.exclusions)
    pkg.write_csv(TASK_LAYOUT["clusters.csv"], _cluster_rows(result))
    if result.missingness is not None and (
        result.missingness.groups or result.missingness.singletons
    ):
        pkg.write_csv(TASK_LAYOUT["missingness.csv"], missingness.rows(result.missingness))
    pkg.write_json(TASK_LAYOUT["recipe.json"], result.recipe)
    if result.split:
        identifier = spec.roles.id if spec.roles.id in final.columns else ""
        pkg.write_csv(
            TASK_LAYOUT["splits.csv"],
            [
                {
                    "Batch": str(final.at[index, identifier]) if identifier else f"row {index}",
                    "Split": final.at[index, "split"] if "split" in final.columns else "training",
                    "Fold": result.folds.get(index, ""),
                }
                for index in final.index
            ],
        )


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


# ---------------------------------------------------------------------------
# Choosing what to run
# ---------------------------------------------------------------------------
def task_files(config: dict[str, Any] | None, given: list[str] | None) -> list[Path]:
    """The task files a command runs: those given (files or folders), else the workspace's."""
    found: list[Path] = []
    for item in given or []:
        path = Path(item).expanduser()
        if path.is_dir():
            found += sorted(path.glob("*.yaml")) + sorted(path.glob("*.yml"))
        else:
            found.append(path)
    if given:
        return found
    if config:
        folder = (config.get("paths") or {}).get("task_files")
        if folder and Path(folder).is_dir():
            return sorted(Path(folder).glob("*.yaml")) + sorted(Path(folder).glob("*.yml"))
    return []


def load_tasks(args, config: dict[str, Any] | None) -> list[TaskSpec]:
    """The tasks to run: the files given, the workspace's task folder, or the old config block."""
    files = task_files(config, getattr(args, "task", None))
    if files:
        return [taskconfig.load(path, config) for path in files]
    if config and config.get("tasks"):
        return [taskconfig.from_legacy(config, config["__path__"])]
    raise TaskConfigError(
        "No task to run: no --task was given, the workspace's task folder (paths.task_files) "
        "has no YAML in it, and the config has no tasks: block. Create one with "
        "`pvf new-task <name>`"
    )


def load_task(args, config: dict[str, Any] | None = None) -> TaskSpec:
    """The first task to run (kept for callers that run one)."""
    return load_tasks(args, config)[0]


def _config(args, start: str | Path | None = None, required: bool = True) -> dict | None:
    try:
        return workspace.load(workspace.find(getattr(args, "config", None), start=start))
    except ConfigError:
        if required or getattr(args, "config", None):
            raise
        return None


# ---------------------------------------------------------------------------
# The helper commands
# ---------------------------------------------------------------------------
def check(config: dict[str, Any] | None, specs: list[TaskSpec]) -> list[str]:
    """Everything that would stop a run, found without running anything."""
    problems: list[str] = []
    if config:
        try:
            rules = corrections.load((config.get("cleaning") or {}).get("corrections"))
            log.success(MODULE, f"Corrections table: {len(rules)} rules")
        except corrections.CorrectionsError as exc:
            problems.append(str(exc))
        location = str((config.get("sources") or {}).get("location", "local"))
        if location == "local":
            for label in workspace.SHAREPOINT_SOURCES:
                value = (config.get("paths") or {}).get(label)
                for path in value if isinstance(value, list) else [value] if value else []:
                    if label != "pvf" and not Path(path).exists():
                        problems.append(f"paths.{label}: {path} does not exist")
    for spec in specs:
        problems += [f"{spec.name}: {p}" for p in _check_task(spec)]
    return problems


def _check_task(spec: TaskSpec) -> list[str]:
    """A task's columns against the PVF and PTF it would read, where they exist."""
    problems: list[str] = []
    if not Path(spec.pvf).exists():
        return [f"inputs.pvf: {spec.pvf} does not exist yet — run `pvf build` first"]
    columns = set(
        pd.read_parquet(spec.pvf).columns
        if spec.pvf.suffix == ".parquet"
        else pd.read_excel(spec.pvf, nrows=0).columns
    )
    named = {
        "target.column": [spec.target.column],
        "columns.id": [spec.roles.id],
        "columns.group": [spec.roles.group],
        "columns.patient": [spec.roles.patient],
        "columns.include": list(spec.roles.include),
        "cohort.filters": [f.column for f in spec.cohort.filters],
        "cohort.date_column": [spec.cohort.date_column],
        "split.order_column": [spec.split.order_column],
    }
    for where, names in named.items():
        for name in names:
            if name and name not in columns:
                problems.append(f"{where}: '{name}' is not a column of the PVF")
    if spec.cohort.sites and spec.cohort.site_column not in columns:
        problems.append(f"cohort.site_column: '{spec.cohort.site_column}' is not in the PVF")
    if Path(spec.ptf).exists() and spec.predictive and spec.availability.cutoff:
        frame = io.read_table(spec.ptf)
        if io.STAGE_COLUMN not in frame.columns and not spec.availability.reviewed:
            problems.append(
                f"availability.cutoff: the PTF has no '{io.STAGE_COLUMN}' column to check it "
                "against; add it, or set availability.reviewed with an unavailable list"
            )
    return problems


def apply_recipe(run: str | Path, rows: pd.DataFrame, out: Path | None) -> Path:
    """Encode batches with a finished run's recipe and write them out."""
    run = Path(run)
    settings = yaml.safe_load((run / TASK_LAYOUT["task.yaml"]).read_text(encoding="utf-8"))
    identifier = ((settings or {}).get("columns") or {}).get("id") or ""
    encoded = recipe.apply(run, rows)
    if identifier and identifier in rows.columns:
        encoded.insert(0, identifier, rows[identifier])
    out = out or Path.cwd() / f"{run.parent.name}-{run.name}-applied.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(out, index=False)
    log.success(MODULE, f"{len(encoded):,} batches encoded with {run.name}'s recipe → {out}")
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pvf",
        description="CAR-T manufacturing data: the PTF, the PVF, and model-ready tasks from it",
        epilog="Start with `pvf init my_workspace --demo`, then `cd my_workspace && pvf all`.",
    )
    parser.add_argument(
        "--config", help="the workspace config (default: $PVF_CONFIG, or the nearest pvf.yaml)"
    )
    parser.add_argument("--debug", action="store_true", help="show tracebacks for every error")
    commands = parser.add_subparsers(dest="stage", required=True, metavar="command")

    init = commands.add_parser("init", help="create a workspace: config, corrections, folders")
    init.add_argument("folder", help="the workspace folder to create")
    init.add_argument("--demo", action="store_true", help="fill it with made-up data and tasks")
    init.add_argument("--force", action="store_true", help="overwrite an existing config")

    new = commands.add_parser("new-task", help="write a task file, in the workspace or anywhere")
    new.add_argument("name", help="a name (goes into tasks/) or a path to a .yaml file")
    new.add_argument("--target", default="", help="the column the task predicts")
    new.add_argument(
        "--purpose", choices=taskconfig.PURPOSES, default="exploratory", help="what it is for"
    )
    new.add_argument("--question", default="", help="the question, in words")
    new.add_argument("--force", action="store_true", help="overwrite an existing task file")

    checking = commands.add_parser("check", help="validate the config and tasks; run nothing")
    checking.add_argument("--task", action="append", help="a task file or folder (repeatable)")

    commands.add_parser("ptf", help="what the sources record that the PTF does not list")
    build = commands.add_parser("build", help="build the PVF from both sites")
    build.add_argument("--mode", default=None, help="PRD or DEV (default: $MODE, else DEV)")
    build.add_argument(
        "--report-only", action="store_true", help="build the report without writing the PVF"
    )
    tasks = commands.add_parser("tasks", help="run tasks and write a folder for each")
    tasks.add_argument(
        "--task",
        action="append",
        help="a task file or folder (repeatable); default: every task in the workspace",
    )
    tasks.add_argument(
        "--report-only",
        action="store_true",
        help="write only the report and its metadata, into a folder of their own",
    )
    everything = commands.add_parser("all", help="run ptf, build and the tasks, in order")
    everything.add_argument("--mode", default=None, help="PRD or DEV (default: $MODE, else DEV)")
    everything.add_argument("--task", action="append", help="a task file or folder (repeatable)")

    applying = commands.add_parser("apply", help="encode batches with a task run's recipe")
    applying.add_argument("--run", required=True, help="the task run folder")
    applying.add_argument("--pvf", help="the batches to encode (default: the workspace PVF)")
    applying.add_argument("--out", help="the CSV to write (default: in the working directory)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except USER_ERRORS as exc:
        if args.debug:
            raise
        print(f"\npvf {args.stage}: {exc}", file=sys.stderr)
        return 2
    except Exception:
        if args.debug:
            raise
        traceback.print_exc()
        print(f"\npvf {args.stage}: failed unexpectedly (details above)", file=sys.stderr)
        return 1


def _run(args) -> int:
    mode = getattr(args, "mode", None) or os.getenv("MODE", "dev")
    if args.stage == "init":
        path = scaffold.init(args.folder, demo_data=args.demo, force=args.force)
        log.facts(
            "Workspace created",
            [
                ("Config", str(path)),
                (
                    "Next",
                    f"cd {Path(args.folder)} && pvf "
                    + ("all" if args.demo else "check   (after putting the sources in data/raw)"),
                ),
            ],
        )
        return 0

    first_task = (getattr(args, "task", None) or [None])[0]
    if args.stage == "new-task":
        config = _config(args)
        path = scaffold.new_task(
            config, args.name, args.target, args.purpose, args.question, args.force
        )
        log.facts(
            "Task created",
            [
                ("File", str(path)),
                ("Check", f"pvf check --task {path}"),
                ("Run", f"pvf tasks --task {path}"),
            ],
        )
        return 0

    if args.stage == "apply":
        config = _config(args, required=not args.pvf)
        rows = io.load_pvf(args.pvf or config["paths"]["pvf"])
        apply_recipe(args.run, rows, Path(args.out) if args.out else None)
        return 0

    if args.stage in ("check", "tasks"):
        config = _config(args, start=first_task, required=False)
        specs = load_tasks(args, config)
        if args.stage == "check":
            problems = check(config, specs)
            for spec in specs:
                log.success(MODULE, f"Task '{spec.name}' is valid", spec.source)
            if problems:
                log.names("Problems that would stop a run", problems, level="ERROR")
                return 2
            log.success(MODULE, "Everything checked out")
            return 0
        for spec in specs:
            run_task(config, spec, report_only=args.report_only)
        return 0

    config = _config(args, start=first_task)
    if args.stage in ("ptf", "all"):
        run_ptf(config)
    if args.stage in ("build", "all"):
        run_build(config, mode=mode, skip_write=getattr(args, "report_only", False))
    if args.stage == "all":
        for spec in load_tasks(args, config):
            run_task(config, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
