"""The commands, and the one place the stages are run from.

    ml validate [--task ...]       validate the config and tasks; run nothing
    ml check-catalog                      what the sources record that the catalog does not list
    ml build                    the batch table: every batch, every parameter, cleaned and derived
    ml task new <folder>            a task workspace: a folder with an annotated task.yaml
    ml task run [--task ...]       each task (default: the one in this folder), as a new run
    ml task rerun <run folder>       a finished run redone from its own folder, and compared
    ml update [--task ...]         catalog, build and the given tasks, in that order
    ml task encode --run <folder>     a task's fitted recipe, applied to new batches

The stages are in that order because each depends on the one before it. The catalog
is the schema, so it comes first; the build fills it in; a task narrows what the
build produced to one question.

Code, configuration and data are kept apart. There is one catalog and one batch table, in a
data folder of your own; ``config.yaml`` names them (found by ``--config``,
``$ML_CONFIG``, the one a task names, or the nearest ``config.yaml`` above the
working directory). Each task is a workspace folder of its own, holding its ``task.yaml``
and every run of it, inputs included.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import find_dotenv, load_dotenv

from . import (
    catalog,
    clean,
    corrections,
    dataset,
    decorrelate,
    enrich,
    features,
    io,
    manifest,
    merge,
    missingness,
    package,
    provenance,
    recipe,
    report,
    scaffold,
    taskconfig,
)
from . import config as workspace
from .config import ConfigError
from .logger import log
from .taskconfig import TaskConfigError, TaskSpec

# A .env in the working directory (or above it); the .env beside config.yaml is
# loaded with its config, see ml.config.load_env.
load_dotenv(find_dotenv(usecwd=True))

MODULE = "cli"

#: Each artefact's place in a run folder, by its file name.
TASK_LAYOUT = {Path(path).name: path for path in package.ARTEFACTS}

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
    """Read config.yaml, validated, with its paths resolved."""
    return workspace.load(path)


def manifest_path(batch_table: str | Path, stage: str) -> Path:
    """Where a stage's manifest lives: beside the batch table, which is where a task looks."""
    return Path(batch_table).parent / f"{stage}_manifest.json"


def _report_path(config: dict[str, Any], stage: str) -> Path:
    directory = Path(config["paths"].get("reports") or Path(config["paths"]["batch_table"]).parent)
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
# Stage 1: the catalog
# ---------------------------------------------------------------------------
def run_catalog(config: dict[str, Any], config_path: str | Path | None = None) -> Path:
    """Compare what the sources record against what the catalog lists."""
    run_start = datetime.now()
    mark = log.mark()
    workspace.require_paths(
        config, ("catalog", "phf_ghent", "phf_raritan", "param_mapping", "batch_table")
    )
    paths = config["paths"]
    log.stage("Catalog")

    reader = workspace.remote_reader(config)
    src = {
        label: workspace.source(config, label, reader)
        for label in ("catalog", "phf_ghent", "phf_raritan", "param_mapping", "investigations")
    }
    catalog_frame = io.read_catalog(src["catalog"])
    catalog_cols = list(catalog_frame["Parameter"])
    ghent = io.load_phf_ghent(src["phf_ghent"])
    raritan = io.load_phf_raritan(src["phf_raritan"])
    mapping = io.load_site_mapping(src["param_mapping"], exclude=_mapping_exclude(config))
    reachable = src["investigations"].path or src["investigations"].is_remote
    investigations = io.load_investigations(src["investigations"]) if reachable else None
    sources = [
        src["catalog"].record(catalog_frame),
        src["phf_ghent"].record(ghent),
        src["phf_raritan"].record(raritan),
        src["param_mapping"].record(_mapping_table(mapping)),
        src["investigations"].record(investigations),
    ]

    found = catalog.find_new_parameters(
        catalog_cols,
        {
            "Ghent PHF": ghent,
            "Raritan batch data": raritan,
            "Investigations Power Query": investigations,
        },
        raritan_mapping=mapping,
    )
    if (config.get("catalog") or {}).get("write_new_parameters") and found.new_parameters:
        catalog.write_new_parameters(found, paths["new_parameters"])

    prov = _provenance(config, sources)
    path = report.write(
        report.build_catalog_payload(
            report.CatalogRun(
                catalog=found, events=log.since(mark), provenance=prov, run_start=run_start
            )
        ),
        _report_path(config, "catalog"),
    )
    log.success(MODULE, f"Report written to {path}")

    target = manifest_path(paths["batch_table"], "catalog")
    warnings, errors = _events(mark)
    manifest.write(
        target,
        manifest.envelope(
            "catalog",
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
            catalog={"location": src["catalog"].location, "parameters": found.catalog_parameters},
            sources_compared=found.sources_read,
            sources_not_read=found.sources_absent,
            not_in_catalog=found.by_source,
            new_parameters=found.new_parameters,
        ),
    )

    for label, names in found.by_source.items():
        if names:
            log.names(f"{label}: columns the catalog does not list", names)
        else:
            log.names(f"{label}: every column is in the catalog", [], level="SUCCESS")
    for label in found.sources_absent:
        log.names(f"{label}: not read, so not compared", [], level="ERROR")
    log.facts(
        "Written",
        [
            ("Report", str(path)),
            ("Manifest", str(target)),
            (
                "New parameters",
                found.written_to or "not written (catalog.write_new_parameters is off)",
            ),
        ],
    )
    log.summary(mark)
    return path


def _mapping_exclude(config: dict[str, Any]) -> list[str]:
    return [str(name) for name in (config.get("mapping") or {}).get("exclude") or []]


# ---------------------------------------------------------------------------
# Stage 2: the batch table
# ---------------------------------------------------------------------------
def run_build(
    config: dict[str, Any],
    mode: str = "dev",
    config_path: str | Path | None = None,
) -> Path:
    """Build the batch table from both sites and write its report."""
    run_start = datetime.now()
    mark = log.mark()
    workspace.require_paths(
        config,
        (
            "catalog",
            "phf_ghent",
            "phf_raritan",
            "param_mapping",
            "remfg",
            "lv_coa",
            "raw_materials",
            "batch_table",
        ),
    )
    paths = config["paths"]
    params = features.Params(**(config.get("features") or {}))
    log.stage("Batch table build")
    log.info(MODULE, f"Mode: {mode.upper()}")
    _check_catalog_is_current(Path(paths["batch_table"]))

    reader = workspace.remote_reader(config)
    src = {
        label: workspace.source(config, label, reader)
        for label in (
            "catalog",
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
    catalog_frame = io.read_catalog(src["catalog"])
    catalog_cols = list(catalog_frame["Parameter"])
    catalog_mapping = dict(zip(catalog_frame["Parameter"], catalog_frame["Value Type"]))

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
        catalog_cols,
        ph_nominal=lv_coa.get("ph_nominal"),
        osmo_nominal=lv_coa.get("osmo_nominal"),
    )
    df_raw = io.load_raw_materials(src["raw_materials"], catalog_cols)

    # Recorded now, as each source was actually read and before cleaning
    # changes the site tables in place.
    sources = [
        src["catalog"].record(catalog_frame),
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
        ["Catalog", f"{len(catalog_cols):,}", f"{catalog_frame.shape[1]}", "loaded"],
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
    cleaning_report = clean.run_all(
        phf, catalog_cols, catalog_mapping, config.get("cleaning") or {}
    )
    if cleaning_report.censored_flags:
        origins["Cleaning (censoring indicators)"] = list(cleaning_report.censored_flags)

    # ── Enrich: original parameters from other systems ────────────────────
    log.section("03 · Joining sources")
    origins["Databricks (Acridine Orange)"] = enrich.add_viability_parameters(
        phf, catalog_cols, enabled=bool((config.get("sources") or {}).get("databricks"))
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
        site: features.apply(df, catalog_cols, params, site=site) for site, df in phf.items()
    }

    # ── Merge ──────────────────────────────────────────────────────────────
    log.section("05 · Merge and checks")
    result = merge.merge_sites(phf)
    merge_report = merge.run_sanity_checks(result, phf, catalog_cols, catalog_mapping)
    origins["Merge"] = ["Site Merged"]

    # ── Write ──────────────────────────────────────────────────────────────
    log.section("06 · Output")
    out_path = Path(paths["batch_table"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(out_path, index=False)
    log.success(MODULE, f"Batch table written to {out_path}")
    # The digest of the table as a task will read it back, which is what a
    # task compares against when it reads the batch table from SharePoint.
    content_digest = provenance.frame_digest(io.read_table(out_path))
    parquet_path = None
    if (config.get("build") or {}).get("write_parquet", True):
        parquet_path = _write_parquet(result, out_path.with_suffix(".parquet"))
    uploaded = _upload(config, mode, out_path)

    prov = _provenance(config, sources)
    path = report.write(
        report.build_batch_table_payload(
            report.BuildRun(
                result=result,
                phf=phf,
                site_shapes=site_shapes,
                catalog_cols=catalog_cols,
                catalog_mapping=catalog_mapping,
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

    written = [("Batch table", str(out_path)), ("Report", str(path))]
    if parquet_path:
        written.insert(1, ("Batch table (Parquet)", str(parquet_path)))
    target = manifest_path(out_path, "batch_table")
    warnings, errors = _events(mark)
    manifest.write(
        target,
        manifest.envelope(
            "batch_table",
            run_id=package.run_id(run_start),
            started=run_start,
            config_path=str(config.get("__path__") or config_path or ""),
            prov=prov,
            sources=sources,
            written=manifest.outputs(
                target.parent,
                {"Batch table": out_path, "Batch table (Parquet)": parquet_path, "report": path},
            ),
            warnings=warnings,
            errors=errors,
            mode=mode.upper(),
            batch_table={
                "path": out_path.name,
                "sha256": provenance.file_sha256(str(out_path)),
                "content_sha256": content_digest,
                "parquet": parquet_path.name if parquet_path else None,
                "rows": int(result.shape[0]),
                "columns": int(result.shape[1]),
                "site_rows": merge_report.site_rows,
            },
            catalog={"location": src["catalog"].location, "parameters": len(catalog_cols)},
            catalog_parameters_not_in_batch_table=merge_report.catalog_missing,
            batch_table_parameters_not_in_catalog=[
                str(c) for c in result.columns if c not in set(catalog_cols)
            ],
            duplicate_batches=merge_report.duplicate_batches,
            uploaded_to=uploaded,
        ),
    )
    written.append(("Manifest", str(target)))
    if uploaded:
        written.append(("Uploaded to", uploaded))

    log.names("Catalog parameters that are not in the batch table", merge_report.catalog_missing)
    log.facts(f"Written — {result.shape[0]:,} batches × {result.shape[1]:,} columns", written)
    log.summary(mark)
    return path


def _check_catalog_is_current(batch_table: Path) -> None:
    """Warn when ``ml check-catalog`` was not run, or found what the catalog does not list."""
    path = manifest_path(batch_table, "catalog")
    if not path.is_file():
        log.warn(
            MODULE, "`ml check-catalog` has not been run, so the catalog is unchecked", str(path)
        )
        return
    new = manifest.read(path).get("new_parameters") or []
    if new:
        log.warn(
            MODULE,
            f"The last `ml check-catalog` found {len(new)} parameters the sources record and "
            "the catalog "
            "does not list; they are not in this batch table until the catalog lists them",
            ", ".join(map(str, new[:10])),
        )


def _site_map_record(site_map: io.Source) -> provenance.Source:
    """A clinical-site workbook as provenance records it, without reading it twice."""
    if site_map.is_remote:
        return provenance.remote(site_map.label, site_map.location)
    if Path(site_map.path).is_file():
        return provenance.local(site_map.label, site_map.path)
    return provenance.missing(site_map.label, site_map.path, "could not be read")


def _write_parquet(frame: pd.DataFrame, path: Path) -> Path | None:
    """The batch table with its column types intact, for tasks to read quickly and exactly.

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
    except (
        Exception
    ) as exc:  # the Excel batch table is the record; the Parquet copy is a convenience
        log.warn(MODULE, f"The Parquet copy of the batch table was not written: {exc}")
        return None
    log.success(MODULE, f"Batch table (Parquet) written to {path}")
    return path


def _upload(config: dict[str, Any], mode: str, written: Path) -> str | None:
    """Put the batch table on the share, when a run is explicitly asked to.

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
    log.success(MODULE, f"Batch table uploaded to {target['location']}")
    return target["location"]


# ---------------------------------------------------------------------------
# Stage 3: one task
# ---------------------------------------------------------------------------
#: What a finished task folder has to contain before it is called finished.
REQUIRED_ARTIFACTS = tuple(
    path for path in package.ARTEFACTS if not path.endswith(("missingness.csv", "_manifest.json"))
)


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
    pkg: package.TaskPackage,
    spec: TaskSpec,
    sources: list[provenance.Source],
    content: str | None,
) -> dict:
    """Copy the catalog and build manifests into the run; check they fit its inputs.

    They are what says how the batch table this task read was made. A batch table manifest that
    describes a different file — by its bytes when the batch table was a local file, by
    its content when it came from SharePoint — belongs to some other build, and
    is said to. So is a catalog that changed after the build: the batch table is then older
    than the schema the task reads it with.
    """
    read, catalog_read = sources
    upstream: dict[str, str] = {}
    batch_table_path = Path(spec.batch_table)
    if batch_table_path.suffix == ".parquet":
        batch_table_path = batch_table_path.with_suffix(".xlsx")
    for stage in ("catalog", "batch_table"):
        source = manifest_path(batch_table_path, stage)
        target = TASK_LAYOUT[f"{stage}_manifest.json"]
        if not source.exists():
            message = (
                f"No {stage} manifest beside the batch table, so the task folder cannot say "
                "how its input was made — rerun "
                f"`ml {'check-catalog' if stage == 'catalog' else 'build'}`"
            )
            log.warn(MODULE, message, str(source))
            pkg.warnings.append(message)
            upstream[stage] = f"not found at {source}"
            continue
        text = source.read_text(encoding="utf-8")
        pkg.write_text(target, text)
        upstream[stage] = target
        if stage == "batch_table":
            recorded = json.loads(text).get("batch_table") or {}
            if read.kind == "local file" and Path(spec.batch_table).suffix != ".parquet":
                same = recorded.get("sha256") in (None, read.digest)
                mine, theirs = read.digest, recorded.get("sha256")
            else:
                same = recorded.get("content_sha256") in (None, content)
                mine, theirs = content, recorded.get("content_sha256")
            if not same:
                message = (
                    "The batch table manifest describes a different file than the batch table "
                    "this task read: "
                    "the batch table changed after that build, or the manifest is stale"
                )
                log.warn(
                    MODULE, message, f"manifest {str(theirs)[:12]}, batch table {str(mine)[:12]}"
                )
                pkg.warnings.append(message)
            inputs = {i["label"]: i.get("digest") for i in json.loads(text).get("inputs") or []}
            built_with = inputs.get("catalog")
            if built_with and catalog_read.digest and built_with != catalog_read.digest:
                message = "The catalog changed after the batch table was built — rerun `ml build`"
                log.warn(MODULE, message)
                pkg.warnings.append(message)
    return upstream


def _task_source(config: dict[str, Any] | None, label: str, path: Path, reader) -> io.Source:
    """The task's batch table or catalog: the SharePoint copy when it is the configured file."""
    if config and reader is not None:
        configured = (config.get("paths") or {}).get(label)
        if configured and Path(configured).resolve() == Path(path).resolve():
            return workspace.source(config, label, reader)
    return io.as_source(path, label)


def _input_name(source: io.Source, label: str) -> str:
    """Where a run keeps its copy of an input: ``inputs/batch_table.xlsx`` or ``catalog.xlsx``."""
    suffix = ".xlsx" if source.is_remote else Path(source.path).suffix
    return f"inputs/{label}{suffix}"


def _copy_input(pkg: package.TaskPackage, source: io.Source, frame: pd.DataFrame, name: str):
    """The input as this run read it, so the run folder alone can redo the run.

    shortcut: a local workbook is copied as is but read back from its first sheet,
    so a batch table or catalog on another sheet (``sheets:`` in config.yaml) needs that sheet
    moved first; write the frame instead if that ever matters.
    """
    target = pkg.path_for(name)
    if source.is_remote:
        frame.to_excel(target, index=False)
    else:
        shutil.copyfile(source.path, target)


def run_task(config: dict[str, Any] | None, spec: TaskSpec) -> Path:
    """Run one task and leave a complete folder behind, or a failed one.

    The run folder is under the task's workspace, ``runs/<run id>``, and holds
    the batch table and catalog it read with their manifests (``inputs/``) and a ``task.yaml``
    that points at those copies — everything needed to redo the analysis.
    """
    run_start = datetime.now()
    mark = log.mark()
    log.stage(f"Task '{spec.name}'")
    pkg = package.open_package(spec.output_root, spec.name)
    inputs: dict[str, str] = {}
    fields: dict[str, Any] = {
        "config_path": spec.source,
        "question": spec.question,
        "workspace": spec.workspace,
    }

    try:
        reader = workspace.remote_reader(config) if config else None
        batch_table_source = _task_source(config, "batch_table", spec.batch_table, reader)
        catalog_source = _task_source(config, "catalog", spec.catalog, reader)
        inputs = {
            "batch_table": _input_name(batch_table_source, "batch_table"),
            "catalog": _input_name(catalog_source, "catalog"),
        }
        settings = spec.as_dict()
        settings["inputs"] = {label: f"../{name}" for label, name in inputs.items()}
        pkg.write_text(
            TASK_LAYOUT["task.yaml"],
            "# The effective settings of this run, defaults included. The inputs are the\n"
            "# copies this run read, so this file and ../inputs reproduce it.\n"
            + yaml.safe_dump(settings, sort_keys=False, allow_unicode=True),
        )
        batch_table = io.load_batch_table(batch_table_source)
        catalog_frame = io.read_catalog(catalog_source)
        _copy_input(pkg, batch_table_source, batch_table, inputs["batch_table"])
        _copy_input(pkg, catalog_source, catalog_frame, inputs["catalog"])
        sources = [batch_table_source.record(batch_table), catalog_source.record(catalog_frame)]
        prov = provenance.capture(
            sources, seed=spec.seed, config_path=spec.source, config=spec.as_dict()
        )
        fields.update({"prov": prov, "sources": sources})

        final, result = dataset.build(batch_table, catalog_frame, spec)
        _check(result, final)
        upstream = _carry_upstream(pkg, spec, sources, provenance.frame_digest(batch_table))
        _write_package(pkg, result, final)

        payload = report.build_task_payload(
            report.TaskRun(
                dataset=final,
                result=result,
                spec=spec,
                events=log.since(mark),
                provenance=prov,
                output_path=str(pkg.destination / TASK_LAYOUT["dataset.csv"]),
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
                "dataset_role": result.dataset_role,
                "upstream_manifests": upstream,
                "counts": {
                    "batch_table_rows": result.cohort.raw_rows,
                    "cohort_rows": result.cohort.rows,
                    "dataset_rows": result.rows,
                    "dataset_columns": int(final.shape[1]),
                    "predictor_columns": result.feature_columns,
                    "transformed_columns": result.transformed_columns,
                    "candidates": len(result.candidates),
                    "clusters": len(result.clusters.clusters),
                    "decisions": len(result.decisions),
                },
                "input_copies": inputs,
            },
            required=REQUIRED_ARTIFACTS + tuple(inputs.values()),
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
            ("Dataset", str(destination / TASK_LAYOUT["dataset.csv"])),
            ("Manifest", str(destination / package.MANIFEST)),
        ],
    )
    log.summary(mark)
    return destination


def rerun(run: str | Path) -> Path:
    """Redo a finished run from its own folder into the same workspace, and compare.

    The run's ``config/task.yaml`` points at the inputs it copied, so nothing
    outside the run folder is read. Says whether ``dataset.csv`` came out the same.
    """
    run = Path(run).expanduser().resolve()
    settings = run / TASK_LAYOUT["task.yaml"]
    if not settings.is_file():
        raise FileNotFoundError(f"{settings} does not exist — is {run} a run folder?")
    spec = replace(
        taskconfig.load(settings), output_root=run.parent, workspace=str(run.parent.parent)
    )
    again = run_task(None, spec)
    name = TASK_LAYOUT["dataset.csv"]
    if (again / name).read_bytes() == (run / name).read_bytes():
        log.success(MODULE, f"dataset.csv is identical to the one in {run.name}")
    else:
        log.warn(MODULE, f"dataset.csv differs from the one in {run.name}", str(again))
    return again


def _write_package(pkg: package.TaskPackage, result: dataset.TaskResult, final: pd.DataFrame):
    pkg.write_csv(TASK_LAYOUT["dataset.csv"], final)
    pkg.write_csv(TASK_LAYOUT["transformed.csv"], result.transformed)
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
    retained = set(result.recipe.get("output_columns") or [])
    pkg.write_csv(TASK_LAYOUT["clusters.csv"], decorrelate.rows(result.clusters, retained))
    if result.missingness is not None and (
        result.missingness.groups or result.missingness.singletons
    ):
        pkg.write_csv(TASK_LAYOUT["missingness.csv"], missingness.rows(result.missingness))
    pkg.write_json(TASK_LAYOUT["recipe.json"], result.recipe)


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
def task_files(given: list[str] | None) -> list[Path]:
    """The task files given: a task.yaml, a task workspace, or a folder of workspaces."""
    found: list[Path] = []
    for item in given or []:
        path = Path(item).expanduser()
        if path.is_dir() and not (path / taskconfig.TASK_FILE).is_file():
            found += sorted(path.glob(f"*/{taskconfig.TASK_FILE}"))
        else:
            found.append(path)
    return found


def load_tasks(args, config: dict[str, Any] | None) -> list[TaskSpec]:
    """The tasks named by ``--task``."""
    files = task_files(getattr(args, "task", None))
    if not files:
        raise TaskConfigError(
            "No task to run: run inside a task workspace or pass --task <folder>. "
            "Create one with `ml task new <folder>`"
        )
    return [taskconfig.load(path, config) for path in files]


def _config(args, start: str | Path | None = None, required: bool = True) -> dict | None:
    """``--config``, else the config the task names, else ``$ML_CONFIG`` or the nearest one."""
    explicit = getattr(args, "config", None) or taskconfig.config_file(start)
    try:
        return workspace.load(workspace.find(explicit, start=start))
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
                    if label != "batch_table" and not Path(path).exists():
                        problems.append(f"paths.{label}: {path} does not exist")
    for spec in specs:
        problems += [f"{spec.name}: {p}" for p in _check_task(spec)]
    return problems


def _check_task(spec: TaskSpec) -> list[str]:
    """A task's columns against the batch table and catalog it would read, where they exist."""
    problems: list[str] = []
    if not Path(spec.batch_table).exists():
        return [f"inputs.batch_table: {spec.batch_table} does not exist yet — run `ml build` first"]
    columns = set(
        pd.read_parquet(spec.batch_table).columns
        if spec.batch_table.suffix == ".parquet"
        else pd.read_excel(spec.batch_table, nrows=0).columns
    )
    named = {
        "target.column": [spec.target.column],
        "columns.id": [spec.roles.id],
        "columns.group": [spec.roles.group],
        "columns.patient": [spec.roles.patient],
        "columns.include": list(spec.roles.include),
        "cohort.filters": [f.column for f in spec.cohort.filters],
        "cohort.date_column": [spec.cohort.date_column],
    }
    for where, names in named.items():
        for name in names:
            if name and name not in columns:
                problems.append(f"{where}: '{name}' is not a column of the batch table")
    if spec.cohort.sites and spec.cohort.site_column not in columns:
        problems.append(
            f"cohort.site_column: '{spec.cohort.site_column}' is not in the batch table"
        )
    if not Path(spec.catalog).exists():
        return problems
    frame = io.read_table(spec.catalog)
    for key in ("include_categories", "exclude_categories"):
        for column, _ in getattr(spec.roles, key):
            if column not in frame.columns:
                problems.append(f"columns.{key}: '{column}' is not a column of the catalog")
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
        prog="ml",
        description="CAR-T manufacturing data: the catalog, the batch table, and datasets from it",
        epilog="Set the paths in config.yaml, run `ml check-catalog` and `ml build`, then "
        "`ml task new <folder>` for a task.",
    )
    parser.add_argument(
        "--config", help="the config (default: $ML_CONFIG, or the nearest config.yaml)"
    )
    parser.add_argument("--debug", action="store_true", help="show tracebacks for every error")
    commands = parser.add_subparsers(dest="stage", required=True, metavar="command")

    new = commands.add_parser("init", help="create a task workspace with a task.yaml")
    new.add_argument("folder", help="the task workspace folder; its name is the task's name")
    new.add_argument("--target", default="", help="the column the task predicts")
    new.add_argument("--question", default="", help="the question, in words")
    new.add_argument("--force", action="store_true", help="overwrite an existing task file")

    checking = commands.add_parser("check", help="validate the config and tasks; run nothing")
    checking.add_argument("--task", action="append", help="a task workspace (repeatable)")

    commands.add_parser("catalog", help="what the sources record that the catalog does not list")
    build = commands.add_parser("build", help="build the batch table from both sites")
    build.add_argument("--mode", default=None, help="PRD or DEV (default: $MODE, else DEV)")
    tasks = commands.add_parser("tasks", help="run tasks and write a folder for each")
    tasks.add_argument(
        "--task",
        action="append",
        help="a task workspace, its task.yaml, or a folder of workspaces (repeatable); "
        "default: the workspace in this folder",
    )
    redo = commands.add_parser("rerun", help="redo a run from its own folder, and compare")
    redo.add_argument("run", help="the run folder, <task workspace>/runs/<run id>")
    everything = commands.add_parser("all", help="run catalog, build and the tasks, in order")
    everything.add_argument("--mode", default=None, help="PRD or DEV (default: $MODE, else DEV)")
    everything.add_argument("--task", action="append", help="a task workspace (repeatable)")

    applying = commands.add_parser("apply", help="encode batches with a task run's recipe")
    applying.add_argument("--run", required=True, help="the task run folder")
    applying.add_argument(
        "--batch-table", help="the batches to encode (default: the configured batch table)"
    )
    applying.add_argument("--out", help="the CSV to write (default: in the working directory)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except USER_ERRORS as exc:
        if args.debug:
            raise
        print(f"\nml {args.stage}: {exc}", file=sys.stderr)
        return 2
    except Exception:
        if args.debug:
            raise
        traceback.print_exc()
        print(f"\nml {args.stage}: failed unexpectedly (details above)", file=sys.stderr)
        return 1


def _run(args) -> int:
    mode = getattr(args, "mode", None) or os.getenv("MODE", "dev")
    if args.stage in ("check", "tasks") and not args.task and Path(taskconfig.TASK_FILE).is_file():
        args.task = ["."]
    first_task = (getattr(args, "task", None) or [None])[0]
    if args.stage == "init":
        found = _config(args, start=args.folder, required=False)
        path = scaffold.init(
            args.folder,
            args.target,
            args.question,
            args.force,
            config=found["__path__"] if found else None,
        )
        log.facts(
            "Task workspace created",
            [
                ("File", str(path)),
                ("Config", found["__path__"] if found else "none found — set config: in the file"),
                ("Next", f"cd {path.parent} && ml validate && ml task run"),
            ],
        )
        return 0

    if args.stage == "rerun":
        rerun(args.run)
        return 0

    if args.stage == "apply":
        config = _config(args, required=not args.batch_table)
        rows = io.load_batch_table(args.batch_table or config["paths"]["batch_table"])
        apply_recipe(args.run, rows, Path(args.out) if args.out else None)
        return 0

    if args.stage in ("check", "tasks"):
        config = _config(args, start=first_task, required=False)
        specs = load_tasks(args, config) if args.stage == "tasks" or args.task else []
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
            run_task(config, spec)
        return 0

    config = _config(args, start=first_task)
    if args.stage in ("catalog", "all"):
        run_catalog(config)
    if args.stage in ("build", "all"):
        run_build(config, mode=mode)
    if args.stage == "all" and args.task:
        for spec in load_tasks(args, config):
            run_task(config, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
