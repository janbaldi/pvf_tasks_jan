"""The commands, and the one place the stages are run from.

    ml validate [--task ...]          validate the config and tasks; run nothing
    ml check-catalog                  what the sources record that the catalog does not list
    ml check-leaks [folder]           any private vocabulary name in the repository
    ml build                          the batch table: every batch, cleaned and derived
    ml update [--task ...]            check-catalog, build and the given tasks, in that order
    ml task new <folder>              a task workspace: a folder with an annotated task.yaml
    ml task run [--task ...]          each task (default: the one in this folder), as a new run
    ml task rerun <run folder>        a finished run redone from its own folder, and compared
    ml task encode --run <folder>     a run's fitted encoding, applied to new batches

The stages are in that order because each depends on the one before it. The
catalog is the schema, so it comes first; the build fills it in; a task narrows
what the build produced to one question.

Code, configuration and data are kept apart, and the data lives on SharePoint.
Every input is read from the location ``config.yaml`` gives it (found by
``--config``, the one a task names, ``$ML_CONFIG``, or the nearest
``config.yaml`` above the working directory), and every stage output is written
there: ``ml check-catalog`` into the catalog's folder, ``ml build`` into the batch
table's folder. Each command lists what it reads and writes, file by file. Each
task is a workspace folder of its own, holding its ``task.yaml`` and every run of
it, inputs included.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
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
    correlation,
    dataset,
    decorrelate,
    enrich,
    features,
    io,
    manifest,
    merge,
    missingness,
    names,
    package,
    provenance,
    recipe,
    report,
    scaffold,
    screening,
    share,
    shifts,
    taskconfig,
)
from . import config as workspace
from .config import ConfigError
from .logger import log
from .taskconfig import TASK_FILE, TaskConfigError, TaskSpec

# A .env in the working directory (or above it); the .env beside config.yaml is
# loaded with its config, see ml.config.load_env.
load_dotenv(find_dotenv(usecwd=True))

MODULE = "cli"

#: Each artefact's place in a run folder, by its file name.
TASK_LAYOUT = {Path(path).name: path for path in package.ARTEFACTS}

#: Errors that are the user's to fix, shown as a message rather than a traceback.
USER_ERRORS = (
    ConfigError,
    names.VocabularyError,
    share.ShareError,
    TaskConfigError,
    dataset.TaskRefused,
    corrections.CorrectionsError,
    screening.ScreeningError,
    FileNotFoundError,
    FileExistsError,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read config.yaml, validated, with its paths resolved."""
    return workspace.load(path)


#: What each stage writes, by the name it has in its SharePoint folder.
CATALOG_OUTPUTS = {
    "report": "catalog_report.html",
    "manifest": "catalog_manifest.json",
    "new parameters": "new_parameters.csv",
}
BUILD_OUTPUTS = {
    "report": "build_report.html",
    "missingness report": "missingness_report.html",
    "missingness table": screening.EVIDENCE_FILE,
    "correlation report": "correlation_report.html",
    "shifts report": "shifts_report.html",
    "task catalog": "task_catalog.xlsx",
    "manifest": "batch_table_manifest.json",
}


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
    return pd.DataFrame(list(mapping.items()), columns=["Site B", "Site A"])


def _show_reads(sources: list[io.Source], title: str = "Reading from SharePoint") -> None:
    """Every file a stage is about to read: its name, then its drive, folder and sheet."""
    if any(source.is_remote for source in sources):
        share.backend()  # unreachable SharePoint stops the stage before it starts
    rows = []
    for source in sources:
        where = source.location
        if where is None:
            rows.append((source.label, Path(source.path).name, str(Path(source.path).parent)))
            continue
        sheet = f"  ·  sheet {where.sheet}" if where.sheet not in (0, "", None) else ""
        rows.append((source.label, where.name, f"{where.where}{sheet}"))
    log.files(title, rows)


def _show_writes(folder: share.Location, names: dict[str, str]) -> None:
    """Every file a stage is about to write, and the one SharePoint folder they go to."""
    log.files(
        "Writing to SharePoint",
        [("folder", folder.where, "")] + [(what, name, "") for what, name in names.items()],
    )


def _show_written(title: str, uploaded: list[tuple[str, share.Location]]) -> None:
    log.files(title, [(what, where.name, where.where) for what, where in uploaded])


@contextmanager
def _preparing(what: str, path: Path, folder: share.Location) -> Iterator[None]:
    """Say which file is being written and where it will go, then show the work running.

    Every output is written in full to a temporary folder first and uploaded
    once all of them are ready, so a failed run leaves nothing half-written on
    SharePoint. The step line names the file and its SharePoint folder; the
    progress bar runs under it until the file is ready.
    """
    log.step(MODULE, f"Preparing {what}", f"{path.name}  →  {folder.where}")
    yield
    if path.is_file():
        log.success(MODULE, f"Prepared {path.name} ({share.size(path)})")


def _manifest_on_share(folder: share.Location, stage: str) -> dict | None:
    """A stage's manifest, read back from the SharePoint folder it was written to."""
    text = share.read_text(folder.file(f"{stage}_manifest.json"))
    return json.loads(text) if text else None


# ---------------------------------------------------------------------------
# Stage 1: the catalog
# ---------------------------------------------------------------------------
def run_catalog(config: dict[str, Any], config_path: str | Path | None = None) -> share.Location:
    """Compare what the sources record against what the catalog lists.

    Reads the catalog and the sources from SharePoint, and writes the report, the
    manifest and the new-parameters list into the catalog's SharePoint folder.
    Returns where the report went.
    """
    run_start = datetime.now()
    mark = log.mark()
    workspace.require(
        config, ("vocabulary", "catalog", "phf_site_a", "phf_site_b", "param_mapping")
    )
    log.stage("Catalog")
    workspace.load_vocabulary(config)

    labels = ["catalog", "phf_site_a", "phf_site_b", "param_mapping"]
    if workspace.has(config, "investigations"):
        labels.append("investigations")
    src = {label: workspace.source(config, label) for label in labels}
    folder = workspace.catalog_folder(config)
    write_new = bool((config.get("catalog") or {}).get("write_new_parameters"))
    outputs = {
        what: name
        for what, name in CATALOG_OUTPUTS.items()
        if write_new or what != "new parameters"
    }
    _show_reads(list(src.values()))
    _show_writes(folder, outputs)

    log.section("Loading")
    catalog_frame = io.read_catalog(src["catalog"])
    catalog_cols = list(catalog_frame["Parameter"])
    first = io.load_phf(src["phf_site_a"], "a")
    second = io.load_phf(src["phf_site_b"], "b")
    mapping = io.load_site_mapping(src["param_mapping"], exclude=_mapping_exclude(config))
    investigations = (
        io.load_investigations(src["investigations"]) if "investigations" in src else None
    )
    sources = [
        src["catalog"].record(catalog_frame),
        src["phf_site_a"].record(first),
        src["phf_site_b"].record(second),
        src["param_mapping"].record(_mapping_table(mapping)),
        src["investigations"].record(investigations)
        if "investigations" in src
        else provenance.missing(
            "investigations", "sharepoint.investigations is not set", "not configured"
        ),
    ]

    log.section("Comparing")
    second_label = f"{names.site('b')} batch records"
    found = catalog.find_new_parameters(
        catalog_cols,
        {
            f"{names.site('a')} batch records": first,
            second_label: second,
            "Investigations Power Query": investigations,
        },
        site_mapping=mapping,
        mapped=(second_label,),
    )

    log.section("Writing")
    with tempfile.TemporaryDirectory(prefix="ml-catalog-") as staging:
        staged = Path(staging)
        files: dict[str, Path] = {}
        if write_new:
            target = staged / CATALOG_OUTPUTS["new parameters"]
            with _preparing("the new-parameters list", target, folder):
                files["new parameters"] = catalog.write_new_parameters(found, target)
            found.written_to = str(folder.file(CATALOG_OUTPUTS["new parameters"]))
        prov = _provenance(config, sources)
        with _preparing("the catalog report", staged / CATALOG_OUTPUTS["report"], folder):
            files["report"] = report.write(
                report.build_catalog_payload(
                    report.CatalogRun(
                        catalog=found, events=log.since(mark), provenance=prov, run_start=run_start
                    )
                ),
                staged / CATALOG_OUTPUTS["report"],
            )
        warnings, errors = _events(mark)
        log.step(
            MODULE, "Preparing the manifest", f"{CATALOG_OUTPUTS['manifest']}  →  {folder.where}"
        )
        files["manifest"] = manifest.write(
            staged / CATALOG_OUTPUTS["manifest"],
            manifest.envelope(
                "catalog",
                run_id=package.run_id(run_start),
                started=run_start,
                config_path=str(config.get("__path__") or config_path or ""),
                prov=prov,
                sources=sources,
                written=manifest.outputs(
                    staged, {what: path for what, path in files.items() if what != "manifest"}
                ),
                warnings=warnings,
                errors=errors,
                written_to=str(folder),
                catalog={"location": src["catalog"].where, "parameters": found.catalog_parameters},
                sources_compared=found.sources_read,
                sources_not_read=found.sources_absent,
                not_in_catalog=found.by_source,
                new_parameters=found.new_parameters,
            ),
        )
        uploaded = share.upload(files, folder)

    for label, columns in found.by_source.items():
        if columns:
            log.names(f"{label}: columns the catalog does not list", columns)
        else:
            log.names(f"{label}: every column is in the catalog", [], level="SUCCESS")
    for label in found.sources_absent:
        log.names(f"{label}: not read, so not compared", [], level="ERROR")
    _show_written("Written to SharePoint", uploaded)
    if not write_new:
        log.info(MODULE, "New parameters not written (catalog.write_new_parameters is off)")
    log.summary(mark)
    return dict(uploaded)["report"]


def _mapping_exclude(config: dict[str, Any]) -> list[str]:
    """The mapping rows known to be wrong: ``mapping.exclude``, as vocabulary keys."""
    return [names.col(str(key)) for key in (config.get("mapping") or {}).get("exclude") or []]


# ---------------------------------------------------------------------------
# Stage 2: the batch table
# ---------------------------------------------------------------------------
BUILD_INPUTS = (
    "catalog",
    "phf_site_a",
    "phf_site_b",
    "param_mapping",
    "remfg",
    "lv_coa",
    "raw_materials",
)


#: The vocabulary key of the date that puts the batches in order, unless build.date_column says.
DATE_COLUMN = "incubator_in_d0"


def run_build(config: dict[str, Any], config_path: str | Path | None = None) -> share.Location:
    """Build the batch table from both sites, and write it with its report.

    Reads every source from SharePoint, and writes the batch table, its Parquet
    copy, the report and the manifest into the batch table's SharePoint folder.
    Returns where the report went.
    """
    run_start = datetime.now()
    mark = log.mark()
    workspace.require(config, ("vocabulary", *BUILD_INPUTS, "batch_table"))
    params = features.Params(**(config.get("features") or {}))
    log.stage("Batch table build")
    workspace.load_vocabulary(config)
    rules = workspace.load_corrections(config)

    src = {label: workspace.source(config, label) for label in BUILD_INPUTS}
    site_maps = workspace.site_map_sources(config)
    target_table = workspace.location(config, "batch_table")
    folder = target_table.folder
    write_parquet = (config.get("build") or {}).get("write_parquet", True)
    date_column = (config.get("build") or {}).get("date_column", DATE_COLUMN)
    parquet_name = f"{Path(target_table.name).stem}.parquet"
    outputs = {"batch table": target_table.name}
    if write_parquet:
        outputs["batch table (Parquet)"] = parquet_name
    outputs.update(BUILD_OUTPUTS)
    _show_reads([*src.values(), *site_maps])
    _show_writes(folder, outputs)
    _check_catalog_is_current(config)

    # ── Load ───────────────────────────────────────────────────────────────
    log.section("01 · Loading")
    catalog_frame = io.read_catalog(src["catalog"])
    catalog_cols = list(catalog_frame["Parameter"])
    catalog_mapping = dict(zip(catalog_frame["Parameter"], catalog_frame["Value Type"]))

    df_first = io.load_phf(src["phf_site_a"], "a")
    second_raw = io.load_phf(src["phf_site_b"], "b")
    mapping = io.load_site_mapping(src["param_mapping"], exclude=_mapping_exclude(config))
    df_second = io.apply_column_mapping(second_raw, mapping)

    df_remfg = io.load_remanufacturing_supplement(src["remfg"])
    df_second = io.attach_supplement(
        df_second, df_remfg, "Patient Lot/Batch #", [names.col("prior_lines")]
    )

    lv_coa = config.get("lv_coa") or {}
    df_lv_coa = io.load_lv_coa(
        src["lv_coa"],
        catalog_cols,
        ph_nominal=lv_coa.get("ph_nominal"),
        osmo_nominal=lv_coa.get("osmo_nominal"),
        censored=(config.get("cleaning") or {}).get("censored", "limit"),
    )
    df_raw = io.load_raw_materials(src["raw_materials"], catalog_cols)

    # Recorded now, as each source was actually read and before cleaning
    # changes the site tables in place.
    sources = [
        src["catalog"].record(catalog_frame),
        src["phf_site_a"].record(df_first),
        src["phf_site_b"].record(second_raw),
        src["param_mapping"].record(_mapping_table(mapping)),
        src["remfg"].record(df_remfg),
        src["lv_coa"].record(df_lv_coa),
        src["raw_materials"].record(df_raw),
    ]

    site_a, site_b = names.sites()
    phf = {site_a: df_first, site_b: df_second}
    site_shapes = {site: df.shape for site, df in phf.items()}

    # Where every column came from, collected as the columns arrive.
    first_cols, second_cols = set(df_first.columns), set(df_second.columns)
    origins: dict[str, list[str]] = {
        "Both sites' batch records": sorted(first_cols & second_cols),
        f"{site_a} batch records only": sorted(first_cols - second_cols),
        f"{site_b} batch records only": sorted(second_cols - first_cols),
        "Re-manufacturing supplement": [names.col("prior_lines")],
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
        phf,
        catalog_cols,
        catalog_mapping,
        config.get("cleaning") or {},
        rules=rules,
        corrections_source=str(
            workspace.location(config, "vocabulary").with_sheet(names.CORRECTIONS_SHEET)
        ),
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
    known = features.availability(catalog_frame, params)
    feature_reports = {
        site: features.apply(df, catalog_cols, params, site=site, known=known)
        for site, df in phf.items()
    }

    # ── Merge ──────────────────────────────────────────────────────────────
    log.section("05 · Merge and checks")
    result = merge.merge_sites(phf)
    merge_report = merge.run_sanity_checks(result, phf, catalog_cols, catalog_mapping)
    origins["Merge"] = ["Site Merged"]
    # Per site: the sites record differently, and pooled they read as each other's gaps.
    missing_by_site, cohorts = {}, {}
    for site in names.sites():
        batches, cohort = _missingness_cohort(result[result["Site Merged"] == site], site, config)
        analysed, counted = _analysed_batches(batches, site, config)
        cohorts[site] = batches, analysed
        missing_by_site[site] = missingness.for_build(
            batches,
            catalog_mapping,
            id_column="Patient Lot/Batch #",
            order_column=names.col(date_column),
            analysed=analysed,
        )
        missing_by_site[site][1].cohort = "; ".join(filter(None, (cohort, counted)))
    standing = _mark_screened(missing_by_site, config)
    # Over the same batches, without what the register already leaves out.
    correlated_by_site, shifted_by_site = {}, {}
    for site, (batches, analysed) in cohorts.items():
        screened = missing_by_site[site][1].screened
        rows, analysis = correlation.for_build(
            batches,
            catalog_mapping,
            id_column="Patient Lot/Batch #",
            order_column=names.col(date_column),
            analysed=analysed,
            screened_out={p: d["Reason"] for p, d in screened.items()},
        )
        analysis.cohort = missing_by_site[site][1].cohort
        analysis.screened = {p: d for p, d in screened.items() if d["Reason"] == correlation.REASON}
        correlated_by_site[site] = rows, analysis
        rows, found = shifts.for_build(
            batches,
            id_column="Patient Lot/Batch #",
            order_column=names.col(date_column),
            analysed=analysed,
            screened_out={p: d["Reason"] for p, d in screened.items()},
        )
        found.cohort = analysis.cohort
        found.screened = {p: d for p, d in screened.items() if d["Reason"] == shifts.REASON}
        shifted_by_site[site] = rows, found

    # ── Write ──────────────────────────────────────────────────────────────
    log.section("06 · Output")
    with tempfile.TemporaryDirectory(prefix="ml-build-") as staging:
        staged = Path(staging)
        files: dict[str, Path] = {"batch table": staged / target_table.name}
        with _preparing(
            f"the batch table ({result.shape[0]:,} × {result.shape[1]:,}, Excel)",
            files["batch table"],
            folder,
        ):
            result.to_excel(files["batch table"], index=False)
        # The digest of the table as a task will read it back from SharePoint,
        # which is what a task compares against.
        log.step(
            MODULE,
            "Reading the batch table back for its content digest",
            f"{target_table.name}, as a task will read it",
        )
        content_digest = provenance.frame_digest(io.read_table(files["batch table"]))
        if write_parquet:
            with _preparing("the batch table (Parquet)", staged / parquet_name, folder):
                parquet = _write_parquet(result, staged / parquet_name)
            if parquet:
                files["batch table (Parquet)"] = parquet

        prov = _provenance(config, sources)
        build_run = report.BuildRun(
            result=result,
            phf=phf,
            site_shapes=site_shapes,
            catalog_cols=catalog_cols,
            catalog_mapping=catalog_mapping,
            cleaning=cleaning_report,
            features=feature_reports,
            merge=merge_report,
            missingness=missing_by_site,
            correlation=correlated_by_site,
            shifts=shifted_by_site,
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
        with _preparing("the build report", staged / BUILD_OUTPUTS["report"], folder):
            files["report"] = report.write(
                report.build_batch_table_payload(build_run), staged / BUILD_OUTPUTS["report"]
            )
        evidence = staged / BUILD_OUTPUTS["missingness table"]
        with _preparing("the missingness table", evidence, folder):
            _write_evidence(
                evidence,
                missing_by_site,
                {
                    (screening.CORRELATION_GROUP, screening.SUGGESTED): correlated_by_site,
                    (screening.SHIFT_GROUP, screening.SHIFT_SUGGESTED): shifted_by_site,
                },
                content_digest,
                run_start,
            )
            files["missingness table"] = evidence
        missing_report = staged / BUILD_OUTPUTS["missingness report"]
        with _preparing("the missingness report", missing_report, folder):
            files["missingness report"] = report.write(
                report.build_missingness_payload(build_run), missing_report
            )
        catalog_for_tasks = staged / BUILD_OUTPUTS["task catalog"]
        with _preparing("the task catalog", catalog_for_tasks, folder):
            # shortcut: the register as this build read it; a later `ml screening add`
            # shows in it only after the next build.
            screening.task_catalog(catalog_frame, result.columns, standing, names.sites()).to_excel(
                catalog_for_tasks, index=False
            )
            files["task catalog"] = catalog_for_tasks
        shifts_report = staged / BUILD_OUTPUTS["shifts report"]
        with _preparing("the shifts report", shifts_report, folder):
            files["shifts report"] = report.write(
                report.build_shifts_payload(build_run), shifts_report
            )
        correlation_report = staged / BUILD_OUTPUTS["correlation report"]
        with _preparing("the correlation report", correlation_report, folder):
            files["correlation report"] = report.write(
                report.build_correlation_payload(build_run), correlation_report
            )
        warnings, errors = _events(mark)
        log.step(
            MODULE, "Preparing the manifest", f"{BUILD_OUTPUTS['manifest']}  →  {folder.where}"
        )
        files["manifest"] = manifest.write(
            staged / BUILD_OUTPUTS["manifest"],
            manifest.envelope(
                "batch_table",
                run_id=package.run_id(run_start),
                started=run_start,
                config_path=str(config.get("__path__") or config_path or ""),
                prov=prov,
                sources=sources,
                written=manifest.outputs(
                    staged, {what: path for what, path in files.items() if what != "manifest"}
                ),
                warnings=warnings,
                errors=errors,
                written_to=str(folder),
                batch_table={
                    "path": target_table.name,
                    "location": str(target_table),
                    "sha256": provenance.file_sha256(str(files["batch table"])),
                    "content_sha256": content_digest,
                    "parquet": parquet_name if "batch table (Parquet)" in files else None,
                    "rows": int(result.shape[0]),
                    "columns": int(result.shape[1]),
                    "site_rows": merge_report.site_rows,
                },
                catalog={"location": src["catalog"].where, "parameters": len(catalog_cols)},
                catalog_parameters_not_in_batch_table=merge_report.catalog_missing,
                batch_table_parameters_not_in_catalog=[
                    str(c) for c in result.columns if c not in set(catalog_cols)
                ],
                duplicate_batches=merge_report.duplicate_batches,
            ),
        )
        uploaded = share.upload(files, folder)

    log.names("Catalog parameters that are not in the batch table", merge_report.catalog_missing)
    _show_written(
        f"Written to SharePoint — {result.shape[0]:,} batches × {result.shape[1]:,} columns",
        uploaded,
    )
    log.summary(mark)
    return dict(uploaded)["report"]


def _register(config: dict[str, Any]) -> screening.Register | None:
    """The screening register config.yaml points at, or None when it points at none."""
    if not workspace.has(config, "screening"):
        return None
    return screening.read(workspace.location(config, "screening"), names.sites())


def _mark_screened(missing_by_site: dict[str, tuple], config: dict[str, Any]) -> dict:
    """Add each parameter's standing screening decision to the missingness rows; the state."""
    try:
        register = _register(config)
    except Exception as exc:  # the report is still worth having without the column
        log.warn(MODULE, f"The screening register could not be read: {exc}")
        return {}
    state = screening.current(register.decisions, names.sites()) if register else {}
    for site, (rows, analysis) in missing_by_site.items():
        for row in rows:
            decision = state.get((row["Parameter"], site))
            row["Screening"] = screening.note(decision)
            if decision is not None and decision.action == screening.EXCLUDE:
                analysis.screened[row["Parameter"]] = {
                    "Decision": decision.id,
                    "Date": decision.date,
                    "Reason": decision.reason,
                    "Rationale": decision.rationale,
                    "Review by": decision.review_by,
                    "Represented by": decision.represented_by,
                }
    return state


def _write_evidence(
    path: Path,
    missing_by_site: dict[str, tuple],
    grouped_by_site: dict[tuple[str, str], dict[str, tuple]],
    content_digest: str,
    run_start: datetime,
) -> Path:
    """Each parameter's missingness, and its correlation and shift group with their suggested
    representative: what `ml screening add` cites. ``grouped_by_site`` maps the two evidence
    columns to the analysis per site that fills them."""
    cited = {}
    for site, (rows, _) in missing_by_site.items():
        cited[site] = [dict(row) for row in rows]
        for (group_column, suggested_column), results in grouped_by_site.items():
            group_of = {m: g for g in results[site][1].groups for m in g.members}
            for row in cited[site]:
                group = group_of.get(row["Parameter"])
                row[group_column] = group.name if group else ""
                row[suggested_column] = group.representative if group else ""
    with pd.ExcelWriter(path) as workbook:
        screening.evidence_frame(cited).to_excel(
            workbook, sheet_name=screening.EVIDENCE_SHEET, index=False
        )
        pd.DataFrame(
            [
                {"Item": "Batch table content digest", "Value": content_digest},
                {"Item": "Built", "Value": run_start.strftime("%Y-%m-%d %H:%M")},
                *(
                    {"Item": f"Batches described at {site}", "Value": analysis.cohort or "all"}
                    for site, (_, analysis) in missing_by_site.items()
                ),
            ]
        ).to_excel(workbook, sheet_name=screening.ABOUT_SHEET, index=False)
    return path


def run_screening_add(
    config: dict[str, Any], proposal: str | Path, author: str, new: bool = False
) -> share.Location:
    """Check a proposal against the latest build, and append it to the register."""
    mark = log.mark()
    log.stage("Screening: adding decisions")
    workspace.require(config, ("vocabulary", "batch_table", "screening"))
    workspace.load_vocabulary(config)
    sites = names.sites()
    where = workspace.location(config, "screening")
    try:
        register = screening.read(where, sites)
    except screening.ScreeningError:
        raise
    except Exception as exc:
        if not new:
            raise FileNotFoundError(
                f"The screening register could not be read from {where} ({exc}). "
                "To start a new one, pass --new"
            ) from exc
        log.info(MODULE, "Starting a new screening register", str(where))
        register = screening.Register([], str(where))
    evidence, digest = _latest_evidence(config)
    added = screening.add(
        register.decisions, io.read_table(proposal), evidence, digest, author, sites
    )
    with tempfile.TemporaryDirectory(prefix="ml-screening-") as staging:
        path = Path(staging) / where.name
        screening.write(register.decisions + added, path)
        share.upload({"screening register": path}, where.folder)
    screening.log_added(added)
    log.facts(
        "Screening register updated",
        [
            ("Register", str(where)),
            ("State", screening.status_line(register.decisions + added, sites)),
            ("Next", "every task run from now on leaves these parameters out"),
        ],
    )
    log.summary(mark)
    return where


def run_screening_status(config: dict[str, Any]) -> list[dict]:
    """What the register excludes today, and which decisions the latest build questions."""
    log.stage("Screening: status")
    workspace.require(config, ("vocabulary", "batch_table", "screening"))
    workspace.load_vocabulary(config)
    sites = names.sites()
    register = screening.read(workspace.location(config, "screening"), sites)
    standing = screening.summary(register.decisions, sites)
    log.facts(
        "Screening register",
        [("Where", register.where), ("State", screening.status_line(register.decisions, sites))],
    )
    log.names(
        "Excluded today",
        level="INFO",
        names=[
            f"{r.Parameter} @ {r.Site} — {r.Decision}, {r.Reason}" for r in standing.itertuples()
        ],
    )
    evidence, _ = _latest_evidence(config)
    findings = screening.review(register.decisions, evidence, sites)
    log.names(
        "Decisions to look at again",
        [
            f"{f['Decision']} {f['Parameter']} @ {f['Site']}: {f['Why look again']}"
            for f in findings
        ],
        level="WARN",
    )
    if not findings:
        log.success(MODULE, "The latest build gives no reason to revisit a decision")
    return findings


def _latest_evidence(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """The latest build's missingness table, and the digest of the batch table it describes."""
    where = workspace.batch_table_folder(config).file(screening.EVIDENCE_FILE)
    try:
        evidence = share.read_excel(where.with_sheet(screening.EVIDENCE_SHEET))
        about = share.read_excel(where.with_sheet(screening.ABOUT_SHEET))
    except Exception as exc:
        raise FileNotFoundError(
            f"The latest build's {screening.EVIDENCE_FILE} could not be read from {where} "
            f"({exc}) — run `ml build` first"
        ) from exc
    items = dict(zip(about["Item"].astype(str), about["Value"].astype(str)))
    return evidence, items.get("Batch table content digest", "")


def _check_catalog_is_current(config: dict[str, Any]) -> None:
    """Warn when ``ml check-catalog`` was not run, or found what the catalog does not list."""
    folder = workspace.catalog_folder(config)
    record = _manifest_on_share(folder, "catalog")
    if record is None:
        log.warn(
            MODULE,
            "No catalog manifest could be read from SharePoint, so the catalog is unchecked "
            "— run `ml check-catalog`",
            str(folder.file(CATALOG_OUTPUTS["manifest"])),
        )
        return
    new = record.get("new_parameters") or []
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
    return provenance.remote(site_map.label, site_map.where)


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
    return path


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
            f"dataset_columns.csv has {len(result.columns)} rows for {final.shape[1]} "
            "dataset columns"
        )
    exported = {row["Column"] for row in result.columns}
    if exported != set(final.columns):
        problems.append("dataset_columns.csv and the dataset do not name the same columns")
    if result.cohort.funnel:
        left = result.cohort.funnel[-1]["Batches left"]
        if left != result.rows:
            problems.append(
                f"the cohort funnel ends at {left} batches but the dataset has {result.rows}"
            )
    if problems:
        raise ValueError("The task package is inconsistent: " + "; ".join(problems))


def _upstream_manifests(config: dict[str, Any] | None, spec: TaskSpec) -> dict[str, tuple]:
    """The catalog and build manifests that say how this task's inputs were made.

    Read from the SharePoint folders ``ml check-catalog`` and ``ml build`` wrote
    them to; a rerun reads the copies its original run kept beside its inputs.
    Each is ``(text or None, where it was looked for)``.
    """
    found: dict[str, tuple] = {}
    for stage, label in (("catalog", "catalog"), ("batch_table", "batch_table")):
        copy = spec.batch_table if label == "batch_table" else spec.catalog
        if copy is not None:
            where = Path(copy).parent / f"{stage}_manifest.json"
            found[stage] = (where.read_text(encoding="utf-8") if where.is_file() else None, where)
        else:
            where = workspace.location(config, label).file(f"{stage}_manifest.json")
            found[stage] = (share.read_text(where), where)
    return found


def _carry_upstream(
    pkg: package.TaskPackage,
    upstream_manifests: dict[str, tuple],
    sources: list[provenance.Source],
    content: str | None,
) -> dict:
    """Copy the catalog and build manifests into the run; check they fit its inputs.

    They are what says how the batch table this task read was made. A batch table
    manifest that describes a different table — by its content as it was read —
    belongs to some other build, and is said to. So is a catalog that changed
    after the build: the batch table is then older than the schema the task reads
    it with.
    """
    _, catalog_read = sources
    upstream: dict[str, str] = {}
    for stage in ("catalog", "batch_table"):
        text, where = upstream_manifests[stage]
        target = TASK_LAYOUT[f"{stage}_manifest.json"]
        if text is None:
            message = (
                f"No {stage} manifest could be read, so the task folder cannot say "
                "how its input was made — rerun "
                f"`ml {'check-catalog' if stage == 'catalog' else 'build'}`"
            )
            log.warn(MODULE, message, str(where))
            pkg.warnings.append(message)
            upstream[stage] = f"not found at {where}"
            continue
        pkg.write_text(target, text)
        upstream[stage] = target
        if stage == "batch_table":
            record = json.loads(text)
            theirs = (record.get("batch_table") or {}).get("content_sha256")
            if theirs not in (None, content):
                message = (
                    "The batch table manifest describes a different table than the batch table "
                    "this task read: "
                    "the batch table changed after that build, or the manifest is stale"
                )
                log.warn(
                    MODULE, message, f"manifest {str(theirs)[:12]}, batch table {content[:12]}"
                )
                pkg.warnings.append(message)
            inputs = {i["label"]: i.get("digest") for i in record.get("inputs") or []}
            built_with = inputs.get("catalog")
            if built_with and catalog_read.digest and built_with != catalog_read.digest:
                message = "The catalog changed after the batch table was built — rerun `ml build`"
                log.warn(MODULE, message)
                pkg.warnings.append(message)
    return upstream


#: The run's copy of the vocabulary it ran with, which a rerun reads.
VOCABULARY_COPY = f"inputs/{workspace.RUN_VOCABULARY}"
#: Where a run keeps the screening register it applied.
SCREENING_COPY = "inputs/screening.xlsx"


def _check_listed_screening(spec: TaskSpec, screened: dict) -> None:
    """Warn about task-file exclusions copied from the register that it no longer makes."""
    for column, reason in spec.predictors.exclude:
        if reason.startswith("screened out by ") and column not in screened:
            log.warn(
                MODULE,
                f"task.yaml still excludes '{column}' ({reason}), but the screening register "
                "no longer does at this task's sites — delete the line to use it again",
            )


def _screened_for_new_task(config: dict[str, Any] | None) -> tuple[list[tuple[str, str]], str]:
    """What the screening register excludes today, at any site, for a new task file."""
    if not config or not workspace.has(config, "screening"):
        return [], ""
    try:
        workspace.load_vocabulary(config)
        register = _register(config)
    except Exception as exc:  # the task is still worth making; its runs apply the register
        log.warn(MODULE, f"The screening register could not be read for the task file: {exc}")
        return [], ""
    excluded = screening.excluded(register.decisions, names.sites())
    last = register.decisions[-1].id if register.decisions else ""
    return [(p, f"screened out by {d.label}") for p, d in sorted(excluded.items())], last


def _task_register(config: dict[str, Any] | None, spec: TaskSpec) -> screening.Register | None:
    """The register a task applies: its run's own copy for a rerun, else SharePoint's."""
    if spec.batch_table is not None:  # a rerun reads nothing but its own folder
        if spec.screening is None:
            return None
        return screening.read_copy(spec.screening, names.sites())
    register = _register(config) if config else None
    if register is None:
        log.info(MODULE, "No screening register is configured; no parameter is screened out")
    else:
        log.info(
            MODULE,
            "Applying the screening register",
            screening.status_line(register.decisions, names.sites()),
        )
    return register


def _input_name(source: io.Source, label: str) -> str:
    """Where a run keeps its copy of an input: ``inputs/batch_table.xlsx`` or ``catalog.xlsx``."""
    return f"inputs/{label}.xlsx"


def _copy_input(pkg: package.TaskPackage, source: io.Source, frame: pd.DataFrame, name: str):
    """The input as this run read it, so the run folder alone can redo the run.

    A SharePoint read is written out as the table that came back; a rerun's own
    copy is copied as is.
    """
    target = pkg.path_for(name)
    if source.is_remote or Path(source.path).suffix.lower() != ".xlsx":
        frame.to_excel(target, index=False)
    else:
        shutil.copyfile(source.path, target)


def run_task(config: dict[str, Any] | None, spec: TaskSpec) -> Path:
    """Run one task and leave a complete folder behind, or a failed one.

    The batch table and catalog are read from SharePoint (a rerun reads the
    copies its original run kept). The run folder is under the task's workspace,
    ``runs/<run id>``, and holds the batch table and catalog it read with their
    manifests (``inputs/``) and a ``task.yaml`` that points at those copies —
    everything needed to redo the analysis.
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
        vocabulary = workspace.task_vocabulary(config, spec.batch_table)
        batch_table_source = workspace.table_source(config, "batch_table", spec.batch_table)
        catalog_source = workspace.table_source(config, "catalog", spec.catalog)
        _show_reads(
            [batch_table_source, catalog_source],
            "Reading from SharePoint" if spec.batch_table is None else "Reading the run's copies",
        )
        log.files("Writing the run folder", [("folder", str(pkg.destination), "")])
        inputs = {
            "batch_table": _input_name(batch_table_source, "batch_table"),
            "catalog": _input_name(catalog_source, "catalog"),
        }
        batch_table = io.load_batch_table(batch_table_source)
        catalog_frame = io.read_catalog(catalog_source)
        _copy_input(pkg, batch_table_source, batch_table, inputs["batch_table"])
        _copy_input(pkg, catalog_source, catalog_frame, inputs["catalog"])
        names.to_frame(vocabulary).to_excel(pkg.path_for(VOCABULARY_COPY), index=False)
        sources = [batch_table_source.record(batch_table), catalog_source.record(catalog_frame)]
        register = _task_register(config, spec)
        if register is not None:
            screening.write(register.decisions, pkg.path_for(SCREENING_COPY))
            inputs["screening"] = SCREENING_COPY
            fields["screening"] = {
                "register": register.where,
                "digest": screening.digest(register.decisions),
                "last_decision": register.decisions[-1].id if register.decisions else "",
                "intact": register.intact,
            }
        prov = provenance.capture(
            sources, seed=spec.seed, config_path=spec.source, config=spec.as_dict()
        )
        fields.update({"prov": prov, "sources": sources})

        settings = spec.as_dict()
        settings["inputs"] = {label: f"../{name}" for label, name in inputs.items()}
        pkg.write_text(
            TASK_LAYOUT["task.yaml"],
            "# The effective settings of this run, defaults included. The inputs are the\n"
            "# copies this run read, so this file and ../inputs reproduce it.\n"
            + yaml.safe_dump(settings, sort_keys=False, allow_unicode=True),
        )
        screened = (
            screening.excluded(register.decisions, names.sites(), spec.cohort.sites)
            if register is not None
            else {}
        )
        _check_listed_screening(spec, screened)
        final, result = dataset.build(
            batch_table,
            catalog_frame,
            spec,
            screened={p: d.label for p, d in screened.items()},
        )
        _check(result, final)
        upstream = _carry_upstream(
            pkg,
            _upstream_manifests(config, spec),
            sources,
            provenance.frame_digest(batch_table),
        )
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
                    "encoded_columns": result.encoded_columns,
                    "candidates": len(result.candidates),
                    "clusters": len(result.clusters.clusters),
                    "decisions": len(result.decisions),
                },
                "input_copies": inputs,
            },
            required=REQUIRED_ARTIFACTS + tuple(inputs.values()) + (VOCABULARY_COPY,),
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
        taskconfig.load(settings, frozen=True),
        output_root=run.parent,
        workspace=str(run.parent.parent),
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
    pkg.write_csv(TASK_LAYOUT["dataset_encoded.csv"], result.encoded)
    pkg.write_csv(TASK_LAYOUT["dataset_columns.csv"], result.columns)
    pkg.write_csv(
        TASK_LAYOUT["encoded_columns.csv"],
        pd.DataFrame(result.dictionary, columns=["output", "source", "strategy", "detail"]).rename(
            columns={
                "output": "Feature",
                "source": "Parameter",
                "strategy": "Encoding",
                "detail": "Meaning",
            }
        ),
    )
    pkg.write_csv(TASK_LAYOUT["parameters.csv"], [d.row() for d in result.decisions])
    pkg.write_csv(TASK_LAYOUT["batches.csv"], result.cohort.exclusions)
    retained = set(result.recipe.get("output_columns") or [])
    pkg.write_csv(TASK_LAYOUT["clusters.csv"], decorrelate.rows(result.clusters, retained))
    if result.missingness is not None and (
        result.missingness.groups or result.missingness.singletons
    ):
        pkg.write_csv(TASK_LAYOUT["missingness.csv"], missingness.rows(result.missingness))
    pkg.write_json(TASK_LAYOUT["encoding.json"], result.recipe)
    pkg.write_json(TASK_LAYOUT["groups.json"], result.groups)


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
    return [taskconfig.load(path) for path in files]


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
def validate(config: dict[str, Any] | None, specs: list[TaskSpec]) -> list[str]:
    """Everything that would stop a run, found without running a stage."""
    problems: list[str] = []
    if config:
        try:
            share.backend()
        except share.ShareError as exc:
            problems.append(str(exc))
            return problems
        try:
            workspace.load_vocabulary(config)
        except (ConfigError, names.VocabularyError) as exc:
            return problems + [str(exc)]
        except Exception as exc:  # unreachable: the stages would stop the same way
            return problems + [f"sharepoint.vocabulary could not be read: {exc}"]
        try:
            rules = workspace.load_corrections(config)
            log.success(MODULE, f"Corrections table: {len(rules)} rules")
        except corrections.CorrectionsError as exc:
            problems.append(str(exc))
        problems += _unknown_keys(config)
        try:
            register = _register(config)
            if register is not None:
                log.success(
                    MODULE,
                    "Screening register",
                    screening.status_line(register.decisions, names.sites()),
                )
        except screening.ScreeningError as exc:
            problems.append(str(exc))
        except Exception as exc:  # not there yet is fine: `ml screening add --new` starts it
            log.warn(MODULE, f"The screening register could not be read: {exc}")
        if (config.get("sources") or {}).get("databricks"):
            try:
                enrich.ao_mapping()
            except RuntimeError as exc:
                problems.append(str(exc))
        missing = [
            label for label in (*BUILD_INPUTS, "batch_table") if not workspace.has(config, label)
        ]
        problems += [f"sharepoint.{label}: not set; `ml build` needs it" for label in missing]
    for spec in specs:
        problems += [f"{spec.name}: {p}" for p in _check_task(config, spec)]
    return problems


def _missingness_cohort(
    batches: pd.DataFrame, site: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, str]:
    """The site's batches the missingness, correlation and shifts reports describe, and how.

    ``build.missingness_cohort`` keeps the batches whose ``column`` holds ``value``
    (both vocabulary keys; compared without case or surrounding spaces), except at
    the sites in ``all_batches_at``, where every batch qualifies. Unset, every
    batch is described.
    """
    rule = (config.get("build") or {}).get("missingness_cohort")
    if not rule:
        return batches, ""
    wanted = names.value(rule["value"])
    if site in {names.site(key) for key in rule.get("all_batches_at") or []}:
        return batches, f"all {len(batches):,} batches (every {wanted} batch at this site)"
    column = names.col(rule["column"])
    if column not in batches.columns:
        raise ValueError(
            f"build.missingness_cohort: {site} has no '{column}' column to select "
            f"{wanted} batches by"
        )
    recorded = batches[column].astype("string").str.strip().str.casefold()
    kept = batches[recorded == wanted.strip().casefold()]
    log.info(
        MODULE,
        f"[{site}] missingness report: {len(kept):,} of {len(batches):,} batches, "
        f"{column} = {wanted}",
    )
    return kept, f"{len(kept):,} of {len(batches):,} batches: {column} = {wanted}"


def _analysed_batches(
    batches: pd.DataFrame, site: str, config: dict[str, Any]
) -> tuple[pd.Series | None, str]:
    """Which of the described batches the missingness statistics are computed over.

    ``build.missingness_cohort.analyse_only`` keeps the batches whose ``column``
    holds ``value`` (vocabulary keys, compared without case); the others are drawn
    in the plot but left out of every statistic and group. A blank counts as not
    holding the value. A site without the column has all its batches analysed.
    """
    rule = ((config.get("build") or {}).get("missingness_cohort") or {}).get("analyse_only")
    if not rule:
        return None, ""
    column, wanted = names.col(rule["column"]), names.value(rule["value"])
    if column not in batches.columns:
        log.warn(
            MODULE,
            f"[{site}] missingness report: no '{column}' column, so every batch is analysed",
        )
        return None, f"no {column} column: all analysed"
    recorded = batches[column].astype("string").str.strip().str.casefold()
    analysed = (recorded == wanted.strip().casefold()).fillna(False)
    log.info(
        MODULE,
        f"[{site}] missingness statistics over {int(analysed.sum()):,} of {len(batches):,} "
        f"batches, {column} = {wanted}; the others are drawn only",
    )
    return analysed, (
        f"statistics over the {int(analysed.sum()):,} with {column} = {wanted}, "
        f"the other {int((~analysed).sum()):,} drawn only"
    )


def _unknown_keys(config: dict[str, Any]) -> list[str]:
    """Settings that name a column by a key the loaded vocabulary does not define."""
    named = {
        "cleaning.yes_no_defaults": list(
            (config.get("cleaning") or {}).get("yes_no_defaults") or {}
        ),
        "mapping.exclude": list((config.get("mapping") or {}).get("exclude") or []),
        "build.date_column": [(config.get("build") or {}).get("date_column", DATE_COLUMN)],
    }
    problems = [
        f"{where}: '{key}' is not a column key of the vocabulary (see ml.names)"
        for where, keys in named.items()
        for key in keys
        if str(key) not in names.current().columns
    ]
    cohort = (config.get("build") or {}).get("missingness_cohort") or {}
    if cohort:
        if str(cohort.get("column")) not in names.current().columns:
            problems.append(
                f"build.missingness_cohort.column: '{cohort.get('column')}' is not a column "
                "key of the vocabulary (see ml.names)"
            )
        if str(cohort.get("value")) not in names.current().values:
            problems.append(
                f"build.missingness_cohort.value: '{cohort.get('value')}' is not a value "
                "key of the vocabulary (see ml.names)"
            )
        only = cohort.get("analyse_only") or {}
        if only and str(only.get("column")) not in names.current().columns:
            problems.append(
                f"build.missingness_cohort.analyse_only.column: '{only.get('column')}' is not "
                "a column key of the vocabulary (see ml.names)"
            )
        if only and str(only.get("value")) not in names.current().values:
            problems.append(
                f"build.missingness_cohort.analyse_only.value: '{only.get('value')}' is not "
                "a value key of the vocabulary (see ml.names)"
            )
        for key in cohort.get("all_batches_at") or []:
            if str(key) not in names.current().sites:
                problems.append(
                    f"build.missingness_cohort.all_batches_at: '{key}' is not a site key (a or b)"
                )
    return problems


def _check_task(config: dict[str, Any] | None, spec: TaskSpec) -> list[str]:
    """A task's columns against the batch table and catalog it would read, where they exist."""
    problems: list[str] = []
    try:
        batch_table = workspace.table_source(config, "batch_table", spec.batch_table).read()
    except ConfigError as exc:
        return [str(exc)]
    except Exception as exc:  # not built yet, or not reachable: either way, say so
        return [f"the batch table could not be read ({exc}) — run `ml build` first"]
    columns = set(batch_table.columns)
    named = {
        "target.column": [spec.target.column],
        "roles.id": [spec.roles.id],
        "roles.group": [spec.roles.group],
        "roles.patient": [spec.roles.patient],
        "predictors.include": list(spec.predictors.include),
        "cohort.filters": [f.column for f in spec.cohort.filters],
        "cohort.date_column": [spec.cohort.date_column],
    }
    for where, wanted in named.items():
        for name in wanted:
            if name and name not in columns:
                problems.append(f"{where}: '{name}' is not a column of the batch table")
    if spec.cohort.sites and spec.cohort.site_column not in columns:
        problems.append(
            f"cohort.site_column: '{spec.cohort.site_column}' is not in the batch table"
        )
    try:
        frame = workspace.table_source(config, "catalog", spec.catalog).read()
    except Exception as exc:  # the catalog check is a bonus; the run reports it properly
        problems.append(f"the catalog could not be read ({exc})")
        return problems
    for key in ("include_by_catalog", "exclude_by_catalog"):
        for column, _ in getattr(spec.predictors, key):
            if column not in frame.columns:
                problems.append(f"predictors.{key}: '{column}' is not a column of the catalog")
    return problems


def check_leaks(config: dict[str, Any], folder: Path) -> int:
    """Search a repository for the private vocabulary's names; 1 when any turns up."""
    from . import leaks

    vocabulary = workspace.load_vocabulary(config)
    rules = workspace.load_corrections(config)
    terms = leaks.vocabulary_terms(vocabulary, rules)
    root = folder.expanduser().resolve()
    files = leaks.tracked_files(root)
    log.step(MODULE, f"Searching {len(files):,} files for {len(terms):,} names", str(root))
    found = leaks.scan(files, terms, root)
    if not found:
        log.success(MODULE, "None of the vocabulary's names is in the repository")
        return 0
    log.names(
        "Lines that hold a name from the private vocabulary",
        [f"{leak.path}:{leak.line}  {leak.name!r}  — {leak.text}" for leak in found],
        level="ERROR",
    )
    return 1


def encode_batches(run: str | Path, rows: pd.DataFrame, out: Path | None) -> Path:
    """Encode batches with a finished run's encoding and write them out."""
    run = Path(run)
    settings = yaml.safe_load((run / TASK_LAYOUT["task.yaml"]).read_text(encoding="utf-8"))
    identifier = ((settings or {}).get("roles") or {}).get("id") or ""
    encoded = recipe.apply(run, rows)
    if identifier and identifier in rows.columns:
        encoded.insert(0, identifier, rows[identifier])
    out = out or Path.cwd() / f"{run.parent.name}-{run.name}-applied.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(out, index=False)
    log.success(MODULE, f"{len(encoded):,} batches encoded with {run.name}'s encoding → {out}")
    return out


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml",
        description="CAR-T manufacturing data: the catalog, the batch table, and datasets from it",
        epilog="Set the SharePoint locations in config.yaml, run `ml check-catalog` and "
        "`ml build`, then "
        "`ml task new <folder>` for a task.",
    )
    parser.add_argument(
        "--config", help="the config (default: $ML_CONFIG, or the nearest config.yaml)"
    )
    parser.add_argument("--debug", action="store_true", help="show tracebacks for every error")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="print only warnings, errors and what was read and written; for log files",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    validate = commands.add_parser("validate", help="validate the config and tasks; run nothing")
    validate.add_argument("--task", action="append", help="a task workspace (repeatable)")
    commands.add_parser("check-catalog", help="what the sources record that the catalog lacks")
    leaks_cmd = commands.add_parser(
        "check-leaks", help="search the repository for any name the private vocabulary holds"
    )
    leaks_cmd.add_argument(
        "folder", nargs="?", default=".", help="the repository to search (default: here)"
    )
    commands.add_parser("build", help="build the batch table from both sites")
    update = commands.add_parser("update", help="check-catalog, build, then the given tasks")
    update.add_argument("--task", action="append", help="a task workspace (repeatable)")

    screen = commands.add_parser(
        "screening", help="leave parameters out of every task, with a recorded reason"
    )
    steps = screen.add_subparsers(dest="action", required=True, metavar="action")
    add = steps.add_parser("add", help="add the decisions in a proposal file to the register")
    add.add_argument("proposal", help="a .csv or .xlsx with one decision per row")
    add.add_argument(
        "--author", default="", help="who takes these decisions, where the proposal has no Author"
    )
    add.add_argument("--new", action="store_true", help="start the register if there is none")
    steps.add_parser("status", help="what is excluded today, and what to look at again")

    task = commands.add_parser("task", help="create, run, rerun and encode with tasks")
    actions = task.add_subparsers(dest="action", required=True, metavar="action")
    new = actions.add_parser("new", help="create a task workspace with a task.yaml")
    new.add_argument("folder", help="the task workspace folder; its name is the task's name")
    new.add_argument("--target", default="", help="the column the task predicts")
    new.add_argument("--question", default="", help="the question, in words")
    new.add_argument("--force", action="store_true", help="overwrite an existing task file")
    run = actions.add_parser("run", help="run tasks, each into a new run folder")
    run.add_argument(
        "--task",
        action="append",
        help="a task workspace, its task.yaml, or a folder of workspaces (repeatable); "
        "default: the workspace in this folder",
    )
    redo = actions.add_parser("rerun", help="redo a run from its own folder, and compare")
    redo.add_argument("run", help="the run folder, <task workspace>/runs/<run id>")
    encode = actions.add_parser("encode", help="encode batches with a run's fitted encoding")
    encode.add_argument("--run", required=True, help="the task run folder")
    encode.add_argument("--out", help="the CSV to write (default: in the working directory)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    log.set_quiet(args.quiet)
    args.command = " ".join(filter(None, (args.command, getattr(args, "action", None))))
    try:
        return _run(args)
    except USER_ERRORS as exc:
        if args.debug:
            raise
        print(f"\nml {args.command}: {exc}", file=sys.stderr)
        return 2
    except Exception:
        if args.debug:
            raise
        traceback.print_exc()
        print(f"\nml {args.command}: failed unexpectedly (details above)", file=sys.stderr)
        return 1


def _run(args) -> int:
    command = args.command
    if command in ("validate", "task run") and not args.task and Path(TASK_FILE).is_file():
        args.task = ["."]
    first_task = (getattr(args, "task", None) or [None])[0]

    if command == "task new":
        found = _config(args, start=args.folder, required=False)
        screened, as_of = _screened_for_new_task(found)
        path = scaffold.init(
            args.folder,
            args.target,
            args.question,
            args.force,
            config=found["__path__"] if found else None,
            screened=screened,
            screened_as_of=as_of,
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

    if command == "task rerun":
        rerun(args.run)
        return 0

    if command == "task encode":
        config = _config(args)
        workspace.load_vocabulary(config)
        rows = io.load_batch_table(workspace.source(config, "batch_table"))
        encode_batches(args.run, rows, Path(args.out) if args.out else None)
        return 0

    if command == "screening add":
        run_screening_add(_config(args), args.proposal, args.author, new=args.new)
        return 0

    if command == "screening status":
        run_screening_status(_config(args))
        return 0

    if command == "check-leaks":
        return check_leaks(_config(args), Path(args.folder))

    if command == "validate":
        config = _config(args, start=first_task, required=False)
        specs = load_tasks(args, config) if args.task else []
        problems = validate(config, specs)
        for spec in specs:
            log.success(MODULE, f"Task '{spec.name}' is valid", spec.source)
        if problems:
            log.names("Problems that would stop a run", problems, level="ERROR")
            return 2
        log.success(MODULE, "Everything checked out")
        return 0

    if command == "task run":
        config = _config(args, start=first_task)
        for spec in load_tasks(args, config):
            run_task(config, spec)
        return 0

    config = _config(args, start=first_task)
    if command in ("check-catalog", "update"):
        run_catalog(config)
    if command in ("build", "update"):
        run_build(config)
    if command == "update" and args.task:
        for spec in load_tasks(args, config):
            run_task(config, spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
