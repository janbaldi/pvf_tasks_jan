"""One manifest format for every stage.

``pvf ptf`` and ``pvf build`` write ``ptf_manifest.json`` and
``pvf_manifest.json`` beside the PVF, which is where a task looks for them; each
task run writes ``manifest.json`` at the top of its folder. All three have the
same envelope, so one reader and one check serve them all:

``manifest_version``  the version of this format
``stage``             ``ptf``, ``pvf`` or ``task``
``status``            ``complete`` or ``failed``
``run_id``            sortable by time, unique
``started``, ``finished``
``config``            the config or task file the run was driven by, and its digest
``inputs``            every source the stage *actually* read: where from, how, digest
``outputs``           every file it wrote: path, size, sha256
``warnings``, ``errors``
``provenance``        commit, seed, environment

Paths in ``outputs`` are relative to the manifest's own folder when the file is
inside it or below it, so a folder can be moved without its manifest going stale.
Stage-specific facts (counts, the PVF's shape, the PTF gap) sit beside these.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from . import provenance

MANIFEST_VERSION = 2
STAGES = ("ptf", "pvf", "task")
STATUSES = ("complete", "failed")
#: Every manifest has these, whatever its stage.
ENVELOPE = (
    "manifest_version",
    "stage",
    "status",
    "run_id",
    "started",
    "finished",
    "config",
    "inputs",
    "outputs",
    "warnings",
    "errors",
    "provenance",
)
#: Each stage's own facts, required on a complete manifest.
STAGE_FIELDS = {
    "ptf": ("ptf", "sources_compared", "not_in_ptf", "new_parameters"),
    "pvf": ("pvf", "ptf", "ptf_parameters_not_in_pvf", "pvf_parameters_not_in_ptf"),
    "task": ("task", "purpose", "dataset_role", "counts", "upstream_manifests"),
}


def output_entry(base: Path, path: str | Path, what: str = "") -> dict[str, Any]:
    """One written file: where (relative to ``base`` when it can be), size and digest."""
    path = Path(path)
    try:
        shown = path.resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        shown = os.path.relpath(path.resolve(), Path(base).resolve()).replace("\\", "/")
    return {
        "what": what or path.name,
        "path": shown,
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": provenance.file_sha256(str(path)) if path.exists() else None,
    }


def outputs(base: Path, files: dict[str, str | Path | None]) -> list[dict[str, Any]]:
    return [output_entry(base, path, what) for what, path in files.items() if path]


def inputs(sources: list[provenance.Source]) -> list[dict[str, Any]]:
    return [asdict(source) for source in sources]


def envelope(
    stage: str,
    *,
    run_id: str,
    started: datetime,
    status: str = "complete",
    config_path: str = "",
    prov: dict | None = None,
    sources: list[provenance.Source] | None = None,
    written: list[dict[str, Any]] | None = None,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    **body: Any,
) -> dict[str, Any]:
    """A manifest in the shared format. ``body`` holds the stage's own facts."""
    prov = prov or {}
    return {
        "manifest_version": MANIFEST_VERSION,
        "stage": stage,
        "status": status,
        "run_id": run_id,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "config": {"path": config_path, "digest": prov.get("config_digest")},
        "inputs": inputs(sources or []),
        "outputs": list(written or []),
        "warnings": list(warnings or []),
        "errors": list(errors or []),
        **body,
        "provenance": prov,
    }


def validate(record: dict[str, Any]) -> list[str]:
    """What is wrong with a manifest, as a list of problems (empty when it is sound)."""
    problems = [f"missing '{key}'" for key in ENVELOPE if key not in record]
    if problems:
        return problems
    if record["manifest_version"] != MANIFEST_VERSION:
        problems.append(f"manifest_version is {record['manifest_version']}, not {MANIFEST_VERSION}")
    if record["stage"] not in STAGES:
        problems.append(f"stage '{record['stage']}' is not one of {', '.join(STAGES)}")
        return problems
    if record["status"] not in STATUSES:
        problems.append(f"status '{record['status']}' is not one of {', '.join(STATUSES)}")
    if record["status"] == "complete":
        problems += [
            f"a complete {record['stage']} manifest has no '{key}'"
            for key in STAGE_FIELDS[record["stage"]]
            if key not in record
        ]
        for entry in record["outputs"]:
            if not entry.get("path") or not entry.get("sha256"):
                name = entry.get("what") or entry.get("path")
                problems.append(f"output {name} has no path or digest")
    for entry in record["inputs"]:
        if not {"label", "location", "kind"} <= set(entry):
            problems.append(f"input {entry} does not say what, where and how it was read")
    return problems


def write(path: str | Path, record: dict[str, Any]) -> Path:
    """Write a manifest, refusing one that does not follow the format."""
    problems = validate(record)
    if problems:
        raise ValueError(f"{path} would not be a valid manifest: " + "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(path: str | Path) -> list[str]:
    """Problems with a manifest on disk: its format, and outputs whose bytes changed."""
    path = Path(path)
    record = read(path)
    problems = validate(record)
    for entry in record.get("outputs") or []:
        target = (path.parent / entry["path"]) if entry.get("path") else None
        if target is None or not target.exists():
            problems.append(f"{entry.get('path')}: listed but not on disk")
        elif entry.get("sha256") and provenance.file_sha256(str(target)) != entry["sha256"]:
            problems.append(f"{entry['path']}: changed since the manifest was written")
    return problems
