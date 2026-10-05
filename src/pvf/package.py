"""The folder one task run leaves behind.

A run writes into a staging directory whose name says it is unfinished, and that
directory only becomes the run's folder once every required artefact is there
and the counts inside them agree. A run that fails keeps its folder, marked
failed, with its log in it — a half-written folder that looks like a finished one
is worse than no folder at all.

Nothing is ever overwritten. Each run gets its own identifier, so two runs of the
same task sit side by side and the earlier one stays exactly as it was.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import manifest
from .logger import log

MODULE = "package"

STAGING_SUFFIX = ".incomplete"
FAILED_SUFFIX = ".failed"
MANIFEST = "manifest.json"

#: What each artefact is, as the manifest names it.
ARTEFACT_ROLES = {
    "task.yaml": "effective task settings",
    "dataset.csv": "dataset",
    "raw_features.csv": "untransformed predictors",
    "transformed.csv": "transformed predictors",
    "splits.csv": "split assignments",
    "columns.csv": "column dictionary",
    "features.csv": "feature dictionary",
    "decisions.csv": "parameter decisions",
    "cohort.csv": "cohort membership",
    "clusters.csv": "correlated groups",
    "missingness.csv": "parameters missing together",
    "recipe.json": "fitted preprocessing",
    "report.html": "report",
    "report_payload.json": "report data",
    "ptf_manifest.json": "PTF stage manifest",
    "pvf_manifest.json": "build stage manifest",
    "run.log": "run log",
}


def run_id(when: datetime | None = None) -> str:
    """A run identifier that sorts by time and does not collide."""
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


@dataclass
class TaskPackage:
    """One run's folder, while it is being written."""

    task: str
    root: Path
    identifier: str
    staging: Path
    destination: Path
    started: datetime = field(default_factory=datetime.now)
    written: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # -- writing ------------------------------------------------------------
    def _path(self, name: str) -> Path:
        path = (self.staging / name).resolve()
        if self.staging.resolve() not in path.parents:
            raise ValueError(f"'{name}' would be written outside the task folder")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self._path(name)
        path.write_text(text, encoding="utf-8")
        self.written.append(name)
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        return self.write_text(name, json.dumps(payload, indent=2, default=str))

    def write_csv(self, name: str, rows: pd.DataFrame | list[dict]) -> Path:
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        path = self._path(name)
        frame.to_csv(path, index=False)
        self.written.append(name)
        return path

    def path_for(self, name: str) -> Path:
        """Where an artefact something else writes — the report — should go."""
        path = self._path(name)
        self.written.append(name)
        return path

    # -- finishing ----------------------------------------------------------
    def artifacts(self) -> list[dict[str, Any]]:
        """Every file in the folder with its size and digest, manifest aside."""
        return [
            manifest.output_entry(self.staging, path, ARTEFACT_ROLES.get(path.name, path.stem))
            for path in sorted(self.staging.rglob("*"))
            if path.is_file() and path.name != MANIFEST
        ]

    def complete(self, fields: dict[str, Any], required: tuple[str, ...]) -> Path:
        """Check, write the manifest, and promote the folder to its final name.

        ``fields`` are the manifest's stage facts plus, optionally, ``config_path``,
        ``prov``, ``sources`` and ``errors`` (see :func:`pvf.manifest.envelope`).
        """
        absent = [name for name in required if not (self.staging / name).exists()]
        if absent:
            raise FileNotFoundError(
                f"The task package is missing {', '.join(absent)} — it was not completed"
            )
        fields = dict(fields)
        warnings = list(self.warnings) + list(fields.pop("warnings", None) or [])
        record = manifest.envelope(
            "task",
            run_id=self.identifier,
            started=self.started,
            status="complete",
            written=self.artifacts(),
            warnings=list(dict.fromkeys(warnings)),
            task=self.task,
            **fields,
        )
        manifest.write(self.staging / MANIFEST, record)
        if self.destination.exists():
            raise FileExistsError(f"{self.destination} already exists; nothing was overwritten")
        self.staging.rename(self.destination)
        log.success(MODULE, f"Task package written to {self.destination}")
        return self.destination

    def fail(self, reason: str, fields: dict[str, Any] | None = None) -> Path:
        """Keep what there is, marked failed, with why it failed inside it."""
        fields = dict(fields or {})
        warnings = list(self.warnings) + list(fields.pop("warnings", None) or [])
        record = manifest.envelope(
            "task",
            run_id=self.identifier,
            started=self.started,
            status="failed",
            written=self.artifacts() if self.staging.exists() else [],
            warnings=warnings,
            errors=[reason, *(fields.pop("errors", None) or [])],
            task=self.task,
            failure=reason,
            **fields,
        )
        self.staging.mkdir(parents=True, exist_ok=True)
        manifest.write(self.staging / MANIFEST, record)
        failed = self.destination.with_name(self.destination.name + FAILED_SUFFIX)
        if not failed.exists():
            self.staging.rename(failed)
        log.error(MODULE, f"Task failed — partial output kept at {failed}", reason)
        return failed


def open_package(root: str | Path, task: str, identifier: str | None = None) -> TaskPackage:
    """Start a run folder under ``<root>/<task>/``, in its staging name."""
    base = Path(root).expanduser() / task
    identifier = identifier or run_id()
    staging = base / f"{identifier}{STAGING_SUFFIX}"
    destination = base / identifier
    if destination.exists() or staging.exists():
        raise FileExistsError(f"{destination} already exists; choose another run identifier")
    staging.mkdir(parents=True)
    log.info(MODULE, f"Writing run {identifier} into {staging}")
    return TaskPackage(
        task=task, root=base, identifier=identifier, staging=staging, destination=destination
    )
