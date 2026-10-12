"""Creating a task workspace.

``pvf new-task <folder>`` makes a folder with a commented ``task.yaml`` in it.
The folder is the task's workspace: every run of the task is written to its
``runs/`` folder, with the PVF and PTF it read, so the folder holds everything
needed to redo the analysis. The PVF and PTF themselves come from ``config.yaml``.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from .config import ConfigError
from .taskconfig import TASK_FILE


def template(name: str) -> str:
    return (resources.files("pvf") / "templates" / name).read_text(encoding="utf-8")


def new_task(
    folder: str | Path,
    target: str = "",
    purpose: str = "exploratory",
    question: str = "",
    force: bool = False,
) -> Path:
    """Write ``<folder>/task.yaml``. Returns its path."""
    path = (Path(folder).expanduser() / TASK_FILE).resolve()
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists; pass --force to overwrite it")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.parent.name).strip("_") or "task"
    text = (
        template("task.yaml")
        .replace("__FOLDER__", str(path.parent))
        .replace("__NAME__", slug)
        .replace("__QUESTION__", question or f"What goes with {target or 'the target'}?")
        .replace("__PURPOSE__", purpose)
        .replace("__TARGET__", target or "FP Flow CAR+ (%)")
        .replace("__TARGET_TYPE__", "numeric")
        .replace("__CUTOFF__", "Harvest" if purpose == "predictive" else '""')
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
