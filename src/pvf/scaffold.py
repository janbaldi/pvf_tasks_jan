"""Creating a workspace, and a task inside or outside one.

``pvf init <folder>`` writes a workspace: the config, the corrections table, an
empty ``tasks/`` folder and the data and output folders the config points at.
``--demo`` fills it with made-up sources and two example tasks, so every stage
runs straight away.

``pvf new-task <name or path>`` writes a commented task file. A bare name goes to
the workspace's ``tasks/`` folder; a path can be anywhere at all — the task file
records where its workspace is, relative to itself, and that is all it needs.
"""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path

from . import demo
from .config import ConfigError

TEMPLATES = ("pvf.yaml", "corrections.yaml")
FOLDERS = ("data/raw", "data/processed", "outputs/reports", "outputs/tasks", "tasks")

RAW_README = """Put the local copies of the sources here, under the names pvf.yaml gives them
(paths: in the config). With sources.location: sharepoint, the sources that have
a sharepoint: entry are read from SharePoint instead and need no copy here.
"""


def template(name: str) -> str:
    return (resources.files("pvf") / "templates" / name).read_text(encoding="utf-8")


def init(folder: str | Path, demo_data: bool = False, force: bool = False) -> Path:
    """Create a workspace. Returns the path of its config."""
    folder = Path(folder).expanduser()
    config_path = folder / "pvf.yaml"
    if config_path.exists() and not force:
        raise ConfigError(f"{config_path} already exists; pass --force to overwrite it")
    if demo_data:
        return demo.write_config(folder)
    for name in FOLDERS:
        (folder / name).mkdir(parents=True, exist_ok=True)
    for name in TEMPLATES:
        target = folder / name
        if not target.exists() or force:
            target.write_text(template(name), encoding="utf-8")
    example = folder / ".env.example"
    if not example.exists():
        example.write_text(template("env.example"), encoding="utf-8")
    readme = folder / "data" / "raw" / "README.txt"
    if not readme.exists():
        readme.write_text(RAW_README, encoding="utf-8")
    return config_path


def task_path(config: dict, name: str) -> Path:
    """Where a new task file goes: a path as given, a bare name into tasks/."""
    candidate = Path(name).expanduser()
    if candidate.suffix in (".yaml", ".yml") or len(candidate.parts) > 1:
        return candidate if candidate.suffix else candidate.with_suffix(".yaml")
    folder = Path(
        (config.get("paths") or {}).get("task_files") or Path(config["__dir__"]) / "tasks"
    )
    return folder / f"{candidate.name}.yaml"


def new_task(
    config: dict,
    name: str,
    target: str = "",
    purpose: str = "exploratory",
    question: str = "",
    force: bool = False,
) -> Path:
    """Write a task file wired to the workspace. Returns its path."""
    path = task_path(config, name).resolve()
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists; pass --force to overwrite it")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "task"
    workspace = os.path.relpath(config["__path__"], path.parent).replace("\\", "/")
    predictive = purpose == "predictive"
    text = (
        template("task.yaml")
        .replace("__FILE__", str(path))
        .replace("__WORKSPACE__", workspace)
        .replace("__NAME__", slug)
        .replace("__QUESTION__", question or f"What goes with {target or 'the target'}?")
        .replace("__PURPOSE__", purpose)
        .replace("__TARGET__", target or "FP Flow CAR+ (%)")
        .replace("__TARGET_TYPE__", "numeric")
        .replace("__CUTOFF__", "Harvest" if predictive else '""')
        .replace("__SPLIT__", "grouped" if predictive else '""')
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
