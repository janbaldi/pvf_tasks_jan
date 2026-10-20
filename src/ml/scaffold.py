"""Creating a task workspace.

``ml task new <folder>`` makes a folder with a commented ``task.yaml`` in it.
The folder is the task's workspace: every run of the task is written to its
``runs/`` folder, with the batch table and catalog it read, so the folder holds everything
needed to redo the analysis. The batch table and catalog themselves are read from
SharePoint, where the ``config.yaml`` the task names locates them.
"""

from __future__ import annotations

import json
import os
import re
from importlib import resources
from pathlib import Path

from .config import ConfigError
from .taskconfig import TASK_FILE


def template(name: str) -> str:
    return (resources.files("ml") / "templates" / name).read_text(encoding="utf-8")


def init(
    folder: str | Path,
    target: str = "",
    question: str = "",
    force: bool = False,
    config: str | Path | None = None,
    screened: list[tuple[str, str]] = (),
    screened_as_of: str = "",
) -> Path:
    """Write ``<folder>/task.yaml``, pointing at ``config``. Returns its path.

    ``screened`` is (parameter, reason) for every parameter the screening register
    excludes when the task is made. They are written out as ``predictors.exclude``
    entries, so the task file itself says what it leaves out and why.
    """
    path = (Path(folder).expanduser() / TASK_FILE).resolve()
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists; pass --force to overwrite it")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.parent.name).strip("_") or "task"
    link = (
        "config: " + os.path.relpath(Path(config).resolve(), path.parent).replace("\\", "/")
        if config
        else "# config: path/to/config.yaml"
    )
    text = (
        template("task.yaml")
        .replace("__CONFIG__", link)
        .replace("__NAME__", slug)
        .replace("__QUESTION__", question or f"What goes with {target or 'the target'}?")
        .replace("__TARGET__", target or "<the column to predict>")
        .replace("__TARGET_TYPE__", "numeric")
        .replace(
            "  exclude: []                     # single columns: [{column: ..., reason: ...}]\n",
            _excludes(screened, screened_as_of),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _excludes(screened: list[tuple[str, str]], as_of: str) -> str:
    """The ``predictors.exclude`` lines of a new task: empty, or the screened parameters."""
    if not screened:
        return "  exclude: []                     # single columns: [{column: ..., reason: ...}]\n"
    lines = [
        "  exclude:                        # single columns: [{column: ..., reason: ...}]",
        f"    # Screened out by the screening register (last decision {as_of}), at any site.",
        "    # Every run applies the register anyway; these lines make it visible here.",
        "    # A run warns when one of them has since been reinstated: then delete the line.",
    ]
    lines += [
        f"    - {{column: {json.dumps(column)}, reason: {json.dumps(reason)}}}"
        for column, reason in screened
    ]
    return "\n".join(lines) + "\n"
