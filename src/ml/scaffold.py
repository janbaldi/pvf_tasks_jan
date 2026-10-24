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
    shifting: list[tuple[str, str]] = (),
    shifting_built: str = "",
    conflicts: list[str] = (),
    site: str = "",
) -> Path:
    """Write ``<folder>/task.yaml``, pointing at ``config``. Returns its path.

    ``screened`` is (parameter, reason) for every parameter the screening register
    excludes when the task is made, and ``shifting`` for every parameter the latest
    build (run at ``shifting_built``) found running in groups over time, when asked
    for. They are written out as ``predictors.exclude`` entries, so the task file
    itself says what it leaves out and why; ``conflicts`` (representatives the
    register excludes too) as comments above them. ``site`` is the key of the site
    the task is about (config.yaml's ``site:``).
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
        .replace("site: __SITE__", f"site: {site}" if site else "# site: a")
        .replace("__QUESTION__", question or f"What goes with {target or 'the target'}?")
        .replace("__TARGET__", target or "<the column to predict>")
        .replace("__TARGET_TYPE__", "numeric")
        .replace(
            "  exclude: []                     # single columns: [{column: ..., reason: ...}]\n",
            _excludes(screened, screened_as_of, shifting, shifting_built, conflicts),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _excludes(
    screened: list[tuple[str, str]],
    as_of: str,
    shifting: list[tuple[str, str]] = (),
    built: str = "",
    conflicts: list[str] = (),
) -> str:
    """The ``predictors.exclude`` lines of a new task: empty, or the parameters left out."""
    if not screened and not shifting:
        return "  exclude: []                     # single columns: [{column: ..., reason: ...}]\n"
    lines = ["  exclude:                        # single columns: [{column: ..., reason: ...}]"]
    if screened:
        lines += [
            f"    # Screened out by the screening register (last decision {as_of}) at this site.",
            "    # Every run applies the register anyway; these lines make it visible here.",
            "    # A run warns when one of them has since been reinstated: then delete the line.",
        ]
        if conflicts:
            lines += [
                "    # CONFLICT: a representative the register excludes too, so its group is",
                "    # left out whole. Resolve it in the correlation report: pick another",
                "    # representative.",
            ]
            lines += [f"    #   {line}" for line in conflicts]
        lines += [_exclude(column, reason) for column, reason in screened]
    if shifting:
        lines += [
            f"    # Run in groups over time at a manufacturing site, as the build of {built}",
            "    # found (shifts_report.html; `ml task new --exclude-shifting`). A lot does this,",
            "    # and so the model would learn time. Delete a line to use the parameter again.",
        ]
        lines += [_exclude(column, reason) for column, reason in shifting]
    return "\n".join(lines) + "\n"


def _exclude(column: str, reason: str) -> str:
    return f"    - {{column: {json.dumps(column)}, reason: {json.dumps(reason)}}}"
