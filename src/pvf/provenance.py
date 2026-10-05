"""Provenance capture for pipeline outputs.

Every artefact this pipeline writes should be traceable back to the exact inputs
that produced it: which commit of the code, which bytes of data, which seed,
which versions of the libraries that can move a number.

Two rules this module exists to keep. A stage records the sources it *actually*
read — not the ones config happens to name — so a run that fell back to a local
copy cannot be read as one that used the share. And nothing here raises: a run
must not die because the code sits outside a git checkout, so anything that
cannot be determined is recorded as unknown rather than quietly omitted.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .logger import log

MODULE = "provenance"


@dataclass(frozen=True)
class Source:
    """One input a stage read, and what can be said about its contents."""

    label: str
    location: str
    kind: str = "local file"
    digest: str | None = None
    digest_of: str = ""
    note: str = ""


def local(label: str, path: str | Path, note: str = "") -> Source:
    """A file on this machine, hashed as it is read."""
    return Source(
        label=label,
        location=str(path),
        kind="local file",
        digest=file_sha256(str(path)),
        digest_of="file bytes",
        note=note,
    )


def remote(label: str, location: str, frame: Any = None, note: str = "") -> Source:
    """A source somewhere else, identified by where it was read from.

    There are no bytes to hash on this side of a SharePoint or warehouse read, so
    what is recorded is the location and a digest of the table that came back —
    labelled as such. A digest of an unused local copy would be worse than none:
    it would look like the remote input's digest.
    """
    return Source(
        label=label,
        location=location,
        kind="remote",
        digest=frame_digest(frame) if frame is not None else None,
        digest_of="the table as it was consumed" if frame is not None else "",
        note=note,
    )


def missing(label: str, location: str, why: str) -> Source:
    """A source that was not read, recorded rather than left out."""
    return Source(label=label, location=location, kind="not read", digest=None, note=why)


def git_commit(repo_dir: Path | None = None) -> str | None:
    """Return the current commit SHA, or ``None`` outside a git checkout.

    Parameters
    ----------
    repo_dir : pathlib.Path, optional
        Directory to inspect. Defaults to the package's own location.

    Returns
    -------
    str or None
        Full SHA, suffixed with ``"-dirty"`` when the working tree has
        uncommitted changes. ``None`` if this is not a git checkout, git is
        not installed, or the repo has no commits yet.
    """
    cwd = repo_dir or Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if sha.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        suffix = "-dirty" if dirty.stdout.strip() else ""
        return sha.stdout.strip() + suffix
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path: str) -> str | None:
    """Hash a file's contents.

    This is the "data version" half of provenance: a filename says nothing
    about whether the file changed under you between runs, a digest does.

    Parameters
    ----------
    path : str
        File to hash.

    Returns
    -------
    str or None
        Hex digest, or ``None`` if the file cannot be read.
    """
    try:
        with open(path, "rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        log.warn(MODULE, f"Could not hash {path} for provenance", str(exc))
        return None


def frame_digest(frame: Any) -> str | None:
    """A digest of a table's values, for a source with no bytes to hash."""
    try:
        import pandas as pd

        hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
        columns = ",".join(map(str, frame.columns)).encode()
        return hashlib.sha256(columns + hashed).hexdigest()
    except Exception as exc:  # a digest is never worth failing a run over
        log.warn(MODULE, "Could not digest a table for provenance", str(exc))
        return None


def config_digest(config: Any) -> str | None:
    """A digest of the effective settings, defaults and all."""
    try:
        text = json.dumps(config, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(text.encode()).hexdigest()


# The libraries whose version can change a number in the report. A pinned commit
# and a data digest do not make a run reproducible if scipy moved underneath it.
_TRACKED_PACKAGES = (
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "plotly",
    "openpyxl",
    "pyyaml",
    "python-dotenv",
)


def environment() -> dict[str, str]:
    """Python and the versions of every library that can move a result.

    Recorded rather than pinned: this does not constrain the environment, it
    documents the one a given report came out of, which is what an auditor
    asking "can you reproduce this" actually needs. The report runtime is in
    here too — it is fetched in the browser, so its version is part of what a
    reader is looking at.
    """
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in _TRACKED_PACKAGES:
        try:
            env[name] = version(name)
        except PackageNotFoundError:
            env[name] = "not installed"
    from . import stlite  # imported here: provenance is also used before reports exist

    env["stlite (report runtime)"] = stlite.STLITE_VERSION
    return env


def is_dirty(repo_dir: Path | None = None) -> bool | None:
    """Whether the working tree has uncommitted changes. ``None`` outside git."""
    commit = git_commit(repo_dir)
    if commit is None:
        return None
    return commit.endswith("-dirty")


def capture(
    sources: dict[str, str] | list[Source],
    seed: int | None = None,
    config_path: str | None = None,
    config: Any = None,
) -> dict[str, Any]:
    """Collect the provenance block recorded alongside a run's outputs.

    Parameters
    ----------
    sources : list of Source, or dict
        What the stage actually read. A plain ``{label: path}`` mapping is taken
        as local files, which is what the earlier stages pass.
    seed : int, optional
        The single configured seed the run derives from.
    config_path : str, optional
        The config or task file the run was driven by.
    config : optional
        The effective settings, defaults included, to digest.

    Returns
    -------
    dict
        ``run_date``, ``git_commit``, ``seed``, ``config``, ``config_digest``,
        ``sources``, ``inputs`` and ``environment``. Any value may be ``None``
        when it could not be determined; nothing is omitted to hide a gap.
    """
    if isinstance(sources, dict):
        sources = [local(label, path) for label, path in sources.items()]
    return {
        "run_date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(Path.cwd()),
        "seed": seed,
        "config": config_path,
        "config_digest": config_digest(config) if config is not None else None,
        "sources": [asdict(source) for source in sources],
        # The flat form the reports have always read.
        "inputs": {source.label: source.digest for source in sources},
        "environment": environment(),
    }
