"""SharePoint: where every input is read from and every output is written to.

This is the one module that talks to the share. Every location comes from
``config.yaml`` (``sharepoint:``) as a :class:`Location` — a drive, a path and a
sheet — and nothing here knows an address.

The calls go to the internal ``io_sharepoint`` package. The demo and the tests
have no share, so they set ``$ML_SHAREPOINT_STANDIN`` to a folder that plays its
part: ``<folder>/<drive id>/<path>``. That is a stand-in for SharePoint, not a
local data option, and every run that uses it says so.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from .logger import log

MODULE = "sharepoint"

#: A folder that stands in for SharePoint. For the demo and the tests only.
STANDIN_ENV = "ML_SHAREPOINT_STANDIN"
#: The ``io_sharepoint`` function that fetches a file as bytes (a manifest, say).
DOWNLOAD = "download_file_from_sharepoint"


class ShareError(RuntimeError):
    """SharePoint cannot be reached, or a location on it cannot be used."""


def readable(path: str) -> str:
    """A SharePoint path as a person reads it: decoded, without doubled slashes."""
    return re.sub(r"/{2,}", "/", unquote(str(path)))


@dataclass(frozen=True)
class Location:
    """A file — or, ending in ``/``, a folder — on SharePoint."""

    drive_id: str
    path: str
    sheet: str | int = 0

    @property
    def name(self) -> str:
        """The file name, decoded."""
        return readable(self.path).rstrip("/").rsplit("/", 1)[-1]

    @property
    def folder(self) -> Location:
        """The folder this file is in, as a location ending in ``/``."""
        cut = self.path.rstrip("/").rfind("/")
        return Location(self.drive_id, self.path[: cut + 1] if cut >= 0 else "")

    def file(self, name: str, sheet: str | int = 0) -> Location:
        """Another file in the same folder (or, for a folder, in it)."""
        folder = self.path if self.path.endswith("/") else self.folder.path
        return Location(self.drive_id, f"{folder}{name}", sheet)

    def with_sheet(self, sheet: str | int) -> Location:
        return replace(self, sheet=sheet)

    @property
    def where(self) -> str:
        """The drive and the folder (this one, for a folder), for the line under a file name."""
        folder = self.path if self.path.endswith("/") else self.folder.path
        return f"{self.drive_id} › {readable(folder) or '/'}"

    def __str__(self) -> str:
        sheet = f" [{self.sheet}]" if self.sheet not in (0, "", None) else ""
        return f"SharePoint {self.drive_id}: {readable(self.path)}{sheet}"


# ---------------------------------------------------------------------------
# The two backends: io_sharepoint, and a folder standing in for it
# ---------------------------------------------------------------------------
class _IoSharepoint:
    """The real share, through the internal ``io_sharepoint`` package."""

    def __init__(self, module):
        self.module = module

    def read_excel(self, location: Location, header: int = 0) -> pd.DataFrame:
        kwargs = {
            "sharepoint_path": location.path,
            "sheet_name": location.sheet,
            "drive_id": location.drive_id,
        }
        if header:
            kwargs["header"] = header
        return self.module.load_excel_from_sharepoint(**kwargs)

    def read_bytes(self, location: Location) -> bytes | None:
        fetch = getattr(self.module, DOWNLOAD, None)
        if fetch is None:
            raise ShareError(f"io_sharepoint has no {DOWNLOAD}(), so it reads workbooks only")
        content = fetch(sharepoint_path=location.path, drive_id=location.drive_id)
        if hasattr(content, "read"):
            content = content.read()
        return content.encode("utf-8") if isinstance(content, str) else content

    def upload(self, local: Path, folder: Location) -> None:
        self.module.upload_file_to_sharepoint(
            local_path=local, sharepoint_target_path=folder.path, drive_id=folder.drive_id
        )


class _Standin:
    """A folder playing SharePoint: ``<folder>/<drive id>/<path>``. Demo and tests only."""

    def __init__(self, root: Path):
        self.standin = root

    def _file(self, location: Location) -> Path:
        return self.standin / location.drive_id / readable(location.path).lstrip("/")

    def read_excel(self, location: Location, header: int = 0) -> pd.DataFrame:
        path = self._file(location)
        if not path.is_file():
            raise FileNotFoundError(f"{location} does not exist")
        return pd.read_excel(path, sheet_name=location.sheet, header=header)

    def read_bytes(self, location: Location) -> bytes | None:
        path = self._file(location)
        return path.read_bytes() if path.is_file() else None

    def upload(self, local: Path, folder: Location) -> None:
        target = self._file(folder.file(Path(local).name))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(local).read_bytes())


_announced: set[str] = set()


def backend():
    """The share, or a :class:`ShareError` that says why it cannot be reached."""
    standin = os.getenv(STANDIN_ENV)
    if standin:
        if standin not in _announced:
            _announced.add(standin)
            log.warn(MODULE, "SharePoint is simulated by a local folder (demo/tests)", standin)
        return _Standin(Path(standin).expanduser())
    try:
        import io_sharepoint
    except ImportError as exc:
        raise ShareError(
            "Every input is read from SharePoint, but io_sharepoint is not installed"
        ) from exc
    return _IoSharepoint(io_sharepoint)


# ---------------------------------------------------------------------------
# What the stages call
# ---------------------------------------------------------------------------
def read_excel(location: Location, header: int = 0) -> pd.DataFrame:
    """A worksheet on the share, as a table."""
    return backend().read_excel(location, header)


def read_text(location: Location) -> str | None:
    """A text file on the share (a manifest), or None when it is not there.

    ``io_sharepoint`` reads workbooks; it fetches other files only if it has
    :data:`DOWNLOAD`. Without it, or when the fetch fails, the reason is logged
    and the caller is told the file could not be read: a finding, not a failure.
    """
    try:
        content = backend().read_bytes(location)
    except Exception as exc:  # an absent file is a finding here, not a failure
        log.warn(MODULE, f"{location.name} could not be read from SharePoint", str(exc))
        return None
    return content.decode("utf-8") if content is not None else None


def upload(files: dict[str, Path], folder: Location) -> list[tuple[str, Location]]:
    """Put each written file into one SharePoint folder; returns where each went."""
    share = backend()
    done: list[tuple[str, Location]] = []
    for what, local in files.items():
        name = Path(local).name
        log.step(MODULE, f"Uploading {what}", f"{name}  →  {folder.where}")
        share.upload(Path(local), folder)
        log.success(MODULE, f"Uploaded {name} ({size(Path(local))})", folder.where)
        done.append((what, folder.file(name)))
    return done


def size(path: Path) -> str:
    """A file's size as a person reads it: 812 KB, 3.4 MB."""
    count = float(path.stat().st_size) if path.is_file() else 0.0
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit in ("B", "KB") else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} GB"
