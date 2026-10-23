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
#: The ``io_sharepoint`` function that writes a string as a file (a manifest, say).
WRITE_TEXT = "write_text_to_sharepoint"
#: The ``io_sharepoint`` function that reads a CSV file as a table.
READ_CSV = "load_csv_from_sharepoint"
#: File endings read as CSV; anything else is read as a workbook.
CSV_SUFFIXES = (".csv", ".txt")


class ShareError(RuntimeError):
    """SharePoint cannot be reached, or a location on it cannot be used."""


def readable(path: str) -> str:
    """A SharePoint path as a person reads it: decoded, without doubled slashes."""
    return re.sub(r"/{2,}", "/", unquote(str(path)))


@dataclass(frozen=True)
class Location:
    """A file — or, ending in ``/``, a folder — on SharePoint.

    ``sheet`` is the worksheet of a workbook; ``sep`` the delimiter of a CSV
    file, None to let it be detected.
    """

    drive_id: str
    path: str
    sheet: str | int = 0
    sep: str | None = None

    @property
    def name(self) -> str:
        """The file name, decoded."""
        return readable(self.path).rstrip("/").rsplit("/", 1)[-1]

    @property
    def is_csv(self) -> bool:
        """Whether the file is read as CSV rather than as a workbook."""
        return self.name.lower().endswith(CSV_SUFFIXES)

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
        sheet = f" [{self.sheet}]" if self.sheet not in (0, "", None) and not self.is_csv else ""
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

    def read_csv(self, location: Location, header: int = 0) -> pd.DataFrame:
        load = self._function(READ_CSV, "reads no CSV files")
        return load(
            sharepoint_path=location.path,
            header=header,
            drive_id=location.drive_id,
            **csv_options(location.sep),
        )

    def read_bytes(self, location: Location) -> bytes | None:
        fetch = self._function(DOWNLOAD, "reads workbooks only")
        content = fetch(sharepoint_path=location.path, drive_id=location.drive_id)
        if hasattr(content, "read"):
            content = content.read()
        return content.encode("utf-8") if isinstance(content, str) else content

    def write_text(self, text: str, location: Location) -> None:
        write = self._function(WRITE_TEXT, "cannot write a manifest")
        write(
            text=text,
            sharepoint_target_folder_path=location.folder.path,
            file_name=location.name,
            drive_id=location.drive_id,
        )

    def upload(self, local: Path, folder: Location) -> None:
        self.module.upload_file_to_sharepoint(
            local_path=local, sharepoint_target_path=folder.path, drive_id=folder.drive_id
        )

    def _function(self, name: str, without: str):
        found = getattr(self.module, name, None)
        if found is None:
            raise ShareError(f"io_sharepoint has no {name}(), so it {without}; update it")
        return found


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

    def read_csv(self, location: Location, header: int = 0) -> pd.DataFrame:
        path = self._file(location)
        if not path.is_file():
            raise FileNotFoundError(f"{location} does not exist")
        return pd.read_csv(path, header=header, **csv_options(location.sep))

    def read_bytes(self, location: Location) -> bytes | None:
        path = self._file(location)
        return path.read_bytes() if path.is_file() else None

    def write_text(self, text: str, location: Location) -> None:
        target = self._file(location)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="")

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


def read_table(location: Location, header: int = 0) -> pd.DataFrame:
    """A file on the share as a table: a CSV file by its ending, else a worksheet."""
    if location.is_csv:
        return backend().read_csv(location, header)
    return backend().read_excel(location, header)


def csv_options(sep: str | None) -> dict:
    """How a CSV file is parsed: a byte-order mark dropped, the delimiter detected unless given."""
    if sep:
        return {"sep": sep, "encoding": "utf-8-sig"}
    return {"sep": None, "engine": "python", "encoding": "utf-8-sig"}


def read_text(location: Location) -> str | None:
    """A text file on the share (a manifest), or None when it is not there.

    It is fetched with ``io_sharepoint``'s :data:`DOWNLOAD`. When the fetch fails
    (an older ``io_sharepoint`` without it, say), the reason is logged and the
    caller is told the file could not be read: a finding, not a failure.
    """
    try:
        content = backend().read_bytes(location)
    except Exception as exc:  # an absent file is a finding here, not a failure
        log.warn(MODULE, f"{location.name} could not be read from SharePoint", str(exc))
        return None
    return content.decode("utf-8") if content is not None else None


def write_text(what: str, text: str, location: Location) -> Location:
    """Write a text file (a manifest) straight to the share; returns where it went."""
    log.step(MODULE, f"Writing {what}", f"{location.name}  →  {location.where}")
    backend().write_text(text, location)
    log.success(MODULE, f"Wrote {location.name} ({_bytes(len(text.encode()))})", location.where)
    return location


def upload(files: dict[str, Path], folder: Location) -> list[tuple[str, Location]]:
    """Put each written file into one SharePoint folder; returns where each went."""
    share = backend()
    done: list[tuple[str, Location]] = []
    for what, local in files.items():
        name = Path(local).name
        log.step(MODULE, f"Uploading {what}", f"{name}  →  {folder.where}")
        try:
            share.upload(Path(local), folder)
        except Exception as exc:
            response = getattr(exc, "response", None)
            if getattr(response, "status_code", None) == 409:
                raise ShareError(
                    f"SharePoint refused to replace {name} in {folder.where} (409 Conflict): it "
                    "is open in Excel (desktop or browser, yours or a colleague's) or checked "
                    "out. Close it everywhere, check it in, wait a minute and run this again"
                ) from exc
            raise
        log.success(MODULE, f"Uploaded {name} ({size(Path(local))})", folder.where)
        done.append((what, folder.file(name)))
    return done


def size(path: Path) -> str:
    """A file's size as a person reads it: 812 KB, 3.4 MB."""
    return _bytes(path.stat().st_size if path.is_file() else 0)


def _bytes(count: float) -> str:
    count = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit in ("B", "KB") else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} GB"
