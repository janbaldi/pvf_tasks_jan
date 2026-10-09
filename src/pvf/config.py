"""The workspace config: where the inputs are, where they come from, and the settings.

Code, configuration and data live apart. This package is code only. A
*workspace* is any folder that holds a ``pvf.yaml`` — the config read here — next
to the data and outputs it names::

    my_workspace/
      pvf.yaml            this file: paths, sources, SharePoint locations, settings
      corrections.yaml    known data corrections, as a table (see pvf.corrections)
      tasks/              one YAML per task
      data/raw/           the local copies of the sources
      data/processed/     the PVF and the manifests of the stages that wrote it
      outputs/            reports and task packages

``pvf init <folder>`` writes that layout. Every relative path in the config is
relative to the config file, so a workspace can be moved or shared as a folder.

Where a source is read from is a setting too. With ``sources.location: local``
every input is the local file under ``paths``. With ``sharepoint``, every input
that has an entry under ``sharepoint:`` is read from that location instead — no
SharePoint address is written into the code.
"""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from . import io
from .features import Params
from .logger import log

MODULE = "config"

#: The file names a workspace config may have, in the order they are looked for.
CONFIG_NAMES = ("pvf.yaml", "pvf.yml")
#: Where configs lived before workspaces, relative to the working directory.
LEGACY_CONFIG = Path("config/config.yaml")
#: The environment variable that names a config, for runs started anywhere.
CONFIG_ENV = "PVF_CONFIG"

#: Config keys under ``paths:`` that name a file, a folder or a list of files.
PATH_KEYS = (
    "ptf",
    "phf_ghent",
    "phf_raritan",
    "param_mapping",
    "remfg",
    "lv_coa",
    "raw_materials",
    "investigations",
    "site_maps",
    "pvf",
    "new_parameters",
    "reports",
    "tasks",
    "task_files",
)
#: Inputs that may be read from SharePoint, by the label they have everywhere else.
SHAREPOINT_SOURCES = (
    "ptf",
    "pvf",
    "phf_ghent",
    "phf_raritan",
    "param_mapping",
    "remfg",
    "lv_coa",
    "raw_materials",
    "investigations",
    "site_maps",
)
TOP_KEYS = (
    "seed",
    "paths",
    "sheets",
    "sources",
    "sharepoint",
    "upload",
    "ptf",
    "build",
    "cleaning",
    "features",
    "lv_coa",
    "mapping",
    "tasks",
)
CLEANING_KEYS = (
    "censored",
    "censored_flags",
    "duration_unit",
    "yes_no_defaults",
    "raritan_type_commercial",
    "raritan_type_commercial_reason",
    "corrections",
    "disable",
)


class ConfigError(ValueError):
    """One or more problems with a workspace config, each naming its field."""


# ---------------------------------------------------------------------------
# Finding and reading
# ---------------------------------------------------------------------------
def find(explicit: str | Path | None = None, start: str | Path | None = None) -> Path:
    """The config a run uses.

    In order: the path given on the command line, ``$PVF_CONFIG``, a ``pvf.yaml``
    in ``start`` (a task file's folder, say) or any folder above it, the same
    from the working directory, and finally the old ``config/config.yaml``.
    """
    if explicit:
        return Path(explicit).expanduser()
    if os.getenv(CONFIG_ENV):
        return Path(os.environ[CONFIG_ENV]).expanduser()
    bases = [Path(start).expanduser().resolve()] if start else []
    bases.append(Path.cwd().resolve())
    for base in bases:
        if base.is_file():
            base = base.parent
        for folder in (base, *base.parents):
            for name in CONFIG_NAMES:
                if (folder / name).is_file():
                    return folder / name
    if LEGACY_CONFIG.is_file():
        return LEGACY_CONFIG
    raise ConfigError(
        "No workspace config found: there is no pvf.yaml here or in any folder above, "
        f"${CONFIG_ENV} is not set and --config was not given. "
        "Create a workspace with `pvf init <folder>` or pass --config <file>"
    )


def load(path: str | Path) -> dict[str, Any]:
    """Read a config, validate it, and resolve its paths against its own folder."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"{path} does not exist")
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"{path} is not a mapping")

    validate(config, str(path))
    base = path.parent
    paths = config.setdefault("paths", {}) or {}
    config["paths"] = paths
    for key in PATH_KEYS:
        value = paths.get(key)
        if isinstance(value, str) and value:
            paths[key] = str(_resolve(value, base))
        elif isinstance(value, list):
            paths[key] = [str(_resolve(str(item), base)) for item in value]
    cleaning = config.get("cleaning") or {}
    if isinstance(cleaning.get("corrections"), str) and cleaning["corrections"]:
        cleaning["corrections"] = str(_resolve(cleaning["corrections"], base))
    config["__path__"] = str(path)
    config["__dir__"] = str(base)
    config["__env__"] = load_env(base)
    return config


def load_env(folder: str | Path) -> str:
    """Load the workspace's ``.env`` — the SharePoint credentials and drive ids.

    Secrets belong with the workspace, next to ``pvf.yaml``, never in the code
    or in git. Variables already set in the environment win, so a scheduler or
    a shell can override the file. Returns the file loaded, or "".
    """
    from dotenv import load_dotenv

    env = Path(folder) / ".env"
    if env.is_file():
        load_dotenv(env, override=False)
        log.info(MODULE, f"Environment loaded from {env}")
        return str(env)
    return ""


def _resolve(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else base / candidate


def settings(config: dict[str, Any]) -> dict[str, Any]:
    """The config without the bookkeeping keys this module adds, for digests."""
    return {key: value for key, value in config.items() if not key.startswith("__")}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _mapping(value: Any, where: str, allowed: tuple[str, ...] | None, errors: list[str]) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        errors.append(f"{where}: expected a mapping, found {type(value).__name__}")
        return {}
    if allowed is not None:
        for key in value:
            if key not in allowed:
                errors.append(f"{where}.{key}: unknown setting (allowed: {', '.join(allowed)})")
    return value


def validate(config: dict[str, Any], source: str = "config") -> None:
    """Refuse a config with anything wrong in it, naming every problem at once."""
    errors: list[str] = []
    _mapping(config, "(root)", TOP_KEYS, errors)

    paths = _mapping(config.get("paths"), "paths", None, errors)
    for key, value in paths.items():
        if key == "site_maps":
            if not (value is None or isinstance(value, list)):
                errors.append("paths.site_maps: expected a list of files")
        elif not (value is None or isinstance(value, str)):
            errors.append(f"paths.{key}: expected a path, found {type(value).__name__}")

    sheets = _mapping(config.get("sheets"), "sheets", SHAREPOINT_SOURCES, errors)
    for key, value in sheets.items():
        if not isinstance(value, (str, int)):
            errors.append(f"sheets.{key}: expected a sheet name or number")

    sources = _mapping(
        config.get("sources"), "sources", ("location", "databricks", "fallback_to_local"), errors
    )
    location = str(sources.get("location", "local")).lower()
    if location not in ("local", "sharepoint"):
        errors.append(f"sources.location: '{location}' is not one of local, sharepoint")
    for flag in ("databricks", "fallback_to_local"):
        if flag in sources and not isinstance(sources[flag], bool):
            errors.append(f"sources.{flag}: expected true or false")

    sharepoint = _mapping(config.get("sharepoint"), "sharepoint", SHAREPOINT_SOURCES, errors)
    for label, entry in sharepoint.items():
        where = f"sharepoint.{label}"
        if label == "site_maps":
            entry = _mapping(entry, where, ("folder", "files", "sheet", "drive_id"), errors)
            if entry and not (isinstance(entry.get("files"), list) and entry.get("files")):
                errors.append(f"{where}.files: name the mapping workbooks in the folder")
            required = ("folder", "drive_id")
        else:
            entry = _mapping(entry, where, ("path", "sheet", "drive_id"), errors)
            required = ("path", "drive_id")
        for key in required:
            if entry and not str(entry.get(key) or "").strip():
                errors.append(f"{where}.{key}: required for a SharePoint source")
    if location == "sharepoint" and not sharepoint:
        errors.append(
            "sharepoint: sources.location is 'sharepoint' but no SharePoint locations are "
            "configured; add a sharepoint: block (see the template from `pvf init`)"
        )

    upload = _mapping(
        config.get("upload"),
        "upload",
        ("enabled", "path", "target", "drive_id", "file_name"),
        errors,
    )
    if "enabled" in upload and not isinstance(upload["enabled"], bool):
        errors.append("upload.enabled: expected true or false")

    _mapping(config.get("ptf"), "ptf", ("write_new_parameters",), errors)
    _mapping(config.get("build"), "build", ("write_parquet",), errors)
    cleaning = _mapping(config.get("cleaning"), "cleaning", CLEANING_KEYS, errors)
    if cleaning.get("disable") is not None and not isinstance(cleaning["disable"], list):
        errors.append("cleaning.disable: expected a list of correction names")
    _mapping(config.get("features"), "features", tuple(f.name for f in fields(Params)), errors)
    _mapping(config.get("lv_coa"), "lv_coa", ("ph_nominal", "osmo_nominal"), errors)
    mapping = _mapping(config.get("mapping"), "mapping", ("exclude",), errors)
    if mapping.get("exclude") is not None and not isinstance(mapping["exclude"], list):
        errors.append("mapping.exclude: expected a list of Raritan names")

    if errors:
        raise ConfigError(f"{source} cannot be used:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Where each input is read from
# ---------------------------------------------------------------------------
def remote_reader(config: dict[str, Any]):
    """The SharePoint reader, but only where the config asked for SharePoint.

    Installing the internal package does not change what a run reads. A config
    that says ``location: sharepoint`` and cannot reach SharePoint stops, unless
    it also allows a fallback — in which case the fallback is logged, and every
    input is then recorded as the local file it really was.
    """
    sources = config.get("sources") or {}
    if str(sources.get("location", "local")).lower() != "sharepoint":
        return None
    try:
        import io_sharepoint
    except ImportError as exc:
        if sources.get("fallback_to_local"):
            log.warn(
                MODULE,
                "io_sharepoint is not installed — every input falls back to its local copy, "
                "as sources.fallback_to_local allows",
            )
            return None
        raise ConfigError(
            "sources.location is 'sharepoint' but io_sharepoint is not installed. "
            "Set sources.location: local, or sources.fallback_to_local: true"
        ) from exc
    return io_sharepoint.load_excel_from_sharepoint


def source(config: dict[str, Any], label: str, reader=None) -> io.Source:
    """One input: its local path, and its SharePoint location when it is read from there."""
    local = str((config.get("paths") or {}).get(label) or "")
    sheet = (config.get("sheets") or {}).get(label, io.DEFAULT_SHEETS.get(label, 0))
    header = io.DEFAULT_HEADERS.get(label, 0)
    entry = (config.get("sharepoint") or {}).get(label)
    if reader is not None and entry:
        remote = {
            "sharepoint_path": entry["path"],
            "sheet_name": entry.get("sheet", sheet),
            "drive_id": entry["drive_id"],
        }
        return io.Source(label, local, sheet, header, remote=remote, loader=reader)
    return io.Source(label, local, sheet, header)


def site_map_sources(config: dict[str, Any], reader=None) -> list[io.Source]:
    """The clinical-site mapping workbooks, one source each."""
    entry = (config.get("sharepoint") or {}).get("site_maps")
    sheet = (config.get("sheets") or {}).get("site_maps", 0)
    if reader is not None and entry:
        folder = str(entry["folder"])
        joiner = "" if folder.endswith("/") else "/"
        return [
            io.Source(
                f"site_map_{number}",
                "",
                sheet,
                remote={
                    "sharepoint_path": f"{folder}{joiner}{name}",
                    "sheet_name": entry.get("sheet", "Sheet1"),
                    "drive_id": entry["drive_id"],
                },
                loader=reader,
            )
            for number, name in enumerate(entry.get("files") or [], start=1)
        ]
    return [
        io.Source(f"site_map_{number}", str(path), sheet)
        for number, path in enumerate((config.get("paths") or {}).get("site_maps") or [], start=1)
    ]


def upload_target(config: dict[str, Any]) -> dict[str, str]:
    """Where the PVF goes when a PRD run uploads it, from config or ``$DATA_LINK``."""
    upload = config.get("upload") or {}
    path = str(upload.get("path") or upload.get("target") or os.getenv("DATA_LINK") or "")
    drive_id = str(upload.get("drive_id") or "")
    if not path or not drive_id:
        raise ConfigError(
            "upload.enabled is set but upload.path and upload.drive_id are not both given "
            "(upload.path may also come from $DATA_LINK)"
        )
    file_name = str(
        upload.get("file_name") or Path((config.get("paths") or {}).get("pvf") or "PVF.xlsx").name
    )
    joiner = "" if path.endswith("/") else "/"
    return {
        "folder": path,
        "drive_id": drive_id,
        "file_name": file_name,
        "location": (f"SharePoint {drive_id}: {path}{joiner}{file_name}"),
    }


def require_paths(config: dict[str, Any], labels: tuple[str, ...]) -> None:
    """Refuse a stage whose inputs the config does not name."""
    paths = config.get("paths") or {}
    reader_labels = set((config.get("sharepoint") or {})) if _is_remote(config) else set()
    missing = [label for label in labels if not paths.get(label) and label not in reader_labels]
    if missing:
        raise ConfigError(
            f"{config.get('__path__', 'config')}: paths.{', paths.'.join(missing)} "
            "must be set for this stage"
        )


def _is_remote(config: dict[str, Any]) -> bool:
    return str((config.get("sources") or {}).get("location", "local")).lower() == "sharepoint"
