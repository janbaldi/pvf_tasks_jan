"""``config.yaml``: where the sources, the catalog and the batch table are, and the settings.

Code, configuration and data live apart, and the data lives on SharePoint. Every
input — the sources, the catalog, the batch table — is read from the location
``config.yaml`` gives it under ``sharepoint:``, and every output is written there
too: ``ml check-catalog`` writes its report, manifest and new-parameters list into
the catalog's folder, ``ml build`` writes the batch table, its Parquet copy, report
and manifest into the batch table's folder. There is no local option; no
SharePoint address is written into the code.
"""

from __future__ import annotations

import os
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from . import corrections, io, names, share
from .features import Params
from .logger import log

MODULE = "config"

#: The file names the config may have, in the order they are looked for.
CONFIG_NAMES = ("config.yaml", "config.yml")
#: The environment variable that names a config, for runs started anywhere.
CONFIG_ENV = "ML_CONFIG"

#: Inputs that may be read from SharePoint, by the label they have everywhere else.
SHAREPOINT_SOURCES = (
    "vocabulary",
    "catalog",
    "batch_table",
    "phf_site_a",
    "phf_site_b",
    "param_mapping",
    "remfg",
    "lv_coa",
    "raw_materials",
    "investigations",
    "site_maps",
)
TOP_KEYS = (
    "seed",
    "sources",
    "sharepoint",
    "catalog",
    "build",
    "cleaning",
    "features",
    "lv_coa",
    "mapping",
)
#: Settings that no longer exist, and what replaced them.
REMOVED = {
    "paths": "every input is read from SharePoint; give its location under sharepoint:",
    "sheets": "the worksheet is the sheet: of each sharepoint: entry",
    "upload": "ml build always writes the batch table to sharepoint.batch_table",
    "sources.location": "every input is read from SharePoint; there is no local option",
    "sources.fallback_to_local": "every input is read from SharePoint; there is no local option",
    "cleaning.corrections": "the corrections are the Corrections sheet of sharepoint.vocabulary",
}
CLEANING_KEYS = (
    "censored",
    "censored_flags",
    "duration_unit",
    "yes_no_defaults",
    "disable",
    "corrections",  # removed, and refused with what replaced it (REMOVED)
)


class ConfigError(ValueError):
    """One or more problems with config.yaml, each naming its field."""


# ---------------------------------------------------------------------------
# Finding and reading
# ---------------------------------------------------------------------------
def find(explicit: str | Path | None = None, start: str | Path | None = None) -> Path:
    """The config a run uses.

    In order: the path given on the command line, ``$ML_CONFIG``, a
    ``config.yaml`` in ``start`` (a task's folder, say) or any folder above it,
    and the same from the working directory.
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
    raise ConfigError(
        "No config found: there is no config.yaml here or in any folder above, "
        f"${CONFIG_ENV} is not set and --config was not given. "
        "Pass --config <file>, or run from the folder that holds config.yaml"
    )


def load(path: str | Path) -> dict[str, Any]:
    """Read a config, validate it, and resolve its file paths against its own folder."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"{path} does not exist")
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"{path} is not a mapping")

    base = path.parent
    env = load_env(base)
    config["sharepoint"] = expand_env(config.get("sharepoint"), str(path))
    validate(config, str(path))
    config["__path__"] = str(path)
    config["__dir__"] = str(base)
    config["__env__"] = env
    return config


#: ``${NAME}`` in a ``sharepoint:`` value: filled in from ``.env`` or the environment.
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(sharepoint: Any, source: str = "config") -> Any:
    """The ``sharepoint:`` block with every ``${NAME}`` replaced from the environment.

    The SharePoint addresses live in ``.env`` (git-ignored), not in
    ``config.yaml``: an entry says ``path: ${SHAREPOINT_CATALOG}`` and ``.env``
    says where that is. A list of workbooks may be one comma-separated variable.
    A variable that is not set stops the run, naming every one that is missing.
    """
    missing: list[str] = []

    def fill(value: Any, where: str) -> Any:
        if isinstance(value, str):

            def one(match: re.Match) -> str:
                name = match.group(1)
                if name not in os.environ:
                    missing.append(f"{where}: ${{{name}}} is not set")
                return os.environ.get(name, "")

            return _VARIABLE.sub(one, value)
        if isinstance(value, dict):
            return {key: fill(item, f"{where}.{key}") for key, item in value.items()}
        if isinstance(value, list):
            return [fill(item, f"{where}[{i}]") for i, item in enumerate(value)]
        return value

    filled = fill(sharepoint, "sharepoint")
    site_maps = (filled or {}).get("site_maps") if isinstance(filled, dict) else None
    if isinstance(site_maps, dict) and isinstance(site_maps.get("files"), str):
        site_maps["files"] = [f.strip() for f in site_maps["files"].split(",") if f.strip()]
    if missing:
        raise ConfigError(
            f"{source}: SharePoint locations are missing from the environment — add them to "
            "the .env next to config.yaml (see .env.example):\n  - " + "\n  - ".join(missing)
        )
    return filled


def load_env(folder: str | Path) -> str:
    """Load the ``.env`` next to ``config.yaml`` — the SharePoint credentials and drive ids.

    Secrets never go in the code or in git. Variables already set in the
    environment win, so a scheduler or a shell can override the file. Returns
    the file loaded, or "".
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
    _mapping(config, "(root)", TOP_KEYS + ("paths", "sheets", "upload"), errors)

    for key, why in REMOVED.items():
        section, _, name = key.partition(".")
        holder = config if not name else config.get(section)
        if isinstance(holder, dict) and (name or section) in holder:
            errors.append(f"{key}: no longer a setting — {why}")

    allowed = ("databricks", "location", "fallback_to_local")
    _mapping(config.get("sources"), "sources", allowed, errors)
    if not isinstance((config.get("sources") or {}).get("databricks", False), bool):
        errors.append("sources.databricks: expected true or false")

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
                errors.append(f"{where}.{key}: required for a SharePoint location")
        if entry and "sheet" in entry and not isinstance(entry["sheet"], (str, int)):
            errors.append(f"{where}.sheet: expected a sheet name or number")
    if not sharepoint:
        errors.append("sharepoint: every input is read from SharePoint; add a sharepoint: block")

    _mapping(config.get("catalog"), "catalog", ("write_new_parameters",), errors)
    _mapping(config.get("build"), "build", ("write_parquet", "date_column"), errors)
    cleaning = _mapping(config.get("cleaning"), "cleaning", CLEANING_KEYS, errors)
    if cleaning.get("disable") is not None and not isinstance(cleaning["disable"], list):
        errors.append("cleaning.disable: expected a list of correction names")
    _mapping(config.get("features"), "features", tuple(f.name for f in fields(Params)), errors)
    _mapping(config.get("lv_coa"), "lv_coa", ("ph_nominal", "osmo_nominal"), errors)
    mapping = _mapping(config.get("mapping"), "mapping", ("exclude",), errors)
    if mapping.get("exclude") is not None and not isinstance(mapping["exclude"], list):
        errors.append("mapping.exclude: expected a list of vocabulary keys")

    if errors:
        raise ConfigError(f"{source} cannot be used:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Where each input is read from, and each output written to
# ---------------------------------------------------------------------------
def location(config: dict[str, Any], label: str) -> share.Location:
    """Where ``label`` is on SharePoint, with the sheet its data is on."""
    entry = (config.get("sharepoint") or {}).get(label)
    if not entry or label == "site_maps":
        raise ConfigError(
            f"{config.get('__path__', 'config')}: sharepoint.{label} is not set — "
            "every input is read from SharePoint"
        )
    sheet = entry.get("sheet", names.sheet(label, 0))
    return share.Location(str(entry["drive_id"]), str(entry["path"]), sheet)


def source(config: dict[str, Any], label: str) -> io.Source:
    """One input, read from its SharePoint location."""
    return io.Source(label, location(config, label), header=io.DEFAULT_HEADERS.get(label, 0))


def has(config: dict[str, Any], label: str) -> bool:
    return bool((config.get("sharepoint") or {}).get(label))


def site_map_sources(config: dict[str, Any]) -> list[io.Source]:
    """The clinical-site mapping workbooks in their SharePoint folder, one source each."""
    entry = (config.get("sharepoint") or {}).get("site_maps")
    if not entry:
        return []
    folder = str(entry["folder"])
    folder = folder if folder.endswith("/") else folder + "/"
    sheet = entry.get("sheet", names.sheet("site_maps", 0))
    return [
        io.Source(
            f"site_map_{number}", share.Location(str(entry["drive_id"]), f"{folder}{name}", sheet)
        )
        for number, name in enumerate(entry.get("files") or [], start=1)
    ]


def load_vocabulary(config: dict[str, Any]) -> names.Vocabulary:
    """Read the vocabulary from SharePoint and make it the one in use.

    Every stage does this first: the code names no column, value, site or sheet
    itself, so nothing can be read until the vocabulary says what they are.
    """
    require(config, ("vocabulary",))
    where = location(config, "vocabulary").with_sheet(names.NAMES_SHEET)
    log.step(MODULE, "Loading the vocabulary", f"{where.name}  ·  {where.where}")
    frame = share.read_excel(where)
    vocabulary = names.use(names.from_frame(frame, str(where)))
    log.success(MODULE, f"Vocabulary loaded — {len(frame):,} names")
    return vocabulary


#: Where a task run keeps the vocabulary it ran with, beside its other inputs.
RUN_VOCABULARY = "vocabulary.xlsx"


def task_vocabulary(config: dict[str, Any] | None, copy: Path | None) -> names.Vocabulary:
    """A task's vocabulary: the run's own copy for a rerun, else SharePoint's."""
    if copy is None:
        if not config:
            raise ConfigError(
                "The vocabulary is read from SharePoint, and no config.yaml says where: "
                "pass --config, or name it under config: in task.yaml"
            )
        return load_vocabulary(config)
    path = Path(copy).parent / RUN_VOCABULARY
    if not path.is_file():
        raise ConfigError(f"{path} does not exist — the run folder has no vocabulary copy")
    import pandas as pd

    return names.use(names.from_frame(pd.read_excel(path), str(path)))


def load_corrections(config: dict[str, Any]) -> list[corrections.Rule]:
    """The known data corrections: the Corrections sheet of the vocabulary workbook."""
    where = location(config, "vocabulary").with_sheet(names.CORRECTIONS_SHEET)
    try:
        frame = share.read_excel(where)
    except Exception as exc:  # a workbook without the sheet has no corrections to apply
        log.warn(MODULE, f"No corrections read from {where}: {exc}")
        return []
    return corrections.from_frame(frame, str(where))


def catalog_folder(config: dict[str, Any]) -> share.Location:
    """Where ``ml check-catalog`` writes: the catalog's own folder."""
    return location(config, "catalog").folder


def batch_table_folder(config: dict[str, Any]) -> share.Location:
    """Where ``ml build`` writes: the batch table's own folder."""
    return location(config, "batch_table").folder


def require(config: dict[str, Any], labels: tuple[str, ...]) -> None:
    """Refuse a stage whose inputs the config does not locate."""
    missing = [label for label in labels if not has(config, label)]
    if missing:
        raise ConfigError(
            f"{config.get('__path__', 'config')}: sharepoint.{', sharepoint.'.join(missing)} "
            "must be set for this stage"
        )


def table_source(config: dict[str, Any] | None, label: str, copy: Path | None) -> io.Source:
    """A task's batch table or catalog: the run's own copy when it has one, else SharePoint."""
    if copy is not None:
        return io.as_source(copy, label)
    if not config:
        raise ConfigError(
            f"The {label.replace('_', ' ')} is read from SharePoint, and no config.yaml says "
            "where: pass --config, or name it under config: in task.yaml"
        )
    return source(config, label)
