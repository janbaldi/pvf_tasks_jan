"""What the sources record that the catalog does not list.

The catalog is the schema: the batch table may only hold parameters it names, and the
feature registry may only create features it names. So the question this stage
answers is the one that comes first — what is in the source systems that the catalog
has never been told about?

It reads the same site files the build reads, plus the investigations Power Query
workbook, and compares their column names against the catalog's parameters. Names are
matched on a flattened form, because ``Harvest  Lactate (g/L)`` and ``harvest
lactate (g/L)`` are the same parameter and a comparison that says otherwise is
noise.

Nothing here decides anything. It reports, and optionally writes the list of new
parameters out for whoever maintains the catalog.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .logger import log

MODULE = "catalog"

#: Investigations columns arrive as ``Query[Parameter Name]``.
_QUERY = re.compile(r"^Query\[([^\]]*)\]$")


@dataclass
class CatalogReport:
    """Which source names the catalog does not have, and where each came from."""

    catalog_parameters: int = 0
    by_source: dict[str, list[str]] = field(default_factory=dict)
    sources_read: dict[str, int] = field(default_factory=dict)
    sources_absent: list[str] = field(default_factory=list)
    new_parameters: list[str] = field(default_factory=list)
    written_to: str = ""

    @property
    def total(self) -> int:
        return sum(len(names) for names in self.by_source.values())


def normalise(name: object) -> str:
    """A parameter name with its case and spacing flattened, for matching."""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _unmapped_name(column: str, mapping: dict[str, str]) -> str:
    """The site A name a site B column maps to, or the column itself."""
    return mapping.get(normalise(column), column)


def find_new_parameters(
    catalog_cols: list[str],
    sources: dict[str, pd.DataFrame],
    site_mapping: dict[str, str] | None = None,
    mapped: tuple[str, ...] = (),
) -> CatalogReport:
    """Compare every source's columns against the catalog's parameters.

    Parameters
    ----------
    catalog_cols
        The catalog parameter names.
    sources
        ``{label: frame}``, one entry per source system.
    site_mapping
        ``{site B name: site A name}``, as the parameter mapping workbook gives it.
    mapped
        The labels whose columns are matched through ``site_mapping`` first,
        because that site's own name for a parameter is not the name the catalog
        holds. A column with no mapping is compared under its own name.

    Returns
    -------
    CatalogReport
        The new names per source, in the order the sources were given, and the
        deduplicated union of them.
    """
    log.step(MODULE, "Comparing source parameters against the catalog")
    known = {normalise(c) for c in catalog_cols}
    mapping = {normalise(k): v for k, v in (site_mapping or {}).items()}

    report = CatalogReport(catalog_parameters=len(catalog_cols))
    seen: set[str] = set()

    for label, frame in sources.items():
        if frame is None:
            report.sources_absent.append(label)
            log.warn(MODULE, f"{label} not available — it was not compared")
            continue

        report.sources_read[label] = frame.shape[1]
        new: list[str] = []
        for column in frame.columns.astype(str):
            name = column
            if label in mapped:
                name = _unmapped_name(column, mapping)
            elif "Power Query" in label or "Investigations" in label:
                # The query tool wraps the parameter name in its own syntax.
                match = _QUERY.match(column)
                name = match.group(1) if match else column

            if normalise(name) in known:
                continue
            new.append(column)
            if normalise(column) not in seen:
                seen.add(normalise(column))
                report.new_parameters.append(column)

        report.by_source[label] = new
        level = log.warn if new else log.success
        level(
            MODULE,
            f"{label}: {len(new)} of {frame.shape[1]} columns are not catalog parameters",
            ", ".join(new[:20]) or None,
        )

    log.success(
        MODULE,
        f"{report.total} names across the sources are new to the catalog, "
        f"{len(report.new_parameters)} of them distinct",
    )
    return report


def write_new_parameters(report: CatalogReport, path: str | Path) -> Path:
    """Write the distinct new names out, for whoever maintains the catalog.

    One column, one name per row, in the order the sources were read — the shape
    the catalog's own Parameter column wants.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Parameter": report.new_parameters}).to_csv(path, index=False)
    report.written_to = str(path)
    log.success(MODULE, f"{len(report.new_parameters)} new parameter names listed in {path.name}")
    return path
