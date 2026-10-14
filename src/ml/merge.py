"""Stacking the two sites into the batch table, and checking what came out.

The merge itself is one concatenation. The checks around it are the point: which
catalog parameters never made it into the table, and which columns only one site has.
"""

from dataclasses import dataclass, field

import pandas as pd

from .logger import log
from .names import site

MODULE = "merge"


@dataclass
class MergeReport:
    total_rows: int = 0
    total_columns: int = 0
    site_rows: dict[str, int] = field(default_factory=dict)
    catalog_missing: list[str] = field(default_factory=list)
    site_a_only_cols: list[str] = field(default_factory=list)
    site_b_only_cols: list[str] = field(default_factory=list)
    duplicate_batches: list[str] = field(default_factory=list)


def merge_sites(phf: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the two sites after stamping a Site Merged column."""
    log.step(MODULE, f"Merging {site('a')} and {site('b')} into batch table")

    for name, df in phf.items():
        df["Site Merged"] = name

    result = pd.concat(
        list(phf.values()),
        axis=0,
        ignore_index=True,
        sort=False,
    )
    log.success(
        MODULE,
        f"Merged — {len(result):,} rows × {result.shape[1]} columns  "
        f"({' + '.join(f'{len(v)} {k}' for k, v in phf.items())})",
    )
    return result


def run_sanity_checks(
    result: pd.DataFrame,
    phf: dict[str, pd.DataFrame],
    catalog_cols: list[str],
    catalog_mapping: dict[str, str],
) -> MergeReport:
    """Validate catalog coverage and cross-site column symmetry."""
    log.step(MODULE, "Running sanity checks on merged batch table")

    report = MergeReport(
        total_rows=len(result),
        total_columns=result.shape[1],
        site_rows={name: len(df) for name, df in phf.items()},
    )

    # A batch that appears twice is counted twice by every task that keeps it.
    key = "Patient Lot/Batch #"
    if key in result.columns:
        ids = result[key].astype("string")
        repeated = ids[ids.notna() & ids.duplicated(keep=False)]
        report.duplicate_batches = sorted(set(repeated))
        if report.duplicate_batches:
            log.warn(
                MODULE,
                f"{len(report.duplicate_batches)} batch numbers appear more than once in "
                "the batch table; "
                "a task whose cohort keeps both rows will refuse to run",
                ", ".join(report.duplicate_batches[:10]),
            )
        else:
            log.success(MODULE, "Every batch number appears once")

    # Catalog columns missing from merged result
    report.catalog_missing = [c for c in catalog_cols if c not in result.columns]
    if report.catalog_missing:
        log.warn(
            MODULE,
            f"{len(report.catalog_missing)} catalog parameters absent from merged batch table",
            ", ".join(report.catalog_missing[:20]),
        )
    else:
        log.success(MODULE, "All catalog parameters present in merged batch table")

    # Site A columns not in site B (and not in catalog mapping)
    a, b = site("a"), site("b")
    report.site_a_only_cols = [
        c for c in phf[a].columns if c not in phf[b].columns and c not in catalog_mapping
    ]
    if report.site_a_only_cols:
        log.info(
            MODULE,
            f"{len(report.site_a_only_cols)} {a}-only columns (not in catalog mapping)",
            ", ".join(report.site_a_only_cols[:20]),
        )

    # Site B columns not in site A
    report.site_b_only_cols = [
        c for c in phf[b].columns if c not in phf[a].columns and c not in catalog_mapping
    ]
    if report.site_b_only_cols:
        log.info(
            MODULE,
            f"{len(report.site_b_only_cols)} {b}-only columns (not in catalog mapping)",
            ", ".join(report.site_b_only_cols[:20]),
        )

    log.success(MODULE, "Sanity checks complete")
    return report
