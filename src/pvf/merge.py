"""Stacking the two sites into the PVF, and checking what came out.

The merge itself is one concatenation. The checks around it are the point: which
PTF parameters never made it into the table, and which columns only one site has.
"""

from dataclasses import dataclass, field

import pandas as pd

from .logger import log

MODULE = "merge"


@dataclass
class MergeReport:
    total_rows: int = 0
    total_columns: int = 0
    site_rows: dict[str, int] = field(default_factory=dict)
    ptf_missing: list[str] = field(default_factory=list)
    ghent_only_cols: list[str] = field(default_factory=list)
    raritan_only_cols: list[str] = field(default_factory=list)
    duplicate_batches: list[str] = field(default_factory=list)


def merge_sites(phf: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate Ghent and Raritan after stamping a Site Merged column."""
    log.step(MODULE, "Merging Ghent and Raritan into PVF")

    for site, df in phf.items():
        df["Site Merged"] = site

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
    ptf_cols: list[str],
    ptf_mapping: dict[str, str],
) -> MergeReport:
    """Validate PTF coverage and cross-site column symmetry."""
    log.step(MODULE, "Running sanity checks on merged PVF")

    report = MergeReport(
        total_rows=len(result),
        total_columns=result.shape[1],
        site_rows={site: len(df) for site, df in phf.items()},
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
                f"{len(report.duplicate_batches)} batch numbers appear more than once in the PVF; "
                "a task whose cohort keeps both rows will refuse to run",
                ", ".join(report.duplicate_batches[:10]),
            )
        else:
            log.success(MODULE, "Every batch number appears once")

    # PTF columns missing from merged result
    report.ptf_missing = [c for c in ptf_cols if c not in result.columns]
    if report.ptf_missing:
        log.warn(
            MODULE,
            f"{len(report.ptf_missing)} PTF parameters absent from merged PVF",
            ", ".join(report.ptf_missing[:20]),
        )
    else:
        log.success(MODULE, "All PTF parameters present in merged PVF")

    # Ghent columns not in Raritan (and not in PTF mapping)
    report.ghent_only_cols = [
        c for c in phf["Ghent"].columns if c not in phf["Raritan"].columns and c not in ptf_mapping
    ]
    if report.ghent_only_cols:
        log.info(
            MODULE,
            f"{len(report.ghent_only_cols)} Ghent-only columns (not in PTF mapping)",
            ", ".join(report.ghent_only_cols[:20]),
        )

    # Raritan columns not in Ghent
    report.raritan_only_cols = [
        c for c in phf["Raritan"].columns if c not in phf["Ghent"].columns and c not in ptf_mapping
    ]
    if report.raritan_only_cols:
        log.info(
            MODULE,
            f"{len(report.raritan_only_cols)} Raritan-only columns (not in PTF mapping)",
            ", ".join(report.raritan_only_cols[:20]),
        )

    log.success(MODULE, "Sanity checks complete")
    return report
