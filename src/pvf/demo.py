"""Made-up sources, shaped like the real ones.

The real inputs live on a corporate share, so there is no way to run this
pipeline — or read its report — without them. This module writes a complete set
of stand-ins: two sites' batch files, the PTF that defines the schema, the
parameter-name mapping between the sites, the vector certificates, the
re-manufacturing supplement, the raw-material export, the investigations Power
Query workbook and the clinical-site mappings.

The numbers are invented and the batches are not real, but the *shapes* are
deliberate. Every path the pipeline can take is represented somewhere in here:
values that need cleaning, a parameter only one site records, a batch whose
vector lot has no certificate, a clinical-site acronym nobody mapped, a feature
whose input is missing, a feature the PTF does not list, a PTF parameter no
source supplies, a parameter for each encoder the dataset stage has, and a group
of parameters correlated strongly enough to be de-correlated. What the reports
show is therefore what the pipeline does.

It writes a complete workspace — config, corrections, task files and sources —
so the demo is also the example of how a workspace is laid out::

    pvf init demo --demo
    cd demo && pvf all
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SEED = 20260917
GHENT_BATCHES = 40
RARITAN_BATCHES = 30

# Deliberate gaps, each one visible in the report. Changing these changes what
# the demo demonstrates, which is the point of naming them here.
BLOCKED_INPUT = "Post Thaw FLOW CD8+ (%)"  # no site records it: two features blocked
RARITAN_ONLY_GAP = "Harvest Glucose (g/L)"  # Ghent only: a feature one site cannot have
UNLISTED_FEATURES = ("Shift", "MOI_per_CD3_pool")  # computable, but not PTF parameters
ORPHAN_PARAMETERS = ("Sterility", "Mycoplasma Method")  # in the PTF, supplied by nothing
UNMAPPED_ACRONYM = "XYZ"  # a clinical site no mapping file covers
LOT_WITHOUT_COA = "LV-909"  # a vector lot with no certificate

# For the dataset stage: one categorical parameter per encoder, an ordinal one,
# and a target a couple of batches are missing.
TARGET = "FP Flow CAR+ (%)"
ONE_HOT_PARAMETER = "Clinical Site"  # few categories
TARGET_ENCODED_PARAMETER = "Shift Team"  # several
HASHED_PARAMETER = "Incubator ID"  # many
ORDINAL_PARAMETER = "Clump Severity"
ORDINAL_ORDER = ("None", "Low", "Medium", "High")
ORDINAL_STRAY = "Severe"  # not one of the PTF's categories
#: What the target is called once the dataset stage has sanitised its name.
TARGET_COLUMN = "fp_flow_car"

# Value types the PTF declares for the columns whose type is not obvious from
# their contents. Everything else is numeric or categorical, decided below.
DATETIME_PARAMETERS = (
    "Thaw Start Time",
    "Date and Time Cells placed in Incubator Day 0",
    "Date and Time Cells Removed From Incubator Day 1",
    "Date and Time Cells placed in Incubator Day 1",
    "Date and Time Cells Removed From Incubator Day 3",
    "Day 3 Date and Time G-Rex A Returned to Incubator",
    "Date and Time Cells Removed From Incubator Day 6",
    "Day 6 Date and Time G-Rex A Returned to Incubator",
    "Date and Time Cells Removed From Incubator Day 8",
    "Day 8 Date and Time G-Rex A Returned to Incubator",
    "Date and Time Cells Removed From Incubator Day 10",
)
DURATION_PARAMETERS = (
    "Day 3 Incubation Time (120min+/-30 min)",
    "Amount of Time G-Rex A in Incubator (120min+/-30 min)",
    "Amount of Time G-Rex B in Incubator (120min+/-30 min)",
)
RATIO_PARAMETERS = ("Post Thaw CD4:CD8 Ratio",)
CATEGORICAL_PARAMETERS = (
    "Patient Lot/Batch #",
    "Patient ID",
    "Vector Lot",
    "Clinical Site",
    "Clinical Site ID",
    "Country",
    "Type",
    "Disposition",
    "OOS Type",
    "Mycoplasma",
    "Non-Conformance Type",
    "Pre-Activation Clumps: Y/N?",
    "day 3 clump pattern",
    "Day 10 Clump Present Bag 1",
    "Day 10 Clump Present Bag 2",
    "Day 10 Clump Present Bag 3",
    "growth profile",
    "CD4 profile",
    "Site Merged",
    "Media Lot",
    "IL-2 Lot",
    "Sterility",
    "Mycoplasma Method",
    "LV CoA Site/Format",
    "Manufacturing and Release Testing Completed? (Y/N)",
    ONE_HOT_PARAMETER,
    TARGET_ENCODED_PARAMETER,
    HASHED_PARAMETER,
)

#: Feature names the PTF lists, so the registry may create them. The two in
#: UNLISTED_FEATURES are left out on purpose.
FEATURE_PARAMETERS = (
    "Log (PT CD4:CD8)",
    "Log (PE CD4:CD8)",
    "PT Total  CD3+ (Viable Cells)",
    "PT CD3+CD4+ (%)",
    "PT Total CD4+ (Viable Cells)",
    "PE CD3+CD4+ (%)",
    "PE CD3+CD8+ (%)",
    "Harvest Lactate Normalized",
    "D6 Lact / E6 cells",
    "D8 Lact / E6 cells",
    "D10 Lact / E6 cells",
    "D6 Glucose / E6 cells",
    "D8 Glucose / E6 cells",
    "D10 Glucose / E6 cells",
    "Lactate_perE6_slope_D6_D10",
    "Lactate_perE6_AUC_D6_D10",
    "Glucose_perE6_slope_D6_D10",
    "Glucose_perE6_AUC_D6_D10",
    "Lactate_to_Glucose_perE6_ratio_D6",
    "Lactate_to_Glucose_perE6_ratio_D8",
    "Lactate_to_Glucose_perE6_ratio_D10",
    "growth profile",
    "CD4 profile",
    "Drop viability D3 vs D0 (%)",
    "Viability_drop_D3_vs_D0_pct",
    "D3-D10 viability change (%)",
    "Viability change during expansion",
    "Viability_change_D3_to_D10_pct",
    "D10 harvest - D10 wash viability change (%)",
    "D3 % cells seeded / target",
    "D3_seeding_error_pct",
    "D3_seeding_within_5pct",
    "Avg_cells_per_bag_D3",
    "Seeded_cells_mean_AB",
    "Seeded_cells_min_AB",
    "Seeded_cells_max_AB",
    "Seeded_cells_std_AB",
    "Seeded_cells_cv_AB",
    "Vector_volume_mean_AB_mL",
    "Vector_volume_min_AB_mL",
    "Vector_volume_max_AB_mL",
    "Vector_volume_std_AB_mL",
    "Vector_volume_cv_AB_mL",
    "MOI_eff_G-Rex_A",
    "log_MOI_eff_G-Rex_A",
    "MOI_eff_G-Rex_B",
    "log_MOI_eff_G-Rex_B",
    "MOI_eff_pool",
    "log_MOI_eff_pool",
    "MOI_eff_quality_adjusted",
    "D3_contact_time_deviation",
    "D3_contact_time_within_window",
    "D3_contact_time_deviation_G-Rex_A",
    "D3_contact_time_within_window_G-Rex_A",
    "D3_contact_time_deviation_G-Rex_B",
    "D3_contact_time_within_window_G-Rex_B",
    "Incubation D0-D1 (min)",
    "Incubation D1-D3",
    "Incubation D3-D6 (min)",
    "Incubation D6-D8 (min)",
    "Incubation D8-D10 (min)",
    "Incubation_total_D3_to_D10_min",
    "Clump_count_reduction_fraction",
    "Clump_size_reduction_fraction",
    "Clump_burden_weighted_pre",
    "Clump_burden_weighted_post_massage",
    "Clump_burden_weighted_post_mixing",
    "Mixing_cycles_x_pre_activation_clump_count",
    "D3_clumps_flag",
    "D10_clump_bag_count",
    "Dose (CAR+ viable cells/kg) all weights",
    "Post formulation dose",
    "D10 Achievable Dose (E6 CAR+ viable T-cells)",
    "D10 Total CAR+ viable T-cells achieved",
    "CAR T produced per cell engaged D3",
    "VCN/cell",
    "Flow accuracy (effective)",
    "Total CS5 cell concentration (E6 cells/mL)",
    "total viable cells used for formulation",
    "total viable cells in formulated suspension",
    "number of viable cells lost during formulation",
    "% cells lost during formulation",
    "total cells lost during LOVO wash (cells)",
    "LVCoA_Impurity_Index_zsum",
    "IL2_IU_total_D6",
    "IL2_IU_total_D8",
    "Day 6 CO2 Saturation (%) dev_from_5.0pct",
    "Day 6 Temperature (°C) dev_from_37.0C",
)


def _cycle(values: list, n: int) -> list:
    """Repeat a short list of values over n batches, so a pattern is visible."""
    return [values[i % len(values)] for i in range(n)]


def _timestamps(n: int, day: int, hour: int) -> list[str]:
    """Batch timestamps as the PHF holds them: text, one batch per day."""
    return [
        f"2026-0{1 + i % 2}-{(i % 20) + 1:02d} {hour:02d}:{(day * 7 + i) % 60:02d}:00"
        for i in range(n)
    ]


def _site_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """The measurements both sites record, under the Ghent names.

    Values are built from a few latent quantities rather than drawn one column at
    a time: batch size drives every cell count, metabolic rate drives lactate and
    glucose, and transduction drives both CAR readings. That is what real
    manufacturing data looks like, and it is what gives the dataset stage
    correlated groups to find and near-duplicate pairs to collapse.
    """
    size = rng.lognormal(0.0, 0.18, n)  # how big the batch is
    metabolism = rng.lognormal(0.0, 0.22, n)  # how fast the cells burn sugar
    transduction = rng.beta(2.5, 3.0, n)  # how well the vector took
    noise = lambda spread: rng.normal(1.0, spread, n)  # noqa: E731 — one line, used ten times

    vessels = np.where(np.arange(n) % 3 == 0, 1, 2)
    seeded = (size * 5.4e5).round(0)
    vsvg = (size * vessels * 5.2e7).round(0)
    prewash = (size * 3.1e9 * noise(0.05)).round(0)
    postwash = (prewash * 0.88 * noise(0.03)).round(0)
    formulation_volume = (size * 58 * noise(0.04)).round(1)
    car_expression = (18 + 44 * transduction * noise(0.05)).round(1)

    return pd.DataFrame(
        {
            "Day 3 Number of G-Rex to seed": vessels,
            "Total Viable Cells for Expansion after VSVg Sampling (cells)": vsvg,
            # 5% more cells were available than were seeded, every time: the pair
            # is a near-duplicate and one of them is dropped.
            "Day 3 Total Viable Cells Available for Seeding G-Rex (cells)": (vsvg * 1.05).round(0),
            "Day 3 Actual Viable Cells Seeded G-Rex A": seeded,
            "Day 3 Actual Viable Cells Seeded G-Rex B": np.where(
                vessels == 2, (seeded * 0.97).round(0), np.nan
            ),
            "# of Culture Bags": np.where(vessels == 2, 4, 2),
            "Volume Vector Added to G-Rex A (mL)": (2.1 * noise(0.06)).round(2),
            "Volume Vector Added to G-Rex B (mL)": np.where(
                vessels == 2, (2.1 * noise(0.06)).round(2), np.nan
            ),
            # Metabolites: one rate, sampled on three days. Day 8 vessel B is
            # absent from both sites on purpose — that feature is blocked while
            # the D6-D10 slope still has two of its three days.
            "Day 6 Lactate G-Rex A (g/L)": (metabolism * 1.15 * noise(0.07)).round(3),
            "Day 6 Lactate G-Rex B (g/L)": np.where(
                vessels == 2, (metabolism * 1.15 * noise(0.07)).round(3), np.nan
            ),
            "Day 8 Lactate G-Rex A (g/L)": (metabolism * 1.9 * noise(0.07)).round(3),
            "Harvest Lactate (g/L)": (metabolism * 2.6 * noise(0.07)).round(3),
            "Day 6 Glucose G-Rex A (g/L)": (3.4 - metabolism * 0.7 * noise(0.06)).round(3),
            "Day 6 Glucose G-Rex B (g/L)": np.where(
                vessels == 2, (3.4 - metabolism * 0.7 * noise(0.06)).round(3), np.nan
            ),
            "Day 8 Glucose G-Rex A (g/L)": (2.6 - metabolism * 0.6 * noise(0.06)).round(3),
            "Day 8 Glucose G-Rex B (g/L)": np.where(
                vessels == 2, (2.6 - metabolism * 0.6 * noise(0.06)).round(3), np.nan
            ),
            "Harvest Glucose (g/L)": (1.9 - metabolism * 0.5 * noise(0.06)).round(3),
            "cPDL (D3 to D10)": (4.6 * metabolism**0.6 * noise(0.06)).round(2),
            # Flow
            "Post Thaw FLOW CD3+ (%)": rng.uniform(55, 88, n).round(1),
            "Post Thaw FLOW CD4+ (%)": rng.uniform(28, 72, n).round(1),
            "Post Wash POS FLOW CD3+ (%)": rng.uniform(80, 97, n).round(1),
            "Post Wash POS FLOW CD4+ (%)": rng.uniform(30, 70, n).round(1),
            "Post Wash POS FLOW CD8+ (%)": rng.uniform(25, 60, n).round(1),
            "Post Thaw CD4:CD8 Ratio": [f"1:{v:.2f}" for v in rng.uniform(0.6, 2.4, n)],
            "Post Thaw Viable Cell Count after sampling(cells)": (size * 4.2e8).round(0),
            # Viability
            "Post Wash POS Viability (%)": rng.uniform(88, 98, n).round(1),
            "Day 3 Average of Pooled Viability (%)": rng.uniform(85, 96, n).round(1),
            "Harvest Pre-Wash Cell Viability (%)": rng.uniform(80, 95, n).round(1),
            "Harvest Post Wash Viability Average (%)": rng.uniform(82, 96, n).round(1),
            # Harvest and formulation: every count follows the batch size, which
            # is what puts them in one correlated cluster.
            "Harvest Pre-Wash Viable Cell Count after Sampling (cells)": prewash,
            "Harvest Post Wash Viable Cell Count (cells)": postwash,
            "Harvest Post Wash Viable Cell Concentration Average (cells/mL)": (
                postwash / formulation_volume
            ).round(0),
            "Volume of Cells Used for Formulation (mL)": formulation_volume,
            "Volume CS5 Used for Formulation (mL)": (formulation_volume * 1.3).round(1),
            "Volume per Bag (mL)": rng.choice([20.0, 30.0], n),
            "Final Formulation CS5 Viable Cell Concentration Average (cells/mL)": (
                postwash / (formulation_volume * 1.3) * noise(0.03)
            ).round(0),
            "Final Formulation CS5 Viability Average (%)": rng.uniform(84, 96, n).round(1),
            # Product and dose. The target and the harvest CAR reading share a
            # latent, as they do in a real process.
            "Harvest Pre-Wash Flow CAR+ Expression (%)": car_expression,
            TARGET: (car_expression * noise(0.08)).round(1),
            "Provirus Vector Copy Number (copies/transduced cell)": (
                1.1 + 2.0 * transduction
            ).round(2),
            "Subject Weight (kg)": rng.uniform(52, 104, n).round(1),
            "Dose (CAR+ viable cells/kg)": (size * 2.4e6).round(0),
            "Dose: Number of CAR+ Viable T-Cells (cells)": (size * 2.6e8).round(0),
            # Timing
            "Thaw Start Time": [f"{8 + i % 6:02d}:{(i * 13) % 60:02d}" for i in range(n)],
            "Date and Time Cells placed in Incubator Day 0": _timestamps(n, 0, 9),
            "Date and Time Cells Removed From Incubator Day 1": _timestamps(n, 1, 10),
            "Date and Time Cells placed in Incubator Day 1": _timestamps(n, 1, 11),
            "Date and Time Cells Removed From Incubator Day 3": _timestamps(n, 3, 9),
            "Day 3 Date and Time G-Rex A Returned to Incubator": _timestamps(n, 3, 12),
            "Date and Time Cells Removed From Incubator Day 6": _timestamps(n, 6, 9),
            "Day 6 Date and Time G-Rex A Returned to Incubator": _timestamps(n, 6, 13),
            "Date and Time Cells Removed From Incubator Day 8": _timestamps(n, 8, 9),
            "Day 8 Date and Time G-Rex A Returned to Incubator": _timestamps(n, 8, 14),
            "Date and Time Cells Removed From Incubator Day 10": _timestamps(n, 10, 8),
            "Day 3 Incubation Time (120min+/-30 min)": [
                f"0{1 + i % 2}:{(i * 11) % 60:02d}" for i in range(n)
            ],
            "Amount of Time G-Rex A in Incubator (120min+/-30 min)": [
                f"0{1 + i % 2}:{(i * 7) % 60:02d}" for i in range(n)
            ],
            "Amount of Time G-Rex B in Incubator (120min+/-30 min)": [
                f"0{1 + i % 2}:{(i * 5) % 60:02d}" for i in range(n)
            ],
            # Clumps
            "Pre-Activation Clumps: # of Clumps before massage": rng.integers(2, 12, n),
            "Pre-Activation Clumps: Size of Clumps before massage": rng.integers(1, 5, n),
            "Post-Massaging: # of Clumps": rng.integers(0, 6, n),
            "Post-Massaging: Size of Clumps": rng.integers(1, 4, n),
            "Post-Mixing Clumps: # of Clumps": rng.integers(0, 4, n).astype(str),
            "Post-Mixing Clumps: Size of Clumps": rng.integers(1, 3, n),
            "Day 1 Number of Mixing Cycles Post Activation": rng.integers(2, 7, n),
            "day 3 clump pattern": rng.choice(["None", "Few small", "Many small"], n),
            "Day 10 Clump Present Bag 1": rng.choice(["Yes", "No"], n),
            "Day 10 Clump Present Bag 2": rng.choice(["Yes", "No"], n),
            "Day 10 Clump Present Bag 3": rng.choice(["Yes", "No"], n),
            ORDINAL_PARAMETER: _cycle([*ORDINAL_ORDER, ORDINAL_STRAY], n),
            # Feeds and environment
            "Day 6 Activity of IL-2 (IU/mg)": rng.uniform(1.4e7, 2.2e7, n).round(0),
            "Day 6 IL-2 Protein Content (µg/vial)": rng.uniform(180, 240, n).round(1),
            "Day 8 Activity of IL-2 (IU/mg)": rng.uniform(1.4e7, 2.2e7, n).round(0),
            "Day 8 IL-2 Protein Content (µg/vial)": rng.uniform(180, 240, n).round(1),
            "Day 6 CO2 Saturation (%)": rng.uniform(4.6, 5.4, n).round(2),
            "Day 6 Temperature (°C)": rng.uniform(36.4, 37.6, n).round(2),
            # Who and where: one categorical per encoder the dataset stage has.
            # Drawn rather than cycled: a strict cycle of six teams over three
            # vector lots gives each lot its own two teams, and grouped folds
            # then have nothing to learn a team's mean from.
            TARGET_ENCODED_PARAMETER: rng.choice(list("ABCDEF"), n),
            HASHED_PARAMETER: _cycle([f"INC-{i:02d}" for i in range(10)], n),
            # Release
            "Endotoxin (EU/mL)": [
                "<0.5 EU/mL" if i % 4 == 0 else f"{v:.2f} EU/mL"
                for i, v in enumerate(rng.uniform(0.1, 1.2, n))
            ],
            "Recovery (%)": rng.uniform(70, 99, n).round(1),
            "Manufacturing and Release Testing Completed? (Y/N)": _cycle(["Y"] * 9 + ["N"], n),
            "Mycoplasma": "mycoplasma not detected",
            "OOS Type": _cycle(["VIability", "Low Dose", "CAR", ""], n),
            "Non-Conformance Type": _cycle(
                ["", "", "", "deviation", "withdrawn", "Withdrawal", "Termination"], n
            ),
        }
    )


# Raritan records some parameters under its own names. The mapping file below is
# what teaches the pipeline they are the same thing.
RARITAN_NAMES = {
    "Patient Lot/Batch #": "Batch Number",
    "Vector Lot": "Vector Lot Number",
    "Subject Weight (kg)": "Weight (kg)",
    "Harvest Pre-Wash Flow CAR+ Expression (%)": "CAR+ Expression Pre-Wash (%)",
}


def ghent_phf(rng: np.random.Generator, n: int = GHENT_BATCHES) -> pd.DataFrame:
    """Ghent batches, with the values that give cleaning something to do."""
    df = _site_frame(n, rng)
    df.insert(0, "Patient Lot/Batch #", [f"GH{i:03d}" for i in range(n)])
    df.insert(1, "Patient ID", [f"EU-{1200 + i % 3:04d}-{i:03d}" for i in range(n)])
    df.insert(2, "Vector Lot", _cycle(["LV-101", "LV-102", LOT_WITHOUT_COA], n))
    df["Clinical Site"] = _cycle(["UZ Gent", "UZ Leuven\xa0", " AZ Sint-Jan", ""], n)
    df["Country"] = _cycle(["Belgium", "Netherlands"], n)
    df["Type"] = _cycle(["Commercial"] * 9 + ["Clinical"], n)
    df["Disposition"] = _cycle(["Released", "Rejected"], n)
    df["Pre-Activation Clumps: Y/N?"] = _cycle(["Yes", "No", None, "Yes"], n)
    df["Target Final Seeding Density per Area (VC/cm^2) G-Rex A"] = rng.uniform(
        4e11, 6e11, n
    ).round(0)
    df["Local Comment"] = "site note"  # not a PTF parameter: a Ghent-only column

    # One batch the integrity corrections know by name.
    df.loc[0, "Patient Lot/Batch #"] = "QCGS03V"
    # A number recorded as text, and a percentage that cannot be one. The column
    # becomes text here exactly as the source file has it — coercion is the
    # pipeline's job, not the fixture's.
    df["Recovery (%)"] = df["Recovery (%)"].astype(object)
    df.loc[1, "Recovery (%)"] = "not measured"
    df.loc[2, "Recovery (%)"] = 105.0
    # A timestamp and a duration nobody filled in properly.
    df.loc[3, "Date and Time Cells Removed From Incubator Day 8"] = "not recorded"
    df.loc[4, "Amount of Time G-Rex B in Incubator (120min+/-30 min)"] = "not started"
    # A duration that ran backwards, and two batches whose target was never read.
    df.loc[2, "Day 3 Incubation Time (120min+/-30 min)"] = "00:00"
    df.loc[[6, 7], TARGET] = np.nan
    return df


def raritan_phf(rng: np.random.Generator, n: int = RARITAN_BATCHES) -> pd.DataFrame:
    """Raritan batches, under Raritan's own column names."""
    df = _site_frame(n, rng)
    df.insert(0, "Batch Number", [f"RA{i:03d}" for i in range(n)])
    df.insert(1, "Vector Lot Number", _cycle(["LV-102", "LV-103", "LV-101"], n))
    df = df.rename(
        columns={
            "Subject Weight (kg)": "Weight (kg)",
            "Harvest Pre-Wash Flow CAR+ Expression (%)": "CAR+ Expression Pre-Wash (%)",
        }
    )
    # Raritan does not record harvest glucose, so one feature cannot be computed
    # there and is reported as blocked for that site alone.
    df = df.drop(columns=[RARITAN_ONLY_GAP], errors="ignore")
    df["Clinical Site"] = _cycle(["MSKCC", "107306", UNMAPPED_ACRONYM], n)
    df["Country"] = "United States"
    # A second column that maps onto Country as well, spelled differently so a
    # test can tell which values ended up under which name.
    df["Country Name"] = "USA"
    df["Local Site Code"] = "RAR-01"  # mapped to nothing, so the mapping drops it
    df["Type"] = "Clinical"  # cleaning forces this to Commercial
    df["Disposition"] = _cycle(["Released", "Discarded"], n)  # a level Ghent never uses
    df["Pre-Activation Clumps: Y/N?"] = _cycle(["Y", "N", None], n)
    df["Total Viable Cells/bag"] = rng.uniform(3e12, 7e12, n).round(0)
    df[ORDINAL_PARAMETER] = _cycle(list(ORDINAL_ORDER), n)
    df["CMD Comment"] = "cmd note"  # not a PTF parameter: a Raritan-only column

    # The clump count that is really a date Excel mangled.
    df.loc[0, "Post-Mixing Clumps: # of Clumps"] = "45691"
    return df


def parameter_mapping() -> pd.DataFrame:
    """Raritan parameter names against their Ghent equivalents."""
    rows = [(raritan, ghent) for ghent, raritan in RARITAN_NAMES.items()]
    rows += [
        ("Country Name", "Country"),  # a second column claiming a taken name
        ("Local Site Code", None),  # no Ghent equivalent: dropped
        ("Non-Conformance Type Calc.", "Non-Conformance Type"),  # known bad entry
    ]
    return pd.DataFrame(
        rows, columns=["Parameter Name Raritan (CMD)", "Parameter Name Ghent (PHF)"]
    )


#: The certificate columns, in the order the workbook holds them. The first is
#: the lot number, which the loader renames to Vector Lot.
COA_COLUMNS = (
    "LV Batch",
    "CoA Available",
    "Site/Format",
    "LV Titer",
    "LV Titer (unadjusted)",
    "Physical Titer p24 ELISA",
    "PI Ratio",
    "CAR% - Donor 1",
    "CAR% - Donor 2",
    "IFNg - Donor 1",
    "IFNg - Donor 2",
    "pH",
    "Osmolality",
    "Volume",
    "DNA size",
    "HCP",
    "pDNA",
    "cDNA",
    "E1a cDNA",
    "E1b cDNA",
    "MFG Date",
)


def lv_coa() -> pd.DataFrame:
    """Vector certificates, written the way the source workbook writes them.

    Numbers arrive as European-locale text, percentages as percent strings, and
    an impurity below the limit of quantitation as ``<0,5`` — all three are
    things the loader has to undo. ``LV-104`` has no certificate yet and is
    filtered out; ``LV-909`` never appears, so the batches built from it have no
    certificate at all.
    """
    rows = [
        [
            "LV-101",
            "Yes",
            "Ghent/250mL",
            "3,40E+07",
            "3,10E+07",
            "1,20E+11",
            "3500",
            "42,0%",
            "38,5%",
            "1250",
            "1190",
            "7,3",
            "305",
            "48,0",
            "1,9",
            "<0,5",
            "2,4",
            "1,1",
            "0,8",
            "0,4",
            "2025-11-02",
        ],
        [
            "LV-102",
            "Yes",
            "Raritan/500mL",
            "2,90E+07",
            "2,70E+07",
            "1,05E+11",
            "3200",
            "39,5%",
            "41,0%",
            "1180",
            "1240",
            "7,1",
            "298",
            "52,0",
            "2,1",
            "0,9",
            "2,8",
            "1,4",
            "0,7",
            "0,6",
            "2025-11-18",
        ],
        [
            "LV-103",
            "Yes",
            "Raritan/500mL",
            "3,10E+07",
            "2,95E+07",
            "1,15E+11",
            "3350",
            "44,5%",
            "40,0%",
            "1310",
            "1275",
            "7,4",
            "312",
            "50,0",
            "2,0",
            "1,4",
            "3,1",
            "1,2",
            "0,9",
            "0,5",
            "2025-12-05",
        ],
        [
            "LV-104",
            "No",
            "Ghent/250mL",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "2026-01-09",
        ],
    ]
    df = pd.DataFrame(rows, columns=list(COA_COLUMNS))
    df.insert(2, 1, np.nan)  # a section marker column, which the loader drops
    return df


def remanufacturing(n: int = RARITAN_BATCHES) -> pd.DataFrame:
    """Prior lines of therapy, keyed by the Raritan batch number."""
    return pd.DataFrame(
        {
            "Atlas Batch Number": [f"RA{i:03d}" for i in range(n)] + [None],
            "Number of Prior Lines of Therapy (LGN)": list(range(1, n + 1)) + [None],
        }
    )


def raw_materials(ghent: int = GHENT_BATCHES, raritan: int = RARITAN_BATCHES) -> pd.DataFrame:
    """The consumables export, with the column names its query tool produces."""
    batches = [f"GH{i:03d}" for i in range(ghent)] + [f"RA{i:03d}" for i in range(raritan)]
    batches[0] = "QCGS03V"
    return pd.DataFrame(
        {
            "Batch": batches,
            "Query[Media Lot]": [f"MED-{100 + i % 4}" for i in range(len(batches))],
            "Query[IL-2 Lot]": [f"IL2-{200 + i % 3}" for i in range(len(batches))],
        }
    )


def investigations(n: int = 12) -> pd.DataFrame:
    """The investigations team's Power Query workbook.

    Only the PTF stage reads it, and only for its column names. The query tool
    wraps each parameter in ``Query[...]``; some of them are PTF parameters and
    some are the team's own working columns, which is exactly the distinction
    that stage is there to draw.
    """
    return pd.DataFrame(
        {
            "Query[Patient Lot/Batch #]": [f"GH{i:03d}" for i in range(n)],
            "Query[Harvest Lactate (g/L)]": np.linspace(1.5, 3.0, n).round(2),
            "Query[Investigation ID]": [f"INV-{i:03d}" for i in range(n)],
            "Query[Root Cause Category]": _cycle(["Operator", "Equipment", "Material"], n),
            "Days to Close": _cycle([7, 14, 30], n),
        }
    )


def site_mappings() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two acronym files, as they arrive from two different meetings."""
    first = pd.DataFrame(
        {"Acronym": ["MSKCC", "MDACC"], "Institution": ["Memorial Sloan Kettering", "MD Anderson"]}
    )
    second = pd.DataFrame(
        {
            "Acronym": ["107306", "MSKCC"],
            "Institution": ["Rambam Health Care Campus", "Memorial Sloan Kettering Cancer Center"],
        }
    )
    return first, second


def _value_type(parameter: str) -> str:
    # An ordinal parameter's value type is its categories, in order. That is how
    # the PTF says "this is a rank, not a set of labels".
    if parameter == ORDINAL_PARAMETER:
        return "[" + ", ".join(ORDINAL_ORDER) + "]"
    if parameter in DATETIME_PARAMETERS:
        return "datetime"
    if parameter in DURATION_PARAMETERS:
        return "duration"
    if parameter in RATIO_PARAMETERS:
        return "ratio"
    if parameter in CATEGORICAL_PARAMETERS or "profile" in parameter:
        return "categorical"
    return "numeric"


#: When in the process each made-up parameter exists, by its name. Earliest match
#: wins. Features are left without a stage on purpose: a task derives theirs from
#: their inputs. Two parameters stay unstaged, so the report shows what that does.
STAGE_RULES = (
    ("Disposition", "Post-release"),
    ("OOS Type", "Post-release"),
    ("Non-Conformance", "Post-release"),
    ("Endotoxin", "Release"),
    ("Mycoplasma", "Release"),
    ("Sterility", "Release"),
    ("Recovery", "Release"),
    ("Manufacturing and Release Testing", "Release"),
    (TARGET, "Final product"),
    ("Provirus Vector Copy Number", "Final product"),
    ("Post Thaw", "Post-thaw"),
    ("Dose", "Formulation"),
    ("Formulation", "Formulation"),
    ("Volume per Bag", "Formulation"),
    ("Post Wash", "Wash"),
    ("Harvest", "Harvest"),
    ("Day 10", "D10"),
    ("Day 10 Clump", "D10"),
    ("cPDL", "D10"),
    ("Day 8", "D8"),
    ("Day 6", "D6"),
    ("Day 3", "D3"),
    ("day 3", "D3"),
    ("# of Culture Bags", "D3"),
    ("Volume Vector Added", "D3"),
    ("Total Viable Cells for Expansion", "D3"),
    ("Total Viable Cells/bag", "D3"),
    ("Target Final Seeding Density", "D3"),
    ("Incubator", "D1"),
    ("Day 1", "D1"),
    ("Clump", "D1"),
    ("Post-Mixing", "D1"),
    ("Post-Massaging", "D1"),
    ("Pre-Activation", "D0"),
    ("Thaw Start", "D0"),
    ("Day 0", "D0"),
    ("LV CoA", "D0"),
    ("Media Lot", "D0"),
    ("IL-2 Lot", "D0"),
    ("Shift Team", "D0"),
    ("Clinical Site", "D0"),
    ("Country", "D0"),
    ("Subject Weight", "D0"),
    ("Number of Prior Lines", "D0"),
    ("Type", "D0"),
    ("Patient", "D0"),
    ("Vector Lot", "D0"),
    ("Site Merged", "D0"),
)
#: Parameters the made-up PTF leaves without a stage.
UNSTAGED = ("Day 6 Temperature (°C)",)


def _stage(parameter: str) -> str:
    if parameter in FEATURE_PARAMETERS or parameter in UNSTAGED:
        return ""
    for fragment, stage in STAGE_RULES:
        if fragment in parameter:
            return stage
    return ""


def ptf(ghent: pd.DataFrame, raritan: pd.DataFrame) -> pd.DataFrame:
    """The PTF: every parameter the PVF may hold, and the type it holds.

    Built from what the sources carry, minus the few things left out on purpose —
    the site comment columns, which the PTF has never accepted, and the two
    features in ``UNLISTED_FEATURES`` — plus parameters nothing supplies, so the
    report has a PTF gap to show.
    """
    to_ghent = {raritan_name: ghent for ghent, raritan_name in RARITAN_NAMES.items()}
    to_ghent["Country Name"] = "Country"
    mapped = dict.fromkeys(to_ghent.get(c, c) for c in raritan.columns)
    certificates = [
        f"LV CoA {column}"
        for column in COA_COLUMNS[2:]
        if column not in ("CoA Available", "LV Titer (unadjusted)", "MFG Date")
    ]

    parameters = dict.fromkeys(
        [
            *ghent.columns,
            *mapped,
            "Vector Lot",
            "Number of Prior Lines of Therapy",
            "Clinical Site ID",
            "Media Lot",
            "IL-2 Lot",
            *certificates,
            *FEATURE_PARAMETERS,
            *ORPHAN_PARAMETERS,
        ]
    )
    for absent in (
        "Local Comment",
        "CMD Comment",
        "Local Site Code",
        "Country Name",
        *UNLISTED_FEATURES,
    ):
        parameters.pop(absent, None)

    return pd.DataFrame(
        {
            "Parameter": list(parameters),
            "Value Type": [_value_type(p) for p in parameters],
            "Available At": [_stage(p) for p in parameters],
        }
    )


def write_sources(directory: Path, seed: int = SEED) -> dict[str, object]:
    """Write one complete set of sources into a folder, and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    ghent = ghent_phf(rng)
    raritan = raritan_phf(rng)
    first_map, second_map = site_mappings()

    paths = {
        "ptf": directory / "PTF.xlsx",
        "phf_ghent": directory / "phf_ghent.xlsx",
        "phf_raritan": directory / "phf_raritan.xlsx",
        "param_mapping": directory / "parameter_requirements.xlsx",
        "remfg": directory / "remfg.xlsx",
        "lv_coa": directory / "lv_coa.xlsx",
        "raw_materials": directory / "raw_materials.csv",
        "investigations": directory / "investigations_power_query.xlsx",
        "site_maps": [directory / "site_map_hannelore.xlsx", directory / "site_map_meeting.xlsx"],
    }

    ptf(ghent, raritan).to_excel(paths["ptf"], index=False)
    ghent.to_excel(paths["phf_ghent"], sheet_name="BR Data", index=False)
    raritan.to_excel(paths["phf_raritan"], sheet_name="BR Data", index=False)
    parameter_mapping().to_excel(paths["param_mapping"], sheet_name="Overall", index=False)
    remanufacturing().to_excel(paths["remfg"], sheet_name="Treatment Line Data", index=False)
    # Two rows of workbook furniture above the header, as the real file has.
    lv_coa().to_excel(paths["lv_coa"], sheet_name="CoA extracts", index=False, startrow=2)
    raw_materials().to_csv(paths["raw_materials"], index=False)
    investigations().to_excel(
        paths["investigations"], sheet_name="Master Query ALL COM", index=False
    )
    first_map.to_excel(paths["site_maps"][0], index=False)
    second_map.to_excel(paths["site_maps"][1], index=False)

    return {
        key: [str(p) for p in value] if isinstance(value, list) else str(value)
        for key, value in paths.items()
    }


def write_config(directory: Path, seed: int = SEED) -> Path:
    """Write a complete demo workspace into ``directory``. Returns its ``pvf.yaml``.

    Sources go to ``data/raw``, the PVF and its manifests to ``data/processed``,
    reports and task runs to ``outputs``, task files to ``tasks``. Every path in
    the config is relative to the config, which is what lets a workspace move.

    The thresholds here are the ones that suit a cohort this small: with forty
    batches, a parameter with a dozen categories is already high-cardinality, and
    a feature needs fewer values before it is worth keeping.
    """
    directory = Path(directory)
    raw = directory / "data" / "raw"
    sources = write_sources(raw, seed=seed)
    for folder in ("data/processed", "outputs"):
        (directory / folder).mkdir(parents=True, exist_ok=True)
    paths = {
        key: [f"data/raw/{Path(p).name}" for p in value]
        if isinstance(value, list)
        else f"data/raw/{Path(value).name}"
        for key, value in sources.items()
    }
    paths.update(
        {
            "pvf": "data/processed/PVF.xlsx",
            "new_parameters": "data/processed/new_parameters.csv",
            "reports": "outputs/reports",
            "tasks": "outputs/tasks",
            "task_files": "tasks",
        }
    )
    shutil.copyfile(_template("corrections.yaml"), directory / "corrections.yaml")

    config = {
        "seed": seed,
        "paths": paths,
        # Which sources a run reads is a setting, not a consequence of what
        # happens to be installed. Everything here is local and nothing uploads.
        "sources": {"location": "local", "databricks": False, "fallback_to_local": False},
        "upload": {"enabled": False},
        "ptf": {"write_new_parameters": True},
        "build": {"write_parquet": True},
        "cleaning": {
            "corrections": "corrections.yaml",
            "censored": "limit",
            "censored_flags": False,
            "duration_unit": "minutes",
            "raritan_type_commercial": True,
            "yes_no_defaults": {"Pre-Activation Clumps: Y/N?": "No"},
        },
        "mapping": {"exclude": ["Non-Conformance Type Calc."]},
        "features": {
            "co2_target": 5.0,
            "temp_target": 37.0,
            "contact_target": 120.0,
            "contact_tolerance": 30.0,
            "seeding_tolerance": 5.0,
            "cpdl_low": 4.0,
            "cpdl_high": 5.0,
            "lactate_low": 12.5,
            "lactate_high": 30.0,
            "cd4_low": 40.0,
            "cd4_high": 60.0,
        },
        # Spec midpoints for the made-up product, so the two-sided deviation
        # features are built rather than skipped.
        "lv_coa": {"ph_nominal": 7.2, "osmo_nominal": 300.0},
    }

    config_path = directory / "pvf.yaml"
    config_path.write_text(
        "# A demo workspace written by `pvf init --demo`. Every source under data/raw is\n"
        "# made up. Every path below is relative to this file.\n"
        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_tasks(directory)
    return config_path


def _template(name: str) -> Path:
    from importlib import resources

    return Path(str(resources.files("pvf") / "templates" / name))


#: The encoder and quality settings both example tasks share, sized for 40 batches.
_SMALL_COHORT = {
    "encoding": {
        "onehot_max_categories": 5,
        "target_max_categories": 6,
        "target_encoding_smoothing": 5.0,
        "target_encoding_folds": 5,
        "hashing_max_buckets": 16,
    },
    # Fitted on the training rows, so the minimum has to be smaller than the
    # training side of a split over a cohort this size.
    "quality": {"min_non_missing": 8, "duplicate_r2_threshold": 0.99, "duplicate_min_overlap": 8},
}

#: A parameter the made-up PTF stages at harvest, excluded by hand on top of the
#: mechanical check: a task can always say more than the PTF does.
DECLARED_UNAVAILABLE = [
    {"column": "Harvest Post Wash Viable Cell Count (cells)", "reason": "counted after the wash"},
]

#: The demo's task files, relative to the workspace.
TASK_FILES = ("tasks/task_car_expression.yaml", "tasks/task_release_outcome.yaml")


def write_tasks(directory: Path) -> list[Path]:
    """Two task files over the same PVF, differing only in what they say.

    The point of the pair is that nothing else changes: same package, same
    commands, same code. Cohort, target, roles, clustering action and destination
    all move in YAML.
    """
    directory = Path(directory)
    car_expression = {
        # Inputs and output folder come from the workspace config one level up.
        "workspace": "../pvf.yaml",
        "task": {
            "name": "ghent_car_expression",
            "question": "Which manufacturing parameters go with final product CAR+ "
            "expression at Ghent?",
            "purpose": "predictive",
            "seed": SEED,
        },
        "cohort": {
            "sites": ["Ghent"],
            "filters": [
                {
                    "column": "Type",
                    "op": "equals",
                    "value": "Commercial",
                    "reason": "manufactured commercially",
                },
                {
                    "column": "Manufacturing and Release Testing Completed? (Y/N)",
                    "op": "equals",
                    "value": "Y",
                    "reason": "manufacturing and testing finished",
                },
                {
                    "column": "Non-Conformance Type",
                    "op": "not_in",
                    "value": ["termination", "withdrawal"],
                    "normalise": True,
                    "missing": "include",
                    "reason": "not terminated or withdrawn, however it is spelled",
                },
            ],
        },
        "target": {"column": TARGET, "type": "numeric"},
        "columns": {
            "id": "Patient Lot/Batch #",
            "group": "Vector Lot",
            "exclude": [
                {
                    "column": "Subject Weight (kg)",
                    "reason": "dose per kilogram already carries it",
                }
            ],
        },
        "availability": {
            # The PTF says from which stage every parameter exists, so naming the
            # cutoff is enough: everything later, or unstaged, is left out.
            "cutoff": "Harvest",
            "unavailable": DECLARED_UNAVAILABLE,
        },
        **_SMALL_COHORT,
        "clustering": {
            "action": "report_only",
            "method": "spearman",
            "threshold": 0.7,
            "min_overlap": 8,
        },
        "split": {"strategy": "grouped", "validation_fraction": 0.3, "folds": 3},
        "report": {"preview_rows": 15},
    }

    release_outcome = {
        "workspace": "../pvf.yaml",
        "task": {
            "name": "raritan_release",
            "question": "What does a released Raritan batch look like, next to a discarded one?",
            "purpose": "exploratory",
            "seed": SEED,
        },
        # A different destination, to show that one moves in YAML too.
        "output": {"root": "../outputs/explorations"},
        "cohort": {
            "sites": ["Raritan"],
            "filters": [
                {
                    "column": "Manufacturing and Release Testing Completed? (Y/N)",
                    "op": "equals",
                    "value": "Y",
                    "reason": "manufacturing and testing finished",
                }
            ],
        },
        "target": {
            "column": "Disposition",
            "type": "binary",
            "positive_class": "Released",
            "negative_class": "Discarded",
        },
        "columns": {
            "id": "Patient Lot/Batch #",
            "group": "Vector Lot",
            "exclude": [{"column": "OOS Type", "reason": "it is the reason for the disposition"}],
        },
        **_SMALL_COHORT,
        # The Raritan cohort is just big enough to show the other two encoders,
        # so this task lowers the sample-size bars they normally have to clear.
        "encoding": {
            **_SMALL_COHORT["encoding"],
            "target_min_rows_per_category": 3,
            "hashing_min_rows": 20,
            "missing_indicators": True,
        },
        "clustering": {
            "action": "linear",
            "method": "spearman",
            "threshold": 0.8,
            "min_overlap": 8,
        },
        "report": {"preview_rows": 10},
    }

    written = []
    for name, task in zip(TASK_FILES, (car_expression, release_outcome)):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# A demo task written by `pvf init --demo`. Paths are relative to this file.\n"
            + yaml.safe_dump(task, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append(path)
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "demo")
    written = write_config(target)
    print(f"A demo workspace is in {target}/. Now run:  cd {target} && pvf all")
