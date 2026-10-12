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

It writes the config, corrections, sources and two task workspaces, so the demo
is also the example of how they are laid out::

    python -m pvf.demo demo
    cd demo && pvf all --task tasks/site_a_car_expression --task tasks/site_b_release
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SEED = 20260917
SITE_A_BATCHES = 40
SITE_B_BATCHES = 30

# Deliberate gaps, each one visible in the report. Changing these changes what
# the demo demonstrates, which is the point of naming them here.
BLOCKED_INPUT = "<pt_cd8_pct>"  # no site records it: two features blocked
SITE_B_ONLY_GAP = "<harvest_glucose>"  # SiteA only: a feature one site cannot have
UNLISTED_FEATURES = ("Shift", "<f_moi_per_cd3>")  # computable, but not PTF parameters
ORPHAN_PARAMETERS = ("Sterility", "<mycoplasma_method>")  # in the PTF, supplied by nothing
UNMAPPED_ACRONYM = "XYZ"  # a clinical site no mapping file covers
LOT_WITHOUT_COA = "LV-909"  # a vector lot with no certificate

# For the dataset stage: one categorical parameter per encoder, an ordinal one,
# and a target a couple of batches are missing.
TARGET = "<fp_car_pct>"
ONE_HOT_PARAMETER = "<clinical_site>"  # few categories
TARGET_ENCODED_PARAMETER = "<team>"  # several
HASHED_PARAMETER = "<incubator_id>"  # many
ORDINAL_PARAMETER = "<ordinal_parameter>"
ORDINAL_ORDER = ("None", "Low", "Medium", "High")
ORDINAL_STRAY = "Severe"  # not one of the PTF's categories
#: What the target is called once the dataset stage has sanitised its name.
TARGET_COLUMN = "fp_flow_car"

# Value types the PTF declares for the columns whose type is not obvious from
# their contents. Everything else is numeric or categorical, decided below.
DATETIME_PARAMETERS = (
    "<thaw_time>",
    "<incubator_in_d0>",
    "<incubator_out_d1>",
    "<incubator_in_d1>",
    "<incubator_out_d3>",
    "<incubator_back_d3>",
    "<incubator_out_d6>",
    "<incubator_back_d6>",
    "<incubator_out_d8>",
    "<incubator_back_d8>",
    "<incubator_out_d10>",
)
DURATION_PARAMETERS = (
    "<contact_time>",
    "<contact_time_a>",
    "<contact_time_b>",
)
RATIO_PARAMETERS = ("<pt_cd4_cd8_ratio>",)
CATEGORICAL_PARAMETERS = (
    "Patient Lot/Batch #",
    "<patient_id>",
    "<vector_lot>",
    "<clinical_site>",
    "<clinical_site_id>",
    "Country",
    "Type",
    "Disposition",
    "<oos_type>",
    "Mycoplasma",
    "<nonconformance>",
    "<pre_activation_clumps_yn>",
    "<d3_clump_pattern>",
    "<clump_bags_1>",
    "<clump_bags_2>",
    "<clump_bags_3>",
    "<f_growth_profile>",
    "<f_cd4_profile>",
    "Site Merged",
    "<media_lot>",
    "<cytokine_lot>",
    "Sterility",
    "<mycoplasma_method>",
    "<coa_prefix><coa_site_format>",
    "<testing_done>",
    ONE_HOT_PARAMETER,
    TARGET_ENCODED_PARAMETER,
    HASHED_PARAMETER,
)

#: Feature names the PTF lists, so the registry may create them. The two in
#: UNLISTED_FEATURES are left out on purpose.
FEATURE_PARAMETERS = (
    "<f_log_pt_cd4_cd8>",
    "<f_log_pw_cd4_cd8>",
    "<f_pt_cd3_cells>",
    "<f_pt_cd4_share>",
    "<f_pt_cd4_cells>",
    "<f_pw_cd4_share>",
    "<f_pw_cd8_share>",
    "<f_harvest_lactate_norm>",
    "<f_lactate_per_cell_1>",
    "<f_lactate_per_cell_2>",
    "<f_lactate_per_cell_3>",
    "<f_glucose_per_cell_1>",
    "<f_glucose_per_cell_2>",
    "<f_glucose_per_cell_3>",
    "<f_lactate_slope>",
    "<f_lactate_auc>",
    "<f_glucose_slope>",
    "<f_glucose_auc>",
    "<f_lactate_glucose_ratio:6>",
    "<f_lactate_glucose_ratio:8>",
    "<f_lactate_glucose_ratio:10>",
    "<f_growth_profile>",
    "<f_cd4_profile>",
    "<f_viability_drop_rel>",
    "<f_viability_drop_pts>",
    "<f_viability_d3_d10_rel>",
    "<f_viability_expansion>",
    "<f_viability_d3_d10_pts>",
    "<f_viability_wash>",
    "<f_seeding_pct>",
    "<f_seeding_error>",
    "<f_seeding_within>",
    "<f_cells_per_bag>",
    "<f_seeded_spread:mean>",
    "<f_seeded_spread:min>",
    "<f_seeded_spread:max>",
    "<f_seeded_spread:std>",
    "<f_seeded_spread:cv>",
    "<f_vector_volume_spread:mean>",
    "<f_vector_volume_spread:min>",
    "<f_vector_volume_spread:max>",
    "<f_vector_volume_spread:std>",
    "<f_vector_volume_spread:cv>",
    "<f_moi_a>",
    "<f_log_moi_a>",
    "<f_moi_b>",
    "<f_log_moi_b>",
    "<f_moi_pool>",
    "<f_log_moi_pool>",
    "<f_moi_quality>",
    "<f_contact_deviation_1>",
    "<f_contact_within_1>",
    "<f_contact_deviation_2>",
    "<f_contact_within_2>",
    "<f_contact_deviation_3>",
    "<f_contact_within_3>",
    "<f_incubation_d0_d1>",
    "<f_incubation_d1_d3>",
    "<f_incubation_d3_d6>",
    "<f_incubation_d6_d8>",
    "<f_incubation_d8_d10>",
    "<f_incubation_total>",
    "<f_clump_count_reduction>",
    "<f_clump_size_reduction>",
    "<f_clump_burden_pre>",
    "<f_clump_burden_post>",
    "<f_clump_burden_mix>",
    "<f_mixing_x_clumps>",
    "<f_d3_clumps>",
    "<f_clump_bag_count>",
    "<f_dose_all>",
    "<f_post_formulation_dose>",
    "<f_achievable_dose>",
    "<f_car_cells>",
    "<f_car_yield>",
    "<f_vcn_per_cell>",
    "<f_flow_accuracy>",
    "<f_formulated_total_concentration>",
    "<f_cells_into_formulation>",
    "<f_cells_out_of_formulation>",
    "<f_cells_lost_formulation>",
    "<f_pct_lost_formulation>",
    "<f_cells_lost_wash>",
    "<f_coa_impurity_index>",
    "<f_il2_d6>",
    "<f_il2_d8>",
    "Day 6 <co2_marker> dev_from_5.0pct",
    "Day 6 <temperature_marker> dev_from_37.0C",
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
    """The measurements both sites record, under the SiteA names.

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
            "<vessel_count>": vessels,
            "<vsvg_cells>": vsvg,
            # 5% more cells were available than were seeded, every time: the pair
            # is a near-duplicate and one of them is dropped.
            "<cells_available>": (vsvg * 1.05).round(0),
            "<cells_seeded_a>": seeded,
            "<cells_seeded_b>": np.where(
                vessels == 2, (seeded * 0.97).round(0), np.nan
            ),
            "<culture_bags>": np.where(vessels == 2, 4, 2),
            "<vector_volume_a>": (2.1 * noise(0.06)).round(2),
            "<vector_volume_b>": np.where(
                vessels == 2, (2.1 * noise(0.06)).round(2), np.nan
            ),
            # Metabolites: one rate, sampled on three days. Day 8 vessel B is
            # absent from both sites on purpose — that feature is blocked while
            # the D6-D10 slope still has two of its three days.
            "<d6_lactate_a>": (metabolism * 1.15 * noise(0.07)).round(3),
            "<d6_lactate_b>": np.where(
                vessels == 2, (metabolism * 1.15 * noise(0.07)).round(3), np.nan
            ),
            "<d8_lactate_a>": (metabolism * 1.9 * noise(0.07)).round(3),
            "<harvest_lactate>": (metabolism * 2.6 * noise(0.07)).round(3),
            "<d6_glucose_a>": (3.4 - metabolism * 0.7 * noise(0.06)).round(3),
            "<d6_glucose_b>": np.where(
                vessels == 2, (3.4 - metabolism * 0.7 * noise(0.06)).round(3), np.nan
            ),
            "<d8_glucose_a>": (2.6 - metabolism * 0.6 * noise(0.06)).round(3),
            "<d8_glucose_b>": np.where(
                vessels == 2, (2.6 - metabolism * 0.6 * noise(0.06)).round(3), np.nan
            ),
            "<harvest_glucose>": (1.9 - metabolism * 0.5 * noise(0.06)).round(3),
            "<cpdl>": (4.6 * metabolism**0.6 * noise(0.06)).round(2),
            # Flow
            "<pt_cd3_pct>": rng.uniform(55, 88, n).round(1),
            "<pt_cd4_pct>": rng.uniform(28, 72, n).round(1),
            "<pw_cd3_pct>": rng.uniform(80, 97, n).round(1),
            "<pw_cd4_pct>": rng.uniform(30, 70, n).round(1),
            "<pw_cd8_pct>": rng.uniform(25, 60, n).round(1),
            "<pt_cd4_cd8_ratio>": [f"1:{v:.2f}" for v in rng.uniform(0.6, 2.4, n)],
            "<pt_viable_cells>": (size * 4.2e8).round(0),
            # Viability
            "<pw_viability>": rng.uniform(88, 98, n).round(1),
            "<d3_viability>": rng.uniform(85, 96, n).round(1),
            "<harvest_viability>": rng.uniform(80, 95, n).round(1),
            "<washed_viability>": rng.uniform(82, 96, n).round(1),
            # Harvest and formulation: every count follows the batch size, which
            # is what puts them in one correlated cluster.
            "<harvest_cells>": prewash,
            "<washed_cells>": postwash,
            "<washed_concentration>": (
                postwash / formulation_volume
            ).round(0),
            "<formulation_cell_volume>": formulation_volume,
            "<formulation_medium_volume>": (formulation_volume * 1.3).round(1),
            "<bag_volume>": rng.choice([20.0, 30.0], n),
            "<formulated_concentration>": (
                postwash / (formulation_volume * 1.3) * noise(0.03)
            ).round(0),
            "<formulated_viability>": rng.uniform(84, 96, n).round(1),
            # Product and dose. The target and the harvest CAR reading share a
            # latent, as they do in a real process.
            "<harvest_car_pct>": car_expression,
            TARGET: (car_expression * noise(0.08)).round(1),
            "<vcn>": (
                1.1 + 2.0 * transduction
            ).round(2),
            "<subject_weight>": rng.uniform(52, 104, n).round(1),
            "<dose_per_kg>": (size * 2.4e6).round(0),
            "<dose_cells>": (size * 2.6e8).round(0),
            # Timing
            "<thaw_time>": [f"{8 + i % 6:02d}:{(i * 13) % 60:02d}" for i in range(n)],
            "<incubator_in_d0>": _timestamps(n, 0, 9),
            "<incubator_out_d1>": _timestamps(n, 1, 10),
            "<incubator_in_d1>": _timestamps(n, 1, 11),
            "<incubator_out_d3>": _timestamps(n, 3, 9),
            "<incubator_back_d3>": _timestamps(n, 3, 12),
            "<incubator_out_d6>": _timestamps(n, 6, 9),
            "<incubator_back_d6>": _timestamps(n, 6, 13),
            "<incubator_out_d8>": _timestamps(n, 8, 9),
            "<incubator_back_d8>": _timestamps(n, 8, 14),
            "<incubator_out_d10>": _timestamps(n, 10, 8),
            "<contact_time>": [
                f"0{1 + i % 2}:{(i * 11) % 60:02d}" for i in range(n)
            ],
            "<contact_time_a>": [
                f"0{1 + i % 2}:{(i * 7) % 60:02d}" for i in range(n)
            ],
            "<contact_time_b>": [
                f"0{1 + i % 2}:{(i * 5) % 60:02d}" for i in range(n)
            ],
            # Clumps
            "<clumps_pre_count>": rng.integers(2, 12, n),
            "<clumps_pre_size>": rng.integers(1, 5, n),
            "<clumps_post_count>": rng.integers(0, 6, n),
            "<clumps_post_size>": rng.integers(1, 4, n),
            "<clumps_mix_count>": rng.integers(0, 4, n).astype(str),
            "<clumps_mix_size>": rng.integers(1, 3, n),
            "<mixing_cycles>": rng.integers(2, 7, n),
            "<d3_clump_pattern>": rng.choice(["None", "Few small", "Many small"], n),
            "<clump_bags_1>": rng.choice(["Yes", "No"], n),
            "<clump_bags_2>": rng.choice(["Yes", "No"], n),
            "<clump_bags_3>": rng.choice(["Yes", "No"], n),
            ORDINAL_PARAMETER: _cycle([*ORDINAL_ORDER, ORDINAL_STRAY], n),
            # Feeds and environment
            "<il2_activity_d6>": rng.uniform(1.4e7, 2.2e7, n).round(0),
            "<il2_content_d6>": rng.uniform(180, 240, n).round(1),
            "<il2_activity_d8>": rng.uniform(1.4e7, 2.2e7, n).round(0),
            "<il2_content_d8>": rng.uniform(180, 240, n).round(1),
            "Day 6 <co2_marker>": rng.uniform(4.6, 5.4, n).round(2),
            "Day 6 <temperature_marker>": rng.uniform(36.4, 37.6, n).round(2),
            # Who and where: one categorical per encoder the dataset stage has.
            # Drawn rather than cycled: a strict cycle of six teams over three
            # vector lots gives each lot its own two teams, and grouped folds
            # then have nothing to learn a team's mean from.
            TARGET_ENCODED_PARAMETER: rng.choice(list("ABCDEF"), n),
            HASHED_PARAMETER: _cycle([f"INC-{i:02d}" for i in range(10)], n),
            # Release
            "<endotoxin>": [
                "<0.5<value:endotoxin_unit>" if i % 4 == 0 else f"{v:.2f}<value:endotoxin_unit>"
                for i, v in enumerate(rng.uniform(0.1, 1.2, n))
            ],
            "<recovery>": rng.uniform(70, 99, n).round(1),
            "<testing_done>": _cycle(["Y"] * 9 + ["N"], n),
            "Mycoplasma": "<value:mycoplasma_raw>",
            "<oos_type>": _cycle(["VIability", "<value:oos_raw>", "CAR", ""], n),
            "<nonconformance>": _cycle(
                ["", "", "", "deviation", "withdrawn", "Withdrawal", "Termination"], n
            ),
        }
    )


# SiteB records some parameters under its own names. The mapping file below is
# what teaches the pipeline they are the same thing.
SITE_B_NAMES = {
    "Patient Lot/Batch #": "<remfg_batch_keys_3>",
    "<vector_lot>": "<site_b_vector_lot>",
    "<subject_weight>": "<site_b_weight>",
    "<harvest_car_pct>": "<site_b_car>",
}


def site_a_phf(rng: np.random.Generator, n: int = SITE_A_BATCHES) -> pd.DataFrame:
    """SiteA batches, with the values that give cleaning something to do."""
    df = _site_frame(n, rng)
    df.insert(0, "Patient Lot/Batch #", [f"GH{i:03d}" for i in range(n)])
    df.insert(1, "<patient_id>", [f"EU-{1200 + i % 3:04d}-{i:03d}" for i in range(n)])
    df.insert(2, "<vector_lot>", _cycle(["LV-101", "LV-102", LOT_WITHOUT_COA], n))
    df["<clinical_site>"] = _cycle(["<value:clinical_sites_a>", "UZ Leuven\xa0", "<value:clinical_sites_a>", ""], n)
    df["Country"] = _cycle(["Belgium", "Netherlands"], n)
    df["Type"] = _cycle(["Commercial"] * 9 + ["Clinical"], n)
    df["Disposition"] = _cycle(["Released", "Rejected"], n)
    df["<pre_activation_clumps_yn>"] = _cycle(["Yes", "No", None, "Yes"], n)
    df["<seeding_density_a>"] = rng.uniform(
        4e11, 6e11, n
    ).round(0)
    df["<site_a_comment>"] = "site note"  # not a PTF parameter: a SiteA-only column

    # One batch the integrity corrections know by name.
    df.loc[0, "Patient Lot/Batch #"] = "<value:bad_lot>"
    # A number recorded as text, and a percentage that cannot be one. The column
    # becomes text here exactly as the source file has it — coercion is the
    # pipeline's job, not the fixture's.
    df["<recovery>"] = df["<recovery>"].astype(object)
    df.loc[1, "<recovery>"] = "not measured"
    df.loc[2, "<recovery>"] = 105.0
    # A timestamp and a duration nobody filled in properly.
    df.loc[3, "<incubator_out_d8>"] = "not recorded"
    df.loc[4, "<contact_time_b>"] = "not started"
    # A duration that ran backwards, and two batches whose target was never read.
    df.loc[2, "<contact_time>"] = "00:00"
    df.loc[[6, 7], TARGET] = np.nan
    return df


def site_b_phf(rng: np.random.Generator, n: int = SITE_B_BATCHES) -> pd.DataFrame:
    """SiteB batches, under SiteB's own column names."""
    df = _site_frame(n, rng)
    df.insert(0, "<remfg_batch_keys_3>", [f"RA{i:03d}" for i in range(n)])
    df.insert(1, "<site_b_vector_lot>", _cycle(["LV-102", "LV-103", "LV-101"], n))
    df = df.rename(
        columns={
            "<subject_weight>": "<site_b_weight>",
            "<harvest_car_pct>": "<site_b_car>",
        }
    )
    # SiteB does not record harvest glucose, so one feature cannot be computed
    # there and is reported as blocked for that site alone.
    df = df.drop(columns=[SITE_B_ONLY_GAP], errors="ignore")
    df["<clinical_site>"] = _cycle(["MSKCC", "<value:clinical_sites_b>", UNMAPPED_ACRONYM], n)
    df["Country"] = "<value:country_to_harmonise>"
    # A second column that maps onto Country as well, spelled differently so a
    # test can tell which values ended up under which name.
    df["<site_b_country>"] = "USA"
    df["<site_b_site_code>"] = "<value:site_b_site_code_value>"  # mapped to nothing, so the mapping drops it
    df["Type"] = "Clinical"  # cleaning forces this to Commercial
    df["Disposition"] = _cycle(["Released", "Discarded"], n)  # a level SiteA never uses
    df["<pre_activation_clumps_yn>"] = _cycle(["Y", "N", None], n)
    df["<cells_per_bag>"] = rng.uniform(3e12, 7e12, n).round(0)
    df[ORDINAL_PARAMETER] = _cycle(list(ORDINAL_ORDER), n)
    df["<site_b_comment>"] = "cmd note"  # not a PTF parameter: a SiteB-only column

    # The clump count that is really a date Excel mangled.
    df.loc[0, "<clumps_mix_count>"] = "<value:mangled_count>"
    return df


def parameter_mapping() -> pd.DataFrame:
    """SiteB parameter names against their SiteA equivalents."""
    rows = [(site_b, site_a) for site_a, site_b in SITE_B_NAMES.items()]
    rows += [
        ("<site_b_country>", "Country"),  # a second column claiming a taken name
        ("<site_b_site_code>", None),  # no SiteA equivalent: dropped
        ("<excluded_mapping_row>", "<nonconformance>"),  # known bad entry
    ]
    return pd.DataFrame(
        rows, columns=["<map_site_b_name>", "<map_site_a_name>"]
    )


#: The certificate columns, in the order the workbook holds them. The first is
#: the lot number, which the loader renames to <vector_lot>.
COA_COLUMNS = (
    "<coa_lot>",
    "<coa_available>",
    "<coa_site_format>",
    "<coa_titer>",
    "<coa_titer_unadjusted>",
    "<coa_p24>",
    "<coa_pi_ratio>",
    "<coa_donor:<coa_car_metric>-1>",
    "<coa_donor:<coa_car_metric>-2>",
    "<coa_donor:IFNg-1>",
    "<coa_donor:IFNg-2>",
    "pH",
    "Osmolality",
    "Volume",
    "<coa_dna_size>",
    "HCP",
    "pDNA",
    "cDNA",
    "<coa_impurities_4>",
    "<coa_impurities_5>",
    "<coa_made>",
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
            "SiteA/250mL",
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
            "SiteB/500mL",
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
            "SiteB/500mL",
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
            "SiteA/250mL",
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


def remanufacturing(n: int = SITE_B_BATCHES) -> pd.DataFrame:
    """Prior lines of therapy, keyed by the SiteB batch number."""
    return pd.DataFrame(
        {
            "<remfg_batch_keys_1>": [f"RA{i:03d}" for i in range(n)] + [None],
            "<prior_lines_source>": list(range(1, n + 1)) + [None],
        }
    )


def raw_materials(site_a: int = SITE_A_BATCHES, site_b: int = SITE_B_BATCHES) -> pd.DataFrame:
    """The consumables export, with the column names its query tool produces."""
    batches = [f"GH{i:03d}" for i in range(site_a)] + [f"RA{i:03d}" for i in range(site_b)]
    batches[0] = "<value:bad_lot>"
    return pd.DataFrame(
        {
            "Batch": batches,
            "Query[<media_lot>]": [f"MED-{100 + i % 4}" for i in range(len(batches))],
            "Query[<cytokine_lot>]": [f"IL2-{200 + i % 3}" for i in range(len(batches))],
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
            "Query[<harvest_lactate>]": np.linspace(1.5, 3.0, n).round(2),
            "Query[<investigation_id>]": [f"INV-{i:03d}" for i in range(n)],
            "Query[<root_cause>]": _cycle(["Operator", "Equipment", "Material"], n),
            "<days_to_close>": _cycle([7, 14, 30], n),
        }
    )


def site_mappings() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two acronym files, as they arrive from two different meetings."""
    first = pd.DataFrame(
        {"Acronym": ["MSKCC", "MDACC"], "Institution": ["<value:institutions_first>", "<value:institutions_first>"]}
    )
    second = pd.DataFrame(
        {
            "Acronym": ["<value:clinical_sites_b>", "MSKCC"],
            "Institution": ["<value:institutions_second>", "<value:institutions_second>"],
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


#: The made-up PTF's 'Available At' column, by parameter name: a category a task
#: can select parameters by. Earliest match wins; features are left without one.
STAGE_RULES = (
    ("Disposition", "Post-release"),
    ("<oos_type>", "Post-release"),
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
    ("<culture_bags>", "D3"),
    ("Volume Vector Added", "D3"),
    ("Total Viable Cells for Expansion", "D3"),
    ("<cells_per_bag>", "D3"),
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
    ("<media_lot>", "D0"),
    ("<cytokine_lot>", "D0"),
    ("<team>", "D0"),
    ("<clinical_site>", "D0"),
    ("Country", "D0"),
    ("Subject Weight", "D0"),
    ("Number of Prior Lines", "D0"),
    ("Type", "D0"),
    ("Patient", "D0"),
    ("<vector_lot>", "D0"),
    ("Site Merged", "D0"),
)
#: Parameters the made-up PTF leaves without a stage.
UNSTAGED = ("Day 6 <temperature_marker>",)


def _stage(parameter: str) -> str:
    if parameter in FEATURE_PARAMETERS or parameter in UNSTAGED:
        return ""
    for fragment, stage in STAGE_RULES:
        if fragment in parameter:
            return stage
    return ""


def ptf(site_a: pd.DataFrame, site_b: pd.DataFrame) -> pd.DataFrame:
    """The PTF: every parameter the PVF may hold, and the type it holds.

    Built from what the sources carry, minus the few things left out on purpose —
    the site comment columns, which the PTF has never accepted, and the two
    features in ``UNLISTED_FEATURES`` — plus parameters nothing supplies, so the
    report has a PTF gap to show.
    """
    to_site_a = {site_b_name: site_a for site_a, site_b_name in SITE_B_NAMES.items()}
    to_site_a["<site_b_country>"] = "Country"
    mapped = dict.fromkeys(to_site_a.get(c, c) for c in site_b.columns)
    certificates = [
        f"<coa_prefix>{column}"
        for column in COA_COLUMNS[2:]
        if column not in ("<coa_available>", "<coa_titer_unadjusted>", "<coa_made>")
    ]

    parameters = dict.fromkeys(
        [
            *site_a.columns,
            *mapped,
            "<vector_lot>",
            "<prior_lines>",
            "<clinical_site_id>",
            "<media_lot>",
            "<cytokine_lot>",
            *certificates,
            *FEATURE_PARAMETERS,
            *ORPHAN_PARAMETERS,
        ]
    )
    for absent in (
        "<site_a_comment>",
        "<site_b_comment>",
        "<site_b_site_code>",
        "<site_b_country>",
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

    site_a = site_a_phf(rng)
    site_b = site_b_phf(rng)
    first_map, second_map = site_mappings()

    paths = {
        "ptf": directory / "PTF.xlsx",
        "phf_site_a": directory / "phf_site_a.xlsx",
        "phf_site_b": directory / "phf_site_b.xlsx",
        "param_mapping": directory / "parameter_requirements.xlsx",
        "remfg": directory / "remfg.xlsx",
        "lv_coa": directory / "lv_coa.xlsx",
        "raw_materials": directory / "raw_materials.csv",
        "investigations": directory / "investigations_power_query.xlsx",
        "site_maps": [directory / "site_map_hannelore.xlsx", directory / "site_map_meeting.xlsx"],
    }

    ptf(site_a, site_b).to_excel(paths["ptf"], index=False)
    site_a.to_excel(paths["phf_site_a"], sheet_name="<sheet:phf_site_a>", index=False)
    site_b.to_excel(paths["phf_site_b"], sheet_name="<sheet:phf_site_a>", index=False)
    parameter_mapping().to_excel(paths["param_mapping"], sheet_name="Overall", index=False)
    remanufacturing().to_excel(paths["remfg"], sheet_name="<sheet:remfg>", index=False)
    # Two rows of workbook furniture above the header, as the real file has.
    lv_coa().to_excel(paths["lv_coa"], sheet_name="<sheet:lv_coa>", index=False, startrow=2)
    raw_materials().to_csv(paths["raw_materials"], index=False)
    investigations().to_excel(
        paths["investigations"], sheet_name="<sheet:investigations>", index=False
    )
    first_map.to_excel(paths["site_maps"][0], index=False)
    second_map.to_excel(paths["site_maps"][1], index=False)

    return {
        key: [str(p) for p in value] if isinstance(value, list) else str(value)
        for key, value in paths.items()
    }


def write_config(directory: Path, seed: int = SEED) -> Path:
    """Write the demo data, config and task workspaces into ``directory``; return the config.

    Sources and the PTF go to ``data/raw``, the PVF with its reports and
    manifests to ``data/processed``, and each task to its own workspace under
    ``tasks/``. Every path in the config is relative to the config.

    The thresholds here are the ones that suit a cohort this small: with forty
    batches, a parameter with a dozen categories is already high-cardinality, and
    a feature needs fewer values before it is worth keeping.
    """
    directory = Path(directory)
    raw = directory / "data" / "raw"
    sources = write_sources(raw, seed=seed)
    (directory / "data" / "processed").mkdir(parents=True, exist_ok=True)
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
            "site_b_type_commercial": True,
            "yes_no_defaults": {"<pre_activation_clumps_yn>": "No"},
        },
        "mapping": {"exclude": ["<excluded_mapping_row>"]},
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

    config_path = directory / "config.yaml"
    config_path.write_text(
        "# The demo config, written by `python -m pvf.demo`. Every source under data/raw\n"
        "# is made up. Every path below is relative to this file.\n"
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
    # The minimum has to be smaller than a cohort this size.
    "quality": {"min_non_missing": 8, "duplicate_r2_threshold": 0.99, "duplicate_min_overlap": 8},
}

#: The made-up PTF's stages after harvest, which the SiteA demo task leaves out.
LATER_THAN_HARVEST = [
    "Wash", "Formulation", "Final product", "Post-thaw", "Release", "Post-release"
]

#: The demo's task workspaces, relative to the config.
TASK_FILES = ("tasks/site_a_car_expression", "tasks/site_b_release")


def write_tasks(directory: Path) -> list[Path]:
    """Two task workspaces over the same PVF, differing only in what their YAML says.

    The point of the pair is that nothing else changes: same package, same
    commands, same code. Cohort, target, roles, columns and clustering action
    all move in YAML.
    """
    directory = Path(directory)
    car_expression = {
        "config": "../../config.yaml",
        "task": {
            "name": "site_a_car_expression",
            "question": "Which manufacturing parameters go with final product CAR+ "
            "expression at SiteA?",
            "seed": SEED,
        },
        "cohort": {
            "sites": ["SiteA"],
            "filters": [
                {
                    "column": "Type",
                    "op": "equals",
                    "value": "Commercial",
                    "reason": "manufactured commercially",
                },
                {
                    "column": "<testing_done>",
                    "op": "equals",
                    "value": "Y",
                    "reason": "manufacturing and testing finished",
                },
                {
                    "column": "<nonconformance>",
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
            "group": "<vector_lot>",
            "exclude": [
                {
                    "column": "<subject_weight>",
                    "reason": "dose per kilogram already carries it",
                },
                {
                    "column": "<washed_cells>",
                    "reason": "counted after the wash",
                },
            ],
            # Everything the PTF says is measured after harvest.
            "exclude_categories": {"Available At": LATER_THAN_HARVEST},
        },
        **_SMALL_COHORT,
        "clustering": {
            "action": "report_only",
            "method": "spearman",
            "threshold": 0.7,
            "min_overlap": 8,
        },
        "report": {"preview_rows": 15},
    }

    release_outcome = {
        "config": "../../config.yaml",
        "task": {
            "name": "site_b_release",
            "question": "What does a released SiteB batch look like, next to a discarded one?",
            "seed": SEED,
        },
        "cohort": {
            "sites": ["SiteB"],
            "filters": [
                {
                    "column": "<testing_done>",
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
            "group": "<vector_lot>",
            "exclude": [{"column": "<oos_type>", "reason": "it is the reason for the disposition"}],
            # Whole categories of the PTF's metadata can be left out by name.
            "exclude_categories": {"Available At": ["Post-release"]},
        },
        **_SMALL_COHORT,
        # The SiteB cohort is just big enough to show the other two encoders,
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
        path = directory / name / "task.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# A demo task workspace written by `python -m pvf.demo`. Its runs go to runs/.\n"
            + yaml.safe_dump(task, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append(path)
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "demo")
    written = write_config(target)
    print(
        f"The demo is in {target}/. Now run:  cd {target} && pvf all "
        + " ".join(f"--task {name}" for name in TASK_FILES)
    )
