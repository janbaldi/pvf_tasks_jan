"""Made-up sources, shaped like the real ones.

The real inputs live on a corporate share, so there is no way to run this
pipeline — or read its report — without them. This module writes a complete set
of stand-ins: two sites' batch files, the catalog that defines the schema, the
parameter-name mapping between the sites, the vector certificates, the
re-manufacturing supplement, the raw-material export, the investigations Power
Query workbook, the clinical-site mappings, and the vocabulary workbook that
names all of them.

The numbers, names and values are invented, but the *shapes* are deliberate.
Every path the pipeline can take is represented somewhere in here: values that
need cleaning, a parameter only one site records, a batch whose vector lot has no
certificate, a clinical-site acronym nobody mapped, a feature whose input is
missing, a feature the catalog does not list, a catalog parameter no source
supplies, a parameter for each encoder the dataset stage has, and a group of
parameters correlated strongly enough to be de-correlated. What the reports show
is therefore what the pipeline does.

Every name goes through a key, exactly as in the package: the columns the
package uses come from the vocabulary in use (:func:`ml.names.demo` unless one is
passed), and the demo's own columns and values from the tables below.

It writes the config, sources and two task workspaces, so the demo is also the
example of how they are laid out. There is no SharePoint here, so the sources go
into ``sharepoint/``, a folder laid out the way the share is, and the demo's
``.env`` points ``$ML_SHAREPOINT_STANDIN`` at it::

    python -m ml.demo demo
    cd demo && ml update --task tasks/expression --task tasks/release
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import corrections, features, names, screening
from .encode import sanitise

SEED = 20260917
SITE_A_BATCHES = 40
SITE_B_BATCHES = 30

#: The demo's own columns — the ones no stage of the package names — by key.
DEMO_COLUMNS: dict[str, str] = {
    "ordinal_parameter": "Mixing severity",
    "team": "Operator team",
    "incubator_id": "Incubator code",
    "recovery": "Recovery rate (%)",
    "testing_done": "Testing complete (Y/N)",
    "mycoplasma": "Contaminant test",
    "oos_type": "Out-of-spec class",
    "nonconformance": "Deviation class",
    "disposition": "Outcome",
    "pt_cd4_cd8_ratio": "Thawed CD4 to CD8 ratio",
    "seeding_density_a": "Seeding density vessel A",
    "seeding_density_b": "Seeding density vessel B",
    "site_a_comment": "Site note",
    "site_b_comment": "Remark",
    "site_b_batch": "Lot reference",
    "site_b_vector_lot": "Vector lot code",
    "site_b_weight": "Body weight (kg)",
    "site_b_car": "Expression before wash (%)",
    "site_b_country": "Nation name",
    "site_b_site_code": "Plant code",
    "cells_per_bag": "Viable cells per bag",
    "cells_for_recovery": "Viable cells recovered (cells)",
    "formulated_concentration_a": "Formulated concentration A (cells/mL)",
    "formulated_concentration_b": "Formulated concentration B (cells/mL)",
    "media_lot": "Medium lot",
    "cytokine_lot": "Cytokine lot",
    "sterility": "Sterile test",
    "mycoplasma_method": "Contaminant method",
    "investigation_id": "Case number",
    "root_cause": "Cause class",
    "days_to_close": "Days open",
}

#: The demo's own category values, by key.
DEMO_VALUES: dict[str, str | tuple[str, ...]] = {
    "batch_prefix_a": "NF",
    "batch_prefix_b": "SB",
    "clinical_sites_a": ("Hill Clinic", "Lake Hospital\xa0", " River Centre", ""),
    "countries_a": ("Northland", "Westland"),
    "type_clinical": "Trial",
    "released": "Accepted",
    "rejected": "Refused",
    "discarded": "Scrapped",
    "oos_raw": ("Viabilty", "Low Count", "Expr", ""),
    "nonconformance_raw": ("", "", "", "deviation", "recalled", "Recall", "Halt"),
    "excluded_statuses": ("halt", "recall"),
    "status_fix": ("recalled", "recall"),
    "mycoplasma_raw": "contaminant not found",
    "mycoplasma_clean": "not found",
    "clinical_sites_b": ("HCC", "20431", "XYZ"),
    "site_b_foreign_code": "20431",
    "foreign_country": "Eastland",
    "site_b_country_short": "SL",
    "site_b_site_code_value": "SB-01",
    "bad_lot": "XK0042",
    "mangled_count": "45123",
    "acronyms_first": ("HCC", "GMC"),
    "institutions_first": ("Harbour Cancer Centre", "Garden Medical College"),
    "acronyms_second": ("20431", "HCC"),
    "institutions_second": ("Eastland General", "Harbour Cancer Centre Main"),
    "oos_fixes_b": ("Viabilty|Viability", "Low Count|Count", "Expr|Expression"),
    "oos_fixes_a": ("Activity Assay|Activity",),
}


def c(key: str, **placeholders) -> str:
    """A column by its key: the vocabulary's when the package uses it, else the demo's."""
    if key in names.COLUMNS:
        return names.col(key, **placeholders)
    return DEMO_COLUMNS[key].format(**placeholders) if placeholders else DEMO_COLUMNS[key]


def cs(key: str) -> tuple[str, ...]:
    return names.cols(key)


def v(key: str):
    """A category value by its key: the vocabulary's, else the demo's (a tuple for a list)."""
    if key in names.VALUES:
        found = names.values(key)
        return found if len(found) > 1 else found[0]
    return DEMO_VALUES[key]


def vs(key: str) -> list:
    found = v(key)
    return list(found) if isinstance(found, tuple) else [found]


# Deliberate gaps, each one visible in the report. Changing these changes what
# the demo demonstrates, which is the point of naming them here.
def blocked_input() -> str:
    return c("pt_cd8_pct")  # no site records it: two features blocked


def site_a_only() -> str:
    return c("harvest_glucose")  # site A only: a feature one site cannot have


def unlisted_features() -> tuple[str, ...]:
    return (c("f_shift"), c("f_moi_per_cd3"))  # computable, but not catalog parameters


def orphan_parameters() -> tuple[str, ...]:
    return (c("sterility"), c("mycoplasma_method"))  # in the catalog, supplied by nothing


UNMAPPED_ACRONYM = "XYZ"  # a clinical site no mapping file covers
LOT_WITHOUT_COA = "LV-909"  # a vector lot with no certificate
BATCH = "Patient Lot/Batch #"

# For the dataset stage: one categorical parameter per encoder, an ordinal one,
# and a target a couple of batches are missing.
ORDINAL_ORDER = ("None", "Low", "Medium", "High")
ORDINAL_STRAY = "Severe"  # not one of the catalog's categories


def target() -> str:
    return c("fp_car_pct")


def target_column() -> str:
    """What the target is called once the dataset stage has sanitised its name."""
    return sanitise(target())


# The catalog's value type and 'Available At' stage, by key. Everything not listed
# is numeric; features and the keys under "" are left without a stage.
DATETIME_KEYS = (
    "date_start",
    "thaw_time",
    "incubator_in_d0",
    "incubator_out_d1",
    "incubator_in_d1",
    "incubator_out_d3",
    "incubator_back_d3",
    "incubator_out_d6",
    "incubator_back_d6",
    "incubator_out_d8",
    "incubator_back_d8",
    "incubator_out_d10",
)
DURATION_KEYS = ("contact_time", "contact_time_a", "contact_time_b")
RATIO_KEYS = ("pt_cd4_cd8_ratio",)
CATEGORICAL_KEYS = (
    "manufacturing_completed",
    "manufacturing_site",
    "patient_id",
    "vector_lot",
    "clinical_site",
    "clinical_site_id",
    "country",
    "batch_type",
    "disposition",
    "oos_type",
    "mycoplasma",
    "nonconformance",
    "pre_activation_clumps_yn",
    "d3_clump_pattern",
    "f_growth_profile",
    "f_cd4_profile",
    "media_lot",
    "cytokine_lot",
    "sterility",
    "mycoplasma_method",
    "testing_done",
    "clinical_site",
    "team",
    "incubator_id",
)
STAGES = {
    "D0": (
        "date_start",
        "manufacturing_completed",
        "manufacturing_site",
        "patient_id",
        "vector_lot",
        "subject_weight",
        "thaw_time",
        "team",
        "clinical_site",
        "country",
        "batch_type",
        "prior_lines",
        "clinical_site_id",
        "media_lot",
        "cytokine_lot",
    ),
    "D1": (
        "incubator_in_d0",
        "incubator_out_d1",
        "incubator_in_d1",
        "contact_time_a",
        "contact_time_b",
        "clumps_pre_count",
        "clumps_pre_size",
        "clumps_post_count",
        "clumps_post_size",
        "clumps_mix_count",
        "clumps_mix_size",
        "mixing_cycles",
        "ordinal_parameter",
        "incubator_id",
        "pre_activation_clumps_yn",
    ),
    "D3": (
        "vessel_count",
        "vsvg_cells",
        "cells_available",
        "cells_seeded_a",
        "cells_seeded_b",
        "culture_bags",
        "vector_volume_a",
        "vector_volume_b",
        "d3_viability",
        "incubator_out_d3",
        "incubator_back_d3",
        "contact_time",
        "d3_clump_pattern",
        "seeding_density_a",
        "cells_per_bag",
    ),
    "D6": (
        "d6_lactate_a",
        "d6_lactate_b",
        "d6_glucose_a",
        "d6_glucose_b",
        "incubator_out_d6",
        "incubator_back_d6",
        "il2_activity_d6",
        "il2_content_d6",
    ),
    "D8": (
        "d8_lactate_a",
        "d8_glucose_a",
        "d8_glucose_b",
        "incubator_out_d8",
        "incubator_back_d8",
        "il2_activity_d8",
        "il2_content_d8",
    ),
    "D10": ("cpdl", "incubator_out_d10"),
    "Harvest": (
        "harvest_lactate",
        "harvest_glucose",
        "harvest_viability",
        "harvest_cells",
        "harvest_car_pct",
    ),
    "Post-thaw": ("pt_cd3_pct", "pt_cd4_pct", "pt_cd4_cd8_ratio", "pt_viable_cells"),
    "Wash": (
        "pw_cd3_pct",
        "pw_cd4_pct",
        "pw_cd8_pct",
        "pw_viability",
        "washed_viability",
        "washed_cells",
        "washed_concentration",
    ),
    "Formulation": (
        "formulation_cell_volume",
        "formulation_medium_volume",
        "bag_volume",
        "formulated_concentration",
        "formulated_viability",
        "dose_per_kg",
        "dose_cells",
    ),
    "Final product": ("fp_car_pct", "vcn"),
    "Release": (
        "endotoxin",
        "recovery",
        "testing_done",
        "mycoplasma",
        "sterility",
        "mycoplasma_method",
    ),
    "Post-release": ("oos_type", "nonconformance", "disposition"),
}

#: The feature keys the catalog lists, so the registry may create them. The two in
#: :func:`unlisted_features` are left out on purpose.
FEATURE_KEYS = (
    "f_log_pt_cd4_cd8",
    "f_log_pw_cd4_cd8",
    "f_pt_cd3_cells",
    "f_pt_cd4_share",
    "f_pt_cd4_cells",
    "f_pw_cd4_share",
    "f_cd4_cd8_shift",
    "f_pw_cd8_share",
)
FEATURE_KEYS_AFTER_METABOLITES = (
    "f_lactate_slope",
    "f_lactate_auc",
    "f_glucose_slope",
    "f_glucose_auc",
)
FEATURE_KEYS_REST = (
    "f_growth_profile",
    "f_cd4_profile",
    "f_viability_drop_pts",
    "f_viability_d3_d10_rel",
    "f_viability_d3_d10_pts",
    "f_viability_wash",
    "f_seeding_pct",
    "f_seeding_error",
    "f_seeding_within",
    "f_cells_per_bag",
    "f_seeding_loss",
    "f_dead_cells_d3",
)
FEATURE_KEYS_MOI = (
    "f_moi_a",
    "f_log_moi_a",
    "f_moi_b",
    "f_log_moi_b",
    "f_moi_pool",
    "f_log_moi_pool",
    "f_moi_quality",
    "f_moi_vs_target",
)
FEATURE_KEYS_LATE = (
    "f_incubation_d0_d1",
    "f_incubation_d1_d3",
    "f_incubation_d3_d6",
    "f_incubation_d6_d8",
    "f_incubation_d8_d10",
    "f_incubation_total",
    "f_thaw_to_incubator",
    "f_clump_count_reduction",
    "f_clump_size_reduction",
    "f_clump_burden_pre",
    "f_clump_burden_post",
    "f_clump_burden_mix",
    "f_mixing_x_clumps",
    "f_d3_clumps",
    "f_clump_bag_count",
    "f_dose_all",
    "f_dose_basis",
    "f_post_formulation_dose",
    "f_achievable_dose",
    "f_car_cells",
    "f_car_yield",
    "f_vcn_per_cell",
    "f_flow_accuracy",
    "f_formulated_total_concentration",
    "f_cells_into_formulation",
    "f_cells_out_of_formulation",
    "f_cells_lost_formulation",
    "f_pct_lost_formulation",
    "f_cells_lost_wash",
    "f_wash_recovery",
    "f_coa_impurity_index",
    "f_coa_impurity_spec_index",
    "f_vector_age",
    "f_il2_d6",
    "f_il2_d8",
)


def co2_reading() -> str:
    return f"Day 6 {c('co2_marker')}"


def temperature_reading() -> str:
    return f"Day 6 {c('temperature_marker')}"


def feature_parameters() -> tuple[str, ...]:
    """The feature names the catalog lists, in the order the made-up catalog has them."""
    statistics = features.SPREAD_STATISTICS
    return (
        *(c(k) for k in FEATURE_KEYS),
        *cs("f_lactate_per_cell"),
        *cs("f_glucose_per_cell"),
        *(c(k) for k in FEATURE_KEYS_AFTER_METABOLITES),
        *cs("f_lactate_increment"),
        *cs("f_glucose_increment"),
        *(c("f_lactate_glucose_ratio", day=day) for day in (6, 8, 10)),
        *(c("f_lactate_yield", day=day) for day in (6, 8, 10)),
        *(c(k) for k in FEATURE_KEYS_REST),
        *(c("f_seeded_spread", statistic=s) for s in statistics),
        *(c("f_vector_volume_spread", statistic=s) for s in statistics),
        *(c(k) for k in FEATURE_KEYS_MOI),
        *(name for pair in zip(cs("f_contact_deviation"), cs("f_contact_within")) for name in pair),
        *(c(k) for k in FEATURE_KEYS_LATE),
        *cs("f_out_of_incubator"),
        f"{co2_reading()} dev_from_5.0pct",
        f"{temperature_reading()} dev_from_37.0C",
    )


def _cycle(values: list, n: int) -> list:
    """Repeat a short list of values over n batches, so a pattern is visible."""
    return [values[i % len(values)] for i in range(n)]


def _timestamps(n: int, day: int, hour: int) -> list[str]:
    """Batch timestamps as the batch records hold them: text, one batch per day."""
    return [
        f"2026-0{1 + i % 2}-{(i % 20) + 1:02d} {hour:02d}:{(day * 7 + i) % 60:02d}:00"
        for i in range(n)
    ]


def _site_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """The measurements both sites record, under site A's names.

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
    unit = v("endotoxin_unit")

    vessels = np.where(np.arange(n) % 3 == 0, 1, 2)
    seeded = (size * 5.4e5).round(0)
    vsvg = (size * vessels * 5.2e7).round(0)
    prewash = (size * 3.1e9 * noise(0.05)).round(0)
    postwash = (prewash * 0.88 * noise(0.03)).round(0)
    formulation_volume = (size * 58 * noise(0.04)).round(1)
    car_expression = (18 + 44 * transduction * noise(0.05)).round(1)

    return pd.DataFrame(
        {
            c("vessel_count"): vessels,
            c("vsvg_cells"): vsvg,
            # 5% more cells were available than were seeded, every time: the pair
            # is a near-duplicate and one of them is dropped.
            c("cells_available"): (vsvg * 1.05).round(0),
            c("cells_seeded_a"): seeded,
            c("cells_seeded_b"): np.where(vessels == 2, (seeded * 0.97).round(0), np.nan),
            c("culture_bags"): np.where(vessels == 2, 4, 2),
            c("vector_volume_a"): (2.1 * noise(0.06)).round(2),
            c("vector_volume_b"): np.where(vessels == 2, (2.1 * noise(0.06)).round(2), np.nan),
            # Metabolites: one rate, sampled on three days. Day 8 vessel B is
            # absent from both sites on purpose — that feature is blocked while
            # the D6-D10 slope still has two of its three days.
            c("d6_lactate_a"): (metabolism * 1.15 * noise(0.07)).round(3),
            c("d6_lactate_b"): np.where(
                vessels == 2, (metabolism * 1.15 * noise(0.07)).round(3), np.nan
            ),
            c("d8_lactate_a"): (metabolism * 1.9 * noise(0.07)).round(3),
            c("harvest_lactate"): (metabolism * 2.6 * noise(0.07)).round(3),
            c("d6_glucose_a"): (3.4 - metabolism * 0.7 * noise(0.06)).round(3),
            c("d6_glucose_b"): np.where(
                vessels == 2, (3.4 - metabolism * 0.7 * noise(0.06)).round(3), np.nan
            ),
            c("d8_glucose_a"): (2.6 - metabolism * 0.6 * noise(0.06)).round(3),
            c("d8_glucose_b"): np.where(
                vessels == 2, (2.6 - metabolism * 0.6 * noise(0.06)).round(3), np.nan
            ),
            c("harvest_glucose"): (1.9 - metabolism * 0.5 * noise(0.06)).round(3),
            c("cpdl"): (4.6 * metabolism**0.6 * noise(0.06)).round(2),
            # Flow
            c("pt_cd3_pct"): rng.uniform(55, 88, n).round(1),
            c("pt_cd4_pct"): rng.uniform(28, 72, n).round(1),
            c("pw_cd3_pct"): rng.uniform(80, 97, n).round(1),
            c("pw_cd4_pct"): rng.uniform(30, 70, n).round(1),
            c("pw_cd8_pct"): rng.uniform(25, 60, n).round(1),
            c("pt_cd4_cd8_ratio"): [f"1:{x:.2f}" for x in rng.uniform(0.6, 2.4, n)],
            c("pt_viable_cells"): (size * 4.2e8).round(0),
            # Viability
            c("pw_viability"): rng.uniform(88, 98, n).round(1),
            c("d3_viability"): rng.uniform(85, 96, n).round(1),
            c("harvest_viability"): rng.uniform(80, 95, n).round(1),
            c("washed_viability"): rng.uniform(82, 96, n).round(1),
            # Harvest and formulation: every count follows the batch size, which
            # is what puts them in one correlated cluster.
            c("harvest_cells"): prewash,
            c("washed_cells"): postwash,
            c("washed_concentration"): (postwash / formulation_volume).round(0),
            c("formulation_cell_volume"): formulation_volume,
            c("formulation_medium_volume"): (formulation_volume * 1.3).round(1),
            c("bag_volume"): rng.choice([20.0, 30.0], n),
            c("formulated_concentration"): (
                postwash / (formulation_volume * 1.3) * noise(0.03)
            ).round(0),
            c("formulated_viability"): rng.uniform(84, 96, n).round(1),
            # Product and dose. The target and the harvest CAR reading share a
            # latent, as they do in a real process.
            c("harvest_car_pct"): car_expression,
            target(): (car_expression * noise(0.08)).round(1),
            c("vcn"): (1.1 + 2.0 * transduction).round(2),
            c("subject_weight"): rng.uniform(52, 104, n).round(1),
            c("dose_per_kg"): (size * 2.4e6).round(0),
            c("dose_cells"): (size * 2.6e8).round(0),
            # Timing
            c("date_start"): _timestamps(n, 0, 7),
            # Three batches are still in production: drawn, not analysed.
            c("manufacturing_completed"): ["N" if i >= n - 3 else "Y" for i in range(n)],
            c("thaw_time"): [f"{8 + i % 6:02d}:{(i * 13) % 60:02d}" for i in range(n)],
            c("incubator_in_d0"): _timestamps(n, 0, 9),
            c("incubator_out_d1"): _timestamps(n, 1, 10),
            c("incubator_in_d1"): _timestamps(n, 1, 11),
            c("incubator_out_d3"): _timestamps(n, 3, 9),
            c("incubator_back_d3"): _timestamps(n, 3, 12),
            c("incubator_out_d6"): _timestamps(n, 6, 9),
            c("incubator_back_d6"): _timestamps(n, 6, 13),
            c("incubator_out_d8"): _timestamps(n, 8, 9),
            c("incubator_back_d8"): _timestamps(n, 8, 14),
            c("incubator_out_d10"): _timestamps(n, 10, 8),
            c("contact_time"): [f"0{1 + i % 2}:{(i * 11) % 60:02d}" for i in range(n)],
            c("contact_time_a"): [f"0{1 + i % 2}:{(i * 7) % 60:02d}" for i in range(n)],
            c("contact_time_b"): [f"0{1 + i % 2}:{(i * 5) % 60:02d}" for i in range(n)],
            # Clumps
            c("clumps_pre_count"): rng.integers(2, 12, n),
            c("clumps_pre_size"): rng.integers(1, 5, n),
            c("clumps_post_count"): rng.integers(0, 6, n),
            c("clumps_post_size"): rng.integers(1, 4, n),
            c("clumps_mix_count"): rng.integers(0, 4, n).astype(str),
            c("clumps_mix_size"): rng.integers(1, 3, n),
            c("mixing_cycles"): rng.integers(2, 7, n),
            c("d3_clump_pattern"): rng.choice(["None", "Few small", "Many small"], n),
            cs("clump_bags")[0]: rng.choice(["Yes", "No"], n),
            cs("clump_bags")[1]: rng.choice(["Yes", "No"], n),
            cs("clump_bags")[2]: rng.choice(["Yes", "No"], n),
            c("ordinal_parameter"): _cycle([*ORDINAL_ORDER, ORDINAL_STRAY], n),
            # Feeds and environment
            c("il2_activity_d6"): rng.uniform(1.4e7, 2.2e7, n).round(0),
            c("il2_content_d6"): rng.uniform(180, 240, n).round(1),
            c("il2_activity_d8"): rng.uniform(1.4e7, 2.2e7, n).round(0),
            c("il2_content_d8"): rng.uniform(180, 240, n).round(1),
            co2_reading(): rng.uniform(4.6, 5.4, n).round(2),
            temperature_reading(): rng.uniform(36.4, 37.6, n).round(2),
            # Who and where: one categorical per encoder the dataset stage has.
            # Drawn rather than cycled: a strict cycle of six teams over three
            # vector lots gives each lot its own two teams, and grouped folds
            # then have nothing to learn a team's mean from.
            c("team"): rng.choice(list("ABCDEF"), n),
            c("incubator_id"): _cycle([f"INC-{i:02d}" for i in range(10)], n),
            # Release
            c("endotoxin"): [
                f"<0.5{unit}" if i % 4 == 0 else f"{x:.2f}{unit}"
                for i, x in enumerate(rng.uniform(0.1, 1.2, n))
            ],
            c("recovery"): rng.uniform(70, 99, n).round(1),
            c("testing_done"): _cycle(["Y"] * 9 + ["N"], n),
            c("mycoplasma"): v("mycoplasma_raw"),
            c("oos_type"): _cycle(vs("oos_raw"), n),
            c("nonconformance"): _cycle(vs("nonconformance_raw"), n),
        }
    )


def site_b_names() -> dict[str, str]:
    """Site B records some parameters under its own names; the mapping file says so."""
    return {
        BATCH: c("site_b_batch"),
        c("vector_lot"): c("site_b_vector_lot"),
        c("subject_weight"): c("site_b_weight"),
        c("harvest_car_pct"): c("site_b_car"),
    }


def site_a_phf(rng: np.random.Generator, n: int = SITE_A_BATCHES) -> pd.DataFrame:
    """Site A's batches, with the values that give cleaning something to do."""
    df = _site_frame(n, rng)
    prefix = v("batch_prefix_a")
    df.insert(0, BATCH, [f"{prefix}{i:03d}" for i in range(n)])
    patient = v("patient_id_site_prefix")
    df.insert(1, c("patient_id"), [f"{patient}{1200 + i % 3:04d}-{i:03d}" for i in range(n)])
    df.insert(2, c("vector_lot"), _cycle(["LV-101", "LV-102", LOT_WITHOUT_COA], n))
    df[c("clinical_site")] = _cycle(vs("clinical_sites_a"), n)
    df[c("country")] = _cycle(vs("countries_a"), n)
    df[c("batch_type")] = _cycle([v("type_commercial")] * 9 + [v("type_clinical")], n)
    df[c("manufacturing_site")] = f"{names.site('a')} plant"  # the shifts report's split
    df[c("disposition")] = _cycle([v("released"), v("rejected")], n)
    df[c("pre_activation_clumps_yn")] = _cycle(["Yes", "No", None, "Yes"], n)
    df[c("seeding_density_a")] = rng.uniform(4e11, 6e11, n).round(0)
    df[c("site_a_comment")] = "site note"  # not a catalog parameter: a site A-only column

    # One batch the integrity corrections know by name.
    df.loc[0, BATCH] = v("bad_lot")
    # A number recorded as text, and a percentage that cannot be one. The column
    # becomes text here exactly as the source file has it — coercion is the
    # pipeline's job, not the fixture's.
    df[c("recovery")] = df[c("recovery")].astype(object)
    df.loc[1, c("recovery")] = "not measured"
    df.loc[2, c("recovery")] = 105.0
    # A timestamp and a duration nobody filled in properly.
    df.loc[3, c("incubator_out_d8")] = "not recorded"
    df.loc[4, c("contact_time_b")] = "not started"
    # A duration that ran backwards, and two batches whose target was never read.
    df.loc[2, c("contact_time")] = "00:00"
    df.loc[[6, 7], target()] = np.nan
    return df


def site_b_phf(rng: np.random.Generator, n: int = SITE_B_BATCHES) -> pd.DataFrame:
    """Site B's batches, under site B's own column names."""
    df = _site_frame(n, rng)
    prefix = v("batch_prefix_b")
    df.insert(0, c("site_b_batch"), [f"{prefix}{i:03d}" for i in range(n)])
    df.insert(1, c("site_b_vector_lot"), _cycle(["LV-102", "LV-103", "LV-101"], n))
    df = df.rename(
        columns={
            c("subject_weight"): c("site_b_weight"),
            c("harvest_car_pct"): c("site_b_car"),
        }
    )
    # Site B does not record harvest glucose, so one feature cannot be computed
    # there and is reported as blocked for that site alone.
    df = df.drop(columns=[site_a_only()], errors="ignore")
    df[c("clinical_site")] = _cycle(vs("clinical_sites_b"), n)
    df[c("country")] = v("country_to_harmonise")
    # A second column that maps onto the country as well, spelled differently so a
    # test can tell which values ended up under which name.
    df[c("site_b_country")] = v("site_b_country_short")
    df[c("site_b_site_code")] = v("site_b_site_code_value")  # mapped to nothing: dropped
    df[c("batch_type")] = v("type_clinical")  # a correction forces this to the commercial type
    df[c("manufacturing_site")] = f"{names.site('b')} plant"
    df[c("disposition")] = _cycle([v("released"), v("discarded")], n)  # a level A never uses
    df[c("pre_activation_clumps_yn")] = _cycle(["Y", "N", None], n)
    df[c("cells_per_bag")] = rng.uniform(3e12, 7e12, n).round(0)
    df[c("ordinal_parameter")] = _cycle(list(ORDINAL_ORDER), n)
    df[c("site_b_comment")] = "cmd note"  # not a catalog parameter: a site B-only column

    # The count that is really a date Excel mangled.
    df.loc[0, c("clumps_mix_count")] = v("mangled_count")
    return df


def parameter_mapping() -> pd.DataFrame:
    """Site B's parameter names against their site A equivalents."""
    rows = [(theirs, ours) for ours, theirs in site_b_names().items()]
    rows += [
        (c("site_b_country"), c("country")),  # a second column claiming a taken name
        (c("site_b_site_code"), None),  # no site A equivalent: dropped
        (c("excluded_mapping_row"), c("nonconformance")),  # known bad entry
    ]
    return pd.DataFrame(rows, columns=[c("map_site_b_name"), c("map_site_a_name")])


def coa_columns() -> tuple[str, ...]:
    """The certificate columns, in the order the workbook holds them.

    The first is the lot number, which the loader renames to the vector lot.
    """
    car, activity = c("coa_car_metric"), c("coa_ifng_metric")
    return (
        c("coa_lot"),
        c("coa_available"),
        c("coa_site_format"),
        c("coa_titer"),
        c("coa_titer_unadjusted"),
        c("coa_p24"),
        c("coa_pi_ratio"),
        c("coa_donor", metric=car, donor=1),
        c("coa_donor", metric=car, donor=2),
        c("coa_donor", metric=activity, donor=1),
        c("coa_donor", metric=activity, donor=2),
        c("coa_ph"),
        c("coa_osmolality"),
        c("coa_volume"),
        c("coa_dna_size"),
        *cs("coa_impurities"),
        c("coa_made"),
    )


def lv_coa() -> pd.DataFrame:
    """Vector certificates, written the way the source workbook writes them.

    Numbers arrive as European-locale text, percentages as percent strings, and
    an impurity below the limit of quantitation as ``<0,5`` — all three are
    things the loader has to undo. ``LV-104`` has no certificate yet and is
    filtered out; ``LV-909`` never appears, so the batches built from it have no
    certificate at all.
    """
    yes, a_format, b_format = (
        v("coa_available_yes"),
        f"{names.site('a')}/250mL",
        (f"{names.site('b')}/500mL"),
    )
    rows = [
        [
            "LV-101",
            yes,
            a_format,
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
            yes,
            b_format,
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
            yes,
            b_format,
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
            a_format,
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
    df = pd.DataFrame(rows, columns=list(coa_columns()))
    df.insert(2, 1, np.nan)  # a section marker column, which the loader drops
    return df


def remanufacturing(n: int = SITE_B_BATCHES) -> pd.DataFrame:
    """Prior lines of therapy, keyed by site B's batch number."""
    prefix = v("batch_prefix_b")
    return pd.DataFrame(
        {
            cs("remfg_batch_keys")[0]: [f"{prefix}{i:03d}" for i in range(n)] + [None],
            c("prior_lines_source"): list(range(1, n + 1)) + [None],
        }
    )


def raw_materials(first: int = SITE_A_BATCHES, second: int = SITE_B_BATCHES) -> pd.DataFrame:
    """The consumables export, with the column names its query tool produces."""
    a, b = v("batch_prefix_a"), v("batch_prefix_b")
    batches = [f"{a}{i:03d}" for i in range(first)] + [f"{b}{i:03d}" for i in range(second)]
    batches[0] = v("bad_lot")
    # A lot lasts six batches in date order, and both lots change on the same ones:
    # what the shifts report groups. Dated as _timestamps dates them.
    place = [
        sorted(range(n), key=lambda i: (i % 2, i % 20, i % 60)).index(i)
        for n in (first, second)
        for i in range(n)
    ]
    return pd.DataFrame(
        {
            cs("raw_materials_batch_keys")[1]: batches,
            f"Query[{c('media_lot')}]": [f"MED-{100 + k // 6 % 4}" for k in place],
            f"Query[{c('cytokine_lot')}]": [f"IL2-{200 + k // 6 % 3}" for k in place],
        }
    )


def investigations(n: int = 12) -> pd.DataFrame:
    """The investigations team's Power Query workbook.

    Only the catalog stage reads it, and only for its column names. The query tool
    wraps each parameter in ``Query[...]``; some of them are catalog parameters and
    some are the team's own working columns, which is exactly the distinction
    that stage is there to draw.
    """
    return pd.DataFrame(
        {
            f"Query[{BATCH}]": [f"{v('batch_prefix_a')}{i:03d}" for i in range(n)],
            f"Query[{c('harvest_lactate')}]": np.linspace(1.5, 3.0, n).round(2),
            f"Query[{c('investigation_id')}]": [f"INV-{i:03d}" for i in range(n)],
            f"Query[{c('root_cause')}]": _cycle(["Operator", "Equipment", "Material"], n),
            c("days_to_close"): _cycle([7, 14, 30], n),
        }
    )


def site_mappings() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two acronym files, as they arrive from two different meetings."""
    acronym, institution = c("site_map_acronym"), c("site_map_institution")
    first = pd.DataFrame({acronym: vs("acronyms_first"), institution: vs("institutions_first")})
    second = pd.DataFrame({acronym: vs("acronyms_second"), institution: vs("institutions_second")})
    return first, second


def _value_type(parameter: str) -> str:
    # An ordinal parameter's value type is its categories, in order. That is how
    # the catalog says "this is a rank, not a set of labels".
    if parameter == c("ordinal_parameter"):
        return "[" + ", ".join(ORDINAL_ORDER) + "]"
    if parameter in {c(key) for key in DATETIME_KEYS} | {c("coa_prefix") + c("coa_made")}:
        return "datetime"
    if parameter in {c(key) for key in DURATION_KEYS}:
        return "duration"
    if parameter in {c(key) for key in RATIO_KEYS}:
        return "ratio"
    categorical = {
        BATCH,
        "Site Merged",
        c("coa_prefix") + c("coa_site_format"),
        *cs("clump_bags")[:3],
        *(c(key) for key in CATEGORICAL_KEYS),
    }
    return "categorical" if parameter in categorical else "numeric"


def _stage(parameter: str) -> str:
    """The made-up catalog's 'Available At' column: a category a task can select by."""
    if parameter in feature_parameters() or parameter == temperature_reading():
        return ""
    if parameter in (BATCH, "Site Merged") or parameter.startswith(c("coa_prefix")):
        return "D0"
    if parameter == co2_reading():
        return "D6"
    if parameter in cs("clump_bags"):
        return "D10"
    for stage, keys in STAGES.items():
        if parameter in {c(key) for key in keys}:
            return stage
    return ""


def catalog(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """The catalog: every parameter the batch table may hold, and the type it holds.

    Built from what the sources carry, minus the few things left out on purpose —
    the site comment columns, which the catalog has never accepted, and the two
    features in :func:`unlisted_features` — plus parameters nothing supplies, so
    the report has a catalog gap to show.
    """
    to_first = {theirs: ours for ours, theirs in site_b_names().items()}
    to_first[c("site_b_country")] = c("country")
    mapped = dict.fromkeys(to_first.get(column, column) for column in second.columns)
    dropped = (c("coa_available"), c("coa_titer_unadjusted"))
    certificates = [
        c("coa_prefix") + column for column in coa_columns()[2:] if column not in dropped
    ]

    parameters = dict.fromkeys(
        [
            *first.columns,
            *mapped,
            c("vector_lot"),
            c("prior_lines"),
            c("clinical_site_id"),
            c("media_lot"),
            c("cytokine_lot"),
            *certificates,
            *feature_parameters(),
            *orphan_parameters(),
        ]
    )
    for absent in (
        c("site_a_comment"),
        c("site_b_comment"),
        c("site_b_site_code"),
        c("site_b_country"),
        *unlisted_features(),
    ):
        parameters.pop(absent, None)

    return pd.DataFrame(
        {
            "Parameter": list(parameters),
            "Value Type": [_value_type(p) for p in parameters],
            "Available At": [_stage(p) for p in parameters],
        }
    )


def demo_corrections() -> list[corrections.Rule]:
    """The demo's known data errors, as the Corrections sheet of its vocabulary holds them."""
    a, b = names.site("a"), names.site("b")

    def pairs(key: str) -> list[list[str]]:
        return [entry.split("|", 1) for entry in vs(key)]

    shift = "stored without the 1e6 divisor"
    return [
        corrections.Rule(
            site=b,
            column=c("clumps_mix_count"),
            action="set",
            value=None,
            name="count_date_serial",
            when_equals=v("mangled_count"),
            reason="a count Excel turned into a date serial",
        ),
        corrections.Rule(
            site=b,
            column=c("country"),
            action="set",
            value=v("foreign_country"),
            name="site_country",
            when_column=c("clinical_site"),
            when_equals=v("site_b_foreign_code"),
            reason="the clinical site is abroad; the export has the wrong country",
        ),
        corrections.Rule(
            site=b,
            column=c("batch_type"),
            action="set",
            value=v("type_commercial"),
            name="site_b_batch_type",
            reason="site B's export does not fill the batch kind",
        ),
        corrections.Rule(
            site=b,
            column=c("oos_type"),
            action="replace_text",
            value=pairs("oos_fixes_b"),
            name="site_b_oos_spelling",
            reason="site B spells its out-of-spec classes its own way",
        ),
        corrections.Rule(
            site=a,
            column=c("oos_type"),
            action="replace_text",
            value=pairs("oos_fixes_a"),
            name="site_a_oos_spelling",
            reason="one spelling for one out-of-spec class",
        ),
        corrections.Rule(
            site=a,
            column=c("pre_activation_clumps_yn"),
            action="set",
            value="No",
            name="bad_lot_clumps",
            when_column=BATCH,
            when_equals=v("bad_lot"),
            reason="a known bad cell for one lot",
        ),
        corrections.Rule(
            site=a,
            column=c("mycoplasma"),
            action="replace_values",
            value={v("mycoplasma_raw"): v("mycoplasma_clean")},
            name="site_a_contaminant_spelling",
            reason="one spelling for a negative result",
        ),
        corrections.Rule(
            site=b,
            column=c("nonconformance"),
            action="replace_values",
            value=dict([vs("status_fix")]),
            name="site_b_status_spelling",
            reason="one spelling for one status",
        ),
        corrections.Rule(b, c("cells_per_bag"), "divide_by", 1e6, reason=shift),
        corrections.Rule(b, c("cells_for_recovery"), "divide_by", 1e6, reason=shift),
        corrections.Rule(b, c("formulated_concentration_a"), "divide_by", 1e6, reason=shift),
        corrections.Rule(b, c("formulated_concentration_b"), "divide_by", 1e6, reason=shift),
        corrections.Rule(a, c("seeding_density_a"), "divide_by", 1e6, reason=shift),
        corrections.Rule(a, c("seeding_density_b"), "divide_by", 1e6, reason=shift),
    ]


#: Each source's file in the demo's stand-in share, by its config label.
FILES = {
    "catalog": "catalog.xlsx",
    "phf_site_a": "site_a_batch_records.xlsx",
    "phf_site_b": "site_b_batch_records.xlsx",
    "param_mapping": "parameter_mapping.xlsx",
    "remfg": "remfg.xlsx",
    "lv_coa": "lv_coa.xlsx",
    "raw_materials": "raw_materials.csv",
    "investigations": "investigations_power_query.xlsx",
    "vocabulary": "vocabulary.xlsx",
    "screening": "parameter_screening.xlsx",
}
SITE_MAP_FILES = ("site_map_first.xlsx", "site_map_second.xlsx")


def write_sources(directory: Path, seed: int = SEED) -> dict[str, object]:
    """Write one complete set of sources into a folder, and return their paths.

    Each workbook's data goes on the sheet the vocabulary in use names for it; the
    consumables export is a CSV file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    first = site_a_phf(rng)
    second = site_b_phf(rng)
    first_map, second_map = site_mappings()
    paths: dict[str, object] = {label: directory / name for label, name in FILES.items()}
    paths["site_maps"] = [directory / name for name in SITE_MAP_FILES]

    def sheet(label: str) -> str:
        found = names.sheet(label, "Sheet1")
        return "Sheet1" if isinstance(found, int) else str(found)

    catalog(first, second).to_excel(paths["catalog"], sheet_name=sheet("catalog"), index=False)
    first.to_excel(paths["phf_site_a"], sheet_name=sheet("phf_site_a"), index=False)
    second.to_excel(paths["phf_site_b"], sheet_name=sheet("phf_site_b"), index=False)
    parameter_mapping().to_excel(
        paths["param_mapping"], sheet_name=sheet("param_mapping"), index=False
    )
    remanufacturing().to_excel(paths["remfg"], sheet_name=sheet("remfg"), index=False)
    # Two rows of workbook furniture above the header, as the real file has.
    lv_coa().to_excel(paths["lv_coa"], sheet_name=sheet("lv_coa"), index=False, startrow=2)
    # A CSV export, as the real one is: semicolon-separated, with a byte-order mark.
    raw_materials().to_csv(paths["raw_materials"], sep=";", index=False, encoding="utf-8-sig")
    investigations().to_excel(
        paths["investigations"], sheet_name=sheet("investigations"), index=False
    )
    for path, frame in zip(paths["site_maps"], (first_map, second_map)):
        frame.to_excel(path, sheet_name=sheet("site_maps"), index=False)
    with pd.ExcelWriter(paths["vocabulary"]) as workbook:
        names.to_frame(names.current()).to_excel(
            workbook, sheet_name=names.NAMES_SHEET, index=False
        )
        corrections.to_frame(demo_corrections()).to_excel(
            workbook, sheet_name=names.CORRECTIONS_SHEET, index=False
        )

    return {
        key: [str(p) for p in value] if isinstance(value, list) else str(value)
        for key, value in paths.items()
    }


def screened_parameter() -> str:
    return c("mixing_cycles")  # screened out at site A: the expression task leaves it out


def screening_decisions() -> list[screening.Decision]:
    """The demo's register: one parameter excluded at site A, as `ml screening add` writes it."""
    return [
        screening.Decision(
            id="S-0001",
            date="2026-09-01",
            author="Demo reviewer",
            action=screening.EXCLUDE,
            parameter=screened_parameter(),
            site=names.site("a"),
            group="",
            reason="irregular",
            rationale="recorded inconsistently between shifts",
            missing="0.0",
            pattern="Complete",
            last_measured="",
            batch_table="",
            review_by="2026-12-01",
            report=screening.MISSINGNESS,
        )
    ]


#: The drive the demo's stand-in share holds everything on.
DRIVE = "DEMO_DRIVE"
#: Where each demo file is on that drive, as config.yaml writes a SharePoint path.
FOLDERS = {"raw": "Data/raw/", "catalog": "Data/catalog/", "batch_data": "Data/batch_data/"}


def standin(directory: Path) -> Path:
    """The folder that plays SharePoint for the demo written into ``directory``."""
    return Path(directory).resolve() / "sharepoint"


def write_config(
    directory: Path, seed: int = SEED, vocabulary: names.Vocabulary | None = None
) -> Path:
    """Write the demo data, config and task workspaces into ``directory``; return the config.

    The sources and the vocabulary go onto the stand-in share
    (``sharepoint/DEMO_DRIVE/Data/raw``), the catalog into ``Data/catalog`` and
    ``ml build`` writes into ``Data/batch_data`` — the layout the real share has.
    Each task goes to its own workspace under ``tasks/``. ``vocabulary`` is the
    one the demo is written in (default: :func:`ml.names.demo`).

    The thresholds here are the ones that suit a cohort this small: with forty
    batches, a parameter with a dozen categories is already high-cardinality, and
    a feature needs fewer values before it is worth keeping.
    """
    names.use(vocabulary or names.demo())
    directory = Path(directory)
    drive = standin(directory) / DRIVE
    sources = write_sources(drive / FOLDERS["raw"], seed=seed)
    (drive / FOLDERS["catalog"]).mkdir(parents=True, exist_ok=True)
    Path(sources["catalog"]).replace(drive / FOLDERS["catalog"] / FILES["catalog"])
    screening.write(screening_decisions(), drive / FOLDERS["catalog"] / FILES["screening"])

    def at(folder: str, name: str) -> dict[str, object]:
        return {"path": f"{FOLDERS[folder]}{name}", "drive_id": DRIVE}

    sharepoint = {
        "vocabulary": at("raw", FILES["vocabulary"]),
        "catalog": at("catalog", FILES["catalog"]),
        "batch_table": at("batch_data", "batch_data.xlsx"),
        "screening": at("catalog", FILES["screening"]),
        **{
            label: at("raw", FILES[label])
            for label in (
                "phf_site_a",
                "phf_site_b",
                "param_mapping",
                "remfg",
                "lv_coa",
                "raw_materials",
                "investigations",
            )
        },
        "site_maps": {
            "folder": FOLDERS["raw"],
            "files": list(SITE_MAP_FILES),
            "drive_id": DRIVE,
        },
    }
    (directory / ".env").write_text(
        "# The demo has no SharePoint: this folder stands in for it.\n"
        f"ML_SHAREPOINT_STANDIN={standin(directory)}\n",
        encoding="utf-8",
    )

    config = {
        "seed": seed,
        "sources": {"databricks": False},
        "sharepoint": sharepoint,
        "catalog": {"write_new_parameters": True},
        "build": {
            "write_parquet": True,
            "date_column": "incubator_in_d0",
            # Commercial batches only; every batch at site B is one.
            "missingness_cohort": {
                "column": "batch_type",
                "value": "type_commercial",
                "all_batches_at": ["b"],
                "analyse_only": {"column": "manufacturing_completed", "value": "completed_yes"},
            },
        },
        "cleaning": {
            "censored": "limit",
            "censored_flags": False,
            "duration_unit": "minutes",
            "yes_no_defaults": {"pre_activation_clumps_yn": "No"},
        },
        "mapping": {"exclude": ["excluded_mapping_row"]},
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
            "seeding_target_per_vessel": 50_000_000.0,
            "shift_cutoff": "11:59",
            # Process facts the made-up product has, so the features that need
            # them are built rather than blocked.
            "moi_target": 0.5,
            "medium_glucose": 4.5,
            "medium_lactate": 0.1,
            "impurity_spec_limits": [5.0, 10.0, 5.0, 3.0, 3.0],
        },
        # Spec midpoints for the made-up product, so the two-sided deviation
        # features are built rather than skipped.
        "lv_coa": {"ph_nominal": 7.2, "osmo_nominal": 300.0},
    }

    config_path = directory / "config.yaml"
    config_path.write_text(
        "# The demo config, written by `python -m ml.demo`. Every source is made up, and\n"
        "# the SharePoint drive is the folder sharepoint/ (see .env).\n"
        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_tasks(directory)
    return config_path


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

#: The made-up catalog's stages after harvest, which the site A demo task leaves out.
LATER_THAN_HARVEST = [
    "Wash",
    "Formulation",
    "Final product",
    "Post-thaw",
    "Release",
    "Post-release",
]

#: The demo's task workspaces, relative to the config.
TASK_FILES = ("tasks/expression", "tasks/release")


def write_tasks(directory: Path) -> list[Path]:
    """Two task workspaces over the same batch table, differing only in what their YAML says.

    The point of the pair is that nothing else changes: same package, same
    commands, same code. Cohort, target, roles, columns and clustering action
    all move in YAML.
    """
    directory = Path(directory)
    a, b = names.site("a"), names.site("b")
    expression = {
        "config": "../../config.yaml",
        "task": {
            "name": "expression",
            "question": f"Which manufacturing parameters go with final product expression at {a}?",
            "seed": SEED,
        },
        "cohort": {
            "sites": [a],
            "filters": [
                {
                    "column": c("batch_type"),
                    "op": "equals",
                    "value": v("type_commercial"),
                    "reason": "manufactured routinely",
                },
                {
                    "column": c("testing_done"),
                    "op": "equals",
                    "value": "Y",
                    "reason": "manufacturing and testing finished",
                },
                {
                    "column": c("nonconformance"),
                    "op": "not_in",
                    "value": vs("excluded_statuses"),
                    "normalise": True,
                    "missing": "include",
                    "reason": "not stopped, however it is spelled",
                },
            ],
        },
        "target": {"column": target(), "type": "numeric"},
        "roles": {"id": BATCH, "group": c("vector_lot")},
        "predictors": {
            "exclude": [
                {
                    "column": c("subject_weight"),
                    "reason": "dose per kilogram already carries it",
                },
                {
                    "column": c("washed_cells"),
                    "reason": "counted after the wash",
                },
            ],
            # Everything the catalog says is measured after harvest.
            "exclude_by_catalog": {"Available At": LATER_THAN_HARVEST},
        },
        **_SMALL_COHORT,
        "clustering": {
            "action": "describe",
            "method": "spearman",
            "threshold": 0.7,
            "min_overlap": 8,
        },
        "report": {"preview_rows": 15},
    }

    release = {
        "config": "../../config.yaml",
        "task": {
            "name": "release",
            "question": f"What does an accepted {b} batch look like, next to a scrapped one?",
            "seed": SEED,
        },
        "cohort": {
            "sites": [b],
            "filters": [
                {
                    "column": c("testing_done"),
                    "op": "equals",
                    "value": "Y",
                    "reason": "manufacturing and testing finished",
                }
            ],
        },
        "target": {
            "column": c("disposition"),
            "type": "binary",
            "positive_class": v("released"),
            "negative_class": v("discarded"),
        },
        "roles": {"id": BATCH, "group": c("vector_lot")},
        "predictors": {
            "exclude": [{"column": c("oos_type"), "reason": "it is the reason for the outcome"}],
            # Whole categories of the catalog's metadata can be left out by name.
            "exclude_by_catalog": {"Available At": ["Post-release"]},
        },
        **_SMALL_COHORT,
        # Site B's cohort is just big enough to show the other two encoders,
        # so this task lowers the sample-size bars they normally have to clear.
        "encoding": {
            **_SMALL_COHORT["encoding"],
            "target_min_rows_per_category": 3,
            "hashing_min_rows": 20,
            "missing_indicators": True,
        },
        "clustering": {
            # Each stable group becomes one score, named after the process stage
            # its members share in the catalog.
            "action": "summarise",
            "name_by": "Available At",
            "method": "spearman",
            "threshold": 0.8,
            "min_overlap": 8,
        },
        "report": {"preview_rows": 10},
    }

    written = []
    for name, task in zip(TASK_FILES, (expression, release)):
        path = directory / name / "task.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# A demo task workspace written by `python -m ml.demo`. Its runs go to runs/.\n"
            + yaml.safe_dump(task, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append(path)
    return written


if __name__ == "__main__":
    target_folder = Path(sys.argv[1] if len(sys.argv) > 1 else "demo")
    write_config(target_folder)
    print(
        f"The demo is in {target_folder}/. Now run:  cd {target_folder} && ml update "
        + " ".join(f"--task {name}" for name in TASK_FILES)
    )
