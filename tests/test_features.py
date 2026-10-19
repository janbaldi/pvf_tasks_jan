"""The feature formulas, one hand-computed case per group.

The end-to-end build proves the registry runs over the demo's sources; these
prove each formula gives the number it says it does, including what it does
with a missing input. Most of the unit and sign problems a feature can have are
visible only like this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import features, names
from ml.names import col, cols, values

PARAMS = features.Params(
    moi_target=0.5,
    medium_glucose=4.5,
    medium_lactate=0.5,
    impurity_spec_limits=(10.0, 10.0, 10.0, 10.0, 10.0),
)


def compute(frame: pd.DataFrame, params: features.Params = PARAMS) -> pd.DataFrame:
    """Every feature the frame supports, as if the catalog listed all of them."""
    df = frame.copy()
    catalog = [f.name for f in features.registry(params, df.columns)]
    features.apply(df, catalog, params)
    return df


def one(**columns) -> pd.DataFrame:
    """A frame from vocabulary keys (or column names, as they are): a list per column."""
    return pd.DataFrame({col(k) if k in names.COLUMNS else k: v for k, v in columns.items()})


def test_cd3_cells_take_the_percentage_as_a_percentage():
    df = compute(one(pt_viable_cells=[1e8], pt_cd3_pct=[80.0], pt_cd4_pct=[50.0]))
    assert df[col("f_pt_cd3_cells")].iloc[0] == pytest.approx(8e7)
    assert df[col("f_pt_cd4_share")].iloc[0] == pytest.approx(40.0)
    assert df[col("f_pt_cd4_cells")].iloc[0] == pytest.approx(4e7)


def test_metabolites_per_cell_follow_the_vessels_and_refuse_a_third():
    df = compute(
        one(
            vessel_count=[1, 2, 3],
            vsvg_cells=[1e8, 1e8, 1e8],
            d6_lactate_a=[1.0, 1.0, 1.0],
            d6_lactate_b=[np.nan, 0.5, 0.5],
            harvest_lactate=[2.0, 2.0, 2.0],
        )
    )
    d6, d10 = cols("f_lactate_per_cell")[0], cols("f_lactate_per_cell")[2]
    assert df[d6].tolist()[:2] == pytest.approx([10.0, 15.0])
    assert np.isnan(df[d6].iloc[2])
    # The pooled harvest sample counts once per vessel.
    assert df[d10].tolist()[:2] == pytest.approx([20.0, 40.0])


def test_the_harvest_duplicate_is_not_rebuilt_and_the_growth_profile_does_without_it():
    df = compute(
        one(vessel_count=[1, 1], vsvg_cells=[1e8, 1e8], harvest_lactate=[0.5, 5.0], cpdl=[4.5, 5.5])
    )
    assert col("f_harvest_lactate_norm") not in df.columns
    under, over, _ = values("growth_profile_classes")
    assert df[col("f_growth_profile")].tolist() == [under, over]


def test_a_growth_profile_needs_both_inputs():
    df = compute(one(vessel_count=[1], vsvg_cells=[1e8], harvest_lactate=[2.0], cpdl=[np.nan]))
    assert df[col("f_growth_profile")].isna().all()


def test_increments_are_the_change_between_days():
    df = compute(
        one(
            vessel_count=[1],
            vsvg_cells=[1e8],
            d6_lactate_a=[1.0],
            d6_lactate_b=[np.nan],
            d8_lactate_a=[1.5],
            d8_lactate_b=[np.nan],
            harvest_lactate=[2.5],
        )
    )
    first, second = cols("f_lactate_increment")
    assert df[first].iloc[0] == pytest.approx(5.0)
    assert df[second].iloc[0] == pytest.approx(10.0)


def test_lactate_yield_uses_the_fresh_medium():
    df = compute(one(harvest_lactate=[2.5], harvest_glucose=[2.5]))
    # (2.5 - 0.5) / (4.5 - 2.5)
    assert df[col("f_lactate_yield", day=10)].iloc[0] == pytest.approx(1.0)


def test_a_feature_waiting_for_a_setting_is_blocked_not_guessed():
    df = one(harvest_lactate=[2.5], harvest_glucose=[2.5])
    name = col("f_lactate_yield", day=10)
    report = features.apply(df, [name], features.Params())
    entry = next(e for e in report.entries if e["name"] == name)
    assert entry["status"] == features.BLOCKED
    assert "features.medium_glucose" in entry["note"]
    assert name not in df.columns


def test_seeding_is_a_percentage_of_the_target_per_vessel():
    df = compute(one(vessel_count=[1, 2], vsvg_cells=[5e7, 1.1e8]))
    assert df[col("f_seeding_pct")].tolist() == pytest.approx([100.0, 110.0])
    assert df[col("f_seeding_error")].tolist() == pytest.approx([0.0, 10.0])
    assert df[col("f_seeding_within")].tolist() == [1.0, 0.0]


def test_seeding_loss_and_dead_cells():
    df = compute(
        one(
            cells_available=[1e8],
            cells_seeded_a=[4e7],
            cells_seeded_b=[5e7],
            d3_viability=[80.0],
        )
    )
    assert df[col("f_seeding_loss")].iloc[0] == pytest.approx(0.1)
    assert df[col("f_dead_cells_d3")].iloc[0] == pytest.approx(2.5e7)


def test_vessel_balance_is_a_mean_and_an_imbalance():
    df = compute(one(cells_seeded_a=[30.0, 40.0], cells_seeded_b=[10.0, np.nan]))
    mean = col("f_seeded_spread", statistic="mean")
    imbalance = col("f_seeded_spread", statistic="imbalance")
    assert df[mean].tolist() == pytest.approx([20.0, 40.0])
    assert df[imbalance].iloc[0] == pytest.approx(0.5)
    assert np.isnan(df[imbalance].iloc[1])
    assert col("f_seeded_spread", statistic="cv") not in df.columns


def test_moi_against_target_and_particles_per_cell():
    prefix = col("coa_prefix")
    df = compute(
        one(
            vector_titer=[1e7],
            vector_volume_a=[2.0],
            vector_volume_b=[np.nan],
            cells_available=[4e7],
            cells_seeded_a=[4e7],
            **{prefix + col("coa_pi_ratio"): [100.0]},
        )
    )
    assert df[col("f_moi_pool")].iloc[0] == pytest.approx(0.5)
    assert df[col("f_moi_vs_target")].iloc[0] == pytest.approx(1.0)
    assert df[col("f_moi_quality")].iloc[0] == pytest.approx(50.0)


def test_contact_within_window_is_missing_when_the_time_is():
    df = compute(one(contact_time=[100.0, 200.0, np.nan]))
    within = df[cols("f_contact_within")[0]]
    assert within.tolist()[:2] == [1.0, 0.0]
    assert np.isnan(within.iloc[2])


def test_handling_times():
    df = compute(
        one(
            thaw_time=["08:30", "23:00"],
            incubator_in_d0=["2026-01-05 09:15:00", "2026-01-05 09:15:00"],
            incubator_out_d3=["2026-01-08 09:00:00", "2026-01-08 09:00:00"],
            incubator_back_d3=["2026-01-08 11:30:00", "2026-01-08 08:00:00"],
        )
    )
    assert df[col("f_thaw_to_incubator")].iloc[0] == pytest.approx(45.0)
    assert np.isnan(df[col("f_thaw_to_incubator")].iloc[1])  # after the incubator: unresolvable
    outside = df[cols("f_out_of_incubator")[0]]
    assert outside.iloc[0] == pytest.approx(150.0)
    assert np.isnan(outside.iloc[1])  # back before it was out


def test_the_shift_follows_the_configured_cut_off_and_skips_a_missing_time():
    df = compute(one(thaw_time=["08:30", "13:00", None]))
    assert df[col("f_shift")].tolist()[:2] == ["Morning", "Afternoon"]
    assert pd.isna(df[col("f_shift")].iloc[2])


def test_dose_basis_says_which_dose_the_record_holds():
    df = compute(one(dose_per_kg=[2e6, np.nan, np.nan], dose_cells=[np.nan, 1e8, np.nan]))
    weight_based, fixed = values("dose_basis_classes")[:2]
    assert df[col("f_dose_basis")].tolist()[:2] == [weight_based, fixed]
    assert pd.isna(df[col("f_dose_basis")].iloc[2])
    assert df[col("f_dose_all")].tolist()[:2] == pytest.approx([2.0, 1.0])


def test_wash_recovery():
    df = compute(one(harvest_cells=[1e9], washed_cells=[8e8]))
    assert df[col("f_wash_recovery")].iloc[0] == pytest.approx(0.8)
    assert df[col("f_cells_lost_wash")].iloc[0] == pytest.approx(2e8)


def test_the_spec_index_needs_no_other_batch():
    prefix = col("coa_prefix")
    impurities = {prefix + name: [5.0] for name in cols("coa_impurities")}
    df = compute(pd.DataFrame(impurities))
    assert df[col("f_coa_impurity_spec_index")].iloc[0] == pytest.approx(2.5)
    # The z-score version cannot say anything about one lot on its own.
    assert np.isnan(df[col("f_coa_impurity_index")].iloc[0])


def test_vector_age_in_days():
    prefix = col("coa_prefix")
    df = compute(one(date_start=["2026-01-11"], **{prefix + col("coa_made"): ["2026-01-01"]}))
    assert df[col("f_vector_age")].iloc[0] == pytest.approx(10.0)


def test_a_feature_known_before_its_input_is_reported():
    params = features.Params(availability_column="Available From Day")
    df = one(cells_available=[1e8], culture_bags=[2])
    name = col("f_cells_per_bag")
    catalog = pd.DataFrame(
        {
            "Parameter": [name, col("cells_available"), col("culture_bags")],
            "Available From Day": [0, 3, 3],
        }
    )
    report = features.apply(df, [name], params, known=features.availability(catalog, params))
    assert any(c["check"] == "Availability" and name in c["detail"] for c in report.checks)


def test_a_feature_known_with_its_inputs_is_not_reported():
    params = features.Params(availability_column="Stage", availability_order=("D0", "D3"))
    df = one(cells_available=[1e8], culture_bags=[2])
    name = col("f_cells_per_bag")
    catalog = pd.DataFrame({"Parameter": [name, col("cells_available")], "Stage": ["D3", "D3"]})
    report = features.apply(df, [name], params, known=features.availability(catalog, params))
    assert not [c for c in report.checks if c["check"] == "Availability"]


def test_the_quality_checks_flag_impossible_arithmetic():
    df = one(
        cpdl=[5.0, 5.0],
        harvest_cells=[3.2e9, 3.2e9],
        vsvg_cells=[1e8, 1e7],
        cells_available=[1e8, 1e8],
        cells_seeded_a=[6e7, 4e7],
        cells_seeded_b=[6e7, 4e7],
    )
    found = {c["check"]: c["batches"] for c in features.quality_checks(df)}
    # log2(3.2e9 / 1e8) = 5, log2(3.2e9 / 1e7) ≈ 8.3
    assert found["Doublings against cell counts"] == 1
    assert found["Seeding arithmetic"] == 1


def test_every_feature_says_what_kind_of_calculation_it_is():
    for feature in features.registry(PARAMS):
        assert feature.kind in (features.ENGINEERED, features.NORMALISATION), feature.name
        if feature.value_type == "numeric":
            assert feature.unit, feature.name
