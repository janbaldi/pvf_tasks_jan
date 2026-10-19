# Calculated parameters / features

Every column in the batch table that is computed from other columns, grouped by
what it describes. Each entry gives the formula and a short description.

**Sources read** (branch `main`, commit `507349b`):
`src/ml/features.py` (the feature registry), `src/ml/io.py` (`load_lv_coa`,
vector certificate columns), `src/ml/clean.py` (censored flags),
`src/ml/dataset.py`, `src/ml/decorrelate.py`, `src/ml/encode.py` (task-stage columns).

**Conventions**

- Names in backticks are *vocabulary keys* (e.g. `pt_cd4_pct`), not the real column
  names. The real names live in the private vocabulary workbook; `vocabulary.csv` in
  the repo shows the display names for most keys. Feature keys start with `f_`.
- "Process day" is the day the feature is attached to in the registry (D0, D1, D3, D6, D8, D10).
- A feature is only created if its name is a catalog parameter and all of its inputs
  exist for that site. Otherwise it is reported as *not in catalog*, *blocked*, or
  *present* (a source already supplies it and the feature stands aside).
- `safe_div(a, b)` = `a / b`, with ±∞ (division by zero) turned into NaN.
- Units are taken from the vocabulary display names. Where a unit factor looks odd
  or I could not confirm what it is meant to do, the entry says **not sure**.
- Config thresholds come from `config.yaml` → `features:` (defaults in brackets).

---

## Contents

1. [Flow cytometry and phenotype](#1-flow-cytometry-and-phenotype-d0)
2. [Metabolites (lactate and glucose)](#2-metabolites-lactate-and-glucose)
3. [Profiles (categorical classes)](#3-profiles-categorical-classes)
4. [Viability changes](#4-viability-changes)
5. [Seeding and vessel balance](#5-seeding-and-vessel-balance-d3)
6. [MOI (multiplicity of infection)](#6-moi-multiplicity-of-infection-d3)
7. [Contact time](#7-contact-time-d3)
8. [Environment (CO₂ and temperature setpoints)](#8-environment-co₂-and-temperature-setpoints)
9. [Incubation durations](#9-incubation-durations)
10. [Clumps](#10-clumps)
11. [Dose and CAR](#11-dose-and-car)
12. [Formulation and wash](#12-formulation-and-wash)
13. [IL-2](#13-il-2)
14. [Context](#14-context)
15. [Vector certificate of analysis (CoA)](#15-vector-certificate-of-analysis-coa)
16. [Censoring flags (cleaning stage)](#16-censoring-flags-cleaning-stage)
17. [Task-stage derived columns](#17-task-stage-derived-columns)
18. [Open questions / "not sure" summary](#18-open-questions--not-sure-summary)

---

## 1. Flow cytometry and phenotype (D0)

| Feature (key) | Formula | Description |
|---|---|---|
| `f_log_pt_cd4_cd8` | `log10(pt_cd4_pct / pt_cd8_pct)` | CD4:CD8 balance of the incoming (post-thaw, "PT") material on a log scale, so a 2-fold shift reads the same in either direction. NaN/inf if CD8 is 0. |
| `f_log_pw_cd4_cd8` | `log10(pw_cd4_pct / pw_cd8_pct)` | Same, after enrichment/selection ("PW"/"PE"). |
| `f_pt_cd3_cells` | `pt_viable_cells × pt_cd3_pct` | Absolute CD3+ cells post thaw. **Not sure** whether `pt_cd3_pct` is stored as a fraction (0–1) or as a percentage (0–100): the code multiplies without `/100`, unlike the rows below. |
| `f_pt_cd4_share` | `pt_cd4_pct × pt_cd3_pct / 100` | CD4+ as a share of *all* cells rather than of the CD3+ gate (post thaw). |
| `f_pt_cd4_cells` | `f_pt_cd4_share × pt_viable_cells` | Absolute CD3+CD4+ cells post thaw. Depends on `f_pt_cd4_share`. **Not sure** about the % vs. fraction units (the share is in % here). |
| `f_pw_cd4_share` | `pw_cd4_pct × pw_cd3_pct / 100` | CD4+ as a share of all cells after enrichment. |
| `f_pw_cd8_share` | `pw_cd8_pct × pw_cd3_pct / 100` | CD8+ as a share of all cells after enrichment. |

## 2. Metabolites (lactate and glucose)

`vessel_count` is the number of G-Rex vessels seeded (1 or 2). `vsvg_cells` is the
number of cells used for expansion (cells). `1e9` is applied to convert the
concentration (g/L) and the cell count (cells) to "per million (E6) cells".
**Not sure** about the exact unit logic of the `1e9` factor (it is `1e3` mL→L
style scaling × `1e6` for "per E6"); the code does not document it.

| Feature (key) | Formula | Description |
|---|---|---|
| `f_harvest_lactate_norm` | `(harvest_lactate × 1e9) × vessel_count / vsvg_cells` | Harvest (D10) lactate per cell seeded, so batches seeded into one or two G-Rex can be compared. |
| `f_lactate_per_cell` (D6) | 1 vessel: `d6_lactate_a × 1e9 / vsvg_cells`<br>2 vessels: `(d6_lactate_a + d6_lactate_b) × 1e9 / vsvg_cells` | Lactate per E6 cells seeded at D6, summed over the vessels in use. Any other `vessel_count` falls into the 2-vessel branch (**not sure** that is intended). |
| `f_lactate_per_cell` (D8) | same with `d8_lactate_a/b` | Same at D8. |
| `f_lactate_per_cell` (D10) | 1 vessel: `harvest_lactate × 1e9 / vsvg_cells`<br>otherwise: `harvest_lactate × 2e9 / vsvg_cells` | D10 version; the harvest is sampled once, so the value is counted twice for two vessels. |
| `f_glucose_per_cell` (D6) | 1 vessel: `d6_glucose_a × 1e9 / vsvg_cells`<br>2 vessels: `(d6_glucose_a + d6_glucose_b) × 1e9 / vsvg_cells` | Glucose remaining per E6 cells seeded at D6. |
| `f_glucose_per_cell` (D8) | same with `d8_glucose_a/b` | Same at D8. |
| `f_glucose_per_cell` (D10) | 1 vessel: `harvest_glucose × 1e9 / vsvg_cells`<br>otherwise: `harvest_glucose × 2 × 1e9 / vsvg_cells` | Same at D10. |
| `f_lactate_slope` | least-squares slope of `f_lactate_per_cell` (D6, D8, D10) against day (6, 8, 10) | How fast normalised lactate climbs across expansion. Needs ≥ 2 readings, otherwise NaN. |
| `f_lactate_auc` | trapezoidal area under `f_lactate_per_cell` over days 6/8/10 | Total normalised lactate exposure. A gap is bridged, not treated as 0. |
| `f_glucose_slope` | least-squares slope of `f_glucose_per_cell` over days 6/8/10 | How fast normalised glucose falls. |
| `f_glucose_auc` | trapezoidal area under `f_glucose_per_cell` over days 6/8/10 | Total normalised glucose exposure. |
| `f_lactate_glucose_ratio` (day = 6, 8, 10) | `safe_div(f_lactate_per_cell[day], f_glucose_per_cell[day])` | Lactate against glucose on the same day; separates cells burning sugar fast from cells simply fed more. |

## 3. Profiles (categorical classes)

| Feature (key) | Formula | Description |
|---|---|---|
| `f_growth_profile` | `class 1` if `cpdl < cpdl_low [4.0]` **or** `f_harvest_lactate_norm < lactate_low [12.5]`<br>`class 2` if `cpdl > cpdl_high [5.0]` **and** `f_harvest_lactate_norm > lactate_high [30.0]`<br>else `class 3` | Below / above / within expected growth. Class labels come from the vocabulary (`growth_profile_classes`; the notes in the vocabulary suggest slow / fast / typical). **Not sure** how rows with a missing `cpdl` or lactate are classed: comparisons against NaN are False, so they fall into the default "typical" class. |
| `f_cd4_profile` | `class 1` if `pt_cd4_pct ≤ cd4_low [40]`<br>`class 2` if `pt_cd4_pct ≥ cd4_high [60]`<br>else `class 3`; NaN if `pt_cd4_pct` is missing | CD4 share of the incoming material in three bands (low / high / mid). A missing reading gets no profile. |

## 4. Viability changes

| Feature (key) | Formula | Description |
|---|---|---|
| `f_viability_drop_rel` | `(d3_viability − pw_viability) / pw_viability × 100` | Relative viability change over activation (D0→D3), %. |
| `f_viability_drop_pts` | `d3_viability − pw_viability` | Same, in percentage points. |
| `f_viability_d3_d10_rel` | `(d3_viability − harvest_viability) / d3_viability × 100` | Relative viability *lost* between D3 and harvest, %. |
| `f_viability_expansion` | `(harvest_viability − d3_viability) / d3_viability × 100` | Same comparison with the opposite sign (viability *gain*), matching the Power Query investigation. It is exactly `−f_viability_d3_d10_rel`. |
| `f_viability_d3_d10_pts` | `washed_viability − d3_viability` | Net viability change D3 → washed harvest, percentage points. Note it uses the *washed* viability, not `harvest_viability`. |
| `f_viability_wash` | `(washed_viability − harvest_viability) / harvest_viability × 100` | Effect of the LOVO wash on viability, %. |

## 5. Seeding and vessel balance (D3)

| Feature (key) | Formula | Description |
|---|---|---|
| `f_seeding_pct` | 1 vessel: `vsvg_cells / 500 000`<br>2 vessels: `vsvg_cells / 1 000 000`<br>otherwise NaN | Cells seeded against the target for the number of vessels. The result is a ratio where 1.0 = on target; the next two features compare it with `100.0`. **Not sure** whether this is a bug (ratio vs. percentage) or whether the factor of 100 is applied elsewhere. |
| `f_seeding_error` | `abs(f_seeding_pct − 100)` | Distance from target seeding in either direction. (Same unit caveat as above.) |
| `f_seeding_within` | `1.0 if abs(f_seeding_pct − 100) ≤ seeding_tolerance [5]` else `0.0` | Whether seeding landed within ±5 % of target. (Same unit caveat.) |
| `f_cells_per_bag` | `safe_div(cells_available, culture_bags)` | Cells per culture bag — the density the cells see. |
| `f_seeded_spread` (`statistic` = mean, min, max, std, cv) | over the pair (`cells_seeded_a`, `cells_seeded_b`):<br>`mean`, `min`, `max`, `std(ddof=0)`, `cv = std / mean` | Vessel balance of seeded viable cells. 5 columns, A–B. |
| `f_vector_volume_spread` (`statistic` = mean, min, max, std, cv) | same over (`vector_volume_a`, `vector_volume_b`) | Vessel balance of the vector volume added. 5 columns, A–B. |

For the spread families a missing vessel is skipped by pandas (so one vessel gives
mean = min = max, std = 0). **Not sure** that this is the desired meaning for a
single-vessel batch.

## 6. MOI (multiplicity of infection) (D3)

`titer` = the measured vector titer: the certificate's titer if present
(`coa_prefix + coa_titer`), otherwise the batch record's `vector_titer`.
`pooled_volume` = `vector_volume_a + vector_volume_b` (a missing vessel counts as 0 as
long as the other has a value). `pooled_cells` = `cells_available` if the column exists,
otherwise `cells_seeded_a + cells_seeded_b`.

| Feature (key) | Formula | Description |
|---|---|---|
| `f_moi_a` | `titer × vector_volume_a / cells_seeded_a` | Infectious units per seeded cell in vessel A. |
| `f_log_moi_a` | `log1p(f_moi_a)` = `ln(1 + MOI_A)` | Vessel A MOI on a log scale. |
| `f_moi_b` | `titer × vector_volume_b / cells_seeded_b` | Same, vessel B. |
| `f_log_moi_b` | `log1p(f_moi_b)` | Vessel B MOI on a log scale. |
| `f_moi_pool` | `titer × pooled_volume / pooled_cells` | MOI over the whole batch. |
| `f_log_moi_pool` | `log1p(f_moi_pool)` | Pooled MOI on a log scale. |
| `f_moi_quality` | `f_moi_pool × coa_pi_ratio` (certificate column) | Pooled MOI weighted by the particle (PI) ratio. **Not sure** about the direction: the code comment says "weighted by how many particles in the lot are infectious", but a PI *ratio* is normally particles/infectious unit, which would be the inverse. Check with the vector team. |
| `f_moi_per_cd3` | `titer × pooled_volume / f_pt_cd3_cells` | Infectious units per CD3+ cell (the actual target cells). Inherits the unit caveat of `f_pt_cd3_cells`. |

## 7. Contact time (D3)

Three variants of the same pair of features: the batch's own reading (`contact_time`)
and vessels A and B (`contact_time_a`, `contact_time_b`). `target` = `contact_target [120]` min,
`tolerance` = `contact_tolerance [30]` min.

| Feature (key) | Formula | Description |
|---|---|---|
| `f_contact_deviation` (batch / A / B) | `abs(contact_time[_a/_b] − 120)` | Minutes away from the transduction contact-time target. |
| `f_contact_within` (batch / A / B) | `1.0 if 90 ≤ contact_time[_a/_b] ≤ 150` else `0.0` | Whether the contact time stayed within target ± tolerance. NaN input gives 0.0 (False), not NaN — **not sure** if intended. |

## 8. Environment (CO₂ and temperature setpoints)

Expanded dynamically: one feature for every column of the frame whose name contains the
vocabulary marker `co2_marker` ("CO2 Saturation (%)") or `temperature_marker`
("Temperature (°C)"). The set therefore differs between sites. Process day is read
from the column name (`Day N` → `DN`, or a D0/D1/D3/D6/D8/D10 tag, else "Context").

| Feature | Formula | Description |
|---|---|---|
| `<column> dev_from_5.0pct` | `abs(<CO2 column> − co2_target [5.0])` | Distance from the CO₂ setpoint: how well the incubator held it. |
| `<column> dev_from_37.0C` | `abs(<temperature column> − temp_target [37.0])` | Distance from the temperature setpoint. |

## 9. Incubation durations

Minutes between two timestamps: `(later − earlier).total_seconds() / 60`.

| Feature (key) | Formula | Description |
|---|---|---|
| `f_incubation_d0_d1` | `incubator_out_d1 − incubator_in_d0` | Minutes in the incubator D0→D1. |
| `f_incubation_d1_d3` | `incubator_out_d3 − incubator_in_d1` | D1→D3. |
| `f_incubation_d3_d6` | `incubator_out_d6 − incubator_back_d3` | D3→D6. |
| `f_incubation_d6_d8` | `incubator_out_d8 − incubator_back_d6` | D6→D8. |
| `f_incubation_d8_d10` | `incubator_out_d10 − incubator_back_d8` | D8→D10. |
| `f_incubation_total` | `f_incubation_d3_d6 + f_incubation_d6_d8 + f_incubation_d8_d10` | Total incubator time D3–D10. Plain `+`, so one missing interval makes the total NaN. |

## 10. Clumps

| Feature (key) | Formula | Description |
|---|---|---|
| `f_clump_count_reduction` | `1 − clumps_post_count / clumps_pre_count` | Share of clumps removed by massaging. |
| `f_clump_size_reduction` | `1 − clumps_post_size / clumps_pre_size` | Share of clump size removed. |
| `f_clump_burden_pre` | `clumps_pre_count × clumps_pre_size` | Size-weighted clump burden before massaging. |
| `f_clump_burden_post` | `clumps_post_count × clumps_post_size` | Burden after massaging. |
| `f_clump_burden_mix` | `clumps_mix_count × clumps_mix_size` | Burden after mixing. |
| `f_mixing_x_clumps` | `mixing_cycles × clumps_pre_count` | Interaction: mixing effort × the burden it was applied to. |
| `f_d3_clumps` | `yn_to_bool(d3_clump_pattern)` | 1 if clumps were recorded at D3. Any present text other than n/no/false/f/0/none/no clumps/absent counts as 1; NaN stays NaN. |
| `f_clump_bag_count` | `Σ yn_to_bool(c)` over the harvest-bag clump columns (`clump_bags`) that exist | Number of harvest bags with clumps. Unfilled bags (NaN) add nothing. |

## 11. Dose and CAR

| Feature (key) | Formula | Description |
|---|---|---|
| `f_dose_all` | `dose_per_kg / 1e6` if present, else `dose_cells / 1e8`, else NaN | One dose column for every batch. **Not sure**: the two branches use different scalings (per-kg dose in millions/kg, absolute dose in units of 100 million cells), so the mixed column may not be on a single scale. |
| `f_post_formulation_dose` | `formulated_concentration / 1e6 × bag_volume × harvest_car_pct / 100 / subject_weight` | Dose per kg implied by what went into the bag (E6 CAR+ cells/kg). Uses harvest CAR% as the CAR fraction. |
| `f_achievable_dose` | `harvest_car_pct / 100 × washed_cells / (subject_weight × 1e6)` | Dose per kg the harvest could support (E6 CAR+ cells/kg). |
| `f_car_cells` | `harvest_car_pct × washed_cells / 100` | Absolute CAR+ cells at harvest (after wash). |
| `f_car_yield` | `harvest_car_pct / 100 × 2^cpdl` | CAR+ cells returned per cell seeded: transduction fraction × expansion (2^doublings). |
| `f_vcn_per_cell` | `fp_car_pct × vcn / 100` | Vector copies per cell across the final product (VCN is per transduced cell). |
| `f_flow_accuracy` | `fp_car_pct / harvest_car_pct` | Final-product CAR+% against the harvest reading; far from 1 points at the assay. |

## 12. Formulation and wash

| Feature (key) | Formula | Description |
|---|---|---|
| `f_formulated_total_concentration` | `formulated_concentration / (formulated_viability / 100) / 1e6` | Total (live + dead) cell concentration of the formulated suspension, E6 cells/mL. |
| `f_cells_into_formulation` | `washed_concentration × formulation_cell_volume` | Viable cells that went into formulation. |
| `f_cells_out_of_formulation` | `formulated_concentration × formulation_medium_volume` | Viable cells in the formulated suspension. |
| `f_cells_lost_formulation` | `f_cells_into_formulation − f_cells_out_of_formulation` | Viable cells lost during formulation. |
| `f_pct_lost_formulation` | `f_cells_lost_formulation / f_cells_into_formulation × 100` | The same loss as a percentage. |
| `f_cells_lost_wash` | `harvest_cells − washed_cells` | Cells the LOVO wash did not return. |

## 13. IL-2

| Feature (key) | Formula | Description |
|---|---|---|
| `f_il2_d6` | `il2_activity_d6 × il2_content_d6 / 1000` | IL-2 units fed at D6: activity (IU/mg) × protein content (µg/vial) / 1000 = IU per vial. |
| `f_il2_d8` | `il2_activity_d8 × il2_content_d8 / 1000` | Same at D8. |

## 14. Context

| Feature (key) | Formula | Description |
|---|---|---|
| `f_shift` | `"Morning"` if `thaw_time < 11:59:00` else `"Afternoon"` | Which shift started the batch. **Not sure**: the cut-off is hard-coded at 11:59, and a missing thaw time probably falls over (the registry would then mark the feature *errored*). |

## 15. Vector certificate of analysis (CoA)

These columns are built while the certificate workbook is loaded
(`ml.io.load_lv_coa`) and carry the prefix `coa_prefix` (display: "LV CoA "). Their
lineage is declared in `features.coa_lineage()` so the task stage can check for
descent from the target.

### 15.1 Dose, titer and infectivity

| Column (key) | Formula | Description |
|---|---|---|
| `coa_delivered_dose` | `dose_titer × coa_volume`, where `dose_titer = coa_titer_unadjusted` if present, else `coa_titer` (adjusted) | Delivered functional dose per G-Rex (titer × volume added). If no unadjusted column exists at all, the adjusted titer is used for every row (with a warning). **Not sure** whether `coa_volume` is the volume *added* to the G-Rex or the fill volume of the lot (the vocabulary calls it "Fill volume"). |
| `coa_delivered_dose_fallback` | `1 if coa_titer_unadjusted is missing else 0` | Flag: rows where the delivered dose fell back to the adjusted titer (possible potency back-channel). Only created when at least one row falls back. |
| `coa_titer_adjustment` | `coa_titer / coa_titer_unadjusted` (0 → NaN) | How hard the release titer was corrected; large values may flag assay drift. |
| `coa_specific_infectivity` | `coa_titer / coa_p24` (0 → NaN) | Infectious titer per physical (p24) particle. |

### 15.2 Impurities

| Column (key) | Formula | Description |
|---|---|---|
| `<impurity> below LOQ` (one per concentration impurity that has `<x` entries) | `1` if the raw cell starts with `<`, else `0` | Left-censoring flag. The numeric value is also replaced by `LOQ / 2` (a low sentinel) in the source column; that replacement is a transformation, not a new column. |
| `coa_last_dna_fraction` | `last impurity (E1b) / Σ(impurities[1:])`, sum with `min_count=1`, 0 → NaN | Last residual-DNA species as a fraction of total residual DNA. Flagged in the code as a lot-fingerprint **leakage risk** (it may encode vector-lot identity). |
| `f_coa_impurity_index` | `Σ zscore(impurity)` across the certificate impurity columns that exist (z-score `ddof=0`, within the site frame) | Residual impurities on one scale, summed. **Site-relative**: standardised within one site's batches, so values are not comparable between sites and cannot be recomputed for a single new batch. Listed in `site_relative()` → exploration only. |

### 15.3 Spec deviations

| Column | Formula | Description |
|---|---|---|
| `<pH column> Abs Deviation` | `abs(pH − ph_nominal [7.2])` | Two-sided deviation from the spec midpoint (U-shaped effect turned monotone). Needs `lv_coa.ph_nominal`; skipped (with a warning) when not set. |
| `<osmolality column> Abs Deviation` | `abs(osmolality − osmo_nominal [300])` | Same for osmolality. Needs `lv_coa.osmo_nominal`. |

### 15.4 Reference-donor summaries

Done for each metric: CAR% (`coa_car_metric`) and IFNγ (`coa_ifng_metric`), using
Donor 1 (`d1`) and Donor 2 (`d2`). Only created if both donor columns exist.

| Column | Formula | Description |
|---|---|---|
| `<metric> Donor MAX` | `max(d1, d2)` | Best donor. |
| `<metric> Donor MIN` | `min(d1, d2)` | Weakest donor; may track patient material better than the mean. |
| `<metric> Donor DIFF` | `max(d1, d2) − min(d1, d2)` | Disagreement between donors. |
| `<metric> Donor DIFF Rel` | `DIFF / mean(d1, d2)` (0 → NaN) | Disagreement relative to the mean. |
| `<metric> Donor AVG` | `mean(d1, d2)` | Reference average. Only computed when the certificate does not already provide its own average column. |

### 15.5 Missingness flags

| Column | Formula | Description |
|---|---|---|
| `<column> missing` | `1 if the numeric certificate column is NaN else 0` | "Not run" ≠ "below LOQ". Created for every numeric certificate column (titer, p24, PI ratio, donor values and averages, pH, osmolality, volume, DNA size, impurities, delivered-mass impurity columns) that has at least one NaN; the unadjusted titer is excluded. |

Note: the "Calc " (delivered-mass impurity) certificate columns are *read* from the
workbook, not computed in this repository. Whether the certificate computes them from
the impurity concentration × volume is **not sure** — the code does not show it.

## 16. Censoring flags (cleaning stage)

| Column | Formula | Description |
|---|---|---|
| `<parameter> censored` | `1 if the raw value was written as "<x"` else `0` | Created only when `cleaning.censored_flags: true`, for every numeric parameter that had at least one censored value. The numeric value itself is rewritten according to `cleaning.censored` (`limit`, `half` or `missing`). Descendants of the target are tracked via `ml.dataset.lineage`. |

## 17. Task-stage derived columns

Not part of the batch table, but created later when a task runs. Listed for
completeness; formulas are from the module docstrings and code.

| Column | Formula | Description |
|---|---|---|
| Cluster score (`clustering` action `summarise`) | `mean over members of  sign × (x − mean) / scale` | One score replacing a group of correlated parameters; `sign` is −1 if the member correlates negatively with the group's reference, else +1; `scale` is the member's std (or 1.0). Members that are missing for a row are ignored. **Not sure** about the exact column naming. |
| `<column>_resid` (`residualise` action) | residual of a (sequential) linear regression of the column on the representative(s) | Part of a column not explained by the retained parameter(s). **Not sure** of the exact regression order — see `decorrelate._sequential_linear`. |
| One-hot / binary / ordinal / target-mean / hashed encodings | see `ml.encode` | Numeric encodings of categorical parameters. Target-mean encodings are fitted on training rows only (cross-fitted). These are encodings rather than calculations from other parameters, so they are not expanded here. |

---

## 18. Open questions / "not sure" summary

1. **Unit of `pt_cd3_pct` / `pw_cd3_pct`** — `f_pt_cd3_cells` multiplies by the percentage without `/100`; the `*_share` features divide by 100. One of the two is probably off.
2. **`1e9` factor** in lactate/glucose per cell — exact unit conversion not documented.
3. **`vessel_count` other than 1** (for example 3 or NaN) is treated as two vessels in the lactate/glucose per-cell features, but gives NaN in `f_seeding_pct`.
4. **`f_seeding_pct`** — ratio (1.0 = on target) vs. percentage, and the downstream `-100` comparisons.
5. **`f_dose_all`** — two branches on different scalings (`/1e6` vs `/1e8`).
6. **`f_moi_quality`** — multiplying by the PI ratio vs. dividing.
7. **`coa_delivered_dose`** — which "volume" (`coa_volume`) is meant.
8. **`f_shift`** — fixed cut-off at 11:59 and handling of missing times.
9. **Spread features and single-vessel batches** — one vessel gives std = 0 and mean = min = max.
10. **`f_contact_within`** — missing contact time gives 0, not NaN.
11. **"Calc " impurity columns** — read from the certificate, not computed here.
12. **Real column names** — only vocabulary keys are visible in the repo; the true names are in the private vocabulary workbook.
