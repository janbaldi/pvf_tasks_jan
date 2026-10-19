# Calculated parameters / features

Every column in the batch table that is computed from other columns, grouped by
what it describes. Each entry gives the formula and a short description.

**Sources read** (branch `main`, commit `507349b`):
`src/ml/features.py` (the feature registry), `src/ml/io.py` (`load_lv_coa`,
vector certificate columns), `src/ml/clean.py` (censored flags),
`src/ml/dataset.py`, `src/ml/decorrelate.py`, `src/ml/encode.py` (task-stage columns).

**Conventions**

- Column names in backticks are the real parameter names from the `Name` column of
  `vocabulary.csv` (e.g. `Post Thaw FLOW CD4+ (%)`). Config settings such as
  `contact_target` are config keys, not columns. Where one feature expands into several
  columns (per day or per vessel) the entry shows the pattern.
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

| Feature | Formula | Description |
|---|---|---|
| `Log (PT CD4:CD8)` | `log10(Post Thaw FLOW CD4+ (%) / Post Thaw FLOW CD8+ (%))` | CD4:CD8 balance of the incoming (post-thaw, "PT") material on a log scale, so a 2-fold shift reads the same in either direction. NaN/inf if CD8 is 0. |
| `Log (PE CD4:CD8)` | `log10(Post Wash POS FLOW CD4+ (%) / Post Wash POS FLOW CD8+ (%))` | Same, after enrichment/selection ("PW"/"PE"). |
| `PT Total  CD3+ (Viable Cells)` | `Post Thaw Viable Cell Count after sampling(cells) × Post Thaw FLOW CD3+ (%)` | Absolute CD3+ cells post thaw. **Not sure** whether `Post Thaw FLOW CD3+ (%)` is stored as a fraction (0–1) or as a percentage (0–100): the code multiplies without `/100`, unlike the rows below. |
| `PT CD3+CD4+ (%)` | `Post Thaw FLOW CD4+ (%) × Post Thaw FLOW CD3+ (%) / 100` | CD4+ as a share of *all* cells rather than of the CD3+ gate (post thaw). |
| `PT Total CD4+ (Viable Cells)` | `PT CD3+CD4+ (%) × Post Thaw Viable Cell Count after sampling(cells)` | Absolute CD3+CD4+ cells post thaw. Depends on `PT CD3+CD4+ (%)`. **Not sure** about the % vs. fraction units (the share is in % here). |
| `PE CD3+CD4+ (%)` | `Post Wash POS FLOW CD4+ (%) × Post Wash POS FLOW CD3+ (%) / 100` | CD4+ as a share of all cells after enrichment. |
| `PE CD3+CD8+ (%)` | `Post Wash POS FLOW CD8+ (%) × Post Wash POS FLOW CD3+ (%) / 100` | CD8+ as a share of all cells after enrichment. |

## 2. Metabolites (lactate and glucose)

`Day 3 Number of G-Rex to seed` is the number of G-Rex vessels seeded (1 or 2). `Total Viable Cells for Expansion after VSVg Sampling (cells)` is the
number of cells used for expansion (cells). `1e9` is applied to convert the
concentration (g/L) and the cell count (cells) to "per million (E6) cells".
**Not sure** about the exact unit logic of the `1e9` factor (it is `1e3` mL→L
style scaling × `1e6` for "per E6"); the code does not document it.

| Feature | Formula | Description |
|---|---|---|
| `Harvest Lactate Normalized` | `(Harvest Lactate (g/L) × 1e9) × Day 3 Number of G-Rex to seed / Total Viable Cells for Expansion after VSVg Sampling (cells)` | Harvest (D10) lactate per cell seeded, so batches seeded into one or two G-Rex can be compared. |
| `D{6,8,10} Lact / E6 cells` (D6) | 1 vessel: `Day 6 Lactate G-Rex A (g/L) × 1e9 / Total Viable Cells for Expansion after VSVg Sampling (cells)`<br>2 vessels: `(Day 6 Lactate G-Rex A (g/L) + Day 6 Lactate G-Rex B (g/L)) × 1e9 / Total Viable Cells for Expansion after VSVg Sampling (cells)` | Lactate per E6 cells seeded at D6, summed over the vessels in use. Any other `Day 3 Number of G-Rex to seed` falls into the 2-vessel branch (**not sure** that is intended). |
| `D{6,8,10} Lact / E6 cells` (D8) | same with `Day 8 Lactate G-Rex A (g/L)/b` | Same at D8. |
| `D{6,8,10} Lact / E6 cells` (D10) | 1 vessel: `Harvest Lactate (g/L) × 1e9 / Total Viable Cells for Expansion after VSVg Sampling (cells)`<br>otherwise: `Harvest Lactate (g/L) × 2e9 / Total Viable Cells for Expansion after VSVg Sampling (cells)` | D10 version; the harvest is sampled once, so the value is counted twice for two vessels. |
| `D{6,8,10} Glucose / E6 cells` (D6) | 1 vessel: `Day 6 Glucose G-Rex A (g/L) × 1e9 / Total Viable Cells for Expansion after VSVg Sampling (cells)`<br>2 vessels: `(Day 6 Glucose G-Rex A (g/L) + Day 6 Glucose G-Rex B (g/L)) × 1e9 / Total Viable Cells for Expansion after VSVg Sampling (cells)` | Glucose remaining per E6 cells seeded at D6. |
| `D{6,8,10} Glucose / E6 cells` (D8) | same with `Day 8 Glucose G-Rex A (g/L)/b` | Same at D8. |
| `D{6,8,10} Glucose / E6 cells` (D10) | 1 vessel: `Harvest Glucose (g/L) × 1e9 / Total Viable Cells for Expansion after VSVg Sampling (cells)`<br>otherwise: `Harvest Glucose (g/L) × 2 × 1e9 / Total Viable Cells for Expansion after VSVg Sampling (cells)` | Same at D10. |
| `Lactate_perE6_slope_D6_D10` | least-squares slope of `D{6,8,10} Lact / E6 cells` (D6, D8, D10) against day (6, 8, 10) | How fast normalised lactate climbs across expansion. Needs ≥ 2 readings, otherwise NaN. |
| `Lactate_perE6_AUC_D6_D10` | trapezoidal area under `D{6,8,10} Lact / E6 cells` over days 6/8/10 | Total normalised lactate exposure. A gap is bridged, not treated as 0. |
| `Glucose_perE6_slope_D6_D10` | least-squares slope of `D{6,8,10} Glucose / E6 cells` over days 6/8/10 | How fast normalised glucose falls. |
| `Glucose_perE6_AUC_D6_D10` | trapezoidal area under `D{6,8,10} Glucose / E6 cells` over days 6/8/10 | Total normalised glucose exposure. |
| `Lactate_to_Glucose_perE6_ratio_D{day}` (day = 6, 8, 10) | `safe_div(D{6,8,10} Lact / E6 cells[day], D{6,8,10} Glucose / E6 cells[day])` | Lactate against glucose on the same day; separates cells burning sugar fast from cells simply fed more. |

## 3. Profiles (categorical classes)

| Feature | Formula | Description |
|---|---|---|
| `growth profile` | `class 1` if `cPDL (D3 to D10) < cpdl_low [4.0]` **or** `Harvest Lactate Normalized < lactate_low [12.5]`<br>`class 2` if `cPDL (D3 to D10) > cpdl_high [5.0]` **and** `Harvest Lactate Normalized > lactate_high [30.0]`<br>else `class 3` | Below / above / within expected growth. Class labels come from the vocabulary (`growth_profile_classes`; the notes in the vocabulary suggest slow / fast / typical). **Not sure** how rows with a missing `cPDL (D3 to D10)` or lactate are classed: comparisons against NaN are False, so they fall into the default "typical" class. |
| `CD4 profile` | `class 1` if `Post Thaw FLOW CD4+ (%) ≤ cd4_low [40]`<br>`class 2` if `Post Thaw FLOW CD4+ (%) ≥ cd4_high [60]`<br>else `class 3`; NaN if `Post Thaw FLOW CD4+ (%)` is missing | CD4 share of the incoming material in three bands (low / high / mid). A missing reading gets no profile. |

## 4. Viability changes

| Feature | Formula | Description |
|---|---|---|
| `Drop viability D3 vs D0 (%)` | `(Day 3 Average of Pooled Viability (%) − Post Wash POS Viability (%)) / Post Wash POS Viability (%) × 100` | Relative viability change over activation (D0→D3), %. |
| `Viability_drop_D3_vs_D0_pct` | `Day 3 Average of Pooled Viability (%) − Post Wash POS Viability (%)` | Same, in percentage points. |
| `D3-D10 viability change (%)` | `(Day 3 Average of Pooled Viability (%) − Harvest Pre-Wash Cell Viability (%)) / Day 3 Average of Pooled Viability (%) × 100` | Relative viability *lost* between D3 and harvest, %. |
| `Viability change during expansion` | `(Harvest Pre-Wash Cell Viability (%) − Day 3 Average of Pooled Viability (%)) / Day 3 Average of Pooled Viability (%) × 100` | Same comparison with the opposite sign (viability *gain*), matching the Power Query investigation. It is exactly `−D3-D10 viability change (%)`. |
| `Viability_change_D3_to_D10_pct` | `Harvest Post Wash Viability Average (%) − Day 3 Average of Pooled Viability (%)` | Net viability change D3 → washed harvest, percentage points. Note it uses the *washed* viability, not `Harvest Pre-Wash Cell Viability (%)`. |
| `D10 harvest - D10 wash viability change (%)` | `(Harvest Post Wash Viability Average (%) − Harvest Pre-Wash Cell Viability (%)) / Harvest Pre-Wash Cell Viability (%) × 100` | Effect of the LOVO wash on viability, %. |

## 5. Seeding and vessel balance (D3)

| Feature | Formula | Description |
|---|---|---|
| `D3 % cells seeded / target` | 1 vessel: `Total Viable Cells for Expansion after VSVg Sampling (cells) / 500 000`<br>2 vessels: `Total Viable Cells for Expansion after VSVg Sampling (cells) / 1 000 000`<br>otherwise NaN | Cells seeded against the target for the number of vessels. The result is a ratio where 1.0 = on target; the next two features compare it with `100.0`. **Not sure** whether this is a bug (ratio vs. percentage) or whether the factor of 100 is applied elsewhere. |
| `D3_seeding_error_pct` | `abs(D3 % cells seeded / target − 100)` | Distance from target seeding in either direction. (Same unit caveat as above.) |
| `D3_seeding_within_5pct` | `1.0 if abs(D3 % cells seeded / target − 100) ≤ seeding_tolerance [5]` else `0.0` | Whether seeding landed within ±5 % of target. (Same unit caveat.) |
| `Avg_cells_per_bag_D3` | `safe_div(Day 3 Total Viable Cells Available for Seeding G-Rex (cells), # of Culture Bags)` | Cells per culture bag — the density the cells see. |
| `Seeded_cells_{statistic}_AB` (`statistic` = mean, min, max, std, cv) | over the pair (`Day 3 Actual Viable Cells Seeded G-Rex A`, `Day 3 Actual Viable Cells Seeded G-Rex B`):<br>`mean`, `min`, `max`, `std(ddof=0)`, `cv = std / mean` | Vessel balance of seeded viable cells. 5 columns, A–B. |
| `Vector_volume_{statistic}_AB_mL` (`statistic` = mean, min, max, std, cv) | same over (`Volume Vector Added to G-Rex A (mL)`, `Volume Vector Added to G-Rex B (mL)`) | Vessel balance of the vector volume added. 5 columns, A–B. |

For the spread families a missing vessel is skipped by pandas (so one vessel gives
mean = min = max, std = 0). **Not sure** that this is the desired meaning for a
single-vessel batch.

## 6. MOI (multiplicity of infection) (D3)

`titer` = the measured vector titer: the certificate's titer if present
(`LV CoA  + LV Titer`), otherwise the batch record's `Vector Titer (IU/mL)`.
`pooled_volume` = `Volume Vector Added to G-Rex A (mL) + Volume Vector Added to G-Rex B (mL)` (a missing vessel counts as 0 as
long as the other has a value). `pooled_cells` = `Day 3 Total Viable Cells Available for Seeding G-Rex (cells)` if the column exists,
otherwise `Day 3 Actual Viable Cells Seeded G-Rex A + Day 3 Actual Viable Cells Seeded G-Rex B`.

| Feature | Formula | Description |
|---|---|---|
| `MOI_eff_G-Rex_A` | `titer × Volume Vector Added to G-Rex A (mL) / Day 3 Actual Viable Cells Seeded G-Rex A` | Infectious units per seeded cell in vessel A. |
| `log_MOI_eff_G-Rex_A` | `log1p(MOI_eff_G-Rex_A)` = `ln(1 + MOI_A)` | Vessel A MOI on a log scale. |
| `MOI_eff_G-Rex_B` | `titer × Volume Vector Added to G-Rex B (mL) / Day 3 Actual Viable Cells Seeded G-Rex B` | Same, vessel B. |
| `log_MOI_eff_G-Rex_B` | `log1p(MOI_eff_G-Rex_B)` | Vessel B MOI on a log scale. |
| `MOI_eff_pool` | `titer × pooled_volume / pooled_cells` | MOI over the whole batch. |
| `log_MOI_eff_pool` | `log1p(MOI_eff_pool)` | Pooled MOI on a log scale. |
| `MOI_eff_quality_adjusted` | `MOI_eff_pool × PI Ratio` (certificate column) | Pooled MOI weighted by the particle (PI) ratio. **Not sure** about the direction: the code comment says "weighted by how many particles in the lot are infectious", but a PI *ratio* is normally particles/infectious unit, which would be the inverse. Check with the vector team. |
| `MOI_per_CD3_pool` | `titer × pooled_volume / PT Total  CD3+ (Viable Cells)` | Infectious units per CD3+ cell (the actual target cells). Inherits the unit caveat of `PT Total  CD3+ (Viable Cells)`. |

## 7. Contact time (D3)

Three variants of the same pair of features: the batch's own reading (`Day 3 Incubation Time (120min+/-30 min)`)
and vessels A and B (`Amount of Time G-Rex A in Incubator (120min+/-30 min)`, `Amount of Time G-Rex B in Incubator (120min+/-30 min)`). `target` = `contact_target [120]` min,
`tolerance` = `contact_tolerance [30]` min.

| Feature | Formula | Description |
|---|---|---|
| `D3_contact_time_deviation[_G-Rex_A/_G-Rex_B]` (batch / A / B) | `abs(<contact time> − 120)`, with `<contact time>` = the batch reading or vessel A / B reading above | Minutes away from the transduction contact-time target. |
| `D3_contact_time_within_window[_G-Rex_A/_G-Rex_B]` (batch / A / B) | `1.0 if 90 ≤ <contact time> ≤ 150` else `0.0` | Whether the contact time stayed within target ± tolerance. NaN input gives 0.0 (False), not NaN — **not sure** if intended. |

## 8. Environment (CO₂ and temperature setpoints)

Expanded dynamically: one feature for every column of the frame whose name contains the
vocabulary marker `CO2 Saturation (%)` ("CO2 Saturation (%)") or `Temperature (°C)`
("Temperature (°C)"). The set therefore differs between sites. Process day is read
from the column name (`Day N` → `DN`, or a D0/D1/D3/D6/D8/D10 tag, else "Context").

| Feature | Formula | Description |
|---|---|---|
| `<column> dev_from_5.0pct` | `abs(<CO2 column> − co2_target [5.0])` | Distance from the CO₂ setpoint: how well the incubator held it. |
| `<column> dev_from_37.0C` | `abs(<temperature column> − temp_target [37.0])` | Distance from the temperature setpoint. |

## 9. Incubation durations

Minutes between two timestamps: `(later − earlier).total_seconds() / 60`.

| Feature | Formula | Description |
|---|---|---|
| `Incubation D0-D1 (min)` | `Date and Time Cells Removed From Incubator Day 1 − Date and Time Cells placed in Incubator Day 0` | Minutes in the incubator D0→D1. |
| `Incubation D1-D3` | `Date and Time Cells Removed From Incubator Day 3 − Date and Time Cells placed in Incubator Day 1` | D1→D3. |
| `Incubation D3-D6 (min)` | `Date and Time Cells Removed From Incubator Day 6 − Day 3 Date and Time G-Rex A Returned to Incubator` | D3→D6. |
| `Incubation D6-D8 (min)` | `Date and Time Cells Removed From Incubator Day 8 − Day 6 Date and Time G-Rex A Returned to Incubator` | D6→D8. |
| `Incubation D8-D10 (min)` | `Date and Time Cells Removed From Incubator Day 10 − Day 8 Date and Time G-Rex A Returned to Incubator` | D8→D10. |
| `Incubation_total_D3_to_D10_min` | `Incubation D3-D6 (min) + Incubation D6-D8 (min) + Incubation D8-D10 (min)` | Total incubator time D3–D10. Plain `+`, so one missing interval makes the total NaN. |

## 10. Clumps

| Feature | Formula | Description |
|---|---|---|
| `Clump_count_reduction_fraction` | `1 − Post-Massaging: # of Clumps / Pre-Activation Clumps: # of Clumps before massage` | Share of clumps removed by massaging. |
| `Clump_size_reduction_fraction` | `1 − Post-Massaging: Size of Clumps / Pre-Activation Clumps: Size of Clumps before massage` | Share of clump size removed. |
| `Clump_burden_weighted_pre` | `Pre-Activation Clumps: # of Clumps before massage × Pre-Activation Clumps: Size of Clumps before massage` | Size-weighted clump burden before massaging. |
| `Clump_burden_weighted_post_massage` | `Post-Massaging: # of Clumps × Post-Massaging: Size of Clumps` | Burden after massaging. |
| `Clump_burden_weighted_post_mixing` | `Post-Mixing Clumps: # of Clumps × Post-Mixing Clumps: Size of Clumps` | Burden after mixing. |
| `Mixing_cycles_x_pre_activation_clump_count` | `Day 1 Number of Mixing Cycles Post Activation × Pre-Activation Clumps: # of Clumps before massage` | Interaction: mixing effort × the burden it was applied to. |
| `D3_clumps_flag` | `yn_to_bool(day 3 clump pattern)` | 1 if clumps were recorded at D3. Any present text other than n/no/false/f/0/none/no clumps/absent counts as 1; NaN stays NaN. |
| `D10_clump_bag_count` | `Σ yn_to_bool(c)` over the harvest-bag clump columns (`Day 10 Clump Present Bag 1–6`) that exist | Number of harvest bags with clumps. Unfilled bags (NaN) add nothing. |

## 11. Dose and CAR

| Feature | Formula | Description |
|---|---|---|
| `Dose (CAR+ viable cells/kg) all weights` | `Dose (CAR+ viable cells/kg) / 1e6` if present, else `Dose: Number of CAR+ Viable T-Cells (cells) / 1e8`, else NaN | One dose column for every batch. **Not sure**: the two branches use different scalings (per-kg dose in millions/kg, absolute dose in units of 100 million cells), so the mixed column may not be on a single scale. |
| `Post formulation dose` | `Final Formulation CS5 Viable Cell Concentration Average (cells/mL) / 1e6 × Volume per Bag (mL) × Harvest Pre-Wash Flow CAR+ Expression (%) / 100 / Subject Weight (kg)` | Dose per kg implied by what went into the bag (E6 CAR+ cells/kg). Uses harvest CAR% as the CAR fraction. |
| `D10 Achievable Dose (E6 CAR+ viable T-cells)` | `Harvest Pre-Wash Flow CAR+ Expression (%) / 100 × Harvest Post Wash Viable Cell Count (cells) / (Subject Weight (kg) × 1e6)` | Dose per kg the harvest could support (E6 CAR+ cells/kg). |
| `D10 Total CAR+ viable T-cells achieved` | `Harvest Pre-Wash Flow CAR+ Expression (%) × Harvest Post Wash Viable Cell Count (cells) / 100` | Absolute CAR+ cells at harvest (after wash). |
| `CAR T produced per cell engaged D3` | `Harvest Pre-Wash Flow CAR+ Expression (%) / 100 × 2^cPDL (D3 to D10)` | CAR+ cells returned per cell seeded: transduction fraction × expansion (2^doublings). |
| `VCN/cell` | `FP Flow CAR+ (%) × Provirus Vector Copy Number (copies/transduced cell) / 100` | Vector copies per cell across the final product (VCN is per transduced cell). |
| `Flow accuracy (effective)` | `FP Flow CAR+ (%) / Harvest Pre-Wash Flow CAR+ Expression (%)` | Final-product CAR+% against the harvest reading; far from 1 points at the assay. |

## 12. Formulation and wash

| Feature | Formula | Description |
|---|---|---|
| `Total CS5 cell concentration (E6 cells/mL)` | `Final Formulation CS5 Viable Cell Concentration Average (cells/mL) / (Final Formulation CS5 Viability Average (%) / 100) / 1e6` | Total (live + dead) cell concentration of the formulated suspension, E6 cells/mL. |
| `total viable cells used for formulation` | `Harvest Post Wash Viable Cell Concentration Average (cells/mL) × Volume of Cells Used for Formulation (mL)` | Viable cells that went into formulation. |
| `total viable cells in formulated suspension` | `Final Formulation CS5 Viable Cell Concentration Average (cells/mL) × Volume CS5 Used for Formulation (mL)` | Viable cells in the formulated suspension. |
| `number of viable cells lost during formulation` | `total viable cells used for formulation − total viable cells in formulated suspension` | Viable cells lost during formulation. |
| `% cells lost during formulation` | `number of viable cells lost during formulation / total viable cells used for formulation × 100` | The same loss as a percentage. |
| `total cells lost during LOVO wash (cells)` | `Harvest Pre-Wash Viable Cell Count after Sampling (cells) − Harvest Post Wash Viable Cell Count (cells)` | Cells the LOVO wash did not return. |

## 13. IL-2

| Feature | Formula | Description |
|---|---|---|
| `IL2_IU_total_D6` | `Day 6 Activity of IL-2 (IU/mg) × Day 6 IL-2 Protein Content (µg/vial) / 1000` | IL-2 units fed at D6: activity (IU/mg) × protein content (µg/vial) / 1000 = IU per vial. |
| `IL2_IU_total_D8` | `Day 8 Activity of IL-2 (IU/mg) × Day 8 IL-2 Protein Content (µg/vial) / 1000` | Same at D8. |

## 14. Context

| Feature | Formula | Description |
|---|---|---|
| `Shift` | `"Morning"` if `Thaw Start Time < 11:59:00` else `"Afternoon"` | Which shift started the batch. **Not sure**: the cut-off is hard-coded at 11:59, and a missing thaw time probably falls over (the registry would then mark the feature *errored*). |

## 15. Vector certificate of analysis (CoA)

These columns are built while the certificate workbook is loaded
(`ml.io.load_lv_coa`) and carry the prefix `LV CoA ` (display: "LV CoA "). Their
lineage is declared in `features.coa_lineage()` so the task stage can check for
descent from the target.

### 15.1 Dose, titer and infectivity

| Column (key) | Formula | Description |
|---|---|---|
| `Delivered Dose per GRex` | `dose_titer × Volume`, where `dose_titer = LV Titer (unadjusted)` if present, else `LV Titer` (adjusted) | Delivered functional dose per G-Rex (titer × volume added). If no unadjusted column exists at all, the adjusted titer is used for every row (with a warning). **Not sure** whether `Volume` is the volume *added* to the G-Rex or the fill volume of the lot (the vocabulary calls it "Fill volume"). |
| `Delivered Dose used adj titer` | `1 if LV Titer (unadjusted) is missing else 0` | Flag: rows where the delivered dose fell back to the adjusted titer (possible potency back-channel). Only created when at least one row falls back. |
| `Titer Adjustment Factor` | `LV Titer / LV Titer (unadjusted)` (0 → NaN) | How hard the release titer was corrected; large values may flag assay drift. |
| `Specific Infectivity` | `LV Titer / Physical Titer p24 ELISA` (0 → NaN) | Infectious titer per physical (p24) particle. |

### 15.2 Impurities

| Column (key) | Formula | Description |
|---|---|---|
| `<impurity> below LOQ` (one per concentration impurity that has `<x` entries) | `1` if the raw cell starts with `<`, else `0` | Left-censoring flag. The numeric value is also replaced by `LOQ / 2` (a low sentinel) in the source column; that replacement is a transformation, not a new column. |
| `E1b Fraction of Total DNA` | `last impurity (E1b) / Σ(impurities[1:])`, sum with `min_count=1`, 0 → NaN | Last residual-DNA species as a fraction of total residual DNA. Flagged in the code as a lot-fingerprint **leakage risk** (it may encode vector-lot identity). |
| `LVCoA_Impurity_Index_zsum` | `Σ zscore(impurity)` across the certificate impurity columns that exist (z-score `ddof=0`, within the site frame) | Residual impurities on one scale, summed. **Site-relative**: standardised within one site's batches, so values are not comparable between sites and cannot be recomputed for a single new batch. Listed in `site_relative()` → exploration only. |

### 15.3 Spec deviations

| Column | Formula | Description |
|---|---|---|
| `<pH column> Abs Deviation` | `abs(pH − ph_nominal [7.2])` | Two-sided deviation from the spec midpoint (U-shaped effect turned monotone). Needs `lv_coa.ph_nominal`; skipped (with a warning) when not set. |
| `<osmolality column> Abs Deviation` | `abs(osmolality − osmo_nominal [300])` | Same for osmolality. Needs `lv_coa.osmo_nominal`. |

### 15.4 Reference-donor summaries

Done for each metric: CAR% (`CAR%`) and IFNγ (`IFNg`), using
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

1. **Unit of `Post Thaw FLOW CD3+ (%)` / `Post Wash POS FLOW CD3+ (%)`** — `PT Total  CD3+ (Viable Cells)` multiplies by the percentage without `/100`; the `*_share` features divide by 100. One of the two is probably off.
2. **`1e9` factor** in lactate/glucose per cell — exact unit conversion not documented.
3. **`Day 3 Number of G-Rex to seed` other than 1** (for example 3 or NaN) is treated as two vessels in the lactate/glucose per-cell features, but gives NaN in `D3 % cells seeded / target`.
4. **`D3 % cells seeded / target`** — ratio (1.0 = on target) vs. percentage, and the downstream `-100` comparisons.
5. **`Dose (CAR+ viable cells/kg) all weights`** — two branches on different scalings (`/1e6` vs `/1e8`).
6. **`MOI_eff_quality_adjusted`** — multiplying by the PI ratio vs. dividing.
7. **`Delivered Dose per GRex`** — which "volume" (`Volume`) is meant.
8. **`Shift`** — fixed cut-off at 11:59 and handling of missing times.
9. **Spread features and single-vessel batches** — one vessel gives std = 0 and mean = min = max.
10. **`D3_contact_time_within_window[_G-Rex_A/_G-Rex_B]`** — missing contact time gives 0, not NaN.
11. **"Calc " impurity columns** — read from the certificate, not computed here.
12. **Certificate columns** — in the batch table they carry the prefix `LV CoA ` (vocabulary `coa_prefix`), which is not repeated in every entry.
