# Calculated parameters / features

Every column in the batch table that is computed from other columns: the feature
registry (`src/ml/features.py`), the vector certificate loader (`src/ml/io.py`), the
cleaning stage's censoring flags and, for completeness, what a task adds later.

Generated from the code on `main` (after `5d65073`), `vocabulary.csv` and `catalog.xlsx`.
The formula column uses the real parameter names from the `Name` column of
`vocabulary.csv`; names in plain text inside a formula (`moi_target`, `cpdl_low`, …)
are `config.yaml` settings, with their default in brackets.

## Conventions

- **Kind** is the `Calculated` column of the catalog:
  - **Normalization** expresses a quantity against a size or a reference so batches of
    different size compare: per cell, per kilogram, per bag, a share of a target, a
    relative change, a fraction, a log or standardised scale.
  - **Engineered** is every other calculation: differences, deviations, products,
    durations, flags, slopes, areas, classes.
  - **No** (catalog only) is a parameter recorded as it is.
- **Calculated By** (catalog) says who calculates it: `pipeline (feature registry)`,
  `pipeline (vector certificate loader)`, or `source record` (a column the batch
  records already deliver calculated).
- **Day** is the feature's process day. **Known** is the catalog's `Available From Day`.
- A feature is computed only when the catalog lists it and every input is there.
  Otherwise the build's run log says it is *not in catalog*, *blocked* (with what is
  missing, including a config setting it waits for) or *present* (a source supplies
  it); the build report's *Catalog coverage* lists the blocked ones too.
- Division by zero gives a missing value, never an infinity.
- **Status** marks what changed in this revision (see the change log at the end).

## Contents

1. [Flow and phenotype](#1-flow-and-phenotype)
2. [Metabolites](#2-metabolites)
3. [Profiles](#3-profiles)
4. [Viability](#4-viability)
5. [Seeding](#5-seeding)
6. [Vessel balance](#6-vessel-balance)
7. [MOI](#7-moi)
8. [Contact time](#8-contact-time)
9. [Incubation](#9-incubation)
10. [Clumps](#10-clumps)
11. [Dose and CAR](#11-dose-and-car)
12. [Formulation and wash](#12-formulation-and-wash)
13. [Vector CoA](#13-vector-coa)
14. [IL-2](#14-il-2)
15. [Context](#15-context)
16. Environment (CO₂ and temperature setpoints)
17. Vector certificate columns (loader)
18. Censoring flags (cleaning stage)
19. Task-stage derived columns
20. Config settings the features read
21. Change log
22. Open questions

## 1. Flow and phenotype

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `Log (PT CD4:CD8)` | log10(`Post Thaw FLOW CD4+ (%)` / `Post Thaw FLOW CD8+ (%)`) | Normalization | log10 ratio | D0 | day 3 | CD4 to CD8 balance of the incoming material, on a log scale so a two-fold shift reads the same in either direction | changed: division by zero → missing |
| `Log (PE CD4:CD8)` | log10(`Post Wash POS FLOW CD4+ (%)` / `Post Wash POS FLOW CD8+ (%)`) | Normalization | log10 ratio | D0 | day 0 | CD4 to CD8 balance after enrichment, on a log scale | changed: division by zero → missing |
| `Log_CD4_CD8_shift_PE_to_PT` | `Log (PT CD4:CD8)` − `Log (PE CD4:CD8)` | Engineered | log10 ratio | D3 | day 3 | How the CD4:CD8 balance moved between the post-enrichment and the post-thaw reading; positive means towards CD4 | **new** |
| `PT Total  CD3+ (Viable Cells)` | `Post Thaw Viable Cell Count after sampling(cells)` × `Post Thaw FLOW CD3+ (%)` / 100 | Engineered | cells | D0 | day 3 | Absolute CD3+ cells post thaw — the population the process actually works with | changed: now divides the CD3 % by 100 |
| `PT CD3+CD4+ (%)` | `Post Thaw FLOW CD4+ (%)` × `Post Thaw FLOW CD3+ (%)` / 100 | Normalization | % | D0 | day 3 | CD4+ as a share of all cells rather than of the CD3+ gate |  |
| `PT Total CD4+ (Viable Cells)` | `PT CD3+CD4+ (%)` / 100 × `Post Thaw Viable Cell Count after sampling(cells)` | Engineered | cells | D0 | day 3 | Absolute CD3+CD4+ cells post thaw | changed: now takes the CD4 share as a percentage |
| `PE CD3+CD4+ (%)` | `Post Wash POS FLOW CD4+ (%)` × `Post Wash POS FLOW CD3+ (%)` / 100 | Normalization | % | D0 | day 0 | CD4+ as a share of all cells after enrichment |  |
| `PE CD3+CD8+ (%)` | `Post Wash POS FLOW CD8+ (%)` × `Post Wash POS FLOW CD3+ (%)` / 100 | Normalization | % | D0 | day 0 | CD8+ as a share of all cells after enrichment |  |

## 2. Metabolites

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `D6 Lact / E6 cells` | 1 vessel: `Day 6 Lactate G-Rex A (g/L)` × 1e9 / `Total Viable Cells for Expansion after VSVg Sampling (cells)`; 2 vessels: (`Day 6 Lactate G-Rex A (g/L)` + `Day 6 Lactate G-Rex B (g/L)`) × 1e9 / `Total Viable Cells for Expansion after VSVg Sampling (cells)`; any other `Day 3 Number of G-Rex to seed`: missing | Normalization | g/L per E6 cells seeded | D6 | day 6 | Lactate per million cells seeded, summed over the vessels in use | changed: NaN for a vessel count other than 1 or 2 |
| `D8 Lact / E6 cells` | 1 vessel: `Day 8 Lactate G-Rex A (g/L)` × 1e9 / `Total Viable Cells for Expansion after VSVg Sampling (cells)`; 2 vessels: (`Day 8 Lactate G-Rex A (g/L)` + `Day 8 Lactate G-Rex B (g/L)`) × 1e9 / `Total Viable Cells for Expansion after VSVg Sampling (cells)`; any other `Day 3 Number of G-Rex to seed`: missing | Normalization | g/L per E6 cells seeded | D8 | day 8 | Lactate per million cells seeded, summed over the vessels in use | changed: NaN for a vessel count other than 1 or 2 |
| `D10 Lact / E6 cells` | `Harvest Lactate (g/L)` × 1e9 × `Day 3 Number of G-Rex to seed` / `Total Viable Cells for Expansion after VSVg Sampling (cells)` | Normalization | g/L per E6 cells seeded | D10 | day 10 | Harvest lactate per million cells seeded; the pooled harvest sample counts once per vessel | changed: harvest × vessel count (same values for 1 or 2 vessels) |
| `D6 Glucose / E6 cells` | 1 vessel: `Day 6 Glucose G-Rex A (g/L)` × 1e9 / `Total Viable Cells for Expansion after VSVg Sampling (cells)`; 2 vessels: (`Day 6 Glucose G-Rex A (g/L)` + `Day 6 Glucose G-Rex B (g/L)`) × 1e9 / `Total Viable Cells for Expansion after VSVg Sampling (cells)`; any other `Day 3 Number of G-Rex to seed`: missing | Normalization | g/L per E6 cells seeded | D6 | day 6 | Glucose remaining per million cells seeded, summed over the vessels in use | changed: NaN for a vessel count other than 1 or 2 |
| `D8 Glucose / E6 cells` | 1 vessel: `Day 8 Glucose G-Rex A (g/L)` × 1e9 / `Total Viable Cells for Expansion after VSVg Sampling (cells)`; 2 vessels: (`Day 8 Glucose G-Rex A (g/L)` + `Day 8 Glucose G-Rex B (g/L)`) × 1e9 / `Total Viable Cells for Expansion after VSVg Sampling (cells)`; any other `Day 3 Number of G-Rex to seed`: missing | Normalization | g/L per E6 cells seeded | D8 | day 8 | Glucose remaining per million cells seeded, summed over the vessels in use | changed: NaN for a vessel count other than 1 or 2 |
| `D10 Glucose / E6 cells` | `Harvest Glucose (g/L)` × 1e9 × `Day 3 Number of G-Rex to seed` / `Total Viable Cells for Expansion after VSVg Sampling (cells)` | Normalization | g/L per E6 cells seeded | D10 | day 10 | Harvest glucose per million cells seeded | changed: harvest × vessel count (same values for 1 or 2 vessels) |
| `Lactate_perE6_slope_D6_D10` | least-squares slope of `D6 Lact / E6 cells`, `D8 Lact / E6 cells`, `D10 Lact / E6 cells` against day 6, 8, 10 (≥ 2 readings) | Engineered | per day | D10 | day 10 | How fast normalised lactate climbs across expansion, as one number instead of three readings |  |
| `Lactate_perE6_AUC_D6_D10` | trapezoidal area under `D6 Lact / E6 cells`, `D8 Lact / E6 cells`, `D10 Lact / E6 cells` over days 6–10 (gaps bridged) | Engineered | day × g/L per E6 cells seeded | D10 | day 10 | Total normalised lactate exposure across expansion |  |
| `Glucose_perE6_slope_D6_D10` | least-squares slope of `D6 Glucose / E6 cells`, `D8 Glucose / E6 cells`, `D10 Glucose / E6 cells` against day 6, 8, 10 | Engineered | per day | D10 | day 10 | How fast normalised glucose falls across expansion |  |
| `Glucose_perE6_AUC_D6_D10` | trapezoidal area under `D6 Glucose / E6 cells`, `D8 Glucose / E6 cells`, `D10 Glucose / E6 cells` over days 6–10 | Engineered | day × g/L per E6 cells seeded | D10 | day 10 | Total normalised glucose exposure across expansion |  |
| `Lactate_perE6_increment_D6_D8` | `D8 Lact / E6 cells` − `D6 Lact / E6 cells` | Engineered | g/L per E6 cells seeded | D8 | day 8 | Change in normalised lactate from day 6 to day 8: when the change happens, which one slope averages away | **new** |
| `Lactate_perE6_increment_D8_D10` | `D10 Lact / E6 cells` − `D8 Lact / E6 cells` | Engineered | g/L per E6 cells seeded | D10 | day 10 | Change in normalised lactate from day 8 to day 10: when the change happens, which one slope averages away | **new** |
| `Glucose_perE6_increment_D6_D8` | `D8 Glucose / E6 cells` − `D6 Glucose / E6 cells` | Engineered | g/L per E6 cells seeded | D8 | day 8 | Change in normalised glucose from day 6 to day 8: when the change happens, which one slope averages away | **new** |
| `Glucose_perE6_increment_D8_D10` | `D10 Glucose / E6 cells` − `D8 Glucose / E6 cells` | Engineered | g/L per E6 cells seeded | D10 | day 10 | Change in normalised glucose from day 8 to day 10: when the change happens, which one slope averages away | **new** |
| `Lactate_to_Glucose_perE6_ratio_D6` | `D6 Lact / E6 cells` / `D6 Glucose / E6 cells` | Engineered | ratio | D6 | day 6 | Lactate against glucose on the same day, which separates cells burning sugar fast from cells simply being fed more |  |
| `Lactate_to_Glucose_perE6_ratio_D8` | `D8 Lact / E6 cells` / `D8 Glucose / E6 cells` | Engineered | ratio | D8 | day 8 | Lactate against glucose on the same day, which separates cells burning sugar fast from cells simply being fed more |  |
| `Lactate_to_Glucose_perE6_ratio_D10` | `D10 Lact / E6 cells` / `D10 Glucose / E6 cells` | Engineered | ratio | D10 | day 10 | Lactate against glucose on the same day, which separates cells burning sugar fast from cells simply being fed more |  |
| `Lactate_yield_on_glucose_D6` | (mean(`Day 6 Lactate G-Rex A (g/L)`, `Day 6 Lactate G-Rex B (g/L)`) − medium_lactate) / (medium_glucose − mean(`Day 6 Glucose G-Rex A (g/L)`, `Day 6 Glucose G-Rex B (g/L)`)); missing when no glucose was used (config: features.medium_glucose, features.medium_lactate) | Normalization | g lactate / g glucose | D6 | day 6 | Lactate made per glucose consumed since fresh medium: the glycolytic state of the culture, independent of how many cells there are. Valid only while no feed has changed the medium the fresh values describe. Needs `features.medium_glucose / medium_lactate` in config; blocked until it is set | **new** |
| `Lactate_yield_on_glucose_D8` | (mean(`Day 8 Lactate G-Rex A (g/L)`, `Day 8 Lactate G-Rex B (g/L)`) − medium_lactate) / (medium_glucose − mean(`Day 8 Glucose G-Rex A (g/L)`, `Day 8 Glucose G-Rex B (g/L)`)); missing when no glucose was used (config: features.medium_glucose, features.medium_lactate) | Normalization | g lactate / g glucose | D8 | day 8 | Lactate made per glucose consumed since fresh medium: the glycolytic state of the culture, independent of how many cells there are. Valid only while no feed has changed the medium the fresh values describe. Needs `features.medium_glucose / medium_lactate` in config; blocked until it is set | **new** |
| `Lactate_yield_on_glucose_D10` | (`Harvest Lactate (g/L)` − medium_lactate) / (medium_glucose − `Harvest Glucose (g/L)`); missing when no glucose was used (config: features.medium_glucose, features.medium_lactate) | Normalization | g lactate / g glucose | D10 | day 10 | Lactate made per glucose consumed since fresh medium: the glycolytic state of the culture, independent of how many cells there are. Valid only while no feed has changed the medium the fresh values describe. Needs `features.medium_glucose / medium_lactate` in config; blocked until it is set | **new** |

## 3. Profiles

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `growth profile` | class 1 if `cPDL (D3 to D10)` < cpdl_low [4.0] or `D10 Lact / E6 cells` < lactate_low [12.5]; class 2 if `cPDL (D3 to D10)` > cpdl_high [5.0] and `D10 Lact / E6 cells` > lactate_high [30.0]; else class 3; missing if either input is missing | Engineered | categorical | D10 | day 10 | Below, within or above the expected growth, from doublings and normalised harvest lactate together; thresholds are in config. A batch missing either input gets no profile rather than the middle one | changed: uses the D10 lactate per cell; missing input → no profile |
| `CD4 profile` | class 1 if `Post Thaw FLOW CD4+ (%)` ≤ cd4_low [40]; class 2 if ≥ cd4_high [60]; else class 3; missing if the input is | Engineered | categorical | D0 | day 0 | The CD4 share of the incoming material in three bands; a batch with no reading gets no profile rather than the middle one |  |

## 4. Viability

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `Viability_drop_D3_vs_D0_pct` | `Day 3 Average of Pooled Viability (%)` − `Post Wash POS Viability (%)` | Engineered | percentage points | D3 | day 3 | Viability change over activation in percentage points, which stays readable when the starting viability is low |  |
| `D3-D10 viability change (%)` | (`Day 3 Average of Pooled Viability (%)` − `Harvest Pre-Wash Cell Viability (%)`) / `Day 3 Average of Pooled Viability (%)` × 100 | Normalization | % | D10 | day 10 | Viability lost between activation and harvest, relative to D3 |  |
| `Viability_change_D3_to_D10_pct` | `Harvest Post Wash Viability Average (%)` − `Day 3 Average of Pooled Viability (%)` | Engineered | percentage points | D10 | day 10 | Net viability change from activation to the washed harvest, in percentage points |  |
| `D10 harvest - D10 wash viability change (%)` | (`Harvest Post Wash Viability Average (%)` − `Harvest Pre-Wash Cell Viability (%)`) / `Harvest Pre-Wash Cell Viability (%)` × 100 | Normalization | % | D10 | day 10 | What the LOVO wash did to viability |  |

## 5. Seeding

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `D3 % cells seeded / target` | `Total Viable Cells for Expansion after VSVg Sampling (cells)` / (`Day 3 Number of G-Rex to seed` × seeding_target_per_vessel [5e7]) × 100 | Normalization | % of target | D3 | day 3 | Cells seeded against the target for the number of vessels used; the target per vessel is in config | changed: target per vessel from config; any vessel count |
| `D3_seeding_error_pct` | \|`D3 % cells seeded / target` − 100\| | Engineered | percentage points | D3 | day 3 | Distance from target seeding in either direction, so over- and under-seeding count the same |  |
| `D3_seeding_within_5pct` | 1 if \|`D3 % cells seeded / target` − 100\| ≤ seeding_tolerance [5] else 0; missing if the input is | Engineered | 0/1 | D3 | day 3 | Whether seeding landed within ±5% of target | changed: missing when the input is |
| `D3_seeding_loss_fraction` | 1 − (`Day 3 Actual Viable Cells Seeded G-Rex A` + `Day 3 Actual Viable Cells Seeded G-Rex B`) / `Day 3 Total Viable Cells Available for Seeding G-Rex (cells)` | Normalization | fraction | D3 | day 3 | Share of the available cells that did not go into a vessel; negative when more was seeded than was recorded as available | **new** |
| `D3_dead_cells_at_seeding` | `Day 3 Total Viable Cells Available for Seeding G-Rex (cells)` × (100 − `Day 3 Average of Pooled Viability (%)`) / `Day 3 Average of Pooled Viability (%)` | Engineered | cells | D3 | day 3 | Dead cells carried into transduction alongside the viable ones; they take up vector and debris slows expansion | **new** |
| `Avg_cells_per_bag_D3` | `Day 3 Total Viable Cells Available for Seeding G-Rex (cells)` / `# of Culture Bags` | Normalization | cells per bag | D3 | day 3 | Cells per culture bag, which is the density the cells actually see |  |

## 6. Vessel balance

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `Seeded_cells_mean_AB` | mean(`Day 3 Actual Viable Cells Seeded G-Rex A`, `Day 3 Actual Viable Cells Seeded G-Rex B`) over the vessels in use | Engineered | cells | D3 | day 3 | Mean seeded viable cells across the vessels in use |  |
| `Seeded_cells_imbalance_AB` | \|`Day 3 Actual Viable Cells Seeded G-Rex A` − `Day 3 Actual Viable Cells Seeded G-Rex B`\| / (`Day 3 Actual Viable Cells Seeded G-Rex A` + `Day 3 Actual Viable Cells Seeded G-Rex B`) | Normalization | fraction | D3 | day 3 | \|A − B\| / (A + B) of the seeded viable cells: 0 for identical vessels, comparable across batch sizes; missing for a single vessel | **new** |
| `Vector_volume_mean_AB_mL` | mean(`Volume Vector Added to G-Rex A (mL)`, `Volume Vector Added to G-Rex B (mL)`) over the vessels in use | Engineered | mL | D3 | day 3 | Mean vector volume added across the vessels in use |  |
| `Vector_volume_imbalance_AB_mL` | \|`Volume Vector Added to G-Rex A (mL)` − `Volume Vector Added to G-Rex B (mL)`\| / (`Volume Vector Added to G-Rex A (mL)` + `Volume Vector Added to G-Rex B (mL)`) | Normalization | fraction | D3 | day 3 | \|A − B\| / (A + B) of the vector volume added: 0 for identical vessels, comparable across batch sizes; missing for a single vessel | **new** |

## 7. MOI

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `MOI_eff_G-Rex_A` | `LV CoA LV Titer` (else `Vector Titer (IU/mL)`) × `Volume Vector Added to G-Rex A (mL)` / `Day 3 Actual Viable Cells Seeded G-Rex A` | Normalization | IU per cell | D3 | day 3 | Infectious units per seeded cell in vessel A, from the measured titer rather than the nominal one |  |
| `log_MOI_eff_G-Rex_A` | ln(1 + `MOI_eff_G-Rex_A`) | Normalization | ln(1 + MOI) | D3 | day 3 | Vessel A MOI on a log scale, where its spread is even enough to model |  |
| `MOI_eff_G-Rex_B` | `LV CoA LV Titer` (else `Vector Titer (IU/mL)`) × `Volume Vector Added to G-Rex B (mL)` / `Day 3 Actual Viable Cells Seeded G-Rex B` | Normalization | IU per cell | D3 | day 3 | Infectious units per seeded cell in vessel B |  |
| `log_MOI_eff_G-Rex_B` | ln(1 + `MOI_eff_G-Rex_B`) | Normalization | ln(1 + MOI) | D3 | day 3 | Vessel B MOI on a log scale |  |
| `MOI_eff_pool` | `LV CoA LV Titer` (else `Vector Titer (IU/mL)`) × (`Volume Vector Added to G-Rex A (mL)` + `Volume Vector Added to G-Rex B (mL)`) / `Day 3 Total Viable Cells Available for Seeding G-Rex (cells)` (else `Day 3 Actual Viable Cells Seeded G-Rex A` + `Day 3 Actual Viable Cells Seeded G-Rex B`) | Normalization | IU per cell | D3 | day 3 | Infectious units per seeded cell over the whole batch; a vessel that was not used contributes nothing rather than making the batch missing |  |
| `log_MOI_eff_pool` | ln(1 + `MOI_eff_pool`) | Normalization | ln(1 + MOI) | D3 | day 3 | Pooled MOI on a log scale |  |
| `MOI_eff_quality_adjusted` | `MOI_eff_pool` × `LV CoA PI Ratio` | Engineered | particles per cell | D3 | day 3 | Physical vector particles per seeded cell: the infectious MOI times the lot's particles per infectious unit. Two lots at the same titer can deliver very different particle loads | changed: relabelled: physical particles per cell |
| `MOI_per_CD3_pool` | `LV CoA LV Titer` (else `Vector Titer (IU/mL)`) × (`Volume Vector Added to G-Rex A (mL)` + `Volume Vector Added to G-Rex B (mL)`) / `PT Total  CD3+ (Viable Cells)` | Normalization | IU per CD3+ cell | D3 | day 3 | Infectious units per CD3+ cell rather than per cell of any kind — only T cells are the target |  |
| `MOI_eff_pool_vs_target` | `MOI_eff_pool` / moi_target (config: features.moi_target) | Normalization | ratio to target | D3 | day 3 | Pooled MOI against the MOI the process aims for, so a low MOI by design and a low MOI by error are told apart. Needs `features.moi_target` in config; blocked until it is set | **new** |

## 8. Contact time

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `D3_contact_time_deviation` | \|`Day 3 Incubation Time (120min+/-30 min)` − 120\| | Engineered | min | D3 | day 3 | Minutes away from the 120 min transduction contact time |  |
| `D3_contact_time_within_window` | 1 if 90 ≤ `Day 3 Incubation Time (120min+/-30 min)` ≤ 150 else 0; missing if the time is missing | Engineered | 0/1 | D3 | day 3 | Whether contact time stayed inside 120±30 min; missing when the time was not recorded | changed: missing when the time is missing |
| `D3_contact_time_deviation_G-Rex_A` | \|`Amount of Time G-Rex A in Incubator (120min+/-30 min)` − 120\| | Engineered | min | D3 | day 3 | Minutes away from the 120 min transduction contact time for vessel A |  |
| `D3_contact_time_within_window_G-Rex_A` | 1 if 90 ≤ `Amount of Time G-Rex A in Incubator (120min+/-30 min)` ≤ 150 else 0; missing if the time is missing | Engineered | 0/1 | D3 | day 3 | Whether contact time stayed inside 120±30 min; missing when the time was not recorded | changed: missing when the time is missing |
| `D3_contact_time_deviation_G-Rex_B` | \|`Amount of Time G-Rex B in Incubator (120min+/-30 min)` − 120\| | Engineered | min | D3 | day 3 | Minutes away from the 120 min transduction contact time for vessel B |  |
| `D3_contact_time_within_window_G-Rex_B` | 1 if 90 ≤ `Amount of Time G-Rex B in Incubator (120min+/-30 min)` ≤ 150 else 0; missing if the time is missing | Engineered | 0/1 | D3 | day 3 | Whether contact time stayed inside 120±30 min; missing when the time was not recorded | changed: missing when the time is missing |

## 9. Incubation

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `Thaw_to_incubator_D0_min` | `Date and Time Cells placed in Incubator Day 0` − (date of `Date and Time Cells placed in Incubator Day 0` + time of day of `Thaw Start Time`), in minutes; missing if negative | Engineered | min | D0 | day 0 | Minutes the thawed cells waited before the day 0 incubator; a thaw recorded after that time is left missing rather than guessed | **new** |
| `Incubation D0-D1 (min)` | (`Date and Time Cells Removed From Incubator Day 1` − `Date and Time Cells placed in Incubator Day 0`) in minutes | Engineered | min | D1 | day 1 | Minutes the cells spent in the incubator between day 0 and day 1 |  |
| `Incubation D1-D3` | (`Date and Time Cells Removed From Incubator Day 3` − `Date and Time Cells placed in Incubator Day 1`) in minutes | Engineered | min | D3 | day 3 | Minutes in the incubator between day 1 and day 3 |  |
| `Incubation D3-D6 (min)` | (`Date and Time Cells Removed From Incubator Day 6` − `Day 3 Date and Time G-Rex A Returned to Incubator`) in minutes | Engineered | min | D6 | day 6 | Minutes in the incubator between day 3 and day 6 |  |
| `Incubation D6-D8 (min)` | (`Date and Time Cells Removed From Incubator Day 8` − `Day 6 Date and Time G-Rex A Returned to Incubator`) in minutes | Engineered | min | D8 | day 8 | Minutes in the incubator between day 6 and day 8 |  |
| `Incubation D8-D10 (min)` | (`Date and Time Cells Removed From Incubator Day 10` − `Day 8 Date and Time G-Rex A Returned to Incubator`) in minutes | Engineered | min | D10 | day 10 | Minutes in the incubator between day 8 and day 10 |  |
| `Incubation_total_D3_to_D10_min` | `Incubation D3-D6 (min)` + `Incubation D6-D8 (min)` + `Incubation D8-D10 (min)` (missing if any is) | Engineered | min | D10 | day 10 | Total time in the incubator across expansion. Missing when any of the three intervals is: a partial sum would read as a short incubation | changed: missing rule documented |
| `Out_of_incubator_D3_min` | (`Day 3 Date and Time G-Rex A Returned to Incubator` − `Date and Time Cells Removed From Incubator Day 3`) in minutes; missing if negative | Engineered | min | D3 | day 3 | Handling time outside 37 °C on D3; the incubation features only measure time inside | **new** |
| `Out_of_incubator_D6_min` | (`Day 6 Date and Time G-Rex A Returned to Incubator` − `Date and Time Cells Removed From Incubator Day 6`) in minutes; missing if negative | Engineered | min | D6 | day 6 | Handling time outside 37 °C on D6; the incubation features only measure time inside | **new** |
| `Out_of_incubator_D8_min` | (`Day 8 Date and Time G-Rex A Returned to Incubator` − `Date and Time Cells Removed From Incubator Day 8`) in minutes; missing if negative | Engineered | min | D8 | day 8 | Handling time outside 37 °C on D8; the incubation features only measure time inside | **new** |

## 10. Clumps

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `Clump_count_reduction_fraction` | 1 − `Post-Massaging: # of Clumps` / `Pre-Activation Clumps: # of Clumps before massage` | Normalization | fraction | D1 | day 1 | Share of the clumps that massaging removed, which measures the intervention rather than the starting material |  |
| `Clump_size_reduction_fraction` | 1 − `Post-Massaging: Size of Clumps` / `Pre-Activation Clumps: Size of Clumps before massage` | Normalization | fraction | D1 | day 1 | Share of the clump size that massaging removed |  |
| `Clump_burden_weighted_pre` | `Pre-Activation Clumps: # of Clumps before massage` × `Pre-Activation Clumps: Size of Clumps before massage` | Engineered | count × size | D1 | day 1 | Count and size together, so many small clumps and one large one are not recorded as the same burden |  |
| `Clump_burden_weighted_post_massage` | `Post-Massaging: # of Clumps` × `Post-Massaging: Size of Clumps` | Engineered | count × size | D1 | day 1 | Size-weighted clump burden left after massaging |  |
| `Clump_burden_weighted_post_mixing` | `Post-Mixing Clumps: # of Clumps` × `Post-Mixing Clumps: Size of Clumps` | Engineered | count × size | D1 | day 1 | Size-weighted clump burden left after mixing |  |
| `Mixing_cycles_x_pre_activation_clump_count` | `Day 1 Number of Mixing Cycles Post Activation` × `Pre-Activation Clumps: # of Clumps before massage` | Engineered | cycles × count | D1 | day 1 | Mixing effort against the burden it was applied to; neither number means much without the other |  |
| `D3_clumps_flag` | 1 if `day 3 clump pattern` records clumps (anything but no/none/absent/…), else 0 | Engineered | 0/1 | D3 | day 3 | Whether clumps were still recorded at day 3 |  |
| `D10_clump_bag_count` | Σ over `Day 10 Clump Present Bag 1` … `Day 10 Clump Present Bag 6` of (clumps present) | Engineered | bags | D10 | day 10 | How many harvest bags had clumps, counted over the bags the batch actually filled |  |

## 11. Dose and CAR

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `Dose (CAR+ viable cells/kg) all weights` | `Dose (CAR+ viable cells/kg)` / 1e6 if present, else `Dose: Number of CAR+ Viable T-Cells (cells)` / 1e8 | Normalization | multiple of nominal dose | D10 | day 11 | One dose column for every batch, as a multiple of the nominal dose: the weight-based dose in E6 cells/kg, a fixed dose in units of 1E8 cells. Read it with the dose basis, which says which of the two a batch has | changed: documented as a multiple of the nominal dose; read with the dose basis |
| `Dose_basis` | 'per kg' if `Dose (CAR+ viable cells/kg)` is present, 'flat' if only `Dose: Number of CAR+ Viable T-Cells (cells)` is, else missing | Engineered | categorical | D10 | day 11 | Whether the batch record holds a weight-based or a fixed dose | **new** |
| `Post formulation dose` | `Final Formulation CS5 Viable Cell Concentration Average (cells/mL)` / 1e6 × `Volume per Bag (mL)` × `Harvest Pre-Wash Flow CAR+ Expression (%)` / 100 / `Subject Weight (kg)` | Normalization | E6 CAR+ cells/kg | D10 | day 11 | Dose per kilogram implied by what actually went into the bag |  |
| `D10 Achievable Dose (E6 CAR+ viable T-cells)` | `Harvest Pre-Wash Flow CAR+ Expression (%)` / 100 × `Harvest Post Wash Viable Cell Count (cells)` / (`Subject Weight (kg)` × 1e6) | Normalization | E6 CAR+ cells/kg | D10 | day 10 | The dose the harvest could have supported, against the dose that was given |  |
| `D10 Total CAR+ viable T-cells achieved` | `Harvest Pre-Wash Flow CAR+ Expression (%)` × `Harvest Post Wash Viable Cell Count (cells)` / 100 | Engineered | cells | D10 | day 10 | Absolute CAR+ cells at harvest |  |
| `CAR T produced per cell engaged D3` | `Harvest Pre-Wash Flow CAR+ Expression (%)` / 100 × 2^`cPDL (D3 to D10)` | Normalization | CAR+ cells per cell seeded | D10 | day 11 | CAR+ cells returned per cell seeded, which folds expansion and transduction into one yield |  |
| `VCN/cell` | `FP Flow CAR+ (%)` × `Provirus Vector Copy Number (copies/transduced cell)` / 100 | Normalization | copies per cell | Final product | day 11 | Copies per cell across the whole product rather than per transduced cell |  |
| `Flow accuracy (effective)` | `FP Flow CAR+ (%)` / `Harvest Pre-Wash Flow CAR+ Expression (%)` | Engineered | ratio | Final product | day 11 | Final product CAR+ against the harvest reading — a ratio far from one points at the assay, not the process |  |

## 12. Formulation and wash

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `Total CS5 cell concentration (E6 cells/mL)` | `Final Formulation CS5 Viable Cell Concentration Average (cells/mL)` / (`Final Formulation CS5 Viability Average (%)` / 100) / 1e6 | Engineered | E6 cells/mL | D10 | day 10 | Total cells in the formulated suspension, dead ones included |  |
| `total viable cells used for formulation` | `Harvest Post Wash Viable Cell Concentration Average (cells/mL)` × `Volume of Cells Used for Formulation (mL)` | Engineered | cells | D10 | day 10 | Viable cells that went into formulation |  |
| `total viable cells in formulated suspension` | `Final Formulation CS5 Viable Cell Concentration Average (cells/mL)` × `Volume CS5 Used for Formulation (mL)` | Engineered | cells | D10 | day 10 | Viable cells that came out of formulation |  |
| `number of viable cells lost during formulation` | `total viable cells used for formulation` − `total viable cells in formulated suspension` | Engineered | cells | D10 | day 10 | The difference between those two, as cells |  |
| `% cells lost during formulation` | `number of viable cells lost during formulation` / `total viable cells used for formulation` × 100 | Normalization | % | D10 | day 10 | The same loss as a share, which is comparable across batch sizes |  |
| `total cells lost during LOVO wash (cells)` | `Harvest Pre-Wash Viable Cell Count after Sampling (cells)` − `Harvest Post Wash Viable Cell Count (cells)` | Engineered | cells | D10 | day 10 | Cells the wash did not return |  |
| `LOVO_wash_recovery_fraction` | `Harvest Post Wash Viable Cell Count (cells)` / `Harvest Pre-Wash Viable Cell Count after Sampling (cells)` | Normalization | fraction | D10 | day 10 | Share of the viable harvest cells the wash returned, comparable across batch sizes where the count of lost cells is not | **new** |

## 13. Vector CoA

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `LVCoA_Impurity_Index_zsum` | Σ z-score(x) over `LV CoA HCP`, `LV CoA pDNA`, `LV CoA cDNA`, `LV CoA E1a cDNA`, `LV CoA E1b cDNA`, z-scored within one site's batches | Normalization | sum of z-scores | Vector CoA | day 3 | Residual impurities on one scale, summed. Standardised within the site being processed, not across the merged batch table, so the two sites' values are not directly comparable. Exploratory only; the spec-normalised index is the one a model can use |  |
| `LVCoA_Impurity_Index_spec` | Σ x / spec_limit(x) over `LV CoA HCP`, `LV CoA pDNA`, `LV CoA cDNA`, `LV CoA E1a cDNA`, `LV CoA E1b cDNA` (config: features.impurity_spec_limits) | Normalization | sum of fractions of spec limit | Vector CoA | day 3 | Each residual impurity as a share of its release limit, summed. Fixed references rather than the data's own spread, so one new lot gets the same value whatever it is processed with. Needs `features.impurity_spec_limits` in config; blocked until it is set | **new** |
| `LV_age_at_use_days` | (`Date of Manufacture Start` − `LV CoA MFG Date`) in days | Engineered | days | D3 | day 3 | Days between the vector lot's manufacture and the start of the batch | **new** |

## 14. IL-2

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `IL2_IU_total_D6` | `Day 6 Activity of IL-2 (IU/mg)` × `Day 6 IL-2 Protein Content (µg/vial)` / 1000 | Engineered | IU per vial | D6 | day 6 | IL-2 units fed at day 6: activity and protein mass mean little apart |  |
| `IL2_IU_total_D8` | `Day 8 Activity of IL-2 (IU/mg)` × `Day 8 IL-2 Protein Content (µg/vial)` / 1000 | Engineered | IU per vial | D8 | day 8 | IL-2 units fed at day 8 |  |

## 15. Context

| Feature | Formula | Kind | Unit | Day | Known | Description | Status |
|---|---|---|---|---|---|---|---|
| `Shift` | 'Morning' if time of day of `Thaw Start Time` < shift_cutoff [11:59], else 'Afternoon'; missing if no time | Engineered | categorical | D0 | day 0 | Which shift started the batch, from the thaw time and the cut-off in config; no thaw time, no shift | changed: cut-off from config; missing thaw time → no shift |

## 16. Environment (CO₂ and temperature setpoints)

One feature per column whose name contains `CO2 Saturation (%)` or
`Temperature (°C)`, so the set depends on what each site records.
The catalog lists 28 of them (family *Deviation from reference*).

| Feature | Formula | Kind | Unit | Description |
|---|---|---|---|---|
| `<column> dev_from_5.0pct` | \|<CO₂ reading> − co2_target [5.0]\| | Engineered | percentage points | Distance from the CO₂ setpoint. |
| `<column> dev_from_37.0C` | \|<temperature reading> − temp_target [37.0]\| | Engineered | °C | Distance from the temperature setpoint. |

## 17. Vector certificate columns (loader)

Built while the certificate workbook is read (`ml.io.load_lv_coa`), prefixed `LV CoA `.
Their lineage is declared in `features.coa_lineage()`, so a task can check them for
descent from its target. In the catalog these are *Calculated By* `pipeline (vector
certificate loader)`. The catalog lists all of them but the `below LOQ` flags and the
donor `AVG`, which the loader builds only when the certificate has no average of its own.

| Column | Formula | Kind | Description |
|---|---|---|---|
| `LV CoA Delivered Dose per GRex` | `LV Titer (unadjusted)` (else `LV Titer`) × `Volume` | Engineered | Delivered functional dose per G-Rex. **Not sure** whether `Volume` is the volume added to the G-Rex or the fill volume of the lot. |
| `LV CoA Delivered Dose used adj titer` | 1 if `LV Titer (unadjusted)` is missing, else 0 | Engineered | Rows where the delivered dose fell back to the adjusted titer (possible potency back-channel). |
| `LV CoA Titer Adjustment Factor` | `LV Titer` / `LV Titer (unadjusted)` | Normalization | How hard the release titer was corrected. |
| `LV CoA Specific Infectivity` | `LV Titer` / `Physical Titer p24 ELISA` | Normalization | Infectious titer per physical particle. |
| `LV CoA E1b Fraction of Total DNA` | `E1b cDNA` / Σ(`pDNA`, `cDNA`, `E1a cDNA`, `E1b cDNA`) | Normalization | Last residual-DNA species as a share of all residual DNA. Lot-fingerprint **leakage risk**. |
| `<impurity> below LOQ` | 1 if the raw value is written `<x`, else 0 | Engineered | Left-censoring flag. **Changed:** the value itself now follows `cleaning.censored` (`limit` → x, `half` → x/2, `missing`), the same policy as every other source; it used to be x/2 regardless. |
| `pH Abs Deviation` | \|pH − lv_coa.ph_nominal [7.2]\| | Engineered | Two-sided deviation from the spec midpoint. |
| `Osmolality Abs Deviation` | \|osmolality − lv_coa.osmo_nominal [300]\| | Engineered | Same for osmolality. |
| `<metric> Donor MAX` / `MIN` | max / min of donor 1 and donor 2 | Engineered | Best and weakest reference donor (CAR% and IFNg). |
| `<metric> Donor DIFF` | max − min | Engineered | Disagreement between the donors. |
| `<metric> Donor DIFF Rel` | DIFF / mean(donor 1, donor 2) | Normalization | Disagreement relative to the mean. |
| `<metric> Donor AVG` | mean(donor 1, donor 2) | Engineered | Only when the certificate has no average of its own. |

`LV CoA MFG Date` is no longer dropped by the loader: the vector age is computed
from it. It is in the catalog as a datetime with *Shift Model Use* `Exclude (leakage)`.

## 18. Censoring flags (cleaning stage)

| Column | Formula | Kind | Description |
|---|---|---|---|
| `<parameter> censored` | 1 if the raw value was written `<x`, else 0 | Engineered | Only with `cleaning.censored_flags: true`. The value itself follows `cleaning.censored`. |

## 19. Task-stage derived columns

Not in the batch table; created when a task runs.

| Column | Formula | Kind | Description |
|---|---|---|---|
| `<label> group` (clustering `summarise`) | mean over members of sign × (x − mean) / sd | Normalization | One score for a stable group of correlated parameters. When two groups share a label the second is now `<label> NN group` (it used to lose the `group` suffix). |
| `<column>_resid` (clustering `residualise`) | residual of a linear regression on the representative | Engineered | The part of a column the kept parameter does not explain. |
| One-hot / binary / ordinal / target-mean / hashed encodings | see `ml.encode` | — | Encodings of categorical parameters; target means are fitted on training rows only. |

## 20. Config settings the features read

All under `features:` in `config.yaml`.

| Setting | Default | Used by |
|---|---|---|
| `co2_target, temp_target` | 5.0, 37.0 | setpoint deviations |
| `contact_target, contact_tolerance` | 120, 30 | contact-time deviation and window |
| `seeding_target_per_vessel` | 50 000 000 | `D3 % cells seeded / target` |
| `seeding_tolerance` | 5 | `D3_seeding_within_5pct` |
| `cpdl_low, cpdl_high, lactate_low, lactate_high` | 4, 5, 12.5, 30 | growth profile |
| `cd4_low, cd4_high` | 40, 60 | CD4 profile |
| `shift_cutoff` | 11:59 | `Shift` |
| `moi_target` | unset | `MOI_eff_pool_vs_target` (blocked while unset) |
| `medium_glucose, medium_lactate` | unset | lactate yield on glucose (blocked while unset) |
| `impurity_spec_limits` | unset | `LVCoA_Impurity_Index_spec` (blocked while unset) |

## 21. Change log

What this revision changed, following `FEATURE_RECOMMENDATIONS.md`.

**Removed** (exact or near duplicates):

- `Harvest Lactate Normalized`: identical to `D10 Lact / E6 cells`. The growth
  profile now reads the latter (or the former when a source supplies it). In the catalog it
  is kept as a source parameter in the redundancy group of the D10 column, not representative.
- `Viability change during expansion` (exact negative of `D3-D10 viability change (%)`) and
  `Drop viability D3 vs D0 (%)` (relative twin of `Viability_drop_D3_vs_D0_pct`): no longer
  computed; their catalog rows stay as source parameters (already non-representative).
- Vessel statistics `min`, `max`, `std`, `cv` for seeded cells and vector volume: with two
  vessels they follow from the mean and the difference. Replaced by one `imbalance` each;
  the eight catalog rows are deleted.

**Fixed:** see the *Status* column. In the catalog, `VCN/cell` and `Flow accuracy (effective)`
were marked known at day 10 although they are calculated from final-product results (day 11),
and `Shift` at day −1 although the thaw time is a day-0 reading; all three are corrected.

**Added:** every row marked **new** above, a unit and a kind on every feature, and formula
tests (`tests/test_features.py`).

**Removed later:** the build report's *Derived features* section with its data checks
(doublings, seeding and formulation arithmetic) and the availability check
(`features.availability_column`, `availability_order`), and the certificate loader's
`<column> missing` indicators.

**Overlap with source columns.** Some new features measure what the batch records already
deliver calculated. They are in the catalog in the same redundancy group, not representative,
so the source column stays the default; compare the two before relying on either:

| New feature | Source column in the same group |
|---|---|
| `Thaw_to_incubator_D0_min` | `Day 0 Processing Time (aph thaw to inc in)` |
| `Out_of_incubator_D3_min` | `Day 3 Processing Time (inc out to inc in)` |
| `Out_of_incubator_D6_min` | `Day 6 Processing Time (inc out to inc in)` |
| `Out_of_incubator_D8_min` | `Day 8 Processing Time (inc out to inc in)` |
| `LOVO_wash_recovery_fraction` | `LOVO Step Yield (%) (as a fraction rather than %)` |

## 22. Open questions

1. **The `1e9` factor** in the metabolites per cell is kept as the source spreadsheets
   define it; its exact unit derivation is **not sure**.
2. **Lactate yield on glucose** is only valid while no feed has changed the medium. If the
   D6/D8 IL-2 feeds add medium, the fresh-medium values have to account for it.
3. **`LV CoA Delivered Dose per GRex`**: which volume `Volume` is (**not sure**).
4. **`Dose (CAR+ viable cells/kg) all weights`** assumes a flat dose of 1E8 cells is the nominal dose; **not sure**.
5. **The post-thaw (`PT`) flow readings** are dated day 3 in the catalog and the post-
   enrichment (`PE`) ones day 0, so the CD4:CD8 shift is PT − PE. Confirm that order.
6. **Catalog value type `TODO`** on one row (pre-existing) is not one the pipeline knows.
7. **The "Calculated" kind of source-record columns** is assigned from the parameter name
   (yield, recovery, change, per G-Rex, ratio, …); review the 129 source rows if it matters.
