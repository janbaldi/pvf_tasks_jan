# Feature engineering and normalisation: recommendations

A second, critical pass over the earlier suggestions (chat, and the "not sure" notes
in `CALCULATED_FEATURES.md`), checked against the code on `main`, the parameter names
in `vocabulary.csv` and the two demo tasks.

Names are the real parameter names from `vocabulary.csv`. Feature names in *italics*
are proposals; pick the final name in the vocabulary.

---

## 0. What the recommendations are judged against

The models these features feed set the bar:

- **Small cohorts.** One site, tens of batches, cross-validated by vector lot
  (`roles.group: Vector lot`). Each extra feature costs degrees of freedom and
  multiple-testing noise. A feature has to carry information the table does not
  already hold.
- **Tree models and rank statistics.** Clustering uses Spearman correlation, and the
  certificate code is written with XGBoost in mind. Both are unaffected by any
  monotone transform (log, scaling, unit conversion) of a *single* column. Those
  transforms only matter for linear models, plots and readability.
- **Availability decides leakage.** The expression task drops everything
  `Available At` Post-thaw, Wash, Formulation, Final product and Release. A feature is
  only usable if it is computed from inputs available before the target exists.

Under that bar, the most useful work is to remove redundancy and fix what is wrong,
and then to add a few features with **new information**: ratios or differences across
columns that a tree cannot build with axis-aligned splits.

---

## 1. Verdict on the earlier suggestions

| Earlier point | Verdict | Why |
|---|---|---|
| CD3 % used without `/100` in `PT Total  CD3+ (Viable Cells)` | **Agree, low impact** | 100× too large, so the meaning and units are wrong. A constant factor does not change a tree or a Spearman correlation. Fix it for meaning. |
| `D3 % cells seeded / target` is a ratio but compared with 100 | **Wrong** | `Total Viable Cells for Expansion after VSVg Sampling (cells) / 500 000` is the percentage of a 50 M target for one G-Rex (`/1 000 000` for 100 M with two vessels). So it *is* a percentage and the `−100` comparison is right. The target numbers (50 M per vessel) should move to config. |
| `Dose (CAR+ viable cells/kg) all weights` mixes scales | **Partly agree** | `/1e6` gives E6 cells/kg. `/1e8` gives the fraction of a 1E8 flat dose. With a 100 kg weight cap both read "1.0 = nominal dose", which looks intentional. A *dose basis* flag (per-kg vs flat) is still missing, so the column mixes two definitions without saying which. |
| `MOI_eff_quality_adjusted` direction is doubtful | **Resolved: keep, relabel** | The code says `PI Ratio` is about 1e4 (particles per infectious unit). So MOI × PI ratio = **physical particles per cell**, a meaningful quantity. The rationale text ("infectious share") is what's wrong, not the formula. |
| Per-cm² G-Rex density | **Drop** | The surface area is fixed per vessel, so this is a constant rescaling of existing columns. It adds no information. |
| Log transforms for skewed columns | **Drop for the tasks** | They are invisible to trees and to Spearman. Only add them if a linear model is introduced. |
| Doubling time | **Drop** | Monotone in `cPDL (D3 to D10)`, so it's the same feature. |
| Specific metabolic rates per cell per day | **Not possible** | No cell counts exist between D3 and D10. |
| Particle-to-infectivity ratio | **Drop** | Already present: `PI Ratio` and its inverse, `Specific Infectivity`. |
| Wash recovery, vector age, timing, CD4/CD8 shift, seeded-vs-available | **Keep** | Detailed below. |
| Fit the impurity index on training rows | **Keep** | It is the only site-relative feature, and it leaks validation-row statistics as built. |
| Units on `Feature`, formula tests | **Keep** | The unit problems above would have been caught. |

---

## 2. Clean up the existing registry first

### 2.1 Remove exact or near duplicates

At this sample size duplicates dilute importance and inflate the clusters that
`decorrelate` then has to undo.

| Remove | Duplicate of | Relation |
|---|---|---|
| `Harvest Lactate Normalized` | `D10 Lact / E6 cells` | Identical for 1 or 2 vessels (`lac × 1e9 × vessels / cells`). Keep one, and point `growth profile` at it. |
| `Viability change during expansion` | `D3-D10 viability change (%)` | Exactly the negative. Keep it in the report if the Power Query comparison needs it, but drop it as a predictor. |
| `Drop viability D3 vs D0 (%)` **or** `Viability_drop_D3_vs_D0_pct` | each other | The relative and absolute change carry almost the same ranking when the starting viability is high. Keep the points version. |
| 5 spread statistics × 2 (`Seeded_cells_{statistic}_AB`, `Vector_volume_{statistic}_AB_mL`) | — | With two values, min, max, std and mean are redundant (`std = |A−B|/2`, `min = mean − std`). Keep **mean** and one imbalance measure, `|A−B| / (A+B)`. That is 4 columns instead of 10. |
| `Log (PT CD4:CD8)` vs `Post Thaw FLOW CD4+ (%)` | — | Not a duplicate, but strongly rank-correlated. No action; `decorrelate` handles it. |

### 2.2 Fix

1. **CD3 percentage units.** Add `/100` in `PT Total  CD3+ (Viable Cells)`, which also
   corrects `MOI_per_CD3_pool`.
2. **Vessel count.** In the lactate and glucose per-cell features, return NaN when
   `Day 3 Number of G-Rex to seed` is not 1 or 2, instead of treating it as 2.
3. **`D3_contact_time_within_window*`.** Return NaN, not 0, when the contact time is
   missing.
4. **`Incubation_total_D3_to_D10_min`.** Decide and document the rule. A plain sum
   makes one missing interval a missing total, which may be right, but say so.
5. **`Shift`.** Guard against a missing `Thaw Start Time`. Today a NaN probably errors
   the whole feature for the site.
6. **Relabel `MOI_eff_quality_adjusted`** as physical particles per seeded cell (see §1).
7. **Hard-coded process constants to config:** the 50 M/100 M seeding targets, the
   11:59 shift cut-off, and the `×2` harvest-vessel assumption for D10 lactate and
   glucose.

### 2.3 Two censoring policies for the same thing

`cleaning.censored` (default `limit`) rewrites `<x` as `x` for batch-record columns.
The certificate loader independently rewrites `<x` as `x/2` for impurities. The same
"below the limit of quantification" ends up as two different numbers depending on
the source. Route the certificate through the same setting, or document why
impurities differ.

---

## 3. New features, ranked

Each proposal brings information the table does not already hold in any single
column. The stage is the latest stage of its inputs, which is what its catalog
`Available At` must be.

### Tier 1: cheap, plausible, available early (worth adding now)

| Proposed feature | Formula | Stage | Why |
|---|---|---|---|
| *Seeding loss* | `1 − (Day 3 Actual Viable Cells Seeded G-Rex A + … G-Rex B) / Day 3 Total Viable Cells Available for Seeding G-Rex (cells)` | D3 | How much of the available material actually went into the vessels. It is not in any single column. |
| *Dead cells at transduction* | `Day 3 Total Viable Cells Available for Seeding G-Rex (cells) × (100 − Day 3 Average of Pooled Viability (%)) / Day 3 Average of Pooled Viability (%)` | D3 | Dead cells and debris compete for the vector and slow expansion. The table holds only viable counts and viability separately, so a tree cannot form this product. |
| *Thaw to incubator (min)* | `Date and Time Cells placed in Incubator Day 0 − Thaw Start Time` | D0 | Time thawed cells spend at room temperature. `Thaw Start Time` is only used for `Shift` today. |
| *Time out of incubator D3 / D6 / D8 (min)* | `Day 3 Date and Time G-Rex A Returned to Incubator − Date and Time Cells Removed From Incubator Day 3` (same for D6, D8) | D3 / D6 / D8 | Handling time outside 37 °C. The registry measures time *in* the incubator only. |
| *Lactate increment D6→D8, D8→D10* | `D8 Lact / E6 cells − D6 Lact / E6 cells`, `D10 Lact / E6 cells − D8 Lact / E6 cells` | D8 / D10 | The slope averages away *when* metabolism takes off. Late acceleration or a plateau are both biologically distinct. Same for glucose if the increments are not too noisy. |
| *Vector physical dose per cell* | relabel of `MOI_eff_quality_adjusted` (§2.2) | D3 | Already computed; listed for completeness. |

### Tier 2: plausible, needs a small code or data change

| Proposed feature | Formula | Stage | Needs |
|---|---|---|---|
| *Vector age at use (days)* | `Date of Manufacture Start − MFG Date` | D3 | The certificate loader drops `MFG Date`. Keep it, prefixed, for this feature. |
| *MOI deviation from plan* | `MOI_eff_pool / target MOI` | D3 | A target MOI (or planned vector volume) in config. Distinguishes "the MOI was low by design" from "it was low by error". |
| *Dose basis* | `per-kg` if `Dose (CAR+ viable cells/kg)` is present, `flat` if only `Dose: Number of CAR+ Viable T-Cells (cells)` | Release | Makes `Dose (CAR+ viable cells/kg) all weights` interpretable. |
| *CD4:CD8 shift through selection* | `Log (PE CD4:CD8) − Log (PT CD4:CD8)` | D0 (after selection) | The change says how selection reshaped the material, which neither ratio says on its own. Check the catalog stage of the post-wash flow columns first; the expression task drops Post-thaw. |
| *Wash recovery* | `Harvest Post Wash Viable Cell Count (cells) / Harvest Pre-Wash Viable Cell Count after Sampling (cells)` | Wash | A ratio is comparable across batch sizes, unlike `total cells lost during LOVO wash (cells)`. Useful for the release task, excluded for expression. |
| *Lactate-on-glucose yield* | `(lactate_t − lactate_fresh) / (glucose_fresh − glucose_t)` | D6–D10 | The best biomass-independent signal of metabolic state (glycolytic vs. oxidative), and it doesn't need cell counts. It **requires** fresh-medium glucose/lactate and the feed schedule (the D6/D8 IL-2 feeds probably add medium). Without those it's wrong, so treat it as a process question first. |

### Tier 3: quality checks rather than predictors

Compute these in the build report, not as model inputs.

| Check | Formula | Purpose |
|---|---|---|
| cPDL consistency | `cPDL (D3 to D10)` vs `log2(Harvest Pre-Wash Viable Cell Count after Sampling (cells) / Total Viable Cells for Expansion after VSVg Sampling (cells))` | Flags transcription errors in either column. |
| Seeding arithmetic | `G-Rex A + G-Rex B` vs `Day 3 Total Viable Cells Available for Seeding G-Rex (cells)` | Batches where the seeded total exceeds what was available. |
| Formulation balance | `% cells lost during formulation` < 0 | Physically impossible gains point at a unit or volume error. |

---

## 4. Normalisations

| Normalisation | Recommendation |
|---|---|
| **Impurity index z-score** | Fit the mean and standard deviation on training rows inside the task, store them in the recipe, and apply them to validation rows. Today it is fitted on the whole site, which leaks. Or replace it with a fixed-reference version (for example each impurity divided by its release-spec limit), which needs no fitting and can be computed for a single new batch. **Preferred: spec-limit normalisation.** |
| **Per-seeded-cell metabolites** | Keep, but note the limit: lactate per *seeded* cell mixes growth and metabolism, because the cells expand ~2^cPDL-fold. Don't add more per-cell variants. The lactate-on-glucose yield (§3) is the better normalisation once the medium facts are known. |
| **Absolute counts vs fractions** | For each cell population, keep one absolute count and the fractions, not every product. The registry already has several overlapping CD4 products (`PT CD3+CD4+ (%)`, `PT Total CD4+ (Viable Cells)`). |
| **Per-kg normalisation** | Already applied where it matters (dose). Don't normalise process parameters by patient weight: the process doesn't see the weight. |
| **Unit scaling and log transforms** | Not needed for trees or Spearman. Add them only alongside a linear model, and then in the task (`encode`), not the batch table, so the stored values stay in their physical units. |
| **Cross-site standardisation** | Not needed while every task is one site. If a pooled task is planned, standardise per site *inside* the task (fitted on training rows), as `decorrelate` already does for correlations. Never in the batch table. |
| **Censored values** | One policy for all sources (§2.3). |

---

## 5. Guardrails worth more than any single feature

1. **Availability check.** A feature's `Available At` is entered by hand in the
   catalog, and nothing checks it against its inputs. Add a build-time check that each
   feature's stage is no earlier than the latest stage of its `requires`. Without it,
   one wrong catalog cell puts a post-harvest quantity into the expression model.
2. **A `unit` field on `Feature`,** shown in the report. Most problems in §2 are unit
   problems.
3. **Formula tests.** One small test per group with hand-computed inputs and outputs.
4. **Task-level note.** The expression task keeps `Harvest Pre-Wash Flow CAR+ Expression (%)`
   as a predictor while the target is `FP Flow CAR+ (%)`. That is legitimate (harvest
   comes before the final product) but it will dominate the model. Run the task with
   and without it, or the process parameters will look unimportant.

---

## 6. Suggested order

1. §2 clean-up and fixes, with tests (removes about 10 columns, fixes 6).
2. §5.1 availability check.
3. Tier 1 features (about 8 columns, all from existing data).
4. Spec-limit impurity normalisation.
5. Tier 2, as the process questions (target MOI, medium composition, feed schedule)
   get answered.
