# ml — training datasets from CAR-T manufacturing data

Turn the batch records of both manufacturing sites into one checked table, then
cut it into a documented, reproducible training dataset for each question you
want a model to answer — and explain that model by what the process actually
does, not by whichever correlated column a fit happened to pick.

```bash
ml check-catalog     # does the parameter catalog still cover what the sources record?
ml build             # every batch, both sites, cleaned, in one batch table
ml task new q3       # a folder for one question
ml task run          # a training dataset, its report, and every decision behind it
```

**What you get from one task run:** `dataset.csv` (one row per batch: id, group,
predictors, target), an encoded version ready for a model, the reason every
parameter is in or out, a report, groups of correlated parameters to explain a
model by, and a copy of the inputs so the run can be redone exactly.

**What it does not do:** fit models or split data into train and validation sets.
That is the job of the tool you fit models with; this one hands it a clean table
and the means to refit the preprocessing inside every fold.

**Four words used throughout**

| Word | What it is |
|---|---|
| **catalog** | `catalog.xlsx` (formerly the PTF): every parameter the batch table may hold, with metadata columns such as value type and process stage |
| **batch table** | `batch_table.xlsx` (formerly the PVF): one row per batch, both sites, cleaned, with derived features |
| **task** | one question asked of the batch table, written as a `task.yaml` in a folder of its own |
| **run** | one execution of a task: a folder under the task's `runs/` holding the dataset and everything about it |

**Contents:** [Install](#install) · [Tutorial](#tutorial-from-source-files-to-an-explained-model) ·
[Use your own data](#use-your-own-data) · [Reference](#reference) ·
[Development](#development) · [License](#license)

## Install

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/janbaldi/pvf_tasks_jan.git
cd pvf_tasks_jan
uv sync                     # installs the exact versions in uv.lock
uv run ml --help
```

Run every command as `uv run ml …` from inside the repository or any folder
below it, such as a task folder. Reading from SharePoint needs the internal
`io_sharepoint` package, and joining Acridine Orange readings needs
`databricks_query_tool`; both are optional, and without them everything runs
from local files.

## Tutorial: from source files to an explained model

This tutorial takes about 20 minutes. It uses the demo: made-up data with the
same shape as the real sources, so every command below runs as written and
prints what is shown. Run IDs (such as `20261012-170843-61f466`) and timings
will differ on your machine.

You will build the batch table, ask one question of it — *which manufacturing
parameters go with CAR+ expression at Ghent?* — refine the dataset four times,
fit a model, find a leak through the model's explanation, and fix it.

### 1. Make the demo data

From the repository root:

```bash
uv run python -m ml.demo demo
cd demo
```

```
config.yaml       paths to the sources, the catalog and the batch table
corrections.yaml  known errors in the sources, as rules
data/raw/         the made-up sources and catalog.xlsx
tasks/            two finished example tasks
```

Every command from now on is run inside `demo/`, where `config.yaml` is.

### 2. Check the catalog and build the batch table

First see whether the configuration is sound. Nothing is read or written yet:

```bash
uv run ml validate
```

```
OK  cli  Corrections table: 14 rules
OK  cli  Everything checked out
```

Then compare every source with the catalog:

```bash
uv run ml check-catalog
```

```
Ghent PHF: columns the catalog does not list (1)
  - Local Comment
Raritan batch data: columns the catalog does not list (2)
  - Local Site Code
  - CMD Comment
Investigations Power Query: columns the catalog does not list (3)
  - Query[Investigation ID]
  - Query[Root Cause Category]
  - Days to Close
```

These are columns the sources record that the batch table will not carry. On
real data this is your to-do list: add the ones you want to `catalog.xlsx`, then
build. The list is also in `data/processed/new_parameters.csv`, and
`catalog_report.html` shows it with every source side by side.

```bash
uv run ml build
```

```
WARN  cli  The last `ml check-catalog` found 6 parameters the sources record and the catalog does not list; …
Written — 70 batches × 221 columns
  Batch table            …/demo/data/processed/batch_table.xlsx
  Batch table (Parquet)  …/demo/data/processed/batch_table.parquet
  Report                 …/demo/data/processed/build_report.html
  Manifest               …/demo/data/processed/batch_table_manifest.json
```

The warning is the catalog check from a moment ago. `build_report.html` shows
what came out: every parameter's source, each cleaning rule that changed
something, and the derived features.

### 3. Create a task

```bash
uv run ml task new tasks/car_expression --target "FP Flow CAR+ (%)"
cd tasks/car_expression
```

```
Task workspace created
  File    …/demo/tasks/car_expression/task.yaml
  Config  …/demo/config.yaml
  Next    cd …/demo/tasks/car_expression && ml validate && ml task run
```

The folder holds an annotated `task.yaml`. It already points at the
`config.yaml` you came from (`config: ../../config.yaml`), so the task finds the
batch table from wherever you run it. Its main sections:

```yaml
cohort:      which batches are rows        (empty: every batch)
target:      what the model will predict   (here FP Flow CAR+ (%), numeric)
roles:       columns that describe a batch rather than predict anything
predictors:  which other parameters may be predictors
clustering:  what to do about parameters that move together
```

### 4. Run it as it is

Inside a task folder no `--task` is needed:

```bash
uv run ml validate
uv run ml task run
```

```
Written — 68 batches × 150 columns
  Task folder  …/tasks/car_expression/runs/20261012-170555-bfe8d0

  WARN  dataset  2 batches have no usable target and are not in the dataset
  WARN  dataset  Only 4 distinct Vector Lot values: any grouped cross-validation rests on 4 groups, not on 68 batches
  WARN  dataset  197 predictor columns for 68 fitted batches (p/n = 2.9): a model needs strong regularisation, …
```

It works, and the run folder already holds a full dataset:

```
runs/20261012-170555-bfe8d0/
  data/dataset.csv               one row per batch: id, group, raw predictors, target
  data/dataset_encoded.csv       the same rows as numbers, ready for a model
  metadata/parameters.csv        every parameter: included or excluded, and why
  metadata/batches.csv           every batch: in or out, and why
  metadata/dataset_columns.csv   every column of dataset.csv: role, source, unit, meaning
  metadata/groups.json           groups of parameters to explain a model by (step 7)
  report/report.html             all of the above with figures; open it in a browser
  inputs/                        the exact batch table and catalog this run read
  config/task.yaml               every setting, defaults included
```

But look at what it contains:

```python
import glob
import pandas as pd

run = sorted(glob.glob("runs/*/"))[-1]                       # the latest run
data = pd.read_csv(f"{run}data/dataset.csv")
print([c for c in data.columns if "Post" in c or "Endotoxin" in c][:4])
```

```
['Day 1 Number of Mixing Cycles Post Activation', 'Endotoxin (EU/mL)',
 'Harvest Post Wash Viability Average (%)', 'Harvest Post Wash Viable Cell Concentration Average (cells/mL)']
```

Endotoxin is a release test and the post-wash counts come after harvest: a
model trained on them would "predict" CAR+ expression from things measured
after it was decided. Both sites are mixed in, as are clinical batches. The
default dataset answers no particular question, so the next step says what the
question is.

### 5. Narrow the cohort and choose the predictors

Edit `task.yaml`. Keep only commercial Ghent batches:

```yaml
cohort:
  sites: [Ghent]
  filters:
    - {column: Type, op: equals, value: Commercial, reason: manufactured commercially}
```

and choose the predictors:

```yaml
predictors:
  exclude:                        # single columns, each with its reason
    - {column: Subject Weight (kg), reason: dose per kilogram already carries it}
  exclude_by_catalog:             # every parameter with these catalog values
    Value Type: [datetime]
    Available At: [Wash, Formulation, Final product, Post-thaw, Release, Post-release]
```

`exclude_by_catalog` looks every parameter up in the catalog and leaves it out
when the named catalog column holds one of the listed values. Here it drops the
timestamps (`Value Type: datetime`) and everything the catalog marks as
measured after harvest (`Available At`). Any catalog column works — a
`Category` column would let you write `{Category: [QC]}` — and
`include_by_catalog` does the opposite, keeping only the listed values.

```bash
uv run ml validate
uv run ml task run
```

```
Written — 34 batches × 115 columns
```

`metadata/parameters.csv` now says why each late parameter is gone:

| Parameter | Decision | Reason | Stage |
|---|---|---|---|
| Post Thaw FLOW CD3+ (%) | excluded | Catalog 'Available At' is 'Post-thaw' | category exclusion |
| Subject Weight (kg) | excluded | dose per kilogram already carries it | task exclusion |
| VCN/cell | excluded | computed from the target: FP Flow CAR+ (%) → VCN/cell | target lineage |
| D3 % cells seeded / target | excluded | duplicates 'Avg_cells_per_bag_D3' (R²=1.0 over 34 shared batches) | near-duplicate |

Some exclusions you never asked for: `VCN/cell` is computed from the target, so
it would be the answer key, and a near-duplicate adds nothing its twin does not
already say. In all, 37 parameters were left out by catalog category, and 138
encoded predictor columns remain for 34 batches.

A typo is caught before anything runs. With `column: FP Flow CAR (%)`:

```
Problems that would stop a run (1)
  - car_expression: target.column: 'FP Flow CAR (%)' is not a column of the batch table
```

### 6. Tame correlated parameters

Manufacturing parameters measure the same few things several times over: four
Day 3 cell counts move together. A model given all four cannot say which
mattered, so its explanations split credit arbitrarily and change from fold to
fold. Every run looks for such groups and writes them to
`metadata/clusters.csv`. In the last run:

| Cluster | Members | Representative | Stability |
|---|---|---|---|
| Cluster 01 | 12 | Lactate_to_Glucose_perE6_ratio_D6 | 0.92 |
| Cluster 02 | 9 | Avg_cells_per_bag_D3 | 0.70 |
| Cluster 03 | 5 | MOI_eff_pool | 0.38 |
| Cluster 05 | 3 | D10 Lact / E6 cells | 0.98 |

*Stability* is how often the group comes back when the clustering is redone on
50 random subsets of 80% of the batches. With 34 batches some groups hang on a
handful of them: Cluster 03 is a coincidence of this cohort, Cluster 05 is
real. Of 24 groups, 17 reach the default bar of 0.8; only those are acted on.

Replace each stable group by one score. In `task.yaml`:

```yaml
clustering:
  action: summarise
```

```bash
uv run ml task run
```

```
Written — 34 batches × 115 columns
```

`dataset.csv` is unchanged — it keeps every raw parameter — but
`dataset_encoded.csv` now has one column per stable group instead of its
members: 46 parameters became 17 scores such as `D10 Lact / E6 cells group`,
each the mean of its members after z-scoring, with negatively related members
flipped. `parameters.csv` records each one:

```
Day 6 Glucose G-Rex A (g/L)  changed  summarised into 'Lactate_to_Glucose_perE6_ratio_D6 group' with 11 other members of Cluster 01
```

If your catalog has a column such as `Process step`, `clustering.name_by:
Process step` names a group after the step its members share.

### 7. Fit a model and explain it by group

`metadata/groups.json` puts every encoded column into one explanation group:
a stable cluster, or a single parameter with all its encoded columns (every
one-hot column of `Shift Team`, say). Explaining by group gives each group its
full credit instead of splitting it between correlated members.

Save this as `fit_and_explain.py` in the task folder and run it with
`uv run python fit_and_explain.py`:

```python
import glob
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.recipe import TaskTransformer, task_data

run = sorted(glob.glob("runs/*/"))[-1]                   # the latest run of this task
groups = json.load(open(f"{run}metadata/groups.json"))   # explanation groups
X, y, folds_by, spec = task_data(".")                    # cohort rows, target, vector lots

rng = np.random.default_rng(0)
scores, drops = [], {name: [] for name in groups}
for train, test in GroupKFold(n_splits=3).split(X, y, folds_by):
    model = make_pipeline(
        TaskTransformer(spec, run=run),   # the task's encoding, refitted on this fold only
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 4, 13)),
    ).fit(X.iloc[train], y.iloc[train])
    held_out, truth = X.iloc[test], y.iloc[test]
    scores.append(model.score(held_out, truth))           # R² on unseen vector lots
    for name, group in groups.items():                    # shuffle a whole group at once
        shuffled = held_out.copy()
        shuffled[group["parameters"]] = held_out[group["parameters"]].to_numpy()[
            rng.permutation(len(held_out))
        ]
        drops[name].append(scores[-1] - model.score(shuffled, truth))

print(f"R² per fold: {np.round(scores, 2)}")
importance = pd.Series({name: np.mean(d) for name, d in drops.items()})
print(importance.sort_values(ascending=False).head(5).round(3).to_string())
```

```
R² per fold: [ 0.46  0.38 -0.15]
Harvest Pre-Wash Flow CAR+ Expression (%)    0.268
Post formulation dose                        0.113
Cluster 11                                   0.077
CAR T produced per cell engaged D3           0.059
Day 8 IL-2 Protein Content (µg/vial)         0.041
```

Three things keep this honest:

- **`task_data(".")`** returns the cohort exactly as `ml task run` selects it,
  the target, and the vector lot of every batch, so whole lots are held out.
- **`TaskTransformer(spec, run=run)`** refits the task's encoding inside every
  fold — nothing is learned from the held-out batches — and uses the run's
  groups in every fold, so every fold explains the same groups.
- **Shuffling a whole group** measures what the group is worth to the model on
  batches it has not seen. With SHAP, sum each row's values over a group's
  `columns` in `groups.json` instead.

(Made-up data: the scores themselves mean nothing.)

### 8. Find a leak through the explanation, and fix it

Read the top of that list as a process engineer would:

- **Harvest Pre-Wash Flow CAR+ Expression (%)** is CAR+ expression measured at
  harvest: the target, measured one step earlier.
- **Post formulation dose** is a derived feature computed after formulation.
  Derived features have no catalog value of their own, so
  `exclude_by_catalog` could not catch it.

Neither crossed the automatic proxy check (|r| ≥ 0.95 with the target), so
only the explanation showed them. Exclude both:

```yaml
predictors:
  exclude:
    - {column: Subject Weight (kg), reason: dose per kilogram already carries it}
    - {column: Harvest Pre-Wash Flow CAR+ Expression (%), reason: the target measured at harvest}
    - {column: Post formulation dose, reason: computed after formulation}
  exclude_by_catalog:
    Value Type: [datetime]
    Available At: [Wash, Formulation, Final product, Post-thaw, Release, Post-release]
```

```bash
uv run ml task run
uv run python fit_and_explain.py
```

```
Written — 34 batches × 113 columns
R² per fold: [ 0.14 -0.09 -0.58]
Cluster 11                                    0.247
Day 8 IL-2 Protein Content (µg/vial)          0.081
CAR T produced per cell engaged D3            0.068
Total CS5 cell concentration (E6 cells/mL)    0.062
Shift Team                                    0.050
```

The score fell: the earlier one was the leak talking. Now `Cluster 11` leads —
`groups.json` shows it is `D10 Achievable Dose` with `D10 Total CAR+ viable
T-cells achieved`, CAR+ counts on day 10. Whether those are known when you want
to predict is your decision; this loop of *run, explain, question, exclude* is
how a task gets to a dataset you can trust.

### 9. Reproduce a run, and encode new batches

A run folder holds everything it needs (`ls runs/` lists yours; the newest is last):

```bash
uv run ml task rerun runs/20261012-170843-61f466
```

```
OK  cli  dataset.csv is identical to the one in 20261012-170843-61f466
```

The rerun reads only that folder — its `config/task.yaml` points at the copies
in `inputs/` — and writes a new run next to it. Archive a run folder and you can
redo it years later, however the batch table has changed since.

To encode batches exactly as a run encoded its cohort, with nothing refitted:

```bash
uv run ml task encode --run runs/20261012-170843-61f466 \
    --batch-table ../../data/processed/batch_table.xlsx --out new_batches_encoded.csv
```

```
OK  cli  70 batches encoded with 20261012-170843-61f466's encoding → new_batches_encoded.csv
```

### 10. A binary target

The finished example `demo/tasks/raritan_release` predicts whether a Raritan
batch was released:

```yaml
target:
  column: Disposition
  type: binary
  positive_class: Released        # becomes 1
  negative_class: Discarded       # becomes 0; any third value stops the run
```

```bash
uv run ml task run --task ../raritan_release
```

```
Written — 27 batches × 142 columns
```

The target column in `dataset.csv` is `disposition_is_released` (15 released,
12 discarded). That task also uses `clustering.action: summarise` with
`name_by: Available At`.

**You now know the whole loop.** To ask a second question, run `ml task new`
again: every question gets its own folder and its own history of runs.

## Use your own data

You do this once. Afterwards you only rerun `ml check-catalog` and `ml build`
when the sources or the catalog change.

**1. Make a data folder** anywhere outside the repository, for example:

```
~/ml_data/
  catalog.xlsx             the parameter catalog
  raw/                     local copies of the sources
    phf_ghent.xlsx
    phf_raritan.xlsx
    parameter_requirements.xlsx
    remfg.xlsx
    lv_coa.xlsx
    raw_materials.csv
    investigations_power_query.xlsx
    site_map_clinical_sites.xlsx
```

`ml build` writes the batch table into this folder, with the reports and
manifests next to it.

**2. Point `config.yaml` at it.** `config.yaml` in the repository root is
annotated. These are the paths to set:

```yaml
paths:
  catalog: ~/ml_data/catalog.xlsx           # read by ml check-catalog, ml build and every task
  batch_table: ~/ml_data/batch_table.xlsx   # written by ml build
  phf_ghent: ~/ml_data/raw/phf_ghent.xlsx
  phf_raritan: ~/ml_data/raw/phf_raritan.xlsx
  param_mapping: ~/ml_data/raw/parameter_requirements.xlsx
  remfg: ~/ml_data/raw/remfg.xlsx
  lv_coa: ~/ml_data/raw/lv_coa.xlsx
  raw_materials: ~/ml_data/raw/raw_materials.csv
  investigations: ~/ml_data/raw/investigations_power_query.xlsx
  site_maps:
    - ~/ml_data/raw/site_map_clinical_sites.xlsx
  new_parameters: ~/ml_data/new_parameters.csv
```

Relative paths are relative to `config.yaml`, and `~` is your home folder. To
read the sources from SharePoint instead, see
[Sources and SharePoint](#sources-and-sharepoint).

**3. Put secrets in `.env`.** If you use SharePoint, copy `.env.example` to
`.env` next to `config.yaml` and fill in the credentials. `.env` is git-ignored.

**4. Run steps 2–8 of the tutorial** with your own data: `ml validate`,
`ml check-catalog`, `ml build`, then `ml task new` for each question. A task
folder can live anywhere — next to the analysis it belongs to, say — because its
`task.yaml` records where `config.yaml` is.

## Reference

### Commands

Global options come before the command: `uv run ml [--config FILE] [--debug] COMMAND …`.

| Command | What it does |
|---|---|
| `ml validate [--task T]` | Validate `config.yaml`, the corrections table and the tasks against the batch table and catalog. Runs nothing. Inside a task workspace, also validates that task. |
| `ml check-catalog` | Compare every source's columns with the catalog. Writes `catalog_report.html`, `catalog_manifest.json` and `new_parameters.csv` beside the batch table. |
| `ml build [--mode PRD\|DEV]` | Build the batch table. Writes `batch_table.xlsx`, `batch_table.parquet`, `build_report.html` and `batch_table_manifest.json`. `--mode PRD` also uploads when `upload.enabled` is set. |
| `ml task new FOLDER [--target COL] [--question TEXT] [--force]` | Create a task workspace with an annotated `task.yaml` linked to the current `config.yaml`. |
| `ml task run [--task T]` | Run tasks. Without `--task`, runs the task in the current folder. |
| `ml task rerun RUN_FOLDER` | Redo a finished run from its own folder and report whether `dataset.csv` is identical. |
| `ml update [--mode M] [--task T]` | `check-catalog`, then `build`, then the given tasks. |
| `ml task encode --run RUN_FOLDER [--batch-table FILE] [--out FILE]` | Encode batches (default: the configured batch table) with a run's fitted encoding. |

`--task` accepts a task workspace folder, a `task.yaml`, or a folder that
contains several workspaces, and can be repeated. `--debug` shows Python
tracebacks instead of one-line error messages.

The config is found in this order: `--config`, the `config:` line of the task,
`$ML_CONFIG`, then the nearest `config.yaml` in the current folder or above it.

### task.yaml

Every key is optional except `target.column`. Unknown keys are
refused, and each error names its field. Quote any column name that contains
`#`, `:` or `%` when it is not inside `{…}` or `[…]`; unquoted,
`Patient Lot/Batch #` is read as `Patient Lot/Batch` followed by a comment.

| Key | Default | Meaning |
|---|---|---|
| `config` | — | Path to the `config.yaml` that supplies the batch table and catalog. `ml task new` fills it in. |
| `inputs.batch_table`, `inputs.catalog` | from config | Read this batch table / catalog instead. |
| `task.name` | `task` | The task's name, shown in the report and manifest. |
| `task.question` | — | The question, in words; heads the report. |
| `task.seed` | `0` | Seed for the target-encoding folds, so a run is repeatable. |
| `cohort.sites` | all | Sites to keep, matched against `cohort.site_column`. |
| `cohort.site_column` | `Site Merged` | The column that says which site a batch is from. |
| `cohort.filters` | — | Rules applied in order: `{column, op, value, missing, normalise, reason}`. |
| `cohort.date_column`, `date_from`, `date_to` | — | Keep batches whose date lies in this range (inclusive). |
| `target.column` | required | The column to predict. |
| `target.type` | `numeric` | `numeric` or `binary`. |
| `target.positive_class` | — | Required for `binary`: the value that becomes 1. |
| `target.negative_class` | the other value | The value that becomes 0. Any third value stops the run. |
| `roles.id` | — | Identifies a batch. Must be unique in the cohort. |
| `roles.group` | — | Grouping column (e.g. vector lot); used for grouped folds. |
| `roles.patient` | — | Joins a patient's batches into one group. |
| `predictors.include` | all | Use exactly these predictors and nothing else. |
| `predictors.exclude` | — | `[{column, reason}]`: single columns to leave out. |
| `predictors.exclude_by_catalog` | — | `{catalog column: [values]}`: leave out parameters with these catalog values. |
| `predictors.include_by_catalog` | — | `{catalog column: [values]}`: keep only parameters with these catalog values. |
| `clustering.action` | `describe` | `off`, `describe`, `keep_one`, `residualise` or `summarise`; see [below](#correlated-parameters). |
| `clustering.min_stability` | `0.8` | Act on a group only if it comes back in this share of subsamples. |
| `clustering.name_by` | — | A catalog column; a group whose members share its value is named after it. |
| `clustering.method` | `spearman` | `spearman` or `pearson`. |
| `clustering.threshold` | `0.7` | Correlation at which parameters are grouped. |
| `clustering.min_size`, `min_overlap` | `2`, `10` | Smallest group; fewest shared batches before a correlation counts. |
| `clustering.prefer` | — | Parameters to pick as a group's representative on a tie. |
| `encoding.*` | see `taskconfig.ENCODING_FIELDS` | Encoder limits, e.g. `missing_indicators: true`, `impute: median`, `target_min_rows_per_category: 5`. |
| `quality.min_non_missing` | `20` | A predictor needs at least this many values. |
| `quality.duplicate_r2_threshold` | `0.99` | Above this R², one of two parameters is a near-duplicate and is dropped. |
| `quality.near_constant_tolerance` | `0.0001` | Columns constant to this precision are dropped. |
| `quality.proxy_correlation` | `0.95` | Predictors this correlated with the target are flagged as possible leaks. |
| `missingness.enabled` | `true` | Describe which parameters go missing together (report only). |
| `report.preview_rows` | `20` | Rows shown in the report's dataset preview. |

Filter operators: `equals`, `not_equals`, `in`, `not_in`, `min`, `max` (both
inclusive, numbers or dates), `is_null`, `not_null`. A filter on a column the
batch table doesn't have stops the run instead of quietly keeping every batch.

### config.yaml

`config.yaml` in the repository root is annotated; this is its shape:

```yaml
seed: 20260917
paths:      {catalog: …, batch_table: …, phf_ghent: …, …, reports: (default: beside the batch table)}
sheets:     {phf_ghent: BR Data}           # worksheet per workbook, where not the default
sources:    {location: local, fallback_to_local: false, databricks: false}
sharepoint: {catalog: {path: …, sheet: …, drive_id: …}, …, site_maps: {folder: …, files: […], …}}
upload:     {enabled: false, path: …, drive_id: …, file_name: PVF.xlsx}
catalog:    {write_new_parameters: true}
build:      {write_parquet: true}
cleaning:   {corrections: …, disable: [], censored: limit, censored_flags: false,
             duration_unit: minutes, yes_no_defaults: {…}}
mapping:    {exclude: [Non-Conformance Type Calc.]}
features:   {co2_target: 5.0, temp_target: 37.0, …}
lv_coa:     {ph_nominal: 7.2, osmo_nominal: 300.0}
```

Unknown keys, wrong types and incomplete SharePoint entries are refused before
anything runs.

#### Sources and SharePoint

With `sources.location: local`, every input is read from its path under
`paths:`. With `sources.location: sharepoint`, every input that has an entry
under `sharepoint:` is read from SharePoint instead, and the rest stay local.
Without `io_sharepoint` installed the run stops, unless
`sources.fallback_to_local: true`. All SharePoint addresses live in
`config.yaml`; none are in the code.

Uploading the batch table is a separate switch. It only happens on
`ml build --mode PRD` with `upload.enabled: true`, to `upload.path` (or
`$DATA_LINK`). `MODE`, `DATA_LINK` and credentials can all live in `.env`;
variables already set in the shell win.

#### Corrections

Known errors in the sources are rules in a `corrections.yaml`
(`cleaning.corrections`; without it the package's own table is used):

```yaml
corrections:
  - name: israel_country
    site: Raritan
    column: Country
    when: {column: Clinical Site, equals: "107306"}
    set: Israel
    reason: the site is in Israel; the export has the wrong country
  - {site: Raritan, column: Total Viable Cells/bag, divide_by: 1000000, reason: stored without the 1e6 divisor}
```

The actions are `set`, `replace_text`, `replace_values` and `divide_by`.
`cleaning.disable: [name]` switches a rule off. The build report lists every
rule that ran and the rows it changed. `cleaning.censored` decides what `<0.5`
becomes (`limit`, `half` or `missing`), and `cleaning.yes_no_defaults` names the
yes/no columns whose blanks may be filled.

### How a dataset is built

For each task, `ml task run`:

1. **Selects the cohort**: sites, then each filter in order, then batches without
   a target. Every batch in the batch table gets a row in `batches.csv` saying
   whether it is in, and why not.
2. **Chooses the predictors**: the catalog parameters present in the batch
   table, minus
   - the target and the `roles` columns;
   - features the registry computed from the target (`VCN/cell` from `FP Flow CAR+ (%)`);
   - `predictors.exclude` and the catalog rules;
   - `LVCoA_Impurity_Index_zsum`, which is standardised within each site.

   Each exclusion is a row in `parameters.csv`.
3. **Fits the preprocessing** on the whole cohort, records it in `encoding.json`,
   and writes the encoded table as `dataset_encoded.csv`:
   - Two categories become 0/1. A few become one column each. More become the
     target's mean per category (cross-fitted, so no row sees its own target).
     Very many are hashed into buckets.
   - Each encoder is used only when there are enough rows; otherwise the next
     one down is used. A column with a different value in nearly every batch is
     treated as an identifier and left out.
   - Near-duplicates, near-constant and too-sparse columns are dropped.
   - Correlated parameters are grouped, and stable groups get the
     `clustering.action`. Every encoded column is assigned to an explanation
     group in `groups.json`.
4. **Writes the dataset**: `dataset.csv` holds the id, group, the raw predictors
   the encoding uses, and the target. It has one row per batch and no split.
5. **Checks** that the row and column counts agree across every file, then
   renames the run folder from `.incomplete` to its final name. A failed run
   keeps its folder as `.failed`, with the log and the reason in its manifest.

#### Correlated parameters

| `clustering.action` | Effect |
|---|---|
| `off` | No groups are looked for. |
| `describe` | Groups are described in the report; every parameter stays. |
| `keep_one` | One member of each group stays; the others are dropped. |
| `residualise` | One member stays; the others are replaced by what is left after regressing them on it. |
| `summarise` | The group becomes one score, `<name> group`: the mean of its members after z-scoring, with negatively related members flipped. Every member feeds it. |

- **How groups are found.** Complete linkage on `1 − |correlation|`, cut at
  `threshold`. Pairs with fewer than `min_overlap` shared batches are unknown and
  never grouped. The representative is the most central member, chosen without
  looking at the target.
- **Only stable groups are acted on.** Each group is redrawn on 50 subsamples of
  80% of the batches. A group that comes back in fewer than `min_stability`
  of them is described and left alone, and isn't an explanation group. With 20–30
  batches many groups hang on a handful of them. `clusters.csv` and the
  report show each group's stability.
- **Within site.** When the cohort spans several sites, each site's mean is
  removed before correlating, so two parameters that only differ by site are not
  grouped.
- **Which action, for explanations.** `describe` keeps everything and leaves the
  grouping to [group-level explanations](#7-fit-a-model-and-explain-it-by-group).
  `summarise` gives one named, stable column per group, which is the easiest to
  explain to a process owner; name groups with `name_by` on a catalog column
  such as a process step. `keep_one` loses the other members' information, and
  residuals are hard to read unless their order means something.

### What a run writes

| File | Contents |
|---|---|
| `data/dataset.csv` | The training dataset: id, group, raw predictors, target. One row per batch. |
| `data/dataset_encoded.csv` | The same rows encoded by `encoding.json`. |
| `metadata/parameters.csv` | Every parameter: included or excluded, with the reason and the step. |
| `metadata/batches.csv` | Every batch in the batch table: in or out, and the first rule that ruled it out. |
| `metadata/dataset_columns.csv` | Every dataset column: role, source, type, catalog value type, unit, meaning. |
| `metadata/encoded_columns.csv` | Every encoded column and the parameter it came from. |
| `metadata/clusters.csv` | Correlated groups: stability, representative and what was done. |
| `metadata/groups.json` | Explanation groups: every encoded column, grouped with what it measures. |
| `metadata/missingness.csv` | Parameters that go missing together (when there are any). |
| `metadata/encoding.json` | The fitted preprocessing, used by `ml task encode` and `TaskTransformer`. |
| `report/report.html` | The report: cohort, data quality, encodings, clusters, dataset, provenance. |
| `report/report_payload.json` | The report's numbers as JSON. |
| `inputs/batch_table.xlsx`, `inputs/catalog.xlsx` | The exact inputs this run read. |
| `inputs/batch_table_manifest.json`, `inputs/catalog_manifest.json` | How those inputs were made. |
| `config/task.yaml` | Every setting, defaults included, pointing at `inputs/`. |
| `logs/run.log` | This run's log. |
| `manifest.json` | Inputs, outputs with SHA-256 checksums, counts, warnings, git commit. |

The reports are self-contained Streamlit pages: they run Python in the browser
via stlite, so the first time you open one it needs network access and takes
10–30 seconds to start. `ml.manifest.verify("…/manifest.json")` reports any
file that changed after the run.

### Troubleshooting

| Message | What to do |
|---|---|
| `No config found: there is no config.yaml here or in any folder above` | Run from the repository, add `config:` to the task, set `$ML_CONFIG`, or pass `--config`. |
| `No task to run: run inside a task workspace or pass --task <folder>` | `cd` into a workspace, or name one with `--task`. |
| `… is not a column of the batch table` | Check the spelling against the batch table; `ml validate` lists every such name. |
| `predictors.exclude_by_catalog: the catalog has no '…' column` | Use a column that exists in the catalog; the message lists them. |
| `(root).columns: unknown setting` (or any `unknown setting`) | The key was renamed, removed or misspelled; the message lists the allowed keys. `columns:` is now `roles:` and `predictors:`. |
| `The catalog changed after the batch table was built — rerun ml build` | Run `ml build`, then the task again. |
| `quality.min_non_missing is 20 but only N rows are being fitted on` | Lower `quality.min_non_missing` or widen the cohort. |

Errors you can fix exit with code 2 and a one-line message; `--debug` shows the
traceback.

## Development

```bash
uv sync --locked              # the environment the tests ran in
uv run pytest                 # every stage, end to end, on the demo data
uv run ruff check src tests   # lint
```

CI runs both on every push (`.github/workflows/ci.yml`). The demo's data and
runs are generated, not committed: only `demo/config.yaml`,
`demo/corrections.yaml` and the two `task.yaml` files are in git.

**Adding a derived feature:** append one `Feature(…)` entry to the registry in
`src/ml/features.py`. It is built only when the catalog lists its name and all
its inputs are present, and it never overwrites a source column. Its `requires` tuple
is what lets a task find, and leave out, features computed from the target.

```
src/ml/
  cli.py          the commands
  config.py       config.yaml: discovery, validation, where each input is read from
  taskconfig.py   task.yaml and its validation
  scaffold.py     ml task new; templates/ holds the task.yaml it writes
  catalog.py io.py clean.py corrections.py enrich.py features.py merge.py   ml check-catalog and ml build
  dataset.py encode.py decorrelate.py missingness.py recipe.py         ml task run
  package.py manifest.py provenance.py logger.py                       run folders and provenance
  report.py plots.py blocks.py streamlit_app.py stlite.py              the reports
  demo.py         the made-up demo data
```

## License

This repository has no license file yet, so no license has been granted. Add one
before sharing it outside the team.
