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
| **catalog** | `catalog.xlsx` on SharePoint, in `Data/catalog/` (formerly the PTF): every parameter the batch table may hold, with metadata columns such as value type and process stage |
| **batch table** | `batch_data.xlsx` on SharePoint, in `Data/batch_data/` (formerly the PVF): one row per batch, both sites, cleaned, with derived features |
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
below it, such as a task folder. All data is read from and written to
SharePoint, through the internal `io_sharepoint` package; there is no local data
option. Joining Acridine Orange readings needs `databricks_query_tool`.

## Tutorial: from source files to an explained model

This tutorial takes about 20 minutes. It uses the demo: made-up data with the
same shape as the real sources, so every command below runs as written and
prints what is shown. Run IDs (such as `20261014-101858-32352c`) and timings
will differ on your machine.

You will build the batch table, ask one question of it — *which manufacturing
parameters go with CAR+ expression at Northfield?* — refine the dataset four times,
fit a model, find a leak through the model's explanation, and fix it.

### 1. Make the demo data

From the repository root:

```bash
uv run python -m ml.demo demo
cd demo
```

```
config.yaml   where on SharePoint the vocabulary, the sources, the catalog and the
              batch table are
.env          points ML_SHAREPOINT_STANDIN at sharepoint/
sharepoint/   a folder standing in for SharePoint: DEMO_DRIVE/Data/raw/ holds the
              made-up sources and their vocabulary.xlsx, DEMO_DRIVE/Data/catalog/
              the catalog
tasks/        two finished example tasks: expression and release
```

The demo has no SharePoint, so `sharepoint/` plays its part; every run says so
with a warning. The demo's two sites are called Northfield (site A) and
Southbank (site B), and every column and value in it is invented; its
`vocabulary.xlsx` maps the code's keys to those names, as the private one does
for the real sources (see [The vocabulary](#the-vocabulary)). Every command from
now on is run inside `demo/`, where `config.yaml` is.

### 2. Check the catalog and build the batch table

First see whether the configuration is sound. Only the vocabulary is read, and
nothing is written:

```bash
uv run ml validate
```

```
OK  config  Vocabulary loaded — 233 names
OK  cli     Corrections table: 14 rules
OK  cli     Everything checked out
```

Then compare every source with the catalog:

```bash
uv run ml check-catalog
```

Every command first lists what it will read and write:

```
Reading from SharePoint
  catalog         catalog.xlsx
                  DEMO_DRIVE › Data/catalog/  ·  sheet Catalog
  phf_site_a      site_a_batch_records.xlsx
                  DEMO_DRIVE › Data/raw/  ·  sheet Batch Records
  …

Writing to SharePoint
  folder          DEMO_DRIVE › Data/catalog/
  report          catalog_report.html
  manifest        catalog_manifest.json
  new parameters  new_parameters.csv
```

and ends with what it found and where each file went:

```
Northfield batch records: columns the catalog does not list (1)
  - Site note
Southbank batch records: columns the catalog does not list (2)
  - Plant code
  - Remark
Investigations Power Query: columns the catalog does not list (3)
  …
  - Days open

Written to SharePoint
  new parameters  new_parameters.csv
                  DEMO_DRIVE › Data/catalog/
  report          catalog_report.html
                  DEMO_DRIVE › Data/catalog/
  manifest        catalog_manifest.json
                  DEMO_DRIVE › Data/catalog/
```

These are columns the sources record that the batch table will not carry. On
real data this is your to-do list: add the ones you want to `catalog.xlsx`, then
build. The list is also in `new_parameters.csv` in the catalog's folder, and
`catalog_report.html` shows it with every source side by side.

```bash
uv run ml build
```

```
WARN  cli  The last `ml check-catalog` found 6 parameters the sources record and the catalog does not list; …
Written to SharePoint — 70 batches × 221 columns
  batch table            batch_data.xlsx
                         DEMO_DRIVE › Data/batch_data/
  batch table (Parquet)  batch_data.parquet
                         DEMO_DRIVE › Data/batch_data/
  report                 build_report.html
                         DEMO_DRIVE › Data/batch_data/
  manifest               batch_table_manifest.json
                         DEMO_DRIVE › Data/batch_data/
```

The warning is the catalog check from a moment ago. `build_report.html` shows
what came out: every parameter's source, each cleaning rule that changed
something, and the derived features.

### 3. Create a task

```bash
uv run ml task new tasks/car_expression --target "Product CAR expression (%)"
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
target:      what the model will predict   (here Product CAR expression (%), numeric)
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
  Task folder  …/tasks/car_expression/runs/20261014-101452-ed627c

  WARN  dataset  2 batches have no usable target and are not in the dataset
  WARN  encode   Mixing severity: 'Severe' is not one of its catalog categories (8 batches) — left missing  (×3)
  WARN  dataset  201 predictor columns for 68 fitted batches (p/n = 3.0): a model needs strong regularisation, …
```

It works, and the run folder already holds a full dataset:

```
runs/20261014-101452-ed627c/
  data/dataset.csv               one row per batch: id, group, raw predictors, target
  data/dataset_encoded.csv       the same rows as numbers, ready for a model
  metadata/parameters.csv        every parameter: included or excluded, and why
  metadata/batches.csv           every batch: in or out, and why
  metadata/dataset_columns.csv   every column of dataset.csv: role, source, unit, meaning
  metadata/groups.json           groups of parameters to explain a model by (step 7)
  report/report.html             all of the above with figures; open it in a browser
  inputs/                        the exact batch table, catalog and vocabulary this run read
  config/task.yaml               every setting, defaults included
```

But look at what it contains:

```python
import glob
import pandas as pd

run = sorted(glob.glob("runs/*/"))[-1]                       # the latest run
data = pd.read_csv(f"{run}data/dataset.csv")
print([c for c in data.columns if "after wash" in c or "Endotoxin" in c][:4])
```

```
['Day 10 concentration after wash (cells/mL)', 'Day 10 viability after wash (%)',
 'Endotoxin level']
```

Endotoxin is a release test and the after-wash counts come after harvest: a
model trained on them would "predict" CAR+ expression from things measured
after it was decided. Both sites are mixed in, as are trial batches. The
default dataset answers no particular question, so the next step says what the
question is.

### 5. Narrow the cohort and choose the predictors

Edit `task.yaml`. Keep only routine Northfield batches:

```yaml
cohort:
  sites: [Northfield]
  filters:
    - {column: Batch kind, op: equals, value: Routine, reason: manufactured routinely}
```

say which column groups batches, so that whole vector lots can be held out
later (step 7):

```yaml
roles:
  id: 'Patient Lot/Batch #'
  group: Vector lot
```

and choose the predictors:

```yaml
predictors:
  exclude:                        # single columns, each with its reason
    - {column: Patient weight (kg), reason: dose per kilogram already carries it}
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
Written — 34 batches × 116 columns

  WARN  dataset  Only 3 distinct Vector lot values: any grouped cross-validation rests on 3 groups, not on 34 batches
```

`metadata/parameters.csv` now says why each late parameter is gone:

| Parameter | Decision | Reason | Stage |
|---|---|---|---|
| Thawed CD3 share (%) | excluded | Catalog 'Available At' is 'Post-thaw' | category exclusion |
| Patient weight (kg) | excluded | dose per kilogram already carries it | task exclusion |
| Vector copies per cell | excluded | computed from the target: Product CAR expression (%) → Vector copies per cell | target lineage |
| D3 seeding vs target (%) | excluded | duplicates 'D3 cells per bag' (R²=1.0 over 34 shared batches) | near-duplicate |

Some exclusions you never asked for: `Vector copies per cell` is computed from
the target, so it would be the answer key, and a near-duplicate adds nothing its
twin does not already say. In all, 36 parameters were left out by catalog
category, and 139 encoded predictor columns remain for 34 batches.

A typo is caught before anything runs. With `column: Product CAR expression`:

```
Problems that would stop a run (1)
  - car_expression: target.column: 'Product CAR expression' is not a column of the batch table
```

### 6. Tame correlated parameters

Manufacturing parameters measure the same few things several times over: the
lactate and glucose readings of days 6 to 10 move together. A model given all four cannot say which
mattered, so its explanations split credit arbitrarily and change from fold to
fold. Every run looks for such groups and writes them to
`metadata/clusters.csv`. In the last run:

| Cluster | Members | Representative | Stability |
|---|---|---|---|
| Cluster 01 | 12 | Lactate to glucose ratio D6 | 0.92 |
| Cluster 02 | 9 | D3 cells per bag | 0.70 |
| Cluster 03 | 5 | MOI pooled | 0.38 |
| Cluster 04 | 3 | D10 lactate per E6 cells | 0.98 |

*Stability* is how often the group comes back when the clustering is redone on
50 random subsets of 80% of the batches. With 34 batches some groups hang on a
handful of them: Cluster 03 is a coincidence of this cohort, Cluster 04 is
real. Of 25 groups, 18 reach the default bar of 0.8; only those are acted on.

Replace each stable group by one score. In `task.yaml`:

```yaml
clustering:
  action: summarise
```

```bash
uv run ml task run
```

```
Written — 34 batches × 116 columns
```

`dataset.csv` is unchanged — it keeps every raw parameter — but
`dataset_encoded.csv` now has one column per stable group instead of its
members: 48 parameters became 18 scores such as `D10 lactate per E6 cells group`,
each the mean of its members after z-scoring, with negatively related members
flipped. `parameters.csv` records each one:

```
Day 6 glucose vessel A (g/L)  changed  summarised into 'Lactate to glucose ratio D6 group' with 11 other members of Cluster 01
```

If your catalog has a column such as `Process step`, `clustering.name_by:
Process step` names a group after the step its members share.

### 7. Fit a model and explain it by group

`metadata/groups.json` puts every encoded column into one explanation group:
a stable cluster, or a single parameter with all its encoded columns (every
one-hot column of `Operator team`, say). Explaining by group gives each group its
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
R² per fold: [ 0.46  0.37 -0.08]
Cluster 20                   0.268
Dose from formulation        0.137
Cluster 12                   0.083
CAR yield per seeded cell    0.060
Cluster 13                   0.044
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

- **Cluster 20** is, `groups.json` shows, `Day 10 CAR expression (%)` with
  `Vector copies per transduced cell`. The first is CAR+ expression measured at
  harvest: the target, measured one step earlier.
- **Dose from formulation** is a derived feature computed after formulation.
  Derived features have no catalog value of their own, so
  `exclude_by_catalog` could not catch it.

Neither crossed the automatic proxy check (|r| ≥ 0.95 with the target), so
only the explanation showed them. Exclude the harvest reading and the dose:

```yaml
predictors:
  exclude:
    - {column: Patient weight (kg), reason: dose per kilogram already carries it}
    - {column: Day 10 CAR expression (%), reason: the target measured at harvest}
    - {column: Dose from formulation, reason: computed after formulation}
  exclude_by_catalog:
    Value Type: [datetime]
    Available At: [Wash, Formulation, Final product, Post-thaw, Release, Post-release]
```

```bash
uv run ml task run
uv run python fit_and_explain.py
```

```
Written — 34 batches × 114 columns
R² per fold: [0.47 0.34 0.02]
Vector copies per transduced cell    0.197
D10 CAR cells                        0.140
D10 achievable dose                  0.040
Cluster 14                           0.035
Cluster 01                           0.035
```

The scores hardly moved, and the new top of the list says why: the leak is still
there, one step removed. `D10 CAR cells` and `D10 achievable dose` are derived
features computed from `Day 10 CAR expression (%)` (their `requires` in
`src/ml/features.py` say so): leaving a parameter out does not leave out what is
computed from it. `Vector copies per transduced cell`, now on its own, is a
vector measurement the catalog marks as available on day 0. Whether those are
known when you want to predict is your decision; this loop of *run, explain,
question, exclude* is how a task gets to a dataset you can trust.

### 9. Reproduce a run, and encode new batches

A run folder holds everything it needs (`ls runs/` lists yours; the newest is last):

```bash
uv run ml task rerun runs/20261014-101858-32352c
```

```
OK  cli  dataset.csv is identical to the one in 20261014-101858-32352c
```

The rerun reads only that folder — its `config/task.yaml` points at the copies
in `inputs/` — and writes a new run next to it. Archive a run folder and you can
redo it years later, however the batch table has changed since.

To encode batches exactly as a run encoded its cohort, with nothing refitted:

```bash
uv run ml task encode --run runs/20261014-101858-32352c --out new_batches_encoded.csv
```

```
OK  cli  70 batches encoded with 20261014-101858-32352c's encoding → new_batches_encoded.csv
```

### 10. A binary target

The finished example `demo/tasks/release` predicts whether a Southbank batch
was accepted:

```yaml
target:
  column: Outcome
  type: binary
  positive_class: Accepted        # becomes 1
  negative_class: Scrapped        # becomes 0; any third value stops the run
```

```bash
uv run ml task run --task ../release
```

```
Written — 27 batches × 142 columns
```

The target column in `dataset.csv` is `outcome_is_accepted` (15 accepted,
12 scrapped). That task also uses `clustering.action: summarise` with
`name_by: Available At`.

### 11. Compare two periods

Sometimes the question is not "what predicts this value" but "what changed
between these batches and those". Did something shift after a process change,
a new supplier or a new site setup? A `periods` target answers that. Name a
date column, give two date windows, and the run builds a dataset whose target
says which window each batch falls in: the second window is 1, the first is 0.
The model then finds what tells the two groups apart.

From `demo/`, make a task and point it at the date each batch started:

```bash
uv run ml task new tasks/before_after --target "Day 0 incubator in"
cd tasks/before_after
```

In `task.yaml`, keep one site, set the grouping column, and replace the target
with the two periods:

```yaml
cohort:
  sites: [Northfield]

target:
  column: "Day 0 incubator in"    # the date that puts a batch in a period
  type: periods
  periods:
    - {name: before, start: 2026-01-01, stop: 2026-01-10}   # becomes 0
    - {name: after,  start: 2026-02-10, stop: 2026-02-20}   # becomes 1

roles:
  id: 'Patient Lot/Batch #'
  group: Vector lot
```

```bash
uv run ml validate
uv run ml task run
```

```
OK     dataset     Cohort: 40 of 70 batches
INFO   dataset     Target: 'after' (2026-02-10 – 2026-02-20) → 1 (12 batches), 'before' (2026-01-01 – 2026-01-10) → 0 (10)
WARN   dataset     'Nation' correlates r=1.0 with the target over 22 fitted batches — check it is not the target measured another way, which would be a leak
WARN   dataset     'Outcome' correlates r=1.0 with the target over 22 fitted batches — check it is not the target measured another way, which would be a leak
…
Written — 22 batches × 129 columns
```

What happened:

- **The target column** in `dataset.csv` is `period_is_after`: 1 for the 12
  batches that started between 10 and 20 February, 0 for the 10 that started
  between 1 and 10 January.
- **Batches outside both periods are left out.** The 18 Northfield batches that
  started between the windows are in `batches.csv` with the reason
  `Day 0 incubator in is outside both periods`. A batch with no readable date
  is left out the same way (`… is not a date`). Both appear as steps of the
  cohort funnel in the report.
- **Both ends of a period count**, and the stop date counts for the whole day:
  a batch that started on 20 February at 18:30 is in `after`.
- **The date column is not a predictor**, and neither is anything computed
  from it. Raw timestamps never are; durations between them can be.
- **Read the leak warnings differently here.** In a numeric or binary task, a
  predictor that matches the target perfectly is usually a leak. When comparing
  periods, it may simply be what changed between them, such as a column that
  changed value from one month to the next. That can be the answer to your
  question. If it isn't interesting (for example, something that only changed
  because time passed), leave it out with `predictors.exclude` and run again.

The periods must not overlap, and each must contain at least one batch of the
cohort, or the run stops and says which one is empty. Use the cohort filters
as usual to narrow which batches are compared: for example, one site, or only
routine batches. Then fit and explain a model on `dataset.csv` exactly as in
step 7. A period target is a binary target, so use a classifier.

**You now know the whole loop.** To ask a second question, run `ml task new`
again: every question gets its own folder and its own history of runs.

## Use your own data

You do this once. Afterwards you only rerun `ml check-catalog` and `ml build`
when the sources or the catalog change.

**1. Put the SharePoint locations in `.env`.** Copy `.env.example` to `.env`
next to `config.yaml` and fill it in: the credentials `io_sharepoint` needs, the
drive each site's files are on, and where each input is. `.env` is git-ignored,
so no SharePoint address is ever committed:

```bash
SHAREPOINT_DRIVE_A="…"                     # the drive site A's files are on
SHAREPOINT_DRIVE_B="…"                     # the drive site B's files are on
SHAREPOINT_VOCABULARY="<team site>/Data/vocabulary.xlsx"
SHAREPOINT_CATALOG="<team site>/Data/catalog/catalog.xlsx"
SHAREPOINT_BATCH_TABLE="<team site>/Data/batch_data/batch_data.xlsx"
…                                          # one variable per input, as .env.example lists them
SHAREPOINT_SITE_MAPS_FOLDER="<team site>/Data/mappings/"
SHAREPOINT_SITE_MAPS_FILES="first.xlsx,second.xlsx"   # comma-separated
```

**2. Check `config.yaml`.** It is annotated, and refers to those variables:
every input has a path and a drive under `sharepoint:`.

```yaml
sharepoint:
  vocabulary:  {path: "${SHAREPOINT_VOCABULARY}", drive_id: "${SHAREPOINT_DRIVE_A}"}
  catalog:     {path: "${SHAREPOINT_CATALOG}", drive_id: "${SHAREPOINT_DRIVE_A}"}
  batch_table: {path: "${SHAREPOINT_BATCH_TABLE}", drive_id: "${SHAREPOINT_DRIVE_A}"}
  phf_site_a:  {path: …, drive_id: "${SHAREPOINT_DRIVE_A}"}   # site A's batch records
  phf_site_b:  {path: …, drive_id: "${SHAREPOINT_DRIVE_B}"}   # site B's batch records
  …
```

A variable that is not set stops the run and is named. The worksheet each input
is on comes from the vocabulary; a `sheet:` here overrides it. The catalog and
batch table locations also say where the stages write: `ml check-catalog` writes
into the catalog's folder and `ml build` into the batch table's.

**3. Fill in the vocabulary.** The code never names a real column, category
value, site or worksheet; it refers to each by a key, and the private vocabulary
workbook at `${SHAREPOINT_VOCABULARY}` says what each key stands for in your
sources, next to the known data corrections. See
[The vocabulary](#the-vocabulary) for its format; `ml validate` names every key
it leaves out.

**4. Install `io_sharepoint`** in the same environment (`uv pip install …`).

**5. Run steps 2–8 of the tutorial** with your own data: `ml validate`,
`ml check-catalog`, `ml build`, then `ml task new` for each question. A task
folder can live anywhere — next to the analysis it belongs to, say — because its
`task.yaml` records where `config.yaml` is.

**6. Run `ml check-leaks` before every push.** It reads the vocabulary and the
corrections from SharePoint and searches the repository's tracked files for any
real name in them, so none is ever committed; see [Commands](#commands).

## Reference

### Commands

Global options come before the command: `uv run ml [--config FILE] [--debug] [--quiet] COMMAND …`.

| Command | What it does |
|---|---|
| `ml validate [--task T]` | Validate `config.yaml`, the vocabulary (naming any key it leaves out), the corrections table and the tasks against the batch table and catalog. Runs nothing. Inside a task workspace, also validates that task. |
| `ml check-leaks [FOLDER]` | Search the git-tracked files of `FOLDER` (default: here) for any name the private vocabulary or its corrections hold. Exits 1 and lists every line when it finds one. Run it before pushing. |
| `ml check-catalog` | Compare every source's columns with the catalog. Writes `catalog_report.html`, `catalog_manifest.json` and `new_parameters.csv` into the catalog's SharePoint folder. |
| `ml build` | Build the batch table. Writes `batch_data.xlsx`, `batch_data.parquet`, `build_report.html` and `batch_table_manifest.json` into the batch table's SharePoint folder. |
| `ml task new FOLDER [--target COL] [--question TEXT] [--force]` | Create a task workspace with an annotated `task.yaml` linked to the current `config.yaml`. |
| `ml task run [--task T]` | Run tasks. Without `--task`, runs the task in the current folder. |
| `ml task rerun RUN_FOLDER` | Redo a finished run from its own folder and report whether `dataset.csv` is identical. |
| `ml update [--task T]` | `check-catalog`, then `build`, then the given tasks. |
| `ml task encode --run RUN_FOLDER [--out FILE]` | Encode the batch table's batches with a run's fitted encoding. |

`ml check-leaks` matches a site's name as a whole word, and any other name that
has a space, a digit or a symbol in it anywhere; a single plain word is too
common to tell from ordinary text and is skipped. Only a run with access to the
share knows the real names, so run it before pushing:

```bash
uv run ml check-leaks          # from the repository root
```

`--task` accepts a task workspace folder, a `task.yaml`, or a folder that
contains several workspaces, and can be repeated. `--debug` shows Python
tracebacks instead of one-line error messages.

Output redirected to a file (`ml build > build.log`) never has colour codes or
progress-bar frames in it; both are switched off when stdout is not a terminal.
`--quiet` (`-q`) also leaves out the step, info and OK lines and the section
rules, and keeps the warnings, errors, the lists of what was read and written,
and the summary: `uv run ml --quiet update > update.log`. The reports and each
task's `logs/run.log` still hold the full log.

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
| `task.name` | `task` | The task's name, shown in the report and manifest. |
| `task.question` | — | The question, in words; heads the report. |
| `task.seed` | `0` | Seed for the target-encoding folds, so a run is repeatable. |
| `cohort.sites` | all | Sites to keep, matched against `cohort.site_column`. |
| `cohort.site_column` | `Site Merged` | The column that says which site a batch is from. |
| `cohort.filters` | — | Rules applied in order: `{column, op, value, missing, normalise, reason}`. |
| `cohort.date_column`, `date_from`, `date_to` | — | Keep batches whose date lies in this range (inclusive). |
| `target.column` | required | The column to predict; for `periods`, the date column that places a batch in a period. |
| `target.type` | `numeric` | `numeric`, `binary` or `periods` (two date windows compared; see [step 11](#11-compare-two-periods)). |
| `target.periods` | — | Required for `periods`: exactly two `{name, start, stop}` entries, dates as `YYYY-MM-DD`, both ends inclusive, not overlapping. The second period is 1, the first 0; batches outside both are left out. |
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
sources:    {databricks: false}
sharepoint: {vocabulary: {path: …, drive_id: …}, catalog: {path: …, sheet: …, drive_id: …},
             batch_table: {…}, phf_site_a: {…}, phf_site_b: {…}, …,
             site_maps: {folder: …, files: […], sheet: …, drive_id: …}}
catalog:    {write_new_parameters: true}
build:      {write_parquet: true}
cleaning:   {disable: [], censored: limit, censored_flags: false,
             duration_unit: minutes, yes_no_defaults: {pre_activation_clumps_yn: 'No'}}
mapping:    {exclude: [excluded_mapping_row]}
features:   {co2_target: 5.0, temp_target: 37.0, …}
lv_coa:     {ph_nominal: 7.2, osmo_nominal: 300.0}
```

`sheet:` is optional: without it, a source's worksheet is the one the
vocabulary's `sheet` rows give. `cleaning.yes_no_defaults` and `mapping.exclude`
name columns by their vocabulary key, not by their name; a key the vocabulary
does not define is refused.

Unknown keys, wrong types and incomplete SharePoint entries are refused before
anything runs. So are the old local settings (`paths:`, `sheets:`, `upload:`,
`sources.location`, `sources.fallback_to_local`), each with what replaced it.

#### Sources and SharePoint

Every input is read from its entry under `sharepoint:`, and every stage output is
written there; there is no local data option. Without `io_sharepoint`
installed a run stops before it reads anything. The SharePoint addresses live in
`.env`, which is git-ignored; `config.yaml` refers to them as `${NAME}`, and none
are in the code. The same goes for the Acridine Orange readings `sources.databricks`
fetches: their Databricks parameter ids are in `$DATABRICKS_AO_MAPPING` in `.env`
(`name|parameter id|type` entries separated by `;`, see `.env.example`).

| Stage | Reads | Writes, into |
|---|---|---|
| `ml check-catalog` | vocabulary, catalog, both sites' batch records, the parameter mapping, investigations | `catalog_report.html`, `catalog_manifest.json`, `new_parameters.csv` → the catalog's folder |
| `ml build` | vocabulary and its corrections, catalog, every source, the site maps, `catalog_manifest.json` | `batch_data.xlsx`, `batch_data.parquet`, `build_report.html`, `batch_table_manifest.json` → the batch table's folder |
| `ml task run` | vocabulary, batch table, catalog, both manifests | the run folder in the task's workspace |

Each command prints these lists — file name, drive, folder and sheet — before it
starts, and again with where each file went when it is done.

Reading a manifest back (the build checks the catalog's, a task copies both)
uses `io_sharepoint.download_file_from_sharepoint` where the package has it;
without it the manifest is reported as not read, and the run carries on.
Credentials and drive ids live in `.env`; variables already set in the shell win.
`ML_SHAREPOINT_STANDIN` makes a local folder stand in for SharePoint; it is for
the demo and the tests only.

#### The vocabulary

The repository holds no real column name, category value, site name or
worksheet name. The code refers to each by a key — `col("vector_titer")`,
`value("coa_available_yes")`, `site("a")` — and the vocabulary says what the key
stands for. The real vocabulary is private: an Excel workbook on SharePoint
(`sharepoint.vocabulary`, its path in `$SHAREPOINT_VOCABULARY`). Every stage
loads it first.

Its `Names` sheet has one row per name:

| Kind | Key | Name |
|---|---|---|
| `column` | `vector_titer` | the column's name in the source |
| `column` | `clump_bags` | one row per name, in order, for a list |
| `value` | `coa_available_yes` | a category value the code compares against |
| `site` | `a` | site A's name (`b`: site B's) |
| `sheet` | `phf_site_a` | the worksheet a source's data is on |

Several rows with the same kind and key form a list, in the order they are in.
A name may hold a `{placeholder}` the code fills in, such as `{day}` in
`Lactate to glucose ratio D{day}`. Other columns (a description, say) are
ignored.

`src/ml/names.py` lists every key the code uses, with an invented name for each:
those are the demo's names, and the description of what each key is. A
vocabulary that leaves out a key the code needs is refused before a stage starts,
naming the key; `ml validate` reports them all. Sites are site A and site B
throughout the code and `config.yaml` (`phf_site_a`, `phf_site_b`); their names
come from the `site` rows, and `cohort.sites` in a task uses those names. A task
run keeps a copy of the vocabulary as `inputs/vocabulary.xlsx`, so
`ml task rerun` works from the run folder alone.

`ml check-leaks` reads the same workbook and fails when any of its names, or a
name in its corrections, turns up in a tracked file: run it before pushing.

#### Corrections

Known errors in the sources are rules on the `Corrections` sheet of the
vocabulary workbook, where a data steward can add one without touching the code.
One row per rule, or per replaced pair for the rules that replace text:

| Column | Meaning |
|---|---|
| `Name` | The rule's name; rows sharing a name form one rule. |
| `Site` | The site's name, as the vocabulary names it. |
| `Column` | The column it corrects. |
| `Action` | `set`, `replace_text`, `replace_values` or `divide_by`. |
| `Old` | `replace_text` / `replace_values`: what is replaced. |
| `New` | `set`: the new value (empty means missing) · `replace_*`: the replacement · `divide_by`: the divisor. |
| `When column` | Optional: only rows where this column… |
| `When equals` | …equals this, compared as text (without `When column`: the column itself). |
| `Reason` | Why. |
| `Enabled` | Optional: `no` or `false` switches the rule off. |

In the demo's workbook, for example:

| Name | Site | Column | Action | Old | New | When column | When equals | Reason |
|---|---|---|---|---|---|---|---|---|
| site_country | Southbank | Nation | set | | Eastland | Treatment centre | 20431 | the clinical site is abroad; the export has the wrong country |
| site_b_oos_spelling | Southbank | Out-of-spec class | replace_text | Viabilty | Viability | | | site B spells its out-of-spec classes its own way |
| site_b_oos_spelling | Southbank | Out-of-spec class | replace_text | Low Count | Count | | | site B spells its out-of-spec classes its own way |
| | Southbank | Viable cells per bag | divide_by | | 1000000 | | | stored without the 1e6 divisor |

`cleaning.disable: [name]` switches a rule off for one workspace. The build report lists every
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
   - features the registry computed from the target (`Vector copies per cell` from
     `Product CAR expression (%)`);
   - `predictors.exclude` and the catalog rules;
   - `Vector impurity index`, which is standardised within each site.

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
| `inputs/batch_table.xlsx`, `inputs/catalog.xlsx`, `inputs/vocabulary.xlsx` | The exact inputs this run read, and the vocabulary it ran with. |
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
| `The vocabulary from … leaves out N names the code needs` | Add a `Names` row for each key listed; `src/ml/names.py` says what each one is. |
| `Lines that hold a name from the private vocabulary` (`ml check-leaks`) | Replace each listed name with its key (code) or an invented name (tests, docs), then run it again. |
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
runs are generated, not committed: only `demo/config.yaml` and the two
`task.yaml` files are in git. Tests and the demo use only the invented names of
`src/ml/names.py`; run `uv run ml check-leaks` before pushing.

**Adding a derived feature:** append one `Feature(…)` entry to the registry in
`src/ml/features.py`. It is built only when the catalog lists its name and all
its inputs are present, and it never overwrites a source column. Its `requires` tuple
is what lets a task find, and leave out, features computed from the target.

```
src/ml/
  cli.py          the commands
  config.py       config.yaml: discovery, validation, where each input is read from
  names.py        the vocabulary: every key the code uses, with the demo's invented name
  leaks.py        ml check-leaks
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
