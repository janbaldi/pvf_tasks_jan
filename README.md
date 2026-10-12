# batch_table — training datasets from CAR-T manufacturing data

Turn the site batch records into one checked table of every batch (the batch table), and
cut that table into a documented, reproducible training dataset for each
question you want to model.

- **`ml check-catalog`** checks that the catalog — the list of parameters and their metadata —
  still covers everything the source systems record.
- **`ml build`** merges both sites into the batch table: one row per batch, cleaned, with
  derived features.
- **`ml task run`** turns the batch table into one training dataset for one question: a
  cohort, a target and the columns you choose. It also writes a report and every
  decision behind it.

What it does **not** do: fit models, split data into train/validation sets, or
decide by itself which parameters come too late in the process. The dataset is
one training table; cross-validation and model fitting belong to the next tool,
and leaving out late parameters is a column choice you make in the task file.

**Contents:** [Install](#install) · [Quick start](#quick-start) ·
[Set up with your own data](#set-up-with-your-own-data) ·
[Walkthrough: your first task](#walkthrough-your-first-task) ·
[Use the dataset](#use-the-dataset-in-your-modelling-code) ·
[Commands](#commands) · [task.yaml reference](#taskyaml-reference) ·
[config.yaml reference](#configyaml-reference) ·
[How a dataset is built](#how-a-dataset-is-built) ·
[What a run writes](#what-a-run-writes) · [Troubleshooting](#troubleshooting) ·
[Development](#development)

## Install

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/janbaldi/pvf_tasks_jan.git
cd pvf_tasks_jan
uv sync                     # installs the exact versions in uv.lock
uv run ml --help
```

Every command below is run with `uv run ml …` from inside the repository (or any
folder below it, such as a task folder).

Reading from SharePoint needs the internal `io_sharepoint` package, and joining
Acridine Orange readings needs `databricks_query_tool`. Both are optional: without
them everything runs from local files.

## Quick start

The repository ships a demo: made-up data with the same shape as the real
sources. These five commands build a batch table and two training datasets from it.

```bash
uv run python -m ml.demo demo       # writes demo/data/raw/… (the made-up sources)
cd demo
uv run ml check-catalog                       # SiteA PHF: columns the catalog does not list (1) …
uv run ml build                     # Written — 70 batches × 221 columns
uv run ml task run --task tasks        # runs both example tasks in demo/tasks/
```

Each task writes a run folder such as
`demo/tasks/site_a_car_expression/runs/20261012-154806-f3dc66/`. Open
`report/report.html` in it in a browser, or load `data/dataset.csv`.

## Set up with your own data

You do this once. Afterwards you only rerun `ml check-catalog` and `ml build` when the
sources or the catalog change.

**1. Make a data folder** anywhere outside the repository, for example:

```
~/ml_data/
  catalog.xlsx                 the parameter list and its metadata
  raw/                     local copies of the sources
    phf_site_a.xlsx
    phf_site_b.xlsx
    parameter_requirements.xlsx
    remfg.xlsx
    lv_coa.xlsx
    raw_materials.csv
    investigations_power_query.xlsx
    site_map_clinical_sites.xlsx
```

`ml build` writes the batch table into this folder, and the reports and manifests go
next to it.

**2. Point `config.yaml` at it.** `config.yaml` in the repository root is
annotated. These are the paths you need to set:

```yaml
paths:
  catalog: ~/ml_data/catalog.xlsx                 # read by ml check-catalog, ml build and every task
  batch_table: ~/ml_data/batch_table.xlsx                 # written by ml build
  phf_site_a: ~/ml_data/raw/phf_site_a.xlsx
  phf_site_b: ~/ml_data/raw/phf_site_b.xlsx
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

**3. Put secrets in `.env`.** If you use SharePoint, copy `.env.example` to `.env`
next to `config.yaml` and fill in the credentials. `.env` is git-ignored.

**4. Check, then build:**

```bash
uv run ml validate      # validates config.yaml and the corrections table; runs nothing
uv run ml check-catalog        # lists source columns the catalog does not know about
uv run ml build      # writes batch_table.xlsx, batch_table.parquet, build_report.html and batch_table_manifest.json
```

`ml check-catalog` writes `catalog_report.html` and `new_parameters.csv`. Add any parameter
you want in the batch table to the catalog, then run `ml build`. If you build without
doing that, `ml build` warns you:

```
WARN   cli  The last `ml check-catalog` found 6 parameters the sources record and the catalog does not list; …
```

## Walkthrough: your first task

A **task** is one question asked of the batch table. It lives in its own folder, the
*task workspace*, which holds its `task.yaml` and every run of it. This
walkthrough uses the demo from the [quick start](#quick-start); run it from
`demo/`.

### 1. Create the workspace

```bash
uv run ml task new tasks/car_expression --target "<fp_car_pct>"
```

```
Task workspace created
  File    …/demo/tasks/car_expression/task.yaml
  Config  …/demo/config.yaml
  Next    cd …/demo/tasks/car_expression && ml validate && ml task run
```

The new `task.yaml` is annotated and already contains `config: ../../config.yaml`,
so the task finds its batch table and catalog from wherever you run it.

### 2. Describe the dataset

Replace the contents of `tasks/car_expression/task.yaml` with:

```yaml
config: ../../config.yaml            # the config.yaml that names the batch table and catalog

task:
  name: car_expression
  question: Which manufacturing parameters go with CAR+ expression at SiteA?
  seed: 20260917                     # makes the run repeatable

cohort:                              # which batches are rows
  sites: [SiteA]
  filters:                           # applied in order
    - {column: Type, op: equals, value: Commercial, reason: manufactured commercially}
    - {column: <nonconformance>, op: not_in, value: [termination, withdrawal],
       normalise: true, missing: include, reason: not terminated or withdrawn}

target:                              # what the model will predict
  column: <fp_car_pct>
  type: numeric

columns:                             # which batch table columns go in
  id: 'Patient Lot/Batch #'          # kept as metadata, never a predictor
  group: <vector_lot>                  # kept as metadata, for grouped cross-validation
  exclude:                           # single columns, each with its reason
    - {column: <subject_weight>, reason: dose per kilogram already carries it}
  exclude_categories:                # every parameter in these catalog categories
    Value Type: [datetime]
    Available At: [Wash, Formulation, Final product, Post-thaw, Release, Post-release]

clustering:
  action: report_only                # describe correlated parameters, change nothing

quality:
  min_non_missing: 8                 # the demo cohort is small; the default is 20
```

What each block does:

- **`cohort`** narrows the batch table's 70 batches to the ones the question is about.
  `missing: include` keeps batches with no value in that column; the default,
  `exclude`, drops them. `normalise: true` compares case-insensitively and
  treats `withdrawn` as `withdrawal`.
- **`target`** names the column to predict. Batches without a target value are
  dropped and listed in `cohort.csv`.
- **`columns`** decides which parameters become predictors. `exclude_categories`
  looks up each parameter in the catalog and leaves it out when the named catalog column
  holds one of the listed values. Here it removes the timestamps
  (`Value Type: datetime`) and everything the catalog marks as measured after harvest
  (`Available At`). Any catalog column works, for example `{Category: [QC]}`.
  `include_categories` does the opposite: it keeps only the listed values.

### 3. Check it

```bash
cd tasks/car_expression
uv run ml validate
```

Every column the task names is checked against the batch table, and every catalog column
against the catalog. Inside a workspace you don't need `--task`. A typo shows up
like this, and nothing runs:

```
Problems that would stop a run (1)
  - car_expression: target.column: 'FP Flow CAR (%)' is not a column of the batch table
```

### 4. Run it

```bash
uv run ml task run
```

```
Written — 21 batches × 114 columns
  Task folder  …/tasks/car_expression/runs/20261012-154806-f3dc66
  Report       …/runs/20261012-154806-f3dc66/report/report.html
  Dataset      …/runs/20261012-154806-f3dc66/data/dataset.csv
  Manifest     …/runs/20261012-154806-f3dc66/manifest.json

  WARN   dataset  Only 3 distinct <vector_lot> values: any grouped cross-validation rests on 3 groups, not on 21 batches
  WARN   dataset  137 predictor columns for 21 fitted batches (p/n = 6.5): a model needs strong regularisation, …
```

The warnings are about the data, not errors: they tell you how much evidence
the table really holds.

### 5. Read the run

Every run is a new folder; earlier runs are never overwritten.

```
runs/20261012-154806-f3dc66/
  data/dataset.csv          21 rows: id, group, 111 raw predictors, target
  data/transformed.csv      the same rows, encoded to numbers by metadata/recipe.json
  metadata/decisions.csv    every parameter: included or excluded, and why
  metadata/cohort.csv       every batch table batch: in or out, and which filter decided
  metadata/columns.csv      every dataset column: role, source parameter, type, unit, meaning
  report/report.html        the same, with figures; open it in a browser
  inputs/                   the exact batch table and catalog this run read, and their manifests
  config/task.yaml          the effective settings, pointing at inputs/
  manifest.json             a checksum of every file
```

`decisions.csv` explains every parameter that was left out. These are real rows
from this run:

| Parameter | Decision | Reason | Stage |
|---|---|---|---|
| Patient Lot/Batch # | excluded | the identifier, kept as metadata rather than encoded | role |
| <subject_weight> | excluded | dose per kilogram already carries it | task exclusion |
| <pt_cd3_pct> | excluded | Catalog 'Available At' is 'Post-thaw' | category exclusion |
| <f_vcn_per_cell> | excluded | computed from the target: <fp_car_pct> → <f_vcn_per_cell> | target lineage |
| <f_seeding_pct> | excluded | duplicates '<f_cells_per_bag>' (R²=1.0 over 21 shared batches) | near-duplicate |
| <f_moi_b> | excluded | '<f_moi_b>' has 7 values, below the 8 this task requires | sparsity |

The target is written under a column-safe name (`fp_flow_car` here).
`columns.csv` maps it back: it is the row whose `Role` is `target`.

### 6. Reproduce it

A run folder contains everything it needs to run again:

```bash
uv run ml task rerun runs/20261012-154806-f3dc66
```

```
OK  cli  dataset.csv is identical to the one in 20261012-154806-f3dc66
```

The rerun reads only that folder (`config/task.yaml` and `inputs/`) and writes
a new run next to it. You can copy or archive a run folder and redo it later,
even after the batch table has moved on.

To ask a second question, create a second workspace with `ml task new`. The two
finished examples in `demo/tasks/` show a binary target
(`site_b_release`: `Disposition`, positive class `Released`) and residualised
clusters (`clustering.action: linear`).

## Use the dataset in your modelling code

`dataset.csv` holds the raw predictors. Every encoding step (category means,
de-duplication, clusters) was fitted on the whole cohort, so to get an honest
cross-validated score, refit that preprocessing inside each fold.
`ml.recipe.TaskTransformer` does this as a scikit-learn step. Run from the task
folder:

```python
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.recipe import TaskTransformer, task_data

X, y, groups, spec = task_data(".")      # the task in this folder, cohort and target applied
model = make_pipeline(
    TaskTransformer(spec),                # refits the task's encoding inside every fold
    SimpleImputer(strategy="median", keep_empty_features=True),
    StandardScaler(),
    RidgeCV(alphas=[0.1, 1, 10, 100]),
)
scores = cross_val_score(model, X, y, groups=groups, cv=GroupKFold(n_splits=3), scoring="r2")
print(f"R² per fold: {scores.round(2)}")
```

```
R² per fold: [ 0.12 -0.19  0.03]
```

(Made-up data: the scores mean nothing.) If your tool does its own
preprocessing, read the files directly:

```python
import glob

import pandas as pd

run = sorted(glob.glob("runs/*/"))[-1]                      # the latest run
data = pd.read_csv(f"{run}data/transformed.csv")            # or dataset.csv for raw values
columns = pd.read_csv(f"{run}metadata/columns.csv")
target = columns.loc[columns["Role"] == "target", "Column"].item()
X = data.drop(columns=["Patient Lot/Batch #", "<vector_lot>", target])
y, groups = data[target], data["<vector_lot>"]
```

### Encode new batches with a finished run

`ml task encode` encodes new batches exactly as a run encoded its cohort, with
nothing refitted:

```bash
uv run ml task encode --run runs/20261012-154806-f3dc66 --batch-table ../../data/processed/batch_table.xlsx --out encoded.csv
```

```
OK  cli  70 batches encoded with 20261012-154806-f3dc66's recipe → encoded.csv
```

## Commands

Global options come before the command: `uv run ml [--config FILE] [--debug] COMMAND …`.

| Command | What it does |
|---|---|
| `ml validate [--task T]` | Validate `config.yaml`, the corrections table and the tasks against the batch table and catalog. Runs nothing. |
| `ml check-catalog` | Compare every source's columns with the catalog. Writes `catalog_report.html`, `catalog_manifest.json` and `new_parameters.csv` beside the batch table. |
| `ml build [--mode PRD\|DEV]` | Build the batch table. Writes `batch_table.xlsx`, `batch_table.parquet`, `build_report.html` and `batch_table_manifest.json`. `--mode PRD` also uploads when `upload.enabled` is set. |
| `ml task new FOLDER [--target COL] [--question TEXT] [--force]` | Create a task workspace with an annotated `task.yaml` linked to the current `config.yaml`. |
| `ml task run [--task T]` | Run tasks. Without `--task`, runs the task in the current folder. |
| `ml task rerun RUN_FOLDER` | Redo a finished run from its own folder and report whether `dataset.csv` is identical. |
| `ml update [--mode M] [--task T]` | `catalog`, then `build`, then the given tasks. |
| `ml task encode --run RUN_FOLDER [--batch-table FILE] [--out FILE]` | Encode batches (default: the configured batch table) with a run's fitted recipe. |

`--task` accepts a task workspace folder, a `task.yaml`, or a folder that
contains several workspaces, and can be repeated. `--debug` shows Python
tracebacks instead of one-line error messages.

The config is found in this order: `--config`, the `config:` line of the task,
`$ML_CONFIG`, then the nearest `config.yaml` in the current folder or above it.

## task.yaml reference

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
| `columns.id` | — | Identifies a batch. Must be unique in the cohort. |
| `columns.group` | — | Grouping column (e.g. vector lot); used for grouped folds. |
| `columns.patient` | — | Joins a patient's batches into one group. |
| `columns.include` | all | Use exactly these predictors and nothing else. |
| `columns.exclude` | — | `[{column, reason}]`: single columns to leave out. |
| `columns.exclude_categories` | — | `{catalog column: [values]}`: leave out parameters with these catalog values. |
| `columns.include_categories` | — | `{catalog column: [values]}`: keep only parameters with these catalog values. |
| `clustering.action` | `report_only` | `off`, `report_only`, `representative` or `linear`; see [below](#correlated-parameters). |
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
inclusive, numbers or dates), `is_null`, `not_null`. A filter on a column the batch table
doesn't have stops the run instead of quietly keeping every batch.

## config.yaml reference

`config.yaml` in the repository root is annotated; this is its shape:

```yaml
seed: 20260917
paths:      {catalog: …, batch_table: …, phf_site_a: …, …, reports: (default: beside the batch table)}
sheets:     {phf_site_a: <sheet:phf_site_a>}           # worksheet per workbook, where not the default
sources:    {location: local, fallback_to_local: false, databricks: false}
sharepoint: {catalog: {path: …, sheet: …, drive_id: …}, …, site_maps: {folder: …, files: […], …}}
upload:     {enabled: false, path: …, drive_id: …, file_name: batch_table.xlsx}
catalog:        {write_new_parameters: true}
build:      {write_parquet: true}
cleaning:   {corrections: …, disable: [], censored: limit, censored_flags: false,
             duration_unit: minutes, yes_no_defaults: {…}}
mapping:    {exclude: [<excluded_mapping_row>]}
features:   {co2_target: 5.0, temp_target: 37.0, …}
lv_coa:     {ph_nominal: 7.2, osmo_nominal: 300.0}
```

Unknown keys, wrong types and incomplete SharePoint entries are refused before
anything runs.

### Sources and SharePoint

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

### Corrections

Known errors in the sources are rules in a `corrections.yaml`
(`cleaning.corrections`; without it the package's own table is used):

```yaml
corrections:
  - name: israel_country
    site: SiteB
    column: Country
    when: {column: <clinical_site>, equals: "<value:clinical_sites_b>"}
    set: Israel
    reason: the site is in Israel; the export has the wrong country
  - {site: SiteB, column: <cells_per_bag>, divide_by: 1000000, reason: stored without the 1e6 divisor}
```

The actions are `set`, `replace_text`, `replace_values` and `divide_by`.
`cleaning.disable: [name]` switches a rule off. The build report lists every
rule that ran and the rows it changed. `cleaning.censored` decides what `<0.5`
becomes (`limit`, `half` or `missing`), and `cleaning.yes_no_defaults` names the
yes/no columns whose blanks may be filled.

## How a dataset is built

For each task, `ml task run`:

1. **Selects the cohort**: sites, then each filter in order, then batches without
   a target. Every batch table batch gets a row in `cohort.csv` saying whether it is in,
   and why not.
2. **Chooses the predictors**: the catalog parameters present in the batch table, minus
   - the target, id, group and patient columns;
   - features the registry computed from the target (`<f_vcn_per_cell>` from `<fp_car_pct>`);
   - `columns.exclude` and the category rules;
   - `<f_coa_impurity_index>`, which is standardised within each site.

   Each exclusion is a row in `decisions.csv`.
3. **Fits the preprocessing** on the whole cohort, records it in `recipe.json`,
   and writes the encoded table as `transformed.csv`:
   - Two categories become 0/1. A few become one column each. More become the
     target's mean per category (cross-fitted, so no row sees its own target).
     Very many are hashed into buckets.
   - Each encoder is used only when there are enough rows; otherwise the next
     one down is used. A column with a different value in nearly every batch is
     treated as an identifier and left out.
   - Near-duplicates, near-constant and too-sparse columns are dropped.
4. **Writes the dataset**: `dataset.csv` holds the id, group, the raw predictors
   the recipe uses, and the target. It has one row per batch and no split.
5. **Checks** that the row and column counts agree across every file, then
   renames the run folder from `.incomplete` to its final name. A failed run
   keeps its folder as `.failed`, with the log and the reason in its manifest.

### Correlated parameters

| `clustering.action` | Effect |
|---|---|
| `off` | No groups are looked for. |
| `report_only` | Groups are described in the report; every parameter stays. |
| `representative` | One member of each group stays; the others are dropped. |
| `linear` | One member stays; the others are replaced by what is left after regressing them on it. |

Groups are built by complete linkage on `1 − |correlation|`, cut at
`threshold`. Pairs with fewer than `min_overlap` shared batches are treated as
unknown and never grouped. The representative is chosen without looking at the
target.

## What a run writes

| File | Contents |
|---|---|
| `data/dataset.csv` | The training dataset: id, group, raw predictors, target. One row per batch. |
| `data/transformed.csv` | The same rows encoded by `recipe.json`. |
| `metadata/decisions.csv` | Every parameter: included or excluded, with the reason and the step. |
| `metadata/cohort.csv` | Every batch table batch: in or out, and the first rule that ruled it out. |
| `metadata/columns.csv` | Every dataset column: role, source, type, catalog value type, unit, meaning. |
| `metadata/features.csv` | Every encoded column and the parameter it came from. |
| `metadata/clusters.csv` | Correlated groups, their representatives and what was done. |
| `metadata/missingness.csv` | Parameters that go missing together (when there are any). |
| `metadata/recipe.json` | The fitted preprocessing, used by `ml task encode` and `TaskTransformer`. |
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

## Troubleshooting

| Message | What to do |
|---|---|
| `No config found: there is no config.yaml here or in any folder above` | Run from the repository, add `config:` to the task, set `$ML_CONFIG`, or pass `--config`. |
| `No task to run: run inside a task workspace or pass --task <folder>` | `cd` into a workspace, or name one with `--task`. |
| `… is not a column of the batch table` | Check the spelling against the batch table; `ml validate` lists every such name. |
| `columns.exclude_categories: the catalog has no '…' column` | Use a column that exists in the catalog; the message lists them. |
| `task.purpose: unknown setting` (or any `unknown setting`) | The key was removed or misspelled; the message lists the allowed keys. |
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
`src/ml/features.py`. It is built only when the catalog lists its name and all its
inputs are present, and it never overwrites a source column. Its `requires` tuple
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
