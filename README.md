# CAR-T manufacturing data

Three stages over one dataset, in one package.

| Stage | What it answers | Writes |
|---|---|---|
| `pvf ptf` | What do the source systems record that the PTF has never been told about? | `ptf_report.html`, `ptf_manifest.json`, optionally `new_parameters.csv` |
| `pvf build` | What happened to every batch, at both sites, in one table? | `PVF.xlsx` (+ `PVF.parquet`), `pvf_manifest.json`, `build_report.html` |
| `pvf tasks` | What does one cohort and one target look like as a dataset someone else can use? | one folder per run under `outputs/tasks/<task name>/` |

They are in that order because each depends on the one before. The PTF is the
schema: it names every parameter the PVF may hold. The build fills that schema in
from the site files and everything joined to them. A task narrows what the build
produced to one question.

## Quick start

```bash
uv sync                                # Python 3.12, the versions in uv.lock
uv run pvf init my_workspace --demo    # a workspace full of made-up data
cd my_workspace
uv run pvf all                         # ptf, build, and every task in tasks/
uv run pvf new-task my_question --target "FP Flow CAR+ (%)" --purpose predictive
uv run pvf check --task tasks/my_question.yaml
uv run pvf tasks --task tasks/my_question.yaml
```

Without `--demo`, `pvf init` writes the same layout with an annotated config and
an empty `data/raw/` for the real sources.

| Command | Does |
|---|---|
| `pvf init <folder> [--demo]` | create a workspace: `pvf.yaml`, `corrections.yaml`, `tasks/`, `data/`, `outputs/` |
| `pvf new-task <name or path>` | write an annotated task file — in `tasks/`, or anywhere if given a path |
| `pvf check [--task …]` | validate the config, the corrections table and the tasks against the PVF; run nothing |
| `pvf ptf` / `pvf build` / `pvf tasks` | the three stages; `pvf tasks` with no `--task` runs every task in `tasks/` |
| `pvf all [--task …]` | the three in order |
| `pvf apply --run <run folder>` | encode new batches with a finished run's fitted recipe |

`--task` takes a file or a folder and can be repeated. Errors that are yours to
fix (a typo in a config, a filter on a column the PVF does not have) are printed
as one message naming the field, with exit code 2; `--debug` shows the traceback.

## Code, configuration and data are kept apart

This repository is code. Configuration and data live in a **workspace**: any
folder with a `pvf.yaml` in it.

```
my_workspace/
  pvf.yaml            paths, sources, SharePoint locations, cleaning and feature settings
  .env                SharePoint credentials and other secrets (git-ignored)
  corrections.yaml    known data corrections, as a table
  tasks/              one YAML per task
  data/raw/           local copies of the sources
  data/processed/     PVF.xlsx, PVF.parquet, ptf_manifest.json, pvf_manifest.json
  outputs/reports/    ptf_report.html, build_report.html
  outputs/tasks/      one folder per task run
```

A command finds its config by `--config <file>`, then `$PVF_CONFIG`, then the
nearest `pvf.yaml` in the working directory or above it (or above the task file),
then the old `config/config.yaml`. **Every relative path in a config or a task
file is relative to that file**, so a workspace — or a task file — can be moved
and still reads the same inputs.

### A task can live anywhere

A task file names its workspace, relative to itself, and inherits the PVF, the
PTF and the output folder from it:

```yaml
workspace: ../../shared/pvf_workspace/pvf.yaml
task: {name: my_question, purpose: predictive, seed: 20260917}
target: {column: FP Flow CAR+ (%), type: numeric}
...
```

`pvf new-task ~/analyses/q3/my_question.yaml` writes exactly that, so a task
for a new question can sit with the analysis it belongs to rather than inside
the package or the workspace. Anything a task wants to say for itself — its own
`inputs: {pvf: …, ptf: …}` or `output: {root: …}` — overrides the workspace.

## The workspace config

`pvf init` writes an annotated `pvf.yaml`; this is its shape.

```yaml
paths:        {ptf: data/raw/PTF.xlsx, phf_ghent: …, pvf: data/processed/PVF.xlsx, tasks: outputs/tasks, task_files: tasks}
sources:      {location: local, fallback_to_local: false, databricks: false}
sharepoint:   {ptf: {path: …, sheet: Catalogue, drive_id: DRIVE_ID_GHENT}, phf_ghent: {…}, site_maps: {folder: …, files: […]}}
upload:       {enabled: false, path: …, drive_id: DRIVE_ID_GHENT, file_name: PVF.xlsx}
build:        {write_parquet: true}
cleaning:     {corrections: corrections.yaml, disable: [], censored: limit, censored_flags: false, …}
mapping:      {exclude: [Non-Conformance Type Calc.]}
features:     {co2_target: 5.0, …}
lv_coa:       {ph_nominal: 7.2, osmo_nominal: 300.0}
```

Unknown settings, wrong types and missing SharePoint fields are refused before
anything runs, each naming its field.

### Sources and SharePoint

Which sources a run reads is a setting, not a consequence of what happens to be
installed. With `sources.location: local` every input is the local file under
`paths:`, and an installed `io_sharepoint` changes nothing. With `sharepoint`,
**every input that has an entry under `sharepoint:` is read from that location**
— the PTF, both PHFs, the mapping, the Re-MFG supplement, the vector
certificates, the consumables workbook, the investigations workbook, the
clinical-site mapping files, and the PVF a task reads. An input without an entry
is still read locally. No SharePoint address is written in the code; they are all
in the config. A missing `io_sharepoint` stops the run unless `fallback_to_local`
allows otherwise, and a fallback that is taken is logged.

Provenance follows the read: a source that came from SharePoint is recorded by its
SharePoint location and a digest of the table that came back, and a local file by
the digest of its bytes. Each stage records only the sources it actually read.

Uploading is a separate switch, and only ever happens on a `--mode PRD` build with
`upload.enabled: true`; the destination is `upload.path` (or `$DATA_LINK`) and
`upload.drive_id`, and the manifest records where it went. Reading from the share
does not imply writing to it, and a local run cannot upload by accident.
Secrets — whatever `io_sharepoint` needs to reach SharePoint — go in a **`.env`
file in the workspace folder, next to `pvf.yaml`** (`pvf init` writes a
`.env.example` to copy). It is loaded with the config, never committed (`.env` is
git-ignored everywhere), and variables already set in the shell or by a scheduler
win over it. A `.env` in the working directory is read too. `MODE=PRD` and
`DATA_LINK` (the upload folder) can live there as well.

### Cleaning and the corrections table

The known data corrections — a clump count Excel turned into a date, a clinical
site under the wrong country, cell counts stored without their 1e6 divisor — are
facts about the sources, not about the code, so they are rules in the workspace's
`corrections.yaml`:

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

Actions are `set`, `replace_text` (substrings, in order), `replace_values` (whole
values) and `divide_by` (applied after numeric coercion). `cleaning.disable`
switches named rules off; `raritan_type_commercial: false` still switches off the
rule that fills the Raritan `Type` column. The build report lists every rule that
ran, the rows it touched and why.

Two cleaning decisions are settings rather than facts:

- `cleaning.censored` — what a value written `<0.5` becomes: `limit`, `half` or
  `missing`. With `censored_flags: true` each such column also gets a
  `<column> censored` 0/1 indicator, so the operator survives whichever policy
  rewrote the number.
- `cleaning.yes_no_defaults` — a blank in a yes/no column is *unknown*. It is
  filled only for the columns this names, in each site's own spelling.

`cleaning.duration_unit` says what a bare number in a duration cell means —
`HH:MM` and `HH:MM:SS` are read as clock times whatever it says — and a ratio
written `a:b` keeps its right-hand side, because these columns are recorded
normalised to 1.

## A task is a YAML file

One task per file: a cohort, a target, the role of every column, what is known at
the prediction point, what to do about parameters that move together, and how to
split. Defining a second task means writing a second file — there is no Python to
edit. `pvf new-task` writes an annotated one; `demo/tasks/` has two that differ in
every one of those things.

```yaml
workspace: ../pvf.yaml
task:    {name: ghent_car_expression, purpose: predictive, seed: 20260917}
cohort:
  sites: [Ghent]
  filters:
    - {column: Type, op: equals, value: Commercial, reason: manufactured commercially}
    - {column: Non-Conformance Type, op: not_in, value: [termination, withdrawal],
       normalise: true, missing: include, reason: not terminated or withdrawn}
target:  {column: FP Flow CAR+ (%), type: numeric}
columns: {id: 'Patient Lot/Batch #', group: Vector Lot, patient: Patient ID}
availability: {cutoff: Harvest}
clustering: {action: report_only, method: spearman, threshold: 0.7, min_overlap: 10}
split:   {strategy: grouped, validation_fraction: 0.25, folds: 5}
```

Everything is validated before the run touches the output directory, and every
complaint names the field and what to do about it.

- **Filters** are a fixed set of operators — `equals`, `not_equals`, `in`,
  `not_in`, `min`, `max`, `is_null`, `not_null` — applied in the order written.
  No expressions and no Python come out of YAML. `missing:` says what happens to
  a row whose value is absent, and `normalise: true` flattens case, spacing and
  the known aliases (`withdrawn` → `withdrawal`) for the comparison while the
  PVF keeps what the source actually said. A filter whose column is missing
  stops the run rather than quietly widening the cohort.
- **Quote a column name with a `#`, `:` or `%` in it.** Unquoted,
  `Patient Lot/Batch #` is read as `Patient Lot/Batch` followed by a comment.
- The old `tasks:` block in a config still works — `pvf tasks` translates it,
  cohort rules included — but it cannot say most of what a task file can.

## Roles, leakage and what the dataset is

A task declares which column is the target, which is the identifier, which is the
grouping metadata and, optionally, which is the patient. Those are authoritative:

- The target and the role columns are removed from the predictor candidates
  **before** anything is encoded, de-duplicated or clustered. They are carried
  into the output as metadata, once each. An identifier that is not unique in the
  cohort stops the run: a batch twice in a dataset is counted twice and can sit on
  both sides of a split.
- Anything the feature registry computed **from the target** is removed with
  them, walked through `Feature.requires`: `VCN/cell` and
  `Flow accuracy (effective)` are both derived from `FP Flow CAR+ (%)`, and
  `decisions.csv` names the chain that got each one excluded. Derived certificate
  columns and censoring indicators are covered too.
- **What is known at the prediction point is checked mechanically.** The PTF's
  optional `Available At` column says from which process stage a parameter exists
  (`D0 D1 D3 D6 D8 D10 Harvest Wash Formulation "Final product" Post-thaw Release
  Post-release`; a task can replace the list in `availability.stages`). A derived
  column is as late as its latest input. A predictive task names its
  `availability.cutoff`, and everything from a later stage — or with no stage at
  all — is left out, with the stage in `decisions.csv`. `availability.unavailable`
  adds exclusions by hand and `availability.available` lets a parameter through
  whatever its stage. Without the PTF column, a predictive task has to list the
  late parameters itself and state `availability.reviewed: true`, as before.
- A predictor that correlates with the target above `quality.proxy_correlation`
  (0.95) on the fitted rows is named in the report: it is often the target
  measured another way.
- `LVCoA_Impurity_Index_zsum` is standardised within its own site's batches, so
  it is exploratory only and a predictive task drops it.

**What is learned is learned on training rows.** Category vocabularies, target
means, near-duplicate selection, clusters, residual regressions, the sparsity and
near-constant checks, indicators and imputation medians are fitted on one
population and applied unchanged to any other:

| The task says | `dataset.csv` is | Beside it | Fitted on |
|---|---|---|---|
| `purpose: exploratory` | the transformed table, labelled exploratory | `raw_features.csv` | the whole cohort |
| `purpose: predictive`, no `split` | the selected raw predictors | `transformed.csv` | the whole cohort, for description only |
| `purpose: predictive` with a `split` | the selected raw predictors, with `split` and `fold` | `transformed.csv` | the training rows alone |

A table preprocessed over every row is not a table to cross-validate on
afterwards. Two ways to do it properly:

- `metadata/recipe.json` holds the fitted decisions as plain JSON and is read
  back by `Preprocessor.from_state` — `pvf apply --run <folder> --pvf <file>`
  encodes new batches exactly as the training rows were encoded.
- `pvf.recipe.TaskTransformer` is the same preprocessing as a scikit-learn step,
  so it can be refitted inside every fold:

  ```python
  from pvf.recipe import TaskTransformer, task_data
  X, y, groups, spec = task_data("tasks/ghent_car_expression.yaml")
  model = make_pipeline(TaskTransformer(spec), SimpleImputer(), StandardScaler(), RidgeCV())
  cross_val_score(model, X, y, groups=groups, cv=GroupKFold(3))
  ```

Target encoding is cross-fitted: each fold's prior *and* its category means come
from that fold's training rows only, folds respect the grouping column or the
chronological order, and a category the training rows never held gets the
training prior.

### Splits and folds

`split.strategy: grouped` holds out whole groups, drawn with the task's seed
(and, for a binary target, keeping the classes balanced where the groups allow).
With `columns.patient`, batches that share a patient are kept together with their
groups, so a re-manufactured batch never sits across the split from the first one.
`chronological` holds out the last batches by `split.order_column`.
`split.folds: k` adds k cross-validation folds over the training rows — grouped
and class-balanced, or time-ordered blocks for a chronological task — written to
`data/splits.csv` and to the `fold` column. Asking for folds without a strategy
gives folds over the whole cohort and no validation rows.

The report says how much evidence the table really holds: how many distinct
groups a grouped estimate rests on, the ratio of predictor columns to fitted
batches, and — for a binary target — the events per variable.

### Encodings

A parameter with two categories is 0/1, one with few categories one column per
category, one with several the target's mean per category, and one with very many
is hashed into buckets — **but only where the rows can support it**: target
encoding needs `encoding.target_min_rows_per_category` (5) rows per category and
hashing `encoding.hashing_min_rows` (100) rows, otherwise the next encoder in line
is used. A column with nearly a category per batch is an identifier and is left
out. An absent value is absent in every encoder's output. A column that is
constant to measurement precision on the fitted rows (`quality.near_constant_tolerance`)
is dropped: an empty one-hot tail, an unused hash bucket, or a ratio that only
differs in its seventh digit — which, scaled to unit variance, would dominate any
model. `encoding.missing_indicators: true` adds a 0/1 column for every column with
gaps, and `encoding.impute: median` fills gaps with the training median.

## Parameters that move together

Discovery and what is done about it are separate settings. `clustering.action`:

| Action | What happens |
|---|---|
| `off` | no groups are looked for |
| `report_only` | the groups are described; every parameter goes into the dataset unchanged |
| `representative` | one member of each group is kept, the rest dropped |
| `linear` | one member is kept, the others residualised on what is retained before them |
| `nonlinear` | the same, by whichever curve fits — experimental, kept for compatibility |

Correlation becomes a distance (`1 - |r|`, Spearman by default), complete linkage
builds the tree and it is cut at the threshold. Below `min_overlap` shared
batches a relationship is **unknown**, drawn blank in the heatmap and never merged
into a group, and residualising is not claimed to preserve everything or to make
columns independent. The representative is chosen without the target.
Near-duplicates are a deterministic sweep, so a dropped parameter always names a
representative that is still in the dataset.

Parameters that go **missing** together are described too (`missingness:` in a
task, on by default): groups of columns with the same gaps, and the categorical
parameter that best explains when they are missing. That is description only.

## What a task run leaves behind

Each run writes a fresh folder, `outputs/tasks/<task name>/<run id>`. Nothing is
ever overwritten. The run writes into a staging folder marked incomplete and
promotes it only once the required artefacts are there and the counts inside them
agree; a run that fails keeps its folder marked `.failed`, with its log and the
reason in the manifest.

| File | What it is |
|---|---|
| `manifest.json` | the run's manifest (see below) |
| `config/task.yaml` | the effective settings, defaults included |
| `data/dataset.csv` | the task's table, with its role stated in the manifest and the report |
| `data/raw_features.csv` | the untransformed predictors, when `dataset.csv` is the transformed view |
| `data/transformed.csv` | the transformed view, when `dataset.csv` is the raw predictors |
| `data/splits.csv` | split and fold assignments, when asked for |
| `metadata/columns.csv` | one row per exported column in output order: role, source, type, unit, transformation, meaning |
| `metadata/features.csv` | the feature dictionary: every column the recipe produces, and its parameter |
| `metadata/decisions.csv` | every parameter included, excluded or changed, with the reason and the stage |
| `metadata/cohort.csv` | every batch in the PVF: in or out, and the first filter that ruled it out |
| `metadata/clusters.csv` | correlated groups: membership, representatives, actions, singletons and skipped parameters |
| `metadata/missingness.csv` | parameters missing together, and their best explainer |
| `metadata/recipe.json` | the fitted preprocessing state, readable by `pvf apply` and `Preprocessor.from_state` |
| `report/report.html` | the Streamlit report |
| `report/report_payload.json` | the same numbers, usable without a browser |
| `provenance/ptf_manifest.json` | the manifest of the `pvf ptf` run that checked the PTF |
| `provenance/pvf_manifest.json` | the manifest of the `pvf build` run that wrote the PVF |
| `logs/run.log` | the events of this run and this stage only |

`--report-only` writes the report and its metadata to a fresh folder of its own.
It still runs the whole analysis, but writes no dataset and its manifest says the
package is a report rather than a dataset.

### Manifests

`pvf ptf` and `pvf build` write `ptf_manifest.json` and `pvf_manifest.json` beside
the PVF; each task run writes `manifest.json`. All three follow one format
(`pvf.manifest`): `manifest_version`, `stage`, `status`, `run_id`, `started`,
`finished`, `config` (path and digest), `inputs` (every source as it was actually
read — where from, local or remote, digest), `outputs` (every file written, with
its size and sha256, by a path relative to the manifest), `warnings`, `errors`,
`provenance` (commit, seed, library versions), and the stage's own facts. A
manifest is validated before it is written, and `pvf.manifest.verify(path)` checks
one on disk, including whether any output has changed since.

A task copies the two stage manifests into its folder and checks that the PVF
manifest describes the PVF it read — by the file's bytes for a local file, by the
table's content for one read from SharePoint — and warns when it does not. A
missing manifest is a warning too, recorded as `not found`.

## Parameters and features

A **parameter** is something a source recorded. It may be cleaned, renamed,
rescaled or harmonised on the way in, but nothing computes it. Parameters are the
business of `io.py`, `clean.py` and `enrich.py`.

A **feature** is arithmetic over one or more parameters — an MOI, a slope, a
ratio, a profile class. Every feature the build computes lives in one ordered
registry in `src/pvf/features.py`, one entry each:

```python
Feature(
    "MOI_per_CD3_pool",
    (TITER, VOL_A, VOL_B, "PT Total  CD3+ (Viable Cells)"),
    lambda d: _first(d, *TITER) * safe_div(_pooled_vector_volume(d), d["PT Total  CD3+ (Viable Cells)"]),
    "MOI", "D3", "numeric",
    "Infectious units per CD3+ cell rather than per cell of any kind — only T cells are the target",
)
```

To add one, append an entry. Nothing else changes. The `requires` tuple is what a
task walks to find the descendants of its target, and to find the stage from
which a feature exists.

- **A feature is created only if the PTF lists its name.**
- **A feature whose inputs are not all present is not created**, and the report
  names the inputs it wanted.
- **A feature never overwrites a column a source supplies.**

The certificate loader derives columns of its own while parsing the workbook —
below-LOQ flags, delivered dose, donor summaries. Each is declared in
`features.coa_lineage()` with its inputs.

## Reports

Every stage writes a Streamlit page that needs no Streamlit server: the file
carries its own Python runtime through stlite. Open it; there is nothing to run.
It fetches that runtime from a CDN, **so a fresh report needs network access
once**, and it spends 10 to 30 seconds starting Python before the first figure
appears. Every number was computed before the file was written.

A task report has eight sections: overview (question, target, cohort, what the
table is for, what needs attention), cohort, data quality, parameters and
encodings (the dictionary, every decision, the availability stages, possible
target proxies), correlated groups, parameters missing together, the dataset, and
provenance with the run log.

## The demo

`pvf init <folder> --demo` writes a workspace of made-up sources (from
`pvf.demo`). The numbers are invented; the *shapes* are not: every path the stages
can take is represented by something in that data. The committed `demo/` is one
such workspace after `pvf all`: a 204-parameter PTF with an `Available At` column,
a 70 × 221 PVF, and two tasks. `demo/run.log` is the console output.

| | `tasks/task_car_expression.yaml` | `tasks/task_release_outcome.yaml` |
|---|---|---|
| Question | what goes with CAR+ expression at Ghent | what a released Raritan batch looks like |
| Cohort | Ghent, commercial, finished, no withdrawals | Raritan, finished |
| Target | `FP Flow CAR+ (%)`, numeric | `Disposition`, binary, positive class `Released` |
| Purpose | predictive, cutoff `Harvest`, grouped 30% validation split, folds | exploratory |
| Clustering | `report_only` | `linear`, threshold 0.8 |
| Result | 21 batches, 83 raw predictors (95 transformed); 50 parameters left out as later than harvest | 27 × 176 transformed, 30 groups residualised |
| Written to | `demo/outputs/tasks/ghent_car_expression/<run>/` | `demo/outputs/explorations/raritan_release/<run>/` |

## Code layout

```
src/pvf/
  cli.py               the commands, and the one place the stages are run from
  config.py            the workspace config: discovery, validation, where each input is read from
  scaffold.py          pvf init and pvf new-task; templates/ holds the files they write
  taskconfig.py        the task YAML, and the validation it has to pass
  ptf.py               stage 1: source parameters the PTF does not list
  io.py                reading every source, local or SharePoint
  clean.py             type coercion and harmonisation
  corrections.py       the corrections table
  enrich.py            parameters joined from other systems
  features.py          the feature registry, and derived-column lineage
  merge.py             stacking the sites, and the checks over the result
  dataset.py           stage 3: cohort, roles, availability, split, fit/transform, assembly
  encode.py            one fit/apply pair per encoding
  decorrelate.py       correlated groups, representatives and residuals
  missingness.py       parameters missing together
  recipe.py            a fitted recipe applied to new rows, and as a scikit-learn transformer
  package.py           the run folder: staging, promotion
  manifest.py          the one manifest format
  provenance.py        commit, input digests, environment, config digest
  blocks.py plots.py report.py streamlit_app.py stlite.py   the reports
  logger.py            the run log the reports show, scoped per run
  demo.py              made-up sources for the demo and the tests
tests/                 one run of every stage over the demo data, plus focused regressions
demo/                  a demo workspace after `pvf all`
```

## Development

Python 3.12 with uv; `uv.lock` pins every version, so `uv sync --locked` gives the
environment the tests ran in. `uv run pytest` runs every stage over made-up data,
and `uv run ruff check src tests` lints; CI runs both on every push
(`.github/workflows/ci.yml`). `io_sharepoint` and `databricks_query_tool` are
internal packages, available where the pipeline runs in production; whether they
are used is `sources.location` and `sources.databricks`, not whether they are
installed. Streamlit is deliberately not a dependency: the reports run it in the
browser.
