# CAR-T manufacturing data

Three stages over one dataset, in one package.

| Stage | What it answers | Writes |
|---|---|---|
| `pvf ptf` | Is the PTF (the feature metadata) still up to date with what the sources record? | `ptf_report.html`, `ptf_manifest.json`, optionally `new_parameters.csv` |
| `pvf build` | What happened to every batch, at both sites, in one table? | `PVF.xlsx` (+ `PVF.parquet`), `pvf_manifest.json`, `build_report.html` |
| `pvf tasks` | What does one cohort and one target look like as a training dataset? | a new run folder in the task's workspace |

There is one PTF and one PVF. They live in a data folder of your own, and
`config.yaml` points at them. Each task — one question asked of the PVF — gets a
workspace folder of its own, and every run of it carries everything needed to
redo it. The dataset is one training table; cross-validation happens in the tool
that fits the models.

## 1. Set up (once)

```bash
git clone <this repository> && cd pvf_tasks_jan
uv sync                      # Python 3.12, the versions in uv.lock
```

1. **Make a data folder** anywhere, for example `~/pvf_data/`, and put the PTF in
   it plus local copies of the sources in `~/pvf_data/raw/` (or read them from
   SharePoint, below).
2. **Edit `config.yaml`** in this repository: point `paths:` at that folder.
   `paths.ptf` is the PTF, `paths.pvf` is where `pvf build` writes the PVF; the
   reports and manifests go beside the PVF. Relative paths are relative to
   `config.yaml`, and `~` is your home folder.
3. **Secrets** (SharePoint credentials, drive ids) go in a `.env` next to
   `config.yaml` — copy `.env.example`. It is git-ignored.
4. **Check, then build:**

   ```bash
   uv run pvf check     # the config and the corrections table; nothing is run
   uv run pvf ptf       # is the PTF still up to date with the sources?
   uv run pvf build     # write the PVF
   ```

Rerun `pvf ptf` and `pvf build` whenever the sources or the PTF change. `pvf
build` warns when the last `pvf ptf` found parameters the PTF does not list, and
a task warns when the PTF changed after the PVF was built.

A command finds `config.yaml` by `--config <file>`, then the `config:` a task
names, then `$PVF_CONFIG`, then the nearest `config.yaml` in or above the working
directory.

## 2. Tutorial: a first task, on the demo data

The demo is made-up data with the same shapes as the real sources, so every step
below runs as written.

**Make the demo and build its PVF.** This writes `demo/config.yaml`, the sources
under `demo/data/raw/` and two example tasks under `demo/tasks/`.

```bash
uv run python -m pvf.demo demo
cd demo
uv run pvf ptf        # "SiteA PHF: columns the PTF does not list (1)" …
uv run pvf build      # "Written — 70 batches × 221 columns"
```

**Create a task workspace.** `pvf init` makes the folder and an annotated
`task.yaml` that already points at this `config.yaml`:

```bash
uv run pvf init tasks/car_expression --target "<fp_car_pct>" --purpose predictive
cd tasks/car_expression
```

**Say what the dataset is.** Open `task.yaml` and change four things: the cohort,
the columns, what is known when the prediction is made, and — because the demo
cohort is small — the minimum number of values a column needs.

```yaml
cohort:
  sites: [SiteA]
  filters:
    - {column: Type, op: equals, value: Commercial, reason: manufactured commercially}

columns:
  id: 'Patient Lot/Batch #'
  group: <vector_lot>
  exclude:                                   # single columns, each with a reason
    - {column: <subject_weight>, reason: dose per kilogram already carries it}
  exclude_categories:                        # every parameter in a PTF category
    Value Type: [datetime]

availability:
  cutoff: Harvest                            # nothing measured after harvest

quality: {min_non_missing: 8}
```

`exclude_categories` takes any PTF column and the categories of it to leave out;
`include_categories` is the opposite: only those categories are kept.

**Check it, then run it.** Inside a workspace no `--task` is needed:

```bash
uv run pvf check      # every column it names is checked against the PVF and PTF
uv run pvf tasks      # "Written — 34 batches × 94 columns"
```

**Look at the run.** It is in `runs/<run id>/`:

```
runs/20261012-145259-dfebcd/
  data/dataset.csv          the training dataset: id, group, raw predictors, target
  data/transformed.csv      the same rows encoded by metadata/recipe.json
  metadata/decisions.csv    every parameter in or out, and why
  report/report.html        open it in a browser
  inputs/                   the PVF and PTF this run read, and their manifests
  config/task.yaml          the effective settings, pointing at inputs/
```

`decisions.csv` has a row for each parameter the task left out — the datetime
parameters say `PTF 'Value Type' is 'datetime'`, the late ones `available at
Post-thaw, after the prediction point (Harvest)`.

**Redo it.** A run folder is enough to reproduce itself:

```bash
uv run pvf rerun runs/20261012-145259-dfebcd   # "dataset.csv is identical to the one in …"
```

The two finished examples in `demo/tasks/` show the rest: a binary target,
residualised clusters and a category exclusion.

## Commands

| Command | Does |
|---|---|
| `pvf check [--task …]` | validate the config, the corrections table and the tasks against the PVF; run nothing |
| `pvf ptf` / `pvf build` | the PTF check and the PVF build |
| `pvf init <folder>` | create a task workspace: the folder and an annotated `task.yaml` |
| `pvf tasks [--task …]` | run tasks; default: the workspace in the current folder |
| `pvf rerun <run folder>` | redo a run from its own folder, and say whether `dataset.csv` came out the same |
| `pvf all [--task …]` | ptf, build and the given tasks, in that order |
| `pvf apply --run <run folder>` | encode new batches with a finished run's fitted recipe |

`--task` takes a task workspace, its `task.yaml`, or a folder of workspaces, and
can be repeated. Errors that are yours to fix (a typo in a config, a filter on a
column the PVF does not have) are printed as one message naming the field, with
exit code 2; `--debug` shows the traceback.

# Reference

## config.yaml

`config.yaml` is annotated; this is its shape.

```yaml
paths:        {ptf: ~/pvf_data/PTF.xlsx, pvf: ~/pvf_data/PVF.xlsx, phf_site_a: …, reports: (default: beside the PVF)}
sources:      {location: local, fallback_to_local: false, databricks: false}
sharepoint:   {ptf: {path: …, sheet: Catalogue, drive_id: DRIVE_ID_SITE_A}, phf_site_a: {…}, site_maps: {folder: …, files: […]}}
upload:       {enabled: false, path: …, drive_id: DRIVE_ID_SITE_A, file_name: PVF.xlsx}
build:        {write_parquet: true}
cleaning:     {corrections: corrections.yaml, disable: [], censored: limit, censored_flags: false, …}
mapping:      {exclude: [<excluded_mapping_row>]}
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
file next to `config.yaml`** (copy `.env.example`). It is loaded with the config, never committed (`.env` is
git-ignored everywhere), and variables already set in the shell or by a scheduler
win over it. A `.env` in the working directory is read too. `MODE=PRD` and
`DATA_LINK` (the upload folder) can live there as well.

### Cleaning and the corrections table

The known data corrections — a clump count Excel turned into a date, a clinical
site under the wrong country, cell counts stored without their 1e6 divisor — are
facts about the sources, not about the code, so they are rules in a
`corrections.yaml` (`cleaning.corrections`; the package's own table when unset):

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

Actions are `set`, `replace_text` (substrings, in order), `replace_values` (whole
values) and `divide_by` (applied after numeric coercion). `cleaning.disable`
switches named rules off; `site_b_type_commercial: false` still switches off the
rule that fills the SiteB `Type` column. The build report lists every rule that
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

## task.yaml

One task per workspace: a cohort, a target, which columns go in, what is known at
the prediction point, and what to do about parameters that move together.
Defining a second task means writing a second file — there is no Python to edit.
`pvf init` writes an annotated one; `demo/tasks/` has two that differ in every
one of those things.

```yaml
config: ../../config.yaml   # the config with the PVF and PTF; or inputs: {pvf: …, ptf: …}
task:    {name: site_a_car_expression, purpose: predictive, seed: 20260917}
cohort:
  sites: [SiteA]
  filters:
    - {column: Type, op: equals, value: Commercial, reason: manufactured commercially}
    - {column: <nonconformance>, op: not_in, value: [termination, withdrawal],
       normalise: true, missing: include, reason: not terminated or withdrawn}
target:  {column: <fp_car_pct>, type: numeric}
columns:
  id: 'Patient Lot/Batch #'
  group: <vector_lot>
  exclude: [{column: <subject_weight>, reason: dose per kilogram already carries it}]
  exclude_categories: {Available At: [Release, Post-release]}
availability: {cutoff: Harvest}
clustering: {action: report_only, method: spearman, threshold: 0.7, min_overlap: 10}
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
- **Which columns go in** is `columns:`. `exclude` leaves out single columns,
  each with its reason. `exclude_categories` leaves out every parameter whose
  value in a PTF column is one of the listed categories — any PTF column, so
  `{Category: [QC]}` works as well as `{Available At: [Release]}`.
  `include_categories` keeps only the listed categories. A PTF column that does
  not exist stops the run. `include` is an explicit predictor list.
  Every exclusion is a row in `decisions.csv`.
- **Quote a column name with a `#`, `:` or `%` in it.** Unquoted,
  `Patient Lot/Batch #` is read as `Patient Lot/Batch` followed by a comment.

## Roles, leakage and what the dataset is

A task declares which column is the target, which is the identifier, which is the
grouping metadata and, optionally, which is the patient. Those are authoritative:

- The target and the role columns are removed from the predictor candidates
  **before** anything is encoded, de-duplicated or clustered. They are carried
  into the output as metadata, once each. An identifier that is not unique in the
  cohort stops the run: a batch twice in a dataset is counted twice and can sit in
  two cross-validation folds at once.
- Anything the feature registry computed **from the target** is removed with
  them, walked through `Feature.requires`: `<f_vcn_per_cell>` and
  `<f_flow_accuracy>` are both derived from `<fp_car_pct>`, and
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
- `<f_coa_impurity_index>` is standardised within its own site's batches, so
  it is exploratory only and a predictive task drops it.

**There is one dataset and no split.** Every batch in the cohort is a row of
`dataset.csv`; cross-validation belongs to the tool that fits the models. Category
vocabularies, target means, near-duplicate selection, clusters, residual
regressions, the sparsity and near-constant checks, indicators and imputation
medians are fitted on the whole cohort and recorded in `recipe.json`.
`dataset.csv` is always the raw predictors the recipe uses; `transformed.csv`
beside it is the same rows through the recipe. `purpose: predictive` adds the
availability checks above; `exploratory` describes stages without leaving
anything out.

A table preprocessed over every row is not a table to cross-validate on as it
stands. For an honest estimate, refit the preprocessing inside every fold:

- `metadata/recipe.json` holds the fitted decisions as plain JSON and is read
  back by `Preprocessor.from_state` — `pvf apply --run <folder> --pvf <file>`
  encodes new batches exactly as the cohort was encoded.
- `pvf.recipe.TaskTransformer` is the same preprocessing as a scikit-learn step:

  ```python
  from pvf.recipe import TaskTransformer, task_data
  X, y, groups, spec = task_data("demo/tasks/site_a_car_expression")
  model = make_pipeline(TaskTransformer(spec), SimpleImputer(), StandardScaler(), RidgeCV())
  cross_val_score(model, X, y, groups=groups, cv=GroupKFold(3))
  ```

Target encoding is cross-fitted: each fold's prior *and* its category means come
from that fold's training rows only, folds keep the grouping column (joined up by
`columns.patient`) whole, and a category the training rows never held gets the
training prior.

The report says how much evidence the table really holds: how many distinct
groups a grouped estimate rests on, the ratio of predictor columns to batches,
and — for a binary target — the events per variable.

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

Each run writes a fresh folder, `<task workspace>/runs/<run id>`. Nothing is
ever overwritten. The run writes into a staging folder marked incomplete and
promotes it only once the required artefacts are there and the counts inside them
agree; a run that fails keeps its folder marked `.failed`, with its log and the
reason in the manifest.

| File | What it is |
|---|---|
| `manifest.json` | the run's manifest (see below) |
| `config/task.yaml` | the effective settings, defaults included; its inputs are the copies in `inputs/` |
| `inputs/PVF.xlsx`, `inputs/PTF.xlsx` | the PVF and PTF this run read (byte for byte when local, as read when from SharePoint) |
| `data/dataset.csv` | the training dataset: identifier, group, the raw predictors the recipe uses, target |
| `data/transformed.csv` | the same rows through `recipe.json` |
| `metadata/columns.csv` | one row per exported column in output order: role, source, type, unit, transformation, meaning |
| `metadata/features.csv` | the feature dictionary: every column the recipe produces, and its parameter |
| `metadata/decisions.csv` | every parameter included, excluded or changed, with the reason and the stage |
| `metadata/cohort.csv` | every batch in the PVF: in or out, and the first filter that ruled it out |
| `metadata/clusters.csv` | correlated groups: membership, representatives, actions, singletons and skipped parameters |
| `metadata/missingness.csv` | parameters missing together, and their best explainer |
| `metadata/recipe.json` | the fitted preprocessing state, readable by `pvf apply` and `Preprocessor.from_state` |
| `report/report.html` | the Streamlit report |
| `report/report_payload.json` | the same numbers, usable without a browser |
| `inputs/ptf_manifest.json` | the manifest of the `pvf ptf` run that checked the PTF |
| `inputs/pvf_manifest.json` | the manifest of the `pvf build` run that wrote the PVF |
| `logs/run.log` | the events of this run and this stage only |

`pvf rerun <run folder>` reads only that folder — its `config/task.yaml` points
at `inputs/` — writes a new run beside it, and says whether `dataset.csv` came
out byte for byte the same.

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

A task copies the two stage manifests into its `inputs/` and checks that the PVF
manifest describes the PVF it read — by the file's bytes for a local file, by the
table's content for one read from SharePoint — and that the PTF has not changed
since that build, and warns when either does not hold. A missing manifest is a
warning too, recorded as `not found`.

## Parameters and features

A **parameter** is something a source recorded. It may be cleaned, renamed,
rescaled or harmonised on the way in, but nothing computes it. Parameters are the
business of `io.py`, `clean.py` and `enrich.py`.

A **feature** is arithmetic over one or more parameters — an MOI, a slope, a
ratio, a profile class. Every feature the build computes lives in one ordered
registry in `src/pvf/features.py`, one entry each:

```python
Feature(
    "<f_moi_per_cd3>",
    (TITER, VOL_A, VOL_B, "<f_pt_cd3_cells>"),
    lambda d: _first(d, *TITER) * safe_div(_pooled_vector_volume(d), d["<f_pt_cd3_cells>"]),
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

`uv run python -m pvf.demo <folder>` writes made-up sources, a `config.yaml` and
two task workspaces (from `pvf.demo`). The numbers are invented; the *shapes* are
not: every path the stages can take is represented by something in that data — a
204-parameter PTF with an `Available At` column and a 70 × 221 PVF. Only
`demo/config.yaml`, `demo/corrections.yaml` and the two `task.yaml` files are in
git; the data and the runs are generated.

| | `tasks/site_a_car_expression/` | `tasks/site_b_release/` |
|---|---|---|
| Question | what goes with CAR+ expression at SiteA | what a released SiteB batch looks like |
| Cohort | SiteA, commercial, finished, no withdrawals | SiteB, finished |
| Target | `<fp_car_pct>`, numeric | `Disposition`, binary, positive class `Released` |
| Purpose | predictive, cutoff `Harvest` | exploratory |
| Columns | one excluded by hand | one excluded by hand, `Available At: [Post-release]` excluded by category |
| Clustering | `report_only` | `linear`, threshold 0.8 |
| Result | 21 batches, 89 raw predictors (113 transformed); 50 parameters left out as later than harvest | 27 batches, 168 transformed columns, 30 groups residualised |

## Code layout

```
src/pvf/
  cli.py               the commands, and the one place the stages are run from
  config.py            config.yaml: discovery, validation, where each input is read from
  scaffold.py          pvf init; templates/ holds the task.yaml it writes
  taskconfig.py        the task YAML, and the validation it has to pass
  ptf.py               stage 1: source parameters the PTF does not list
  io.py                reading every source, local or SharePoint
  clean.py             type coercion and harmonisation
  corrections.py       the corrections table
  enrich.py            parameters joined from other systems
  features.py          the feature registry, and derived-column lineage
  merge.py             stacking the sites, and the checks over the result
  dataset.py           stage 3: cohort, roles, column selection, availability, fit/transform, assembly
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
demo/                  the demo's config and task files; `python -m pvf.demo demo` adds the data
config.yaml            the config: set the paths to your data folder here
.env.example           what goes in .env, next to config.yaml
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
