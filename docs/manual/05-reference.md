# 5. Reference

## Commands

Run as `uv run ml [--config FILE] [--debug] [--quiet] COMMAND …`.

| Command | What it does |
|---|---|
| `ml validate [--task T]` | Check `config.yaml`, the vocabulary, the corrections, the screening register and the given tasks. Writes nothing. Inside a task folder, also checks that task. |
| `ml check-catalog` | Compare every source's columns with the catalog. Writes `catalog_report.html`, `catalog_manifest.json` and `new_parameters.csv` into the catalog's folder. |
| `ml build` | Build the batch table. Writes `batch_data.xlsx`, `batch_data.parquet`, `build_report.html`, `missingness_report.html`, `missingness.xlsx` and `batch_table_manifest.json` into the batch table's folder. |
| `ml screening add PROPOSAL [--author NAME] [--new]` | Check a proposal (written by the missingness report, or by hand) against the latest build and add its decisions to the screening register. `--author` fills an empty `Author` column. `--new` starts the register when there is none. |
| `ml screening status` | List what the register excludes today, and the decisions the latest build gives a reason to look at again. |
| `ml task new FOLDER [--target COL] [--question TEXT] [--force]` | Create a task folder with an annotated `task.yaml`. |
| `ml task run [--task T]` | Run tasks; without `--task`, the task in the current folder. |
| `ml task rerun RUN_FOLDER` | Redo a run from its own folder, and say whether `dataset.csv` came out identical. |
| `ml task encode --run RUN_FOLDER [--out FILE]` | Encode the batch table's batches with a run's fitted encoding. |
| `ml update [--task T]` | `check-catalog`, `build`, then the given tasks. |
| `ml check-leaks [FOLDER]` | Search the repository's tracked files for any name the private vocabulary holds. Run it before pushing. |

`--task` takes a task folder, a `task.yaml`, or a folder of task folders, and can
be repeated. `--debug` shows tracebacks instead of one-line errors. `--quiet`
keeps only warnings, errors, what was read and written, and the summary.
Output redirected to a file has no colour codes.

The config is found in this order: `--config`, the `config:` line of the task,
`$ML_CONFIG`, then the nearest `config.yaml` in the current folder or above.

## Who reads and writes what

| Stage | Reads | Writes, into |
|---|---|---|
| `ml check-catalog` | vocabulary, catalog, both sites' batch records, parameter mapping, investigations | the catalog's folder |
| `ml build` | vocabulary and corrections, catalog, every source, site maps, the catalog manifest, the screening register (for the report) | the batch table's folder |
| `ml screening add` | vocabulary, screening register, the latest `missingness.xlsx` | the screening register |
| `ml task run` | vocabulary, batch table, catalog, both manifests, screening register | the run folder |

Every command lists what it will read and write before it starts, and where each
file went when it is done.

## config.yaml

The shipped `config.yaml` is annotated. Its shape:

```yaml
seed: 20260917
sources:    {databricks: false}
sharepoint: {vocabulary: {path: …, drive_id: …}, catalog: {…}, batch_table: {…},
             screening: {…}, phf_site_a: {…}, phf_site_b: {…}, …,
             site_maps: {folder: …, files: […], drive_id: …}}
catalog:    {write_new_parameters: true}
build:      {write_parquet: true, date_column: date_start,
             missingness_cohort: {column: batch_type, value: type_commercial, all_batches_at: [b],
                                  analyse_only: {column: manufacturing_completed, value: completed_yes}}}
cleaning:   {disable: [], censored: limit, censored_flags: false,
             duration_unit: minutes, yes_no_defaults: {pre_activation_clumps_yn: 'No'}}
mapping:    {exclude: [excluded_mapping_row]}
features:   {co2_target: 5.0, temp_target: 37.0, …}
lv_coa:     {ph_nominal: 7.2, osmo_nominal: 300.0}
```

| Setting | Meaning |
|---|---|
| `sharepoint.<input>` | Where each input is: `path`, `drive_id`, optional `sheet`. Values come from `.env` as `${NAME}`. `screening` is optional: without it, no parameter is screened out. |
| `build.date_column` | The vocabulary key of the date that orders batches in the missingness report. |
| `build.missingness_cohort` | Which batches the missingness report describes: those whose `column` holds `value` (vocabulary keys, case ignored), and every batch at the sites in `all_batches_at`. Leave it out to describe every batch. |
| `build.missingness_cohort.analyse_only` | Of those, the batches the statistics and groups are computed over: `column` holds `value`. The others are drawn in the plot, their missing cells in green. A site without the column has all its batches analysed. |
| `cleaning.censored` | What `<0.5` becomes: `limit` (0.5), `half` (0.25) or `missing`. Applies to every source, the vector certificates included. |
| `cleaning.censored_flags` | Also add a `<column> censored` 0/1 column. |
| `cleaning.disable` | Names of corrections to switch off. |
| `cleaning.yes_no_defaults` | Yes/no columns whose blanks may be filled, and with what. |
| `mapping.exclude` | Rows of the site mapping known to be wrong. |
| `features.*` | Setpoints and thresholds of the calculated features. Some features need a process fact (`moi_target`, `medium_glucose`, `medium_lactate`, `impurity_spec_limits`) and are blocked until it is set. `availability_column` is the catalog column the build checks feature availability against. See `CALCULATED_FEATURES.md`. |
| `lv_coa.ph_nominal`, `osmo_nominal` | Spec midpoints for the certificate's deviation features. |

Unknown keys, wrong types and incomplete SharePoint entries are refused before
anything runs, each naming its field.

## task.yaml

Every key is optional except `target.column`. Quote a column name that holds
`#`, `:` or `%` outside `{…}` or `[…]`.

| Key | Default | Meaning |
|---|---|---|
| `config` | — | The `config.yaml` that locates the batch table and catalog. `ml task new` fills it in. |
| `task.name`, `task.question`, `task.seed` | `task`, —, `0` | Name, question in words, and the seed of the target-encoding folds. |
| `cohort.sites` | all | Sites to keep. |
| `cohort.site_column` | `Site Merged` | The column that says which site a batch is from. |
| `cohort.filters` | — | Rules in order: `{column, op, value, missing, normalise, reason}`. Operators: `equals`, `not_equals`, `in`, `not_in`, `min`, `max`, `is_null`, `not_null`. |
| `cohort.date_column`, `date_from`, `date_to` | — | Keep batches whose date is in this range (inclusive). |
| `target.column` | required | The column to predict; for `periods`, the date that places a batch in a period. |
| `target.type` | `numeric` | `numeric`, `binary` or `periods`. |
| `target.positive_class`, `negative_class` | —, the other value | For `binary`: the values that become 1 and 0. |
| `target.periods` | — | For `periods`: two `{name, start, stop}` windows, `YYYY-MM-DD`, not overlapping. |
| `roles.id` | — | Identifies a batch; must be unique in the cohort. |
| `roles.group` | — | Grouping column (e.g. vector lot) for grouped folds. |
| `roles.patient` | — | Joins a patient's batches into one group. |
| `predictors.include` | all | Use exactly these predictors. A screened parameter here stops the run. |
| `predictors.exclude` | — | `[{column, reason}]`: single columns to leave out. |
| `predictors.exclude_by_catalog`, `include_by_catalog` | — | `{catalog column: [values]}`: leave out, or keep only, parameters with these catalog values. |
| `clustering.action` | `describe` | `off`, `describe`, `keep_one`, `residualise` or `summarise`. |
| `clustering.min_stability` | `0.8` | Act on a group only if it comes back in this share of resamples. |
| `clustering.name_by` | — | A catalog column; a group whose members share its value is named after it. |
| `clustering.method`, `threshold` | `spearman`, `0.7` | Correlation method, and the \|r\| at which parameters group. |
| `clustering.min_size`, `min_overlap`, `prefer` | `2`, `10`, — | Smallest group; fewest shared batches; preferred representatives. |
| `encoding.*` | | Encoder limits, e.g. `missing_indicators: true`, `impute: median`. |
| `quality.min_non_missing` | `20` | A predictor needs at least this many values. |
| `quality.duplicate_r2_threshold` | `0.99` | Above this R², one of two parameters is a near-duplicate. |
| `quality.proxy_correlation` | `0.95` | Predictors this correlated with the target are flagged as possible leaks. |
| `missingness.enabled` | `true` | Describe which parameters go missing together in the task report. |
| `report.preview_rows` | `20` | Rows shown in the report's dataset preview. |

## How a dataset is built

For each task, `ml task run`:

1. **Selects the cohort**: sites, then each filter in order, then batches without
   a target. Every batch gets a row in `batches.csv` saying whether it's in, and
   why not.
2. **Chooses the predictors**: the catalog parameters in the batch table, minus
   the target and the roles, `predictors.exclude` and the catalog rules, the
   screening register's exclusions at the cohort's sites, features calculated
   from the target, and the site-standardised impurity index. Each exclusion is
   a row in `parameters.csv`, with its stage.
3. **Fits the preprocessing** on the cohort and records it in `encoding.json`:
   categories become 0/1, one column each, the target's mean (cross-fitted) or
   hashed buckets, by how many there are. Near-duplicates, near-constant and
   too-sparse columns are dropped. Correlated parameters are grouped, and stable
   groups get the `clustering.action`.
4. **Writes the dataset**: `dataset.csv` (id, group, raw predictors, target) and
   `dataset_encoded.csv`. One row per batch, no split.
5. **Checks** that the counts agree across every file, then gives the run folder
   its final name. A failed run keeps its folder as `….failed`, with the reason
   in its manifest.

## Correlated parameters

| `clustering.action` | Effect |
|---|---|
| `off` | No groups are looked for. |
| `describe` | Groups are described; every parameter stays. |
| `keep_one` | One member of each stable group stays; the others are dropped. |
| `residualise` | One member stays; the others become what's left after regressing them on it. |
| `summarise` | The group becomes one score, `<name> group`: the mean of its z-scored members, negatively related members flipped. |

Groups are found by complete linkage on `1 − |correlation|`, cut at `threshold`.
Pairs with fewer than `min_overlap` shared batches never group. Each group is
redrawn on 50 resamples of 80% of the batches, and only groups that come back in
`min_stability` of them are acted on. Over several sites, each site's mean is
removed first, so parameters that differ only by site don't group.

## What a run writes

| File | Contents |
|---|---|
| `data/dataset.csv` | The training dataset: id, group, raw predictors, target. |
| `data/dataset_encoded.csv` | The same rows, encoded. |
| `metadata/parameters.csv` | Every parameter: included or excluded, the reason and the stage. |
| `metadata/batches.csv` | Every batch: in or out, and the first rule that ruled it out. |
| `metadata/dataset_columns.csv` | Every dataset column: role, source, type, unit, meaning. |
| `metadata/encoded_columns.csv` | Every encoded column and the parameter it came from. |
| `metadata/clusters.csv` | Correlated groups: stability, representative, what was done. |
| `metadata/groups.json` | Explanation groups. |
| `metadata/missingness.csv` | Parameters that go missing together in this cohort. |
| `metadata/encoding.json` | The fitted preprocessing. |
| `report/report.html`, `report_payload.json` | The report, and its numbers. |
| `inputs/` | The batch table, catalog, vocabulary and screening register the run read, and the catalog and build manifests. |
| `config/task.yaml` | Every setting, defaults included, pointing at `inputs/`. |
| `logs/run.log` | The run's log. |
| `manifest.json` | Inputs with digests, outputs with checksums, counts, warnings, the git commit, and which screening register (digest and last decision) the run applied. |

The reports are self-contained pages that run Python in the browser: the first
time you open one it needs network access and 10 to 30 seconds to start.

## The screening register

`parameter_screening.xlsx`, sheet `Decisions`, one row per decision and
parameter:

| Column | Filled by | Meaning |
|---|---|---|
| `ID` | tool | `S-0001`, … one per proposal row; a group's members share it |
| `Date`, `Author` | tool, proposal | When, and the proposal's `Author` (or `--author`) |
| `Action` | you | `exclude` or `reinstate` |
| `Parameter`, `Site` | you (tool, for a group) | What, and where (a site's name or `all`) |
| `Group` | you | The report's group label, as context |
| `Reason`, `Rationale` | you | One of the fixed reasons, and why in words |
| `Missing %`, `Pattern`, `Last measured` | tool | The evidence, from the latest build |
| `Batch table` | tool | Digest of the batch table the decision was made on |
| `Review by` | you, or three months on | When to look again |

The `Check` sheet holds the digest the register was last written with. Reasons:
`never recorded`, `retired`, `introduced late`, `too sparse`, `irregular`,
`missing with group`, `data quality`, `other`.

## Troubleshooting

| Message | What to do |
|---|---|
| `No config found …` | Run from the repository, add `config:` to the task, set `$ML_CONFIG`, or pass `--config`. |
| `… ${NAME} is not set` | Add the variable to `.env`; `.env.example` lists them all. |
| `No task to run …` | `cd` into a task folder, or pass `--task`. |
| `… is not a column of the batch table` | Check the spelling; `ml validate` lists every such name. |
| `The vocabulary … leaves out N names the code needs` | Add a `Names` row for each key listed; `src/ml/names.py` says what each is. |
| `The catalog changed after the batch table was built` | Run `ml build`, then the task again. |
| `The proposal was not added: …` | Fix each listed row; nothing was added. |
| `The screening register could not be read … pass --new` | The first time: `ml screening add … --new`. Otherwise check `sharepoint.screening`. |
| `The screening register was changed outside ml screening` | Someone edited the register by hand. Find out what changed (the run folders keep copies), and restore it. |
| `The latest build's missingness.xlsx could not be read` | Run `ml build` first. |
| `predictors.include: '…' was asked for … but is excluded for another reason` | The parameter is screened or excluded otherwise; see `parameters.csv`. |
| `Windows would not rename … copying it instead` | Nothing to do: a file in the new run folder was open elsewhere. |
| `Lines that hold a name from the private vocabulary` (`ml check-leaks`) | Replace each listed name with its key (code) or an invented name (tests, docs). |

Errors you can fix exit with code 2 and a one-line message; `--debug` shows the
traceback.
