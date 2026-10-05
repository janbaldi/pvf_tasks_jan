# CAR-T manufacturing data

Three stages over one dataset, in one package.

| Stage | What it answers | Writes |
|---|---|---|
| `pvf ptf` | What do the source systems record that the PTF has never been told about? | `outputs/reports/ptf_report.html`, `data/processed/ptf_manifest.json`, optionally `new_parameters.csv` |
| `pvf build` | What happened to every batch, at both sites, in one table? | `data/processed/PVF.xlsx`, `data/processed/pvf_manifest.json`, `outputs/reports/build_report.html` |
| `pvf tasks --task <file>.yaml` | What does one cohort and one target look like as a dataset someone else can use? | one folder per run under `outputs/tasks/<task name>/` |

They are in that order because each depends on the one before. The PTF is the
schema: it names every parameter the PVF may hold. The build fills that schema in
from the site files and everything joined to them. A task narrows what the build
produced to one question.

```bash
uv sync
uv run pvf ptf                                              # stage 1
uv run pvf build                                            # stage 2
uv run pvf tasks --task config/task_ghent_car_expression.yaml
uv run pvf all --task config/task_ghent_car_expression.yaml  # the three in order
uv run pytest                                                # one full run over made-up data
```

`--config config/config.yaml` drives the first two stages; a task is a file of
its own. **Every relative path in a config or a task file is relative to that
file**, not to the working directory, so a run started from anywhere reads the
same inputs. Absolute paths are used as they are.

Each stage prints its steps in aligned columns, in colour when the output is
a terminal (set `NO_COLOR=1` to turn colour off, `FORCE_COLOR=1` to keep it in a
pipe). Light blue marks the structure; green, yellow and red mark OK, warnings
and errors. There are no icons: rules, aligned columns and colour carry it.
When a step computes for more than half a second without printing, a small
progress bar (`[  -=#   ] working 4s`) runs under it until the next line. It
only runs on a terminal, so a redirected log never contains it. Each stage ends
with a block of results and a summary line, coloured by how it went, that
repeats every warning once, with a count where the same warning came up more
than once. A console whose code page has no line-drawing characters (a Windows
console in cp1252, say) rules with `=` and `-` instead, and any other character
it cannot print becomes `?` rather than an error. `pvf ptf`
lists, source by source, every column the PTF does not have. `pvf build` lists
the PTF parameters that did not make it into the PVF. Both print the path of
each file they wrote.

No access to the real files? There is a complete set of stand-ins:

```bash
uv run python tests/dummy_data.py demo
uv run pvf --config demo/config.yaml all --task demo/task_car_expression.yaml
uv run pvf --config demo/config.yaml tasks --task demo/task_release_outcome.yaml
```

## A task is a YAML file

One task per file: a cohort, a target, the role of every column, what to do
about parameters that move together, and where the result goes. Defining a
second task means writing a second file — there is no Python to edit.
`config/task_ghent_car_expression.yaml` is the annotated example; `demo/` has
two that differ in every one of those things.

```yaml
task:    {name: ghent_car_expression, purpose: predictive, seed: 20260917}
inputs:  {pvf: ../data/processed/PVF.xlsx, ptf: ../data/raw/PTF.xlsx}
output:  {root: ../outputs/tasks}
cohort:
  sites: [Ghent]
  filters:
    - {column: Type, op: equals, value: Commercial, reason: manufactured commercially}
    - {column: Non-Conformance Type, op: not_in, value: [termination, withdrawal],
       normalise: true, missing: include, reason: not terminated or withdrawn}
target:  {column: FP Flow CAR+ (%), type: numeric}
columns: {id: 'Patient Lot/Batch #', group: Vector Lot}
availability: {cutoff: harvest (day 10), reviewed: true, unavailable: [...]}
clustering: {action: report_only, method: spearman, threshold: 0.7, min_overlap: 10}
split:   {strategy: grouped, validation_fraction: 0.25}
```

Everything is validated before the run touches the output directory, and every
complaint names the field and what to do about it: unknown settings, wrong
types, thresholds outside their range, a column that is two roles at once, a
predictive task nobody reviewed, a filter on a column the PVF does not have.

- **Filters** are a fixed set of operators — `equals`, `not_equals`, `in`,
  `not_in`, `min`, `max`, `is_null`, `not_null` — applied in the order written.
  No expressions and no Python come out of YAML. `missing:` says what happens to
  a row whose value is absent, and `normalise: true` flattens case, spacing and
  the known aliases (`withdrawn` → `withdrawal`) for the comparison while the
  PVF keeps what the source actually said. A filter whose column is missing
  stops the run rather than quietly widening the cohort.
- **Quote a column name with a `#`, `:` or `%` in it.** Unquoted,
  `Patient Lot/Batch #` is read as `Patient Lot/Batch` followed by a comment.
- The old `tasks:` block in `config/config.yaml` still works — `pvf tasks` with
  no `--task` translates it, cohort rules included — but it cannot say most of
  what a task file can.

## Roles, leakage and what the dataset is

A task declares which column is the target, which is the identifier and which is
the grouping metadata. Those are authoritative:

- The target, the identifier and the grouping column are removed from the
  predictor candidates **before** anything is encoded, de-duplicated or
  clustered. They are carried into the output as metadata, once each.
- Anything the feature registry computed **from the target** is removed with
  them, walked through `Feature.requires`: `VCN/cell` and
  `Flow accuracy (effective)` are both derived from `FP Flow CAR+ (%)`, and
  `decisions.csv` names the chain that got each one excluded. Derived
  certificate columns are covered too, through `features.coa_lineage()`.
- A parameter that only exists after the prediction point is excluded where the
  task declares it. Nothing is inferred from a column's spelling, so a
  `purpose: predictive` task must either give `columns.include` or state
  `availability.reviewed: true`.
- `LVCoA_Impurity_Index_zsum` is standardised within its own site's batches, so
  it is exploratory only and a predictive task drops it.

**What is learned is learned on training rows.** Category vocabularies, target
means, near-duplicate selection, clusters and residual regressions are fitted on
one population and applied unchanged to any other:

| The task says | `dataset.csv` is | Fitted on |
|---|---|---|
| `purpose: exploratory` | the transformed table, labelled exploratory | the whole cohort |
| `purpose: predictive`, no `split` | the selected raw predictors | the whole cohort, for description only |
| `purpose: predictive` with a `split` | the selected raw predictors, with a `split` column | the training rows alone |

A table preprocessed over every row is not a table to cross-validate on
afterwards, and the report says so where that is what you have. `recipe.json`
holds the fitted decisions — vocabularies, priors, category means, retained
parameters, cluster membership, regressors, coefficients — as plain JSON, so
applying them to new batches needs this module rather than this process.

Target encoding is cross-fitted: each fold's prior *and* its category means come
from that fold's training rows only, folds respect the grouping column or the
chronological order where the task asks for them, and a category the training
rows never held gets the training prior. Where a category turns out to be
confounded with the grouping column, the report says how many rows fell back to
the prior instead of pretending the column carries something.

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
builds the tree and it is cut at the threshold. Two things this does not do: it
does not treat a pair with too few shared batches as uncorrelated — below
`min_overlap` the relationship is **unknown**, drawn blank in the heatmap and
never merged into a group — and it does not claim that residualising preserves
everything or makes the columns independent. Each regression is fitted on the
rows where all its inputs are present, so columns with different gaps are only
orthogonal there, and least-squares orthogonality is neither zero rank
correlation nor independence.

The representative is chosen without the target: candidates must reach the
group's median coverage, and among those the highest mean pairwise R² wins, with
coverage, an explicit `clustering.prefer` and the name breaking ties. Members are
residualised most-complete-first, so a sparse parameter never becomes the
regressor a complete one is fitted on; where there is too little data, the
parameter is kept as it was recorded and the fallback is written down rather
than a centred sparse series being called a residual.

Near-duplicates are a deterministic sweep rather than a list of pairs: candidates
are considered most-complete first, each is kept or dropped against something
already kept, so a dropped parameter always names a representative that is still
in the dataset and a chain never drops C for resembling a B that itself went.

## What a task run leaves behind

Each run writes a fresh folder, `outputs/tasks/<task name>/<run id>`. Nothing is
ever overwritten: two runs sit side by side. The run writes into a staging
folder marked incomplete and promotes it only once the required artefacts are
there and the counts inside them agree; a run that fails keeps its folder marked
`.failed`, with its log and the reason in the manifest.

| File | What it is |
|---|---|
| `manifest.json` | status, timestamps, provenance, environment, counts, warnings, artefact hashes |
| `config/task.yaml` | the effective settings, defaults included |
| `data/dataset.csv` | the task's table, with its role stated in the manifest and the report |
| `data/raw_features.csv` | the untransformed predictors, when `dataset.csv` is the transformed view |
| `data/splits.csv` | the row assignments, when a split was asked for |
| `metadata/columns.csv` | one row per exported column in output order: role, source, type, unit, transformation, meaning |
| `metadata/decisions.csv` | every parameter included, excluded or changed, with the reason and the stage |
| `metadata/cohort.csv` | every batch in the PVF: in or out, and the first filter that ruled it out |
| `metadata/clusters.csv` | correlated groups: membership, representatives, actions, singletons and skipped parameters |
| `metadata/recipe.json` | the fitted preprocessing state, and what it was fitted on |
| `report/report.html` | the Streamlit report |
| `report/report_payload.json` | the same numbers, usable without a browser |
| `provenance/ptf_manifest.json` | the manifest of the `pvf ptf` run that checked the PTF |
| `provenance/pvf_manifest.json` | the manifest of the `pvf build` run that wrote the PVF |
| `logs/run.log` | the events of this run and this stage only |

The two stage manifests are copied from beside the PVF, where `pvf ptf` and
`pvf build` write them. They record what each stage read (from where, with a
digest), what it found, what it wrote and every warning. The PVF manifest also
carries the PVF's own digest. If that digest is not the digest of the PVF the
task read, the task warns that the manifest belongs to another build. A missing
manifest is a warning too, and the run manifest says `not found` rather than
leaving the entry out.

`--report-only` writes the report and its metadata to a fresh folder of its own.
It still runs the whole analysis — there is no other way to have something to
report — but it writes no dataset, uploads nothing, and its manifest says the
package is a report rather than a dataset.

## Parameters and features

The package draws a line between the two, and so do the reports.

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

To add one, append an entry. Nothing else changes. Entries are computed in list
order, so a feature may require a column an earlier entry produced. The
`requires` tuple is not only documentation: it is what a task walks to find the
descendants of its target.

Three rules the registry follows, each visible in the build's report:

- **A feature is created only if the PTF lists its name.** The PTF is the schema
  of the PVF; a column it does not know about has nowhere to go.
- **A feature whose inputs are not all present is not created**, and the report
  names the inputs it wanted.
- **A feature never overwrites a column a source supplies.**

The certificate loader in `io.py` derives columns of its own while parsing the
workbook — below-LOQ flags, delivered dose, donor summaries. Those are not PTF
parameters, deliberately, but each is declared in `features.coa_lineage()` with
its inputs, and the loader says so if it ever produces one that is not.

## Sources, cleaning and uploads

Which sources a run reads is a setting, not a consequence of what happens to be
installed:

```yaml
sources: {location: local, databricks: false, fallback_to_local: false}
upload:  {enabled: false, target: ""}
```

With `location: local` an installed `io_sharepoint` changes nothing. With
`location: sharepoint` a missing one stops the run unless `fallback_to_local`
allows otherwise, and a fallback that is taken is recorded. Uploading is a
separate switch again, and only ever happens on a `--mode PRD` run: reading from
the share does not imply writing to it, and a local test cannot upload by
accident.

SharePoint serves the two PHFs, the parameter mapping, the Re-MFG supplement
and the investigations workbook; their locations are `io.SHAREPOINT`. The PTF,
the LV CoA workbook, the site maps and the raw-materials CSV are always read
from the local paths in config. The CSV is local because `io_sharepoint` reads
Excel workbooks only. Provenance follows the read: a source that came from
SharePoint is recorded by its SharePoint location and a digest of the table that
came back, and a local file by the digest of its bytes. Each stage records only
the sources it actually read.

Two cleaning decisions are settings rather than facts:

- `cleaning.censored` — what a value written `<0.5` becomes: `limit`, `half` or
  `missing`. Stripping the operator changes what the number means, so the policy
  is named and the count of values it touched is in the build report.
- `cleaning.yes_no_defaults` — a blank in a yes/no column is *unknown*. It is
  filled only for the columns this names, in each site's own spelling, with the
  count.

`cleaning.raritan_type_commercial` keeps the established correction that fills
the Raritan `Type` column, with the number of rows it changed; turn it off for a
task where that distinction matters. `cleaning.duration_unit` says what a bare
number in a duration cell means — `HH:MM` and `HH:MM:SS` are read as clock times
whatever it says — and a ratio written `a:b` keeps its right-hand side, because
these columns are recorded normalised to 1.

## Reports

Every stage writes a Streamlit page that needs no Streamlit server: the file
carries its own Python runtime through stlite. Open it; there is nothing to run.
It fetches that runtime from a CDN, **so a fresh report needs network access
once** — it is not an offline, self-contained file — and it spends 10 to 30
seconds starting Python before the first figure appears. The page says so while
it boots, and says what to do if the runtime never arrives.

Every number was computed before the file was written. Filtering, sorting and
picking a cluster on the page change what is on screen, never what was computed.
Heavy content waits to be asked for: a cluster's heatmaps are behind a picker and
one cluster is drawn at a time, because Streamlit runs what is inside a closed
expander whether or not anyone opens it.

The five report modules depend in one direction only: `blocks` defines what a
unit of content is, `plots` builds figures out of blocks, `report` assembles
sections, `streamlit_app` renders them in the browser, `stlite` wraps the page
into one file. `streamlit_app.py` runs under Pyodide and may import only the
standard library, `streamlit` and `plotly`.

A task report is seven sections: overview (question, target, cohort, what the
table is for, what needs attention), cohort (the full funnel including the
missing-target step), data quality, parameters and encodings (the searchable
dictionary and every decision), correlated groups, the dataset itself, and
provenance with the run log. The correlated groups open with a summary table of
every cluster and a table of every member of every cluster; a dropdown then
picks the cluster whose numbers, members, heatmap and fits are drawn. The dictionary, decisions, cohort exclusions,
cluster membership and the dataset are downloadable from the page, and what
downloads is what the package holds.

## Seeing it work: the demo

`tests/dummy_data.py` writes a full set of made-up sources, a config and two task
files. The numbers are invented. The *shapes* are not: every path the stages can
take is represented by something in that data, so the reports are a tour of what
the package does.

The committed demo in `demo/` is one such run over 40 Ghent and 30 Raritan
batches: a 204-parameter PTF, a 70 × 221 PVF, and two tasks over it.

| | `task_car_expression.yaml` | `task_release_outcome.yaml` |
|---|---|---|
| Question | what goes with CAR+ expression at Ghent | what a released Raritan batch looks like |
| Cohort | Ghent, commercial, finished, no withdrawals | Raritan, finished |
| Target | `FP Flow CAR+ (%)`, numeric | `Disposition`, binary, positive class `Released` |
| Purpose | predictive, grouped 30% validation split | exploratory |
| Clustering | `report_only` | `linear`, threshold 0.8 |
| Result | 21 × 127 raw modelling inputs, 29 groups described | 27 × 186 transformed, 30 groups residualised |
| Written to | `demo/tasks/ghent_car_expression/<run>/` | `demo/explorations/raritan_release/<run>/` |

`demo/run.log` is the console output of both.

| What the dummy data does | Where it shows up |
|---|---|
| The two sites' comment columns, and the investigations team's own working columns | **PTF report** — parameters new to the PTF, with the source that records each |
| Raritan's own names (`Batch Number`, `Weight (kg)`, …) are in no PTF | **PTF report** — absent from the new list, because they are compared through the name mapping |
| Two Raritan columns claim the name `Country` | **Build** — the first keeps it, the second is `Country_1`, each with its own values |
| Clinical sites written as `UZ Leuven\xa0`, ` AZ Sint-Jan` and `` | **Build** — cells stripped, blanks turned into missing values |
| A clump count of `45691`, a batch at clinical site `107306` | **Build** — integrity corrections, with the reason and the row count for each |
| Endotoxin as `<0.5 EU/mL`; a timestamp reading `not recorded`; `Recovery (%)` holding `not measured` and `105` | **Build** — the censoring policy and what it touched, parse failures, out-of-range percentages |
| Vector lot `LV-909` has no certificate; impurities as `<0,5`, titers as `3,40E+07` | **Build** — certificate coverage, European-locale text read as numbers, a below-LOQ flag |
| No site records `Post Thaw FLOW CD8+ (%)`; Raritan does not record `Harvest Glucose (g/L)` | **Build** — features blocked, one of them at a single site |
| `VCN/cell` and `Flow accuracy (effective)` are computed from the target | **Task** — excluded, with the chain, in `decisions.csv` |
| `Disposition`, `OOS Type` and release testing happen after harvest | **Task** — excluded as unavailable at the declared cutoff |
| Non-conformances spelled `withdrawn`, `Withdrawal` and `Termination` | **Task** — all excluded by one normalised filter |
| `Clinical Site` (4 categories), `Shift Team` (6), `Incubator ID` (10), `Patient ID` (one per batch) | **Task** — one-hot, target encoding, hashing, and left out as an identifier |
| `Clump Severity`, whose PTF value type is `[None, Low, Medium, High]`, plus a stray `Severe` | **Task** — ranked 1 to 4, with the stray left missing |
| `Day 3 Total Viable Cells Available` is always 5% above the VSVg count | **Task** — collapsed as a near-duplicate, naming the parameter that survived |
| Cell counts follow one latent batch size; lactate and glucose one metabolic rate | **Task** — correlated groups found, with the signed heatmap and the overlap counts |

## Layout

```
config/config.yaml                  the PTF and build stages
config/task_*.yaml                  one task each
src/pvf/
  cli.py               the stages, and the one place they are run from
  taskconfig.py        the task YAML, and the validation it has to pass
  ptf.py               stage 1: source parameters the PTF does not list
  io.py                reading every source
  clean.py             type coercion and harmonisation
  enrich.py            parameters joined from other systems
  features.py          the feature registry, and derived-column lineage
  merge.py             stacking the sites, and the checks over the result
  dataset.py           stage 3: cohort, roles, fit/transform, assembly
  encode.py            one fit/apply pair per encoding
  decorrelate.py       correlated groups, representatives and residuals
  package.py           the run folder: staging, promotion, manifest
  blocks.py plots.py report.py streamlit_app.py stlite.py
  logger.py            the run log the reports show, scoped per run
  provenance.py        commit, input digests, environment, config digest
data/raw/              inputs, read-only
data/processed/        PVF.xlsx, ptf_manifest.json, pvf_manifest.json
outputs/reports/       ptf_report.html, build_report.html
outputs/tasks/         one folder per task run
tests/dummy_data.py    made-up sources, a config and two task files
tests/                 one run of every stage over them, plus focused regressions
demo/                  a generated run: sources, outputs, two task packages, console log
```

## Environment

Python 3.12 with uv. `io_sharepoint` and `databricks_query_tool` are internal
packages, available where the pipeline runs in production; whether they are used
is `sources.location` and `sources.databricks`, not whether they are installed.
Git metadata is recorded where there is a checkout and reported as unknown where
there is not — no run depends on it. Streamlit is deliberately not a dependency:
the reports run it in the browser.

Environment variables, read from `.env`: `MODE` (`PRD` allows the SharePoint
upload, if `upload.enabled` is also set) and `DATA_LINK` (its target, when
`upload.target` is empty).
