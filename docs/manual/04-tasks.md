# 4. Tasks

A **task** is one question asked of the batch table, such as *which
manufacturing parameters go with CAR+ expression at Northfield?* It lives in a
folder of its own with a `task.yaml`. Every **run** of it writes a new folder
under `runs/` with the training dataset, the reason every parameter and batch is
in or out, a report, and copies of everything it read.

This chapter walks through one task on the demo data (start with
`uv run python -m ml.demo demo && cd demo`, and build it as in the
[README](../../README.md#try-it-in-ten-minutes)). Your numbers can differ
slightly from the ones shown. The *New task* and *Reproduce a run* routes of
[the workflow map](../workflow.html) show what a task reads and writes.

## 1. Create a task

```bash
uv run ml task new tasks/car_expression --target "Product CAR expression (%)"
cd tasks/car_expression
```

The folder holds an annotated `task.yaml` that already points at the
`config.yaml` you came from, so the task finds the batch table from anywhere.
Every parameter the screening register excludes today (at any site) is already
listed under `predictors.exclude`, with its decision:

```yaml
predictors:
  exclude:
    # Screened out by the screening register (last decision S-0003), at any site.
    - {column: "Day 1 mixing cycles", reason: "screened out by S-0001: irregular (Northfield)"}
```

To see what is still usable, open `task_catalog.xlsx` next to the batch table:
the catalog minus what the register excludes, per site, with what each
representative of a correlated group stands for. A parameter excluded as
`correlated with group` names its representative in the line above, so use that
one instead. A parameter excluded in both reports lists both decisions, joined
by `; `. When a representative is excluded itself (a conflict), `task.yaml` says
so in a `# CONFLICT:` comment above the exclusions, and every run warns; resolve
it in the correlation report ([chapter 3](03-screening.md#how-it-works)).

Every run applies the register anyway; the lines make it visible in the task
itself. If a decision is later reinstated, the run warns that the task still
excludes the parameter: delete the line to use it again. A line for a site your
task doesn't use can be deleted too.

To also leave out every parameter that runs in groups over time, as a lot does
(the [shifts report](02-monthly-update.md#the-shifts-report)), add
`--exclude-shifting`:

```bash
uv run ml task new tasks/car_expression_no_lots --target "Product CAR expression (%)" --exclude-shifting
```

It lists them from the latest build's `shifts.xlsx`, at any manufacturing site,
each with why, a group of parameters that change on the same days one after the
other:

```yaml
    # Run in groups over time at a manufacturing site, as the build of 2026-10-21 14:03
    # found (shifts_report.html; `ml task new --exclude-shifting`). A lot does this,
    # and so the model would learn time. Delete a line to use the parameter again.
    - {column: "Cytokine lot", reason: "runs in groups over time: 7 groups at Northfield plant, changes with Shifting 01"}
    - {column: "Medium lot", reason: "runs in groups over time: 7 groups at Northfield plant, changes with Shifting 01"}
```

These lines are the task's own: a later build does not change them. Make the
task again to pick up a newer build's list.

Its main sections:

```yaml
cohort:      which batches are rows          (empty: every batch)
target:      what the model will predict
roles:       columns that describe a batch rather than predict anything
predictors:  which other parameters may be predictors
clustering:  what to do about parameters that move together
```

## 2. Run it as it is

Inside a task folder no `--task` is needed:

```bash
uv run ml validate
uv run ml task run
```

```
WARN  dataset  2 batches have no usable target and are not in the dataset
WARN  dataset  203 predictor columns for 68 fitted batches (p/n = 3.0): a model needs strong regularisation, …
Written — 68 batches × 152 columns
```

The run folder:

```
runs/<run id>/
  data/dataset.csv               one row per batch: id, group, raw predictors, target
  data/dataset_encoded.csv       the same rows as numbers, ready for a model
  metadata/parameters.csv        every parameter: included or excluded, and why
  metadata/batches.csv           every batch: in or out, and why
  metadata/groups.json           groups of parameters to explain a model by
  report/report.html             all of it with figures; open it in a browser
  inputs/                        the batch table, catalog, vocabulary and screening register this run read
  config/task.yaml               every setting, defaults included
```

It works, but it answers no particular question yet. Both sites and trial
batches are mixed in, and the predictors include release tests and after-wash
counts, measured after the outcome was decided.

## 3. Say what the question is

Edit `task.yaml`. Keep routine Northfield batches, say which column groups the
batches (so whole vector lots can be held out later), and leave out what is
known too late:

```yaml
cohort:
  sites: [Northfield]
  filters:
    - {column: Batch kind, op: equals, value: Routine, reason: manufactured routinely}

roles:
  id: 'Patient Lot/Batch #'
  group: Vector lot

predictors:
  exclude:                        # single columns, each with its reason
    - {column: Patient weight (kg), reason: dose per kilogram already carries it}
  exclude_by_catalog:             # every parameter with these catalog values
    Value Type: [datetime]
    Available At: [Wash, Formulation, Final product, Post-thaw, Release, Post-release]
```

`exclude_by_catalog` looks every parameter up in the catalog and leaves it out
when the named column holds one of the values. Any catalog column works.
`include_by_catalog` does the opposite.

```bash
uv run ml validate
uv run ml task run
```

```
WARN  dataset  Only 3 distinct Vector lot values: any grouped cross-validation rests on 3 groups, not on 34 batches
Written — 34 batches × 117 columns
```

`metadata/parameters.csv` now gives a reason for every exclusion, including
ones you didn't ask for:

| Parameter | Decision | Reason | Stage |
|---|---|---|---|
| Patient weight (kg) | excluded | dose per kilogram already carries it | task exclusion |
| Day 1 mixing cycles | excluded | screened out by S-0001: irregular (Northfield) | screening |
| Vector copies per cell | excluded | computed from the target: Product CAR expression (%) → Vector copies per cell | target lineage |
| D3 seeding vs target (%) | excluded | duplicates 'D3 cells per bag' (R²=1.0 over 34 shared batches) | near-duplicate |

- **screening**: the register leaves the parameter out of every task ([chapter 3](03-screening.md)).
- **target lineage**: the parameter is calculated from the target, so it would be the answer key.
- **near-duplicate**: it adds nothing its twin doesn't already say.

A typo is caught before anything runs, and named:

```
Problems that would stop a run (1)
  - car_expression: target.column: 'Product CAR expression' is not a column of the batch table
```

## 4. Tame correlated parameters

Manufacturing parameters measure the same few things several times over. A
model given all of them cannot say which one mattered. Every run finds such
groups and lists them in `metadata/clusters.csv` with their **stability**: how
often the group comes back when the batches are resampled. Only groups that come
back in at least 80% of resamples are acted on.

To replace each stable group by one score, set in `task.yaml`:

```yaml
clustering:
  action: summarise
```

`dataset.csv` still holds every raw parameter. `dataset_encoded.csv` gets one
column per stable group instead of its members (`… group`: the mean of the
z-scored members), and `parameters.csv` records each member as *changed*. The
other actions are `describe` (the default: report only), `keep_one` and
`residualise`; see [the reference](05-reference.md#correlated-parameters).

## 5. Fit a model and explain it by group

`metadata/groups.json` puts every encoded column into one explanation group: a
stable cluster, or a single parameter with all its encoded columns. Explaining by
group gives each group its full credit instead of splitting it between
correlated members.

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

Three things keep this honest:

- **`task_data(".")`** returns the cohort exactly as `ml task run` selects it, the
  target, and the vector lot of every batch, so whole lots are held out.
- **`TaskTransformer(spec, run=run)`** refits the task's encoding inside every
  fold, so nothing is learned from the held-out batches. It applies the same
  screening as the task.
- **Shuffling a whole group** measures what the group is worth on batches the
  model hasn't seen. With SHAP, sum each row's values over a group's `columns` in
  `groups.json` instead.

(On the made-up demo data the scores mean nothing.)

## 6. Find a leak through the explanation

Read the top of the importance list as a process engineer would. In the demo, a
group holding `Day 10 CAR expression (%)` comes out on top: that's CAR+
expression measured at harvest, so the target measured one step earlier.
Exclude it, with its reason, and run again:

```yaml
predictors:
  exclude:
    - {column: Patient weight (kg), reason: dose per kilogram already carries it}
    - {column: Day 10 CAR expression (%), reason: the target measured at harvest}
```

If the scores hardly move, look at the new top of the list. Calculated features
such as `D10 CAR cells` are computed from `Day 10 CAR expression (%)`, and
leaving a parameter out does not leave out what is calculated from it
(`CALCULATED_FEATURES.md` lists every input). This loop of *run, explain,
question, exclude* is how a task gets to a dataset you can trust. When an
exclusion holds for every question, not just this one, record it as a screening
decision instead ([chapter 3](03-screening.md)).

## 7. Reproduce a run, and encode new batches

A run folder holds everything it needs:

```bash
uv run ml task rerun runs/<run id>
```

```
OK  cli  dataset.csv is identical to the one in <run id>
```

The rerun reads only that folder (the batch table, catalog, vocabulary and
screening register in `inputs/`) and writes a new run next to it. Archive a run
folder and you can redo it years later.

To encode batches exactly as a run encoded its cohort, with nothing refitted:

```bash
uv run ml task encode --run runs/<run id> --out new_batches_encoded.csv
```

## 8. Other kinds of target

**Binary.** The demo task `tasks/release` predicts whether a Southbank batch was
accepted:

```yaml
target:
  column: Outcome
  type: binary
  positive_class: Accepted        # becomes 1
  negative_class: Scrapped        # becomes 0; any third value stops the run
```

**Two periods.** To find what changed between two sets of batches (before and
after a process change, say), name a date column and two date windows. The
second window becomes 1 and the first 0, and batches outside both are left out:

```yaml
target:
  column: "Day 0 incubator in"
  type: periods
  periods:
    - {name: before, start: 2026-01-01, stop: 2026-01-10}   # becomes 0
    - {name: after,  start: 2026-02-10, stop: 2026-02-20}   # becomes 1
```

Both ends of a period count, and the stop date counts for the whole day. The
periods must not overlap and each must hold at least one batch. With periods,
a predictor that matches the target perfectly may simply be what changed: read
the leak warnings as candidates for the answer, and exclude what only changed
because time passed.

To ask a second question, run `ml task new` again: every question gets its own
folder and its own history of runs.
