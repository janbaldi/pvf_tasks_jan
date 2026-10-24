# ml — training datasets from CAR-T manufacturing data

`ml` turns the batch records of one manufacturing site at a time into a checked
**batch table**, and cuts that table into a documented, reproducible **training
dataset** for each question you want a model to answer. Which site is one
setting, `site:` in `config.yaml`; each site's outputs stay in a folder of its
own. Every step leaves a
report and a record of what it read, what it decided and why.

It does not fit models or split data. It hands your modelling tool a clean
table, and the means to refit the preprocessing inside every fold.

## The workflow in one picture

```
 once a month, when new batches are in                      whenever you have a question
 ─────────────────────────────────────                      ────────────────────────────
 ml check-catalog   is the catalog still complete?
 ml build           every batch of the site, in one table   ml task new <folder>
        │           + build report, missingness,            ml task run
        │             correlation and shifts reports               │
        │                                                   reads the batch table,
        ▼                                                   applies the screening,
 ml screening add   leave unusable or redundant parameters  writes a run folder
 ml screening status   out of every task, with a recorded
                       reason (and representative)
```

The full picture is [`docs/workflow.html`](docs/workflow.html): an interactive
map of every command, input and output, from the catalog to a task's run
folder. Download it and open it in a browser; click any box for what it reads,
writes and which settings change it, or pick a route (monthly update, screening
loop, new task, reproduce a run) to highlight it.

| Word | What it is |
|---|---|
| **catalog** | `catalog.xlsx` on SharePoint: every parameter the batch table may hold, with its type, stage and availability |
| **site** | the manufacturing site everything is about: `site:` in `config.yaml` (`a` or `b`, as the vocabulary names them) |
| **batch table** | `batch_data.xlsx` on SharePoint, in the site's folder: one row per batch of that site, cleaned, with calculated features |
| **missingness report** | `missingness_report.html`: how every parameter goes missing at the site, batch by batch |
| **correlation report** | `correlation_report.html`: which numerical parameters rank the site's batches alike, and which one could represent each group |
| **shifts report** | `shifts_report.html`: which parameters run in groups over time, as a lot does, and which change on the same days, per manufacturing site; `ml task new --exclude-shifting` leaves them out of a task |
| **task catalog** | `task_catalog.xlsx`, in a task folder: the catalog parameters that task can use as input features, written by `ml task new` (again: `ml task catalog`) |
| **screening register** | `parameter_screening.xlsx` on SharePoint: the log of parameters you decided to leave out of every task, and why |
| **task** | one question, written as a `task.yaml` in a folder of its own |
| **run** | one execution of a task: a folder holding the dataset, every decision behind it, and copies of its inputs |

## Install

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/janbaldi/pvf_tasks_jan.git
cd pvf_tasks_jan
uv sync                     # the exact versions in uv.lock
uv run ml --help
```

All data is read from and written to SharePoint through the internal
`io_sharepoint` package. To set up the real data, see
[Setting up](docs/manual/01-setup.md).

## Try it in ten minutes

The demo has made-up data in the same shape as the real sources, and a folder
that stands in for SharePoint. Every command below runs as written.

```bash
uv run python -m ml.demo demo      # write the demo into demo/
cd demo
uv run ml validate                 # is the setup sound? reads, writes nothing
uv run ml check-catalog            # what do the sources record that the catalog lacks?
uv run ml build                    # site A's batch table and its reports
uv run ml screening status         # what is screened out at site A today
uv run ml task run --task tasks/expression
```

The demo's `config.yaml` is about site A (Northfield); `config_site_b.yaml` is the
same with `site: b`. The `release` task is about site B and names that config, so
`uv run ml update --task tasks/release` builds site B and runs it.

Then open, in a browser:

- `sharepoint/DEMO_DRIVE/Data/batch_data/Northfield/build_report.html`: where every column came from and what cleaning changed.
- `sharepoint/DEMO_DRIVE/Data/batch_data/Northfield/missingness_report.html`: how each parameter goes missing.
- `sharepoint/DEMO_DRIVE/Data/batch_data/Northfield/correlation_report.html`: which numerical parameters rank the batches alike.
- `sharepoint/DEMO_DRIVE/Data/batch_data/Northfield/shifts_report.html`: which parameters run in groups over time, as the lots do, and which change on the same days.
- `tasks/expression/runs/<run id>/report/report.html`: the training dataset and every decision behind it.

## The manual

| Chapter | Read it when |
|---|---|
| [1. Setting up](docs/manual/01-setup.md) | You connect `ml` to the real SharePoint data: `.env`, `config.yaml`, the vocabulary, corrections. |
| [2. The monthly update](docs/manual/02-monthly-update.md) | New batches are in: check the catalog, build, read the reports. A checklist. |
| [3. Screening parameters](docs/manual/03-screening.md) | The missingness or correlation report shows parameters you don't want in any task. |
| [4. Tasks](docs/manual/04-tasks.md) | You have a question: create a task, narrow it, fit and explain a model, reproduce a run. |
| [5. Reference](docs/manual/05-reference.md) | You need the exact meaning of a command, a setting or an output file. |
| [6. Development](docs/manual/06-development.md) | You change the code: tests, adding a calculated feature, the code layout. |

`docs/workflow.html` maps the whole workflow, and `CALCULATED_FEATURES.md` lists
every calculated column with its formula.

## License

This repository has no license file yet, so no license has been granted. Add one
before sharing it outside the team.
