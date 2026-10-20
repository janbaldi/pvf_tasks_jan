# ml — training datasets from CAR-T manufacturing data

`ml` turns the batch records of both manufacturing sites into one checked
**batch table**, and cuts that table into a documented, reproducible **training
dataset** for each question you want a model to answer. Every step leaves a
report and a record of what it read, what it decided and why.

It does not fit models or split data. It hands your modelling tool a clean
table, and the means to refit the preprocessing inside every fold.

## The workflow in one picture

```
 once a month, when new batches are in                      whenever you have a question
 ─────────────────────────────────────                      ────────────────────────────
 ml check-catalog   is the catalog still complete?
 ml build           every batch, both sites, in one table   ml task new <folder>
        │           + build report, missingness report      ml task run
        ▼                                                          │
 ml screening add   leave unusable parameters out of        reads the batch table,
 ml screening status   every task, with a recorded reason   applies the screening,
                                                            writes a run folder
```

| Word | What it is |
|---|---|
| **catalog** | `catalog.xlsx` on SharePoint: every parameter the batch table may hold, with its type, stage and availability |
| **batch table** | `batch_data.xlsx` on SharePoint: one row per batch, both sites, cleaned, with calculated features |
| **missingness report** | `missingness_report.html`: how every parameter goes missing, per site, batch by batch |
| **correlation report** | `correlation_report.html`: which numerical parameters rank the batches alike, per site, and which one could represent each group |
| **task catalog** | `task_catalog.xlsx`, next to the batch table: the catalog parameters a task can still use after screening |
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
uv run ml build                    # the batch table and its reports
uv run ml screening status         # what is screened out today
uv run ml task run --task tasks/expression
```

Then open, in a browser:

- `sharepoint/DEMO_DRIVE/Data/batch_data/build_report.html`: where every column came from and what cleaning changed.
- `sharepoint/DEMO_DRIVE/Data/batch_data/missingness_report.html`: how each parameter goes missing.
- `sharepoint/DEMO_DRIVE/Data/batch_data/correlation_report.html`: which numerical parameters rank the batches alike.
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

`CALCULATED_FEATURES.md` lists every calculated column with its formula.

## License

This repository has no license file yet, so no license has been granted. Add one
before sharing it outside the team.
