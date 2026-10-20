# 1. Setting up

You do this once. Afterwards the monthly routine ([chapter 2](02-monthly-update.md))
needs no setup.

## What lives where

| Where | What | Who changes it |
|---|---|---|
| This repository | The code, `config.yaml`, the task folders if you like | Developers; you, for `config.yaml` |
| `.env` next to `config.yaml` (never in git) | SharePoint credentials and the address of every file | You, once |
| SharePoint | The sources, the vocabulary, the catalog, the batch table, the reports, the screening register | The stages write their outputs; you edit the catalog and the vocabulary |
| A task's `runs/` folder | One folder per run, with copies of everything it read | `ml task run`, never by hand |

`ml` reads every input from SharePoint and writes every output there. There is
no local data option.

## Step 1: the SharePoint addresses in `.env`

Copy `.env.example` to `.env` next to `config.yaml` and fill it in. One variable
per file; a path is relative to its drive, and `%20` encoding is fine.

```bash
SHAREPOINT_DRIVE_A="…"                        # the drive site A's files are on
SHAREPOINT_DRIVE_B="…"                        # the drive site B's files are on
SHAREPOINT_VOCABULARY="<team site>/Data/vocabulary.xlsx"
SHAREPOINT_CATALOG="<team site>/Data/catalog/catalog.xlsx"
SHAREPOINT_BATCH_TABLE="<team site>/Data/batch_data/batch_data.xlsx"
SHAREPOINT_SCREENING="<team site>/Data/catalog/parameter_screening.xlsx"
…                                             # the rest, as .env.example lists them
```

A variable that is not set stops every command, and is named. Variables already
set in the shell win over `.env`, so a scheduler can override them.

## Step 2: check `config.yaml`

`config.yaml` in the repository root is annotated. Under `sharepoint:` it gives
every input a path and a drive, by the variables above:

```yaml
sharepoint:
  vocabulary:  {path: "${SHAREPOINT_VOCABULARY}", drive_id: "${SHAREPOINT_DRIVE_A}"}
  catalog:     {path: "${SHAREPOINT_CATALOG}", drive_id: "${SHAREPOINT_DRIVE_A}"}
  batch_table: {path: "${SHAREPOINT_BATCH_TABLE}", drive_id: "${SHAREPOINT_DRIVE_A}"}
  screening:   {path: "${SHAREPOINT_SCREENING}", drive_id: "${SHAREPOINT_DRIVE_A}"}
  …
```

The catalog's folder is where `ml check-catalog` writes, and the batch table's
folder is where `ml build` writes. The rest of `config.yaml` holds the settings
of the build (cleaning rules, feature thresholds, which batches the missingness,
correlation and shifts reports describe). [The reference](05-reference.md#configyaml) lists them all.

## Step 3: the vocabulary

The code never names a real column, value, site or worksheet. It refers to each
by a key, such as `col("vector_titer")`, and the private **vocabulary** workbook
says what each key is called in your sources. Its `Names` sheet has one row per
name:

| Kind | Key | Name |
|---|---|---|
| `column` | `vector_titer` | the column's name in the source |
| `column` | `clump_bags` | one row per name, in order, for a list |
| `value` | `type_commercial` | a category value the code compares against |
| `site` | `a` | site A's name (`b`: site B's) |
| `sheet` | `phf_site_a` | the worksheet a source's data is on |

`src/ml/names.py` lists every key the code uses, with what it means. A
vocabulary that leaves a key out is refused, naming the key. `vocabulary.csv`
in this repository is a copy of the `Names` sheet for reference.

## Step 4: known corrections

Known errors in the sources are rules on the `Corrections` sheet of the same
workbook, so a data steward can add one without touching the code:

| Column | Meaning |
|---|---|
| `Name` | The rule's name; rows sharing a name form one rule. |
| `Site`, `Column` | Where it applies, by the site's and the column's names. |
| `Action` | `set`, `replace_text`, `replace_values` or `divide_by`. |
| `Old`, `New` | What is replaced, and by what (`divide_by`: the divisor). |
| `When column`, `When equals` | Optional: only the rows where this column equals this. |
| `Reason` | Why. |
| `Enabled` | Optional: `no` switches the rule off. |

The build report lists every rule that ran and the rows it changed.

## Step 5: install `io_sharepoint`, and check

Install `io_sharepoint` in the same environment (`uv pip install …`), then:

```bash
uv run ml validate
```

```
OK  config  Vocabulary loaded — 253 names
OK  cli     Corrections table: 14 rules
OK  cli     Screening register
            1 parameter excluded at some site; 1 rows, last decision S-0001
OK  cli     Everything checked out
```

`ml validate` reads the vocabulary, the corrections and the screening register,
and writes nothing. If the register does not exist yet, it says so, and that is
fine: the first `ml screening add --new` creates it
([chapter 3](03-screening.md)).

## Before every push: `ml check-leaks`

```bash
uv run ml check-leaks          # from the repository root
```

It reads the private vocabulary and searches the repository's tracked files for
any real name. It exits with an error and lists every line where it finds one.
