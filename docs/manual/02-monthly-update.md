# 2. The monthly update

When a new month of batches is in the sources, bring the batch table up to date.
This chapter is the routine, in order, with what to look at in each report.
Allow an hour the first time and a quarter of an hour after that.

## Checklist

1. `ml validate`: the setup still works.
2. `ml check-catalog`: the sources record nothing the catalog doesn't know about, or you add it.
3. `ml build`: the batch table, the build report and the missingness report.
4. Read the build report.
5. Read the missingness report, and record any screening decision ([chapter 3](03-screening.md)).
6. `ml screening status`: no standing decision is contradicted by the new data.
7. Rerun the tasks you maintain ([chapter 4](04-tasks.md)).

Run every command from the repository (or any folder below it), as `uv run ml …`.

## 1. Validate

```bash
uv run ml validate
```

It reads the vocabulary, the corrections and the screening register, and writes
nothing. Fix whatever it lists before going on.

## 2. Check the catalog

```bash
uv run ml check-catalog
```

It compares every source's columns with the catalog, and ends with what it found:

```
Northfield batch records: columns the catalog does not list (1)
  - Site note
Southbank batch records: columns the catalog does not list (2)
  - Plant code
  - Remark
```

These columns are in the sources but won't be in the batch table. **This is your
to-do list:** add the rows you want to `catalog.xlsx` (with a value type and
availability), and leave the rest out on purpose. The same list is in
`new_parameters.csv`, and `catalog_report.html` shows it source by source; both
are in the catalog's folder on SharePoint.

## 3. Build

```bash
uv run ml build
```

```
Written to SharePoint — 70 batches × 230 columns
  batch table            batch_data.xlsx
  batch table (Parquet)  batch_data.parquet
  report                 build_report.html
  missingness table      missingness.xlsx
  missingness report     missingness_report.html
  manifest               batch_table_manifest.json
```

Everything goes into the batch table's folder on SharePoint. The manifest records
what was read (with digests), what was written, and every warning, so any batch
table can be traced back to the sources it came from.

If it warns that the last `ml check-catalog` found parameters the catalog
doesn't list, that's step 2's list: the build goes ahead without them.

## 4. Read the build report

Open `build_report.html` in a browser. The first time, it needs network access
and 10 to 30 seconds to start. Look at these sections:

| Section | Look for |
|---|---|
| **Overview** | The batch and column counts per site: do they match what you expect for the month? |
| **Cleaning** | Every correction that changed something, and every value that could not be read (a date that isn't a date, a number written as text). A new kind of error here usually means a new correction rule. |
| **Catalog coverage** | Catalog parameters that are not in the batch table, and why. |
| **Derived features** | Calculated features that were blocked (an input missing, or a setting it waits for), and the data checks: doublings against cell counts, seeding and formulation arithmetic, features the catalog says are known before their inputs. Each names batches or catalog rows to look at. |
| **Run log** | Every warning of the run. |

## 5. Read the missingness report

Open `missingness_report.html`. It describes **commercial batches only**: Ghent
batches whose `Type` is `Commercial`, and every Raritan batch
(`build.missingness_cohort` in `config.yaml`). The top of the page says how many
batches each site's analysis covers.

Pick a site in the sidebar. The page has three parts:

- **The pattern plot.** One row per parameter, one cell per batch in date order,
  newest on the right. Hover a cell for its batch and date.
- **The group table.** Parameters that go missing together. Tick a group to draw
  all its members.
- **The parameter table** (collapsed). Every parameter with its **pattern**, its
  statistics and, in the **Screening** column, the decision standing for it at
  this site. Tick parameters to add them to the plot.

The pattern says how a parameter goes missing:

| Pattern | Meaning | Usually |
|---|---|---|
| Complete | Never missing. | Fine. |
| Never recorded | Missing on every batch. | Screen out at that site. |
| Introduced late | Measured from some batch on. | Keep it, or screen it out until there is enough history. |
| Retired | No longer measured. | Screen out: new batches will never have it. |
| Measured in a window | Missing at the start and the end. | Look closer. |
| Gap in the middle | One stretch missing. | Look closer: a recording problem? |
| Clumped | Several long gaps. | Look closer. |
| Random gaps | As many gaps as chance would give. | Usually fine. |
| Regular skipping | Every n-th batch skipped. | Look closer: a sampling plan? |

The *Help* button in the sidebar explains every statistic. The parameter table
downloads as a CSV, and `missingness.xlsx` in the same folder holds all of it,
for both sites.

When you decide that a parameter or a group should not be used by any task,
record it now: [chapter 3](03-screening.md).

## 6. Check the standing decisions

```bash
uv run ml screening status
```

```
Excluded today (1)
  - Day 1 mixing cycles @ Northfield — S-0001, irregular

Decisions to look at again (0)
OK  The latest build gives no reason to revisit a decision
```

It compares every standing exclusion with the new build. It lists a decision
when the parameter is measured again after being *retired* or *never recorded*,
when its missing rate moved by 20 points or more, when it is no longer in the
batch table, or when its review date has passed. Nothing changes until you
decide: see [Changing your mind](03-screening.md#changing-your-mind).

## 7. Rerun your tasks

```bash
uv run ml task run --task tasks/expression --task tasks/release
```

Each run reads the new batch table and the current screening register, and
writes a new run folder next to the previous ones. Nothing is overwritten.

`ml update --task …` does steps 2, 3 and 7 in one go. Run it unattended only
when you don't need to read the reports in between.
