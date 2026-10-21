# 2. The monthly update

When a new month of batches is in the sources, bring the batch table up to date.
This chapter is the routine, in order, with what to look at in each report.
Allow an hour the first time and a quarter of an hour after that.

## Checklist

1. `ml validate`: the setup still works.
2. `ml check-catalog`: the sources record nothing the catalog doesn't know about, or you add it.
3. `ml build`: the batch table, the build report, and the missingness, correlation and shifts reports.
4. Read the build report.
5. Read the missingness report, then the correlation and shifts reports, and record any screening decision ([chapter 3](03-screening.md)).
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
  correlation report     correlation_report.html
  shifts report          shifts_report.html
  task catalog           task_catalog.xlsx
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
batches whose `Type` is `Commercial`, and every Raritan batch. Of those, the
**statistics and groups use only batches whose manufacturing is completed**
(`Manufacturing Completed?` = `Y`). Batches still in production are drawn in the
plot but counted nowhere else. Both rules are `build.missingness_cohort` in
`config.yaml`. **About this report** in the sidebar says how many batches each
site's analysis covers.

Pick a site in the sidebar. The page opens on the plot, and has these parts:

- **The pattern plot.** It shows the largest group until you tick something.
  One row per parameter, one cell per batch in date order,
  newest on the right. Missing cells are blue, or **green for a batch whose
  manufacturing is not completed**: a gap there may just mean "not measured
  yet". A ⊘ before a name marks a parameter screened out at this site. Hover a
  cell for its batch and date.
- **The group table.** Parameters that go missing together. Tick a group to draw
  all its members. *Screened members* counts the ones already screened out.
- **Screened out at this site.** The screening register's standing exclusions:
  decision, date, reason, rationale and review date. Tick one to draw it, or to
  reinstate it.
- **The parameter table** (collapsed). Every parameter missing on at least one
  batch at this site and in no group (those are in the group table), with its
  **pattern**, its statistics and, in the **Screening** column, the decision
  standing for it at this site. Tick parameters to add them to the plot. The CSV
  download still has every parameter.
- **Propose a screening decision.** Turns everything you ticked into a
  `proposal.csv` ([chapter 3](03-screening.md#recording-a-decision)).

The pattern says how a parameter goes missing:

| Pattern | Meaning | Usually |
|---|---|---|
| Never recorded | Missing on every batch. | Nothing to do: the build screens it out at that site. |
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

### The correlation report

Open `correlation_report.html` next. It looks and works like the missingness
report, over the same batches, for the **numerical** parameters (catalog
`Value Type` numeric, duration or ratio) that are not already screened out at
the site. It groups the parameters that **put the batches in the same order**
(Spearman rank correlation, sign ignored): a decision tree only uses that order,
so to a tree the members of a group are interchangeable.

The parameter table lists only the parameters in no group, each with its closest
parameter; tick one to draw just that pair. The same goes for the missingness
report: a parameter in a group is in the group table, not the parameter table.

- **The scatterplots.** One dot per batch with a value for both parameters;
  members of a group lie along a rising or falling curve. A group of up to four
  parameters shows every pair. A larger one shows one parameter (*Against*, the
  representative by default) on the x-axis of a small plot per other member,
  titled with that member; pick another parameter in *Against* to see its pairs.
- **The group table.** Members, the **suggested representative** (the member
  that carries most of the others' information: the highest mean |ρ| to the other
  members, weighted by how many of their batches it has a value on), mean and
  weakest |ρ|, the fewest batches a pair shares, and the weakest pair's q
  (Benjamini-Hochberg adjusted p-value). A group is drawn from its representative.
- **Screened out at this site, the parameter table and the proposal** work as in
  the missingness report. To **represent a group by one parameter**, tick the
  group: the proposal form sets the reason `correlated with group` and *Represented
  by* to the suggestion (change it if you know better). The proposal excludes
  every other member and records the representative in the register; the
  representative stays usable. Excluded members stay in this report, marked ⊘,
  so they can be reinstated; a parameter screened out for any other reason is
  left out. A group you propose nothing for stays whole: tasks use all of it.
  Every new task file lists the excluded members under `predictors.exclude`, each
  with its decision as the reason ([chapter 4](04-tasks.md)).

### The shifts report

Open `shifts_report.html`. It works like the correlation report, but **per
manufacturing site** (the `Manufacturing Site` column) rather than per data
source: time, and the lots, run per plant. It uses the batches the missingness
report counts in its statistics, split by plant, for parameters of **any type** whose batches, in date order, fall into
**level groups**: stretches of consecutive batches where the value is very
nearly the same, followed by a jump to another level. A raw material or
consumable lot does this, and with it the lot's certificate values, its expiry
date, or anything measured per lot with little noise. Parameters whose level
groups cover **the same batches** are grouped, since they all say which lot a
batch was made with.

A parameter only counts when, in its level groups, the values are very close to
the same (the jump between groups is at least four times the spread within
them) and do not drift. So a trend is not caught, nor a date that advances with
every batch, nor a parameter that merely leans with the lot.

- **The plot.** One row per parameter, one cell per batch in date order. A dark
  line starts a new level group; each level group is one shade, light for the
  parameter's lowest level and dark for its highest. Rows with their lines in
  the same places have the same level groups. Hover a cell for its group's
  value. Each row says the adjusted Rand index (ARI, 1 = the same groups) of its
  level groups and the first row's.
- **The parameter table.** Every parameter that moves in level groups: how many,
  their mean length, the median jump and the spread within groups (in the
  parameter's own unit), the drift within groups, and the permutation p.
- **The group table, screening and the proposal** as in the correlation report,
  with the reason `shifts with group`. Group names end with the plant
  (`Shifting 01 @ …`). The register records decisions per data source, so a
  proposal names the source the plant's batches come from, or `all` when they
  come from both.

Parameters screened out for any other reason (missingness, correlation) are left
out, so a parameter already represented elsewhere is not counted twice.

### The task catalog

`task_catalog.xlsx` is the catalog cut down to what a task can still use: every
catalog parameter in the batch table that the screening register (as this build
read it) does not exclude at every site. *Usable at* names the sites that allow
it (a task over both sites needs both), and *Represents* the parameters it stands
in for. Pick predictors from it when you write a task. A decision added after the
build shows in it from the next build.

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
batch table, when the parameter that represents it is screened out too, or when
its review date has passed. Nothing changes until you
decide: see [Changing your mind](03-screening.md#changing-your-mind).

## 7. Rerun your tasks

```bash
uv run ml task run --task tasks/expression --task tasks/release
```

Each run reads the new batch table and the current screening register, and
writes a new run folder next to the previous ones. Nothing is overwritten.

`ml update --task …` does steps 2, 3 and 7 in one go. Run it unattended only
when you don't need to read the reports in between.
