# 2. The monthly update

When a new month of batches is in the sources, bring the batch table up to date.
This chapter is the routine, in order, with what to look at in each report.
Allow an hour the first time and a quarter of an hour after that.

**Everything is about one site at a time**: the one `site:` in `config.yaml`
names. Go through the checklist for that site; to bring the other site up to
date too, switch `site:` (or pass `--config` with a config about it) and go
through it again. Each site's outputs go into a folder named after it.

## Checklist

1. `ml validate`: the setup still works.
2. `ml check-catalog`: the sources record nothing the catalog doesn't know about, or you add it.
3. `ml build`: the batch table, the build report, and the missingness, correlation and shifts reports.
4. Read the build report.
5. Read the missingness report, then the correlation and shifts reports, and record any screening decision ([chapter 3](03-screening.md)).
6. `ml screening status`: no standing decision is contradicted by the new data.
7. Rerun the tasks you maintain ([chapter 4](04-tasks.md)).

Run every command from the repository (or any folder below it), as `uv run ml …`.
The *Monthly update* route of [the workflow map](../workflow.html) shows these
steps, and what each reads and writes.

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

It compares the site's sources' columns with the catalog (site B's under the
names the parameter mapping gives them), and ends with what it found:

```
Northfield batch records: columns the catalog does not list (1)
  - Site note
```

These columns are in the sources but won't be in the batch table. **This is your
to-do list:** add the rows you want to `catalog.xlsx` (with a value type and
availability), and leave the rest out on purpose. The same list is in
`new_parameters.csv`, and `catalog_report.html` shows it source by source; both
are in the site's folder next to the catalog on SharePoint (`<catalog
folder>/<site name>/`).

## 3. Build

```bash
uv run ml build
```

```
Written to SharePoint — 40 batches × 227 columns
  batch table            batch_data.xlsx
  batch table (Parquet)  batch_data.parquet
  report                 build_report.html
  missingness table      missingness.xlsx
  missingness report     missingness_report.html
  correlation report     correlation_report.html
  shifts report          shifts_report.html
  shifts table           shifts.xlsx
  manifest               batch_table_manifest.json
```

Everything goes into the site's folder next to the batch table on SharePoint
(`<batch table folder>/<site name>/`). The batch table holds the site's batches
only, and only the catalog parameters the site records: one empty in every batch
is left out, and the build report lists it. The manifest records
what was read (with digests), what was written, and every warning, so any batch
table can be traced back to the sources it came from.

If it warns that the last `ml check-catalog` found parameters the catalog
doesn't list, that's step 2's list: the build goes ahead without them.
If it warns that no catalog manifest could be read although step 2 ran, the
warning just above it says why the read failed. The build reads
`catalog_manifest.json` back with `io_sharepoint.download_file_from_sharepoint`
(and writes the manifests with `io_sharepoint.write_text_to_sharepoint`); an
older `io_sharepoint` without them cannot, and the warning says so.

## 4. Read the build report

Open `build_report.html` in a browser. The first time, it needs network access
and 10 to 30 seconds to start. Look at these sections:

| Section | Look for |
|---|---|
| **Overview** | The batch and column counts: do they match what you expect for the month? |
| **Cleaning** | Every correction that changed something, and every value that could not be read (a date that isn't a date, a number written as text). A new kind of error here usually means a new correction rule. |
| **Catalog coverage** | Catalog parameters that are not in the batch table, and why: no source supplies it, or a calculated feature is blocked (an input missing, or a setting it waits for). And, collapsed, the ones this site never records. |
| **Run log** | Every warning of the run. |

## 5. Read the missingness report

Open `missingness_report.html`. It describes **commercial batches only**: at
Ghent the batches whose `Type` is `Commercial`, at Raritan every batch. Of those,
the **statistics and groups use only batches whose manufacturing is completed**
(`Manufacturing Completed?` = `Y`). Batches still in production are drawn in the
plot but counted nowhere else. Both rules are `build.missingness_cohort` in
`config.yaml`. **About this report** in the sidebar says how many batches the
analysis covers.

The sidebar names the site. The tables work the same way in all three reports:
**click a row** to draw it (the row is shaded), and **tick the box** at the
start of rows, in any of the tables, to propose a screening decision about
them. The page has these parts:

- **The pattern plot.** Empty until you click a row in a table.
  One row per parameter, one cell per batch in date order,
  newest on the right. Missing cells are blue, or **green for a batch whose
  manufacturing is not completed**: a gap there may just mean "not measured
  yet". A ⊘ before a name marks a parameter this report screened out at this
  site, ◇ one the correlation report screened out. Hover a cell for its batch and
  date.
- **⚠ Screening conflict** (only when there is one). A correlation
  representative that is excluded itself, which leaves its whole group out:
  resolve it in the correlation report ([chapter 3](03-screening.md#how-it-works)).
- **The group table.** Parameters that go missing together. Click a group to
  draw all its members. *Excluded here* counts the members this report already
  screened out, *Excluded elsewhere* the ones the correlation report did.
- **Screened out at this site.** This report's standing exclusions at the site:
  decision, date, reason, rationale. Click one to draw it; tick it to reinstate
  it.
- **Screened out by the correlation report** (collapsed). That report's
  exclusions here, with their representative: shown, not changed here.
- **The parameter table** (collapsed). Every parameter missing on at least one
  batch at this site and in no group (those are in the group table), with its
  **pattern**, its statistics and, in the **Screening** column, every standing
  decision on it at this site, from either report, and what it represents. Click
  one to draw it.
- **Propose a screening decision.** Turns everything you ticked, in every table, into a
  `proposal.csv` ([chapter 3](03-screening.md#recording-a-decision)).

The pattern says how a parameter goes missing:

| Pattern | Meaning | Usually |
|---|---|---|
| Never recorded | Missing on every batch described (it has a value in some other batch). | Screen it out, or check why the commercial batches lack it. |
| Introduced late | Measured from some batch on. | Keep it, or screen it out until there is enough history. |
| Retired | No longer measured. | Screen out: new batches will never have it. |
| Measured in a window | Missing at the start and the end. | Look closer. |
| Gap in the middle | One stretch missing. | Look closer: a recording problem? |
| Clumped | Several long gaps. | Look closer. |
| Random gaps | As many gaps as chance would give. | Usually fine. |
| Regular skipping | Every n-th batch skipped. | Look closer: a sampling plan? |

The *Help* button in the sidebar explains every statistic. `missingness.xlsx`
in the same folder holds every parameter's row. The tables have
no CSV download; the only download is the proposal.

When you decide that a parameter or a group should not be used by any task,
record it now: [chapter 3](03-screening.md).

### The correlation report

Open `correlation_report.html` next. It looks and works like the missingness
report, over the same batches, for the **numerical** parameters (catalog
`Value Type` numeric, duration or ratio), **whatever the screening register says
of them**: the groups and suggestions depend on the data alone. *Not compared*
lists the other catalog parameters, and why. It groups the parameters that **put the batches in the same order**
(Spearman rank correlation, sign ignored): a decision tree only uses that order,
so to a tree the members of a group are interchangeable.

The parameter table lists only the parameters in no group, each with its closest
parameter; click one to draw it against its closest. Tick
several to propose, for each, that its closest parameter is excluded as
`correlated with group`, represented by the ticked one. The **Register** column
says what the register already holds, from either report: *excluded S-… (missingness:
retired)*, *excluded S-… (correlation: …, represented by …)*, or *represents …*. The same
goes for the missingness report: a parameter in a group is in the group table,
not the parameter table.

- **The scatterplots.** One dot per batch with a value for both parameters;
  members of a group lie along a rising or falling curve. A group of up to four
  parameters shows every pair. A larger one shows one parameter (*Against*, the
  representative by default) on the x-axis of a small plot per other member,
  titled with that member; pick another parameter in *Against* to see its pairs.
- **The group table.** Members, the **suggested representative** (the member
  that carries most of the others' information: the highest mean |ρ| to the other
  members, weighted by how many of their batches it has a value on), mean and
  weakest |ρ|, the fewest batches a pair shares, and the weakest pair's q
  (Benjamini-Hochberg adjusted p-value), the **best usable representative** (the
  best-scoring member the missingness report leaves in; `= suggested` when that
  is the suggestion), and *Excluded here* / *Excluded elsewhere*. A group is
  drawn from its representative.
- **Screened out at this site, the other report's exclusions, the parameter
  table and the proposal** work as in the missingness report, ⊘ and ◇ too. To
  **represent a group by one parameter**, tick the
  group (or several groups): the proposal form sets the reason `correlated with
  group` and, per group, *Represented by* to the suggestion (change it if you
  know better; it refuses one the missingness report excludes). The proposal
  excludes every other member and records the representative in the register;
  the representative stays usable. To move a group to another representative,
  tick it and pick the new one: the proposal also reinstates it.
  A group you propose nothing for stays whole: tasks use all of it.
  Every new task file lists the excluded members under `predictors.exclude`, each
  with its decision as the reason ([chapter 4](04-tasks.md)).

### The shifts report

Open `shifts_report.html`. It finds the parameters, of **any type**, whose value
**runs in groups over time**: it keeps exactly one value for a stretch of days
(blanks allowed), then changes to another. A raw material or consumable lot does
this, and with it the lot's certificate values and expiry date. A model can learn
such a parameter as a stand-in for time. The report also groups the parameters
that **change on the same days**, since they say the same thing: which lot a
batch was made with.

It works **per manufacturing site** (the `Manufacturing Site` column) within the
site's batches, since time and the lots run per plant, over the batches the
missingness report counts in its statistics; with one plant, the sidebar just
names it. Dates count by the day: the batches
of one day have no order, and on the day a lot changes some of them can have the
old lot and some the new one; that day ends the old lot's group and starts the
new one's.

A parameter counts when its groups last more than a day on average (so a value
set per day, such as a date, does not) and it has far fewer groups than its
values shuffled at random would give (so a rare value scattered at random does
not).

- **The plot.** One row per parameter, one bar per group: one value, from the
  first day it was used to the last, with the value written in it where it fits
  (hover for it and its batches). A row's bars alternate dark and light only to
  tell neighbours apart; a blank stretch has no value. Thin grey lines mark the
  days the first row changes value, so the rows that change with it line up.
- **The group table.** Groups of parameters that change on the same days: every
  pair in a group shares at least 80% of its change days. Click one to draw it.
  Names end with the plant (`Shifting 01 @ …`).
- **The parameter table.** Every parameter that runs in groups and is in no
  group of the group table: its groups, days per group, the permutation p, and
  the parameter it shares most change days with. Click one to draw it.

This report stands apart from screening: it neither reads the register nor
writes to it, so every parameter takes part and its tables have no boxes to
tick. To leave all of them out of a task, the grouped ones too, create it with
`ml task new <folder> --exclude-shifting`
([chapter 4](04-tasks.md#1-create-a-task)): it reads every parameter from
`shifts.xlsx`. One the register already screens out is listed once, as
screened.

## 6. Check the standing decisions

```bash
uv run ml screening status
```

```
Excluded today (1)
  - Day 1 mixing cycles @ Northfield — S-0001, missingness: irregular

Conflicts: representatives that are excluded themselves (0)

Decisions to look at again (0)
OK  The latest build gives no reason to revisit a decision
```

It compares every standing exclusion at the site (and at `all`) with the site's
new build. It lists a decision
when the parameter is measured again after being *retired* or *never recorded*,
when its missing rate moved by 20 points or more, when it is no longer in the
batch table, when the parameter that represents it is screened out too, or when
its review date has passed. Nothing changes until you
decide: see [Changing your mind](03-screening.md#changing-your-mind).

## 7. Rerun your tasks

```bash
uv run ml task run --task tasks/expression
```

Each run reads its site's new batch table and the current screening decisions at
that site, and writes a new run folder next to the previous ones. Nothing is
overwritten. A task names its site (`site:` in its `task.yaml`), and refuses to
run with a config about the other site: rerun each site's tasks after that
site's build.

`ml update --task …` does steps 2, 3 and 7 in one go. Run it unattended only
when you don't need to read the reports in between.
