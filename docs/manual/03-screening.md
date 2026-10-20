# 3. Screening parameters

Some parameters are not fit to predict with: a site never records them, they
were retired, or they are recorded so irregularly that a model would learn the
gaps. Others are redundant: they rank the batches as another parameter does
(the correlation report), or fall into the same level groups over time, as a lot
and its certificate values do (the shifts report). **Screening** leaves such a
parameter out of every task, with the reason written down (and, for a redundant
one, the parameter that represents it), until you decide otherwise.

## How it works

- **One register.** `parameter_screening.xlsx` on SharePoint
  (`sharepoint.screening` in `config.yaml`) holds every screening decision ever
  made.
- **A log, not a list.** Each row is one decision: *exclude* or *reinstate* one
  parameter at one site (or `all`), with who, when, why, and the evidence. Rows
  are only added, by `ml screening add`; a change of mind is a new row. Today's
  state is the log read from top to bottom: per parameter and site, the last
  decision stands.
- **Every task applies it.** When a task picks its predictors, it leaves out
  every parameter the register excludes at the task's sites. A task over both
  sites leaves out what either site excludes. `ml task new` also lists them in the
  new `task.yaml` under `predictors.exclude`, so the task file shows them.
- **Every run keeps the evidence.** The run's `parameters.csv` names the decision
  behind each exclusion, its `manifest.json` records which register it applied,
  and `inputs/screening.xlsx` is a copy of that register. `ml task rerun` applies
  the copy, so a rerun applies exactly the same decisions however the register
  has changed since.
- **Nothing is deleted.** The batch table and the reports keep every parameter.
  Screening only decides what a task may use. `task_catalog.xlsx`, written by
  every build, lists what is left.

## Deciding: what to look at

Open the latest `missingness_report.html`, `correlation_report.html` and `shifts_report.html`
([chapter 2](02-monthly-update.md#5-read-the-missingness-report)).
Good candidates for screening:

| What you see | Reason to record |
|---|---|
| *Never recorded* at a site | `never recorded` |
| *Retired*, with no value in the latest batches | `retired` |
| *Introduced late*, with too little history yet | `introduced late` |
| Mostly missing, or a value in too few batches to learn from | `too sparse` |
| *Regular skipping*, *Clumped*, *Gap in the middle* with no process reason | `irregular` |
| A group that goes missing together, explained by one of the above | `missing with group` |
| A member of a group in the correlation report, represented by another member | `correlated with group`, with `Represented by` |
| A member of a group in the shifts report, represented by another member | `shifts with group`, with `Represented by` |
| Values you know to be wrong or incomparable | `data quality` |
| Anything else | `other`, and say why |

Screen at the site where the problem is. A parameter that is fine at Raritan and
never recorded at Ghent is excluded at Ghent; only tasks that include Ghent
batches lose it.

## Recording a decision

**1. Write the proposal from the report.** In the report (`missingness_report.html`,
`correlation_report.html` or `shifts_report.html`), pick the site and tick what
the decision is about: a group in the group table,
parameters in the parameter table, or (to reinstate them) rows of *Screened out
at this site*. Open **Propose a screening decision** below the tables and fill in:

- **Action**: exclude or reinstate;
- **Reason**: one of the list above;
- **Rationale**: why, in a sentence;
- **Review by**: three months from today unless you change it;
- **Author**: your name;
- **Represented by** (correlation and shifts reports, exclude only): the member
  that stays, starting on the report's suggestion;
- **Leave out of the proposal**: ticked parameters the proposal should skip.

The form shows the rows it will write, and **Download proposal.csv** saves them,
one row per parameter with every column filled in:

```
Action,Parameter,Group,Site,Reason,Rationale,Review by,Author,Represented by
exclude,Day 3 clump observation,Missing 05,Northfield,missing with group,clump observations recorded on some batches only,2027-01-19,S. Degroeve,
exclude,D3 clumps present,Missing 05,Northfield,missing with group,clump observations recorded on some batches only,2027-01-19,S. Degroeve,
exclude,Cytokine lot,Shifting 01,Northfield,shifts with group,changes with the medium lot,2027-01-19,S. Degroeve,Medium lot
```

Only parameters that are screened out at that site can be reinstated; the form
says which ticked ones it leaves out, and why.

From the correlation or shifts report, a ticked group becomes an exclusion of every member
but its representative, with the representative in the `Represented by` column.
`ml screening status` lists such an exclusion again when its representative is
screened out later, since the group would then have no parameter left.

You can also write the file by hand, in Excel or a text editor:

| Column | Fill in |
|---|---|
| `Action` | `exclude` or `reinstate` |
| `Parameter` | The parameter's exact name, as in the report |
| `Group` | A group from a report's group table, such as `Missing 05`, `Correlated 02` or `Shifting 01`. With no `Parameter`, the row stands for every member of the group at that site; with one, it is context |
| `Site` | A site's name, or `all` (not with a group: groups are found per site) |
| `Reason` | One of the reasons in the table above |
| `Rationale` | Why, in a sentence. Required |
| `Review by` | Optional date (`YYYY-MM-DD`) to look at it again; empty means three months from today |
| `Author` | Who decides; or leave it empty and pass `--author` |
| `Represented by` | The parameter that stands in for the excluded ones; it is never excluded itself and must be usable at the site. Required with `correlated with group` and `shifts with group`: left empty, it is that report's suggestion for the group |

**2. Add it.**

```bash
uv run ml screening add proposal.csv
```

```
OK  screening  S-0002: exclude 2 parameters at Northfield — missing with group
               Day 3 clump observation, D3 clumps present
OK  screening  S-0003: exclude 1 parameter at Southbank — never recorded
               Seeding density vessel A

Screening register updated
  Register  SharePoint DEMO_DRIVE: Data/catalog/parameter_screening.xlsx
  State     4 parameters excluded at some site; 4 rows, last decision S-0003
  Next      every task run from now on leaves these parameters out
```

`ml screening add` checks the whole proposal against the latest build before it
writes anything. It refuses a proposal with an unknown parameter or group, a
reason that isn't on the list, an empty rationale, or a reinstatement of
something that isn't excluded. It names every problem at once and adds nothing.

Each decision gets an ID (`S-0002`). The rows of one group with the same action,
reason, rationale, review date and author share one ID: they are one decision.
A group is stored as its members, because group numbers can change with the next
build and the members are what was decided. The tool fills in the date,
the author, and the evidence from the latest build: missing %, pattern, last
measurement, and the digest of the batch table it was decided on.

The first time, there is no register yet: add `--new` to start one.

**3. Check.**

```bash
uv run ml screening status
```

The next build shows the decision in the missingness report's *Screening*
column, and the next task run lists the parameters as excluded:

| Parameter | Decision | Reason | Stage |
|---|---|---|---|
| Day 3 clump observation | excluded | screened out by S-0002: missing with group (Northfield) | screening |

## Changing your mind

When new months of data come in, `ml screening status` lists the decisions the
new build questions ([chapter 2, step 6](02-monthly-update.md#6-check-the-standing-decisions)).
For each one, decide:

- **Reinstate it:** a proposal row with `Action: reinstate`, the same parameter
  and site, and why. The parameter is back in every task from the next run.
- **Keep it, and look again later:** add an `exclude` row with fresh evidence and
  a new `Review by` date. The old row stays in the log.
- **Change the scope:** to move an exclusion from `all` to one site, reinstate at
  the site where the parameter is fine.

Never edit or delete a row in the register by hand. The register's `Check` sheet
holds a digest of its content. When the rows no longer match it, every command
warns that the register was changed outside `ml screening`, and each task run
records that in its manifest.

## Good practice

- **Decide on the commercial cohort** the build's reports describe, not on a
  single task's cohort, so the decision holds for every task.
- **Work through the reports in order:** missingness, then correlation, then
  shifts. Each leaves out what is already screened out for another reason, so
  after a missingness decision, the correlation and shifts groups of the next
  build are the ones to decide on.
- **Check the representative.** The suggestion is the member that carries most
  of the others' information; prefer a parameter you can explain, or that a
  task needs, if it is nearly as good.
- **One reason per decision, in words someone else understands in a year.** "Not
  recorded since the new batch record template" beats "sparse".
- **Use review dates.** Three months is the default; use a shorter one for
  *introduced late* parameters that are gaining history.
- **Review the status every month** (checklist step 6), even when nothing seems
  to have changed.
- **When a task must use a screened parameter** for a specific question, it can't
  override the register: reinstate the parameter, with the reason, or ask the
  question with a task that doesn't need it. That keeps every exception on
  record.
