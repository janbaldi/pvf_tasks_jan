# 3. Screening parameters

Some parameters are not fit to predict with: a site never records them, they
were retired, or they are recorded so irregularly that a model would learn the
gaps. Others are redundant: they rank the batches as another parameter does
(the correlation report). **Screening** leaves such a parameter out of every
task, with the reason written down (and, for a redundant one, the parameter that
represents it), until you decide otherwise. Parameters that run in groups over
time, as a lot does (the shifts report), are left out per task instead, with
`ml task new --exclude-shifting` ([chapter 4](04-tasks.md#1-create-a-task)): the
shifts report neither reads nor writes the register.

The *Screening loop* route of [the workflow map](../workflow.html) shows the
whole cycle: reports, proposal, register, and back into the next build and
every task.

## How it works

- **One register.** `parameter_screening.xlsx` on SharePoint
  (`sharepoint.screening` in `config.yaml`) holds every screening decision ever
  made.
- **A log, not a list.** Each row is one decision: *exclude* or *reinstate* one
  parameter at one site (or `all`), with who, when, why, and the evidence. Rows
  are only added, by `ml screening add`; a change of mind is a new row.
- **Two reports, two tracks.** Each row records the report it was made from in
  its `Report` column: `missingness` (every reason but one) or `correlation`
  (`correlated with group`). Today's state is the log read from top to bottom:
  per parameter, site **and report**, the last decision stands. So a parameter
  can be excluded in both reports at once, and reinstating it in one leaves the
  other's exclusion standing. A parameter is out when either report excludes it.
- **Every task applies it.** When a task picks its predictors, it leaves out
  every parameter the register excludes at the task's sites. A task over both
  sites leaves out what either site excludes. `ml task new` also lists them in the
  new `task.yaml` under `predictors.exclude`, so the task file shows them.
- **Every run keeps the evidence.** The run's `parameters.csv` names the decision
  behind each exclusion, its `manifest.json` records which register it applied,
  and `inputs/screening.xlsx` is a copy of that register. `ml task rerun` applies
  the copy, so a rerun applies exactly the same decisions however the register
  has changed since.
- **Every build shows it, and is not steered by it.** The missingness and
  correlation reports each analyse every parameter, whatever the register says:
  neither depends on the other's decisions. Both show the whole register. ⊘ marks
  what the report itself excluded (listed under *Screened out at this site*,
  where you can reinstate it), ◇ what the other report excluded (listed under
  *Screened out by the … report*, read only), and the *Screening* / *Register*
  column spells out every decision on a parameter, and what it represents. Each
  report's lead names the register it shows (rows, last decision), or says why
  none could be read; the build log counts the exclusions per site and report and
  warns about decisions on parameters the batch table does not have.
  `task_catalog.xlsx` is cut by it. And the build adds its own `never recorded`
  exclusions to it (below).
- **Conflicts are named, never resolved silently.** A representative that is
  excluded itself would leave its whole group out. `ml screening add` refuses a
  proposal that makes one, in either order. The build's own `never recorded`
  exclusions can still make one: then a ⚠ banner names it in both reports, the
  build log warns, `ml screening status` lists it, `ml task new` writes it as a
  comment in `task.yaml`, and every task run warns. Resolve it in the correlation
  report (below).
- **Nothing is deleted.** The batch table and the reports keep every parameter.
  Screening only decides what a task may use. `task_catalog.xlsx`, written by
  every build, lists what is left.

## Deciding: what to look at

Open the latest `missingness_report.html` and `correlation_report.html`
([chapter 2](02-monthly-update.md#5-read-the-missingness-report)).
Good candidates for screening:

| What you see | Reason to record |
|---|---|
| *Retired*, with no value in the latest batches | `retired` |
| *Introduced late*, with too little history yet | `introduced late` |
| Mostly missing, or a value in too few batches to learn from | `too sparse` |
| *Regular skipping*, *Clumped*, *Gap in the middle* with no process reason | `irregular` |
| A group that goes missing together, explained by one of the above | `missing with group` |
| A member of a group in the correlation report, represented by another member | `correlated with group`, with `Represented by` |
| Values you know to be wrong or incomparable | `data quality` |
| Anything else | `other`, and say why |

A parameter missing on every batch at a site needs no decision: each build
excludes it there itself, as `never recorded`, with `ml build` as the author and
one decision ID per site. It does so only once, and never for a parameter the
missingness report already decides on at that site, so a reinstatement stands. Parameters
that are never missing are left out of the missingness report altogether.

Screen at the site where the problem is. A parameter that is fine at Raritan and
never recorded at Ghent is excluded at Ghent; only tasks that include Ghent
batches lose it.

## Recording a decision

**1. Write the proposal from the report.** In the report (`missingness_report.html`
or `correlation_report.html`), pick the site and tick, with the box at the
start of a row, what the decision is about: groups in the group table,
parameters in the parameter table, or (to reinstate them) rows of *Screened out
at this site*. You can tick rows in several tables at once. Clicking a row
elsewhere only draws it. Open **Propose a screening decision** below the tables
and fill in:

- **Action**: exclude or reinstate;
- **Reason**: one of the list above;
- **Rationale** (optional): why, in a sentence;
- **Author**: your name;
- **Represented by** (correlation report, per ticked group): the member
  that stays, starting on the report's suggestion;
- **Leave out of the proposal**: ticked parameters the proposal should skip.

The form shows the rows it will write, and **Download proposal.csv** saves them,
one row per parameter with every column filled in:

```
Action,Parameter,Group,Site,Reason,Rationale,Author,Represented by,Report
exclude,Day 3 clump observation,Missing 05,Northfield,missing with group,clump observations recorded on some batches only,S. Degroeve,,missingness
exclude,D3 clumps present,Missing 05,Northfield,missing with group,clump observations recorded on some batches only,S. Degroeve,,missingness
```

Each report offers only its own reasons, and can only reinstate what it screened
out itself at that site; the form says which ticked ones it leaves out, and why.
To reinstate a parameter the other report excluded, do it in that report.

From the correlation report, each ticked group becomes an exclusion of every member
but its representative, with the representative in the `Represented by` column.
The group table's *Best usable representative* is the best-scoring member the
missingness report leaves in (`= suggested` when that is the suggestion): pick
it when the suggestion is excluded there. The form refuses a representative the
missingness report excludes. **To move a group to another representative**, tick
the group and pick the new one: the proposal reinstates it (if this report
excluded it) and excludes the others, represented by it, in one go.

You can also write the file by hand, in Excel or a text editor:

| Column | Fill in |
|---|---|
| `Action` | `exclude` or `reinstate` |
| `Parameter` | The parameter's exact name, as in the report |
| `Group` | A group from a report's group table, such as `Missing 05` or `Correlated 02`. With no `Parameter`, the row stands for every member of the group at that site; with one, it is context |
| `Site` | A site's name, or `all` (not with a group and no `Parameter`: groups are found per site) |
| `Reason` | One of the reasons in the table above |
| `Rationale` | Why, in a sentence. Optional |
| `Author` | Who decides; or leave it empty and pass `--author` |
| `Represented by` | The parameter that stands in for the excluded ones; it must stay usable at the site. Required with `correlated with group`: left empty, it is the report's suggestion for the group |
| `Report` | `missingness` or `correlation`: the report the decision belongs to. Left empty, the report of its reason |

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
reason that isn't on the list or belongs to the other report, a reinstatement
of something that report hasn't excluded, or a conflict: a representative that
would be excluded, or an exclusion of a parameter that represents others. The
conflict check is on the register as the whole proposal leaves it, so moving a
group to another representative in one proposal is fine. It names every problem
at once and adds nothing.

Each decision gets an ID (`S-0002`). The rows of one group with the same action,
reason, rationale and author share one ID: they are one decision.
A group is stored as its members, because group numbers can change with the next
build and the members are what was decided. The tool fills in the date,
the author, the evidence from the latest build (missing %, pattern, last
measurement, and the digest of the batch table it was decided on) and, for an
exclusion, a review date three months on.

The first time, there is no register yet: add `--new` to start one.

**Starting over.** `uv run ml screening reset --yes` replaces the register with
an empty one. The old one is kept next to it on SharePoint, renamed
`parameter_screening_before_reset_<date>.xlsx`. The next `ml build` adds the
`never recorded` exclusions again; the reports propose the rest. Tasks already
run keep their own copy, so their reruns are unchanged.

If SharePoint answers that it cannot replace the register (409 Conflict), the
file is open in Excel somewhere (desktop or browser, yours or a colleague's) or
checked out: close it everywhere, check it in, wait a minute and run the command
again. Nothing was written.

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
- **Keep it, and look again later:** add an `exclude` row; it gets fresh evidence
  and a new review date. The old row stays in the log.
- **Change the scope:** to move an exclusion from `all` to one site, reinstate at
  the site where the parameter is fine.

Never edit or delete a row in the register by hand. The register's `Check` sheet
holds a digest of its content. When the rows no longer match it, every command
warns that the register was changed outside `ml screening`, and each task run
records that in its manifest.

## Good practice

- **Decide on the commercial cohort** the build's reports describe, not on a
  single task's cohort, so the decision holds for every task.
- **The reports stand on their own,** so you can work through them in any
  order. Missingness first is still the natural one: the correlation report then
  shows those decisions as ◇, and its *Best usable representative* skips them.
- **Check the representative.** The suggestion is the member that carries most
  of the others' information; prefer a parameter you can explain, or that a
  task needs, if it is nearly as good.
- **One reason per decision, in words someone else understands in a year.** "Not
  recorded since the new batch record template" beats "sparse".
- **Let the review dates work.** Every exclusion is due for review three months
  on, and `ml screening status` lists it then; reinstate an *introduced late*
  parameter sooner if it gains history.
- **Review the status every month** (checklist step 6), even when nothing seems
  to have changed.
- **When a task must use a screened parameter** for a specific question, it can't
  override the register: reinstate the parameter, with the reason, or ask the
  question with a task that doesn't need it. That keeps every exception on
  record.
