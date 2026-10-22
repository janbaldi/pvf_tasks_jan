# 6. Development

```bash
uv sync --locked              # the environment the tests ran in
uv run pytest                 # every stage, end to end, on the demo data
uv run ruff check src tests   # lint
```

CI runs both on every push (`.github/workflows/ci.yml`). The demo's data and
runs are generated, not committed: only `demo/config.yaml` and the two task files
are in git. Tests and the demo use only the invented names of `src/ml/names.py`;
run `uv run ml check-leaks` before pushing.

## Adding a calculated feature

Append one `Feature(…)` to the registry in `src/ml/features.py`:

- It is built only when the catalog lists its name and all its inputs are present,
  and it never overwrites a source column.
- Its `requires` tuple is what lets a task find, and leave out, features
  calculated from the target.
- Give it a `unit` and a `kind` (`Engineered` or `Normalization`, the catalog's
  `Calculated` column).
- When it needs a `features:` setting with no safe default, name it in `setting`:
  the feature is then blocked until config sets it.
- Add its key to `src/ml/names.py` and its real name to the vocabulary, add a row
  to the catalog, a hand-computed case to `tests/test_features.py`, and its formula
  to `CALCULATED_FEATURES.md`.

## The code

```
src/ml/
  cli.py          the commands
  config.py       config.yaml: discovery, validation, where each input is
  names.py        the vocabulary: every key the code uses, with the demo's invented name
  leaks.py        ml check-leaks
  taskconfig.py   task.yaml and its validation
  scaffold.py     ml task new; templates/ holds the task.yaml it writes
  screening.py    the screening register: reading, replaying, adding, reviewing
  catalog.py io.py clean.py corrections.py enrich.py features.py merge.py   check-catalog, build
  missingness.py correlation.py shifts.py                              the build's screening reports
  dataset.py encode.py decorrelate.py recipe.py                         ml task run
  package.py manifest.py provenance.py logger.py                       run folders, provenance
  report.py plots.py blocks.py streamlit_app.py stlite.py              the reports
  share.py        SharePoint (and the demo's stand-in folder)
  demo.py         the made-up demo data
```
