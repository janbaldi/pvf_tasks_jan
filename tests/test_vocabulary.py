"""The vocabulary: the code names nothing itself, and every name comes from one place.

Every column, category value, site and sheet the sources use is a key in the
code and a name in the vocabulary — the demo's invented one here, a private
workbook on SharePoint for real runs. These tests hold the code to that.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

import dummy_data
from conftest import ROOT
from ml import cli, leaks, names

SOURCE = ROOT / "src" / "ml"


def _keys_used(pattern: str, text: str) -> set[str]:
    return set(re.findall(pattern + r'\(\s*"([a-z0-9_]+)"', text))


def test_every_key_the_code_uses_is_registered():
    for path in SOURCE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if path.name == "demo.py":
            columns = _keys_used(r"\bcs?", text)
            assert columns <= set(names.COLUMNS) | set(dummy_data.DEMO_COLUMNS), path.name
            values = _keys_used(r"\bvs?", text)
            assert values <= set(names.VALUES) | set(dummy_data.DEMO_VALUES), path.name
            continue
        for call, registry in (("col", names.COLUMNS), ("cols", names.COLUMNS)):
            assert _keys_used(rf"\b{call}", text) <= set(registry), (path.name, call)
        for call in ("value", "values"):
            assert _keys_used(rf"\b{call}", text) <= set(names.VALUES), (path.name, call)


def test_the_code_names_no_column_or_value_itself():
    """No module but the registry and the demo holds a vocabulary name: they go by key."""
    code = "\n".join(
        path.read_text(encoding="utf-8")
        for path in SOURCE.glob("*.py")
        if path.name not in ("names.py", "demo.py")
    )
    vocabulary = names.demo()
    terms = leaks.vocabulary_terms(vocabulary)
    assert terms, "the demo vocabulary has names worth looking for"
    found = [name for name, pattern in terms.items() if pattern.search(code)]
    assert found == []


def test_a_vocabulary_that_leaves_a_key_out_is_refused_by_name():
    vocabulary = names.demo()
    del vocabulary.columns["vector_titer"]
    with pytest.raises(names.VocabularyError, match="column vector_titer"):
        names.use(vocabulary)
    names.use(names.demo())


def test_a_vocabulary_survives_its_sheet(tmp_path):
    path = tmp_path / "vocabulary.xlsx"
    names.to_frame(names.demo()).to_excel(path, sheet_name=names.NAMES_SHEET, index=False)
    back = names.from_frame(pd.read_excel(path, sheet_name=names.NAMES_SHEET))
    demo = names.demo()
    assert (back.columns, back.values, back.sites, back.sheets) == (
        demo.columns,
        demo.values,
        demo.sites,
        demo.sheets,
    )


def test_a_name_template_takes_its_placeholder():
    assert names.col("f_lactate_glucose_ratio", day=8).endswith("D8")
    assert "{" not in names.col("coa_donor", metric="X", donor=2)


def test_the_leak_check_finds_names_and_ignores_plain_words(tmp_path):
    vocabulary = names.demo()
    terms = leaks.vocabulary_terms(vocabulary, dummy_data.demo_corrections())
    clean = tmp_path / "clean.py"
    clean.write_text('x = col("vector_titer")  # a Country and a Volume are just words\n')
    leaky = tmp_path / "leaky.md"
    leaky.write_text(
        f"The {names.site('a')} site records {names.col('vector_titer')}.\n"
        f"Lot {dummy_data.v('bad_lot')} is fixed by hand.\n"
    )
    found = leaks.scan([clean, leaky], terms, tmp_path)
    assert {leak.path for leak in found} == {"leaky.md"}
    assert {leak.name for leak in found} >= {
        names.site("a"),
        names.col("vector_titer"),
        dummy_data.v("bad_lot"),
    }
    assert "Patient Lot/Batch #" not in terms, "kept visible on purpose"


def test_check_leaks_reads_the_vocabulary_from_sharepoint_and_reports(built, tmp_path):
    config = str(built["config_path"])
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "notes.md").write_text("Nothing private here.\n")
    assert cli.main(["--config", config, "check-leaks", str(clean)]) == 0

    leaky = tmp_path / "leaky"
    leaky.mkdir()
    (leaky / "notes.md").write_text(f"{names.col('harvest_car_pct')} went up.\n")
    assert cli.main(["--config", config, "check-leaks", str(leaky)]) == 1


def test_the_shipped_config_and_template_name_no_column(built):
    """config.yaml names columns by key; the task template uses placeholders."""
    text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    template = (SOURCE / "templates" / "task.yaml").read_text(encoding="utf-8")
    assert "pre_activation_clumps_yn" in text and "excluded_mapping_row" in text
    for name in (names.col("pre_activation_clumps_yn"), names.col("excluded_mapping_row")):
        assert name not in text and name not in template
    assert Path(SOURCE / "templates" / "corrections.yaml").exists() is False
