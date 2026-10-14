"""Finding a real name in files that should hold none.

The code refers to every column, value, site and sheet by a key, and the real
names live in the private vocabulary on SharePoint. ``ml check-leaks`` reads that
vocabulary and its corrections, and searches the repository's files for any of
them — the check to run before pushing, since only a run with access to the share
knows what to look for.

A name is matched the way it could leak:

* a site's name, as a whole word anywhere;
* a name with a space, a digit or a symbol in it (``Day 3 …``, ``A12/B``), anywhere.

A single plain word (``Country``, ``Sheet1`` aside) is too common to tell a leak
from an ordinary word or a table heading, so it is not searched for.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import corrections, names

#: Shorter than this, a name is too common to mean anything when it turns up.
MIN_LENGTH = 4
#: Names kept visible on purpose: the batch identifier every stage joins on.
VISIBLE = frozenset({"Patient Lot/Batch #"})


@dataclass(frozen=True)
class Leak:
    path: str
    line: int
    name: str
    text: str


def vocabulary_terms(
    vocabulary: names.Vocabulary, rules: list[corrections.Rule] = ()
) -> dict[str, re.Pattern]:
    """Every name and value worth looking for, with the pattern that finds it."""
    terms: dict[str, re.Pattern] = {}

    def add(value, whole_word: bool = False) -> None:
        for piece in re.split(r"\{[^}]*\}", str(value)):
            piece = piece.strip()
            if len(piece) < MIN_LENGTH or piece in terms or piece in VISIBLE:
                continue
            escaped = re.escape(piece)
            if whole_word:
                terms[piece] = re.compile(rf"\b{escaped}\b")
            elif re.fullmatch(r"[A-Za-z]+\d?", piece):
                continue  # a plain word: as common in code and prose as anywhere
            else:
                terms[piece] = re.compile(escaped)

    for site in vocabulary.sites.values():
        add(site, whole_word=True)
    for name in vocabulary.names():
        add(name)
    for rule in rules:
        add(rule.column)
        add(rule.when_column)
        if rule.when_equals is not None:
            add(rule.when_equals)
        if rule.action == "set" and rule.value is not None:
            add(rule.value)
        if rule.action in ("replace_text", "replace_values"):
            pairs = rule.value if rule.action == "replace_text" else rule.value.items()
            for old, new in pairs:
                add(old)
                add(new)
    return terms


def tracked_files(root: Path) -> list[Path]:
    """The files git tracks under ``root``, or every file when it is not a checkout."""
    try:
        listed = subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.splitlines()
        return [root / line for line in listed if line]
    except (OSError, subprocess.CalledProcessError):
        skip = {".git", ".venv", "__pycache__", "node_modules"}
        return [
            path
            for path in root.rglob("*")
            if path.is_file() and not skip & set(path.relative_to(root).parts)
        ]


def scan(files: list[Path], terms: dict[str, re.Pattern], root: Path | None = None) -> list[Leak]:
    """Every line of ``files`` that holds one of ``terms``."""
    found: list[Leak] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # a binary file holds no names a reader would see
        shown = str(path.relative_to(root)) if root else str(path)
        for number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in terms.items():
                if pattern.search(line):
                    found.append(Leak(shown, number, name, line.strip()[:120]))
    return found
