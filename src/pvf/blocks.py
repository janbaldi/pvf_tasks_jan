"""What a unit of report content is.

A block is a plain dict tagged with ``kind``, and nothing more. The payload the
report writes is a tree of these, so anything that can be JSON-encoded can be
rendered, and the renderer in the browser never has to know how a number was
computed.

Eight kinds: ``md``, ``table``, ``figure``, ``image``, ``code``, ``download``,
``collapsed`` (child blocks behind a closed expander), ``deferred`` (child blocks
behind a closed expander *and* a toggle, so a figure nobody opened is never
drawn) and ``chooser`` (one set of child blocks out of many, picked on the page).

The last two exist because a closed expander is not free: Streamlit runs what is
inside it whether or not anyone opens it, so a report with a heatmap per cluster
draws every heatmap on every load.

This module imports nothing from the rest of the package and knows nothing about
sections. Both :mod:`pvf.plots` and :mod:`pvf.report` need this
vocabulary, and putting it in ``report`` would make ``plots`` import ``report``
while ``report`` already imports ``plots``.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

Block = dict[str, Any]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def md(text: str) -> Block:
    """A run of markdown prose."""
    return {"kind": "md", "text": text}


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], note: str = "") -> Block:
    """A table. Cells are markdown, so a cell may hold a link or bold text."""
    return {
        "kind": "table",
        "headers": [str(h) for h in headers],
        "rows": [[str(cell) for cell in row] for row in rows],
        "note": note,
    }


def _scalar(value: Any) -> Any:
    """A cell a sortable table can hold: a plain number, a plain string, or nothing.

    Numbers stay numbers all the way to the browser. That is the whole point of
    a sortable table — a column of pre-formatted strings sorts alphabetically,
    which puts 0.9 above 0.09 and every negative number in the wrong place.
    """
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return str(value)


def sortable(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    formats: Sequence[str | None] | None = None,
    flag: dict[str, Any] | None = None,
    note: str = "",
    filter_column: str = "",
) -> Block:
    """A table the reader can reorder by any column.

    Cells hold raw values rather than formatted text, and ``formats`` gives a
    printf pattern per column for the display. Cells are not markdown here — the
    page renders this as a data grid, not as prose — so nothing is escaped.

    ``flag`` marks one numeric column below a threshold, as ``{"column": ...,
    "below": ...}``. ``filter_column`` puts a keyword box above the table that
    keeps only the rows whose value in that column contains what was typed.
    """
    return {
        "kind": "table",
        "sort": True,
        "headers": [str(h) for h in headers],
        "rows": [[_scalar(cell) for cell in row] for row in rows],
        "formats": list(formats) if formats else [None] * len(headers),
        "flag": flag or {},
        "note": note,
        "filter": str(filter_column),
    }


def facts(pairs: Sequence[Sequence[Any]]) -> Block:
    """A two-column key/value table, for the handful of numbers that name a result."""
    return table(["", ""], [[bold(str(k)), v] for k, v in pairs])


def figure(spec: dict[str, Any] | None, caption: str = "") -> Block | None:
    """A plotly figure, already reduced to JSON. ``None`` in, ``None`` out."""
    if spec is None:
        return None
    return {"kind": "figure", "spec": spec, "caption": caption}


def image(src: str, alt: str = "") -> Block:
    """An image given as a data URI."""
    return {"kind": "image", "src": src, "alt": alt}


def code(text: str, label: str = "", language: str = "text") -> Block:
    """Verbatim text: an estimator's own output, a config dump."""
    return {"kind": "code", "text": text, "label": label, "language": language}


def collapsed(label: str, children: Sequence[Block | None]) -> Block | None:
    """Reference material, present but out of the way. Closed by default."""
    kept = [block for block in children if block]
    if not kept:
        return None
    return {"kind": "collapsed", "label": label, "blocks": kept}


def download(label: str, filename: str, text: str, note: str = "") -> Block | None:
    """A file the reader can save, with the same bytes the package holds."""
    if not text:
        return None
    return {
        "kind": "download",
        "label": label,
        "filename": filename,
        "text": text,
        "note": note,
    }


def deferred(label: str, children: Sequence[Block | None], hint: str = "") -> Block | None:
    """Heavy content behind an expander *and* a toggle, so it is drawn on request."""
    kept = [block for block in children if block]
    if not kept:
        return None
    return {"kind": "deferred", "label": label, "hint": hint, "blocks": kept}


def chooser(label: str, options: Sequence[dict[str, Any]]) -> Block | None:
    """One set of blocks out of many. Only the chosen one is rendered."""
    kept = [
        {"label": str(option["label"]), "blocks": [b for b in option["blocks"] if b]}
        for option in options
        if any(option["blocks"])
    ]
    if not kept:
        return None
    return {"kind": "chooser", "label": label, "options": kept}


def csv_text(rows: Sequence[dict[str, Any]] | pd.DataFrame) -> str:
    """Rows as CSV text, so a download in the page is the artefact on disk."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    return frame.to_csv(index=False)


def clean(blocks: Sequence[Block | None]) -> list[Block]:
    """Drop the blocks that decided there was nothing to draw."""
    return [block for block in blocks if block]


# ---------------------------------------------------------------------------
# Markdown-safe text
# ---------------------------------------------------------------------------
def esc(text: Any) -> str:
    """Neutralise markdown in a value that came out of the data.

    A column called ``rate_*`` or ``A|B`` has to survive into a table cell as
    itself rather than as emphasis or as a new column.
    """
    out = str(text)
    for char in ("\\", "`", "*", "_", "[", "]", "|", "$", "<"):
        out = out.replace(char, "\\" + char)
    return out.replace("\n", " ")


def bold(text: str) -> str:
    return f"**{text}**"


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------
def num(value: Any, digits: int = 4) -> str:
    """Format a number for a table cell, or "n/a" when there is nothing to show."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if not np.isfinite(value):
        return "n/a"
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))  # counts and level tallies, not measurements
    if abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.{digits}g}"
    return f"{value:.{digits}f}"


def frame(df: pd.DataFrame, digits: int = 4, label: Any = None) -> Block:
    """A DataFrame as a table block, numbers formatted for reading."""
    display = df.copy()
    for column in display.columns:
        if display[column].dtype.kind in "fc":
            display[column] = display[column].map(lambda v: num(v, digits))
        else:
            display[column] = display[column].map(esc)
    headers = [esc(label(c) if label else c) for c in display.columns]
    return table(headers, display.to_numpy().tolist())
