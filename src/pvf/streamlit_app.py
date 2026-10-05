"""Render the report payload. Computes nothing.

This is the only module that runs in the browser, under Pyodide, so it may
import the standard library, ``streamlit`` and ``plotly`` and nothing else —
none of the rest of this package exists there. It reads ``payload.json`` from
the virtual file system, which :mod:`pvf.stlite` mounts from the JSON embedded
in the page.

Every number was decided before this file ran. All it does is walk the block
tree and hand each block to Streamlit.
"""

import html
import json
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# pandas is here for one reason: a sortable table is a Streamlit data grid, and
# a data grid wants a DataFrame. Streamlit installs it into the runtime itself,
# so it is always present alongside streamlit and plotly.

PAYLOAD = json.loads(Path("payload.json").read_text(encoding="utf-8"))

# Styling touches only this module's own markup. Streamlit's internal class names
# move between runtime versions, and a report that renders differently after an
# upgrade is not a record of anything. Colours are expressed against the text
# colour the viewer's theme already set, so light and dark both work.
CSS = """
<style>
.pvf-nav { display: flex; flex-direction: column; gap: 0.15rem; margin-top: 0.5rem; }
.pvf-nav a {
  display: block; padding: 0.4rem 0.7rem; border-radius: 8px;
  text-decoration: none; color: inherit; font-size: 0.92rem; line-height: 1.3;
  border: 1px solid transparent; transition: background 120ms, border-color 120ms;
}
.pvf-nav a:hover {
  background: color-mix(in srgb, currentColor 10%, transparent);
  border-color: color-mix(in srgb, currentColor 22%, transparent);
}
.pvf-nav a:active { background: color-mix(in srgb, currentColor 18%, transparent); }
.pvf-lead { font-size: 1.02rem; opacity: 0.85; margin-bottom: 0.2rem; }
.pvf-sub { font-size: 0.82rem; opacity: 0.6; }
.pvf-flag { color: #c0392b; font-weight: 600; }
.pvf-caption { font-size: 0.8rem; opacity: 0.65; margin-top: -0.4rem; }
.pvf-scroll { overflow-x: auto; max-width: 100%; margin-bottom: 0.9rem; }
.pvf-scroll table { border-collapse: collapse; font-size: 0.86rem; }
.pvf-scroll th, .pvf-scroll td {
  padding: 0.3rem 0.6rem; text-align: left; vertical-align: top;
  border-bottom: 1px solid color-mix(in srgb, currentColor 15%, transparent);
  white-space: nowrap;
}
.pvf-scroll th { font-weight: 600; opacity: 0.75; }
.pvf-scroll a { color: inherit; }
</style>
"""

# The data grid's own row height, passed to it rather than assumed, so the height
# computed here and the height it draws are the same number. GUTTER is the slack
# for the border and for the horizontal scrollbar, which is drawn inside the
# element and would otherwise sit on top of the last row.
ROW_HEIGHT = 35
GUTTER = 20
MAX_GRID_ROWS = 25

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_RED = re.compile(r":red\[(.+?)\]")
_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")


def cell_html(text: str) -> str:
    """One table cell: markdown in, safe HTML out.

    Cells arrive as the small markdown dialect the report writes — bold, inline
    code and a red span — with every character that came out of the data
    backslash-escaped. Those escapes are parked before the patterns run, so a
    parameter literally called ``Lact / E6 *`` can never turn into markup, and
    restored as themselves afterwards.
    """
    parked: list[str] = []

    def park(match: re.Match) -> str:
        parked.append(match.group(1))
        return f"\x00{len(parked) - 1}\x00"

    # Park first, escape second: a parked "<" has to be escaped as itself on the
    # way out, not as whatever escaping turned it into on the way in.
    escaped = html.escape(re.sub(r"\\(.)", park, text), quote=False)
    escaped = _RED.sub(r'<span class="pvf-flag">\1</span>', escaped)
    escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
    escaped = _CODE.sub(r"<code>\1</code>", escaped)
    return _PLACEHOLDER.sub(lambda m: html.escape(parked[int(m.group(1))], quote=False), escaped)


def render_sortable(block: dict, key: str) -> None:
    """A table as a data grid: click a header to reorder, type a word to filter.

    The values arrive typed, so numeric columns sort as numbers rather than as
    text. ``formats`` puts the display back where it was before the numbers were
    kept raw.
    """
    frame = pd.DataFrame(block["rows"], columns=block["headers"])

    column = block.get("filter")
    if column in frame.columns:
        keyword = st.text_input(
            f"Filter by {column.lower()}",
            key=f"filter{key}",
            placeholder=f"a word in the {column} column",
        )
        if keyword:
            # A word, not a pattern: a parameter name with "(%)" in it must match
            # itself rather than blow up as a regular expression.
            frame = frame[
                frame[column].astype(str).str.contains(keyword, case=False, na=False, regex=False)
            ].reset_index(drop=True)
            if frame.empty:
                st.caption(f"Nothing matches “{keyword}”.")
                return

    # A column with a number format is a number column: coerce it, so a cell
    # with nothing in it is blank rather than the word "None".
    config = {}
    for header, fmt in zip(block["headers"], block["formats"]):
        if not fmt:
            continue
        config[header] = st.column_config.NumberColumn(header, format=fmt)
        if header in frame.columns:
            frame[header] = pd.to_numeric(frame[header], errors="coerce")

    data = frame
    flag = block.get("flag") or {}
    if flag.get("column") in frame.columns:
        data = frame.style.map(
            lambda v: (
                "color: #c0392b; font-weight: 600"
                if isinstance(v, (int, float)) and v == v and v < flag["below"]
                else ""
            ),
            subset=[flag["column"]],
        )

    # Every row on screen up to a limit, then the grid scrolls on its own rather
    # than pushing the rest of the section off the page.
    shown = min(len(frame), MAX_GRID_ROWS)
    st.dataframe(
        data,
        column_config=config,
        hide_index=True,
        width="stretch",
        row_height=ROW_HEIGHT,
        height=ROW_HEIGHT * (shown + 1) + GUTTER,
    )
    if block.get("note"):
        st.markdown(
            f'<p class="pvf-caption">{cell_html(block["note"])}</p>',
            unsafe_allow_html=True,
        )


def render_table(block: dict, key: str) -> None:
    """A table in its own horizontally scrolling box, so the page never scrolls."""
    if block.get("sort"):
        render_sortable(block, key)
        return

    head = "".join(f"<th>{cell_html(h)}</th>" for h in block["headers"])
    body = "".join(
        "<tr>" + "".join(f"<td>{cell_html(c)}</td>" for c in row) + "</tr>" for row in block["rows"]
    )
    st.markdown(
        f'<div class="pvf-scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    if block.get("note"):
        st.markdown(
            f'<p class="pvf-caption">{cell_html(block["note"])}</p>',
            unsafe_allow_html=True,
        )


def render_block(block: dict, key: str) -> None:
    kind = block["kind"]
    if kind == "md":
        st.markdown(block["text"])
    elif kind == "table":
        render_table(block, key)
    elif kind == "figure":
        st.plotly_chart(go.Figure(block["spec"]), width="stretch", key=f"fig{key}")
        if block.get("caption"):
            st.markdown(
                f'<p class="pvf-caption">{html.escape(block["caption"])}</p>',
                unsafe_allow_html=True,
            )
    elif kind == "image":
        st.markdown(
            f'<img src="{block["src"]}" alt="{html.escape(block.get("alt", ""))}" '
            'style="max-width:100%">',
            unsafe_allow_html=True,
        )
    elif kind == "code":
        if block.get("label"):
            st.markdown(f"**{block['label']}**")
        st.code(block["text"], language=block.get("language") or "text")
    elif kind == "download":
        st.download_button(
            block["label"],
            data=block["text"],
            file_name=block["filename"],
            mime="text/csv" if block["filename"].endswith(".csv") else "application/json",
            key=f"dl{key}",
        )
        if block.get("note"):
            st.markdown(
                f'<p class="pvf-caption">{cell_html(block["note"])}</p>', unsafe_allow_html=True
            )
    elif kind == "collapsed":
        with st.expander(block["label"], expanded=False):
            for i, child in enumerate(block["blocks"]):
                render_block(child, f"{key}-{i}")
    elif kind == "deferred":
        # An expander runs its contents whether or not it is open, so the figures
        # inside one wait behind a toggle instead: nothing is drawn until asked for.
        with st.expander(block["label"], expanded=False):
            if st.toggle(block.get("hint") or "Show", key=f"show{key}"):
                for i, child in enumerate(block["blocks"]):
                    render_block(child, f"{key}-{i}")
            else:
                st.caption("Nothing here is drawn until you ask for it.")
    elif kind == "chooser":
        labels = [option["label"] for option in block["options"]]
        chosen = st.selectbox(block["label"], labels, key=f"pick{key}")
        option = next(o for o in block["options"] if o["label"] == chosen)
        for i, child in enumerate(option["blocks"]):
            render_block(child, f"{key}-{labels.index(chosen)}-{i}")
    else:
        # A block kind the builder emits and this renderer forgot is a silent
        # hole in the report, so say so on the page rather than in a console
        # nobody reads in a browser.
        st.warning(f"Unrenderable block kind: {kind}")


st.set_page_config(page_title=PAYLOAD["title"], layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### {PAYLOAD['title']}")
    st.caption(PAYLOAD["subtitle"])
    links = "".join(
        f'<a href="#{section["id"]}">{html.escape(section["title"])}</a>'
        for section in PAYLOAD["sections"]
    )
    st.markdown(f'<nav class="pvf-nav">{links}</nav>', unsafe_allow_html=True)

st.title(PAYLOAD["title"])
if PAYLOAD.get("lead"):
    st.markdown(f'<p class="pvf-lead">{html.escape(PAYLOAD["lead"])}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="pvf-sub">{html.escape(PAYLOAD["subtitle"])}</p>', unsafe_allow_html=True)

for section in PAYLOAD["sections"]:
    st.header(section["title"], anchor=section["id"], divider="gray")
    for i, block in enumerate(section["blocks"]):
        render_block(block, f"{section['id']}-{i}")
