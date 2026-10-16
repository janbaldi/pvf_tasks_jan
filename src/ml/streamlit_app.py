"""Render the report payload. Computes nothing.

This is the only module that runs in the browser, under Pyodide, so it may
import the standard library, ``streamlit`` and ``plotly`` and nothing else —
none of the rest of this package exists there. It reads ``payload.json`` from
the virtual file system, which :mod:`ml.stlite` mounts from the JSON embedded
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
import streamlit.components.v1 as components

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
.batch_table-nav { display: flex; flex-direction: column; gap: 0.15rem; margin-top: 0.5rem; }
.batch_table-nav a {
  display: block; padding: 0.4rem 0.7rem; border-radius: 8px;
  text-decoration: none; color: inherit; font-size: 0.92rem; line-height: 1.3;
  border: 1px solid transparent; transition: background 120ms, border-color 120ms;
}
.batch_table-nav a:hover {
  background: color-mix(in srgb, currentColor 10%, transparent);
  border-color: color-mix(in srgb, currentColor 22%, transparent);
}
.batch_table-nav a:active { background: color-mix(in srgb, currentColor 18%, transparent); }
.batch_table-lead { font-size: 1.02rem; opacity: 0.85; margin-bottom: 0.2rem; }
.batch_table-sub { font-size: 0.82rem; opacity: 0.6; }
.batch_table-flag { color: #c0392b; font-weight: 600; }
.batch_table-caption { font-size: 0.8rem; opacity: 0.65; margin-top: -0.4rem; }
.batch_table-scroll { overflow-x: auto; max-width: 100%; margin-bottom: 0.9rem; }
.batch_table-scroll table { border-collapse: collapse; font-size: 0.86rem; }
.batch_table-scroll th, .batch_table-scroll td {
  padding: 0.3rem 0.6rem; text-align: left; vertical-align: top;
  border-bottom: 1px solid color-mix(in srgb, currentColor 15%, transparent);
  white-space: nowrap;
}
.batch_table-scroll th { font-weight: 600; opacity: 0.75; }
.batch_table-scroll a { color: inherit; }
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
    escaped = _RED.sub(r'<span class="batch_table-flag">\1</span>', escaped)
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
            f'<p class="batch_table-caption">{cell_html(block["note"])}</p>',
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
        f'<div class="batch_table-scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    if block.get("note"):
        st.markdown(
            f'<p class="batch_table-caption">{cell_html(block["note"])}</p>',
            unsafe_allow_html=True,
        )


# The pattern strip: a fixed size per batch, so a long history scrolls sideways
# instead of being squeezed to the page. Drawn on a canvas in its own frame,
# which keeps thousands of cells cheap and the hover immediate.
CELL, ROW, LABELS = 5, 16, 280
STRIP = """
<style>
  body { margin: 0; font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #7a828c; }
  #wrap { display: flex; }
  #names { flex: none; width: %(labels)dpx; }
  #names div { height: %(row)dpx; line-height: %(row)dpx; margin-bottom: 2px; padding-right: 8px;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: right; }
  #scroll { overflow-x: auto; flex: 1; }
  canvas { display: block; }
  #tip { position: fixed; display: none; pointer-events: none; padding: 4px 7px; border-radius: 4px;
         background: #1f2328; color: #fff; white-space: nowrap; font-size: 12px; }
  #legend { margin: 0 0 6px %(labels)dpx; }
  #legend b { display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 10px;
              vertical-align: -1px; border-radius: 2px; }
</style>
<div id="legend"><b style="background:#c3c9d0;margin-left:0"></b>measured
  <b style="background:#e8692c"></b>missing &nbsp;·&nbsp; one cell per batch, newest on the right;
  scroll left to go back in time
</div>
<div id="wrap"><div id="names"></div><div id="scroll"><canvas id="c"></canvas></div></div>
<div id="tip"></div>
<script>
const D = %(data)s, W = %(cell)d, H = %(row)d, GAP = 2, AXIS = 22;
const n = D.batches.length, step = W + 1, ratio = window.devicePixelRatio || 1;
const canvas = document.getElementById("c"), ctx = canvas.getContext("2d");
const width = n * step, height = D.rows.length * (H + GAP) + AXIS;
canvas.width = width * ratio; canvas.height = height * ratio;
canvas.style.width = width + "px"; canvas.style.height = height + "px";
ctx.scale(ratio, ratio);
const names = document.getElementById("names");
D.rows.forEach((row, r) => {
  const label = document.createElement("div");
  label.textContent = row.name; label.title = row.name; names.appendChild(label);
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = row.bits[i] === "1" ? "#e8692c" : "#c3c9d0";
    ctx.fillRect(i * step, r * (H + GAP), W, H);
  }
});
// A date under the strip wherever the month changes, with room between labels.
ctx.fillStyle = "#7a828c"; ctx.font = "11px system-ui, sans-serif";
let last = -1e9, month = "";
D.dates.forEach((date, i) => {
  const m = (date || "").slice(0, 7);
  if (m && m !== month && i * step - last > 70) {
    ctx.fillRect(i * step, D.rows.length * (H + GAP), 1, 4);
    ctx.fillText(m, i * step + 2, D.rows.length * (H + GAP) + 15);
    last = i * step;
  }
  if (m) month = m;
});
const tip = document.getElementById("tip");
canvas.addEventListener("mousemove", (e) => {
  const i = Math.floor(e.offsetX / step), r = Math.floor(e.offsetY / (H + GAP));
  if (i < 0 || i >= n || r < 0 || r >= D.rows.length) { tip.style.display = "none"; return; }
  const row = D.rows[r];
  tip.textContent = D.batches[i] + (D.dates[i] ? " · " + D.dates[i] : "") + " · " + row.name +
    ": " + (row.bits[i] === "1" ? "missing" : "measured");
  tip.style.display = "block";
  const left = e.clientX + 12;
  tip.style.left = Math.min(left, window.innerWidth - tip.offsetWidth - 4) + "px";
  tip.style.top = (e.clientY + 14) + "px";
});
canvas.addEventListener("mouseleave", () => { tip.style.display = "none"; });
// Open on the most recent batches.
const scroller = document.getElementById("scroll");
scroller.scrollLeft = scroller.scrollWidth;
</script>
"""


def _pick_table(headers, rows, formats, mode: str, key: str):
    """A data grid whose rows the reader ticks; the frame and the ticked row numbers."""
    frame = pd.DataFrame(rows, columns=headers)
    config = {}
    for header, fmt in zip(headers, formats):
        if fmt:
            config[header] = st.column_config.NumberColumn(header, format=fmt)
            frame[header] = pd.to_numeric(frame[header], errors="coerce")
    event = st.dataframe(
        frame,
        column_config=config,
        hide_index=True,
        width="stretch",
        row_height=ROW_HEIGHT,
        height=ROW_HEIGHT * (min(len(frame), MAX_GRID_ROWS) + 1) + GUTTER,
        on_select="rerun",
        selection_mode=mode,
        key=key,
    )
    return frame, event.selection.rows


def render_missing_viewer(block: dict, key: str) -> None:
    """The strip on top; below it the groups (one to draw) and the parameters (any to add)."""
    labels = [site["label"] for site in block["sites"]]
    chosen = st.radio("Site", labels, horizontal=True, key=f"site{key}")
    site = block["sites"][labels.index(chosen)]
    top = st.container()

    picked: list[str] = []
    with st.expander(
        f"Groups of parameters missing together ({len(site['groups'])})", expanded=True
    ):
        if site["groups"]:
            groups, ticked = _pick_table(
                site["group_headers"],
                site["group_rows"],
                site["group_formats"],
                "single-row",
                f"groups{key}-{chosen}",
            )
            picked += [m for i in ticked for m in site["groups"][groups.iloc[i]["Group"]]]
            st.markdown(
                '<p class="batch_table-caption">Tick a group to draw all its parameters.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No group of parameters goes missing together at this site.")
    with st.expander(f"Parameters ({len(site['rows'])})", expanded=False):
        frame, ticked = _pick_table(
            site["headers"], site["rows"], site["formats"], "multi-row", f"rows{key}-{chosen}"
        )
        picked += [frame.iloc[i]["Parameter"] for i in ticked]
        st.markdown(
            '<p class="batch_table-caption">Tick parameters to add them to the plot. Sort by a '
            "column header; search with the magnifier at the table's top right.</p>",
            unsafe_allow_html=True,
        )
        st.download_button(
            f"Download {site['csv_name']}",
            data=site["csv"],
            file_name=site["csv_name"],
            mime="text/csv",
            key=f"dl{key}-{chosen}",
        )

    picked = list(dict.fromkeys(picked))
    n = len(site["batches"])
    with top:
        if not picked:
            st.info("Tick a group or parameters in the tables below to draw them here.")
            return
        data = {
            "batches": site["batches"],
            "dates": site["dates"],
            # A parameter with no blank at this site has no pattern stored: all measured.
            "rows": [{"name": p, "bits": site["patterns"].get(p, "0" * n)} for p in picked],
        }
        page = STRIP % {
            "data": json.dumps(data).replace("<", "\\u003c"),
            "cell": CELL,
            "row": ROW,
            "labels": LABELS,
        }
        components.html(page, height=len(picked) * (ROW + 2) + 22 + 40, scrolling=False)


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
                f'<p class="batch_table-caption">{html.escape(block["caption"])}</p>',
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
                f'<p class="batch_table-caption">{cell_html(block["note"])}</p>',
                unsafe_allow_html=True,
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
    elif kind == "missing_viewer":
        render_missing_viewer(block, key)
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


@st.dialog((PAYLOAD.get("help") or {}).get("title", "Help"), width="large")
def show_help() -> None:
    """The report's method, out of the way of the report itself."""
    for i, block in enumerate(PAYLOAD["help"]["blocks"]):
        render_block(block, f"help-{i}")


st.set_page_config(page_title=PAYLOAD["title"], layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### {PAYLOAD['title']}")
    st.caption(PAYLOAD["subtitle"])
    links = "".join(
        f'<a href="#{section["id"]}">{html.escape(section["title"])}</a>'
        for section in PAYLOAD["sections"]
    )
    st.markdown(f'<nav class="batch_table-nav">{links}</nav>', unsafe_allow_html=True)
    if PAYLOAD.get("help"):
        st.divider()
        if st.button(f"Help: {PAYLOAD['help']['title']}", icon=":material/help:", key="help"):
            show_help()

st.title(PAYLOAD["title"])
if PAYLOAD.get("lead"):
    st.markdown(
        f'<p class="batch_table-lead">{html.escape(PAYLOAD["lead"])}</p>', unsafe_allow_html=True
    )
st.markdown(
    f'<p class="batch_table-sub">{html.escape(PAYLOAD["subtitle"])}</p>', unsafe_allow_html=True
)

for section in PAYLOAD["sections"]:
    st.header(section["title"], anchor=section["id"], divider="gray")
    for i, block in enumerate(section["blocks"]):
        render_block(block, f"{section['id']}-{i}")
