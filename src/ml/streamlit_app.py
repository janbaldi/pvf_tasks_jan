"""Render the report payload. Computes nothing.

This is the only module that runs in the browser, under Pyodide, so it may
import the standard library, ``streamlit`` and ``plotly`` and nothing else —
none of the rest of this package exists there. It reads ``payload.json`` from
the virtual file system, which :mod:`ml.stlite` mounts from the JSON embedded
in the page.

Every number was decided before this file ran. All it does is walk the block
tree and hand each block to Streamlit.
"""

import datetime
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

# The look is the CAR-T manufacturing dashboard's (its dashboard-style skill): the
# theme and fonts come from ml.stlite, the density rules below are the dashboard's
# own, and the rest styles only this module's markup.
CSS = """
<style>
[data-testid="stHeader"] {background: transparent; pointer-events: none;}
[data-testid="stHeader"] button, [data-testid="stHeader"] a {pointer-events: auto;}
[data-testid="stMainBlockContainer"] {padding: 0.75rem 3rem 1.5rem;}
[data-testid="stSidebarHeader"] {height: 1.75rem; margin-bottom: 0; padding-top: 0.25rem;}
[data-testid="stSidebarUserContent"] {padding: 0 1rem 1.5rem;}
[data-testid="stVerticalBlock"] {gap: 0.6rem;}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {gap: 0.45rem;}
[data-testid="stWidgetLabel"] {min-height: 1.1rem; margin-bottom: 0.1rem;}
[data-testid="stWidgetLabel"] p {font-size: 0.85rem;}
[data-testid^="stBaseButton"] {min-height: 1.9rem; padding: 0.1rem 0.7rem;}

.batch_table-name {font-size: 1rem; font-weight: 600; line-height: 1.35; margin: .3rem 0 .2rem;}
.batch_table-title {font-size: 1.25rem; font-weight: 600; line-height: 1.4; margin: .4rem 0 0;}
.batch_table-lead {font-size: .9rem; line-height: 1.45; margin: 0;}
.batch_table-sub {font-size: .8rem; opacity: .65; margin: 0;}
.batch_table-flag {color: #0047ab; font-weight: 600;}
.batch_table-caption {font-size: .8rem; opacity: .65; margin-top: -.4rem;}
.batch_table-scroll {overflow-x: auto; max-width: 100%; margin-bottom: .6rem;}
.batch_table-scroll table {border-collapse: collapse; font-size: .85rem; border: none;}
.batch_table-scroll th, .batch_table-scroll td {
  padding: .25rem .6rem; text-align: left; vertical-align: top; border: none;
  border-bottom: 1px solid #dee2e6; white-space: nowrap;
}
.batch_table-scroll th {font-weight: 600; font-size: .8rem; opacity: .75;
                        border-bottom-color: #c3cad2;}
.batch_table-scroll a {color: inherit;}

/* Contents: the dashboard help page's docs-style navigation list. */
.toc-head {font-size: .75rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
           opacity: .6; margin: .6rem 0 .3rem .6rem;}
.toc {display: flex; flex-direction: column; gap: 1px;
       border-left: 1px solid rgba(44, 62, 80, .15);}
.toc a {display: block; text-decoration: none !important; color: inherit !important;
        padding: .22rem .6rem; border-left: 2px solid transparent; margin-left: -1px;
        border-radius: 0 .25rem .25rem 0; line-height: 1.3; font-size: .85rem;}
.toc a:hover {background: rgba(0, 102, 204, .08); color: #0066cc !important;
              border-left-color: #0066cc;}
.toc a.toc-l2 {font-weight: 600; margin-top: .45rem;}
.toc a.toc-l2:first-child {margin-top: 0;}
.toc a.toc-l3 {font-size: .8rem; padding-left: 1.3rem; opacity: .8;}
.toc a.toc-l3:hover {opacity: 1;}
.toc-none {font-size: .8rem; opacity: .6; padding-left: .6rem;}
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
                "color: #0047ab; font-weight: 600"
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
  body { margin: 0; font: 12px "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
         color: #5d6d7e; }
  #wrap { display: flex; }
  #names { flex: none; width: %(labels)dpx; }
  #names div { height: %(row)dpx; line-height: %(row)dpx; margin-bottom: 2px; padding-right: 8px;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: right; }
  #scroll { overflow-x: auto; flex: 1; }
  canvas { display: block; }
  #tip { position: fixed; display: none; pointer-events: none; padding: 4px 7px; border-radius: 4px;
         background: #2c3e50; color: #fff; white-space: nowrap; font-size: 12px; }
  #legend { margin: 0 0 6px %(labels)dpx; }
  #legend b { display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 10px;
              vertical-align: -1px; border-radius: 2px; }
</style>
<div id="legend"><b style="background:#d3d2cc;margin-left:0"></b>measured
  <b style="background:#0047ab"></b>missing
  <b style="background:#2e8b57"></b>missing, batch not in the statistics
  &nbsp;·&nbsp; one cell per batch, newest on the right; scroll left to go back in time;
  ⊘ marks a parameter screened out at this site
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
    const counted = !D.analysed || D.analysed[i] === "1";
    ctx.fillStyle = row.bits[i] !== "1" ? "#d3d2cc" : counted ? "#0047ab" : "#2e8b57";
    ctx.fillRect(i * step, r * (H + GAP), W, H);
  }
});
// A date under the strip wherever the month changes, with room between labels.
ctx.fillStyle = "#5d6d7e"; ctx.font = "11px system-ui, sans-serif";
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
    ": " + (row.bits[i] === "1" ? "missing" : "measured") +
    (D.analysed && D.analysed[i] !== "1" ? " (not in the statistics)" : "");
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


def proposal_rows(
    picked: list,
    group_of: dict,
    screened: dict,
    site: str,
    action: str,
    reason: str,
    rationale: str,
    review_by: str,
    author: str,
) -> tuple:
    """The proposal rows for the ticked parameters, and the ones that cannot be in it.

    One row per parameter, with its group as context, in the columns
    ``ml screening add`` reads. Only a parameter screened out at this site can
    be reinstated; the others are returned as skipped.
    """
    rows, skipped = [], []
    for parameter in picked:
        if action == "reinstate" and parameter not in screened:
            skipped.append(parameter)
            continue
        rows.append(
            {
                "Action": action,
                "Parameter": parameter,
                "Group": group_of.get(parameter, ""),
                "Site": site,
                "Reason": reason,
                "Rationale": rationale.strip(),
                "Review by": review_by if action == "exclude" else "",
                "Author": author.strip(),
            }
        )
    return rows, skipped


def render_proposal(form: dict, site: dict, picked: list, key: str) -> None:
    """Turn the ticked parameters into a proposal.csv for `ml screening add`."""
    name, group = site["headers"].index("Parameter"), site["headers"].index("Group")
    group_of = {row[name]: row[group] or "" for row in site["rows"]}
    with st.expander(f"Propose a screening decision ({len(picked)} ticked)", expanded=False):
        if not picked:
            st.caption(
                "Tick a group, parameters, or screened parameters above; they are what the "
                "proposal is about."
            )
            return
        left, right = st.columns(2)
        action = left.radio("Action", form["actions"], horizontal=True, key=f"act{key}")
        reason = right.selectbox("Reason", form["reasons"], key=f"why{key}")
        rationale = st.text_area("Rationale: why, in words", key=f"text{key}")
        left, right = st.columns(2)
        review = left.date_input(
            "Review by",
            datetime.date.today() + datetime.timedelta(days=form["review_days"]),
            key=f"when{key}",
            disabled=action != "exclude",
        )
        author = right.text_input("Author", key=f"who{key}")
        rows, skipped = proposal_rows(
            picked,
            group_of,
            site["screened"],
            site["label"],
            action,
            reason,
            rationale,
            review.isoformat(),
            author,
        )
        if skipped:
            st.warning(
                f"Not screened out at this site, so nothing to reinstate: {', '.join(skipped)}"
            )
        missing = [
            name
            for name, value in (("a rationale", rationale), ("an author", author))
            if not value.strip()
        ]
        if missing:
            st.info(f"Fill in {' and '.join(missing)} to write the proposal.")
            return
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=form["columns"])
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button(
            f"Download proposal.csv ({len(rows)} rows)",
            data=frame.to_csv(index=False),
            file_name="proposal.csv",
            mime="text/csv",
            key=f"prop{key}",
        )
        st.markdown(
            '<p class="batch_table-caption">Then: <code>uv run ml screening add proposal.csv'
            "</code>. It checks every row against the latest build before it adds any.</p>",
            unsafe_allow_html=True,
        )


def render_missing_viewer(block: dict, key: str) -> None:
    """The strip on top; below it the groups (one to draw) and the parameters (any to add)."""
    labels = [site["label"] for site in block["sites"]]
    chosen = st.radio("Site", labels, horizontal=True, key=f"site{key}")
    site = block["sites"][labels.index(chosen)]
    screened = site.get("screened") or {}
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
                '<p class="batch_table-caption">Tick a group to draw all its parameters. '
                "<i>Screened members</i> counts the ones already screened out here.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No group of parameters goes missing together at this site.")
    with st.expander(f"Screened out at this site ({len(screened)})", expanded=bool(screened)):
        if screened:
            headers = ["Parameter", *next(iter(screened.values()))]
            rows = [[name, *decision.values()] for name, decision in screened.items()]
            frame, ticked = _pick_table(
                headers, rows, [None] * len(headers), "multi-row", f"screened{key}-{chosen}"
            )
            picked += [frame.iloc[i]["Parameter"] for i in ticked]
            st.markdown(
                '<p class="batch_table-caption">The screening register\'s standing exclusions '
                "at this site. Every task leaves these out. Tick them to draw them, or to "
                "reinstate them.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Nothing is screened out at this site.")
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
    if block.get("proposal"):
        render_proposal(block["proposal"], site, picked, f"{key}-{chosen}")
    n = len(site["batches"])
    with top:
        if not picked:
            st.info("Tick a group or parameters in the tables below to draw them here.")
            return
        data = {
            "batches": site["batches"],
            "dates": site["dates"],
            "analysed": site.get("in_statistics") or "",
            # A parameter with no blank at this site has no pattern stored: all measured.
            "rows": [
                {
                    "name": ("⊘ " if p in screened else "") + p,
                    "bits": site["patterns"].get(p, "0" * n),
                }
                for p in picked
            ],
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


_HEADING = re.compile(r"^(#{2,4}) (.+)$")


def _anchor(title: str, used: set[str]) -> str:
    anchor = base = "help-" + (re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section")
    n = 1
    while anchor in used:
        n += 1
        anchor = f"{base}-{n}"
    used.add(anchor)
    return anchor


def help_sections(blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split the help blocks at their markdown headings: the intro, and the sections.

    A section is ``{level, title, anchor, parent, blocks}``; ``level`` is 2 for the
    shallowest heading the help uses and 3 for anything below it.
    """
    found = [
        len(m.group(1))
        for b in blocks
        if b["kind"] == "md"
        for m in map(_HEADING.match, b["text"].splitlines())
        if m
    ]
    top = min(found, default=2)
    intro: list[dict] = []
    sections: list[dict] = []
    parent, used = "", set()

    def add(block: dict) -> None:
        (sections[-1]["blocks"] if sections else intro).append(block)

    for block in blocks:
        if block["kind"] != "md":
            add(block)
            continue
        lines: list[str] = []
        for line in block["text"].splitlines():
            match = _HEADING.match(line)
            if not match:
                lines.append(line)
                continue
            if "\n".join(lines).strip():
                add({"kind": "md", "text": "\n".join(lines).strip()})
            lines = []
            level, title = (2 if len(match.group(1)) == top else 3), match.group(2).strip()
            parent = title if level == 2 else parent
            sections.append(
                {
                    "level": level,
                    "title": title,
                    "anchor": _anchor(title, used),
                    "parent": parent if level == 3 else "",
                    "blocks": [],
                }
            )
        if "\n".join(lines).strip():
            add({"kind": "md", "text": "\n".join(lines).strip()})
    return intro, sections


def _words(query: str) -> list[str]:
    return query.lower().split()


def _text_of(block: dict) -> str:
    if block["kind"] == "md":
        return block["text"]
    if block["kind"] == "table":
        return (
            " ".join(map(str, block["headers"]))
            + " "
            + " ".join(str(c) for row in block["rows"] for c in row)
        )
    return ""


def highlight(markdown: str, query: str) -> str:
    """Wrap the query words in <mark>, leaving `code` spans alone."""
    found = sorted(set(_words(query)), key=len, reverse=True)
    if not found:
        return markdown
    pattern = re.compile("|".join(re.escape(w) for w in found), re.IGNORECASE)
    parts = markdown.split("`")
    for i in range(0, len(parts), 2):  # even parts are outside code spans
        parts[i] = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", parts[i])
    return "`".join(parts)


def contents(links: list[tuple[int, str, str]], head: str) -> None:
    """The sidebar's docs-style table of contents: (level, title, anchor) per entry."""
    items = (
        "".join(
            f"<a class='toc-l{level}' href='#{anchor}'>{html.escape(title)}</a>"
            for level, title, anchor in links
        )
        or "<span class='toc-none'>Nothing matches.</span>"
    )
    st.markdown(
        f"<div class='toc-head'>{head}</div><nav class='toc'>{items}</nav>",
        unsafe_allow_html=True,
    )


def title_block(title: str) -> None:
    st.markdown(
        f'<div class="batch_table-title">{html.escape(title)}</div>', unsafe_allow_html=True
    )


def help_page() -> None:
    """The report's method on a page of its own, searchable, as the dashboard's help is."""
    help_ = PAYLOAD["help"]
    intro, sections = help_sections(help_["blocks"])

    query = st.text_input(
        "Search the help",
        placeholder="Type words to find",
        key="help_search",
        icon=":material/search:",
    )
    words = _words(query)
    hits = [
        s
        for s in sections
        if all(
            w in " ".join([s["parent"], s["title"], *map(_text_of, s["blocks"])]).lower()
            for w in words
        )
    ]

    with st.sidebar:
        st.button(
            "Back to the report",
            icon=":material/arrow_back:",
            on_click=st.session_state.update,
            kwargs={"page": "report"},
            width="stretch",
        )
        count = f" · {len(hits)} matching" if words else ""
        contents([(s["level"], s["title"], s["anchor"]) for s in hits], f"Contents{count}")

    if not words:
        title_block(f"Help · {help_['title']}")
        for i, block in enumerate(intro):
            render_block(block, f"help-intro-{i}")
    elif hits:
        st.caption(f"{len(hits)} of {len(sections)} sections contain every word of the search.")
    else:
        st.info("No section contains all these words. Try fewer or shorter words.")

    for n, s in enumerate(hits):
        if s["level"] == 2:
            st.header(s["title"], anchor=s["anchor"])
        else:
            if words:
                st.caption(s["parent"])
            st.subheader(s["title"], anchor=s["anchor"])
        for i, block in enumerate(s["blocks"]):
            if block["kind"] == "md":
                st.markdown(highlight(block["text"], query), unsafe_allow_html=True)
            else:
                render_block(block, f"help-{n}-{i}")


def report_page() -> None:
    with st.sidebar:
        contents([(2, s["title"], s["id"]) for s in PAYLOAD["sections"]], "Contents")
        if PAYLOAD.get("help"):
            st.divider()
            st.button(
                f"Help: {PAYLOAD['help']['title']}",
                icon=":material/help:",
                key="help",
                on_click=st.session_state.update,
                kwargs={"page": "help"},
                width="stretch",
            )

    title_block(PAYLOAD["title"])
    if PAYLOAD.get("lead"):
        st.markdown(
            f'<p class="batch_table-lead">{html.escape(PAYLOAD["lead"])}</p>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<p class="batch_table-sub">{html.escape(PAYLOAD["subtitle"])}</p>',
        unsafe_allow_html=True,
    )

    # Each section in a bordered box, the dashboard's expander look; expanders
    # themselves cannot hold the folded blocks a section may contain.
    for section in PAYLOAD["sections"]:
        with st.container(border=True):
            st.subheader(section["title"], anchor=section["id"])
            for i, block in enumerate(section["blocks"]):
                render_block(block, f"{section['id']}-{i}")


st.set_page_config(page_title=PAYLOAD["title"], layout="wide", initial_sidebar_state="expanded")
st.html(CSS)

with st.sidebar:
    st.markdown(
        f'<div class="batch_table-name">{html.escape(PAYLOAD["title"])}</div>',
        unsafe_allow_html=True,
    )
    st.caption(PAYLOAD["subtitle"])

if PAYLOAD.get("help") and st.session_state.get("page") == "help":
    help_page()
else:
    report_page()
