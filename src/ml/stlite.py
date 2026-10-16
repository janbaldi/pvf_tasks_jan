"""Wrap the payload and the page module into one HTML file.

The report is a Streamlit page that needs no Streamlit server: the file carries
its own Python runtime through stlite, which is Streamlit compiled onto Pyodide.
Open the file; there is nothing to run.

What that costs, and the page says so while it boots: the runtime is fetched
from a CDN, so a fresh report needs network access once, and it spends roughly
10 to 30 seconds starting Python before the first figure appears. A report that
must open instantly and offline cannot be built this way.

The stlite version is pinned. A report that renders differently next month is
not a record of anything.

The look is the CAR-T manufacturing dashboard's: its light clinical theme (one
blue accent), IBM Plex, and its compact type scale. The fonts are embedded in
the page rather than fetched, so the only download is the runtime itself.
"""

import base64
import html
import json
from pathlib import Path
from typing import Any

import plotly

STLITE_VERSION = "1.8.1"
# The browser installs plotly into Pyodide when the page opens. Pinned to the
# version that serialised the figures, for the same reason stlite is pinned.
PLOTLY_VERSION = plotly.__version__
CDN = f"https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build"

APP_PATH = Path(__file__).with_name("streamlit_app.py")
FONTS = Path(__file__).with_name("fonts")

# The dashboard's .streamlit/config.toml, as the options stlite passes to Streamlit.
# Light only: the colours in the page module and the figures assume it. stlite
# refuses anything but strings, numbers and booleans here, so a list goes in as
# JSON text, which Streamlit reads as it does a list given on the command line.
STREAMLIT_CONFIG = {
    "client.toolbarMode": "viewer",
    "theme.base": "light",
    "theme.baseFontSize": 14,
    "theme.headingFontSizes": json.dumps(
        ["1.9rem", "1.35rem", "1.15rem", "1rem", "0.9rem", "0.85rem"]
    ),
    "theme.primaryColor": "#0066cc",
    "theme.backgroundColor": "#fafbfc",
    "theme.secondaryBackgroundColor": "#f4f6f8",
    "theme.textColor": "#2c3e50",
    "theme.linkColor": "#0052a3",
    "theme.borderColor": "#dee2e6",
    "theme.showWidgetBorder": True,
    "theme.baseRadius": "0.25rem",
    "theme.font": "IBMPlexSans",
    "theme.headingFont": "IBMPlexSans",
    "theme.codeFont": "IBMPlexMono",
    "theme.codeBackgroundColor": "#f1f3f4",
    "theme.showSidebarBorder": True,
    "theme.sidebar.backgroundColor": "#f1f4f7",
    "theme.sidebar.secondaryBackgroundColor": "#e8ecf1",
}

# (family, weight, file in FONTS); IBM Plex is under the SIL Open Font Licence (fonts/OFL.txt).
FONT_FACES = [
    ("IBMPlexSans", 400, "IBMPlexSans-Regular.woff2"),
    ("IBMPlexSans", 500, "IBMPlexSans-Medium.woff2"),
    ("IBMPlexSans", 600, "IBMPlexSans-SemiBold.woff2"),
    ("IBMPlexSans", 700, "IBMPlexSans-Bold.woff2"),
    ("IBMPlexMono", 400, "IBMPlexMono-Regular.woff2"),
    ("IBMPlexMono", 600, "IBMPlexMono-SemiBold.woff2"),
]

PAYLOAD_ID = "report-payload"
APP_ID = "report-app"

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{cdn}/stlite.css">
<style>
{fonts}
  html, body {{ margin: 0; padding: 0; background: #fafbfc; color: #2c3e50; }}
  #boot {{
    font-family: IBMPlexSans, system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 14px; max-width: 34rem; margin: 18vh auto; padding: 0 1.5rem; line-height: 1.5;
  }}
  #boot h1 {{ font-size: 1.25rem; font-weight: 600; margin: 0 0 0.6rem; }}
  #boot p {{ margin: 0.4rem 0; opacity: 0.75; font-size: 0.9rem; }}
  #boot code {{
    font-family: IBMPlexMono, monospace; font-size: 0.85rem;
    background: #f1f3f4; padding: 0 0.2rem; border-radius: 0.25rem;
  }}
  #boot.failed h1 {{ color: #0b2a5b; }}
  #boot.failed {{ border-left: 2px solid #0066cc; padding-left: 1.2rem; }}
</style>
</head>
<body>
<div id="root">
  <div id="boot">
    <h1>{title}</h1>
    <p>Starting Python in your browser. This takes roughly 10 to 30 seconds the
       first time, and needs network access once to fetch the runtime.</p>
    <p>Every number in this report was computed before the file was written;
       nothing is being recalculated.</p>
  </div>
</div>
<script type="application/json" id="{payload_id}">{payload}</script>
<script type="application/json" id="{app_id}">{app}</script>
<script>
  // The runtime comes from a CDN. When it cannot be fetched — no network, a
  // proxy in the way, a blocked CDN — the page would otherwise sit on "starting
  // Python" for ever, so say what happened and what can be done about it.
  window.__mlStarted = false;
  window.__mlFailed = function (detail) {{
    var boot = document.getElementById("boot");
    if (!boot || boot.dataset.failed) return;
    boot.dataset.failed = "1";
    boot.className = "failed";
    boot.innerHTML =
      "<h1>This report could not start</h1>" +
      "<p>The page fetches its Python runtime from " +
      "<code>cdn.jsdelivr.net</code> the first time it is opened, and that request did " +
      "not finish.</p>" +
      "<p>What usually fixes it: connect to the network and reload; open the file in a " +
      "browser that is allowed to reach that CDN; or ask whoever generated the report " +
      "for the numbers directly — in a task folder, " +
      "<code>report/report_payload.json</code> holds every value on this page and " +
      "<code>data/</code> and <code>metadata/</code> hold the tables." +
      "</p>" +
      (detail ? "<p>Details: <code>" + String(detail).slice(0, 300) + "</code></p>" : "");
  }};
  setTimeout(function () {{
    if (!window.__mlStarted) window.__mlFailed("the runtime did not load within 60 seconds");
  }}, 60000);
</script>
<script type="module">
try {{
  const {{ mount }} = await import("{cdn}/stlite.js");
  window.__mlStarted = true;
  mount(
    {{
      requirements: ["plotly=={plotly_version}"],
      entrypoint: "streamlit_app.py",
      files: {{
        "streamlit_app.py": JSON.parse(document.getElementById("{app_id}").textContent),
        "payload.json": document.getElementById("{payload_id}").textContent,
      }},
      streamlitConfig: {config},
    }},
    document.getElementById("root"),
  );
}} catch (error) {{
  window.__mlFailed(error && error.message);
}}
</script>
</body>
</html>
"""


def _embed(value: Any) -> str:
    """JSON for a ``<script>`` element.

    Every ``<`` becomes its JSON escape, so nothing in the data — a column name,
    an error message, a config comment — can produce a closing ``</script>`` and
    end the element early. The result is still valid JSON, which is how a test
    reads a report's numbers back without a browser.
    """
    return json.dumps(value, default=str).replace("<", "\\u003c")


def _font_faces() -> str:
    """``@font-face`` rules with the fonts inline, so the page fetches no font."""
    rules = []
    for family, weight, name in FONT_FACES:
        data = base64.b64encode((FONTS / name).read_bytes()).decode("ascii")
        rules.append(
            f"  @font-face {{ font-family: {family}; font-style: normal; font-weight: {weight}; "
            f'font-display: swap; src: url(data:font/woff2;base64,{data}) format("woff2"); }}'
        )
    return "\n".join(rules)


def render(payload: dict[str, Any]) -> str:
    """The whole page, as a string."""
    return _TEMPLATE.format(
        title=html.escape(str(payload.get("title") or "Report")),
        fonts=_font_faces(),
        config=json.dumps(STREAMLIT_CONFIG),
        cdn=CDN,
        plotly_version=PLOTLY_VERSION,
        payload=_embed(payload),
        app=_embed(APP_PATH.read_text(encoding="utf-8")),
        payload_id=PAYLOAD_ID,
        app_id=APP_ID,
    )


def write(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(payload), encoding="utf-8")
    return path


def read_payload(page: str) -> dict[str, Any]:
    """Pull the payload back out of a written report.

    Tests read a report's numbers through here rather than by parsing markup —
    the markup is Streamlit's business and changes with its version, while this
    is the data the report was built from.
    """
    opening = f'<script type="application/json" id="{PAYLOAD_ID}">'
    start = page.index(opening) + len(opening)
    return json.loads(page[start : page.index("</script>", start)])
