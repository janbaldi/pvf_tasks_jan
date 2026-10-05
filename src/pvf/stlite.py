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
"""

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
  html, body {{ margin: 0; padding: 0; }}
  #boot {{
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    max-width: 34rem; margin: 18vh auto; padding: 0 1.5rem; line-height: 1.55;
  }}
  #boot h1 {{ font-size: 1.15rem; font-weight: 600; margin: 0 0 0.6rem; }}
  #boot p {{ margin: 0.4rem 0; opacity: 0.75; font-size: 0.92rem; }}
  #boot code {{ font-size: 0.85rem; }}
  #boot.failed h1 {{ color: #c0392b; }}
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
  window.__pvfStarted = false;
  window.__pvfFailed = function (detail) {{
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
    if (!window.__pvfStarted) window.__pvfFailed("the runtime did not load within 60 seconds");
  }}, 60000);
</script>
<script type="module">
try {{
  const {{ mount }} = await import("{cdn}/stlite.js");
  window.__pvfStarted = true;
  mount(
    {{
      requirements: ["plotly=={plotly_version}"],
      entrypoint: "streamlit_app.py",
      files: {{
        "streamlit_app.py": JSON.parse(document.getElementById("{app_id}").textContent),
        "payload.json": document.getElementById("{payload_id}").textContent,
      }},
      streamlitConfig: {{ "client.toolbarMode": "viewer" }},
    }},
    document.getElementById("root"),
  );
}} catch (error) {{
  window.__pvfFailed(error && error.message);
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


def render(payload: dict[str, Any]) -> str:
    """The whole page, as a string."""
    return _TEMPLATE.format(
        title=html.escape(str(payload.get("title") or "Report")),
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
