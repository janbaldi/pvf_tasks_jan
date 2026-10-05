"""Every figure in the report, as plotly JSON.

One function per figure, each returning ``None`` when there is nothing to draw,
so the section it belongs to drops out rather than rendering an empty axis.
This module imports :mod:`pvf.blocks` and nothing else from the project.

Colour is spent only where it carries something: the two sites share an axis and
get one colour each, and the registry outcomes are a severity scale. Everything
single-series is neutral ink.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .blocks import Block, figure

INK = "#3f4750"
MUTED = "#9aa5b1"
SITE_COLOURS = ("#3f4750", "#7c9cbf")
#: created, already present, blocked, not in PTF, errored
STATUS_COLOURS = {
    "created": "#3f4750",
    "present": "#b9c2cc",
    "blocked": "#c79a3e",
    "not in PTF": "#7c9cbf",
    "errored": "#c0392b",
}


def _fig(fig: go.Figure, caption: str = "") -> Block:
    """The one choke point every figure passes through.

    ``to_json`` rather than ``to_dict`` because it is what encodes numpy and
    pandas values to plain JSON, and the payload has to survive a round trip
    through a script tag. The default template is dropped: it serialises trace
    defaults for every trace type this plotly knows about, and the plotly in the
    browser rejects the ones it has since renamed.
    """
    spec = json.loads(fig.to_json())
    spec.get("layout", {}).pop("template", None)
    return figure(spec, caption)


def _style(fig: go.Figure, title: str, x: str = "", y: str = "", height: int = 360) -> go.Figure:
    """Layout shared by every figure: no background, so the page theme shows through.

    The title is pinned to the top edge and a legend, when there is one, gets a
    row of its own between the title and the plot. Placed inside the title's
    margin, the two are drawn over each other.
    """
    legend = bool(fig.layout.showlegend)
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14),
            x=0,
            xanchor="left",
            y=1,
            yref="container",
            yanchor="top",
            pad=dict(t=10),
        ),
        xaxis_title=x,
        yaxis_title=y,
        height=height + (28 if legend else 0),
        margin=dict(l=10, r=10, t=72 if legend else 44, b=10),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.22)", zeroline=False)
    return fig


def profile_distribution(
    profiles: Mapping[str, Mapping[str, Mapping[str, int]]],
    column: str,
    order: Sequence[str],
) -> Block | None:
    """Batches per profile class, one bar group per site.

    ``profiles`` is ``{site: {column: {class: count}}}``. Sites share the axis,
    which is what earns them a colour each.
    """
    counts = {site: by_column.get(column, {}) for site, by_column in profiles.items()}
    counts = {site: values for site, values in counts.items() if values}
    if not counts:
        return None

    classes = [c for c in order if any(c in values for values in counts.values())]
    classes += sorted({c for values in counts.values() for c in values if c not in order})

    fig = go.Figure()
    for colour, (site, values) in zip(SITE_COLOURS, counts.items()):
        fig.add_bar(
            name=site,
            x=classes,
            y=[values.get(c, 0) for c in classes],
            marker_color=colour,
            hovertemplate="%{x}: %{y} batches<extra>" + site + "</extra>",
        )
    fig.update_layout(barmode="group", showlegend=True)
    return _fig(_style(fig, f"{column} — batches per class", y="batches"), "")


def registry_outcome(reports: Sequence) -> Block | None:
    """How each feature group fared, stacked by status.

    One bar per group, so a group that is entirely blocked for want of one input
    is visible without reading the table under it.
    """
    entries = [entry for report in reports for entry in report.entries]
    if not entries:
        return None

    groups = sorted({entry["group"] for entry in entries})
    fig = go.Figure()
    for status, colour in STATUS_COLOURS.items():
        counts = [
            sum(1 for e in entries if e["group"] == group and e["status"] == status)
            for group in groups
        ]
        if not any(counts):
            continue
        fig.add_bar(
            name=status,
            x=groups,
            y=counts,
            marker_color=colour,
            hovertemplate="%{x}: %{y} " + status + "<extra></extra>",
        )
    fig.update_layout(barmode="stack", showlegend=True)
    return _fig(
        _style(fig, "Feature outcomes by group", y="features", height=400),
        "Counted across sites, so a feature computed at both sites appears twice.",
    )


def coverage_bars(report, limit: int = 30) -> Block | None:
    """The created features with the least data behind them.

    A feature that computed cleanly over three batches is still an empty column,
    and that only shows up as coverage.
    """
    created = [
        (entry["name"], 100 * entry["non_null"] / report.rows)
        for entry in report.entries
        if entry["status"] == "created" and report.rows
    ]
    if not created:
        return None

    created.sort(key=lambda item: item[1])
    shown = created[:limit]
    fig = go.Figure(
        go.Bar(
            x=[pct for _, pct in shown],
            y=[name for name, _ in shown],
            orientation="h",
            marker_color=INK,
            hovertemplate="%{y}: %{x:.1f}% of batches<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 100], showgrid=True, gridcolor="rgba(128,128,128,0.22)")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _fig(
        _style(
            fig,
            f"{report.site} — thinnest features by coverage",
            x="% of batches with a value",
            height=max(320, 22 * len(shown) + 80),
        ),
        f"The {len(shown)} sparsest of {len(created)} features created at {report.site}.",
    )


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------
def cluster_fits(cluster) -> Block | None:
    """How well each member is explained by the parameters it was fitted on.

    Observed against predicted, not the member against the representative: a
    sequential residual is fitted on several parameters at once, so a single
    curve drawn against one of them would suggest a relationship that was never
    fitted.

    Members are drawn in units of their own spread, because a cell count in the
    billions and a slope around one cannot share an axis — on a common scale the
    smaller one collapses onto the origin and reads as a perfect fit. The axes
    say so, and the hover text carries the measured values.
    """
    fits = [fit for fit in cluster.fits if fit.y and fit.fitted]
    if not fits:
        return None

    palette = ("#3f4750", "#7c9cbf", "#a68a64", "#6b8f71", "#9b7d9b", "#c0864f")
    figure_ = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Observed against predicted, on the rows the fit used",
            "Residual against predicted — what the dataset gets",
        ),
    )

    for index, fit in enumerate(fits):
        colour = palette[index % len(palette)]
        short = fit.column if len(fit.column) <= 34 else fit.column[:33] + "…"
        detail = f"{fit.model}; {fit.rows} batches"
        observed = np.asarray(fit.y, dtype="float64")
        predicted = np.asarray(fit.fitted, dtype="float64")
        residual = np.asarray(fit.residual, dtype="float64")
        centre = float(np.mean(observed))
        spread = float(np.std(observed)) or 1.0

        figure_.add_scatter(
            x=(predicted - centre) / spread,
            y=(observed - centre) / spread,
            mode="markers",
            name=short,
            legendgroup=short,
            marker=dict(color=colour, size=6, opacity=0.65),
            customdata=np.column_stack([predicted, observed]),
            hovertemplate=(
                f"<b>{fit.column}</b><br>predicted %{{customdata[0]:.4g}}, "
                f"observed %{{customdata[1]:.4g}}<extra>{detail}</extra>"
            ),
            row=1,
            col=1,
        )
        figure_.add_scatter(
            x=(predicted - centre) / spread,
            y=residual / spread,
            mode="markers",
            name=short,
            legendgroup=short,
            showlegend=False,
            marker=dict(color=colour, size=6, opacity=0.65),
            customdata=np.column_stack([predicted, residual]),
            hovertemplate=(
                f"<b>{fit.column}</b><br>predicted %{{customdata[0]:.4g}}, "
                f"residual %{{customdata[1]:.4g}}<extra>{detail}</extra>"
            ),
            row=1,
            col=2,
        )

    figure_.add_shape(
        type="line",
        x0=-3,
        y0=-3,
        x1=3,
        y1=3,
        line=dict(color=MUTED, width=1, dash="dash"),
        row=1,
        col=1,
    )
    figure_.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"), row=1, col=2)
    figure_.update_xaxes(
        title_text="predicted, in standard deviations of the member",
        showgrid=False,
        zeroline=False,
    )
    figure_.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.22)", zeroline=False)
    figure_.update_yaxes(title_text="observed (standardised)", row=1, col=1)
    figure_.update_yaxes(title_text="residual (standardised)", row=1, col=2)
    figure_.update_layout(
        title=dict(
            text=f"{cluster.name} — members against their fits",
            font=dict(size=14),
            x=0,
            xanchor="left",
        ),
        height=460,
        # Room under the axis titles for the legend, which is a row of parameter
        # names and would otherwise be drawn straight through them.
        margin=dict(l=10, r=10, t=76, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
    )
    return _fig(
        figure_,
        "Each member is standardised by its own spread so they can share an axis; the "
        "measured values are in the hover text. Points on the dashed line are fitted "
        "exactly, and the right-hand cloud is what the dataset receives.",
    )


def cluster_correlation(cluster) -> Block | None:
    """The cluster's correlation matrix before and after, signed.

    Colour is the axis here — the sign of a correlation is half of what it says —
    which is the case where a diverging scale earns its place. A pair with too
    few shared batches is left blank rather than drawn as zero, because those are
    different statements.
    """
    before, after = cluster.corr_matrix_before, cluster.corr_matrix_after
    if not before:
        return None

    figure_ = make_subplots(rows=1, cols=2, subplot_titles=("Before", "After"))
    names = [name if len(name) <= 40 else name[:39] + "…" for name in cluster.members]
    codes_before = [str(i + 1) for i in range(len(before))]
    codes_after = [str(i + 1) for i in range(len(after))]
    overlap = cluster.overlap_matrix or [[0] * len(before) for _ in before]
    hover = [
        [
            f"{names[row]}<br>vs {names[column]}<br>"
            + (
                "no supported estimate"
                if before[row][column] is None
                else f"r = {before[row][column]:+.2f}"
            )
            + f"<br>{overlap[row][column]} shared batches"
            for column in range(len(before))
        ]
        for row in range(len(before))
    ]

    figure_.add_heatmap(
        z=before,
        x=codes_before,
        y=codes_before,
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        reversescale=True,
        showscale=False,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        row=1,
        col=1,
    )
    if after:
        labels_after = [
            name if len(name) <= 40 else name[:39] + "…" for name in cluster.labels_after
        ]
        after_hover = [
            [
                f"{labels_after[row]}<br>vs {labels_after[column]}<br>"
                + (
                    "no supported estimate"
                    if after[row][column] is None
                    else f"r = {after[row][column]:+.2f}"
                )
                for column in range(len(after))
            ]
            for row in range(len(after))
        ]
        figure_.add_heatmap(
            z=after,
            x=codes_after,
            y=codes_after,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="r", thickness=12),
            text=after_hover,
            hovertemplate="%{text}<extra></extra>",
            row=1,
            col=2,
        )

    figure_.update_yaxes(autorange="reversed", showgrid=False)
    figure_.update_xaxes(showgrid=False)
    figure_.update_layout(
        title=dict(
            text=f"{cluster.name} — {cluster.action} · correlation within the cluster",
            font=dict(size=14),
            x=0,
            xanchor="left",
        ),
        height=340,
        margin=dict(l=10, r=10, t=76, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return _fig(
        figure_,
        "Blank means too few shared batches to estimate, which is not the same as no "
        "correlation. Axis numbers are the parameters listed below; full names are in the "
        "hover text.",
    )


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------
def target_distribution(values, name: str, unit: str = "", classes=None) -> Block | None:
    """What the thing being predicted actually looks like across the cohort."""
    numbers = [float(v) for v in values if v == v]
    if not numbers:
        return None

    if classes:
        counts = {label: numbers.count(float(code)) for code, label in classes.items()}
        figure_ = go.Figure(
            go.Bar(
                x=list(counts),
                y=list(counts.values()),
                marker_color=INK,
                hovertemplate="%{x}: %{y} batches<extra></extra>",
            )
        )
        return _fig(
            _style(figure_, f"{name} — class balance", y="batches", height=300),
            f"{counts.get(list(counts)[-1], 0)} of {len(numbers)} batches are the positive class.",
        )

    figure_ = go.Figure(
        go.Histogram(x=numbers, marker_color=INK, hovertemplate="%{x}: %{y}<extra></extra>")
    )
    return _fig(
        _style(
            figure_,
            f"{name} — distribution over the cohort",
            x=unit or name,
            y="batches",
            height=300,
        ),
        f"{len(numbers)} batches, from {min(numbers):.4g} to {max(numbers):.4g}.",
    )


def missingness(rows, limit: int = 25) -> Block | None:
    """The columns with the least data, worst first."""
    worst = sorted(rows, key=lambda row: row["missing_pct"], reverse=True)[:limit]
    worst = [row for row in worst if row["missing_pct"] > 0]
    if not worst:
        return None
    labels = [
        row["column"] if len(row["column"]) <= 42 else row["column"][:41] + "…" for row in worst
    ]
    figure_ = go.Figure(
        go.Bar(
            x=[row["missing_pct"] for row in worst],
            y=labels,
            orientation="h",
            marker_color=INK,
            customdata=[[row["column"], row["present"]] for row in worst],
            hovertemplate="<b>%{customdata[0]}</b><br>%{x:.1f}% missing, "
            "%{customdata[1]} batches with a value<extra></extra>",
        )
    )
    figure_ = _style(
        figure_,
        f"The {len(worst)} columns with the most missing values",
        x="% of batches with no value",
        height=max(300, 22 * len(worst)),
    )
    figure_.update_yaxes(autorange="reversed")
    return _fig(figure_, "Full names are in the hover text.")


def encoder_split(routes) -> Block | None:
    """How many categorical parameters went to each encoder."""
    if not routes:
        return None
    order = ["binary", "one_hot", "target", "hashing", "skip"]
    labels = {
        "binary": "binary",
        "one_hot": "one-hot",
        "target": "target encoding",
        "hashing": "hashing",
        "skip": "left out",
    }
    counts = [sum(1 for r in routes if r.strategy == strategy) for strategy in order]
    if not any(counts):
        return None

    figure_ = go.Figure(
        go.Bar(
            x=[labels[s] for s in order],
            y=counts,
            marker_color=INK,
            hovertemplate="%{x}: %{y} parameters<extra></extra>",
        )
    )
    return _fig(
        _style(figure_, "Categorical parameters per encoder", y="parameters", height=300),
        "Every categorical parameter goes to exactly one of these.",
    )


def missing_pattern(group) -> Block | None:
    """Which batches miss which member of a group: one row per member, one column per batch."""
    if not group.pattern:
        return None
    names = [name if len(name) <= 40 else name[:39] + "…" for name in group.members]
    figure_ = go.Figure(
        go.Heatmap(
            z=group.pattern,
            x=list(range(1, len(group.batch_labels) + 1)),
            y=names,
            colorscale=[[0.0, "#eef1f4"], [1.0, INK]],
            zmin=0,
            zmax=1,
            showscale=False,
            customdata=[group.batch_labels for _ in group.members],
            hovertemplate="%{y}<br>batch %{customdata}<br>missing = %{z}<extra></extra>",
        )
    )
    figure_ = _style(
        figure_,
        f"{group.name} — missing values per batch",
        x="batch (in the order shown in the caption)",
        height=max(260, 26 * len(group.members) + 120),
    )
    figure_.update_yaxes(autorange="reversed")
    return _fig(figure_, "Dark cells are missing values.")


def missing_by_category(group) -> Block | None:
    """How often the group is missing, per category of its best explainer."""
    explainer = group.explainer
    if explainer is None or not explainer.rates:
        return None
    categories = list(explainer.rates)
    figure_ = go.Figure(
        go.Bar(
            x=categories,
            y=[100 * explainer.rates[c] for c in categories],
            marker_color=INK,
            customdata=[explainer.batches.get(c, 0) for c in categories],
            hovertemplate="%{x}: %{y:.0f}% of %{customdata} batches missing<extra></extra>",
        )
    )
    return _fig(
        _style(
            figure_,
            f"{group.name} — how often the group is missing, by {explainer.column}",
            y="% of batches missing at least one member",
            height=320,
        )
    )
