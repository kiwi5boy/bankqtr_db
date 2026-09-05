"""Chart builders for the explorer.

Colour here is not a taste decision.  Every palette below was run through the
data-viz validator before it was used: the four provenance hues pass the
adjacent-pair gates in both modes, and the three series hues pass the stricter
all-pairs gates that a chart with non-adjacent marks needs.  Two of the light
hues sit under 3:1 against the light surface, so every chart that uses them
ships direct labels and a table twin -- that is the documented relief, not an
oversight.

The rules the charts hold to, in case one gets extended later: one y-axis
ever; colour follows the entity, so filtering the bank list never repaints
the survivors; hues are assigned in fixed slot order and never cycled past
the fourth.
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import polars as pl

# --------------------------------------------------------------------------
# Palettes -- selected per mode, not flipped

@dataclass(frozen=True)
class Palette:
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    band: str
    absent: str
    series: tuple[str, ...]


LIGHT = Palette(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    band="#c3c2b7",
    absent="#eeede8",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
)

DARK = Palette(
    surface="#1a1a19",
    ink="#ffffff",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    band="#383835",
    absent="#242422",
    series=("#3987e5", "#d95926", "#199e70", "#c98500"),
)

# Fixed in both modes by design -- a status colour means the same thing
# whatever the surface, and never doubles as a series hue.  Paired with a
# label everywhere it appears, so state is never carried by colour alone.
# Only the two problem states get a status colour.  "ok" is the absence of
# a problem, and on this chart it is also the overwhelming majority of every
# bar -- painting it a saturated green would put a wall of loud colour behind
# the small marks that actually carry the signal.  It takes the neutral, and
# the saturated steps stay for the short segments, which is what saturated
# fills are for.
STATUS = {"sparse": "#fab219", "missing": "#d03b3b"}
STATUS_ORDER = ("ok", "sparse", "missing")

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def palette(mode: str) -> Palette:
    return DARK if mode == "dark" else LIGHT


def _dress(chart: alt.Chart, mode: str) -> alt.Chart:
    """Recessive chrome: hairline grid, no view border, quiet ink."""
    p = palette(mode)
    return (
        chart.configure_view(strokeWidth=0, fill=p.surface)
        .configure_axis(
            labelFont=FONT, titleFont=FONT, labelColor=p.muted,
            titleColor=p.ink2, gridColor=p.grid, domainColor=p.axis,
            tickColor=p.axis, gridDash=[], labelFontSize=11, titleFontSize=11,
        )
        .configure_legend(
            labelFont=FONT, titleFont=FONT, labelColor=p.ink2,
            titleColor=p.ink2, labelFontSize=11, titleFontSize=11,
        )
        .configure_title(font=FONT, color=p.ink, fontSize=13, anchor="start")
        .configure_text(font=FONT)
    )


# --------------------------------------------------------------------------
# The screen: how many banks carry each variable, and how well

def coverage_bar(screen: pl.DataFrame, mode: str, height: int = 420) -> alt.Chart:
    """One stacked bar per variable, segments = bank counts by status.

    Sorted by usable banks, so the top of the chart is the answer to "what
    can I model with" and the bottom is the answer to "what is basically
    not here".
    """
    p = palette(mode)
    # n/a is deliberately not a segment.  A bank that correctly has no such
    # book is not a gap, and stacking it beside the real gaps would make a
    # variable look thin for a reason that has nothing to do with the data.
    # It stays in the tooltip and in the table twin.
    long = (
        screen.select("variable", "family", "banks_ok", "banks_sparse",
                      "banks_missing", "banks_na")
        .rename({"banks_ok": "ok", "banks_sparse": "sparse",
                 "banks_missing": "missing"})
        .unpivot(
            index=["variable", "family", "banks_na"],
            variable_name="status",
            value_name="banks",
        )
        .filter(pl.col("banks") > 0)
    )
    reporting = int(
        (screen["banks_ok"] + screen["banks_sparse"] + screen["banks_missing"]).max()
        or 1
    )
    order = (
        screen.sort(["banks_ok", "banks_sparse"], descending=True)["variable"].to_list()
    )
    return _dress(
        alt.Chart(long)
        .mark_bar(
            # A 2px stroke in the surface colour reads as a gap between
            # segments -- it is spacing, not a border drawn round the mark.
            stroke=p.surface,
            strokeWidth=2,
            height={"band": 0.8},
        )
        .encode(
            y=alt.Y("variable:N", sort=order, title=None,
                    axis=alt.Axis(labelLimit=220)),
            x=alt.X(
                "banks:Q",
                title="banks that could report it",
                stack="zero",
                scale=alt.Scale(domain=[0, reporting], nice=False),
                axis=alt.Axis(tickCount=6),
            ),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(
                    domain=list(STATUS_ORDER),
                    range=[p.band, STATUS["sparse"], STATUS["missing"]],
                ),
                legend=alt.Legend(title="coverage", orient="top"),
            ),
            order=alt.Order("color_status_sort_index:Q"),
            tooltip=[
                "variable:N", "family:N", "status:N", "banks:Q",
                alt.Tooltip("banks_na:Q", title="banks it does not apply to"),
            ],
        )
        .properties(width="container", height=height),
        mode,
    )


# --------------------------------------------------------------------------
# One variable: where the values came from, and where the holes are

def provenance_bar(prov: pl.DataFrame, mode: str) -> alt.Chart:
    """A single 100%-wide bar: the mix of sources behind the populated cells.

    Part-to-whole with at most four segments, direct-labelled -- which is
    also the relief the light palette's contrast warning requires.
    """
    p = palette(mode)
    total = prov["n"].sum() or 1
    data = prov.with_columns(
        pct=(100 * pl.col("n") / total).round(1), row=pl.lit("source")
    )
    scale = alt.Scale(
        domain=data["source"].to_list(),
        range=list(p.series[: data.height]),
    )
    base = alt.Chart(data).encode(
        x=alt.X("n:Q", stack="normalize", title=None,
                axis=alt.Axis(format="%", grid=False)),
        y=alt.Y("row:N", title=None, axis=None),
        order=alt.Order("n:Q", sort="descending"),
        tooltip=["source:N", "n:Q", alt.Tooltip("pct:Q", title="% of cells")],
    )
    bars = base.mark_bar(
        stroke=p.surface, strokeWidth=2, cornerRadius=4, height=28
    ).encode(color=alt.Color("source:N", scale=scale, legend=alt.Legend(orient="top", title=None)))
    # Label only the segments wide enough to hold the text; the rest are in
    # the legend and the tooltip, and all of them are in the table twin.
    labels = base.transform_filter(alt.datum.pct >= 12).encode(
        # Without an explicit band position the text lands on the *end* of the
        # stacked segment, which pushes the widest label off the chart.
        x=alt.X("n:Q", stack="normalize", bandPosition=0.5, title=None,
                axis=alt.Axis(format="%", grid=False)),
    ).mark_text(
        color=p.surface, fontWeight=600, fontSize=11, font=FONT
    ).encode(text=alt.Text("pct:Q", format=".0f"))
    return _dress(
        (bars + labels).properties(width="container", height=52), mode
    )


def presence_grid(
    avail: pl.DataFrame, mode: str, title: str = "", height: int = 380
) -> alt.Chart:
    """Bank x quarter, filled where every selected variable is populated.

    Two states, so two colours: present takes the first series slot, absent
    takes a step barely off the surface.  The holes are the subject of the
    chart, and they read as holes rather than as a third category.
    """
    p = palette(mode)
    order = (
        avail.group_by("ticker")
        .agg(n=pl.col("complete").sum())
        .sort("n", descending=True)["ticker"]
        .to_list()
    )
    return _dress(
        alt.Chart(avail)
        .mark_rect()
        .encode(
            x=alt.X("yearquarter(period):T", title=None),
            y=alt.Y("ticker:N", sort=order, title=None),
            color=alt.Color(
                "complete:N",
                scale=alt.Scale(
                    domain=[True, False],
                    range=[p.series[0], p.absent],
                ),
                legend=alt.Legend(
                    title=None, orient="top",
                    labelExpr="datum.label == 'true' ? 'populated' : 'missing'",
                ),
            ),
            tooltip=["ticker:N", "quarter:N", "complete:N"],
        )
        .properties(width="container", height=height, title=title),
        mode,
    )


# --------------------------------------------------------------------------
# One variable through time, with the seams drawn on

def series_chart(
    band: pl.DataFrame,
    lines: pl.DataFrame,
    seams: pl.DataFrame,
    mode: str,
    title: str = "",
    height: int = 320,
) -> alt.Chart:
    """The peer band, up to three named banks, and the seams behind them.

    The band is the cross-sectional interquartile range per quarter, not a
    confidence interval -- on a sparse variable the early quarters can be
    two banks wide, so the count rides in the tooltip.

    Identity is carried by the endpoint label, not by hue: with only three
    validated slots against an arbitrary user-chosen subset, a hue cannot
    stay pinned to a bank across selection changes, so the label is the
    thing that must be readable and the colour is only separation.
    """
    p = palette(mode)
    highlighted = lines["ticker"].unique(maintain_order=True).to_list()[:3]
    lines = lines.filter(pl.col("ticker").is_in(highlighted))
    scale = alt.Scale(
        domain=highlighted, range=list(p.series[: len(highlighted)])
    )
    x = alt.X("period:T", title=None)

    area = (
        alt.Chart(band)
        .mark_area(opacity=0.30, color=p.band, line=False)
        .encode(x=x, y=alt.Y("p25:Q", title=None), y2=alt.Y2("p75:Q"))
    )
    median = (
        alt.Chart(band)
        .mark_line(color=p.muted, strokeWidth=1.5)
        .encode(
            x=x,
            y="p50:Q",
            tooltip=[
                alt.Tooltip("quarter:N", title="quarter"),
                alt.Tooltip("p50:Q", title="peer median", format=",.4g"),
                alt.Tooltip("n:Q", title="banks behind it"),
            ],
        )
    )
    layers = [area, median]

    if lines.height:
        bank_lines = (
            alt.Chart(lines)
            .mark_line(strokeWidth=2)
            .encode(
                x=x,
                y=alt.Y("value:Q", title=None),
                color=alt.Color("ticker:N", scale=scale,
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=[
                    alt.Tooltip("ticker:N"),
                    alt.Tooltip("quarter:N"),
                    alt.Tooltip("value:Q", format=",.4g"),
                ],
            )
        )
        # Direct-label the right-hand end of each line: the legend says which
        # colour, the label says it again where the eye already is.
        ends = (
            alt.Chart(lines)
            # The label belongs on the last quarter the bank actually
            # reported, not on a trailing hole.
            .transform_filter("isValid(datum.value)")
            .transform_window(
                rank="rank(period)", sort=[alt.SortField("period", "descending")],
                groupby=["ticker"],
            )
            .transform_filter(alt.datum.rank == 1)
            .mark_text(align="left", dx=6, fontSize=11, font=FONT, fontWeight=600)
            .encode(x=x, y="value:Q",
                    text="ticker:N",
                    color=alt.Color("ticker:N", scale=scale, legend=None))
        )
        layers += [bank_lines, ends]

    if seams.height:
        # Solid, one weight up from the hairline grid, and always labelled --
        # a seam is an annotation about the data's meaning, not a threshold
        # and not a gridline, so it must not be mistakable for either.
        marks = seams.unique(subset=["period", "label"])
        rules = (
            alt.Chart(marks)
            .mark_rule(color=p.ink2, strokeWidth=1, opacity=0.7)
            .encode(x=x, tooltip=[alt.Tooltip("label:N"), alt.Tooltip("why:N")])
        )
        captions = (
            alt.Chart(marks)
            .mark_text(align="left", baseline="top", dx=4, dy=2,
                       fontSize=10, font=FONT, color=p.ink2)
            .encode(x=x, text="label:N")
        )
        layers += [rules, captions]

    return _dress(
        alt.layer(*layers)
        .resolve_scale(color="independent")
        .properties(width="container", height=height, title=title),
        mode,
    )


def cost_bar(cost: pl.DataFrame, mode: str, height: int = 380) -> alt.Chart:
    """What each candidate variable would cost the sample, in rows.

    One measure, one series, one colour -- the bar length already carries
    the magnitude, so shading it by size would spend the colour channel on
    information the chart shows twice.
    """
    p = palette(mode)
    order = cost.sort("rows_lost")["variable"].to_list()
    return _dress(
        alt.Chart(cost)
        .mark_bar(color=p.series[0], cornerRadiusEnd=4,
                  height={"band": 0.75})
        .encode(
            y=alt.Y("variable:N", sort=order, title=None,
                    axis=alt.Axis(labelLimit=220)),
            x=alt.X(
                "rows_lost:Q",
                title="bank-quarters lost if added",
                scale=alt.Scale(domain=[0, int(cost["rows_lost"].max() or 1)],
                                nice=False),
            ),
            tooltip=[
                "variable:N", "family:N",
                alt.Tooltip("rows_lost:Q", title="rows lost"),
                alt.Tooltip("rows_after:Q", title="rows left"),
                alt.Tooltip("banks_lost:Q", title="banks lost"),
                alt.Tooltip("banks_after:Q", title="banks left"),
            ],
        )
        .properties(width="container", height=height),
        mode,
    )
