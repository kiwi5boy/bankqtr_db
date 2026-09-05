# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair",
#     "marimo",
#     "polars",
#     "pyarrow",
# ]
# ///
"""bankqtr_db variable explorer -- a marimo app.

Run it:

    uv run --group app marimo run apps/explorer.py --no-sandbox
    uv run --group app marimo edit apps/explorer.py --no-sandbox

``--no-sandbox`` is required, not incidental: the inline script metadata
above makes marimo offer to build a throwaway venv from it on every launch,
and the ``app`` dependency group is already that environment.  The metadata
is not for the local run at all -- it is what the WASM export reads to know
which wheels to install into the browser kernel.  Both extras beyond the
obvious are load-bearing: without ``altair`` the notebook loads and every
chart cell dies, and without ``pyarrow`` polars cannot read a parquet at all
under Pyodide -- it has no native reader there and falls back to arrow.

It answers "which variables are worth modelling", in three passes: what is
here (the screen), what one variable is actually made of (provenance and
holes), and what sample a chosen *set* of them leaves you (the builder).
The counting lives in ``apps/explore.py`` and the chart specs in
``apps/viz.py``; this file is layout.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="bankqtr_db variable explorer")


@app.cell
def _():
    import sys

    import marimo as mo

    sys.path.insert(0, str(mo.notebook_dir()))

    import explore
    import polars as pl
    import viz

    # A WASM build has no filesystem holding the panels: the modules arrive
    # as wheels and the data is served next to the notebook's own assets.
    if sys.platform == "emscripten":
        explore.set_data_base(str(mo.notebook_location() / "public" / "data"))

    # Charts are drawn for the theme the app is actually running in -- the
    # dark palette is its own selected set of steps, not an inverted light one.
    mode = mo.app_meta().theme or "light"
    return explore, mo, pl, viz, mode


@app.cell
def _(mo):
    mo.md(
        """
        # Which variables can I model with?

        Two panels, ~270 measured variables each. Half the columns in the
        EDGAR parquet are `__source` companions to the other half, so the
        browsable width is roughly what you see here, not what a spreadsheet
        shows.

        Nothing below assumes a model. It reports what is populated, where it
        came from, and what a chosen set of variables leaves you to fit on.
        """
    )


@app.cell
def _(mo):
    panel_ui = mo.ui.radio(
        options={"EDGAR — SEC, 2013–": "edgar", "FFIEC — Call Reports, 2001–": "ffiec"},
        value="EDGAR — SEC, 2013–",
        inline=True,
    )
    return (panel_ui,)


@app.cell
def _(explore, mo, panel_ui):
    panel = panel_ui.value
    _families = sorted({explore.family_of(c) for c in explore.value_columns(panel)})
    banks = sorted(explore.load_panel(panel)["ticker"].unique().to_list())

    family_ui = mo.ui.multiselect(
        options=_families, value=[], label="families", full_width=False
    )
    min_ok_ui = mo.ui.slider(
        start=0, stop=len(banks), value=0, step=1,
        label="min banks with usable coverage", show_value=True,
    )
    return banks, family_ui, min_ok_ui, panel


@app.cell
def _(family_ui, min_ok_ui, mo, panel_ui):
    # One filter row above everything it scopes -- every section below reads
    # the same slice, and no chart carries a filter of its own.
    mo.hstack(
        [panel_ui, family_ui, min_ok_ui], justify="start", gap=1.5, align="center"
    )


@app.cell
def _(explore, family_ui, min_ok_ui, panel, pl):
    screened = explore.screen(panel).filter(pl.col("banks_ok") >= min_ok_ui.value)
    if family_ui.value:
        screened = screened.filter(pl.col("family").is_in(family_ui.value))
    return (screened,)


@app.cell
def _(mo, screened):
    mo.md(
        f"""
        ## 1 · What is here

        **{screened.height}** variables match the filter. `ok` and `sparse`
        are the shipped thresholds — a bank is `ok` on a variable when at
        least 60% of its quarters are populated. `n/a` is not a gap: it is a
        bank that correctly has no such book, and it is excluded from the
        median so a custody bank does not drag down a credit-card variable.
        """
    )


@app.cell
def _(mo, mode, screened, viz):
    _cap = 45
    _shown = screened.sort(["banks_ok", "banks_sparse"], descending=True).head(_cap)
    _note = (
        mo.md(
            f"*Charting the {_cap} best-covered of {screened.height} — "
            "narrow the family filter to see the rest.*"
        )
        if screened.height > _cap
        else mo.md("")
    )
    mo.vstack(
        [
            viz.coverage_bar(_shown, mode, height=max(220, 18 * _shown.height)),
            _note,
        ]
    )


@app.cell
def _(mo, screened):
    # The table twin: every number in the chart above, sortable and
    # searchable, and the only place the window and cell counts appear.
    mo.ui.table(screened, selection=None, page_size=15)


@app.cell
def _(banks, mo, screened):
    _options = screened["variable"].to_list()
    variable_ui = mo.ui.dropdown(
        options=_options,
        value=_options[0] if _options else None,
        label="variable",
        searchable=True,
    )
    # Three is the cap the validated line palette supports; past it, identity
    # stops being reliable and the peer band is the better read anyway.
    highlight_ui = mo.ui.multiselect(
        options=banks, value=banks[:2], label="highlight banks",
        max_selections=3,
    )
    return highlight_ui, variable_ui


@app.cell
def _(mo):
    mo.md("## 2 · What one variable is made of")


@app.cell
def _(highlight_ui, mo, variable_ui):
    mo.hstack([variable_ui, highlight_ui], justify="start", gap=1.5, align="center")


@app.cell
def _(mo, pl, screened, variable_ui):
    _row = (
        screened.filter(pl.col("variable") == variable_ui.value)
        if variable_ui.value
        else None
    )
    _stats = None
    if _row is not None and _row.height:
        _r = _row.row(0, named=True)
        _pct = "—" if _r["pct_median"] is None else f"{_r['pct_median']:.0f}%"
        _stats = mo.hstack(
            [
                mo.stat(f"{_r['banks_ok']}", label="banks with usable coverage",
                        caption=f"of {_r['banks_ok'] + _r['banks_sparse'] + _r['banks_missing']} that could report it",
                        bordered=True),
                mo.stat(_pct, label="median coverage",
                        caption="across the banks it applies to", bordered=True),
                mo.stat(f"{_r['first_q']} – {_r['last_q']}", label="window",
                        caption=f"{_r['n_cells']:,} populated bank-quarters",
                        bordered=True),
            ],
            justify="start",
            gap=1,
        )
    _stats  # noqa: B018 -- a marimo cell's last expression is its output


@app.cell
def _(explore, highlight_ui, mode, panel, variable_ui, viz):
    _chart = None
    if variable_ui.value:
        _band, _lines = explore.series(panel, variable_ui.value, highlight_ui.value)
        _chart = viz.series_chart(
            _band,
            _lines,
            explore.seams_for(panel, highlight_ui.value),
            mode,
            title=f"{variable_ui.value} — highlighted banks against the peer band",
        )
    _chart  # noqa: B018 -- a marimo cell's last expression is its output


@app.cell
def _(mo):
    mo.md(
        """
        The grey band is the interquartile range across whichever banks
        report the variable that quarter — it narrows and widens as coverage
        changes, so read it with the count in the tooltip. Vertical rules are
        seams: the quarter a bank's numbers change meaning for a reason that
        is not the bank. CECL adoption is read off the panel's own `basis`
        column rather than assumed at 2020Q1, because one bank in this
        universe deferred to 2020Q4.
        """
    ).callout(kind="neutral")


@app.cell
def _(explore, mo, mode, panel, variable_ui, viz):
    _left = mo.md("")
    _right = mo.md("")
    if variable_ui.value:
        _prov = explore.provenance(panel, variable_ui.value)
        _left = mo.vstack(
            [
                mo.md("**Where the values came from**"),
                viz.provenance_bar(_prov, mode),
                mo.ui.table(_prov, selection=None, page_size=5),
            ]
        )
        _right = viz.presence_grid(
            explore.availability(panel, [variable_ui.value]),
            mode,
            title="populated bank-quarters",
            height=360,
        )
    mo.hstack([_left, _right], widths=[1, 1], gap=2, align="start")


@app.cell
def _(mo):
    mo.md(
        """
        ## 3 · What a set of variables leaves you

        This is the number that decides a model, and it is not the coverage
        of the weakest variable in the set. Two variables each 80% populated
        intersect anywhere between 60% and 80% depending on whether the same
        banks are missing both — so the intersection is computed, not
        inferred from the table above.
        """
    )


@app.cell
def _(explore, mo, panel):
    chosen_ui = mo.ui.multiselect(
        options=explore.value_columns(panel),
        value=[],
        label="variables in the set",
        full_width=True,
    )
    chosen_ui  # noqa: B018 -- a marimo cell's last expression is its output
    return (chosen_ui,)


@app.cell
def _(chosen_ui, explore, mo, panel):
    sample = explore.complete_case(panel, chosen_ui.value)
    mo.hstack(
        [
            mo.stat(f"{sample.n_rows:,}", label="complete bank-quarters",
                    caption=f"{sample.pct_of_panel}% of the panel", bordered=True),
            mo.stat(f"{sample.n_banks}", label="banks left",
                    caption=f"{len(sample.dropped)} dropped entirely", bordered=True),
            mo.stat(f"{sample.first_q or '—'} – {sample.last_q or '—'}",
                    label="window that survives", bordered=True),
        ],
        justify="start",
        gap=1,
    )
    return (sample,)


@app.cell
def _(chosen_ui, explore, mo, mode, panel, sample, viz):
    _grid = viz.presence_grid(
        explore.availability(panel, chosen_ui.value),
        mode,
        title="bank-quarters where every selected variable is populated",
        height=380,
    )
    _dropped = (
        mo.md(f"**Dropped entirely:** {', '.join(sample.dropped)}")
        if sample.dropped
        else mo.md("*No bank is dropped entirely by this set.*")
    )
    mo.vstack([_grid, _dropped])


@app.cell
def _(chosen_ui, explore, mo, mode, panel, screened, viz):
    # Priced against the *filtered* candidate list: pricing all ~270 on every
    # keystroke is work nobody asked for, and the family filter above is
    # already the question being asked.
    _cands = screened["variable"].to_list()
    _cap = 40
    _cost = explore.marginal_cost(panel, chosen_ui.value, _cands[:_cap])
    mo.vstack(
        [
            mo.md(
                "**What each candidate would cost, added to the set above.** "
                "Two variables with the same coverage on paper can cost very "
                "different amounts here — what separates them is whether "
                "their holes fall on banks the set has already lost."
            ),
            viz.cost_bar(_cost, mode, height=max(200, 18 * _cost.height)),
            mo.ui.table(_cost, selection=None, page_size=10),
        ]
    )
