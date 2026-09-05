"""Data layer for the variable explorer -- everything except the widgets.

The explorer answers one question: *which of these variables can I actually
model with?*  That is not a rendering problem, it is a counting problem, and
counting the wrong thing here produces a plausible-looking answer -- so the
counting lives in this module, with tests, and ``explorer.py`` only draws it.

Three counts matter, and they are not the same number:

``screen``
    per variable, how many banks carry it and how well.  Rolls the shipped
    per-bank coverage tables up to one row per variable.
``provenance``
    per variable, *where the values came from*.  A variable that is 80%
    populated off the HTML fallback is a different proposition from one that
    is 80% populated off XBRL, and the panel records this per cell in a
    parallel ``<var>__source`` column.
``complete_case``
    the one that decides a model.  A set of variables is usable over the
    *intersection* of their coverage, which is always smaller -- often much
    smaller -- than the coverage of the worst member, because the gaps are
    not aligned across banks.  ``marginal_cost`` prices each candidate
    variable in rows lost before it is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from io import BytesIO
from pathlib import Path

import polars as pl

# --------------------------------------------------------------------------
# Where the data lives

# A clean checkout has no ``data/out`` until something is built, but the Pages
# payload in ``docs/data`` is committed and is the same two panels.  Prefer a
# local build when there is one, so the explorer shows what you just built.
_ROOT = Path(__file__).resolve().parent.parent
_SEARCH = (_ROOT / "data" / "out", _ROOT / "docs" / "data")

# A WASM build has no filesystem to find the panels on: the modules arrive as
# wheels and the data arrives over HTTP alongside them.  ``set_data_base``
# lets the notebook point the loaders at wherever its own assets are served
# from, and the loaders fetch instead of opening.  marimo patches urllib
# through JS fetch on startup, so one sync path covers both worlds.
_DATA_BASE: str | None = None

PANELS = ("edgar", "ffiec")

_FILES = {
    "edgar": ("panel.parquet", "panel_coverage.csv"),
    "ffiec": ("call_panel.parquet", "call_panel_coverage.csv"),
}

# Columns that describe the row rather than measure the bank.  Excluded from
# every variable list -- they are always populated and would read as the
# best-covered "variables" in the panel.
META = frozenset(
    {
        "bank", "ticker", "cik", "rssd", "quarter", "period", "basis",
        "charters", "n_charters", "predecessors", "predecessor_count",
        "n_predecessor_charters", "has_predecessor", "predecessor_failed",
        "n_insured_not_filing", "n_flow_resets", "n_tfr_backfilled",
    }
)

SOURCE_SUFFIX = "__source"


def set_data_base(base: str | None) -> None:
    """Serve the panels from ``base`` (a directory or a URL prefix) instead.

    Passing ``None`` restores the local search.  Cached frames are dropped,
    because they came from the old location.
    """
    global _DATA_BASE
    _DATA_BASE = base.rstrip("/") if base else None
    load_panel.cache_clear()
    load_coverage.cache_clear()


def _find(name: str) -> Path | str:
    if _DATA_BASE is not None:
        # A base is a URL only if it says so.  Deciding by type instead would
        # send "C:\data\out" to urllib, which parses the drive letter as a
        # scheme and fails with "unknown url type: c".
        if "://" in _DATA_BASE:
            return f"{_DATA_BASE}/{name}"
        return Path(_DATA_BASE) / name
    for base in _SEARCH:
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{name} not found in {' or '.join(str(b) for b in _SEARCH)}. "
        "Build the panel, or run from a checkout with docs/data committed."
    )


def _open(name: str) -> BytesIO:
    target = _find(name)
    if isinstance(target, Path):
        return BytesIO(target.read_bytes())
    import urllib.request

    with urllib.request.urlopen(target) as response:
        return BytesIO(response.read())


@cache
def load_panel(panel: str) -> pl.DataFrame:
    return pl.read_parquet(_open(_FILES[panel][0]))


@cache
def load_coverage(panel: str) -> pl.DataFrame:
    """Per bank x variable coverage, on one schema across both panels.

    The two builds ship different columns: the EDGAR table carries a peer
    ``category`` and a ``not_applicable`` flag, because a custody bank
    genuinely has no credit-card book and counting that as a gap would be
    wrong.  The FFIEC table has neither -- every filer files the same
    schedules on the same form -- so those columns are filled with a null
    category and ``not_applicable = False``, which is the truth there rather
    than a placeholder.
    """
    raw = pl.read_csv(_open(_FILES[panel][1]))
    if panel == "ffiec":
        raw = raw.rename(
            {"quarters": "n_quarters", "populated": "n_present", "pct": "pct_present"}
        ).with_columns(
            bank=pl.col("ticker"),
            category=pl.lit(None, dtype=pl.String),
            not_applicable=pl.lit(False),
            # Same thresholds reconcile.py applies to the EDGAR side, so a
            # status means the same thing whichever panel you are reading.
            status=pl.when(pl.col("pct_present") == 0)
            .then(pl.lit("missing"))
            .when(pl.col("pct_present") < 60)
            .then(pl.lit("sparse"))
            .otherwise(pl.lit("ok")),
        )
    return raw.select(
        "bank", "ticker", "variable", "n_quarters", "n_present",
        "pct_present", "category", "not_applicable", "status",
    )


# --------------------------------------------------------------------------
# Variable naming

# Columns are ``<family>_<category>``, but both halves contain underscores
# (``charge_offs_cre_owner_occupied``), so the split is by longest known
# prefix rather than by position.  Ordered longest-first: ``nco_rate`` must
# win over ``nco``, ``cq_fico`` over ``cq``.
FAMILIES: tuple[str, ...] = (
    "vintage_criticized", "vintage_total", "charge_offs", "nco_rate",
    "cq_internal", "cq_fico", "nonaccrual", "recoveries", "provision",
    "commitments", "securities", "deposits", "reserve", "pd_dpd", "loans",
    "risk_weighted_assets", "total_capital", "noninterest", "intangibles",
    "tangible", "interest", "acl", "nco", "dpd", "cq", "npa", "pd",
    "tier1", "cet1", "office",
)


def family_of(name: str) -> str:
    """The family a variable belongs to, or ``"other"`` for the singletons.

    ``other`` is not a failure -- roughly a fifth of each panel is one-off
    ratios and balance-sheet totals (``assets``, ``roa``, ``texas_ratio``)
    that form no family.  They are worth browsing as a group.
    """
    for fam in FAMILIES:
        if name == fam or name.startswith(fam + "_"):
            return fam
    # The portfolio-mix shares are named for their slice, not their family
    # (``cre_pct``, ``credit_card_pct``), so no prefix catches them.
    if name.endswith("_pct"):
        return "mix"
    return "other"


def value_columns(panel: str) -> list[str]:
    """The measured columns -- no metadata, no ``__source`` companions.

    This is the number that matters when someone says the panel has 548
    columns: on the EDGAR side 271 of those are provenance companions to the
    277 measures, so the browsable width is roughly half what a spreadsheet
    shows.
    """
    cols = load_panel(panel).columns
    return [c for c in cols if c not in META and not c.endswith(SOURCE_SUFFIX)]


def source_column(panel: str, variable: str) -> str | None:
    """The provenance companion for ``variable``, if the panel records one.

    Only the EDGAR build does.  The FFIEC panel needs no per-cell provenance:
    every value on it comes off the same filed form, and the one thing that
    *is* reconstructed -- the pre-2012 thrift quarters -- is flagged per row
    by ``n_tfr_backfilled`` instead.
    """
    col = variable + SOURCE_SUFFIX
    return col if col in load_panel(panel).columns else None


# --------------------------------------------------------------------------
# Seams -- the places a series breaks for a reason that is not the bank

@dataclass(frozen=True)
class Seam:
    quarter: str
    label: str
    tickers: tuple[str, ...]  # empty means every bank
    why: str


# Discontinuities in the entity, not the disclosure.  These are the ones the
# README calls out: a series that steps here stepped because the firm changed
# shape, and reading it as growth is the documented way to get a wrong answer.
LINEAGE_SEAMS: tuple[Seam, ...] = (
    Seam("2019Q4", "SunTrust merger", ("TFC",),
         "TFC files on BB&T's CIK; the merger roughly doubles the balance sheet."),
    Seam("2022Q1", "CIT consolidated", ("FCNCA",),
         "FCNCA excludes CIT before this quarter."),
    Seam("2023Q2", "SVB acquired", ("FCNCA",),
         "FCNCA excludes SVB before this quarter."),
    Seam("2008Q3", "WaMu consolidated", ("JPM",),
         "The quarter the Call Report first consolidates Washington Mutual; "
         "the loan book jumps by WaMu's entirety and reads as growth."),
)


def _calendar(panel: str) -> pl.DataFrame:
    """The panel's quarter labels against their period-end dates.

    Charts plot on a real time axis rather than on 54 ordinal string ticks,
    so every frame that reaches a chart carries ``period`` alongside
    ``quarter``.
    """
    return (
        load_panel(panel)
        .select("quarter", "period")
        .unique()
        .sort("quarter")
    )


def cecl_seams(panel: str) -> tuple[Seam, ...]:
    """The CECL adoption quarter, read off the panel rather than assumed.

    Almost everyone adopted at 2020Q1, but not everyone -- the CARES Act let
    a bank defer, and at least one in this universe did.  Drawing the seam at
    a hardcoded 2020Q1 would put the line in the wrong place for that bank,
    on the one chart whose whole purpose is to say where the meaning changed.
    """
    df = load_panel(panel)
    if "basis" not in df.columns:
        return ()
    first = (
        df.filter(pl.col("basis") == "cecl")
        .group_by("ticker")
        .agg(quarter=pl.col("quarter").min())
    )
    return tuple(
        Seam(row["quarter"], "CECL adoption", (row["ticker"],),
             "The allowance changes meaning here; the step in reserve "
             "coverage looks exactly like a credit event.")
        for row in first.iter_rows(named=True)
    )


def seams_for(panel: str, tickers: list[str]) -> pl.DataFrame:
    """Every seam that applies to the banks on screen, as rules to draw."""
    wanted = set(tickers)
    rows = [
        {"quarter": s.quarter, "label": s.label, "ticker": t, "why": s.why}
        for s in (*LINEAGE_SEAMS, *cecl_seams(panel))
        for t in (s.tickers or tuple(wanted))
        if t in wanted
    ]
    schema = {"quarter": pl.String, "label": pl.String,
              "ticker": pl.String, "why": pl.String}
    out = pl.DataFrame(rows, schema=schema)
    # A seam outside the panel's own window is not wrong, it is just not on
    # this chart -- the WaMu step predates the EDGAR panel's 2013 start.  The
    # join to the calendar drops those and supplies the date the x-axis uses.
    return out.join(_calendar(panel), on="quarter", how="inner")


# --------------------------------------------------------------------------
# Counting 1 -- the screen

def _windows(panel: str) -> pl.DataFrame:
    """Per variable: the first and last quarter carrying a value, and how many."""
    df = load_panel(panel)
    names = value_columns(panel)

    def agg(expr) -> pl.DataFrame:
        return df.select([expr(v).alias(v) for v in names]).unpivot(
            variable_name="variable"
        )

    first = agg(lambda v: pl.col("quarter").filter(pl.col(v).is_not_null()).min())
    last = agg(lambda v: pl.col("quarter").filter(pl.col(v).is_not_null()).max())
    cells = agg(lambda v: pl.col(v).is_not_null().sum())
    return (
        first.rename({"value": "first_q"})
        .join(last.rename({"value": "last_q"}), on="variable")
        .join(cells.rename({"value": "n_cells"}), on="variable")
    )


def screen(panel: str) -> pl.DataFrame:
    """One row per variable -- the table that answers "what is even here?".

    ``pct_median`` is taken over the banks the variable *applies to*.
    Including the not-applicable ones would rank a variable down for the
    banks that correctly have no such book: a custody bank reporting no
    credit-card charge-offs is not a gap in ``charge_offs_credit_card``.
    """
    cov = load_coverage(panel)
    rolled = cov.group_by("variable").agg(
        n_banks=pl.len(),
        banks_ok=(pl.col("status") == "ok").sum(),
        banks_sparse=(pl.col("status") == "sparse").sum(),
        banks_missing=(pl.col("status") == "missing").sum(),
        banks_na=(pl.col("status") == "n/a").sum(),
        pct_median=pl.col("pct_present")
        .filter(~pl.col("not_applicable"))
        .median()
        .round(1),
    )
    return (
        _windows(panel)
        .join(rolled, on="variable", how="left")
        .with_columns(
            family=pl.col("variable").map_elements(family_of, return_dtype=pl.String)
        )
        .select(
            "variable", "family", "banks_ok", "banks_sparse", "banks_missing",
            "banks_na", "pct_median", "first_q", "last_q", "n_cells",
        )
        .sort(["family", "variable"])
    )


# --------------------------------------------------------------------------
# Counting 2 -- provenance

def provenance(panel: str, variable: str) -> pl.DataFrame:
    """Where the populated cells of ``variable`` came from.

    On the EDGAR side this is recorded per cell.  On the FFIEC side there is
    nothing to record -- every value is off the same filed form -- except the
    one thing that genuinely is reconstructed, the pre-2012 thrift quarters,
    which are flagged per row rather than per cell.
    """
    df = load_panel(panel).filter(pl.col(variable).is_not_null())
    col = source_column(panel, variable)
    if col is not None:
        out = df.group_by(col).agg(n=pl.len()).rename({col: "source"})
        return out.with_columns(
            pl.col("source").fill_null("unrecorded")
        ).sort("n", descending=True)

    backfilled = pl.col("n_tfr_backfilled") > 0 if "n_tfr_backfilled" in df.columns \
        else pl.lit(False)
    return (
        df.with_columns(
            source=pl.when(backfilled)
            .then(pl.lit("tfr backfill"))
            .otherwise(pl.lit("call report"))
        )
        .group_by("source")
        .agg(n=pl.len())
        .sort("n", descending=True)
    )


# --------------------------------------------------------------------------
# Counting 3 -- the sample a set of variables actually leaves you

@dataclass(frozen=True)
class Sample:
    """The complete-case slice for a set of variables."""

    variables: tuple[str, ...]
    n_rows: int
    n_banks: int
    first_q: str | None
    last_q: str | None
    per_bank: pl.DataFrame
    dropped: tuple[str, ...]

    @property
    def pct_of_panel(self) -> float:
        return self._pct

    _pct: float = 0.0


def complete_case(panel: str, variables: list[str]) -> Sample:
    """Bank-quarters where *every* one of ``variables`` is populated.

    This is the number that decides a model, and it is not the coverage of
    the worst variable in the set.  The gaps are not aligned: two variables
    each 80% populated can intersect anywhere between 60% and 80% depending
    on whether the same banks are missing both.  Reading the per-variable
    coverage table and taking the minimum will overstate the sample, so the
    intersection is computed rather than inferred.
    """
    df = load_panel(panel)
    keys = ["ticker", "quarter"]
    kept = df.drop_nulls(subset=list(variables)) if variables else df
    per_bank = (
        kept.group_by("ticker")
        .agg(
            n_rows=pl.len(),
            first_q=pl.col("quarter").min(),
            last_q=pl.col("quarter").max(),
        )
        .sort("n_rows", descending=True)
    )
    all_banks = set(df["ticker"].unique().to_list())
    return Sample(
        variables=tuple(variables),
        n_rows=kept.height,
        n_banks=per_bank.height,
        first_q=kept["quarter"].min() if kept.height else None,
        last_q=kept["quarter"].max() if kept.height else None,
        per_bank=per_bank.select(*keys[:1], "n_rows", "first_q", "last_q"),
        dropped=tuple(sorted(all_banks - set(per_bank["ticker"].to_list()))),
        _pct=round(100 * kept.height / df.height, 1) if df.height else 0.0,
    )


def marginal_cost(
    panel: str, chosen: list[str], candidates: list[str]
) -> pl.DataFrame:
    """What each candidate variable would cost, in rows, if added to ``chosen``.

    The point of pricing them one at a time is that the cheap-looking
    variable and the expensive one are often the same coverage on paper --
    what separates them is whether their holes fall on banks the set has
    already lost.
    """
    base = complete_case(panel, chosen)
    rows = []
    for cand in candidates:
        if cand in chosen:
            continue
        after = complete_case(panel, [*chosen, cand])
        rows.append(
            {
                "variable": cand,
                "family": family_of(cand),
                "rows_after": after.n_rows,
                "rows_lost": base.n_rows - after.n_rows,
                "banks_after": after.n_banks,
                "banks_lost": base.n_banks - after.n_banks,
            }
        )
    schema = {
        "variable": pl.String, "family": pl.String, "rows_after": pl.Int64,
        "rows_lost": pl.Int64, "banks_after": pl.Int64, "banks_lost": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema).sort(["rows_lost", "variable"])


def availability(panel: str, variables: list[str]) -> pl.DataFrame:
    """Bank x quarter, flagged complete or not -- the grid behind the count."""
    df = load_panel(panel)
    if not variables:
        return df.select("ticker", "quarter", "period").with_columns(
            complete=pl.lit(True)
        )
    complete = pl.all_horizontal([pl.col(v).is_not_null() for v in variables])
    return df.select("ticker", "quarter", "period", complete=complete)


def series(
    panel: str, variable: str, highlight: list[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Peer band and highlighted lines for one variable.

    Returns ``(band, lines)``.  The band is the cross-sectional quartiles per
    quarter with the count behind them -- the count matters, because early
    quarters on a sparse variable can have a "peer median" of two banks.
    """
    df = load_panel(panel).select("ticker", "quarter", "period", variable)
    band = (
        df.drop_nulls(variable)
        .group_by("quarter", "period")
        .agg(
            p25=pl.col(variable).quantile(0.25),
            p50=pl.col(variable).median(),
            p75=pl.col(variable).quantile(0.75),
            n=pl.len(),
        )
        .sort("quarter")
    )
    # The highlighted banks keep their null quarters.  A line drawn over a
    # frame with the gaps dropped joins the two sides of a hole with a
    # straight segment that looks like data -- on a panel whose holes are the
    # whole subject, that is the one thing the chart must not do.
    lines = (
        df.filter(pl.col("ticker").is_in(highlight))
        .rename({variable: "value"})
        .sort(["ticker", "quarter"])
    )
    return band, lines
