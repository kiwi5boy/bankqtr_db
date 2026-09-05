"""Regression tests for the variable explorer's counting.

Same rule as the rest of this suite: every test corresponds to a way of
counting that produces a plausible-looking but wrong answer.  The explorer is
a browsing tool, so nothing here raises when it is wrong -- it just draws a
different number, and someone picks a variable set on the strength of it.

The fixtures build a two-bank panel in memory and point the loaders at it
through ``set_data_base``, which is the same indirection the WASM build uses.
That keeps the tests independent of ``data/out`` and exercises that path.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps"))

import explore

QUARTERS = ("2019Q4", "2020Q1", "2020Q2", "2020Q3")


def _period(quarter: str) -> dt.date:
    ends = {"Q1": (3, 31), "Q2": (6, 30), "Q3": (9, 30), "Q4": (12, 31)}
    month, day = ends[quarter[4:]]
    return dt.date(int(quarter[:4]), month, day)


def _rows(**series: list) -> list[dict]:
    """One row per bank-quarter; ``series`` maps a column to its values."""
    out = []
    pairs = [(t, q) for t in ("AAA", "BBB") for q in QUARTERS]
    for i, (ticker, quarter) in enumerate(pairs):
        row = {
            "bank": ticker,
            "ticker": ticker,
            "quarter": quarter,
            "period": _period(quarter),
            "basis": "incurred" if quarter == "2019Q4" else "cecl",
        }
        row.update({name: values[i] for name, values in series.items()})
        out.append(row)
    return out


def _cov(**by_variable: list[tuple[str, float, bool]]) -> list[dict]:
    """Coverage rows in the shipped EDGAR schema, statuses at the real cuts."""
    return [
        {
            "bank": ticker,
            "ticker": ticker,
            "variable": variable,
            "n_quarters": 4,
            "n_present": round(pct / 25),
            "pct_present": pct,
            "category": "regional",
            "not_applicable": na,
            "status": (
                "n/a"
                if na
                else "missing"
                if pct == 0
                else "sparse"
                if pct < 60
                else "ok"
            ),
        }
        for variable, entries in by_variable.items()
        for ticker, pct, na in entries
    ]


@pytest.fixture
def panel_at(tmp_path: Path):
    """Write a panel + coverage pair to disk and point the loaders at it."""

    def build(rows: list[dict], coverage: list[dict]) -> Path:
        pl.DataFrame(rows).write_parquet(tmp_path / "panel.parquet")
        pl.DataFrame(coverage).write_csv(tmp_path / "panel_coverage.csv")
        explore.set_data_base(str(tmp_path))
        return tmp_path

    yield build
    explore.set_data_base(None)


# --------------------------------------------------------------------------


def test_complete_case_is_an_intersection_not_the_weakest_variable(panel_at):
    """Two variables at 7 and 6 rows can leave 5 together, not 6.

    The holes fall in different quarters, so reading the coverage table and
    taking the minimum promises a sample that does not exist.  This is the
    number the sample builder is for, and the one most likely to be eyeballed
    off the screen table instead.
    """
    panel_at(
        _rows(
            a=[1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            b=[1.0, 2.0, 3.0, None, 5.0, None, 7.0, 8.0],
        ),
        _cov(a=[("AAA", 75.0, False)], b=[("AAA", 75.0, False)]),
    )
    assert explore.complete_case("edgar", ["a"]).n_rows == 7
    assert explore.complete_case("edgar", ["b"]).n_rows == 6
    assert explore.complete_case("edgar", ["a", "b"]).n_rows == 5


def test_complete_case_reports_the_banks_it_drops(panel_at):
    """A set that silently loses a bank changes what the panel is about."""
    panel_at(
        _rows(a=[1.0, 2.0, 3.0, 4.0, None, None, None, None]),
        _cov(a=[("AAA", 100.0, False), ("BBB", 0.0, False)]),
    )
    sample = explore.complete_case("edgar", ["a"])
    assert sample.dropped == ("BBB",)
    assert sample.n_banks == 1
    assert sample.pct_of_panel == 50.0


def test_marginal_cost_is_priced_against_the_chosen_set(panel_at):
    """A candidate is cheap or expensive only relative to what is already in.

    ``b``'s second hole falls on a row ``a`` has already dropped, so adding it
    to a set holding ``a`` costs one row, not two.  Pricing every candidate
    against the whole panel would rank the two identically.
    """
    panel_at(
        _rows(
            a=[1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            b=[1.0, None, 3.0, None, 5.0, 6.0, 7.0, 8.0],
        ),
        _cov(a=[("AAA", 75.0, False)], b=[("AAA", 75.0, False)]),
    )
    alone = explore.marginal_cost("edgar", [], ["b"]).row(0, named=True)
    after_a = explore.marginal_cost("edgar", ["a"], ["b"]).row(0, named=True)
    assert alone["rows_lost"] == 2
    assert after_a["rows_lost"] == 1


def test_not_applicable_banks_are_excluded_from_the_median(panel_at):
    """A bank with no such book is not a gap in the variable.

    A custody bank reporting no credit-card charge-offs would otherwise pull
    the median to zero and bury a variable that is fully populated everywhere
    it applies.
    """
    panel_at(
        _rows(cards=[1.0] * 8),
        _cov(cards=[("AAA", 100.0, False), ("BBB", 0.0, True)]),
    )
    row = explore.screen("edgar").row(0, named=True)
    assert row["pct_median"] == 100.0
    assert row["banks_na"] == 1
    assert row["banks_missing"] == 0


def test_source_companions_are_not_offered_as_variables(panel_at):
    """``loans_total__source`` is provenance, not something to model.

    Half the EDGAR panel's columns are these companions.  Offering them would
    double the apparent width of the panel with columns that never hold a
    number.
    """
    rows = _rows(loans_total=[1.0] * 8)
    for row in rows:
        row["loans_total__source"] = "xbrl"
    panel_at(rows, _cov(loans_total=[("AAA", 100.0, False)]))

    assert explore.value_columns("edgar") == ["loans_total"]
    assert explore.source_column("edgar", "loans_total") == "loans_total__source"


def test_metadata_columns_are_not_offered_as_variables(panel_at):
    """``basis`` and ``quarter`` are always populated and never a measure.

    Left in, they sort to the top of any coverage ranking and read as the
    best-covered variables in the panel.
    """
    panel_at(_rows(assets=[1.0] * 8), _cov(assets=[("AAA", 100.0, False)]))
    offered = explore.value_columns("edgar")
    assert offered == ["assets"]
    for name in ("bank", "ticker", "quarter", "period", "basis"):
        assert name not in offered


@pytest.mark.parametrize(
    ("variable", "family"),
    [
        ("nco_rate_ci", "nco_rate"),
        ("nco_ci", "nco"),
        ("cq_fico_prime", "cq_fico"),
        ("cq_classified", "cq"),
        ("charge_offs_cre_owner_occupied", "charge_offs"),
        ("vintage_criticized_2021", "vintage_criticized"),
        ("credit_card_pct", "mix"),
        ("roa", "other"),
    ],
)
def test_family_is_the_longest_matching_prefix(variable: str, family: str):
    """``nco_rate_ci`` is a rate, not a charge-off level.

    Splitting on the first underscore, or on the first prefix that happens to
    match, files rates under ``nco`` and puts a ratio in a family of dollar
    amounts -- which is how a variable gets picked for the wrong reason.
    """
    assert explore.family_of(variable) == family


def test_series_keeps_the_holes_in_a_highlighted_bank(panel_at):
    """A line drawn over a gap-free frame invents a trajectory.

    Dropping the null quarters before charting joins the two sides of a hole
    with a straight segment that looks exactly like data.  On a tool whose
    subject is missing data, that is the one thing the chart must not do, so
    the highlighted rows keep their nulls and the line breaks.
    """
    panel_at(
        _rows(a=[1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        _cov(a=[("AAA", 75.0, False)]),
    )
    band, lines = explore.series("edgar", "a", ["AAA"])

    assert lines.height == len(QUARTERS)
    assert lines["value"].null_count() == 1
    # The band is over the banks that did report, so 2020Q1 stands on one.
    assert band.filter(pl.col("quarter") == "2020Q1")["n"].item() == 1


def test_cecl_seam_is_read_off_the_panel_not_assumed(panel_at):
    """One bank in the real universe deferred adoption past 2020Q1.

    Drawing the seam at a hardcoded quarter puts the line in the wrong place
    for that bank, on the one annotation whose entire job is to say where the
    numbers changed meaning.
    """
    rows = _rows(a=[1.0] * 8)
    for row in rows:
        if row["ticker"] == "BBB":
            row["basis"] = "cecl" if row["quarter"] == "2020Q3" else "incurred"
    panel_at(rows, _cov(a=[("AAA", 100.0, False)]))

    seams = {(s.tickers[0], s.quarter) for s in explore.cecl_seams("edgar")}
    assert ("AAA", "2020Q1") in seams
    assert ("BBB", "2020Q3") in seams


def test_seams_outside_the_window_are_dropped(panel_at):
    """The WaMu step predates the EDGAR panel, so it is not on that chart.

    A rule at a quarter the panel does not hold either lands on the axis edge
    or stretches the time scale to reach a year with no data on it.
    """
    panel_at(_rows(a=[1.0] * 8), _cov(a=[("AAA", 100.0, False)]))
    seams = explore.seams_for("edgar", ["AAA", "BBB"])

    assert seams.height  # the CECL seams survive
    assert "2008Q3" not in seams["quarter"].to_list()
    assert seams["period"].dtype == pl.Date


def test_availability_marks_a_row_complete_only_if_every_variable_is(panel_at):
    """The grid behind the count has to agree with the count."""
    panel_at(
        _rows(
            a=[1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            b=[1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0],
        ),
        _cov(a=[("AAA", 75.0, False)], b=[("AAA", 75.0, False)]),
    )
    grid = explore.availability("edgar", ["a", "b"])
    assert grid["complete"].sum() == explore.complete_case("edgar", ["a", "b"]).n_rows


def test_ffiec_coverage_uses_the_same_status_thresholds(tmp_path: Path):
    """A status has to mean the same thing on both panels.

    The FFIEC coverage table ships without a status column, so one is derived.
    Different cut-points would make "ok" mean 60% on one panel and something
    else on the other, and the screen would be comparing two questions.
    """
    pl.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "variable": ["loans_total"] * 3,
            "quarters": [100, 100, 100],
            "populated": [0, 59, 60],
            "pct": [0.0, 59.0, 60.0],
        }
    ).write_csv(tmp_path / "call_panel_coverage.csv")
    pl.DataFrame(
        {
            "bank": ["AAA"],
            "ticker": ["AAA"],
            "quarter": ["2020Q1"],
            "period": [_period("2020Q1")],
            "basis": ["cecl"],
            "loans_total": [1.0],
        }
    ).write_parquet(tmp_path / "call_panel.parquet")

    explore.set_data_base(str(tmp_path))
    try:
        got = explore.load_coverage("ffiec")
        assert got["status"].to_list() == ["missing", "sparse", "ok"]
        # FFIEC has no not-applicable concept: every filer files the same form.
        assert got["not_applicable"].to_list() == [False, False, False]
    finally:
        explore.set_data_base(None)
