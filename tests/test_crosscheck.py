"""Regression tests for the EDGAR-against-FFIEC cross-check.

``crosscheck`` writes ``source_diff.csv``, and the verdict column in it is a
work queue: ``differ`` means "go read the filings for this bank-quarter".  So
the failure mode that matters here is not a crash but a *miscategorised* row --
a verdict that sends a reader after a disagreement that is not there, or that
lets a real one pass as agreement.  Every test below is one of those.

The frames are built by hand rather than read from ``data/``, so each test
states the exact shape it is about and none of them need a build.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
import pytest

from callrpt_db import crosscheck

# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------


def _panel(rows: list[dict]) -> pl.DataFrame:
    """A minimal panel: whatever columns the rows name, plus the join key."""
    return pl.DataFrame(rows)


def _pair(
    variable: str,
    edgar_values: Sequence[float | None],
    ffiec_values: Sequence[float | None],
    *,
    ticker: str = "XYZ",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    periods = [f"20{20 + i // 4}Q{i % 4 + 1}" for i in range(len(edgar_values))]

    def frame(values: Sequence[float | None]) -> pl.DataFrame:
        return _panel(
            [
                {"ticker": ticker, "period": p, variable: v}
                for p, v in zip(periods, values, strict=True)
            ]
        )

    return frame(edgar_values), frame(ffiec_values)


def _compare(
    variable: str, edgar: pl.DataFrame, ffiec: pl.DataFrame, *, ratio: bool = False
) -> pl.DataFrame:
    return crosscheck.compare(
        edgar,
        ffiec,
        columns=() if ratio else (variable,),
        ratios=(variable,) if ratio else (),
    )


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def test_two_sources_reporting_the_same_zero_do_not_disagree() -> None:
    """Both sources saying "none" is agreement, not a disagreement.

    ``difference_pct`` divides by the EDGAR value, so it is null whenever that
    value is zero, and the verdict chain used to fall through such rows to
    ``differ``.  A bank that genuinely holds no card book -- ALLY at
    2020-12-31, on the shipped panels -- was reported as a source conflict in
    every quarter it held none, on both the balance and the ratio derived from
    it.  91 rows of the shipped ``source_diff.csv`` were this.
    """
    edgar, ffiec = _pair("loans_credit_card", [0.0], [0.0])
    diff = _compare("loans_credit_card", edgar, ffiec)

    assert diff["verdict"].to_list() == ["agree"]
    assert diff["difference"].to_list() == [0.0]
    # The percentage stays null: it is genuinely undefined, and the verdict no
    # longer depends on it being otherwise.
    assert diff["difference_pct"].to_list() == [None]


def test_a_book_appearing_out_of_nothing_is_still_a_disagreement() -> None:
    """The zero-case fix must not swallow the one conflict zero can produce.

    EDGAR reading zero where FFIEC reads a real balance is the most
    interesting disagreement there is -- one source lost the book entirely.
    It has the same null ``difference_pct`` as the case above, so it is the
    row a careless fix silently reclassifies as agreement.
    """
    edgar, ffiec = _pair("loans_ci", [0.0], [5_000.0])
    diff = _compare("loans_ci", edgar, ffiec)

    assert diff["verdict"].to_list() == ["differ"]
    assert diff["difference"].to_list() == [5_000.0]


def test_a_missing_reading_is_not_a_disagreement() -> None:
    """A null on one side is ``one_source_only``, never ``differ``.

    The two panels start in different years and disclose different things;
    absence is a coverage fact, and mixing it into ``differ`` would bury the
    real conflicts under it.
    """
    edgar, ffiec = _pair("assets", [None, 100.0], [50.0, None])
    diff = _compare("assets", edgar, ffiec)

    assert diff["verdict"].to_list() == ["one_source_only", "one_source_only"]


def test_rounding_between_thousands_and_units_is_not_a_disagreement() -> None:
    """The Call Report is filed in thousands and XBRL in units.

    A sub-tolerance gap on a large book is the two filing conventions, not a
    conflict, and flagging it would make the report useless by volume.
    """
    edgar, ffiec = _pair("loans_total", [500_000_000_000.0], [500_000_000_500.0])

    assert _compare("loans_total", edgar, ffiec)["verdict"].to_list() == ["agree"]


def test_a_gap_just_past_the_tolerance_is_a_disagreement() -> None:
    """The tolerance is a threshold, not a shrug -- pin which side is which."""
    tol = crosscheck.AGREEMENT_TOLERANCE_PCT
    edgar, ffiec = _pair("loans_total", [100.0, 100.0], [100.0 + tol, 100.0 + tol * 2])

    assert _compare("loans_total", edgar, ffiec)["verdict"].to_list() == ["agree", "differ"]


def test_the_gap_is_signed_from_edgar_to_ffiec() -> None:
    """``difference`` is ffiec - edgar, and the ratio is ffiec / edgar.

    The direction carries the meaning: a ratio below 1 is a holding company
    lending outside its charters, above 1 is intercompany lending running the
    other way.  Inverting either would flip the interpretation of the whole
    report while every number in it stayed plausible.
    """
    edgar, ffiec = _pair("loans_total", [100.0], [60.0])
    diff = _compare("loans_total", edgar, ffiec)

    assert diff["difference"].to_list() == [-40.0]
    assert diff["difference_pct"].to_list() == [-40.0]
    assert diff["ffiec_over_edgar"].to_list() == [pytest.approx(0.6)]


def test_a_negative_edgar_reading_keeps_the_gap_signed() -> None:
    """``difference_pct`` divides by ``abs(edgar)`` on purpose.

    Charge-offs and provisions go negative.  Dividing by a signed negative
    would flip the sign of the gap, so a FFIEC reading *above* EDGAR would
    report as a shortfall.
    """
    edgar, ffiec = _pair("provision_total", [-100.0], [-50.0])
    diff = _compare("provision_total", edgar, ffiec)

    assert diff["difference"].to_list() == [50.0]
    assert diff["difference_pct"].to_list() == [50.0]


def test_levels_and_ratios_are_labelled_apart() -> None:
    """``kind`` is what lets a reader weigh a row.

    A level gap is usually entity scope; a ratio gap is not, because scope
    largely divides out of a ratio.  The two must not be read together.
    """
    edgar, ffiec = _pair("cre_pct", [10.0], [12.0])
    assert _compare("cre_pct", edgar, ffiec, ratio=True)["kind"].to_list() == ["ratio"]

    edgar, ffiec = _pair("loans_cre_total", [10.0], [12.0])
    assert _compare("loans_cre_total", edgar, ffiec)["kind"].to_list() == ["level"]


# --------------------------------------------------------------------------
# What is compared at all
# --------------------------------------------------------------------------


def test_a_column_only_one_source_carries_is_not_compared() -> None:
    """Comparison needs both sides; a one-sided column is silently skipped.

    Not an error: the panels legitimately carry different columns, and
    ``compare`` is called with the full default list against whatever the
    build produced.
    """
    edgar = _panel([{"ticker": "XYZ", "period": "2020Q1", "loans_total": 1.0, "equity": 5.0}])
    ffiec = _panel([{"ticker": "XYZ", "period": "2020Q1", "loans_total": 1.0}])
    diff = crosscheck.compare(edgar, ffiec, columns=("loans_total", "equity"), ratios=())

    assert diff["variable"].unique().to_list() == ["loans_total"]


def test_a_bank_quarter_only_one_panel_has_does_not_appear() -> None:
    """The join is inner, and the absence is reported by ``coverage_gained``.

    The FFIEC panel reaches back to 2001 and covers the IHCs that file no
    10-K.  Carrying those into ``source_diff.csv`` as null-versus-value rows
    would swamp the file with a fact that is already reported elsewhere.
    """
    edgar = _panel([{"ticker": "XYZ", "period": "2020Q1", "loans_total": 1.0}])
    ffiec = _panel(
        [
            {"ticker": "XYZ", "period": "2020Q1", "loans_total": 1.0},
            {"ticker": "XYZ", "period": "2001Q1", "loans_total": 2.0},
            {"ticker": "IHC", "period": "2020Q1", "loans_total": 3.0},
        ]
    )
    diff = crosscheck.compare(edgar, ffiec, columns=("loans_total",), ratios=())

    assert diff.height == 1
    assert diff["period"].to_list() == ["2020Q1"]


@pytest.mark.parametrize(
    "edgar, ffiec",
    [
        (pl.DataFrame(), pl.DataFrame()),
        (pl.DataFrame(), _panel([{"ticker": "XYZ", "period": "2020Q1", "loans_total": 1.0}])),
        (_panel([{"ticker": "XYZ", "period": "2020Q1", "loans_total": 1.0}]), pl.DataFrame()),
    ],
)
def test_an_absent_panel_gives_an_empty_report_not_a_crash(
    edgar: pl.DataFrame, ffiec: pl.DataFrame
) -> None:
    """The EDGAR panel is optional -- ``build_call_panel.py`` runs without it."""
    assert crosscheck.compare(edgar, ffiec).is_empty()


# --------------------------------------------------------------------------
# The summary
# --------------------------------------------------------------------------


def test_pct_agree_counts_every_quarter_compared_including_one_sided_ones() -> None:
    """``pct_agree`` is over all quarters joined, not over the comparable ones.

    Pinning the denominator because it is the one figure in the summary that
    can be read two ways: a variable both sources agree on wherever both
    report it, but which EDGAR only carries half the time, reads 50 -- not
    100.  ``n_one_source`` beside it is what makes that recoverable, so the
    two must stay together.
    """
    edgar, ffiec = _pair("assets", [100.0, 100.0, None, None], [100.0, 100.0, 50.0, 50.0])
    summary = crosscheck.summarise(_compare("assets", edgar, ffiec))

    row = summary.to_dicts()[0]
    assert (row["quarters"], row["n_agree"], row["n_one_source"]) == (4, 2, 2)
    assert row["pct_agree"] == 50.0


def test_the_headline_ratio_is_the_median_not_the_mean() -> None:
    """One bad quarter must not move the structural ratio it is measured against.

    The report's whole method is "stable ratio = structure, excursion = bug".
    A mean would let the excursion drag the baseline toward itself and make
    the excursion look smaller than it is.
    """
    edgar, ffiec = _pair("loans_total", [100.0] * 4, [60.0, 60.0, 60.0, 6.0])
    summary = crosscheck.summarise(_compare("loans_total", edgar, ffiec))

    assert summary["median_ratio"][0] == pytest.approx(0.6)


def test_a_single_quarter_has_no_instability_rather_than_zero() -> None:
    """One observation cannot be stable or unstable, and must not read stable.

    ``ratio_std`` is null for a group of one.  Filling it with zero would put
    every newly-added bank at the bottom of the instability queue as if it
    had been checked.
    """
    edgar, ffiec = _pair("loans_total", [100.0], [60.0])
    summary = crosscheck.summarise(_compare("loans_total", edgar, ffiec))

    assert summary["ratio_std"].to_list() == [None]
    assert crosscheck.unstable(summary).is_empty()


# --------------------------------------------------------------------------
# Instability -- where bugs live
# --------------------------------------------------------------------------


def test_a_stable_gap_is_structure_and_is_not_flagged() -> None:
    """A steady 60% is a fact about the bank, not a thing to go read.

    State Street's charters really do hold a fraction of the holding
    company's book, quarter after quarter.
    """
    edgar, ffiec = _pair("loans_total", [100.0] * 6, [60.0, 60.5, 59.5, 60.2, 59.8, 60.0])
    summary = crosscheck.summarise(_compare("loans_total", edgar, ffiec))

    assert crosscheck.unstable(summary).is_empty()


def test_a_ratio_that_jumps_for_one_quarter_is_flagged() -> None:
    """The same 60% dropping to 6% for one quarter is the pipeline, not the bank."""
    edgar, ffiec = _pair("loans_total", [100.0] * 6, [60.0, 60.0, 6.0, 60.0, 60.0, 60.0])
    summary = crosscheck.summarise(_compare("loans_total", edgar, ffiec))
    shaky = crosscheck.unstable(summary)

    assert shaky.height == 1
    assert shaky["instability"][0] > 0.15


def test_the_most_unstable_pair_is_reported_first() -> None:
    """The output is a work queue, so its order is part of what it says."""
    frames = []
    for ticker, values in (("AAA", [60.0, 60.0, 30.0]), ("BBB", [60.0, 60.0, 6.0])):
        edgar, ffiec = _pair("loans_total", [100.0] * 3, values, ticker=ticker)
        frames.append(_compare("loans_total", edgar, ffiec))
    shaky = crosscheck.unstable(crosscheck.summarise(pl.concat(frames)))

    assert shaky["ticker"].to_list() == ["BBB", "AAA"]


# --------------------------------------------------------------------------
# What the second source buys
# --------------------------------------------------------------------------


def _ffiec_universe() -> pl.DataFrame:
    return _panel(
        [
            {"ticker": t, "period": p, "bank": f"{t} Inc", "loans_total": v, "n_charters": n}
            for t, p, v, n in (
                ("BIG", "2020Q1", 900.0, 2),
                ("BIG", "2020Q2", 1_000.0, 3),
                ("IHC", "2020Q1", 400.0, 1),
                ("IHC", "2020Q2", 500.0, 1),
            )
        ]
    )


def test_a_firm_with_no_edgar_counterpart_is_marked_as_gained() -> None:
    """The seven IHCs file no 10-K at all -- this column is why the package exists."""
    edgar = _panel([{"ticker": "BIG", "period": "2020Q1", "loans_total": 1.0}])
    gained = crosscheck.coverage_gained(edgar, _ffiec_universe())

    assert dict(zip(gained["ticker"], gained["in_edgar_panel"], strict=True)) == {
        "BIG": True,
        "IHC": False,
    }


def test_the_firms_edgar_cannot_reach_are_listed_first() -> None:
    """This table is read top-down for what the FFIEC build adds.

    The sort is ascending on ``in_edgar_panel`` and descending on size, so
    False sorts above True.  Inverting it would bury the answer under the
    firms both sources already have.
    """
    edgar = _panel([{"ticker": "BIG", "period": "2020Q1", "loans_total": 1.0}])
    gained = crosscheck.coverage_gained(edgar, _ffiec_universe())

    assert gained["ticker"].to_list() == ["IHC", "BIG"]


def test_the_latest_loan_balance_is_the_latest_quarters_not_the_last_rows() -> None:
    """``loans_total_latest`` must follow ``last_quarter``, whatever the row order.

    It is a ``last`` aggregate, so on an unsorted frame it returns the last
    *row* of the group -- a real balance, from the wrong quarter, printed
    beside a correctly computed ``last_quarter``.  It is also the sort key of
    the table, so the ordering goes wrong with it.  The holding panel happens
    to arrive sorted today; this pins that the function does not depend on it.
    """
    shuffled = _ffiec_universe().sort("period", descending=True)
    gained = crosscheck.coverage_gained(pl.DataFrame(), shuffled)

    row = gained.filter(pl.col("ticker") == "BIG").to_dicts()[0]
    assert row["last_quarter"] == "2020Q2"
    assert row["loans_total_latest"] == 1_000.0


def test_the_charter_count_is_the_most_a_firm_ever_had() -> None:
    """A firm that shed a charter still needed the wider read while it had it."""
    gained = crosscheck.coverage_gained(pl.DataFrame(), _ffiec_universe())

    assert gained.filter(pl.col("ticker") == "BIG")["max_charters"][0] == 3


def test_coverage_is_reported_with_no_edgar_panel_to_compare_against() -> None:
    """The FFIEC build stands alone.

    ``--no-crosscheck`` and a missing ``panel.parquet`` must still leave this
    table meaningful, since it is the one output that does not need a second
    source to mean anything.
    """
    gained = crosscheck.coverage_gained(pl.DataFrame(), _ffiec_universe())

    assert gained["in_edgar_panel"].to_list() == [False, False]
