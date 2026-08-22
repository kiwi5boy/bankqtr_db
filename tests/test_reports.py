"""Regression tests for the two reports shipped beside the EDGAR panel.

``panel_coverage.csv`` and ``panel_flags.csv`` are how a user decides whether a
number is safe to use, so a defect in either is worse than a defect in the
panel: it makes a wrong number look checked.  The two failure modes both look
like success --

* a structural absence reported as a gap, which inflates the work queue until
  nobody reads it; and
* a real gap or a real outlier reported as fine, which is the one thing these
  files exist to prevent.

Every test below is one of those, on frames built by hand so no build is needed.
"""

from __future__ import annotations

import polars as pl
import pytest

from bankqtr_db import reconcile
from bankqtr_db.config import Bank

# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------

BANKS = (
    Bank(name="Regional Bancorp", cik="0000000001", ticker="REG", category="regional"),
    Bank(name="Custody Trust", cik="0000000002", ticker="CUS", category="custody"),
)


def _panel(rows: list[dict]) -> pl.DataFrame:
    """A panel with the full key, so ``PANEL_KEY`` selections resolve.

    Value columns are declared ``Float64`` even where every row is null.  A
    real panel is built with an explicit schema and does the same; letting a
    wholly empty column arrive as ``Null`` dtype would make the fixture skip
    it for a reason no build ever produces.
    """
    key = ("bank", "ticker", "cik", "quarter", "period")
    filled = []
    for i, row in enumerate(rows):
        ticker = row.get("ticker", "REG")
        filled.append(
            {
                "bank": row.get("bank", f"{ticker} Inc"),
                "ticker": ticker,
                "cik": row.get("cik", "0000000001"),
                "quarter": row.get("quarter", f"2020Q{i % 4 + 1}"),
                "period": row.get("period", f"2020-0{i % 4 * 3 + 3}-30"),
                **{k: v for k, v in row.items() if k not in ("bank", "ticker", "cik")},
            }
        )
    schema = {c: pl.String for c in key}
    for row in filled:
        for column in row:
            if column not in key:
                schema.setdefault(
                    column, pl.String if column.endswith("__source") else pl.Float64
                )
    return pl.DataFrame(filled, schema=schema)


def _quarters(ticker: str, column: str, values: list[float | None]) -> list[dict]:
    return [
        {"ticker": ticker, "quarter": f"2020Q{i + 1}", column: v}
        for i, v in enumerate(values)
    ]


# --------------------------------------------------------------------------
# Coverage: a gap that is not a gap
# --------------------------------------------------------------------------


def test_a_custody_bank_with_no_cre_book_is_not_a_coverage_failure() -> None:
    """"A custody bank with no CRE book is not a coverage failure; a regional
    bank with no CRE column is."  That distinction is the whole report.

    Both banks below have an empty ``loans_cre_total``.  Reporting them the
    same way would put a structural fact at the top of the work queue and keep
    it there forever, since no amount of extraction effort can fill it.
    """
    panel = _panel(
        _quarters("REG", "loans_cre_total", [None, None])
        + _quarters("CUS", "loans_cre_total", [None, None])
    )
    report = reconcile.coverage_report(panel, banks=BANKS)
    status = dict(
        zip(
            report.filter(pl.col("variable") == "loans_cre_total")["ticker"],
            report.filter(pl.col("variable") == "loans_cre_total")["status"],
            strict=True,
        )
    )

    assert status == {"CUS": "n/a", "REG": "missing"}


def test_a_bank_outside_the_registry_is_not_silently_given_a_category() -> None:
    """An unknown ticker gets an empty category, and so no exemptions.

    Defaulting it to a category with exemptions would let a genuine gap be
    reported as structure for any bank the registry had not caught up with.
    """
    panel = _panel(_quarters("NEW", "loans_cre_total", [None, None]))
    report = reconcile.coverage_report(panel, banks=BANKS)

    assert report["category"].to_list() == [""]
    assert report["not_applicable"].to_list() == [False]
    assert report["status"].to_list() == ["missing"]


@pytest.mark.parametrize(
    "values, status",
    [
        ([1.0, 1.0, 1.0, 1.0], "ok"),
        ([1.0, 1.0, 1.0, None], "ok"),
        ([1.0, 1.0, None, None], "sparse"),
        ([1.0, None, None, None], "sparse"),
        ([None, None, None, None], "missing"),
    ],
)
def test_partial_coverage_is_sparse_and_only_nothing_is_missing(
    values: list[float | None], status: str
) -> None:
    """"Sparse" and "missing" are different work.

    A half-populated column is an extraction to improve; an empty one is an
    extraction that never fired.  Collapsing the two -- in either direction --
    loses which of the two a user is looking at.  The boundary sits at 60%.
    """
    panel = _panel(_quarters("REG", "loans_cre_total", values))
    report = reconcile.coverage_report(panel, banks=BANKS)

    assert report.filter(pl.col("variable") == "loans_cre_total")["status"][0] == status


def test_provenance_columns_are_not_reported_as_variables() -> None:
    """``*__source`` companions are metadata, not things to have coverage of.

    The EDGAR panel carries hundreds of them.  Letting them into the report
    would roughly double it with rows nobody can act on.
    """
    panel = _panel(
        [
            {"ticker": "REG", "quarter": "2020Q1", "loans_total": 1.0, "loans_total__source": "xbrl"},
        ]
    )
    report = reconcile.coverage_report(panel, banks=BANKS)

    assert report["variable"].to_list() == ["loans_total"]


def test_an_empty_panel_gives_an_empty_report_rather_than_raising() -> None:
    """The reports run at the end of a build that may have produced nothing."""
    assert reconcile.coverage_report(pl.DataFrame()).is_empty()
    assert reconcile.gap_summary(pl.DataFrame()).is_empty()
    assert reconcile.quality_flags(pl.DataFrame()).is_empty()


# --------------------------------------------------------------------------
# The work queue
# --------------------------------------------------------------------------


def test_structural_absences_are_kept_out_of_the_work_queue() -> None:
    """``gap_summary`` ranks what is worth extracting, so n/a rows must not count.

    Were they counted, a variable only two of thirty-one banks can ever have
    would sort to the top of the queue as the least covered thing in the
    panel, every single build.
    """
    panel = _panel(
        _quarters("REG", "loans_cre_total", [1.0, 1.0])
        + _quarters("CUS", "loans_cre_total", [None, None])
    )
    summary = reconcile.gap_summary(reconcile.coverage_report(panel, banks=BANKS))
    row = summary.filter(pl.col("variable") == "loans_cre_total").to_dicts()[0]

    assert (row["n_banks"], row["n_ok"], row["pct_banks_ok"]) == (1, 1, 100.0)


def test_the_least_covered_variable_is_ranked_first() -> None:
    """The queue is read top-down, so its order is the recommendation."""
    panel = _panel(
        _quarters("REG", "loans_total", [1.0, 1.0])
        + _quarters("REG", "loans_office_cre", [None, None])
    )
    summary = reconcile.gap_summary(reconcile.coverage_report(panel, banks=BANKS))

    assert summary["variable"].to_list()[0] == "loans_office_cre"


# --------------------------------------------------------------------------
# Quality flags: the things that should never be true
# --------------------------------------------------------------------------


def _flag_names(panel: pl.DataFrame) -> set[str]:
    flagged = reconcile.quality_flags(panel)
    if flagged.is_empty():
        return set()
    return {
        c
        for c in flagged.columns
        if c not in reconcile.PANEL_KEY and c != "n_flags" and bool(flagged[c][0])
    }


def test_an_unpopulated_row_raises_no_flag() -> None:
    """A null is a coverage fact and belongs in the coverage report, not here.

    Every check is a comparison, and a comparison against null is null.  If
    those nulls became flags, ``panel_flags.csv`` would list every thinly
    covered bank-quarter in the panel and say nothing about quality.
    """
    panel = _panel(
        [
            {
                "ticker": "REG",
                "quarter": "2020Q1",
                "loans_total": None,
                "acl_total": None,
                "loans_cre_total": None,
                "loans_construction": None,
                "reserve_coverage": None,
                "cre_pct": None,
            }
        ]
    )

    assert reconcile.quality_flags(panel).is_empty()


def test_a_negative_balance_is_flagged() -> None:
    """A loan book cannot be negative; a sign error in a fallback parse can be."""
    panel = _panel([{"ticker": "REG", "quarter": "2020Q1", "loans_total": -1.0}])

    assert "neg_loans" in _flag_names(panel)


def test_a_component_larger_than_its_parent_is_flagged() -> None:
    """Construction is inside CRE, so exceeding it means one of them is misread.

    This is the signature of a dimension member landing in the wrong column --
    the number is plausible on its own and only the relationship gives it away.
    """
    panel = _panel(
        [
            {
                "ticker": "REG",
                "quarter": "2020Q1",
                "loans_construction": 60.0,
                "loans_cre_total": 50.0,
            }
        ]
    )

    assert "construction_exceeds_cre" in _flag_names(panel)


def test_a_rounding_sized_overshoot_is_not_flagged() -> None:
    """The 2% tolerance is deliberate.

    A component can exceed its parent by a hair when the two come from
    different tables of the same filing, rounded differently.  Flagging that
    would fill the file with arithmetic and hide the real inversions.
    """
    panel = _panel(
        [
            {
                "ticker": "REG",
                "quarter": "2020Q1",
                "loans_construction": 50.5,
                "loans_cre_total": 50.0,
            }
        ]
    )

    assert reconcile.quality_flags(panel).is_empty()


def test_a_genuinely_thin_reserve_is_not_flagged_as_implausible() -> None:
    """The reserve band is wide because two real banks sit near its edges.

    Schwab's pledged-asset lending carries about 0.08% and Synchrony's card
    book about 10.5%.  A band drawn around the typical regional would flag
    both as errors, in every quarter, forever -- and a reader who learns to
    ignore those stops reading the file.
    """
    panel = _panel(
        [
            {"ticker": "REG", "quarter": "2020Q1", "reserve_coverage": 0.08},
            {"ticker": "REG", "quarter": "2020Q2", "reserve_coverage": 10.5},
        ]
    )

    assert reconcile.quality_flags(panel).is_empty()


def test_an_impossible_reserve_is_still_flagged() -> None:
    """A reserve of a quarter of the book is a scale error, not a credit view."""
    panel = _panel([{"ticker": "REG", "quarter": "2020Q1", "reserve_coverage": 25.0}])

    assert "reserve_coverage_implausible" in _flag_names(panel)


def test_a_mix_that_overshoots_the_book_is_flagged() -> None:
    """``mix_coverage_pct`` above 110 means the categories double-count.

    The README tells users to check this column before ranking peers, so it
    is the one number whose incoherence must never pass silently.
    """
    panel = _panel([{"ticker": "REG", "quarter": "2020Q1", "mix_coverage_pct": 130.0}])

    assert "mix_incoherent" in _flag_names(panel)


def test_a_share_of_the_book_above_one_hundred_percent_is_flagged() -> None:
    """Every ``_pct`` ratio is a share of loans and cannot exceed the loans.

    The bound is applied by iterating ``RATIOS``, so this also pins that a
    newly added percentage ratio is bounded by construction rather than by
    somebody remembering to add a check for it.
    """
    panel = _panel([{"ticker": "REG", "quarter": "2020Q1", "cre_pct": 140.0}])

    assert "cre_pct_out_of_range" in _flag_names(panel)


def test_the_worst_bank_quarter_is_listed_first() -> None:
    """``panel_flags.csv`` is triage, and ``n_flags`` is the triage order."""
    panel = _panel(
        [
            {"ticker": "REG", "quarter": "2020Q1", "loans_total": 100.0, "cre_pct": 140.0},
            {
                "ticker": "REG",
                "quarter": "2020Q2",
                "loans_total": -100.0,
                "cre_pct": 140.0,
                "reserve_coverage": 25.0,
            },
        ]
    )
    flagged = reconcile.quality_flags(panel)

    assert flagged["quarter"].to_list() == ["2020Q2", "2020Q1"]
    assert flagged["n_flags"].to_list() == [3, 1]


def test_a_clean_panel_produces_no_flag_file_rows() -> None:
    """The report must be empty when nothing is wrong, not merely all-False.

    A file of thousands of unflagged rows reads as "checks ran" rather than
    "nothing failed", and buries the rows that did fail.
    """
    panel = _panel(
        [
            {
                "ticker": "REG",
                "quarter": "2020Q1",
                "loans_total": 100.0,
                "acl_total": 1.0,
                "loans_cre_total": 40.0,
                "loans_construction": 10.0,
                "reserve_coverage": 1.0,
                "cre_pct": 40.0,
            }
        ]
    )

    assert reconcile.quality_flags(panel).is_empty()


def test_a_check_whose_columns_are_absent_is_skipped_rather_than_raising() -> None:
    """Every check is guarded on the columns it needs, and the guard must work.

    ``components_exceed_total`` sums whichever of the two component columns
    the panel carries.  With neither present that sum was over an empty list,
    and ``pl.sum_horizontal([])`` raises -- at the moment the argument is
    built, which is *before* the guard that was written to skip the check ever
    runs.  So the guard could not fire, and a panel carrying only
    ``loans_total`` took the whole report down with a ComputeError about fold
    widths.  Builds run without the fallback, and any caller handing over a
    subset of columns, are exactly this shape.
    """
    panel = _panel([{"ticker": "REG", "quarter": "2020Q1", "loans_total": 100.0}])

    assert reconcile.quality_flags(panel).is_empty()


def test_the_component_check_still_fires_on_whichever_half_is_present() -> None:
    """Skipping the check when neither column is there must not skip it when
    one is.  A commercial book alone can already exceed the total."""
    panel = _panel(
        [
            {
                "ticker": "REG",
                "quarter": "2020Q1",
                "loans_total": 100.0,
                "loans_commercial_total": 150.0,
            }
        ]
    )

    assert "components_exceed_total" in _flag_names(panel)


def test_thin_book_tickers_are_read_from_the_registry() -> None:
    """The custody and broker exemptions come from one place.

    ``THIN_LOAN_BOOK`` and ``NOT_APPLICABLE`` are two spellings of the same
    judgement, and a bank exempted by one but not the other would be reported
    inconsistently between the coverage report and the panel itself.
    """
    assert reconcile.thin_book_tickers(BANKS) == {"CUS"}
    for category in reconcile.THIN_LOAN_BOOK:
        assert category in reconcile.NOT_APPLICABLE
