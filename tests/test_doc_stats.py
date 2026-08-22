"""Regression tests for the documentation-count sync.

``sync_doc_stats.py`` rewrites numbers *inside committed prose*, and CI runs it
on every push to ``main``.  That combination makes its failure modes unusually
expensive: a bad substitution is committed, is not visibly wrong, and is quoted
back by anyone reading the README.  The three that matter are

* rewriting the wrong span -- prose eaten, a table broken, or a number put
  where a sentence was;
* refreshing one half of a sentence and not the other, which states something
  no build ever produced; and
* reporting a document as current when a sentinel in it was never reached.

Only the pure half of the script is exercised here -- the substitution table
and the rewriter.  ``refresh_stats`` and ``load_stats`` read ``data/`` and a
completed build, and are covered by the integration test at the end, which
skips when the panels are not present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    """Import the script by path -- ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "sync_doc_stats", ROOT / "scripts" / "sync_doc_stats.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sync = _load_script()


# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------


def _stats(**overrides: Any) -> dict:
    """The smallest stats file ``values`` will accept, plus overrides."""
    base: dict[str, Any] = {
        "edgar": {
            "rows": 1_648,
            "columns": 548,
            "banks": 31,
            "first_quarter": "2013Q1",
            "last_quarter": "2026Q2",
        },
        "ffiec": {
            "rows": 3_763,
            "columns": 279,
            "banks": 38,
            "first_quarter": "2001Q1",
            "last_quarter": "2026Q2",
        },
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# The substitution table
# --------------------------------------------------------------------------


def test_thousands_are_grouped_and_bank_counts_are_not() -> None:
    """The rendering is part of the prose and has to match how it reads.

    "3,763 rows" and "38 firms" are both right; "3763 rows" and "38,0 firms"
    are both wrong in a way a diff makes noisy and a reader makes nothing of.
    """
    table = sync.values(_stats())

    assert table["ffiec-rows"] == "3,763"
    assert table["ffiec-banks"] == "38"
    assert table["edgar-cols"] == "548"


def test_a_window_is_rendered_with_the_dash_the_documents_use() -> None:
    """The README's table cells hold an en dash with spaces around it.

    A hyphen here would be rewritten into the document on every run and show
    up as a perpetual diff that never converges.
    """
    assert sync.values(_stats())["ffiec-window"] == "2001Q1 – 2026Q2"


def test_the_extent_line_names_firms_for_ffiec_and_banks_for_edgar() -> None:
    """The two panels count different things and the prose says so.

    The FFIEC panel's entities are firms including the IHCs; the EDGAR
    panel's are banks that file a 10-K.  Swapping the nouns would make the
    README claim a coverage it does not have.
    """
    table = sync.values(_stats())

    assert table["edgar-extent"] == "1,648 rows, 31 banks, 548 columns"
    assert table["ffiec-extent"] == "3,763 rows, 38 firms, 279 columns"


def test_a_figure_the_build_did_not_record_produces_no_key_at_all() -> None:
    """An absent figure must be a missing key, not a zero or an empty string.

    ``sync_file`` reports an unknown key as an error, so a figure that no
    build produced surfaces loudly.  Rendering it as "0" would instead write
    a confident, wrong number into the prose.
    """
    table = sync.values(_stats())

    assert "tests" not in table
    assert "ffiec-bank-quarters" not in table
    assert "thrift-sought" not in table


def test_the_shipped_build_is_unprefixed_and_the_comparison_is_prefixed() -> None:
    """The documents quote the ``--tfr`` build as the panel, and the plain
    build only as the thing the thrift backfill is measured against.

    Emitting the plain build unprefixed would put the pre-backfill row count
    in the sentence that describes the shipped file.
    """
    table = sync.values(
        _stats(
            build_tfr={"bank_quarters": 3_763, "firms": 38},
            build_plain={"bank_quarters": 3_247, "firms": 38},
        )
    )

    assert table["ffiec-bank-quarters"] == "3,763"
    assert table["plain-bank-quarters"] == "3,247"


def test_the_added_rows_are_derived_rather_than_quoted() -> None:
    """"before", "after" and "added" are three halves of one sentence.

    Deriving the third from the other two is what stops a refresh from
    updating two of them and leaving a subtraction that no longer holds.
    """
    table = sync.values(
        _stats(build_tfr={"coverage_delta": {"before": 1_000, "after": 3_763}})
    )

    assert table["ffiec-delta-before"] == "1,000"
    assert table["ffiec-delta-after"] == "3,763"
    assert table["ffiec-delta-added"] == "2,763"


def test_a_partition_percentage_is_not_computed_from_nothing() -> None:
    """A build that checked no charter-quarters has no tie rate.

    The percentage is guarded on the denominator; without that the sync would
    raise on the one build shape -- ``--tfr``, which suppresses the holding
    partition check -- that the documents specifically compare against.
    """
    table = sync.values(
        _stats(build_tfr={"partition": {"checked": 0, "tied": 0}})
    )

    assert table["ffiec-partition-tied"] == "0"
    assert "ffiec-partition-tied-pct" not in table


def test_a_partition_percentage_is_computed_when_there_is_a_denominator() -> None:
    table = sync.values(
        _stats(build_tfr={"partition": {"checked": 1_000, "tied": 994}})
    )

    assert table["ffiec-partition-tied-pct"] == "99.4"


def test_lineage_figures_prefer_the_shipped_build_over_the_comparison() -> None:
    """Lineage is a property of the registries, not of which build ran.

    Both records carry it, so the shipped one must win -- otherwise the
    number in the prose depends on the order the two builds happened to be
    refreshed in.
    """
    table = sync.values(
        _stats(
            build_tfr={"lineage": {"predecessors": 214}},
            build_plain={"lineage": {"predecessors": 199}},
        )
    )

    assert table["lineage-predecessors"] == "214"


def test_a_bank_short_of_the_window_is_named_not_merely_counted() -> None:
    """"N of 31 banks have all 28 quarters" is only safe if the exceptions
    are recomputable too.

    Both halves come from the same panel, which is the stated rule for
    wrapping a sentence at all.  A count without the exceptions would let the
    two drift apart with nothing to catch it.
    """
    table = sync.values(
        _stats(
            edgar={
                **_stats()["edgar"],
                "early": {
                    "window_quarters": 28,
                    "banks_complete": 29,
                    "banks_short": [
                        {"ticker": "FCNCA", "quarters": 24},
                        {"ticker": "TFC", "quarters": 27},
                    ],
                },
            }
        )
    )

    assert table["edgar-early-quarters"] == "28"
    assert table["edgar-early-complete"] == "29"
    assert table["edgar-early-short"] == "FCNCA 24, TFC 27"


def test_no_bank_short_of_the_window_reads_as_none_not_as_blank() -> None:
    """An empty list must render as a word, since it sits mid-sentence.

    "...except none" reads; "...except " does not, and would look like the
    sync had failed halfway.
    """
    table = sync.values(
        _stats(
            edgar={
                **_stats()["edgar"],
                "early": {
                    "window_quarters": 28,
                    "banks_complete": 31,
                    "banks_short": [],
                },
            }
        )
    )

    assert table["edgar-early-short"] == "none"


# --------------------------------------------------------------------------
# Rewriting a document
# --------------------------------------------------------------------------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_only_the_number_moves_and_the_sentence_around_it_does_not(
    tmp_path: Path,
) -> None:
    """The point of the sentinels: prose keeps reading like prose."""
    doc = "The panel holds <!--stats:ffiec-rows-->3,741<!--/stats--> bank-quarters.\n"
    path = _write(tmp_path, doc)

    sync.sync_file(path, {"ffiec-rows": "3,763"}, write=True)

    assert path.read_text(encoding="utf-8") == (
        "The panel holds <!--stats:ffiec-rows-->3,763<!--/stats--> bank-quarters.\n"
    )


def test_two_sentinels_on_one_line_are_rewritten_independently(tmp_path: Path) -> None:
    """These sit in markdown table cells, so several share a line.

    The match is non-greedy for exactly this reason.  A greedy one would
    treat everything from the first opening marker to the last closing one as
    a single value and swallow the cells between them.
    """
    path = _write(
        tmp_path,
        "| <!--stats:edgar-rows-->1<!--/stats--> | <!--stats:ffiec-rows-->2<!--/stats--> |\n",
    )

    sync.sync_file(path, {"edgar-rows": "1,648", "ffiec-rows": "3,763"}, write=True)

    assert path.read_text(encoding="utf-8") == "| <!--stats:edgar-rows-->1,648<!--/stats--> | <!--stats:ffiec-rows-->3,763<!--/stats--> |\n"


def test_a_table_spanning_several_lines_is_not_swallowed(tmp_path: Path) -> None:
    """The match stops at the end of a line, so an unclosed marker cannot
    consume the rows beneath it."""
    doc = (
        "| a | <!--stats:edgar-rows-->1<!--/stats--> |\n"
        "| b | second row |\n"
        "| c | <!--stats:ffiec-rows-->2<!--/stats--> |\n"
    )
    path = _write(tmp_path, doc)

    sync.sync_file(path, {"edgar-rows": "1,648", "ffiec-rows": "3,763"}, write=True)
    updated = path.read_text(encoding="utf-8")

    assert "| b | second row |" in updated
    assert updated.count("\n") == 3


def test_a_value_already_current_reports_no_change(tmp_path: Path) -> None:
    """``--check`` fails on any reported change, so "current" has to mean it."""
    path = _write(tmp_path, "<!--stats:ffiec-rows-->3,763<!--/stats-->\n")

    assert sync.sync_file(path, {"ffiec-rows": "3,763"}, write=True) == []


def test_a_stale_value_is_reported_with_both_sides(tmp_path: Path) -> None:
    """The message is the whole of what CI shows, so it must name the drift."""
    path = _write(tmp_path, "<!--stats:ffiec-rows-->3,741<!--/stats-->\n")

    changes = sync.sync_file(path, {"ffiec-rows": "3,763"}, write=False)

    assert len(changes) == 1
    assert "3,741" in changes[0] and "3,763" in changes[0]


def test_a_check_run_does_not_touch_the_file(tmp_path: Path) -> None:
    """``--check`` is what CI runs on a clean tree; writing there would leave
    the working copy dirty and turn a report into a failure."""
    doc = "<!--stats:ffiec-rows-->3,741<!--/stats-->\n"
    path = _write(tmp_path, doc)

    sync.sync_file(path, {"ffiec-rows": "3,763"}, write=False)

    assert path.read_text(encoding="utf-8") == doc


def test_an_unknown_key_leaves_the_document_alone(tmp_path: Path) -> None:
    """A key the table has no value for must not be blanked.

    Substituting an empty string would silently delete a number from the
    prose, and the run would report success.
    """
    doc = "<!--stats:not-a-real-key-->42<!--/stats-->\n"
    path = _write(tmp_path, doc)

    changes = sync.sync_file(path, {"ffiec-rows": "3,763"}, write=True)

    assert path.read_text(encoding="utf-8") == doc
    assert len(changes) == 1
    assert "unknown sentinel" in changes[0]


def test_a_value_containing_a_backreference_is_inserted_literally(tmp_path: Path) -> None:
    """Substitution goes through ``re.sub``, and rendered values are data.

    ``edgar-early-short`` interpolates tickers, and a value that happened to
    look like a group reference would be expanded rather than written.
    """
    path = _write(tmp_path, "<!--stats:edgar-early-short-->old<!--/stats-->\n")

    sync.sync_file(path, {"edgar-early-short": r"\1 and \g<2>"}, write=True)

    assert path.read_text(encoding="utf-8") == (
        "<!--stats:edgar-early-short-->\\1 and \\g<2><!--/stats-->\n"
    )


def test_a_document_with_no_sentinels_is_left_untouched(tmp_path: Path) -> None:
    """Most wiki pages carry none, and are passed in anyway by ``--wiki``."""
    doc = "# A page with prose and no counts in it.\n"
    path = _write(tmp_path, doc)

    assert sync.sync_file(path, {"ffiec-rows": "3,763"}, write=True) == []
    assert path.read_text(encoding="utf-8") == doc


def test_a_missing_target_is_dropped_rather_than_raising(tmp_path: Path) -> None:
    """``--wiki`` points at a checkout that may not be there."""
    assert sync.targets(tmp_path / "no-such-wiki") == [ROOT / "README.md"]


def test_wiki_pages_are_synced_in_a_stable_order(tmp_path: Path) -> None:
    """The change log is read by a human, and ordering it by filesystem
    iteration would reshuffle it between runs for no reason."""
    for name in ("Traps.md", "Quick-Start.md", "FAQ.md"):
        (tmp_path / name).write_text("x\n", encoding="utf-8")

    names = [p.name for p in sync.targets(tmp_path)]

    assert names == ["README.md", "FAQ.md", "Quick-Start.md", "Traps.md"]


# --------------------------------------------------------------------------
# Against the committed panels
# --------------------------------------------------------------------------

_PANELS = (ROOT / "data" / "panel.parquet", ROOT / "data" / "call_panel.parquet")
needs_panels = pytest.mark.skipif(
    not all(p.exists() for p in _PANELS), reason="the committed panels are not present"
)


@needs_panels
def test_the_committed_readme_has_no_sentinel_the_table_cannot_fill() -> None:
    """An unknown key in the README is always an error, per the script itself.

    It means a sentinel was written into prose that no build produces a value
    for, and that number is then frozen at whatever was typed -- the exact
    failure the sentinels exist to end.
    """
    table = sync.values(sync.load_stats(verify=False))
    changes = sync.sync_file(ROOT / "README.md", table, write=False)

    assert [c for c in changes if "unknown sentinel" in c] == []


@needs_panels
def test_the_stats_file_still_describes_the_committed_panels() -> None:
    """``load_stats`` cross-checks the recorded facts against the parquets.

    This is the guard that stops a stale ``panel_stats.json`` being copied
    into the documents as confidently as a current one, so it is worth
    knowing it passes on what is committed rather than only that it exists.
    """
    stats = sync.load_stats(verify=True)

    assert stats["ffiec"]["rows"] > 0
