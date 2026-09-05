"""Regression tests for the site's variable dictionary.

The dictionary's failure mode is not an exception, it is an omission or a
wrong sentence. A column with no entry is invisible to someone searching for
it, and they conclude the panel does not have it; a category with no label
ships a row that says ``fund_finance`` and reads as a bug. Both are silent,
so both are asserted here.

These run against the committed panels at ``data/`` and are skipped when they
are absent, the same way the integration tests elsewhere in this suite are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_dictionary as bd

pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "panel.parquet").exists()
    or not (ROOT / "data" / "call_panel.parquet").exists(),
    reason="the committed panels are not present",
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return bd.collect()


def test_every_column_has_a_definition(payload: dict) -> None:
    """No column may be missing from the dictionary.

    ``collect`` raises on an unresolved column rather than dropping it, which
    is the whole point: a dictionary that quietly omits the variable someone
    is looking for is worse than no dictionary at all. This asserts the count
    it produced matches the panels it read.
    """
    per_panel: dict[str, int] = {}
    for entry in payload["variables"]:
        per_panel[entry["panel"]] = per_panel.get(entry["panel"], 0) + 1

    for panel, meta in payload["panels"].items():
        assert per_panel.get(panel) == meta["variables"], panel


def test_an_unresolvable_column_fails_the_build(monkeypatch) -> None:
    """The guard has to actually fire, not just exist.

    A resolver that returns ``None`` for an unknown name is only useful if
    something stops on it, so this removes a rule and checks the build refuses
    rather than shipping 270 entries and calling it done.
    """
    monkeypatch.setattr(bd, "resolve_edgar", lambda name, family: None)
    with pytest.raises(SystemExit) as raised:
        bd.collect()
    assert "must have a definition" in str(raised.value)


def test_every_canonical_category_has_a_label() -> None:
    """A category added to the taxonomy must be named before it can ship.

    Otherwise the page shows the slug, which reads as an unfinished feature
    on a page whose entire job is to explain what the slugs mean.
    """
    bd._check_labels()


def test_a_category_without_a_label_fails_the_build(monkeypatch) -> None:
    monkeypatch.setitem(bd.LOAN_LABELS, "ci", None)
    monkeypatch.delitem(bd.LOAN_LABELS, "ci")
    with pytest.raises(SystemExit) as raised:
        bd._check_labels()
    assert "no label" in str(raised.value)


def test_labels_are_names_not_definitions(payload: dict) -> None:
    """The first column of a table is a name, and names are short.

    A VarDef's ``description`` is a definition -- "Loans held for investment,
    gross of allowance" -- and using it as the label made every sliced row
    unreadable once a category was appended. The definition still travels, in
    the description.
    """
    for entry in payload["variables"]:
        assert entry["label"], entry["name"]
        assert len(entry["label"]) <= 80, (entry["name"], entry["label"])
        assert entry["description"], entry["name"]


def test_no_label_is_just_the_column_name(payload: dict) -> None:
    """"charge_offs_construction" is not a meaning.

    Two thirds of the FFIEC item table carries no ``description`` -- it was
    obvious to whoever added it -- and the first cut of this page fell back to
    the raw slug, printing the name twice and explaining nothing on a page
    whose only job is to explain. A composed name from the family and the
    category is the floor.
    """
    for entry in payload["variables"]:
        assert entry["label"] != entry["name"], entry["name"]


def test_multi_schedule_items_are_lists_not_tuples(payload: dict) -> None:
    """Six FFIEC items are read off whichever schedule the filer used.

    ``ItemSpec.schedule`` is a string for most of them and a tuple for those
    six. Passed through unconverted it survived ``json.dumps`` as a list, so
    the file no longer matched what the builder produced and every ``--check``
    reported a stale dictionary; it also rendered as a Python repr on the page.
    """
    seen_multi = False
    for entry in payload["variables"]:
        schedules = entry["definition"].get("schedules")
        if schedules is None:
            continue
        assert isinstance(schedules, list), entry["name"]
        assert all(isinstance(s, str) for s in schedules), entry["name"]
        seen_multi = seen_multi or len(schedules) > 1
    assert seen_multi, "expected at least one item spanning two schedules"


def test_the_payload_survives_a_json_round_trip(payload: dict) -> None:
    """``--check`` compares the committed file to a fresh build.

    Anything that does not round-trip through JSON -- a tuple, a set, a date --
    makes that comparison fail every time, which turns the freshness gate into
    noise that gets ignored.
    """
    import json

    assert json.loads(json.dumps(payload)) == payload


def test_ffiec_entries_name_their_filing_line(payload: dict) -> None:
    """The point of the FFIEC half is that every number traces to a form line.

    An entry with no schedule, no formula and no components is one nobody can
    check against the Call Report, which is the one thing this panel is for.
    """
    for entry in payload["variables"]:
        if entry["panel"] != "ffiec":
            continue
        definition = entry["definition"]
        traceable = (
            definition.get("mdrm_items")
            or definition.get("numerator")
            or definition.get("sum_of")
            or definition.get("minus")
            or definition.get("derived_from")
        )
        # The remainder are the composites described in prose by hand.
        if not traceable:
            assert entry["name"] in bd.FFIEC_SINGLETONS or (
                entry["name"] in bd.CAPITAL_RATIOS
            ), entry["name"]


def test_counts_come_from_the_panel_not_the_definitions(payload: dict) -> None:
    """The counts describe the built panel, and have to agree with each other.

    A window that runs backwards, or a bank count of zero against populated
    cells, is the kind of wrong that looks fine in a table cell.
    """
    for entry in payload["variables"]:
        assert entry["n_cells"] > 0, entry["name"]
        assert entry["n_banks"] > 0, entry["name"]
        assert entry["first_q"] <= entry["last_q"], entry["name"]

        panel = payload["panels"][entry["panel"]]
        assert entry["n_banks"] <= panel["banks"], entry["name"]
        assert entry["n_cells"] <= panel["rows"], entry["name"]

        if entry["sources"]:
            assert sum(entry["sources"].values()) == entry["n_cells"], entry["name"]


def test_no_defined_ffiec_item_is_missing_from_the_dictionary(payload: dict) -> None:
    """The FFIEC item table is the definition of what that panel holds.

    The dictionary is built from the panel's columns, so anything the build
    defines but never writes out would be absent without a word -- and someone
    who cannot find ``deposits_brokered`` concludes the panel has no such
    thing. Today the two sets coincide exactly; this is what says so.
    """
    from callrpt_db import mdrm

    listed = {e["name"] for e in payload["variables"] if e["panel"] == "ffiec"}
    defined = {spec.name for spec in mdrm.ALL_ITEMS}
    assert defined - listed == set()
