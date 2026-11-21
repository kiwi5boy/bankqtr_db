"""Download FFIEC bulk data into the local cache.

uv run python scripts/fetch_call.py                       # 2001Q1 onward, ~850 MB
uv run python scripts/fetch_call.py --since 2013-01-01
uv run python scripts/fetch_call.py --structure-only       # NIC files only
uv run python scripts/fetch_call.py --list-periods

Fetching and building are separate on purpose, exactly as they are for the
EDGAR side: ``build_call_panel.py`` reads only what is already cached and never
touches the network, so a rebuild is reproducible.

One quarter is a 6-9 MB zip holding every schedule for every filer -- 4,400
institutions today, 8,900 in 2001 -- so a 2013-start window is ~350 MB and a
2001-start one ~850 MB, at about two quarters a minute behind the one rate
limiter in ``cdr.py``.  The NIC structure files (attributes, relationships
and the transformations table the lineage walk reads) are fetched first.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from callrpt_db import cdr, nic
from callrpt_db.config import ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fetch_call")


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _fetch_tfr(periods: list[str], *, refresh: bool = False) -> None:
    """Cache BankFind financials for the depositories that filed no Call Report.

    Which institutions those are depends on the lineages, and so on the
    universe.  This asks ``panel.unfiled_depositories`` -- the question
    ``build_call_panel.py --tfr`` itself asks -- rather than reading a built
    panel's ``insured_not_filing`` column.  The column is narrower: a quarter
    in which none of a firm's charters filed has no row to carry it, so a firm
    whose only depository was a thrift (Ameriprise, 2006-2011) was never
    fetched, and the build then warned that its certificate was not cached.

    The universe is widened to the comparators so a ``--comparators`` build
    is covered too.  The cache is per certificate, so the superset costs a few
    requests once and satisfies every narrower build.
    """
    from callrpt_db import panel, tfr

    holdings = panel.select_holdings(comparators=True)
    gap = panel.unfiled_depositories(
        periods, holdings, panel.window_lineages(periods, holdings)
    )
    _, info = tfr.backfill(gap, periods, refresh=refresh)
    log.info(
        "tfr: %d rows cached for %d of %d non-filing depositories; identities %s",
        info.get("rows", 0),
        info.get("resolved", 0),
        info.get("sought", 0),
        info.get("identities", {}),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", type=_date, default=dt.date(2001, 1, 1))
    ap.add_argument("--until", type=_date, default=None)
    ap.add_argument("--refresh", action="store_true", help="re-download cached files")
    ap.add_argument(
        "--structure-only",
        action="store_true",
        help="fetch only the NIC entity/relationship files",
    )
    ap.add_argument(
        "--list-periods",
        action="store_true",
        help="print the periods CDR offers and exit",
    )
    ap.add_argument(
        "--tfr",
        action="store_true",
        help=(
            "also fetch FDIC BankFind financials for the insured depositories "
            "that filed a Thrift Financial Report rather than a Call Report "
            "(WaMu, Golden West, Sovereign, IndyMac); reads only the cached "
            "Call Reports and NIC files, so no panel need be built first"
        ),
    )
    args = ap.parse_args(argv)

    ensure_dirs()

    if args.list_periods:
        periods = cdr.available_periods()
        for period in sorted(periods):
            print(f"{period}\t{cdr.period_label(period)}\tid={periods[period]}")
        print(f"\n{len(periods)} periods offered")
        return 0

    log.info("fetching NIC structure files")
    nic.fetch_structure(refresh=args.refresh)
    if args.structure_only:
        return 0

    until = args.until or dt.datetime.now(dt.UTC).date()
    wanted = cdr.periods_between(args.since, until)

    with cdr.new_client() as client:
        offered = cdr.available_periods(client)
        missing = [p for p in wanted if p not in offered]
        if missing:
            log.warning(
                "CDR does not offer %d of the requested periods: %s",
                len(missing),
                ", ".join(missing),
            )
        todo = [p for p in wanted if p in offered]
        log.info("%d periods to consider (%s..%s)", len(todo), todo[0], todo[-1])

        failed: list[tuple[str, str]] = []
        for i, period in enumerate(todo, 1):
            try:
                path = cdr.fetch_period(
                    period, refresh=args.refresh, client=client, periods=offered
                )
            except Exception as exc:  # noqa: BLE001 - one bad quarter must not lose the rest
                log.error("%s failed: %s", period, exc)
                failed.append((period, str(exc)))
                continue
            log.info(
                "[%d/%d] %s -> %s (%.1f MB)",
                i,
                len(todo),
                period,
                path.name,
                path.stat().st_size / 1e6,
            )

    have = cdr.cached_periods()
    log.info("cache now holds %d periods (%s..%s)", len(have), have[0], have[-1])

    if args.tfr:
        _fetch_tfr(have, refresh=args.refresh)
    if failed:
        log.error("%d periods failed: %s", len(failed), ", ".join(p for p, _ in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
