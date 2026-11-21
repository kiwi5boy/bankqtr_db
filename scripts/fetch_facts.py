"""Download and cache companyfacts + submissions for the whole bank universe."""

from __future__ import annotations

import argparse
import logging
import sys

from bankqtr_db import edgar
from bankqtr_db.config import BANKS, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="re-download cached payloads; without it a new quarter is never seen",
    )
    args = ap.parse_args(argv)
    ensure_dirs()
    failures = []
    for bank in BANKS:
        for label, fn in (("facts", edgar.company_facts), ("subs", edgar.submissions)):
            try:
                payload = fn(bank.cik, refresh=args.refresh)
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append((bank.name, label, str(exc)))
                print(f"FAIL {bank.ticker:5s} {label}: {exc}", flush=True)
            else:
                n = (
                    len(payload.get("facts", {}).get("us-gaap", {}))
                    if label == "facts"
                    else 0
                )
                print(f"ok   {bank.ticker:5s} {label} ({n} us-gaap tags)", flush=True)
    if failures:
        print(f"\n{len(failures)} failures", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
