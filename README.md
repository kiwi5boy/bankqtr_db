# bankqtr_db — DFAST bank peer benchmarking

Bank-quarter panel databases for peer benchmarking, built **twice from two
independent sources**. One row per bank-quarter, columns per variable, with
provenance and coverage reporting attached.

| Package | Source | Entity | Output | Window |
|---|---|---|---|---|
| `bankqtr_db` | SEC EDGAR — XBRL, filing HTML, IR supplements | the holding company, as it reports itself | `panel.parquet` | <!--stats:edgar-window-->2013Q1 – 2026Q2<!--/stats--> |
| `callrpt_db` | FFIEC — CDR bulk Call Reports, NIC structure data; FDIC BankFind for the thrifts | its bank charters, summed | `call_panel.parquet` | <!--stats:ffiec-window-->2001Q1 – 2026Q2<!--/stats--> |

They are not redundant. The EDGAR panel is what the market sees and covers the
whole consolidated firm; the FFIEC panel is a regulatory filing on a fixed form,
so its categories cross-foot, it reaches back to 2001, and it covers the seven US
intermediate holding companies of foreign banks that file no 10-K at all. Where
the two disagree, `source_diff.csv` says by how much and whether the gap is
stable.

Before 2012Q1 that second panel is missing the thrifts by construction: an
OTS-supervised savings institution filed a Thrift Financial Report, not a Call
Report, so CDR holds nothing for Washington Mutual, Golden West, Countrywide
Bank, IndyMac or Sovereign in the quarters they existed. The `--tfr` build
supplements those quarters from FDIC BankFind, which compiles both report types
onto one schema. Those rows carry `source = "tfr"` on coarser loan categories,
so a reconstructed quarter is always distinguishable from a filed one. Without
that flag the pre-2012 history is a floor, and `n_insured_not_filing` says where.

📖 **[Full documentation is in the wiki.](https://github.com/kiwi5boy/bankqtr_db/wiki)**
📊 **[Panel downloads and charts.](https://kiwi5boy.github.io/bankqtr_db/)**
📓 **[A worked example, start to finish.](examples/peer_nco_benchmark.ipynb)**

## Quick start

```bash
uv sync
export BANKQTR_UA="you@yourdomain.com"          # SEC requires a real contact

uv run python scripts/fetch_facts.py            # companyfacts + submissions
uv run python scripts/fetch_instances.py --since 2013-01-01
uv run python scripts/fetch_ir.py --since 2020-01-01       # IR supplements
uv run python scripts/build_panel.py --since 2013-01-01 --html-fallback --html-since 2013-01-01 --ir
```

The FFIEC build is independent and can be run on its own. It starts at **2001Q1**
and follows each firm back through its predecessors:

```bash
uv run python scripts/fetch_call.py                      # 102 quarters, ~850 MB
uv run python scripts/resolve_rssd.py --write-config     # bank -> RSSD, once
uv run python scripts/build_call_panel.py                # 2001Q1 onward, with lineage
uv run python scripts/fetch_call.py --tfr                # the thrifts CDR has nothing for
uv run python scripts/build_call_panel.py --tfr          # ...folded in
```

Fetching and building are separate on purpose: every `build_*` script reads only
what is already cached and never touches the network, so a rebuild is
reproducible. A full 2013-start EDGAR build takes a few minutes.

See **[Quick Start](https://github.com/kiwi5boy/bankqtr_db/wiki/Quick-Start)**
for the narrower windows and what each flag costs.

## Outputs

Both builds land in `data/out/`:

| File | Contents |
|---|---|
| `panel.parquet` / `.csv` | the EDGAR bank-quarter panel — <!--stats:edgar-extent-->1,648 rows, 31 banks, 548 columns<!--/stats--> |
| `call_panel.parquet` / `.csv` | the FFIEC panel — <!--stats:ffiec-extent-->3,763 rows, 38 firms, 279 columns<!--/stats--> |
| `call_panel_charters.parquet` | one row per bank charter per quarter, with its predecessor |
| `panel_coverage.csv` / `call_panel_coverage.csv` | per bank and variable: how many quarters are populated |
| `panel_flags.csv` / `call_panel_flags.csv` | bank-quarters failing a sanity check |
| `source_diff.csv` | EDGAR against FFIEC, per bank-quarter per variable |
| `rssd_lineage.csv` | every predecessor of every firm: who was absorbed into whom, when, how |

Full list and column reference: **[Outputs](https://github.com/kiwi5boy/bankqtr_db/wiki/Outputs)**
and **[Data Dictionary](https://github.com/kiwi5boy/bankqtr_db/wiki/Data-Dictionary)**.

## A worked example

**[`examples/peer_nco_benchmark.ipynb`](examples/peer_nco_benchmark.ipynb)** — a risk
analyst benchmarks one bank's net charge-off rate against its DFAST peers: where it
ranks, how that has moved since 2001, and whether the whole-book number is hiding an
outlier in C&I, CRE or card. It reads both panels straight from the release assets, so it
runs against a clean checkout with no build:

```bash
uv sync --group examples          # matplotlib and a kernel; not needed to build the panels
cd examples && uv run jupyter execute --inplace peer_nco_benchmark.ipynb
```

That re-runs it in place against the current release. To read or edit it interactively,
open it in VS Code or your own Jupyter — the `examples` group installs the kernel, and
the notebook is committed with its outputs, so it also reads fine without running at all.
The home bank is a variable in the first cell.

## Looking a variable up

**[The variable dictionary](https://kiwi5boy.github.io/bankqtr_db/dictionary.html)**
on the site lists every column in both panels — searchable by name, by meaning,
by MDRM item or by XBRL element. Each entry says what the number is, how it is
defined, and how many banks and quarters carry it:

| Panel | An entry says |
|---|---|
| FFIEC | the schedule and MDRM item it is read from, which prefix wins (`RCFD` before `RCON`, and why that matters), and the quarter it was first filed |
| EDGAR | the XBRL tag candidates in priority order, the axis and category it is sliced on, and what it rolls up into |
| Both | for a ratio or a roll-up, the exact formula over other columns; and the populated cell count, bank count and window from the built panel |

It is generated from the definitions the build itself uses — `variables.py`,
`taxonomy.py`, `mdrm.py` — so it cannot drift from the panels the way a
hand-written page does:

```bash
uv run python scripts/build_dictionary.py            # writes docs/variables.json
uv run python scripts/build_dictionary.py --check    # CI: complete and current?
```

Two things fail that build rather than shipping quietly: a column with no
definition, and a category the taxonomy has but nobody has named. Both would
otherwise reach the page as an omission or a raw slug, and both are silent.

## Browsing the variables

Before a model there is a prior question — *which* of these variables is worth
modelling. `apps/explorer.py` is a [marimo](https://marimo.io) app for exactly
that, and it assumes nothing about what you intend to fit:

```bash
uv sync --group app
uv run --group app marimo run apps/explorer.py --no-sandbox    # or `marimo edit`
```

`--no-sandbox` is not optional noise: the notebook carries inline PEP 723
metadata so the WASM export knows what to install in the browser, and without
the flag marimo stops to ask whether to build a throwaway venv from it. The
`app` group is already that environment.

It reads `data/out` when there is a build there and falls back to the committed
`docs/data`. Three passes, in order:

| Pass | Answers |
|---|---|
| **What is here** | per variable, how many banks carry it and how well, over what window. `n/a` is kept out of the bar and out of the median — a custody bank with no card book is not a gap. |
| **What one variable is made of** | its provenance mix (`xbrl` / `html` / `ir_supplement` / `derived_ir`), the bank-quarters it actually populates, and its path against the peer band with the seams drawn on. |
| **What a set of variables leaves you** | the complete-case sample for a chosen set, and what each further candidate would cost in bank-quarters before you add it. |

That third pass is the one that decides a model. The usable sample is the
*intersection* of the set's coverage, which is not the coverage of its weakest
member: two variables each 80% populated leave anywhere between 60% and 80%
depending on whether the same banks are missing both. The explorer computes it
rather than inferring it, because the inference is always optimistic.

To hand it to someone without a Python environment, export it as a static
WebAssembly site — the notebook, its modules and both panels, running in the
browser under Pyodide:

```bash
uv run --group app python scripts/export_explorer.py --out build/explorer
python -m http.server --directory build/explorer      # must be served, not opened
```

## Read this before using the data

Five things will produce a wrong answer if you skip them. None is a bug; each is
a property of the underlying disclosure that the panel reports rather than hides.

1. **The CECL seam at 2020Q1.** Every row carries a `basis` column — `incurred`
   or `cecl`. The allowance changes meaning at that line, and the step shows up
   in `reserve_coverage` as a jump that looks exactly like a credit event.
2. **Two discontinuities that read as growth.** `TFC`'s CIK is BB&T's, and the
   SunTrust merger roughly doubles the balance sheet in 2019Q4. `FCNCA` excludes
   CIT before 2022 and SVB before 2023.
3. **The 2013 start is survivors-only.** Any cross-sectional statistic over the
   early years is conditioned on surviving to 2026.
4. **Check `mix_coverage_pct` before ranking peers.** On the EDGAR panel it is an
   estimate of coherence, not a guarantee — and ≈100 is necessary, not sufficient.
5. **The pre-2012 thrifts are invisible without `--tfr`.** CDR is
   Call-Report-only, so an OTS-supervised savings institution contributes zero
   for every quarter it existed. JPMorgan's loan book jumps by Washington
   Mutual's entirety at 2008Q3 — the quarter the Call Report first consolidates
   it — and reads as growth. The shipped panel is the `--tfr` build, which
   reconstructs <!--stats:thrift-rows-->2,749<!--/stats--> charter-quarters from
   FDIC BankFind; `n_insured_not_filing` counts what is still missing.

The full account — the other two, and the rest of what the long FFIEC window
adds: **[Reading the Panel](https://github.com/kiwi5boy/bankqtr_db/wiki/Reading-the-Panel)**
and **[Going Back to 2001](https://github.com/kiwi5boy/bankqtr_db/wiki/Going-Back-to-2001)**.

## The one thing to know first

**The SEC's companyfacts API cannot produce a portfolio mix.** It returns only
*undimensioned* facts. Banks tag CRE loans, C&I loans and total loans with the
*same* XBRL element and distinguish them purely by dimension member — and
companyfacts drops every dimensioned fact. Verified on JPMorgan's FY2024 10-K:
companyfacts exposes 1 fact for the loans element at 2024-12-31; the filing's own
instance document carries **146**.

So extraction runs in two tiers — `xbrl.companyfacts_long` for consolidated
totals, and `xbrl.dimensional_long`, which parses each filing's XBRL **instance
document** and keeps the dimensions. Portfolio mix, criticized balances and
delinquency buckets all come from the second.

## Testing

```bash
uv run pytest tests/ -q
uv run ruff check .
```

<!--stats:tests-->410<!--/stats--> tests across both builds. Every one corresponds to a defect that produced
**plausible but wrong** numbers during development — the dangerous kind, since
nothing raises and the panel still renders. The catalogue of what they guard is
in the wiki: **[Traps](https://github.com/kiwi5boy/bankqtr_db/wiki/Traps)**.

## Documentation

| Page | What it covers |
|---|---|
| [Quick Start](https://github.com/kiwi5boy/bankqtr_db/wiki/Quick-Start) | Clean checkout to built panel, both sources |
| [Choosing a Source](https://github.com/kiwi5boy/bankqtr_db/wiki/Choosing-a-Source) | Which panel answers your question, and what to do when they disagree |
| [Reading the Panel](https://github.com/kiwi5boy/bankqtr_db/wiki/Reading-the-Panel) | The six caveats that matter before analysis |
| [Outputs](https://github.com/kiwi5boy/bankqtr_db/wiki/Outputs) | What each file in `data/out/` contains |
| [Data Dictionary](https://github.com/kiwi5boy/bankqtr_db/wiki/Data-Dictionary) | Column naming scheme, provenance, and the families |
| [Universe](https://github.com/kiwi5boy/bankqtr_db/wiki/Universe) | Who is in, who is deliberately out |
| [Glossary](https://github.com/kiwi5boy/bankqtr_db/wiki/Glossary) | RSSD, MDRM, RCFD/RCON, CECL, EX-13 |
| [FAQ](https://github.com/kiwi5boy/bankqtr_db/wiki/FAQ) | Short answers with links |
| [EDGAR Panel](https://github.com/kiwi5boy/bankqtr_db/wiki/EDGAR-Panel) | Two-tier extraction, and why the panel logic is not a groupby |
| [IR Supplements](https://github.com/kiwi5boy/bankqtr_db/wiki/IR-Supplements) | Office CRE and the disclosures XBRL has no tag for |
| [HTML Fallback](https://github.com/kiwi5boy/bankqtr_db/wiki/HTML-Fallback) | Parsing the filing's own tables |
| [Extending to 2013](https://github.com/kiwi5boy/bankqtr_db/wiki/Extending-to-2013) | Build record for the XBRL back-extension |
| [Call Reports](https://github.com/kiwi5boy/bankqtr_db/wiki/Call-Reports) | What the second source buys, and what it does not |
| [Going Back to 2001](https://github.com/kiwi5boy/bankqtr_db/wiki/Going-Back-to-2001) | Lineage, predecessors, and the thrift gap |
| [Traps](https://github.com/kiwi5boy/bankqtr_db/wiki/Traps) | Every catalogued failure, indexed by failure mode |
| [Testing](https://github.com/kiwi5boy/bankqtr_db/wiki/Testing) | The two suites and what they pin |
| [Extending](https://github.com/kiwi5boy/bankqtr_db/wiki/Extending) | Adding a variable, a bank, a disclosure or an era |

## A note on the counts in this file

Figures that describe a build rather than the text — row counts, columns, the
window, the test count — are wrapped in sentinels and rewritten from the
committed panels by `scripts/sync_doc_stats.py`, which CI runs on every push to
`main`. Do not edit them by hand; run `--write`. See
[Extending](https://github.com/kiwi5boy/bankqtr_db/wiki/Extending#counts-quoted-in-prose).

## Layout

```
bankqtr_db/     the EDGAR build   (filings, edgar, instance, xbrl, taxonomy,
                                   variables, panel, html_fallback, ir,
                                   ir_extract, reconcile)
callrpt_db/     the FFIEC build   (cdr, nic, lineage, mdrm, schedules, panel,
                                   crosscheck, tfr)
apps/           the variable explorer  (explore = the counting, viz = the
                                       charts, explorer = the marimo layout)
scripts/        fetch_* and build_* entry points
tests/          one test per defect (count under Testing, above)
docs/           the GitHub Pages site: downloads, charts, variable dictionary
```
