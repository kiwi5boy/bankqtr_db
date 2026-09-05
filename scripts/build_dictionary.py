"""Generate the site's variable dictionary from the definitions themselves.

The wiki's Data Dictionary describes the *naming scheme*; this describes every
actual column, and it is generated, so it cannot drift from the build the way
a hand-written page does.  Both panels already carry their definitions in
code -- ``bankqtr_db.variables`` holds the EDGAR concepts and their XBRL tag
candidates, ``callrpt_db.mdrm`` holds every FFIEC item with its schedule and
MDRM codes -- and the columns are those definitions crossed with a canonical
category from ``bankqtr_db.taxonomy``.  So a dictionary entry is a join, not
a transcription.

The one thing written by hand here is the *label* for each canonical category:
``ci`` reads as "C&I", not "Ci", and no titlecasing gets there.  That map is
checked against the taxonomy at build time, so adding a category without a
label fails here rather than shipping a row that says ``fund_finance``.

**Every column must resolve.** A dictionary that silently omits the variable
someone is looking for is worse than no dictionary -- they conclude it does
not exist.  So an unresolved column is a build failure, listed by name.

    uv run python scripts/build_dictionary.py            # writes docs/variables.json
    uv run python scripts/build_dictionary.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple, TypedDict

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# The explorer already decides which family a variable belongs to.  Importing
# it rather than restating the rule keeps the dictionary and the explorer from
# grouping the same column two different ways.
sys.path.insert(0, str(ROOT / "apps"))

from explore import family_of

from bankqtr_db import taxonomy as tax
from bankqtr_db import variables as var
from callrpt_db import mdrm
from callrpt_db import panel as callpanel

OUT = ROOT / "docs" / "variables.json"

# The panels are committed at data/; data/out/ is a local build.
PANEL_FILES = {
    "edgar": ("panel.parquet", "SEC EDGAR"),
    "ffiec": ("call_panel.parquet", "FFIEC Call Reports"),
}
SEARCH = (ROOT / "data", ROOT / "data" / "out")

META = frozenset(
    {
        "bank", "ticker", "cik", "rssd", "quarter", "period", "basis",
        "charters", "n_charters", "predecessors", "predecessor_count",
        "n_predecessor_charters", "has_predecessor", "predecessor_failed",
        "n_insured_not_filing", "n_flow_resets", "n_tfr_backfilled",
    }
)

# --------------------------------------------------------------------------
# Labels for the canonical categories
#
# Presentation, deliberately kept out of taxonomy.py: the taxonomy's job is to
# decide which member is which category, and it does not care what a reader
# calls it.  Checked for completeness against the taxonomy below.

LOAN_LABELS: dict[str, str] = {
    "total": "All loans",
    "commercial_total": "Commercial, total",
    "cre_total": "CRE, total",
    "cre_investor": "CRE, investor (non-owner-occupied)",
    "cre_owner_occupied": "CRE, owner-occupied",
    "construction": "Construction and land development",
    "multifamily": "Multifamily",
    "ci": "C&I",
    "small_business": "Small business",
    "lease": "Leases",
    "municipal": "Municipal / public sector",
    "fund_finance": "Fund finance / subscription lines",
    "financial_institutions": "Loans to financial institutions",
    "commercial_other": "Commercial, other",
    "consumer_total": "Consumer, total",
    "resi_mortgage": "Residential mortgage",
    "home_equity": "Home equity",
    "credit_card": "Credit card",
    "auto": "Auto",
    "student": "Student",
    "securities_based": "Securities-based lending",
    "consumer_other": "Consumer, other",
    "other": "Other loans",
}

CQ_LABELS: dict[str, str] = {
    "pass": "Pass-rated",
    "special_mention": "Special mention",
    "substandard": "Substandard",
    "doubtful": "Doubtful",
    "loss": "Loss",
    "classified": "Classified (substandard and worse)",
    "criticized": "Criticized (special mention and worse)",
    "nonaccrual": "Nonaccrual",
    "performing": "Performing",
    "nonperforming": "Nonperforming",
    "investment_grade": "Investment grade",
    "non_investment_grade": "Non-investment grade",
    "unrated": "Unrated",
    "internal_grade_low": "Internal grade, low risk",
    "internal_grade_mid": "Internal grade, medium risk",
    "internal_grade_high": "Internal grade, high risk",
    "fico_prime": "FICO prime",
    "fico_near_prime": "FICO near prime",
    "fico_subprime": "FICO subprime",
}

PD_LABELS: dict[str, str] = {
    "current": "Current",
    "dpd_30_59": "30–59 days past due",
    "dpd_60_89": "60–89 days past due",
    "dpd_30_89": "30–89 days past due",
    "dpd_90_plus": "90+ days past due",
    "past_due_total": "Past due, total",
    "nonaccrual": "Nonaccrual",
    "nonperforming": "Nonperforming",
    "in_foreclosure": "In foreclosure",
}


def _check_labels() -> None:
    """Fail if the taxonomy grew a category nobody named."""
    missing = {
        "loan": set(tax.LOAN_CATEGORIES) - set(LOAN_LABELS),
        "credit quality": set(tax.CREDIT_QUALITY_CATEGORIES) - set(CQ_LABELS),
        "past due": {r.category for r in tax.PAST_DUE_RULES} - set(PD_LABELS),
    }
    bad = {axis: names for axis, names in missing.items() if names}
    if bad:
        raise SystemExit(
            "categories with no label: "
            + "; ".join(f"{axis}: {sorted(n)}" for axis, n in bad.items())
        )


# --------------------------------------------------------------------------


def short(text: str, limit: int = 58) -> str:
    """A name cut from a definition.

    The FFIEC item table and the ratio table both carry prose written to be
    *read*: "Restructured loans in compliance with modified terms (accruing
    TDRs): HK25 from 2017, 1616 before" is exactly right in a description and
    259 characters wide in the first column of a table.  The clause before the
    first colon or semicolon is almost always the name; a parenthesis is not a
    clause boundary, because "Other (non-owner-occupied) nonfarm
    nonresidential" cut there leaves the word "Other".
    """
    head = re.split(r"\s*[:;]\s|\s+--\s+", text)[0].strip().rstrip(".,")
    if len(head) <= limit:
        return head
    head = head.split(",")[0].strip()
    if len(head) <= limit:
        return head
    return head[:limit].rsplit(" ", 1)[0] + "…"


class Concept(NamedTuple):
    """A column that is its own concept, described by hand."""

    label: str
    description: str
    kind: str = "stock"
    unit: str = "USD"


@dataclass
class Entry:
    panel: str
    name: str
    family: str
    label: str
    description: str
    kind: str
    unit: str
    # How the number is defined: XBRL tag candidates on the EDGAR side, the
    # schedule and MDRM items on the FFIEC side.
    definition: dict = field(default_factory=dict)
    # Filled in from the panel itself.
    n_cells: int = 0
    n_banks: int = 0
    first_q: str | None = None
    last_q: str | None = None
    sources: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# EDGAR: a concept crossed with a canonical category

_RATIOS = {r.name: r for r in var.RATIOS}
_BASES = {v.name: v for v in var.ALL_VARS}
# Longest first, so a short concept name cannot shadow a longer one.
_BASE_ORDER = tuple(sorted(_BASES, key=len, reverse=True))

# A VarDef's ``description`` is a definition, not a name -- "Loans held for
# investment, gross of allowance" is right in a tooltip and unreadable as the
# first column of a table once a category is appended to it.  These are the
# names; the definitions still travel in the description field.
BASE_LABELS: dict[str, str] = {
    "loans": "Loans",
    "acl": "Allowance (ACL)",
    "nonaccrual": "Nonaccrual",
    "oreo": "OREO",
    "loans_by_credit_quality": "Loans by credit quality",
    "loans_by_past_due": "Loans by delinquency",
    "assets": "Total assets",
    "equity": "Equity",
    "deposits": "Deposits",
    "unfunded_commitments": "Unfunded commitments",
    "nco": "Net charge-offs",
    "charge_offs": "Gross charge-offs",
    "recoveries": "Recoveries",
    "provision": "Provision",
    "net_income": "Net income",
    "nii": "Net interest income",
    "revenue": "Revenue",
}

GROWTH_SUFFIXES = {
    "_yoy_pct": (
        "year-over-year growth",
        "% change on the same quarter a year earlier",
    ),
    "_qoq_ann_pct": (
        "annualised quarterly growth",
        "% change on the prior quarter, annualised",
    ),
}

# Columns that are their own concept rather than a concept crossed with a
# category.  Each is a real disclosure with no VarDef of its own.
EDGAR_SINGLETONS: dict[str, Concept] = {
    "cq_rated_total": Concept(
        "Loans carrying a credit-quality grade",
        "Total loans the bank assigns to any internal risk grade or credit "
        "score band -- the denominator the other cq_ balances are shares of.",
        "stock",
        "USD",
    ),
    "loans_by_credit_quality_total": Concept(
        "Loans in the credit-quality table",
        "Total loans appearing in the credit-quality disclosure. It exists to "
        "check that the graded components sum to the book.",
        "stock",
        "USD",
    ),
    "loans_by_past_due_total": Concept(
        "Loans in the past-due table",
        "Total loans appearing in the aging disclosure. It exists to check "
        "that the delinquency buckets sum to the book.",
        "stock",
        "USD",
    ),
    "loans_office_cre": Concept(
        "Office CRE loans",
        "Office exposure within investor CRE. XBRL has no tag for it, so it "
        "comes from the bank's own investor supplement.",
        "stock",
        "USD",
    ),
    "acl_office_cre": Concept(
        "Allowance on office CRE",
        "Allowance for credit losses held against the office book, from the "
        "investor supplement.",
        "stock",
        "USD",
    ),
    "nonaccrual_office_cre": Concept(
        "Nonaccrual office CRE",
        "Office CRE loans on nonaccrual, from the investor supplement.",
        "stock",
        "USD",
    ),
    "nco_office_cre": Concept(
        "Net charge-offs, office CRE",
        "Net charge-offs on the office book, from the investor supplement.",
        "flow",
        "USD",
    ),
    "office_cre_share_of_cre": Concept(
        "Office as a share of CRE",
        "Office exposure as a percentage of the investor CRE book, taken as "
        "the bank states it in its supplement rather than recomputed.",
        "ratio",
        "%",
    ),
    "npa_total": Concept(
        "Nonperforming assets",
        "Nonaccrual loans plus OREO. A missing nonaccrual total stays missing "
        "rather than becoming zero -- \"not disclosed\" and \"no problem "
        "loans\" are not the same thing.",
        "stock",
        "USD",
    ),
    "mix_coverage_pct": Concept(
        "Portfolio-mix coherence",
        "The leaf loan categories summed over reported total loans. It is a "
        "diagnostic, not a measure of the bank: check it before ranking peers "
        "on any mix variable. Near 100 is necessary but not sufficient -- it "
        "says the parts add up, not that they were split the same way.",
        "ratio",
        "%",
    ),
}


def _growth_of(name: str) -> tuple[str, str, str] | None:
    for suffix, (label, gloss) in GROWTH_SUFFIXES.items():
        if name.endswith(suffix):
            return name[: -len(suffix)], label, gloss
    return None


def _loan_sliced(name: str) -> tuple[var.VarDef, str] | None:
    for base in _BASE_ORDER:
        if name.startswith(base + "_"):
            category = name[len(base) + 1 :]
            if category in tax.LOAN_CATEGORIES:
                return _BASES[base], category
    return None


class _Common(TypedDict):
    panel: str
    name: str
    family: str


def resolve_edgar(name: str, family: str) -> Entry | None:
    """One EDGAR column -> its definition, or None if nothing claims it."""
    common: _Common = {"panel": "edgar", "name": name, "family": family}

    if name in EDGAR_SINGLETONS:
        concept = EDGAR_SINGLETONS[name]
        return Entry(**common, **concept._asdict())

    if name in _RATIOS:
        return _ratio_entry(name, family, "edgar", _RATIOS[name])

    grown = _growth_of(name)
    if grown is not None:
        stem, label, gloss = grown
        inner = resolve_edgar(stem, family)
        # The suffix costs ~26 characters, so the stem is cut harder than a
        # standalone label would be.
        stem_label = short(inner.label, 44) if inner else stem
        return Entry(
            **common,
            label=f"{stem_label} — {label}",
            kind="growth",
            unit="%",
            description=f"{stem_label}: {gloss}.",
            definition={"derived_from": stem},
        )

    if name in _BASES:
        spec = _BASES[name]
        return Entry(
            **common,
            label=BASE_LABELS[name],
            kind=spec.kind,
            unit=spec.unit,
            description=f"{spec.description}.",
            definition={"xbrl_tags": list(spec.tags)},
        )

    sliced = _loan_sliced(name)
    if sliced is not None:
        spec, category = sliced
        category_label = LOAN_LABELS[category]
        parent = tax.LOAN_CATEGORIES[category]
        return Entry(
            **common,
            label=f"{BASE_LABELS[spec.name]} — {category_label}",
            kind=spec.kind,
            unit=spec.unit,
            description=(
                f"{spec.description}, for the {category_label} portfolio."
                + (f" Rolls up into {LOAN_LABELS[parent]}." if parent else "")
            ),
            definition={
                "xbrl_tags": list(spec.tags),
                "axis": "loan",
                "category": category,
                "rolls_up_into": parent,
            },
        )

    if name.startswith("cq_") and name[3:] in tax.CREDIT_QUALITY_CATEGORIES:
        category = name[3:]
        return Entry(
            **common,
            label=f"Loans graded {CQ_LABELS[category]}",
            kind="stock",
            unit="USD",
            description=(
                "Loan balances the bank discloses as "
                f"{CQ_LABELS[category]} on its credit-quality axis."
            ),
            definition={"axis": "credit_quality", "category": category},
        )

    if name.startswith("pd_") and name[3:] in PD_LABELS:
        category = name[3:]
        return Entry(
            **common,
            label=f"Loans {PD_LABELS[category].lower()}",
            kind="stock",
            unit="USD",
            description=(
                f"Loan balances in the {PD_LABELS[category]} bucket of the "
                "delinquency aging disclosure."
            ),
            definition={"axis": "past_due", "category": category},
        )

    for bucket in ("dpd_30_89", "dpd_90_plus"):
        if not name.startswith(bucket + "_"):
            continue
        category = name[len(bucket) + 1 :]
        if category in tax.LOAN_CATEGORIES:
            category_label = LOAN_LABELS[category]
        elif category == "cre_nonfarm_nonres":
            # The Call Report's own category boundary, kept as filed rather
            # than forced onto the panel's CRE split.
            category_label = "CRE (nonfarm nonresidential)"
        else:
            continue
        return Entry(
            **common,
            label=f"{category_label} — {PD_LABELS[bucket]}",
            kind="stock",
            unit="USD",
            description=(
                f"{category_label} loans {PD_LABELS[bucket].lower()}, from the "
                "aging disclosure."
            ),
            definition={
                "axis": "past_due",
                "bucket": bucket,
                "category": category,
            },
        )

    vintage = re.fullmatch(r"vintage_(total|criticized)_(\d{4})", name)
    if vintage is not None:
        measure, year = vintage.groups()
        what = "Total" if measure == "total" else "Criticized"
        return Entry(
            **common,
            label=f"{what} balances originated in {year}",
            kind="stock",
            unit="USD",
            description=(
                f"{what} loan balances still outstanding from the {year} "
                "origination vintage, as disclosed in the vintage table."
            ),
            definition={
                "axis": "vintage",
                "vintage_year": int(year),
                "measure": measure,
            },
        )

    return None


# --------------------------------------------------------------------------
# FFIEC: a filed item, or something built from filed items
#
# Every number on this panel is either an MDRM item off a schedule or a
# documented combination of them, so a description can always say which --
# which is the thing the wiki page cannot do for 263 columns by hand.

_ITEMS = mdrm.BY_NAME
# CALL_RATIOS are the Call-Report-specific ones; the rest are the EDGAR
# definitions reused, and those carry prose the Call ones mostly do not.
_CALL_RATIOS = {r.name: r for r in (*callpanel.RATIOS, *callpanel.CALL_RATIOS)}

# Two thirds of the FFIEC item table carries no description: whoever added
# ``charge_offs_construction`` did not need one, and neither does anyone who
# already knows the naming scheme -- which is precisely the reader this page is
# not for. Falling back to the raw slug puts "charge_offs_construction" in the
# Meaning column, which reads as an unfinished page. These compose a name from
# the parts the name is already made of.
#
# The categories are the *Call Report's* own, not the panel's canonical set:
# RC-C splits farmland and foreign offices that the EDGAR taxonomy never sees,
# and keeps nonfarm nonresidential whole where the panel splits owner-occupied
# out. Naming them after the panel's categories would quietly assert the two
# agree.
CALL_CATEGORY_LABELS: dict[str, str] = {
    **LOAN_LABELS,
    "farmland": "Secured by farmland",
    "foreign_office": "Foreign offices",
    "re_foreign": "Real estate in foreign offices",
    "depository_institutions": "Loans to depository institutions",
    "agricultural": "Agricultural production",
    "consumer_installment": "Consumer instalment",
    "consumer_noncard": "Consumer excluding cards",
    "foreign_govt": "Foreign governments",
    "other_loans": "Other loans",
    "cre_nonfarm_nonres": "CRE, nonfarm nonresidential",
}

# Family prefixes, longest first so ``dpd_30_89`` beats nothing shorter and
# ``charge_offs`` is not read as ``charge``.
CALL_FAMILY_LABELS: dict[str, str] = {
    "dpd_30_89": "30–89 days past due",
    "dpd_90_plus": "90+ days past due",
    "charge_offs": "Gross charge-offs",
    "recoveries": "Recoveries",
    "nonaccrual": "Nonaccrual",
    "commitments": "Unused commitments",
    "loans": "Loans",
    "acl": "Allowance (ACL)",
    "nco": "Net charge-offs",
}

# What is left once the composed names are taken: items that are their own
# concept and never carried prose.
CALL_ITEM_LABELS: dict[str, str] = {
    "assets": "Total assets",
    "equity": "Total equity capital",
    "trading_assets": "Trading assets",
    "loans_unearned_income": "Unearned income on loans",
    "loans_held_for_sale": "Loans held for sale",
    "commitments_securities_underwriting": "Securities underwriting commitments",
    "net_income": "Net income",
    "net_interest_income": "Net interest income",
    "noninterest_income": "Noninterest income",
    "noninterest_expense": "Noninterest expense",
    "interest_income": "Total interest income",
    "interest_expense": "Total interest expense",
    "cet1_capital": "Common equity tier 1 capital",
    "tier1_capital": "Tier 1 capital",
    "total_capital": "Total regulatory capital",
    "risk_weighted_assets": "Risk-weighted assets",
}


def _call_label(name: str) -> str | None:
    """A name composed from the family and category the column is named for."""
    if name in CALL_ITEM_LABELS:
        return CALL_ITEM_LABELS[name]
    for family in sorted(CALL_FAMILY_LABELS, key=len, reverse=True):
        if not name.startswith(family + "_"):
            continue
        category = name[len(family) + 1 :]
        if category in CALL_CATEGORY_LABELS:
            return f"{CALL_FAMILY_LABELS[family]} — {CALL_CATEGORY_LABELS[category]}"
    return None


FFIEC_SINGLETONS: dict[str, Concept] = {
    "npa_total": Concept(
        "Nonperforming assets",
        "Nonaccrual loans plus OREO. A missing nonaccrual total stays missing "
        "rather than becoming zero.",
        "stock",
        "USD",
    ),
    "noncurrent_total": Concept(
        "Noncurrent loans",
        "Loans 90+ days past due and still accruing, plus nonaccrual loans -- "
        "the regulatory definition of noncurrent.",
        "stock",
        "USD",
    ),
    "tangible_assets": Concept(
        "Tangible assets",
        "Total assets less goodwill and other intangibles.",
        "stock",
        "USD",
    ),
    "tangible_equity": Concept(
        "Tangible equity",
        "Total equity less goodwill and other intangibles.",
        "stock",
        "USD",
    ),
    "tangible_common_equity": Concept(
        "Tangible common equity",
        "Tangible equity less preferred stock -- the TCE ratio's numerator.",
        "stock",
        "USD",
    ),
    "wholesale_funding": Concept(
        "Wholesale funding",
        "Brokered deposits, fed funds purchased and repo, and other borrowed "
        "money: the funding that leaves fastest under stress.",
        "stock",
        "USD",
    ),
    "ppnr": Concept(
        "Pre-provision net revenue",
        "Net interest income plus noninterest income less noninterest "
        "expense -- earnings before the provision, which is the quantity a "
        "stress model projects.",
        "flow",
        "USD",
    ),
    "commitments_total": Concept(
        "Unused commitments, total",
        "The unused-commitment components of Schedule RC-L, summed.",
        "stock",
        "USD",
    ),
    "loans_unallocated": Concept(
        "Loans not allocated to a category",
        "Total loans less the categories that were reported: the residual "
        "that says how much of the book the mix does not explain.",
        "stock",
        "USD",
    ),
    "rcc_residual": Concept(
        "Schedule RC-C residual",
        "RC-C total loans less the sum of its own categories. A non-zero "
        "residual means the schedule's parts do not foot to its total.",
        "stock",
        "USD",
    ),
    "rcc_residual_pct": Concept(
        "Schedule RC-C residual, % of loans",
        "The RC-C residual over total loans -- the coherence check to read "
        "before trusting a category on this panel.",
        "ratio",
        "%",
    ),
    "rcn_total_built": Concept(
        "Schedule RC-N total, rebuilt",
        "The RC-N delinquency total reconstructed from its components, for "
        "quarters where the filed total is absent or disagrees.",
        "stock",
        "USD",
    ),
    "insured_not_filing": Concept(
        "Insured depositories not filing",
        "Charters the FDIC records as insured for the quarter but for which "
        "CDR holds no Call Report -- the measure of what the roll-up is "
        "missing, and the pre-2012 thrift gap's fingerprint.",
        "stock",
        "count",
    ),
    "mix_coverage_pct": Concept(
        "Portfolio-mix coherence",
        "The leaf loan categories summed over reported total loans. A "
        "diagnostic, not a measure of the bank.",
        "ratio",
        "%",
    ),
    "securities_afs_unrealized": Concept(
        "AFS securities, unrealised gain/loss",
        "Available-for-sale securities at fair value less amortised cost.",
        "stock",
        "USD",
    ),
    "securities_htm_unrealized": Concept(
        "HTM securities, unrealised gain/loss",
        "Held-to-maturity securities at fair value less amortised cost -- the "
        "loss that does not pass through capital until it is realised.",
        "stock",
        "USD",
    ),
    "intangibles_other": Concept(
        "Other intangibles",
        "Intangible assets other than goodwill.",
        "stock",
        "USD",
    ),
    "deposits_uninsured_coverage_pct": Concept(
        "Uninsured deposit coverage",
        "Estimated uninsured deposits as a share of total deposits.",
        "ratio",
        "%",
    ),
    "texas_ratio": Concept(
        "Texas ratio",
        "Noncurrent loans plus OREO over tangible common equity plus the "
        "allowance: problem assets against what absorbs them.",
        "ratio",
        "%",
    ),
}

# Capital ratios are filed as ratios on Schedule RC-R in the modern series and
# rebuilt from components in the Basel I era, so they belong to neither the
# item table nor the ratio table cleanly.
CAPITAL_RATIOS: dict[str, Concept] = {
    "cet1_ratio": Concept(
        "CET1 ratio",
        "Common equity tier 1 capital over risk-weighted assets.",
        "ratio",
        "%",
    ),
    "tier1_ratio": Concept(
        "Tier 1 ratio",
        "Tier 1 capital over risk-weighted assets.",
        "ratio",
        "%",
    ),
    "tier1_leverage_ratio": Concept(
        "Tier 1 leverage ratio",
        "Tier 1 capital over average assets, with no risk weighting.",
        "ratio",
        "%",
    ),
    "total_capital_ratio": Concept(
        "Total capital ratio",
        "Total regulatory capital over risk-weighted assets.",
        "ratio",
        "%",
    ),
    "tier1_ratio_basel1": Concept(
        "Tier 1 ratio (Basel I)",
        "Tier 1 capital over Basel I risk-weighted assets, for quarters before "
        "the Basel III series begins.",
        "ratio",
        "%",
    ),
    "total_capital_ratio_basel1": Concept(
        "Total capital ratio (Basel I)",
        "Total capital over Basel I risk-weighted assets, for quarters before "
        "the Basel III series begins.",
        "ratio",
        "%",
    ),
}


def _operand_label(name: str, panel: str) -> str:
    """A short name for one side of a ratio, from that operand's own entry."""
    if name in _RATIOS or name in _CALL_RATIOS:
        return name  # a ratio of ratios would recurse; the raw name is honest
    resolve = resolve_edgar if panel == "edgar" else resolve_ffiec
    entry = resolve(name, family_of(name))
    return short(entry.label, 40) if entry is not None else name


def _ratio_entry(name: str, family: str, panel: str, ratio) -> Entry:
    """A ratio entry, named from its operands when it carries no prose.

    Half the ratios on each panel have an empty ``description`` -- they were
    obvious to whoever added them, and ``nco_rate_ci`` is indeed obvious once
    you know the naming scheme, which is the thing this page exists for people
    who do not. Humanising the slug gives "Nco rate ci"; naming it from what
    it divides gives "Net charge-offs -- C&I / Loans -- C&I", which is the
    answer to the question actually being asked.
    """
    over = _operand_label(ratio.numerator, panel)
    under = _operand_label(ratio.denominator, panel)
    formula = f"{ratio.numerator} / {ratio.denominator}" + (
        ", annualised" if ratio.annualize else ""
    )
    prose = short(ratio.description)
    return Entry(
        panel=panel,
        name=name,
        family=family,
        label=prose or f"{over} / {under}",
        kind="ratio",
        unit="%" if name.endswith("_pct") or "rate" in name or "ratio" in name
        else "ratio",
        description=(
            (ratio.description + ". " if ratio.description else "")
            + f"Computed as {formula}."
        ),
        definition={
            "numerator": ratio.numerator,
            "denominator": ratio.denominator,
            "annualised": bool(ratio.annualize),
        },
    )


def resolve_ffiec(name: str, family: str) -> Entry | None:
    """One FFIEC column -> its definition, or None if nothing claims it."""
    common: _Common = {"panel": "ffiec", "name": name, "family": family}

    if name in _ITEMS:
        spec = _ITEMS[name]
        # A handful of items are read from whichever of several schedules the
        # filer used -- goodwill off RC or RC-M, brokered deposits off RC-E or
        # its part I -- so ``schedule`` is sometimes a tuple.  Always a list
        # here, or it renders as a Python repr on the page.
        schedules = (
            [spec.schedule] if isinstance(spec.schedule, str) else list(spec.schedule)
        )
        return Entry(
            **common,
            label=short(spec.description) or _call_label(name) or name,
            kind=spec.kind,
            unit="USD",
            description=(
                (spec.description + ". " if spec.description else "")
                + "Schedule "
                + " or ".join(schedules)
                + ", MDRM item"
                + ("s " if len(spec.items) > 1 else " ")
                + ", ".join(spec.items)
                + f" (read as {' then '.join(spec.prefixes)})."
                + (f" Filed from {spec.since}." if spec.since else "")
            ),
            definition={
                "schedules": schedules,
                "mdrm_items": list(spec.items),
                "prefixes": list(spec.prefixes),
                "since": spec.since,
                "alternatives": [list(a) for a in spec.alternatives],
            },
        )

    if name in FFIEC_SINGLETONS:
        return Entry(**common, **FFIEC_SINGLETONS[name]._asdict())

    if name in CAPITAL_RATIOS:
        return Entry(**common, **CAPITAL_RATIOS[name]._asdict())

    if name in _CALL_RATIOS:
        return _ratio_entry(name, family, "ffiec", _CALL_RATIOS[name])

    if name in mdrm.DERIVED_LOAN_GROUPS:
        parts = mdrm.DERIVED_LOAN_GROUPS[name]
        category = name.removeprefix("loans_")
        label = LOAN_LABELS.get(category, category.replace("_", " ").capitalize())
        return Entry(
            **common,
            label=f"Loans — {label}",
            kind="stock",
            unit="USD",
            description=(
                f"{label} loans, summed from the filed categories: "
                + ", ".join(parts)
                + "."
            ),
            definition={"sum_of": list(parts)},
        )

    grown = _growth_of(name)
    if grown is not None:
        stem, label, gloss = grown
        inner = resolve_ffiec(stem, family)
        stem_label = short(inner.label, 44) if inner else stem
        return Entry(
            **common,
            label=f"{stem_label} — {label}",
            kind="growth",
            unit="%",
            description=f"{stem_label}: {gloss}.",
            definition={"derived_from": stem},
        )

    if name.startswith("nco_"):
        suffix = name.removeprefix("nco_")
        if suffix == "cre_total":
            parts = ["nco_cre_nonfarm_nonres", "nco_construction", "nco_multifamily"]
            return Entry(
                **common,
                label="Net charge-offs — CRE, total",
                kind="flow",
                unit="USD",
                description=(
                    "Net charge-offs across the CRE classes, summed: "
                    + ", ".join(parts)
                    + "."
                ),
                definition={"sum_of": parts},
            )
        category_label = LOAN_LABELS.get(
            suffix, suffix.replace("_", " ").capitalize()
        )
        return Entry(
            **common,
            label=f"Net charge-offs — {category_label}",
            kind="flow",
            unit="USD",
            description=(
                f"Net charge-offs on {category_label.lower()} loans. The Call "
                "Report never reports a net figure, so this is always gross "
                f"charge-offs less recoveries: charge_offs_{suffix} - "
                f"recoveries_{suffix}."
            ),
            definition={"minus": [f"charge_offs_{suffix}", f"recoveries_{suffix}"]},
        )

    return None


# --------------------------------------------------------------------------
# What the panel actually holds


def _panel_path(panel: str) -> Path:
    filename = PANEL_FILES[panel][0]
    for base in SEARCH:
        candidate = base / filename
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"{filename} not found in {' or '.join(str(b) for b in SEARCH)} -- "
        "build the panels first."
    )


def value_columns(frame: pl.DataFrame) -> list[str]:
    return [
        c
        for c in frame.columns
        if c not in META and not c.endswith("__source")
    ]


def enrich(entry: Entry, frame: pl.DataFrame) -> None:
    """Attach what the built panel says about a variable.

    Deliberately the plain counts -- populated cells, distinct banks, the
    window -- and not the ok/sparse judgement.  That judgement needs the
    not-applicable map and a threshold, and it belongs to the explorer, which
    is where someone goes to decide whether a variable is *usable*.  A
    dictionary's job is to say what a column is and roughly how much of it
    there is.
    """
    present = frame.filter(pl.col(entry.name).is_not_null())
    entry.n_cells = present.height
    if not present.height:
        return
    entry.n_banks = present["ticker"].n_unique()
    entry.first_q = present["quarter"].min()
    entry.last_q = present["quarter"].max()

    source_column = entry.name + "__source"
    if source_column in present.columns:
        counts = (
            present.group_by(source_column)
            .agg(n=pl.len())
            .sort("n", descending=True)
        )
        entry.sources = {
            (row[source_column] or "unrecorded"): row["n"]
            for row in counts.iter_rows(named=True)
        }


def collect() -> dict:
    """Resolve and enrich every column on both panels."""
    _check_labels()
    resolvers = {"edgar": resolve_edgar, "ffiec": resolve_ffiec}

    entries: list[Entry] = []
    panels: dict[str, dict] = {}
    unresolved: dict[str, list[str]] = {}

    for panel, (_, source) in PANEL_FILES.items():
        frame = pl.read_parquet(_panel_path(panel))
        names = value_columns(frame)
        panels[panel] = {
            "source": source,
            "rows": frame.height,
            "banks": frame["ticker"].n_unique(),
            "variables": len(names),
            "first_q": frame["quarter"].min(),
            "last_q": frame["quarter"].max(),
        }
        missing = []
        for name in names:
            entry = resolvers[panel](name, family_of(name))
            if entry is None:
                missing.append(name)
                continue
            enrich(entry, frame)
            entries.append(entry)
        if missing:
            unresolved[panel] = missing

    if unresolved:
        lines = [
            f"  {panel}: {', '.join(names)}"
            for panel, names in sorted(unresolved.items())
        ]
        raise SystemExit(
            "every column must have a definition; these have none:\n"
            + "\n".join(lines)
            + "\n\nAdd a rule or a singleton entry in scripts/build_dictionary.py."
        )

    entries.sort(key=lambda e: (e.panel, e.family, e.name))
    return {
        "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "panels": panels,
        "variables": [asdict(e) for e in entries],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="resolve everything and report, but write nothing",
    )
    args = ap.parse_args()

    payload = collect()
    counts = ", ".join(
        f"{panel} {meta['variables']}" for panel, meta in payload["panels"].items()
    )
    print(f"resolved every column ({counts})")

    if args.check:
        if not OUT.exists():
            raise SystemExit(f"{OUT.relative_to(ROOT)} is missing -- run this script")
        # The timestamp moves on every run and says nothing about the content,
        # so a rebuild is not a change.  Everything else is.
        committed = json.loads(OUT.read_text(encoding="utf-8"))
        fresh = dict(payload)
        committed.pop("generated_utc", None)
        fresh.pop("generated_utc", None)
        if committed != fresh:
            raise SystemExit(
                f"{OUT.relative_to(ROOT)} is out of date with the panels and the "
                "definitions.\nRun: uv run python scripts/build_dictionary.py"
            )
        print(f"{OUT.relative_to(ROOT)} is current")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    size = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
