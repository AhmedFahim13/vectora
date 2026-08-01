"""DSE company page parser (displayCompany.php?name=SYMBOL).

Facts are label/value pairs in mixed th/td and td/td tables; all seven fact
keys are always present (None when missing/unparseable) plus the caller's
symbol, so downstream upserts see a stable schema keyed on symbol.

Shareholding percentages appear in "Share Holding Percentage [as on <date>]"
blocks with cells like "Sponsor/Director:<br>90.00". Pages carry multiple
blocks (year-end plus one or more later months — GP shows three), a free
historical time series: ALL valid blocks are returned, sorted ascending by
as_of, each tagged with the symbol to match the (symbol, as_of) holdings PK.
A block is valid when at least 3 of the 5 percentages actually parse.
"""
import re
from datetime import datetime

from bs4 import BeautifulSoup

from vectora.http import PoliteSession
from vectora.settings import DSE_BASE

URL = f"{DSE_BASE}/displayCompany.php"

_FACT_LABELS = {
    "Paid-up Capital (mn)": ("paid_up_capital_mn", "num"),
    "Face/par Value": ("face_value", "num"),
    "Total No. of Outstanding Securities": ("outstanding_shares", "int"),
    "Type of Instrument": ("instrument_type", "str"),
    "Market Lot": ("market_lot", "int"),
    "Sector": ("sector", "str"),
    "Market Category": ("category", "str"),
}

_HOLDING_KEYS = {
    "Sponsor/Director": "sponsor_pct",
    "Govt": "govt_pct",
    "Institute": "institute_pct",
    "Foreign": "foreign_pct",
    "Public": "public_pct",
}

_AS_ON_RE = re.compile(r"\[as on ([^\]\(]+)")


def fetch_company(session: PoliteSession, symbol: str) -> str:
    return session.get(URL, params={"name": symbol})


def _num(text: str) -> float | None:
    t = text.strip().replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def _parse_as_of(text: str) -> str | None:
    m = _AS_ON_RE.search(text)
    if not m:
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(m.group(1).strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _extract_facts(soup: BeautifulSoup, symbol: str) -> dict:
    facts: dict = {"symbol": symbol}
    facts.update({key: None for key, _ in _FACT_LABELS.values()})
    seen: set[str] = set()
    for cell in soup.find_all(["th", "td"]):
        label = cell.get_text(strip=True)
        if label not in _FACT_LABELS:
            continue
        key, kind = _FACT_LABELS[label]
        if key in seen:
            continue  # first occurrence wins
        seen.add(key)
        value_cell = cell.find_next_sibling("td")
        if value_cell is None:
            continue
        raw = value_cell.get_text(" ", strip=True)
        if key == "category":
            # some pages render "-" for uncategorized instruments (bonds etc.)
            c = raw.strip()[:1]
            facts[key] = c if c.isalpha() else None
        elif kind == "str":
            facts[key] = raw or None
        elif kind == "num":
            facts[key] = _num(raw)
        elif kind == "int":
            v = _num(raw)
            facts[key] = int(v) if v is not None else None
    return facts


def _extract_holdings(soup: BeautifulSoup, symbol: str) -> list[dict]:
    blocks: list[dict] = []
    for node in soup.find_all(string=re.compile(r"Share Holding Percentage")):
        label_cell = node.find_parent("td")
        if label_cell is None:
            continue
        as_of = _parse_as_of(label_cell.get_text(" ", strip=True))
        # percentages live in the sibling <td> (nested table) of the same row
        row = label_cell.find_parent("tr")
        if row is None:
            continue
        text = row.get_text(" ", strip=True)
        holdings: dict = {"symbol": symbol, "as_of": as_of}
        parsed = 0
        for label, key in _HOLDING_KEYS.items():
            m = re.search(rf"{re.escape(label)}\s*:\s*([\d.,]+)", text)
            value = _num(m.group(1)) if m else None
            holdings[key] = value
            parsed += 1 if value is not None else 0
        if parsed < 3:
            continue  # not a real holdings block
        blocks.append(holdings)
    blocks.sort(key=lambda h: h["as_of"] or "")
    return blocks


def parse_company(html: str, symbol: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    if soup.find("table") is None:
        return {"facts": {}, "holdings": []}
    return {
        "facts": _extract_facts(soup, symbol),
        "holdings": _extract_holdings(soup, symbol),
    }

# --- fundamentals (Phase 6C) -------------------------------------------------
# All of these live on the same company page the sweep already fetches, so
# they cost no extra requests. DSE quotes dividends as a percentage of the
# FACE value (usually 10 taka), never of the market price — the derived
# yield below applies that convention rather than the naive pct/100.
_FUND_LABELS = {
    "Market Capitalization (mn)": ("market_cap_mn", "num"),
    "Free Float Market Cap. (mn)": ("free_float_mcap_mn", "num"),
    "Reserve & Surplus without OCI (mn)": ("reserve_surplus_mn", "num"),
    "Trailing P/E Ratio": ("trailing_pe", "num"),
    "Listing Year": ("listing_year", "int"),
    "Year End": ("year_end", "str"),
    "Latest Dividend Status (%)": ("_dividend_raw", "str"),
}
_FUND_KEYS = ("market_cap_mn", "free_float_mcap_mn", "reserve_surplus_mn",
              "trailing_pe", "listing_year", "year_end",
              "latest_dividend_pct", "latest_bonus_pct", "dividend_year",
              "face_value")
# "215.00 for 2025" | "5%B for 2024" | "175.00, 10%B for 2025" | "n/a"
# A trailing B marks a BONUS (stock) dividend. It must never be folded into
# the cash figure: bonus shares pay the holder nothing, and a yield computed
# from them would be fabricated income.
_DIV_YEAR_RE = re.compile(r"for\s*(\d{4})")
_DIV_PART_RE = re.compile(r"([\d.]+)\s*%?\s*(B)?", re.IGNORECASE)


def parse_dividend_status(text: str | None) -> dict:
    """Split DSE's 'Latest Dividend Status (%)' into cash, bonus and year."""
    out = {"latest_dividend_pct": None, "latest_bonus_pct": None,
           "dividend_year": None}
    if not text:
        return out
    head = text.split("for")[0]
    for part in head.split(","):
        m = _DIV_PART_RE.search(part)
        if not m:
            continue
        value = _num(m.group(1))
        if value is None:
            continue
        key = "latest_bonus_pct" if m.group(2) else "latest_dividend_pct"
        if out[key] is None:
            out[key] = value
    year = _DIV_YEAR_RE.search(text)
    if year:
        out["dividend_year"] = int(year.group(1))
    return out


def parse_fundamentals(html: str, symbol: str) -> dict:
    """Headline fundamentals from the company page; missing fields stay None."""
    soup = BeautifulSoup(html, "lxml")
    out: dict = {k: None for k in _FUND_KEYS}
    out["symbol"] = symbol
    seen: set = set()
    for cell in soup.find_all(["th", "td"]):
        label = cell.get_text(strip=True)
        spec = _FUND_LABELS.get(label)
        if spec is None or label in seen:
            continue
        seen.add(label)
        key, kind = spec
        value_cell = cell.find_next_sibling("td")
        if value_cell is None:
            continue
        raw = value_cell.get_text(" ", strip=True)
        if kind == "num":
            out[key] = _num(raw)
        elif kind == "int":
            v = _num(raw)
            out[key] = int(v) if v is not None else None
        else:
            out[key] = raw or None
    # face value feeds the dividend conversion; reuse the facts parser
    facts = _extract_facts(soup, symbol)
    out["face_value"] = facts.get("face_value")
    out.update(parse_dividend_status(out.pop("_dividend_raw", None)))
    return out


def derive_metrics(fund: dict, close: float | None) -> dict:
    """Metrics that need the live price: dividend per share, yield, EPS.

    Cash only — a bonus issue changes the share count, not the holder's
    income, so it is reported separately and never enters the yield.
    """
    face = fund.get("face_value") or 10.0
    pct = fund.get("latest_dividend_pct")
    dps = (pct / 100.0) * face if pct is not None else None
    pe = fund.get("trailing_pe")
    return {
        "dividend_per_share": dps,
        "dividend_yield": (dps / close) if (dps and close) else None,
        "eps_trailing": (close / pe) if (close and pe) else None,
    }
