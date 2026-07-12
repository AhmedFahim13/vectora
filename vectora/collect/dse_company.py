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
