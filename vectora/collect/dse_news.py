"""DSE news/announcements scraper.

Endpoint (verified 2026-07-12):
  /old_news.php?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&criteria=4&archive=news

All items on the page live inside a single <table class="table-news">, not
one table per item. Each item is a run of label/value rows in this order:
  "Trading Code:" / "News Title:" / "News:" / "Post Date:"
followed by separator rows (<th colspan="2"><hr/></th> and a "&nbsp;" row)
before the next item's "Trading Code:" row starts.

Symbol source: the "Trading Code:" field is authoritative — exchange/regulator
notices carry synthetic bucket codes (EXCH, REGL) which map to None. The
"SYMBOL: Subject" title prefix is only a fallback when the field is absent.
"""
import hashlib
import re

from bs4 import BeautifulSoup

from vectora.http import PoliteSession
from vectora.settings import DSE_BASE

URL = f"{DSE_BASE}/old_news.php"

# ticker: 2+ chars, uppercase alnum, starts the title and ends with a colon
_SYMBOL_RE = re.compile(r"^([A-Z0-9]{2,20}):\s")


def fetch_news(session: PoliteSession, start: str, end: str) -> str:
    return session.get(URL, params={
        "startDate": start, "endDate": end, "criteria": "4", "archive": "news",
    })


def symbol_from_title(title: str) -> str | None:
    m = _SYMBOL_RE.match(title.strip())
    if not m:
        return None
    sym = m.group(1)
    # exchange/regulator notices, not tickers ("DSE", "DSENEWS", "BSECNEWS", ...)
    if sym in ("DSE", "BSEC") or sym.endswith("NEWS"):
        return None
    return sym


_SYNTHETIC_CODES = {"EXCH", "REGL", "", "-"}


def _symbol(fields: dict[str, str]) -> str | None:
    if "Trading Code" in fields:
        code = fields["Trading Code"].strip()
        return None if code in _SYNTHETIC_CODES else code
    return symbol_from_title(fields.get("News Title", ""))


def _build_item(fields: dict[str, str]) -> dict | None:
    title = fields.get("News Title", "")
    body = fields.get("News", "")
    post_date = fields.get("Post Date", "")
    if not title or not post_date:
        return None
    digest = hashlib.sha256(f"{title}|{post_date}|{body}".encode()).hexdigest()
    return {
        "id": digest,
        "post_date": post_date,
        "symbol": _symbol(fields),
        "title": title,
        "body": body[:5000],
        "source": "dse_news",
    }


def parse_news(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="table-news")
    if table is None:
        return []

    items: list[dict] = []
    fields: dict[str, str] = {}
    for row in table.find_all("tr"):
        th, td = row.find("th"), row.find("td")
        if th is None or td is None:
            continue  # separator rows (<hr/>, &nbsp;) have no <td>
        key = th.get_text(strip=True).rstrip(":")
        if key == "Trading Code" and fields:
            item = _build_item(fields)
            if item is not None:
                items.append(item)
            fields = {}
        if key:
            fields[key] = td.get_text(" ", strip=True)

    item = _build_item(fields)
    if item is not None:
        items.append(item)

    return items
