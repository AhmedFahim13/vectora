"""Day-end archive scraper.

Endpoint (verified 2026-07-12):
  /day_end_archive.php?startDate=&endDate=&inst=All Instrument&archive=data
Table: <table class="shares-table fixedHeader">
Columns: # | DATE | TRADING CODE | LTP | HIGH | LOW | OPENP | CLOSEP | YCP
         | TRADE | VALUE (mn) | VOLUME
"""
from bs4 import BeautifulSoup

from vectora.http import PoliteSession
from vectora.settings import DSE_BASE

URL = f"{DSE_BASE}/day_end_archive.php"


def fetch_day_end(session: PoliteSession, start: str, end: str) -> str:
    return session.get(URL, params={
        "startDate": start, "endDate": end,
        "inst": "All Instrument", "archive": "data",
    })


def _num(text: str) -> float | None:
    t = text.strip().replace(",", "")
    if t in ("", "-", "--", "0.00-"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _int(text: str) -> int:
    v = _num(text)
    return int(v) if v is not None else 0


def parse_day_end(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="shares-table")
    if table is None:
        return []  # holiday / empty range renders no table
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 12:
            continue
        cells = [td.get_text(strip=True) for td in tds]
        rows.append({
            "symbol": cells[2],
            "date": cells[1],
            "ltp": _num(cells[3]),
            "high": _num(cells[4]),
            "low": _num(cells[5]),
            "open": _num(cells[6]),
            "close": _num(cells[7]),
            "ycp": _num(cells[8]),
            "trades": _int(cells[9]),
            "value_mn": _num(cells[10]),
            "volume": _int(cells[11]),
        })
    return rows
