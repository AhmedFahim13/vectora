"""Intraday snapshot collector (spec §16, Phase 4D).

Endpoint (verified 2026-07-17): /latest_share_price_scroll_l.php —
same shares-table markup as the day-end archive, 11 columns:
# | TRADING CODE | LTP | HIGH | LOW | CLOSEP | YCP | CHANGE | TRADE
| VALUE (mn) | VOLUME. DSE refreshes chart data hourly, 4x per trading
day; the workflow polls exactly those publication points.
"""
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from vectora import db as vdb
from vectora.collect.dse_eod import _int, _num
from vectora.collect.raw_store import save_raw
from vectora.http import PoliteSession
from vectora.settings import DSE_BASE, RAW_DIR

URL = f"{DSE_BASE}/latest_share_price_scroll_l.php"


def fetch_latest(session: PoliteSession) -> str:
    return session.get(URL)


def parse_latest(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="shares-table")
    if table is None:
        return []
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 11:
            continue
        c = [td.get_text(strip=True) for td in tds]
        rows.append({
            "symbol": c[1], "ltp": _num(c[2]), "high": _num(c[3]),
            "low": _num(c[4]), "closep": _num(c[5]), "ycp": _num(c[6]),
            "change": _num(c[7]), "trades": _int(c[8]),
            "value_mn": _num(c[9]), "volume": _int(c[10]),
        })
    return rows


def collect_intraday(con, session, ts: str | None = None,
                     raw_dir: Path = RAW_DIR) -> int:
    stamp = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    day, hhmm = stamp[:10], stamp[11:16].replace(":", "")
    html = fetch_latest(session)
    save_raw(raw_dir, "dse_intraday", day, f"latest_{hhmm}", html, url=URL)
    rows = parse_latest(html)
    if not rows:
        return 0
    return vdb.upsert(con, "intraday_snapshots",
                      [{**r, "ts": stamp} for r in rows])
