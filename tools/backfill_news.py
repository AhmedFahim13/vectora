# tools/backfill_news.py
"""Backfill the DSE announcement archive month-by-month via old_news.php.

Usage: uv run python tools/backfill_news.py [START [END]]
Defaults: 2013-01-01 through today. ~165 polite requests (1.5s spacing),
raw pages land in data/raw/dse_news_backfill/<month>/, parsed items upsert
into events (sha256 ids make re-runs idempotent). Event history is the
fuel for spec §12.2 event-impact studies.
"""
import calendar
import sys
from datetime import date
from pathlib import Path

from vectora import db as vdb
from vectora.collect import dse_news
from vectora.collect.raw_store import save_raw
from vectora.http import PoliteSession
from vectora.settings import DB_PATH, RAW_DIR


def month_windows(start: str, end: str) -> list[tuple[str, str]]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    cur = s
    while cur <= e:
        last_day = date(cur.year, cur.month,
                        calendar.monthrange(cur.year, cur.month)[1])
        win_end = min(last_day, e)
        out.append((cur.isoformat(), win_end.isoformat()))
        cur = last_day.replace(day=1)
        cur = (date(cur.year + 1, 1, 1) if cur.month == 12
               else date(cur.year, cur.month + 1, 1))
    return out


def backfill(con, session, start: str, end: str,
             raw_dir: Path = RAW_DIR) -> dict:
    months = items = 0
    for win_start, win_end in month_windows(start, end):
        html = dse_news.fetch_news(session, win_start, win_end)
        save_raw(raw_dir, "dse_news_backfill", win_start[:7], "old_news",
                 html, url=dse_news.URL)
        parsed = dse_news.parse_news(html)
        if parsed:
            items += vdb.upsert(con, "events", parsed)
        months += 1
        print(f"{win_start[:7]}: {len(parsed)} items", flush=True)
    return {"months": months, "items": items}


def main() -> int:
    start = sys.argv[1] if len(sys.argv) > 1 else "2013-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()
    con = vdb.connect(DB_PATH)
    try:
        vdb.init_schema(con)
        result = backfill(con, PoliteSession(), start, end)
        print(result)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
