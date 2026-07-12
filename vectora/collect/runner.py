"""Collect stage: fetch -> save raw -> parse -> upsert -> watermark.

Watermarks only advance when data actually landed; an empty EOD page on
an expected trading day is recorded in no_trade_days for validation
(Task 10) to assess, and the watermark stays put so a later re-run
retries the date.
"""
from datetime import date
from pathlib import Path

from vectora import db as vdb
from vectora.collect import dse_eod, dse_indices, dse_news
from vectora.collect.raw_store import save_raw
from vectora.settings import RAW_DIR


def collect_eod(con, session, run_date: date, raw_dir: Path = RAW_DIR) -> int:
    ds = run_date.isoformat()
    html = dse_eod.fetch_day_end(session, ds, ds)
    save_raw(raw_dir, "dse_eod", ds, "day_end_archive", html)
    rows = dse_eod.parse_day_end(html)
    if not rows:
        vdb.upsert(con, "no_trade_days", [{"date": ds, "reason": "no day-end table"}])
        return 0
    for r in rows:
        r["source"] = "dse_eod"
    n = vdb.upsert(con, "prices_raw", rows)
    vdb.set_watermark(con, "collect", "eod", ds)
    return n


def collect_news(con, session, run_date: date, raw_dir: Path = RAW_DIR) -> int:
    ds = run_date.isoformat()
    html = dse_news.fetch_news(session, ds, ds)
    save_raw(raw_dir, "dse_news", ds, "old_news", html)
    items = dse_news.parse_news(html)
    n = vdb.upsert(con, "events", items) if items else 0
    vdb.set_watermark(con, "collect", "news", ds)
    return n


def collect_indices(con, session, run_date: date, raw_dir: Path = RAW_DIR) -> int:
    ds = run_date.isoformat()
    html = dse_indices.fetch_homepage(session)
    save_raw(raw_dir, "dse_indices", ds, "homepage", html)
    parsed = dse_indices.parse_homepage(html)
    idx_rows = [{**i, "date": ds} for i in parsed["indices"]]
    if idx_rows:
        vdb.upsert(con, "indices", idx_rows)
    if parsed["totals"] is not None:
        vdb.upsert(con, "market_totals", [{**parsed["totals"], "date": ds}])
    if idx_rows or parsed["totals"]:
        vdb.set_watermark(con, "collect", "indices", ds)
    return 1 if idx_rows else 0
