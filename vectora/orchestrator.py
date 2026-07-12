"""EOD pipeline orchestration: collect -> validate, stage-isolated failures.

run_eod is the testable core with injected dependencies; run_eod_live wires
the real session, database, and calendar for the CLI and CI. A stage failure
is recorded and the remaining stages still run — a broken news page must not
cost us the day's prices. The quality score decides usability downstream.
"""
from datetime import date
from pathlib import Path

from vectora import calendar as cal
from vectora import db as vdb
from vectora.collect import runner
from vectora.http import PoliteSession
from vectora.settings import DB_PATH, RAW_DIR
from vectora.validate import checks


def run_eod(con, session, run_date: date, raw_dir: Path = RAW_DIR,
            holidays: set[date] | None = None) -> dict:
    summary: dict = {
        "date": run_date.isoformat(), "skipped": None,
        "eod_rows": None, "news_items": None, "index_rows": None,
        "quality_score": None,
    }
    if not cal.is_trading_day(run_date, holidays):
        summary["skipped"] = "not a trading day"
        return summary

    errors: list[str] = []

    def _stage(name: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - stage isolation is the point
            errors.append(f"{name}: {exc}")
            return None

    summary["eod_rows"] = _stage(
        "eod", lambda: runner.collect_eod(con, session, run_date, raw_dir=raw_dir))
    summary["news_items"] = _stage(
        "news", lambda: runner.collect_news(con, session, run_date, raw_dir=raw_dir))
    summary["index_rows"] = _stage(
        "indices", lambda: runner.collect_indices(con, session, run_date, raw_dir=raw_dir))

    prev = cal.previous_trading_day(run_date, holidays)
    result = _stage("validate", lambda: checks.validate_day(con, run_date, prev=prev))
    if result is not None:
        summary["quality_score"] = result["score"]

    if errors:
        summary["errors"] = errors
    return summary


def run_eod_live(date_str: str | None) -> dict:
    """Wire real dependencies and run one EOD cycle (used by CLI/CI)."""
    holidays = cal.load_holidays()
    run_date = date.fromisoformat(date_str) if date_str else date.today()
    con = vdb.connect(DB_PATH)
    try:
        vdb.init_schema(con)
        return run_eod(con, PoliteSession(), run_date, holidays=holidays)
    finally:
        con.close()
