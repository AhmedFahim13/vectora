"""DSE trading calendar: Sun-Thu trading, Fri/Sat weekend, plus holidays CSV.

The holidays file (data/reference/holidays.csv: date,description) is
maintained by hand; missing file just means "no known holidays yet" and a
malformed row is skipped rather than crashing the pipeline — a dropped
holiday degrades safely because an unexpected no-data day is recorded as
a no_trade_day by validation (Task 10), not treated as an error.
"""
import csv
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from vectora.settings import HOLIDAYS_CSV

WEEKEND = (4, 5)  # Friday, Saturday


@lru_cache(maxsize=4)
def _load_holidays_cached(path_str: str) -> frozenset[date]:
    path = Path(path_str)
    if not path.exists():
        return frozenset()
    holidays: set[date] = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                holidays.add(date.fromisoformat((row.get("date") or "").strip()))
            except ValueError:
                continue  # skip malformed hand-edited rows
    return frozenset(holidays)


def load_holidays(path: Path = HOLIDAYS_CSV) -> set[date]:
    return set(_load_holidays_cached(str(path)))


def is_trading_day(d: date, holidays: set[date] | None = None) -> bool:
    if d.weekday() in WEEKEND:
        return False
    hs = load_holidays() if holidays is None else holidays
    return d not in hs


def previous_trading_day(today: date, holidays: set[date] | None = None) -> date:
    hs = load_holidays() if holidays is None else holidays
    d = today - timedelta(days=1)
    while not is_trading_day(d, hs):
        d -= timedelta(days=1)
    return d
