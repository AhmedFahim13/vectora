"""DSE trading calendar: Sun-Thu trading, Fri/Sat weekend, plus holidays CSV.

The holidays file (data/reference/holidays.csv: date,description) is
maintained by hand; missing file just means "no known holidays yet".
An unexpected no-data day is caught by validation instead (Task 10).
"""
import csv
from datetime import date, timedelta
from pathlib import Path

from vectora.settings import HOLIDAYS_CSV

WEEKEND = (4, 5)  # Friday, Saturday


def load_holidays(path: Path = HOLIDAYS_CSV) -> set[date]:
    if not Path(path).exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {date.fromisoformat(row["date"]) for row in csv.DictReader(f)}


def is_trading_day(d: date, holidays: set[date] | None = None) -> bool:
    if d.weekday() in WEEKEND:
        return False
    hs = load_holidays() if holidays is None else holidays
    return d not in hs


def last_trading_day(today: date, holidays: set[date] | None = None) -> date:
    hs = load_holidays() if holidays is None else holidays
    d = today - timedelta(days=1)
    while not is_trading_day(d, hs):
        d -= timedelta(days=1)
    return d
