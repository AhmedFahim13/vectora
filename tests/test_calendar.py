# tests/test_calendar.py
from datetime import date

from vectora import calendar as cal


def test_friday_saturday_are_not_trading_days():
    assert not cal.is_trading_day(date(2026, 7, 10), holidays=set())  # Friday
    assert not cal.is_trading_day(date(2026, 7, 11), holidays=set())  # Saturday
    assert cal.is_trading_day(date(2026, 7, 12), holidays=set())      # Sunday


def test_holiday_is_not_trading_day():
    eid = date(2026, 3, 20)
    assert not cal.is_trading_day(eid, holidays={eid})


def test_last_trading_day_skips_weekend_and_holidays():
    # From Sunday 2026-07-12, previous trading day is Thursday 2026-07-09
    assert cal.last_trading_day(date(2026, 7, 12), holidays=set()) == date(2026, 7, 9)
    # If Thursday were a holiday, it must skip to Wednesday
    assert cal.last_trading_day(
        date(2026, 7, 12), holidays={date(2026, 7, 9)}
    ) == date(2026, 7, 8)


def test_load_holidays_parses_csv(tmp_path):
    p = tmp_path / "holidays.csv"
    p.write_text("date,description\n2026-03-20,Eid holiday\n2026-12-16,Victory Day\n",
                 encoding="utf-8")
    hs = cal.load_holidays(p)
    assert date(2026, 3, 20) in hs and date(2026, 12, 16) in hs
    assert len(hs) == 2


def test_load_holidays_missing_file_returns_empty(tmp_path):
    assert cal.load_holidays(tmp_path / "nope.csv") == set()
