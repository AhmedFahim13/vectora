# tests/validate/test_checks.py
from datetime import date

from vectora import db as vdb
from vectora.validate import checks


def _price(symbol, d, close=100.0, ycp=100.0, high=101.0, low=99.0, volume=1000):
    return dict(symbol=symbol, date=d, open=100.0, high=high, low=low, close=close,
                ltp=close, ycp=ycp, trades=10, value_mn=1.0, volume=volume,
                source="dse_eod")


def _seed_two_days(con, n_prev=5, n_today=5):
    prev = [_price(f"S{i}", "2026-07-08") for i in range(n_prev)]
    today = [_price(f"S{i}", "2026-07-09") for i in range(n_today)]
    vdb.upsert(con, "prices_raw", prev + today)


def test_clean_day_scores_100(test_db):
    _seed_two_days(test_db)
    result = checks.validate_day(test_db, date(2026, 7, 9), prev=date(2026, 7, 8))
    assert result["score"] == 100
    assert result["issues"] == []
    row = test_db.execute(
        "SELECT score FROM data_quality WHERE source = 'dse_eod'").fetchone()
    assert row == (100,)


def test_row_count_collapse_deducts(test_db):
    _seed_two_days(test_db, n_prev=10, n_today=3)  # 70% drop
    result = checks.validate_day(test_db, date(2026, 7, 9), prev=date(2026, 7, 8))
    assert result["score"] < 100
    assert any("row count" in i for i in result["issues"])


def test_impossible_ohlc_deducts(test_db):
    _seed_two_days(test_db)
    vdb.upsert(test_db, "prices_raw",
               [_price("BAD", "2026-07-09", high=90.0, low=95.0)])  # high < low
    result = checks.validate_day(test_db, date(2026, 7, 9), prev=date(2026, 7, 8))
    assert result["score"] < 100
    assert any("high<low" in i for i in result["issues"])


def test_excessive_move_deducts(test_db):
    _seed_two_days(test_db)
    vdb.upsert(test_db, "prices_raw",
               [_price("JUMP", "2026-07-09", close=150.0, ycp=100.0)])  # +50% move
    result = checks.validate_day(test_db, date(2026, 7, 9), prev=date(2026, 7, 8))
    assert result["score"] < 100
    assert any("band" in i for i in result["issues"])


def test_missing_day_scores_zero(test_db):
    result = checks.validate_day(test_db, date(2026, 7, 9), prev=date(2026, 7, 8))
    assert result["score"] == 0
    assert any("no rows" in i for i in result["issues"])


def test_score_floor_is_zero(test_db):
    # many bad rows must not push the score negative
    _seed_two_days(test_db, n_prev=50, n_today=1)
    bad = [_price(f"B{i}", "2026-07-09", high=90.0, low=95.0) for i in range(30)]
    vdb.upsert(test_db, "prices_raw", bad)
    result = checks.validate_day(test_db, date(2026, 7, 9), prev=date(2026, 7, 8))
    assert 0 <= result["score"] <= 100
