# tests/features/test_base.py
import polars as pl

from vectora import db as vdb
from vectora.features import base


def _seed(con):
    rows = [
        # scraped rows: ycp is ex-date adjusted, so ret must use close/ycp.
        # 2026-07-06: close 110 vs ycp 100 -> +10%
        dict(symbol="GP", date="2026-07-06", open=100, high=111, low=99, close=110,
             ltp=110, ycp=100, trades=10, value_mn=5.0, volume=1000, source="dse_eod"),
        # backfill rows: no ycp -> close/prev_close, clipped to +/-12%
        dict(symbol="ACI", date="2026-07-05", open=10, high=10, low=10, close=10.0,
             ltp=None, ycp=None, trades=None, value_mn=None, volume=500,
             source="mendeley"),
        dict(symbol="ACI", date="2026-07-06", open=10, high=11, low=10, close=11.0,
             ltp=None, ycp=None, trades=None, value_mn=None, volume=600,
             source="mendeley"),
        # a 50% "gap" (unadjusted rights/split) must clip to the 12% band
        dict(symbol="ACI", date="2026-07-07", open=16, high=17, low=16, close=16.5,
             ltp=None, ycp=None, trades=None, value_mn=None, volume=700,
             source="mendeley"),
    ]
    vdb.upsert(con, "prices_raw", rows)


def test_panel_has_canonical_return(test_db):
    _seed(test_db)
    df = base.load_panel(test_db)
    assert isinstance(df, pl.DataFrame)
    gp = df.filter(pl.col("symbol") == "GP")
    assert abs(gp["ret"][0] - 0.10) < 1e-9  # close/ycp - 1


def test_backfill_return_uses_prev_close_and_clips(test_db):
    _seed(test_db)
    df = base.load_panel(test_db).filter(pl.col("symbol") == "ACI").sort("date")
    rets = df["ret"].to_list()
    assert rets[0] is None                      # no previous close
    assert abs(rets[1] - 0.10) < 1e-9           # 11/10 - 1
    assert abs(rets[2] - base.RET_CLIP) < 1e-9  # 16.5/11 - 1 = 50% -> clipped


def test_panel_sorted_by_symbol_date(test_db):
    _seed(test_db)
    df = base.load_panel(test_db)
    # per-symbol dates strictly increasing
    for _, g in df.group_by("symbol"):
        ds = g.sort("date")["date"].to_list()
        assert ds == sorted(ds)


def test_backfill_return_prefers_adjusted_chain(test_db, tmp_path, monkeypatch):
    # unadjusted closes show a fake 2:1 split gap; adjusted chain is smooth
    vdb.upsert(test_db, "prices_raw", [
        dict(symbol="SPL", date="2026-07-05", open=100, high=100, low=100,
             close=100.0, ltp=None, ycp=None, trades=None, value_mn=None,
             volume=1, source="mendeley"),
        dict(symbol="SPL", date="2026-07-06", open=51, high=51, low=51,
             close=51.0, ltp=None, ycp=None, trades=None, value_mn=None,
             volume=1, source="mendeley"),
    ])
    adj = tmp_path / "adj.parquet"
    pl.DataFrame({
        "symbol": ["SPL", "SPL"],
        "date": ["2026-07-05", "2026-07-06"],
        "adj_close": [50.0, 51.0],
    }).with_columns(pl.col("date").cast(pl.Date)).write_parquet(adj)
    monkeypatch.setattr(base, "ADJUSTED_PARQUET", adj)
    df = base.load_panel(test_db).filter(pl.col("symbol") == "SPL").sort("date")
    # adjusted chain: 51/50 - 1 = +2%, NOT the clipped -12% the raw gap gives
    assert abs(df["ret"][1] - 0.02) < 1e-9


def test_backfill_without_adjusted_falls_back(test_db, tmp_path, monkeypatch):
    monkeypatch.setattr(base, "ADJUSTED_PARQUET", tmp_path / "missing.parquet")
    _seed(test_db)   # existing helper: ACI rows incl. the 50% gap day
    df = base.load_panel(test_db).filter(pl.col("symbol") == "ACI").sort("date")
    assert abs(df["ret"][2] - base.RET_CLIP) < 1e-9   # old clipped behavior


def test_zero_prices_become_a_zero_range_day_not_a_zero_price(test_db):
    """A zero low is a scrape gap. As 0 it becomes the lowest low in every
    window it touches; as null it blanks those windows for a year. Neither
    is right — the day's close is."""
    vdb.upsert(test_db, "prices_raw", [dict(
        symbol="BAD", date="2026-07-06", open=0.0, high=0.0, low=0.0,
        close=42.0, ltp=None, ycp=None, trades=1, value_mn=1.0, volume=10,
        source="dse_eod")])
    df = base.load_panel(test_db).filter(pl.col("symbol") == "BAD")
    row = df.row(0, named=True)
    assert row["low"] == 42.0
    assert row["high"] == 42.0
    assert row["open"] == 42.0


def test_good_prices_are_left_alone(test_db):
    vdb.upsert(test_db, "prices_raw", [dict(
        symbol="OK", date="2026-07-06", open=10.0, high=12.0, low=9.0,
        close=11.0, ltp=None, ycp=None, trades=1, value_mn=1.0, volume=10,
        source="dse_eod")])
    row = base.load_panel(test_db).filter(
        pl.col("symbol") == "OK").row(0, named=True)
    assert (row["open"], row["high"], row["low"]) == (10.0, 12.0, 9.0)
