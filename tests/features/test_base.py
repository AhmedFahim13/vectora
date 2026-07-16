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
