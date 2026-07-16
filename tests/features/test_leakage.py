# tests/features/test_leakage.py
"""Anti-leakage invariant (spec §8): features for date t must not change
when future data (t+1...) changes. Every feature uses trailing windows or
per-date cross-sections only; this test catches any future violation."""
import datetime as dt

import polars as pl

from vectora import db as vdb
from vectora.features import engine


def _seed(con, n_days, close_fn):
    vdb.upsert(con, "symbols", [
        dict(symbol="AAA", name=None, sector="Bank", instrument_type="Equity",
             category="A", listing_status="active",
             first_seen="2020-01-01", last_seen="2026-12-31"),
        dict(symbol="BBB", name=None, sector="Textile", instrument_type="Equity",
             category="B", listing_status="active",
             first_seen="2021-01-01", last_seen="2026-12-31"),
    ])
    rows = []
    d0 = dt.date(2026, 1, 1)
    for i in range(n_days):
        d = (d0 + dt.timedelta(days=i)).isoformat()
        for sym, base_px in (("AAA", 100.0), ("BBB", 40.0)):
            px = close_fn(i, base_px)
            rows.append(dict(symbol=sym, date=d, open=px, high=px * 1.01,
                             low=px * 0.99, close=px, ltp=px, ycp=px,
                             trades=30, value_mn=3.0, volume=500 + i,
                             source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)


def test_future_data_does_not_change_past_features(tmp_path):
    def close_fn(i, base):
        return base * (1 + 0.001 * (i % 9))

    con1 = vdb.connect(tmp_path / "a.duckdb")
    vdb.init_schema(con1, backfill_parquet=tmp_path / "none.parquet")
    _seed(con1, 70, close_fn)
    f1 = engine.compute(con1, out_path=tmp_path / "f1.parquet")
    con1.close()

    def close_fn2(i, base):  # identical history, WILD different future
        return close_fn(i, base) if i < 70 else base * 3.0

    con2 = vdb.connect(tmp_path / "b.duckdb")
    vdb.init_schema(con2, backfill_parquet=tmp_path / "none.parquet")
    _seed(con2, 100, close_fn2)
    f2 = engine.compute(con2, out_path=tmp_path / "f2.parquet")
    con2.close()

    cutoff = dt.date(2026, 1, 1) + dt.timedelta(days=69)
    a = f1.filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
    b = f2.filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
    assert a.height == b.height
    for col in a.columns:
        if col in ("symbol", "date", "sector", "first_seen"):
            continue
        av, bv = a[col].to_list(), b[col].to_list()
        for x, y in zip(av, bv, strict=True):
            if x is None and y is None:
                continue
            assert x == y or abs(x - y) < 1e-12, \
                f"LEAKAGE: {col} changed for a past date when future data changed"
