import datetime as dt

import numpy as np
import polars as pl

from vectora import db as vdb
from vectora.regime import state


def _seed(con, n_days=300, n_syms=60, seed=4, vol_mult=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = dt.date(2025, 1, 1)
    px = {f"S{i:02d}": 100.0 for i in range(n_syms)}
    for day in range(n_days):
        d = d0 + dt.timedelta(days=day)
        for sym in px:
            px[sym] *= float(np.exp(rng.normal(0.0003, 0.01 * vol_mult)))
            p = round(max(px[sym], 1.0), 2)
            rows.append(dict(symbol=sym, date=d, open=p, high=p * 1.01,
                             low=p * 0.99, close=p, ltp=p, ycp=p, trades=20,
                             value_mn=2.0, volume=int(rng.integers(500, 5000)),
                             source="dse_eod"))
    # bulk insert: 18k rows through executemany-based upsert takes minutes,
    # a registered polars frame takes milliseconds
    df = pl.DataFrame(rows)  # noqa: F841 - registered by name below
    con.execute("INSERT INTO prices_raw SELECT * FROM df")


def test_regimes_table_exists(test_db):
    tables = {r[0] for r in test_db.execute("SHOW TABLES").fetchall()}
    assert "regimes" in tables


def test_market_state_shape_and_columns(test_db):
    _seed(test_db)
    st = state.market_state(test_db)
    assert set(st.columns) >= {"date", "med_ret", "mkt_level", "ma50",
                               "ma200", "ret_21d", "vol_21d", "vol_pctile",
                               "breadth", "activity_z"}
    assert st.height == 300                      # one row per trading date
    assert st["date"].is_sorted()


def test_market_state_values_sane(test_db):
    _seed(test_db)
    st = state.market_state(test_db)
    last = st.tail(1).row(0, named=True)
    assert last["ma200"] is not None             # 300 days > 200 warmup
    assert 0.0 <= last["breadth"] <= 1.0
    assert 0.0 <= last["vol_pctile"] <= 1.0
    assert last["mkt_level"] > 0
    # med_ret of ~0.03% drift stays small
    assert abs(last["med_ret"]) < 0.05


def test_sparse_dates_are_dropped(test_db):
    _seed(test_db, n_syms=60)
    # one extra date with only 3 symbols must not produce a state row
    vdb.upsert(test_db, "prices_raw", [
        dict(symbol=f"S{i:02d}", date="2026-06-01", open=10, high=10, low=10,
             close=10, ltp=10, ycp=10, trades=1, value_mn=0.1, volume=10,
             source="dse_eod") for i in range(3)])
    st = state.market_state(test_db)
    assert str(st["date"].max()) != "2026-06-01"
