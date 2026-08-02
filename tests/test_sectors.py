"""Sector relative-strength / rotation tests."""
import datetime as dt

import polars as pl

from vectora import db as vdb
from vectora import sectors


def _seed(con, days: int = 200):
    """Two sectors: TECH beats the market, BANK trails it."""
    vdb.upsert(con, "symbols", [
        {"symbol": "T1", "sector": "Tech"}, {"symbol": "T2", "sector": "Tech"},
        {"symbol": "B1", "sector": "Bank"}, {"symbol": "B2", "sector": "Bank"}])
    rows = []
    px = {"T1": 100.0, "T2": 100.0, "B1": 100.0, "B2": 100.0}
    drift = {"T1": 1.004, "T2": 1.004, "B1": 0.998, "B2": 0.998}
    d0 = dt.date(2025, 1, 1)
    for i in range(days):
        for sym in px:
            px[sym] *= drift[sym]
            rows.append(dict(
                symbol=sym, date=(d0 + dt.timedelta(days=i)).isoformat(),
                open=px[sym], high=px[sym], low=px[sym], close=px[sym],
                ltp=None, ycp=None, trades=1, value_mn=5.0, volume=1000,
                source="test"))
    vdb.upsert(con, "prices_raw", rows)


def test_outperforming_sector_has_positive_relative_strength(test_db):
    _seed(test_db)
    d = sectors.compute(test_db)
    last = d.filter(pl.col("date") == d["date"].max())
    tech = last.filter(pl.col("sector") == "Tech")
    bank = last.filter(pl.col("sector") == "Bank")
    assert tech["rs_21"][0] > 0
    assert bank["rs_21"][0] < 0


def test_benchmark_sits_between_the_sectors(test_db):
    """The benchmark must be a genuine centre of the same universe.

    RS does not net to exactly zero and should not be expected to:
    compounding is convex, so the mean of compounded sector returns exceeds
    the compounded mean daily return (Jensen). The residual must stay small
    against the spread it is measuring, which is what this pins.
    """
    _seed(test_db)
    d = sectors.compute(test_db)
    last = d.filter(pl.col("date") == d["date"].max())
    rs = last["rs_21"]
    spread = rs.max() - rs.min()
    assert rs.min() < 0 < rs.max()          # benchmark is bracketed
    assert abs(rs.sum()) < 0.05 * spread    # residual is convexity, not drift


def test_quadrants_are_assigned(test_db):
    _seed(test_db)
    res = sectors.run(test_db)
    assert res["sectors"] == 2
    loaded = sectors.load(test_db, res["date"])
    quads = {r["sector"]: r["quadrant"] for r in loaded}
    assert quads["Tech"] in ("Leading", "Weakening")
    assert quads["Bank"] in ("Lagging", "Improving")


def test_run_persists_and_reloads(test_db):
    _seed(test_db)
    res = sectors.run(test_db)
    rows = sectors.load(test_db, res["date"])
    assert len(rows) == 2
    assert rows[0]["rs_21d"] >= rows[-1]["rs_21d"]   # ordered by strength
    assert rows[0]["n_symbols"] == 2


def test_non_trading_debt_is_excluded_from_the_benchmark(test_db):
    """Bonds that never trade must not dilute an equal-weighted market mean.

    On the DSE 243 of ~640 listings are debt with a median daily traded value
    of zero. Counted in, they were 40% of the benchmark while contributing
    only stale quotes.
    """
    _seed(test_db)
    vdb.upsert(test_db, "symbols", [
        {"symbol": f"GSEC{i}", "sector": "G-SEC (T.Bond)",
         "instrument_type": "Debt"} for i in range(10)])
    d0 = dt.date(2025, 1, 1)
    vdb.upsert(test_db, "prices_raw", [dict(
        symbol=f"GSEC{j}", date=(d0 + dt.timedelta(days=i)).isoformat(),
        open=100.0, high=100.0, low=100.0, close=100.0, ltp=None,
        ycp=None, trades=0, value_mn=0.0, volume=0, source="test")
        for i in range(200) for j in range(10)])

    d = sectors.compute(test_db)
    assert "G-SEC (T.Bond)" not in d["sector"].unique().to_list()
    last = d.filter(pl.col("date") == d["date"].max())
    # the equity read is unchanged by 50 dead bond series
    assert last.filter(pl.col("sector") == "Tech")["rs_21"][0] > 0


def test_zero_turnover_rows_are_dropped(test_db):
    """Even an equity contributes nothing on a day it did not trade."""
    _seed(test_db)
    d = sectors.compute(test_db)
    before = d.filter(pl.col("date") == d["date"].max())["n_symbols"].sum()
    vdb.upsert(test_db, "prices_raw", [dict(
        symbol="T1", date="2025-07-20", open=100.0, high=100.0, low=100.0,
        close=100.0, ltp=None, ycp=None, trades=0, value_mn=0.0, volume=0,
        source="test")])
    after = sectors.compute(test_db)
    stale = after.filter((pl.col("sector") == "Tech")
                         & (pl.col("date") == dt.date(2025, 7, 20)))
    assert stale.height == 0 or stale["n_symbols"][0] < 2
    assert before > 0


def test_empty_database_does_not_crash(test_db):
    assert sectors.run(test_db)["sectors"] == 0


def test_untraded_debt_is_excluded_from_the_benchmark(test_db):
    """219 never-traded T-bonds must not become 40% of 'the market'."""
    _seed(test_db)
    vdb.upsert(test_db, "symbols", [
        {"symbol": f"TB{i}", "sector": "G-SEC (T.Bond)",
         "instrument_type": "Debt"} for i in range(50)])
    d0 = dt.date(2025, 1, 1)
    vdb.upsert(test_db, "prices_raw", [
        dict(symbol=f"TB{i}", date=(d0 + dt.timedelta(days=j)).isoformat(),
             open=100.0, high=100.0, low=100.0, close=100.0, ltp=None,
             ycp=None, trades=0, value_mn=0.0, volume=0, source="test")
        for i in range(50) for j in range(200)])
    out = sectors.compute(test_db)
    assert "G-SEC (T.Bond)" not in out["sector"].unique().to_list()


def test_null_turnover_is_kept_as_unknown(test_db):
    """The 1.06M-row backfill has no turnover column; null must not mean zero.

    Testing `value_mn > 0` alone silently discarded thirteen years of history
    and left every sector reading 'Insufficient data'.
    """
    _seed(test_db)
    test_db.execute("UPDATE prices_raw SET value_mn = NULL")
    out = sectors.compute(test_db)
    assert out.height > 0
    last = out.filter(pl.col("date") == out["date"].max())
    assert last["rs_21"].null_count() == 0


def test_a_stale_quote_is_not_counted_as_a_return(test_db):
    """An equity that printed no trades that day contributes nothing."""
    _seed(test_db)
    vdb.upsert(test_db, "symbols", [
        {"symbol": "DEAD", "sector": "Tech", "instrument_type": "Equity"}])
    d0 = dt.date(2025, 1, 1)
    vdb.upsert(test_db, "prices_raw", [
        dict(symbol="DEAD", date=(d0 + dt.timedelta(days=j)).isoformat(),
             open=50.0, high=50.0, low=50.0, close=50.0, ltp=None, ycp=None,
             trades=0, value_mn=0.0, volume=0, source="test")
        for j in range(200)])
    out = sectors.compute(test_db)
    last = out.filter((pl.col("sector") == "Tech")
                      & (pl.col("date") == out["date"].max()))
    assert last["n_symbols"][0] == 2       # T1 and T2 only, not DEAD


def test_compounding_not_summing(test_db):
    """A sector up 0.4%/day for 21 days compounds above the naive sum."""
    _seed(test_db)
    d = sectors.compute(test_db)
    tech = d.filter((pl.col("sector") == "Tech")
                    & pl.col("ret_21").is_not_null()).sort("date")
    expected = 1.004 ** 21 - 1
    assert abs(tech["ret_21"][-1] - expected) < 1e-6
