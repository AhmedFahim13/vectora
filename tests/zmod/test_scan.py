# tests/zmod/test_scan.py
import datetime as dt

import numpy as np
import polars as pl

from vectora import db as vdb
from vectora.zmod import scan


def _seed_market(con, n_days=90, n_syms=35, seed=13):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = dt.date(2026, 4, 1)
    px = {f"S{i:02d}": 50.0 for i in range(n_syms)}
    prev = dict(px)
    for day in range(n_days):
        d = d0 + dt.timedelta(days=day)
        for sym in px:
            # S00 is the pump: strong drift + volume surge in the last month
            pumping = sym == "S00" and day >= n_days - 10
            drift = 0.03 if pumping else 0.0
            px[sym] *= float(np.exp(rng.normal(drift, 0.01)))
            p = round(max(px[sym], 1.0), 2)
            vol = int(rng.integers(20000, 40000)) if pumping \
                else int(rng.integers(1000, 3000))
            y = round(max(prev[sym], 1.0), 2)
            prev[sym] = px[sym]
            rows.append(dict(symbol=sym, date=d, open=p, high=p * 1.01,
                             low=p * 0.99, close=p, ltp=p, ycp=y, trades=20,
                             value_mn=2.0, volume=vol, source="dse_eod"))
    df = pl.DataFrame(rows)  # noqa: F841
    con.execute("INSERT INTO prices_raw SELECT * FROM df")
    vdb.upsert(con, "symbols", [
        dict(symbol=s, name=None, sector="Bank", instrument_type="Equity",
             category="Z" if s == "S00" else "A", listing_status="active",
             first_seen="2020-01-01", last_seen="2026-12-31")
        for s in px])
    return (d0 + dt.timedelta(days=n_days - 1)).isoformat()


def test_zscan_flags_the_pump_and_writes_zwatch(test_db, tmp_path):
    last = _seed_market(test_db)
    result = scan.run_zscan(test_db, date_str=last,
                            features_path=tmp_path / "f.parquet")
    assert result["pump_flags"] >= 1
    top = test_db.execute(
        "SELECT symbol, score, phase FROM zwatch WHERE kind='pump' "
        "ORDER BY score DESC LIMIT 1").fetchone()
    assert top[0] == "S00"
    assert top[1] > 60
    assert top[2] in ("markup", "distribution")


def test_zscan_idempotent(test_db, tmp_path):
    last = _seed_market(test_db)
    scan.run_zscan(test_db, date_str=last, features_path=tmp_path / "f.parquet")
    scan.run_zscan(test_db, date_str=last, features_path=tmp_path / "g.parquet")
    n = test_db.execute(
        "SELECT count(*) FROM (SELECT DISTINCT date, symbol, kind FROM zwatch)"
    ).fetchone()[0]
    total = test_db.execute("SELECT count(*) FROM zwatch").fetchone()[0]
    assert n == total
