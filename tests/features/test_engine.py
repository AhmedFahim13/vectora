# tests/features/test_engine.py
import polars as pl

from vectora import db as vdb
from vectora.features import engine, registry


def _seed(con, n_days=80):
    vdb.upsert(con, "symbols", [
        dict(symbol="AAA", name=None, sector="Bank", instrument_type="Equity",
             category="A", listing_status="active",
             first_seen="2020-01-01", last_seen="2026-07-09"),
    ])
    rows = []
    import datetime as dt
    d0 = dt.date(2026, 3, 1)
    for i in range(n_days):
        d = (d0 + dt.timedelta(days=i)).isoformat()
        px = 100 + (i % 7)
        rows.append(dict(symbol="AAA", date=d, open=px, high=px + 1, low=px - 1,
                         close=px, ltp=px, ycp=px - (i % 3 == 0), trades=50,
                         value_mn=5.0, volume=1000, source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)


def test_compute_produces_all_registered_columns(test_db, tmp_path):
    _seed(test_db)
    out_path = tmp_path / "features.parquet"
    df = engine.compute(test_db, out_path=out_path)
    expected = {s.name for s in registry.load()}
    assert expected <= set(df.columns)
    assert {"symbol", "date", "ret"} <= set(df.columns)
    assert out_path.exists()
    assert pl.read_parquet(out_path).height == df.height


def test_compute_row_count_matches_panel(test_db, tmp_path):
    _seed(test_db, n_days=40)
    df = engine.compute(test_db, out_path=tmp_path / "f.parquet")
    n = test_db.execute("SELECT count(*) FROM prices").fetchone()[0]
    assert df.height == n
