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


def test_event_and_regime_base_columns(test_db, tmp_path):
    _seed(test_db, n_days=80)
    vdb.upsert(test_db, "events", [dict(
        id="ev1", post_date="2026-04-20", symbol="AAA",
        title="AAA: Q1 Financials", body="", source="dse_news")])
    vdb.upsert(test_db, "event_labels", [dict(
        event_id="ev1", event_type="earnings_release", materiality=3)])
    vdb.upsert(test_db, "regimes", [dict(
        date="2026-05-10", regime="Bull", confidence=0.8, method="rules")])
    df = engine.compute(test_db, out_path=tmp_path / "f.parquet")
    import datetime as dt
    row = df.filter((pl.col("symbol") == "AAA")
                    & (pl.col("date") == dt.date(2026, 4, 25)))
    assert row["days_since_event"][0] == 5
    assert row["board_meeting_soon"][0] == 0
    before = df.filter((pl.col("symbol") == "AAA")
                       & (pl.col("date") == dt.date(2026, 4, 10)))
    assert before["days_since_event"][0] is None    # no event yet
    reg = df.filter(pl.col("date") == dt.date(2026, 5, 10))
    assert (reg["regime_code"] == 5).all()          # Bull -> 5
    unclass = df.filter(pl.col("date") == dt.date(2026, 4, 25))
    assert (unclass["regime_code"] == 0).all()      # unclassified -> 0
