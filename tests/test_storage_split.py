# tests/test_storage_split.py
from vectora import db as vdb


def _price(symbol, d, close, source):
    return dict(symbol=symbol, date=d, open=close, high=close, low=close,
                close=close, ltp=close, ycp=close, trades=1, value_mn=1.0,
                volume=100, source=source)


def test_prices_view_without_parquet_is_prices_raw(test_db):
    vdb.upsert(test_db, "prices_raw", [_price("GP", "2026-07-09", 10.0, "dse_eod")])
    rows = test_db.execute("SELECT count(*) FROM prices").fetchone()[0]
    assert rows == 1


def test_prices_view_unions_backfill_parquet(tmp_path):
    import polars as pl
    pq = tmp_path / "backfill.parquet"
    pl.DataFrame({
        "symbol": ["GP"], "date": ["2013-01-02"], "open": [5.0], "high": [5.0],
        "low": [5.0], "close": [5.0], "ltp": [None], "ycp": [None],
        "trades": [None], "value_mn": [None], "volume": [1000],
        "source": ["mendeley"],
    }).with_columns(pl.col("date").cast(pl.Date)).write_parquet(pq)
    con = vdb.connect(tmp_path / "t.duckdb")
    vdb.init_schema(con, backfill_parquet=pq)
    vdb.upsert(con, "prices_raw", [_price("GP", "2026-07-09", 10.0, "dse_eod")])
    rows = con.execute("SELECT count(*) FROM prices").fetchone()[0]
    assert rows == 2
    srcs = {r[0] for r in con.execute("SELECT DISTINCT source FROM prices").fetchall()}
    assert srcs == {"mendeley", "dse_eod"}
    con.close()


def test_model_registry_table_exists(test_db):
    cols = {r[0] for r in test_db.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'model_registry'").fetchall()}
    assert {"model_id", "family", "target", "trained_at", "metrics", "active"} <= cols
