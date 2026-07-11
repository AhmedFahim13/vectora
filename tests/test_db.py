# tests/test_db.py
import pytest

from vectora import db as vdb


def test_schema_creates_all_tables(test_db):
    tables = {r[0] for r in test_db.execute("SHOW TABLES").fetchall()}
    assert {
        "symbols", "prices_raw", "indices", "events", "company_snapshot",
        "holdings", "data_quality", "watermarks", "no_trade_days",
    } <= tables


def test_watermark_roundtrip(test_db):
    assert vdb.get_watermark(test_db, "collect", "eod") is None
    vdb.set_watermark(test_db, "collect", "eod", "2026-07-10")
    assert vdb.get_watermark(test_db, "collect", "eod") == "2026-07-10"
    vdb.set_watermark(test_db, "collect", "eod", "2026-07-12")  # overwrite
    assert vdb.get_watermark(test_db, "collect", "eod") == "2026-07-12"


def test_upsert_prices_is_idempotent(test_db):
    row = dict(symbol="GP", date="2026-07-09", open=280.0, high=285.0, low=279.0,
               close=284.1, ltp=284.0, ycp=280.5, trades=1500, value_mn=120.5,
               volume=425000, source="dse_eod")
    vdb.upsert(test_db, "prices_raw", [row, row])
    vdb.upsert(test_db, "prices_raw", [row])
    n = test_db.execute("SELECT count(*) FROM prices_raw").fetchone()[0]
    assert n == 1


def test_upsert_replaces_on_conflict(test_db):
    r1 = dict(symbol="GP", date="2026-07-09", open=1.0, high=1.0, low=1.0, close=1.0,
              ltp=1.0, ycp=1.0, trades=1, value_mn=1.0, volume=1, source="dse_eod")
    r2 = {**r1, "close": 2.0}
    vdb.upsert(test_db, "prices_raw", [r1])
    vdb.upsert(test_db, "prices_raw", [r2])
    close = test_db.execute("SELECT close FROM prices_raw").fetchone()[0]
    assert close == 2.0


def test_upsert_same_pk_different_values_in_one_batch(test_db):
    r1 = dict(symbol="GP", date="2026-07-09", open=1.0, high=1.0, low=1.0, close=1.0,
              ltp=1.0, ycp=1.0, trades=1, value_mn=1.0, volume=1, source="dse_eod")
    r2 = {**r1, "close": 2.0}
    vdb.upsert(test_db, "prices_raw", [r1, r2])  # same PK, different close
    rows = test_db.execute("SELECT close FROM prices_raw").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 2.0  # last row wins


def test_upsert_failed_batch_leaves_no_partial_rows(test_db):
    good = dict(symbol="GP", date="2026-07-09", open=1.0, high=1.0, low=1.0, close=1.0,
                ltp=1.0, ycp=1.0, trades=1, value_mn=1.0, volume=1, source="dse_eod")
    good2 = {**good, "symbol": "BEXIMCO"}
    bad = {**good, "symbol": "ROBI", "close": "not-a-number"}
    with pytest.raises(Exception):  # noqa: B017 - any conversion error counts
        vdb.upsert(test_db, "prices_raw", [good, good2, bad])
    n = test_db.execute("SELECT count(*) FROM prices_raw").fetchone()[0]
    assert n == 0  # atomic: nothing from the failed batch persists
