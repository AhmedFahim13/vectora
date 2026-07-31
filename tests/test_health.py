# tests/test_health.py
import datetime as dt

from vectora import db as vdb
from vectora import health


def _seed_fresh(con, d="2026-07-16"):
    vdb.upsert(con, "prices_raw", [dict(
        symbol="GP", date=d, open=1, high=1, low=1, close=1.0, ltp=1, ycp=1,
        trades=1, value_mn=1.0, volume=1, source="dse_eod")])
    vdb.upsert(con, "data_quality", [dict(
        date=d, source="dse_eod", score=100, issues="[]")])
    vdb.upsert(con, "regimes", [dict(
        date=d, regime="Sideways", confidence=0.5, method="rules")])
    vdb.upsert(con, "predictions", [dict(
        id=f"{d}_g5_h10_GP", symbol="GP", date=d, target="g5_h10",
        probability=0.4, model_id="m", quality_score=100, is_signal=False,
        suppressed_reason="below-probability-threshold")])
    vdb.set_watermark(con, "collect", "eod", d)


def test_all_green_when_fresh(test_db):
    _seed_fresh(test_db)
    # 2026-07-17 was Friday (weekend): last expected trading day = Thu 07-16
    result = health.check(test_db, today=dt.date(2026, 7, 18),
                          holidays=set())
    assert result["ok"] is True
    names = {c["name"] for c in result["checks"]}
    assert {"freshness", "quality", "tables", "watermark"} <= names


def test_stale_prices_fail_freshness(test_db):
    _seed_fresh(test_db, d="2026-07-09")   # a week old
    result = health.check(test_db, today=dt.date(2026, 7, 18),
                          holidays=set())
    assert result["ok"] is False
    fresh = next(c for c in result["checks"] if c["name"] == "freshness")
    assert fresh["ok"] is False and "2026-07-16" in fresh["detail"]


def test_low_quality_fails(test_db):
    _seed_fresh(test_db)
    vdb.upsert(test_db, "data_quality", [dict(
        date="2026-07-16", source="dse_eod", score=40, issues="[]")])
    result = health.check(test_db, today=dt.date(2026, 7, 18),
                          holidays=set())
    assert next(c for c in result["checks"]
                if c["name"] == "quality")["ok"] is False


def test_canaries_with_fake_session(test_db):
    _seed_fresh(test_db)

    class GoodSession:
        def get(self, url, params=None):
            return "<table class='shares-table'></table><div class='midrow'>"

    class BrokenSession:
        def get(self, url, params=None):
            return "<html>redesigned!</html>"

    ok = health.check(test_db, today=dt.date(2026, 7, 18), holidays=set(),
                      session=GoodSession())
    assert next(c for c in ok["checks"]
                if c["name"] == "canary")["ok"] is True
    bad = health.check(test_db, today=dt.date(2026, 7, 18), holidays=set(),
                       session=BrokenSession())
    assert bad["ok"] is False


def test_stale_predictions_fail_even_when_prices_fresh(test_db):
    """The 2026-07-19 outage: prices kept flowing while predict was dead."""
    _seed_fresh(test_db)
    test_db.execute("DELETE FROM predictions")
    result = health.check(test_db, today=dt.date(2026, 7, 18), holidays=set())
    assert result["ok"] is False
    assert next(c for c in result["checks"]
                if c["name"] == "predictions")["ok"] is False
