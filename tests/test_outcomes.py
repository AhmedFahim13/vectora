import datetime as dt

from vectora import db as vdb
from vectora.outcomes import resolver


def _price(symbol, d, close):
    return dict(symbol=symbol, date=d, open=close, high=close, low=close,
                close=close, ltp=close, ycp=close, trades=10, value_mn=1.0,
                volume=100, source="dse_eod")


def _pred(symbol, d, target="g5_h3", prob=0.6):
    return dict(id=f"{d}_{target}_{symbol}", symbol=symbol, date=d,
                target=target, probability=prob, model_id="m",
                quality_score=100, is_signal=True, suppressed_reason=None)


def _seed(con, closes, symbol="GP", start="2026-07-01"):
    d0 = dt.date.fromisoformat(start)
    rows = [_price(symbol, (d0 + dt.timedelta(days=i)).isoformat(), c)
            for i, c in enumerate(closes)]
    vdb.upsert(con, "prices_raw", rows)


def test_matured_prediction_resolves_hit(test_db):
    # close 100 on day0; next 3 closes: 103, 106, 104 -> max +6% >= 5% -> hit
    _seed(test_db, [100, 103, 106, 104])
    vdb.upsert(test_db, "predictions", [_pred("GP", "2026-07-01")])
    result = resolver.resolve(test_db)
    assert result == {"resolved": 1, "pending": 0}
    row = test_db.execute(
        "SELECT hit, realized_max, realized_min FROM outcomes").fetchone()
    assert row[0] is True
    assert abs(row[1] - 0.06) < 1e-9
    assert abs(row[2] - 0.03) < 1e-9


def test_matured_prediction_resolves_miss(test_db):
    _seed(test_db, [100, 101, 102, 101])   # max +2% < 5% -> miss
    vdb.upsert(test_db, "predictions", [_pred("GP", "2026-07-01")])
    resolver.resolve(test_db)
    assert test_db.execute("SELECT hit FROM outcomes").fetchone()[0] is False


def test_immature_prediction_stays_pending(test_db):
    _seed(test_db, [100, 103])             # only 1 forward row, horizon 3
    vdb.upsert(test_db, "predictions", [_pred("GP", "2026-07-01")])
    result = resolver.resolve(test_db)
    assert result == {"resolved": 0, "pending": 1}
    assert test_db.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 0


def test_resolve_is_idempotent_and_incremental(test_db):
    _seed(test_db, [100, 103, 106, 104])
    vdb.upsert(test_db, "predictions", [_pred("GP", "2026-07-01")])
    resolver.resolve(test_db)
    result = resolver.resolve(test_db)     # nothing new to do
    assert result == {"resolved": 0, "pending": 0}
    assert test_db.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 1


def test_multiple_targets_resolve_independently(test_db):
    _seed(test_db, [100, 103, 106, 104, 108, 112, 111, 115, 113, 116, 118])
    vdb.upsert(test_db, "predictions", [
        _pred("GP", "2026-07-01", target="g5_h3"),
        _pred("GP", "2026-07-01", target="g10_h10", prob=0.4),
    ])
    result = resolver.resolve(test_db)
    assert result["resolved"] == 2
    rows = dict(test_db.execute(
        "SELECT prediction_id, hit FROM outcomes").fetchall())
    assert rows["2026-07-01_g5_h3_GP"] is True     # +6% within 3
    assert rows["2026-07-01_g10_h10_GP"] is True   # +18% within 10
