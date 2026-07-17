# tests/test_intraday_alerts.py
import polars as pl

from vectora import db as vdb
from vectora.alerts import intraday


def _seed_history(con, symbols=("NORM", "SURGE", "LIMIT"), days=30):
    import datetime as dt
    rows = []
    d0 = dt.date(2026, 6, 1)
    for i in range(days):
        d = d0 + dt.timedelta(days=i)
        for s in symbols:
            rows.append(dict(symbol=s, date=d, open=10.0, high=10.1, low=9.9,
                             close=10.0, ltp=10.0, ycp=10.0, trades=50,
                             value_mn=5.0, volume=10000, source="dse_eod"))
    df = pl.DataFrame(rows)  # noqa: F841
    con.execute("INSERT INTO prices_raw SELECT * FROM df")
    vdb.upsert(con, "symbols", [
        dict(symbol=s, name=None, sector="Bank", instrument_type="Equity",
             category="A", listing_status="active", first_seen="2020-01-01",
             last_seen="2026-12-31") for s in symbols])


def _snap(symbol, volume=10000, ltp=10.0, ycp=10.0, value_mn=5.0):
    return dict(symbol=symbol, ltp=ltp, high=ltp, low=ltp, closep=ltp,
                ycp=ycp, change=0.0, trades=100, value_mn=value_mn,
                volume=volume)


def test_volume_surge_and_limit_detected(test_db):
    _seed_history(test_db)
    snaps = [_snap("NORM"),
             _snap("SURGE", volume=40000),          # 4x median full day
             _snap("LIMIT", ltp=10.9, ycp=10.0)]    # +9%
    anomalies = intraday.detect(test_db, snaps, "2026-07-17 12:00:00")
    kinds = {(a["symbol"], a["kind"]) for a in anomalies}
    assert ("SURGE", "volume_surge") in kinds
    assert ("LIMIT", "near_circuit") in kinds
    assert not any(a["symbol"] == "NORM" for a in anomalies)


def test_illiquid_names_ignored(test_db):
    _seed_history(test_db)
    snaps = [_snap("SURGE", volume=40000, value_mn=0.05)]  # dust turnover
    assert intraday.detect(test_db, snaps, "2026-07-17 12:00:00") == []


def test_cooldown_suppresses_repeat(test_db):
    _seed_history(test_db)
    snaps = [_snap("SURGE", volume=40000)]
    first = intraday.filter_and_log(
        test_db, intraday.detect(test_db, snaps, "2026-07-17 12:00:00"),
        "2026-07-17")
    assert [a["symbol"] for a in first] == ["SURGE"]
    second = intraday.filter_and_log(
        test_db, intraday.detect(test_db, snaps, "2026-07-17 13:00:00"),
        "2026-07-17")
    assert second == []


def test_daily_email_cap(test_db):
    for i in range(3):
        vdb.upsert(test_db, "alerts_log", [dict(
            id=f"2026-07-17_intraday_email_{i}", alert_type="intraday_email",
            symbol=None, alert_date="2026-07-17", prediction_id=None)])
    assert intraday.email_allowed(test_db, "2026-07-17") is False
    assert intraday.email_allowed(test_db, "2026-07-16") is True


def test_render_body_mentions_anomalies():
    body = intraday.render(
        [{"symbol": "SURGE", "kind": "volume_surge", "ratio": 4.0,
          "detail": "40,000 vs 21d median 10,000"},
         {"symbol": "LIMIT", "kind": "near_circuit", "ratio": 0.09,
          "detail": "+9.0% vs YCP"}], "2026-07-17 12:00:00")
    assert "SURGE" in body and "volume surge" in body.lower()
    assert "LIMIT" in body and "9.0%" in body
    assert "not investment advice" in body.lower()
