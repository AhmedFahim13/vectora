# tests/zmod/test_footprint.py
import datetime as dt

import polars as pl

from vectora import db as vdb
from vectora.zmod import footprint


def _features(rows):
    """Minimal feature frame: symbol, date, ret, volume_z_21d."""
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _mk(symbol, d0, days, volz=0.0, ret=0.0, spike_days=(), spike_volz=4.0):
    out = []
    for i in range(days):
        d = d0 + dt.timedelta(days=i)
        out.append(dict(symbol=symbol, date=d,
                        ret=ret, volume_z_21d=spike_volz
                        if i in spike_days else volz))
    return out


def test_event_footprint_measures_pre_event_window(test_db):
    d0 = dt.date(2026, 6, 1)
    feats = _features(_mk("GP", d0, 20, spike_days=(10, 11, 12, 13, 14)))
    # event on day 15: prior 5 days are all spiked
    vdb.upsert(test_db, "events", [dict(
        id="ev1", post_date=(d0 + dt.timedelta(days=15)).isoformat(),
        symbol="GP", title="GP: Dividend Declaration", body="",
        source="dse_news")])
    vdb.upsert(test_db, "event_labels", [dict(
        event_id="ev1", event_type="dividend_declared", materiality=3)])
    result = footprint.compute_event_footprints(test_db, feats)
    assert result == {"computed": 1}
    row = test_db.execute(
        "SELECT pre_vol_z FROM event_footprints").fetchone()
    assert abs(row[0] - 4.0) < 1e-9      # mean of the 5 spiked days


def test_incremental_skips_done_events(test_db):
    d0 = dt.date(2026, 6, 1)
    feats = _features(_mk("GP", d0, 20, spike_days=(12,)))
    vdb.upsert(test_db, "events", [dict(
        id="ev1", post_date=(d0 + dt.timedelta(days=15)).isoformat(),
        symbol="GP", title="t", body="", source="dse_news")])
    vdb.upsert(test_db, "event_labels", [dict(
        event_id="ev1", event_type="earnings_release", materiality=3)])
    footprint.compute_event_footprints(test_db, feats)
    assert footprint.compute_event_footprints(test_db, feats) == {"computed": 0}


def test_daily_watch_flags_footprint_lookalikes(test_db):
    d0 = dt.date(2026, 6, 1)
    # historical footprints: three events with pre_vol_z 1.0, 2.0, 3.0
    for i, v in enumerate((1.0, 2.0, 3.0)):
        vdb.upsert(test_db, "event_footprints", [dict(
            event_id=f"e{i}", pre_vol_z=v, pre_ret=0.02)])
    today = d0 + dt.timedelta(days=9)
    feats = _features(
        _mk("HOT", d0, 10, spike_days=(5, 6, 7, 8, 9), spike_volz=5.0,
            ret=0.01)
        + _mk("COLD", d0, 10, volz=0.1))
    flagged = footprint.daily_watch(test_db, feats, today.isoformat())
    assert [f["symbol"] for f in flagged] == ["HOT"]
    f = flagged[0]
    assert f["kind"] == "footprint" and f["score"] >= 5.0


def test_daily_watch_empty_without_history(test_db):
    d0 = dt.date(2026, 6, 1)
    feats = _features(_mk("HOT", d0, 10, spike_days=(9,), spike_volz=9.0))
    assert footprint.daily_watch(
        test_db, feats, (d0 + dt.timedelta(days=9)).isoformat()) == []
