"""Database maintenance: shrink the committed binary, lose nothing.

The schema's rule is that rows are never deleted. These tests pin the two
ways that rule is honoured: regenerable rows may go because prices can
rebuild them exactly, and unregenerable rows must land in Parquet before
they leave the database.
"""
import datetime as dt

import polars as pl

from vectora import db as vdb
from vectora import maintenance


def _dates(con, table, n=10):
    d0 = dt.date(2026, 1, 1)
    vdb.upsert(con, table, [
        {"date": (d0 + dt.timedelta(days=i)).isoformat(), "symbol": f"S{j}",
         "score": 1, "band": "Buy", "votes": "[]", "rsi": 50.0,
         "macd_hist": 0.0, "bb_pos": 0.5, "st_dir": 1}
        for i in range(n) for j in range(3)])


def test_regenerable_tables_keep_only_recent_dates(test_db):
    _dates(test_db, "ta_ratings", n=10)
    removed = maintenance.prune_regenerable(test_db, {"ta_ratings": 3})
    kept = test_db.execute(
        "SELECT count(DISTINCT date) FROM ta_ratings").fetchone()[0]
    assert kept == 3
    assert removed["ta_ratings"] == 21          # 7 dates x 3 symbols


def test_pruning_keeps_the_newest_not_the_oldest(test_db):
    _dates(test_db, "ta_ratings", n=10)
    maintenance.prune_regenerable(test_db, {"ta_ratings": 2})
    newest = test_db.execute("SELECT max(date) FROM ta_ratings").fetchone()[0]
    assert newest == dt.date(2026, 1, 10)


def test_intraday_is_archived_before_it_is_pruned(test_db, tmp_path):
    old = dt.datetime.now() - dt.timedelta(days=90)
    recent = dt.datetime.now() - dt.timedelta(days=2)
    vdb.upsert(test_db, "intraday_snapshots", [
        {"symbol": "GP", "ts": ts.isoformat(), "ltp": 100.0, "high": 101.0,
         "low": 99.0, "closep": 100.0, "ycp": 100.0, "change": 0.0,
         "trades": 5, "value_mn": 1.0, "volume": 10}
        for ts in (old, recent)])
    out = maintenance.archive_and_prune(test_db, root=tmp_path,
                                        intraday_days=30)
    assert out["intraday_snapshots"] == 1
    # the old row left the database ...
    assert test_db.execute(
        "SELECT count(*) FROM intraday_snapshots").fetchone()[0] == 1
    # ... and is still readable on disk
    files = list((tmp_path / "intraday_snapshots").glob("*.parquet"))
    assert len(files) == 1
    assert pl.read_parquet(files[0]).height == 1


def test_archive_merges_rather_than_overwrites_a_month(test_db, tmp_path):
    """A late row must not erase what the month already held."""
    base = dt.datetime.now() - dt.timedelta(days=90)
    vdb.upsert(test_db, "intraday_snapshots", [{
        "symbol": "GP", "ts": base.isoformat(), "ltp": 100.0, "high": 1.0,
        "low": 1.0, "closep": 1.0, "ycp": 1.0, "change": 0.0, "trades": 1,
        "value_mn": 1.0, "volume": 1}])
    maintenance.archive_and_prune(test_db, root=tmp_path, intraday_days=30)
    vdb.upsert(test_db, "intraday_snapshots", [{
        "symbol": "BEXIMCO", "ts": (base + dt.timedelta(hours=1)).isoformat(),
        "ltp": 50.0, "high": 1.0, "low": 1.0, "closep": 1.0, "ycp": 1.0,
        "change": 0.0, "trades": 1, "value_mn": 1.0, "volume": 1}])
    maintenance.archive_and_prune(test_db, root=tmp_path, intraday_days=30)
    files = list((tmp_path / "intraday_snapshots").glob("*.parquet"))
    syms = set(pl.read_parquet(files[0])["symbol"].to_list())
    assert syms == {"GP", "BEXIMCO"}, syms


def test_open_predictions_keep_their_explanation(test_db, tmp_path):
    """An unresolved prediction is still on the dashboard; do not archive it."""
    old = (dt.date.today() - dt.timedelta(days=200)).isoformat()
    vdb.upsert(test_db, "predictions", [{
        "id": "p1", "symbol": "GP", "date": old, "target": "g5_h10",
        "probability": 0.4, "model_id": "m", "quality_score": 100,
        "is_signal": True, "suppressed_reason": None}])
    vdb.upsert(test_db, "explanations", [{
        "prediction_id": "p1", "drivers": "[]", "analogs": "{}",
        "rendered": "why"}])
    out = maintenance.archive_and_prune(test_db, root=tmp_path,
                                        explanation_days=45)
    assert out["explanations"] == 0
    assert test_db.execute(
        "SELECT count(*) FROM explanations").fetchone()[0] == 1


def test_nothing_to_do_is_safe(test_db, tmp_path):
    out = maintenance.archive_and_prune(test_db, root=tmp_path)
    assert out == {"intraday_snapshots": 0, "explanations": 0}


def test_signal_explanations_are_never_archived(test_db, tmp_path):
    """Signals are the track record. Everything else is the growth problem."""
    old = (dt.date.today() - dt.timedelta(days=200)).isoformat()
    vdb.upsert(test_db, "predictions", [
        {"id": "sig", "symbol": "GP", "date": old, "target": "g5_h10",
         "probability": 0.6, "model_id": "m", "quality_score": 100,
         "is_signal": True, "suppressed_reason": None},
        {"id": "noise", "symbol": "XY", "date": old, "target": "g5_h10",
         "probability": 0.3, "model_id": "m", "quality_score": 100,
         "is_signal": False, "suppressed_reason": "below bar"}])
    vdb.upsert(test_db, "outcomes", [
        {"prediction_id": "sig", "realized_max": 0.1, "realized_min": 0.0,
         "hit": True},
        {"prediction_id": "noise", "realized_max": 0.0, "realized_min": 0.0,
         "hit": False}])
    vdb.upsert(test_db, "explanations", [
        {"prediction_id": "sig", "drivers": "[]", "analogs": "{}",
         "rendered": "kept"},
        {"prediction_id": "noise", "drivers": "[]", "analogs": "{}",
         "rendered": "archived"}])
    out = maintenance.archive_and_prune(test_db, root=tmp_path,
                                        explanation_days=15)
    assert out["explanations"] == 1
    left = [r[0] for r in test_db.execute(
        "SELECT prediction_id FROM explanations").fetchall()]
    assert left == ["sig"]


def test_alerts_survive_a_run_that_does_not_commit_the_database(test_db,
                                                                tmp_path):
    """The intraday dedup check must still see this morning's alert.

    Intraday no longer commits the database, so each run starts from the EOD
    checkout. Without the archive round-trip the cooldown would see nothing
    and re-send the same symbol four times a day.
    """
    today = dt.date.today().isoformat()
    vdb.upsert(test_db, "alerts_log", [{
        "id": f"{today}_intraday_GP", "alert_type": "intraday",
        "symbol": "GP", "alert_date": today, "prediction_id": None}])
    assert maintenance.archive_alerts(test_db, root=tmp_path) == 1

    # a fresh checkout: same schema, no alerts_log rows
    fresh_path = tmp_path / "fresh.duckdb"
    fresh = vdb.connect(fresh_path)
    vdb.init_schema(fresh, backfill_parquet=tmp_path / "none.parquet")
    assert fresh.execute("SELECT count(*) FROM alerts_log").fetchone()[0] == 0

    restored = maintenance.restore_alerts(fresh, today, root=tmp_path)
    assert restored == 1
    seen = fresh.execute(
        "SELECT symbol FROM alerts_log WHERE alert_type = 'intraday' "
        "AND alert_date = ?", [today]).fetchall()
    assert seen == [("GP",)]
    fresh.close()


def test_restore_is_a_noop_with_no_archive(test_db, tmp_path):
    assert maintenance.restore_alerts(test_db, "2026-08-20", root=tmp_path) == 0


def test_compaction_is_skipped_when_it_would_not_help(tmp_path):
    """Rebuilding an already-compact file just churns bytes in git."""
    path = tmp_path / "small.duckdb"
    con = vdb.connect(path)
    vdb.init_schema(con, backfill_parquet=tmp_path / "none.parquet")
    con.close()
    res = maintenance.compact(path, min_saving_mb=2.0)
    assert res["saved_mb"] == 0.0
    assert res["skipped"] is not None
