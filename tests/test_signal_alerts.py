from vectora import db as vdb
from vectora.alerts import signals


def _signal(symbol, d):
    return dict(id=f"{d}_g5_h10_{symbol}", symbol=symbol, date=d,
                target="g5_h10", probability=0.62, model_id="m",
                quality_score=100, is_signal=True, suppressed_reason=None)


def test_new_signals_are_logged(test_db):
    vdb.upsert(test_db, "predictions",
               [_signal("GP", "2026-07-16"), _signal("ACI", "2026-07-16")])
    new = signals.log_signal_alerts(test_db, "2026-07-16")
    assert sorted(new) == ["ACI", "GP"]
    n = test_db.execute("SELECT count(*) FROM alerts_log").fetchone()[0]
    assert n == 2


def test_cooldown_suppresses_repeat_within_2_days(test_db):
    vdb.upsert(test_db, "predictions", [_signal("GP", "2026-07-16")])
    signals.log_signal_alerts(test_db, "2026-07-16")
    vdb.upsert(test_db, "predictions", [_signal("GP", "2026-07-17")])
    new = signals.log_signal_alerts(test_db, "2026-07-17")
    assert new == []   # same symbol within cooldown -> suppressed
    n = test_db.execute("SELECT count(*) FROM alerts_log").fetchone()[0]
    assert n == 1


def test_signal_after_cooldown_logs_again(test_db):
    vdb.upsert(test_db, "predictions", [_signal("GP", "2026-07-10")])
    signals.log_signal_alerts(test_db, "2026-07-10")
    vdb.upsert(test_db, "predictions", [_signal("GP", "2026-07-16")])
    new = signals.log_signal_alerts(test_db, "2026-07-16")
    assert new == ["GP"]


def test_rerun_same_day_is_idempotent(test_db):
    vdb.upsert(test_db, "predictions", [_signal("GP", "2026-07-16")])
    signals.log_signal_alerts(test_db, "2026-07-16")
    new = signals.log_signal_alerts(test_db, "2026-07-16")
    assert new == []
    assert test_db.execute("SELECT count(*) FROM alerts_log").fetchone()[0] == 1
