"""State mirroring: a discarded database must not lose which model is live.

On 2026-08-23 a rebase resolved data/vectora.duckdb by keeping one side.
The model artifacts were committed; the registry rows that activated them
were not. The system went back to serving a model trained through
2024-11-21 and nothing errored. These tests pin the recovery.
"""
import json

from vectora import db as vdb
from vectora import state


def _register(con, model_id, train_end, active=False):
    vdb.upsert(con, "model_registry", [{
        "model_id": model_id, "family": "lgbm", "target": "g5_h10",
        "trained_at": "2026-08-23", "train_end": train_end,
        "metrics": "{}", "artifact_dir": f"models/{model_id}",
        "active": active}])


def test_export_then_restore_recovers_lost_rows(test_db, tmp_path):
    _register(test_db, "new", "2026-08-09", active=True)
    vdb.set_watermark(test_db, "collect", "eod", "2026-08-23")
    state.export_state(test_db, tmp_path)

    # simulate the rebase: the whole registry vanishes with the binary
    test_db.execute("DELETE FROM model_registry")
    test_db.execute("DELETE FROM watermarks")
    out = state.restore_state(test_db, tmp_path)

    assert out["model_registry"] == 1
    row = test_db.execute(
        "SELECT model_id, CAST(train_end AS VARCHAR), active "
        "FROM model_registry").fetchone()
    assert row == ("new", "2026-08-09", True)
    assert vdb.get_watermark(test_db, "collect", "eod") == "2026-08-23"


def test_restore_reactivates_the_right_model(test_db, tmp_path):
    """The exact 2026-08-23 failure: the old model is still present and
    still flagged active, and the promoted one is gone."""
    _register(test_db, "old", "2024-11-21", active=False)
    _register(test_db, "new", "2026-08-09", active=True)
    state.export_state(test_db, tmp_path)

    test_db.execute("DELETE FROM model_registry WHERE model_id = 'new'")
    test_db.execute("UPDATE model_registry SET active = true "
                    "WHERE model_id = 'old'")
    state.restore_state(test_db, tmp_path)

    active = test_db.execute(
        "SELECT model_id FROM model_registry WHERE active").fetchall()
    assert active == [("new",)], "the reverted model is still being served"


def test_restore_never_rolls_back_newer_state(test_db, tmp_path):
    """A mirror from this morning must not undo work from this afternoon."""
    _register(test_db, "old", "2024-11-21", active=True)
    state.export_state(test_db, tmp_path)
    _register(test_db, "newer", "2026-08-20", active=False)
    state.restore_state(test_db, tmp_path)
    ids = {r[0] for r in test_db.execute(
        "SELECT model_id FROM model_registry").fetchall()}
    assert ids == {"old", "newer"}          # nothing deleted


def test_only_one_model_stays_active(test_db, tmp_path):
    _register(test_db, "a", "2026-08-01", active=True)
    state.export_state(test_db, tmp_path)
    _register(test_db, "b", "2026-08-02", active=True)   # double activation
    state.restore_state(test_db, tmp_path)
    active = test_db.execute(
        "SELECT model_id FROM model_registry WHERE active").fetchall()
    assert active == [("a",)]


def test_divergence_reports_missing_rows(test_db, tmp_path):
    _register(test_db, "new", "2026-08-09", active=True)
    state.export_state(test_db, tmp_path)
    assert state.divergence(test_db, tmp_path) == []
    test_db.execute("DELETE FROM model_registry")
    problems = state.divergence(test_db, tmp_path)
    assert problems and "model_registry" in problems[0]


def test_export_is_byte_stable_between_runs(test_db, tmp_path):
    """An unsorted dump would churn the diff on every commit and bury the
    one change that mattered."""
    _register(test_db, "b", "2026-08-02")
    _register(test_db, "a", "2026-08-01")
    state.export_state(test_db, tmp_path)
    first = (tmp_path / "model_registry.json").read_text(encoding="utf-8")
    state.export_state(test_db, tmp_path)
    assert (tmp_path / "model_registry.json").read_text(
        encoding="utf-8") == first
    ids = [r["model_id"] for r in json.loads(first)]
    assert ids == sorted(ids)


def test_restore_with_no_mirror_is_a_noop(test_db, tmp_path):
    assert state.restore_state(test_db, tmp_path) == {"reactivated": 0}
