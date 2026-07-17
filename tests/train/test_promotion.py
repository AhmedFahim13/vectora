# tests/train/test_promotion.py
import json

import numpy as np

from vectora import db as vdb
from vectora.train import trainer


def test_promote_flags_and_demotes(test_db):
    for mid, active, brier in (("old", True, 0.210), ("new", False, 0.205)):
        vdb.upsert(test_db, "model_registry", [{
            "model_id": mid, "family": "lgbm", "target": "g5_h10",
            "trained_at": "2026-07-01T00:00:00", "train_end": "2026-06-01",
            "metrics": json.dumps({"brier": brier}),
            "artifact_dir": f"models/{mid}", "active": active}])
    assert trainer.promote_if_better(
        test_db, target="g5_h10", model_id="new", new_brier=0.205) is True
    rows = dict(test_db.execute(
        "SELECT model_id, active FROM model_registry").fetchall())
    assert rows == {"old": False, "new": True}


def test_worse_challenger_stays_inactive(test_db):
    for mid, active, brier in (("old", True, 0.200), ("new", False, 0.215)):
        vdb.upsert(test_db, "model_registry", [{
            "model_id": mid, "family": "lgbm", "target": "g5_h10",
            "trained_at": "2026-07-01T00:00:00", "train_end": "2026-06-01",
            "metrics": json.dumps({"brier": brier}),
            "artifact_dir": f"models/{mid}", "active": active}])
    assert trainer.promote_if_better(
        test_db, target="g5_h10", model_id="new", new_brier=0.215) is False
    rows = dict(test_db.execute(
        "SELECT model_id, active FROM model_registry").fetchall())
    assert rows == {"old": True, "new": False}


def test_no_incumbent_auto_promotes(test_db):
    vdb.upsert(test_db, "model_registry", [{
        "model_id": "new", "family": "lgbm", "target": "g5_h10",
        "trained_at": "2026-07-01T00:00:00", "train_end": "2026-06-01",
        "metrics": json.dumps({"brier": 0.30}), "artifact_dir": "models/new",
        "active": False}])
    assert trainer.promote_if_better(
        test_db, target="g5_h10", model_id="new", new_brier=0.30) is True


def test_pooled_calibrator_is_monotone():
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 1, 3000)
    y = (rng.uniform(size=3000) < raw ** 1.5).astype(int)  # miscalibrated
    cal = trainer.fit_deploy_calibrator(raw, y)
    grid = cal.predict(np.linspace(0, 1, 50))
    assert (np.diff(grid) >= -1e-9).all()
    assert 0 <= grid.min() and grid.max() <= 1
