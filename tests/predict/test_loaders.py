import json

import pytest

from vectora import db as vdb
from vectora.predict import loaders


def test_prediction_tables_exist(test_db):
    tables = {r[0] for r in test_db.execute("SHOW TABLES").fetchall()}
    assert {"predictions", "risk_blocks", "explanations"} <= tables


def _register(con, tmp_path, target="g5_h10", active=True, features=("ret_1d",)):
    art = tmp_path / f"{target}_lgbm_test"
    art.mkdir(parents=True, exist_ok=True)
    (art / "meta.json").write_text(json.dumps(
        {"model_id": f"{target}_lgbm_test", "family": "lgbm",
         "target": target, "features": list(features)}), encoding="utf-8")
    vdb.upsert(con, "model_registry", [{
        "model_id": f"{target}_lgbm_test", "family": "lgbm", "target": target,
        "trained_at": "2026-07-16T00:00:00", "train_end": "2026-01-01",
        "metrics": "{}", "artifact_dir": str(art), "active": active,
    }])
    return art


def test_active_model_returns_meta(test_db, tmp_path):
    _register(test_db, tmp_path)
    m = loaders.active_model(test_db, "g5_h10")
    assert m["model_id"] == "g5_h10_lgbm_test"
    assert m["features"] == ["ret_1d"]
    assert m["artifact_dir"].endswith("g5_h10_lgbm_test")


def test_active_model_none_when_inactive(test_db, tmp_path):
    _register(test_db, tmp_path, active=False)
    assert loaders.active_model(test_db, "g5_h10") is None


def test_active_model_latest_wins(test_db, tmp_path):
    _register(test_db, tmp_path)
    art2 = tmp_path / "newer"
    art2.mkdir()
    (art2 / "meta.json").write_text(json.dumps(
        {"model_id": "newer", "family": "lgbm", "target": "g5_h10",
         "features": ["ret_1d", "ret_3d"]}), encoding="utf-8")
    vdb.upsert(test_db, "model_registry", [{
        "model_id": "newer", "family": "lgbm", "target": "g5_h10",
        "trained_at": "2026-07-17T00:00:00", "train_end": "2026-02-01",
        "metrics": "{}", "artifact_dir": str(art2), "active": True,
    }])
    assert loaders.active_model(test_db, "g5_h10")["model_id"] == "newer"


def test_load_artifacts_missing_dir_raises(test_db, tmp_path):
    _register(test_db, tmp_path)
    m = loaders.active_model(test_db, "g5_h10")
    with pytest.raises(FileNotFoundError):
        loaders.load_artifacts(m)  # no lgbm.txt/calibrator.pkl written
