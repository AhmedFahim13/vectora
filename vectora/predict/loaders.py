"""Load active models from the registry with their on-disk artifacts."""
import json
import pickle
from pathlib import Path

import lightgbm as lgb


def active_model(con, target: str) -> dict | None:
    row = con.execute(
        """
        SELECT model_id, artifact_dir FROM model_registry
        WHERE target = ? AND family = 'lgbm' AND active
        ORDER BY trained_at DESC LIMIT 1
        """, [target]).fetchone()
    if row is None:
        return None
    meta = json.loads(
        (Path(row[1]) / "meta.json").read_text(encoding="utf-8"))
    return {"model_id": row[0], "artifact_dir": row[1],
            "target": target, "features": meta["features"]}


def load_artifacts(model: dict):
    """Returns (booster, calibrator)."""
    art = Path(model["artifact_dir"])
    booster_path, cal_path = art / "lgbm.txt", art / "calibrator.pkl"
    if not booster_path.exists() or not cal_path.exists():
        raise FileNotFoundError(f"artifacts missing in {art}")
    booster = lgb.Booster(model_file=str(booster_path))
    calibrator = pickle.loads(cal_path.read_bytes())
    return booster, calibrator
