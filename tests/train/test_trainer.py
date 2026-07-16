# tests/train/test_trainer.py
import datetime as dt

import numpy as np
import pytest

from vectora import db as vdb
from vectora.train import trainer

pytestmark = pytest.mark.slow  # ~2-4 min: real LightGBM fits; skip via -m "not slow"


def _seed_realistic(con, n_days=900, n_syms=8, seed=3):
    rng = np.random.default_rng(seed)
    vdb.upsert(con, "symbols", [
        dict(symbol=f"S{i:02d}", name=None, sector="Bank" if i % 2 else "Textile",
             instrument_type="Equity", category="A", listing_status="active",
             first_seen="2019-01-01", last_seen="2026-12-31")
        for i in range(n_syms)])
    rows = []
    d0 = dt.date(2021, 1, 1)
    px = {f"S{i:02d}": 100.0 for i in range(n_syms)}
    for day in range(n_days):
        d = (d0 + dt.timedelta(days=day)).isoformat()
        for sym in px:
            px[sym] *= float(np.exp(rng.normal(0, 0.02)))
            p = round(max(px[sym], 1.0), 2)
            rows.append(dict(symbol=sym, date=d, open=p, high=p * 1.02,
                             low=p * 0.98, close=p, ltp=p, ycp=p,
                             trades=int(rng.integers(20, 200)),
                             value_mn=float(rng.uniform(2, 30)),
                             volume=int(rng.integers(1000, 50000)),
                             source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)


def test_trainer_end_to_end_writes_report_registry_and_artifacts(test_db, tmp_path):
    _seed_realistic(test_db)
    result = trainer.run(
        test_db, target="g5_h10",
        models_dir=tmp_path / "models", reports_dir=tmp_path / "reports",
        features_path=tmp_path / "features.parquet",
        min_train_days=500, test_days=120, embargo_days=10,
        min_median_value_mn=0.1)
    assert result["folds"] >= 2
    assert 0 < result["lgbm_brier"] < 1 and 0 < result["logistic_brier"] < 1
    report = list((tmp_path / "reports").glob("train_g5_h10_*.md"))
    assert len(report) == 1
    text = report[0].read_text(encoding="utf-8")
    assert "Brier" in text and "reliability" in text.lower()
    n = test_db.execute("SELECT count(*) FROM model_registry").fetchone()[0]
    assert n == 2  # one row per family
    art = list((tmp_path / "models").glob("*/meta.json"))
    assert len(art) == 2


def test_trainer_refuses_unknown_target(test_db, tmp_path):
    import pytest
    with pytest.raises(ValueError, match="target"):
        trainer.run(test_db, target="nonsense",
                    models_dir=tmp_path, reports_dir=tmp_path,
                    features_path=tmp_path / "f.parquet")
