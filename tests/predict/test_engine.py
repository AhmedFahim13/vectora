import datetime as dt
import json
import pickle

import numpy as np

from vectora import db as vdb
from vectora.predict import engine as pengine
from vectora.train import models as M


def _seed_market(con, n_days=140, n_syms=6, seed=11):
    """Enough history for 21/63d features and analog labels."""
    rng = np.random.default_rng(seed)
    vdb.upsert(con, "symbols", [
        dict(symbol=f"S{i}", name=None, sector="Bank",
             instrument_type="Equity", category="Z" if i == 5 else "A",
             listing_status="active", first_seen="2020-01-01",
             last_seen="2026-12-31")
        for i in range(n_syms)])
    rows = []
    d0 = dt.date(2026, 1, 1)
    px = {f"S{i}": 50.0 * (i + 1) for i in range(n_syms)}
    for day in range(n_days):
        d = (d0 + dt.timedelta(days=day)).isoformat()
        for sym in px:
            px[sym] *= float(np.exp(rng.normal(0.0002, 0.015)))
            p = round(max(px[sym], 1.0), 2)
            rows.append(dict(symbol=sym, date=d, open=p, high=p * 1.01,
                             low=p * 0.99, close=p, ltp=p, ycp=p,
                             trades=50, value_mn=float(rng.uniform(2, 10)),
                             volume=int(rng.integers(5000, 20000)),
                             source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)
    last = (d0 + dt.timedelta(days=n_days - 1)).isoformat()
    vdb.upsert(con, "data_quality",
               [{"date": last, "source": "dse_eod", "score": 100,
                 "issues": "[]"}])
    return last


def _train_and_register(con, tmp_path):
    """Tiny real model over the seeded market, saved like trainer does."""
    from vectora import labels as lab
    from vectora.features import engine as fengine
    from vectora.features import registry
    feat_names = [s.name for s in registry.load()]
    df = fengine.compute(con, out_path=tmp_path / "f.parquet")
    df = lab.make_labels(df, thresholds=(0.05,), horizons=(10,),
                         continuous=True)
    train = df.filter(df["y_g5_h10"].is_not_null())
    X = train.select(feat_names).to_numpy().astype(np.float64)
    y = train["y_g5_h10"].to_numpy().astype(int)
    cut = int(len(y) * 0.8)
    m = M.fit_lgbm(X[:cut], y[:cut], X[cut:], y[cut:])
    cal = M.fit_calibrator(M.predict(m, X[cut:]), y[cut:])
    art = tmp_path / "g5_h10_lgbm_test"
    art.mkdir()
    m.booster_.save_model(str(art / "lgbm.txt"))
    (art / "calibrator.pkl").write_bytes(pickle.dumps(cal))
    (art / "meta.json").write_text(json.dumps(
        {"model_id": "g5_h10_lgbm_test", "family": "lgbm",
         "target": "g5_h10", "features": feat_names}), encoding="utf-8")
    vdb.upsert(con, "model_registry", [{
        "model_id": "g5_h10_lgbm_test", "family": "lgbm", "target": "g5_h10",
        "trained_at": "2026-07-16T00:00:00", "train_end": "2026-05-01",
        "metrics": "{}", "artifact_dir": str(art), "active": True,
    }])


def test_run_predict_persists_all_three_tables(test_db, tmp_path):
    last = _seed_market(test_db)
    _train_and_register(test_db, tmp_path)
    result = pengine.run_predict(test_db, date_str=last,
                                 features_path=tmp_path / "f2.parquet",
                                 min_median_value_mn=0.1)
    assert result["predictions"] == 6            # all six symbols scored
    n_pred = test_db.execute("SELECT count(*) FROM predictions").fetchone()[0]
    n_risk = test_db.execute("SELECT count(*) FROM risk_blocks").fetchone()[0]
    n_expl = test_db.execute("SELECT count(*) FROM explanations").fetchone()[0]
    assert n_pred == n_risk == n_expl == 6
    row = test_db.execute(
        "SELECT probability, quality_score FROM predictions LIMIT 1").fetchone()
    assert 0.0 <= row[0] <= 1.0 and row[1] == 100


def test_z_category_never_signals(test_db, tmp_path):
    last = _seed_market(test_db)
    _train_and_register(test_db, tmp_path)
    pengine.run_predict(test_db, date_str=last,
                        features_path=tmp_path / "f2.parquet",
                        min_median_value_mn=0.1)
    z = test_db.execute(
        "SELECT is_signal, suppressed_reason FROM predictions "
        "WHERE symbol = 'S5'").fetchone()
    assert z[0] is False and z[1] == "z-category-gate"


def test_low_quality_day_suppresses_signals(test_db, tmp_path):
    last = _seed_market(test_db)
    vdb.upsert(test_db, "data_quality",
               [{"date": last, "source": "dse_eod", "score": 60,
                 "issues": "[]"}])
    _train_and_register(test_db, tmp_path)
    pengine.run_predict(test_db, date_str=last,
                        features_path=tmp_path / "f2.parquet",
                        min_median_value_mn=0.1)
    n_signals = test_db.execute(
        "SELECT count(*) FROM predictions WHERE is_signal").fetchone()[0]
    assert n_signals == 0
    reasons = {r[0] for r in test_db.execute(
        "SELECT DISTINCT suppressed_reason FROM predictions "
        "WHERE NOT is_signal").fetchall()}
    assert "quality-below-floor" in reasons


def test_rerun_is_idempotent(test_db, tmp_path):
    last = _seed_market(test_db)
    _train_and_register(test_db, tmp_path)
    pengine.run_predict(test_db, date_str=last,
                        features_path=tmp_path / "f2.parquet",
                        min_median_value_mn=0.1)
    pengine.run_predict(test_db, date_str=last,
                        features_path=tmp_path / "f3.parquet",
                        min_median_value_mn=0.1)
    assert test_db.execute("SELECT count(*) FROM predictions").fetchone()[0] == 6


def test_no_active_model_returns_zero(test_db, tmp_path):
    last = _seed_market(test_db)
    result = pengine.run_predict(test_db, date_str=last,
                                 features_path=tmp_path / "f.parquet",
                                 min_median_value_mn=0.1)
    assert result["predictions"] == 0 and result["targets"] == []
