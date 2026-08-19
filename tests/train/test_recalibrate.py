"""Live recalibration: a correction layer fitted on realized outcomes.

The model is well calibrated above 0.4 and badly overconfident below it
(live, 2026-08-20: the 0.1-0.2 bin predicted 15.4% and realized 3.0%).
This layer corrects that. It must never be installed on faith — the guard
is leave-one-COHORT-out, because rows sharing a prediction date share a
market and leave-one-row-out would rubber-stamp anything.
"""
import numpy as np

from vectora.train import recalibrate as rc


def _overconfident(n_cohorts=8, per=200, seed=0):
    """Model says p; truth is roughly p/2 below 0.4, p above it."""
    rng = np.random.default_rng(seed)
    p, y, c = [], [], []
    for k in range(n_cohorts):
        pk = rng.uniform(0.05, 0.6, per)
        truth = np.where(pk < 0.4, pk * 0.4, pk * 0.95)
        p.append(pk)
        y.append((rng.uniform(size=per) < truth).astype(int))
        c.append(np.full(per, f"d{k}"))
    return np.concatenate(p), np.concatenate(y), np.concatenate(c)


def test_correction_pulls_down_overconfident_region():
    p, y, _ = _overconfident()
    g = rc.fit_correction(p, y)
    lows = np.array([0.10, 0.15, 0.20, 0.30])
    corrected = rc.apply_correction(g, lows)
    assert np.all(corrected < lows), corrected
    # and it must not invent confidence it never had
    assert np.all(corrected >= 0.0) and np.all(corrected <= 1.0)


def test_correction_preserves_ranking():
    """Isotonic is monotone, so the ordering — and the AUC — must survive."""
    p, y, _ = _overconfident()
    g = rc.fit_correction(p, y)
    probe = np.linspace(0.05, 0.6, 50)
    out = rc.apply_correction(g, probe)
    assert np.all(np.diff(out) >= -1e-12), "correction reordered the book"


def test_cv_holds_out_whole_cohorts():
    p, y, c = _overconfident()
    seen = []
    for tr, te in rc._cohort_folds(c):
        seen.append(set(c[te]))
        assert not (set(c[tr]) & set(c[te])), "cohort leaked across the split"
    assert len(seen) == 8


def test_cv_reports_improvement_on_genuinely_miscalibrated_data():
    p, y, c = _overconfident()
    res = rc.cross_validate(p, y, c)
    assert res["brier_corrected"] < res["brier_base"]
    assert res["cohorts"] == 8


def test_guard_refuses_when_already_calibrated():
    """If the model is honest, the correction must not be installed."""
    rng = np.random.default_rng(3)
    p, y, c = [], [], []
    for k in range(6):
        pk = rng.uniform(0.05, 0.9, 300)
        p.append(pk)
        y.append((rng.uniform(size=300) < pk).astype(int))
        c.append(np.full(300, f"d{k}"))
    p, y, c = np.concatenate(p), np.concatenate(y), np.concatenate(c)
    res = rc.cross_validate(p, y, c)
    assert res["improved"] is False


def test_single_cohort_cannot_be_validated():
    """One date is one observation; there is nothing to hold out."""
    p = np.linspace(0.1, 0.6, 100)
    y = (p > 0.45).astype(int)
    c = np.full(100, "d1")
    res = rc.cross_validate(p, y, c)
    assert res["improved"] is False
    assert res["cohorts"] == 1


def test_guard_refuses_a_correction_that_wins_often_but_loses_big():
    """The real 2026-08-20 shape: better on 7 of 9 cohorts, but the two it
    misses it misses by several times the average gain. Pooled Brier calls
    that an improvement; a paired test across cohorts does not."""
    rng = np.random.default_rng(7)
    p, y, c = [], [], []
    for k in range(9):
        pk = rng.uniform(0.05, 0.6, 330)
        # eight ordinary days the correction helps on, one anomalous day
        # where the market delivers far more than the model expected
        truth = pk * (2.0 if k == 1 else 0.75)
        p.append(pk)
        y.append((rng.uniform(size=330) < np.clip(truth, 0, 1)).astype(int))
        c.append(np.full(330, f"d{k}"))
    p, y, c = np.concatenate(p), np.concatenate(y), np.concatenate(c)
    res = rc.cross_validate(p, y, c)
    assert res["improved"] is False
    assert res["t_stat"] > res["t_critical"]


def test_verdict_is_a_paired_test_not_pooled_brier():
    """Pinning the distinction explicitly: the fields must disagree when
    pooled Brier improves but the cohort spread says it is noise."""
    p, y, c = _overconfident(n_cohorts=8, per=200)
    res = rc.cross_validate(p, y, c)
    # genuinely miscalibrated data: consistent gains, so this DOES pass
    assert res["improved"] is True
    assert res["t_stat"] <= res["t_critical"]
    assert res["delta_mean"] < 0


def test_run_refuses_and_logs_on_a_single_cohort(test_db, tmp_path):
    from vectora import db as vdb
    from vectora.train import recalibrate
    vdb.upsert(test_db, "predictions", [
        {"id": f"p{i}", "symbol": f"S{i}", "date": "2026-07-16",
         "target": "g5_h10", "probability": 0.4, "model_id": "m",
         "quality_score": 100, "is_signal": False, "suppressed_reason": None}
        for i in range(300)])
    vdb.upsert(test_db, "outcomes", [
        {"prediction_id": f"p{i}", "realized_max": 0.0, "realized_min": 0.0,
         "hit": i % 3 == 0} for i in range(300)])
    res = recalibrate.run(test_db, "g5_h10", models_dir=tmp_path)
    assert res["improved"] is False
    assert "too few" in res["verdict"]
    assert not recalibrate.correction_path("g5_h10", tmp_path).exists()
    logged = test_db.execute(
        "SELECT installed, cohorts, verdict FROM calibration_log").fetchall()
    assert logged and logged[0][0] is False and logged[0][1] == 1


def test_load_correction_absent_returns_none(tmp_path):
    from vectora.train import recalibrate
    assert recalibrate.load_correction("g5_h10", tmp_path) is None
