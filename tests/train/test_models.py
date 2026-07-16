# tests/train/test_models.py
import numpy as np

from vectora.train import models


def _synthetic(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.3 * rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_logistic_baseline_learns():
    X, y = _synthetic()
    m = models.fit_logistic(X[:3000], y[:3000])
    p = models.predict(m, X[3000:])
    assert models.auc(y[3000:], p) > 0.80


def test_lgbm_learns_and_beats_chance():
    X, y = _synthetic()
    m = models.fit_lgbm(X[:2500], y[:2500], X[2500:3000], y[2500:3000])
    p = models.predict(m, X[3000:])
    assert models.auc(y[3000:], p) > 0.80


def test_isotonic_calibration_improves_or_maintains_brier():
    X, y = _synthetic()
    m = models.fit_lgbm(X[:2000], y[:2000], X[2000:2500], y[2000:2500])
    p_val = models.predict(m, X[2500:3200])
    cal = models.fit_calibrator(p_val, y[2500:3200])
    p_test_raw = models.predict(m, X[3200:])
    p_test_cal = models.apply_calibrator(cal, p_test_raw)
    assert models.brier(y[3200:], p_test_cal) <= models.brier(y[3200:], p_test_raw) + 0.005
    assert (p_test_cal >= 0).all() and (p_test_cal <= 1).all()


def test_reliability_table_shape():
    X, y = _synthetic()
    m = models.fit_logistic(X[:3000], y[:3000])
    p = models.predict(m, X[3000:])
    tab = models.reliability_table(y[3000:], p, bins=10)
    assert len(tab) <= 10
    for row in tab:
        assert set(row) == {"bin_lo", "bin_hi", "n", "p_mean", "y_rate"}
