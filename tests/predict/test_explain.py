import numpy as np

from vectora.predict import explain
from vectora.train import models as M


def _tiny_model(n=800, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0).astype(int)
    m = M.fit_lgbm(X[:600], y[:600], X[600:], y[600:])
    return m.booster_, X


def test_drivers_rank_the_signal_feature_first():
    booster, X = _tiny_model()
    names = ["alpha", "beta", "gamma", "delta"]
    drivers = explain.drivers(booster, X[0], names, top=3)
    assert len(drivers) == 3
    assert drivers[0]["feature"] == "alpha"      # the only real signal
    assert set(drivers[0]) == {"feature", "contribution", "value"}


def test_render_mentions_key_facts():
    d = [{"feature": "volume_z_21d", "contribution": 0.31, "value": 4.2},
         {"feature": "ret_21d", "contribution": -0.12, "value": -0.05}]
    a = {"hit_rate": 0.65, "median_up": 0.081, "median_down": -0.032,
         "max_drawdown": -0.19, "n": 20}
    r = {"exit_days": 4.2, "category": "B", "vol_21d": 0.03,
         "rr_ratio": 2.5, "expected_up": 0.081, "expected_down": -0.032,
         "analog_hit_rate": 0.65, "analog_n": 20,
         "analog_max_drawdown": -0.19, "liquidity_value_mn": 0.6}
    text = explain.render("GP", "g5_h10", 0.62, d, a, r, quality=100)
    assert "62%" in text
    assert "volume_z_21d" in text and "supports" in text
    assert "ret_21d" in text and "works against" in text
    assert "13 of 20" in text            # analog hit count
    assert "worst analog" in text and "-19.0%" in text
    assert "thin book" in text           # exit_days > 3 warning


def test_render_flags_low_quality_and_z_category():
    d, a = [], {"hit_rate": 0.5, "median_up": 0.05, "median_down": -0.05,
                "max_drawdown": -0.1, "n": 10}
    r = {"exit_days": 1.0, "category": "Z", "vol_21d": 0.05,
         "rr_ratio": 1.0, "expected_up": 0.05, "expected_down": -0.05,
         "analog_hit_rate": 0.5, "analog_n": 10,
         "analog_max_drawdown": -0.1, "liquidity_value_mn": 2.0}
    text = explain.render("ZSTOCK", "g5_h10", 0.7, d, a, r, quality=75)
    assert "Z-category" in text
    assert "quality 75" in text
