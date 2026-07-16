import numpy as np
import polars as pl

from vectora.predict import analogs


def _history(n=300, seed=1):
    """Labeled history: outcome correlates with feature f1."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    fwdmax = 0.04 + 0.03 * f1 + rng.normal(0, 0.01, n)   # up-move scales with f1
    fwdmin = -0.03 + 0.01 * f1 - np.abs(rng.normal(0, 0.01, n))
    y = (fwdmax >= 0.05).astype(np.int8)
    return pl.DataFrame({
        "f1": f1, "f2": f2, "fwdmax_h10": fwdmax, "fwdmin_h10": fwdmin,
        "y_g5_h10": y,
    })


def test_analog_stats_reflect_neighbourhood():
    hist = _history()
    idx = analogs.AnalogIndex.fit(hist, feature_names=["f1", "f2"],
                                  label_col="y_g5_h10",
                                  fwdmax_col="fwdmax_h10",
                                  fwdmin_col="fwdmin_h10")
    # a query deep in high-f1 territory should find high hit-rate analogs
    hi = idx.query(np.array([2.5, 0.0]), k=20)
    lo = idx.query(np.array([-2.5, 0.0]), k=20)
    assert hi["hit_rate"] > lo["hit_rate"]
    assert hi["median_up"] > lo["median_up"]
    assert hi["n"] == 20
    assert hi["max_drawdown"] <= hi["median_down"] <= 0.05
    assert set(hi) == {"hit_rate", "median_up", "median_down",
                       "max_drawdown", "n"}


def test_nan_features_are_imputed_not_fatal():
    hist = _history()
    idx = analogs.AnalogIndex.fit(hist, feature_names=["f1", "f2"],
                                  label_col="y_g5_h10",
                                  fwdmax_col="fwdmax_h10",
                                  fwdmin_col="fwdmin_h10")
    out = idx.query(np.array([np.nan, 0.5]), k=10)
    assert out["n"] == 10 and 0.0 <= out["hit_rate"] <= 1.0


def test_fit_drops_rows_without_labels():
    hist = _history().with_columns(
        pl.when(pl.int_range(pl.len()) < 50).then(None)
        .otherwise(pl.col("y_g5_h10")).alias("y_g5_h10"))
    idx = analogs.AnalogIndex.fit(hist, feature_names=["f1", "f2"],
                                  label_col="y_g5_h10",
                                  fwdmax_col="fwdmax_h10",
                                  fwdmin_col="fwdmin_h10")
    assert idx.n_rows == 250


def test_fit_caps_rows_to_most_recent():
    import datetime as dt
    hist = _history(n=300).with_columns(
        pl.date_range(dt.date(2025, 1, 1), dt.date(2025, 10, 27),
                      eager=True).alias("date"))
    idx = analogs.AnalogIndex.fit(hist, feature_names=["f1", "f2"],
                                  label_col="y_g5_h10",
                                  fwdmax_col="fwdmax_h10",
                                  fwdmin_col="fwdmin_h10", max_rows=100)
    assert idx.n_rows == 100
