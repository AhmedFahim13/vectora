# vectora/train/models.py
"""Model zoo for Phase 2: regularized logistic (the mandatory baseline —
spec §10.1: if GBMs can't beat it out-of-sample, the features carry no
signal) and LightGBM, plus isotonic calibration and the metrics that decide
promotion (Brier is primary; it prices calibration, not just ranking)."""
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LGBM_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, verbosity=-1, seed=42,
)


def fit_logistic(X, y):
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ])
    return pipe.fit(X, y)


def fit_lgbm(X, y, X_val, y_val):
    m = LGBMClassifier(**LGBM_PARAMS)
    m.fit(X, y, eval_set=[(X_val, y_val)],
          callbacks=[early_stopping(50, verbose=False)])
    return m


def predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def fit_calibrator(p_val, y_val) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0
                              ).fit(p_val, y_val)


def apply_calibrator(cal, p) -> np.ndarray:
    return cal.predict(p)


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def reliability_table(y, p, bins: int = 10) -> list[dict]:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.sum() == 0:
            continue
        out.append({"bin_lo": float(lo), "bin_hi": float(hi),
                    "n": int(mask.sum()), "p_mean": float(p[mask].mean()),
                    "y_rate": float(y[mask].mean())})
    return out
