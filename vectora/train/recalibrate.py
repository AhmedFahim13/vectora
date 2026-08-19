"""Live recalibration: a correction layer fitted on realized outcomes.

The deployment calibrator is fitted on pooled out-of-sample BACKTEST
predictions. That is the best available estimate before anything has
traded, but it is not the deployment distribution, and the 2026-08-20
review showed where the two part company: above 0.4 the model is well
calibrated, below it the model is badly overconfident — the 0.1-0.2 bin
predicted 15.4% and realized 3.0%.

This module fits a second isotonic map g on LIVE (probability, outcome)
pairs and composes it after the existing calibrator:

    p_final = g(calibrator(raw))

Composing rather than refitting is deliberate. It keeps the backtest
calibrator — which carries thirteen years and every market regime — and
corrects only the deployment bias on top. Both maps are monotone, so the
composition cannot reorder the book: this fixes CALIBRATION, and does
nothing for the separate compression problem in the ranking.

Nothing is installed on faith. The guard is leave-one-COHORT-out, and the
verdict is a paired t-test across those cohorts rather than a comparison of
pooled Brier. Both choices exist for the same reason: rows sharing a
prediction date share a market. Holding out rows would let a correction
memorise its own cohort, and pooling rows would let a correction that wins
small and often but loses large and rarely look like an improvement.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from vectora import db as vdb
from vectora.settings import MODELS_DIR
from vectora.train.models import brier as brier_score

MIN_COHORTS = 3
MIN_ROWS = 200
ALPHA = 0.05          # one-sided: the correction must be better, not different


def fit_correction(p, y) -> IsotonicRegression:
    """Isotonic map from stated probability to realized frequency."""
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0
                              ).fit(np.asarray(p, dtype=float),
                                    np.asarray(y, dtype=float))


def apply_correction(g, p) -> np.ndarray:
    return np.clip(g.predict(np.asarray(p, dtype=float)), 0.0, 1.0)


def _cohort_folds(cohorts):
    """Leave-one-cohort-out index splits."""
    cohorts = np.asarray(cohorts)
    out = []
    for c in sorted(set(cohorts.tolist())):
        te = np.flatnonzero(cohorts == c)
        tr = np.flatnonzero(cohorts != c)
        out.append((tr, te))
    return out


def cross_validate(p, y, cohorts) -> dict:
    """Does the correction beat doing nothing, on cohorts it never saw?"""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)
    cohorts = np.asarray(cohorts)
    k = len(set(cohorts.tolist()))
    res = {"cohorts": k, "n": int(len(y)), "improved": False,
           "brier_base": None, "brier_corrected": None, "per_cohort": []}
    if k < MIN_COHORTS or len(y) < MIN_ROWS:
        # one date is one observation: there is nothing to hold out, and a
        # correction validated on itself is not validated at all
        return res

    base_all, corr_all, y_all = [], [], []
    for tr, te in _cohort_folds(cohorts):
        if len(np.unique(y[tr])) < 2:
            continue
        g = fit_correction(p[tr], y[tr])
        pc = apply_correction(g, p[te])
        res["per_cohort"].append({
            "cohort": str(cohorts[te][0]), "n": int(len(te)),
            "brier_base": brier_score(y[te], p[te]),
            "brier_corrected": brier_score(y[te], pc),
        })
        base_all.append(p[te])
        corr_all.append(pc)
        y_all.append(y[te])
    if not y_all:
        return res
    y_all = np.concatenate(y_all)
    res["brier_base"] = brier_score(y_all, np.concatenate(base_all))
    res["brier_corrected"] = brier_score(y_all, np.concatenate(corr_all))
    res["cohorts_improved"] = sum(
        1 for c in res["per_cohort"]
        if c["brier_corrected"] < c["brier_base"])

    # The verdict is a PAIRED test across cohorts, not a comparison of
    # pooled Brier. Pooling rows is what makes a handful of trading days
    # look like thousands of observations, and it would wave through a
    # correction that wins often but loses catastrophically when it loses.
    deltas = np.array([c["brier_corrected"] - c["brier_base"]
                       for c in res["per_cohort"]])
    res["delta_mean"] = float(deltas.mean())
    res["delta_worst"] = float(deltas.max())
    if len(deltas) >= 2 and deltas.std(ddof=1) > 0:
        se = deltas.std(ddof=1) / np.sqrt(len(deltas))
        t_stat = float(deltas.mean() / se)
        crit = float(stats.t.ppf(ALPHA, len(deltas) - 1))   # negative
        res["delta_se"] = float(se)
        res["t_stat"] = t_stat
        res["t_critical"] = crit
        res["ci95"] = (float(deltas.mean() - 1.96 * se),
                       float(deltas.mean() + 1.96 * se))
        res["improved"] = bool(t_stat <= crit)
    else:
        res["improved"] = False
    return res


def correction_path(target: str, models_dir=None) -> Path:
    base = Path(models_dir or MODELS_DIR) / "calibration"
    return base / f"live_correction_{target}.pkl"


def load_live(con, target: str) -> tuple:
    """(probability, hit, cohort_date, regime) for every resolved prediction."""
    rows = con.execute(
        """
        SELECT p.probability, CASE WHEN o.hit THEN 1 ELSE 0 END,
               CAST(p.date AS VARCHAR), coalesce(g.regime, 'unclassified')
        FROM outcomes o JOIN predictions p ON p.id = o.prediction_id
        LEFT JOIN regimes g ON g.date = p.date
        WHERE p.target = ?
        """, [target]).fetchall()
    if not rows:
        return (np.array([]), np.array([]), np.array([]), [])
    return (np.array([r[0] for r in rows], dtype=float),
            np.array([r[1] for r in rows], dtype=int),
            np.array([r[2] for r in rows]),
            sorted({r[3] for r in rows}))


def run(con, target: str = "g5_h10", models_dir=None) -> dict:
    """Fit, validate and install-or-refuse a live correction for one target."""
    p, y, cohorts, regimes = load_live(con, target)
    if len(y) == 0:
        return {"target": target, "installed": False,
                "verdict": "no resolved predictions yet"}
    res = cross_validate(p, y, cohorts)
    res["target"] = target
    res["fit_regimes"] = regimes

    if res["improved"]:
        g = fit_correction(p, y)
        path = correction_path(target, models_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"model": g, "regimes": regimes,
                                       "target": target}))
        verdict = (f"installed: {res['cohorts_improved']}/{res['cohorts']} "
                   f"cohorts improved, t={res['t_stat']:.2f}")
    elif res["cohorts"] < MIN_COHORTS:
        verdict = (f"refused: {res['cohorts']} cohort(s) is too few to "
                   "validate anything")
    else:
        verdict = (
            f"refused: {res['cohorts_improved']}/{res['cohorts']} cohorts "
            f"improved but t={res.get('t_stat', float('nan')):.2f} misses "
            f"{res.get('t_critical', float('nan')):.2f}; worst cohort "
            f"{res.get('delta_worst', 0):+.4f}")
    res["verdict"] = verdict
    vdb.upsert(con, "calibration_log", [{
        "target": target, "cohorts": res["cohorts"], "n": res["n"],
        "brier_base": res["brier_base"],
        "brier_corrected": res["brier_corrected"],
        "delta_mean": res.get("delta_mean"), "delta_se": res.get("delta_se"),
        "t_stat": res.get("t_stat"), "t_critical": res.get("t_critical"),
        "cohorts_improved": res.get("cohorts_improved"),
        "installed": res["improved"], "fit_regimes": json.dumps(regimes),
        "verdict": verdict,
    }])
    return res


def load_correction(target: str, models_dir=None):
    """Installed correction, or None. Returns {model, regimes, target}."""
    path = correction_path(target, models_dir)
    if not path.exists():
        return None
    return pickle.loads(path.read_bytes())
