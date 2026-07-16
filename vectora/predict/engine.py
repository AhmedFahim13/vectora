"""Daily prediction stage (spec §9.3): score the tradable universe with each
active model, gate signals, persist predictions + risk + explanations.

Gate order (first failing gate is the recorded reason):
quality floor -> target enabled -> Z category -> probability threshold.
Predictions are always stored regardless of gating; is_signal marks the
survivors. Idempotent per (date, target, symbol)."""
import json
from datetime import date

import numpy as np
import polars as pl

from vectora import db as vdb
from vectora import labels as lab
from vectora.features import engine as fengine
from vectora.predict import analogs, explain, loaders, risk
from vectora.predict.explain import drivers as shap_drivers
from vectora.settings import ANALOG_K, MIN_QUALITY_SCORE, SIGNAL_THRESHOLDS
from vectora.train import models as M
from vectora.universe import tradable_universe

TARGETS = ("g5_h10", "g10_h30")
_TARGET_HORIZON = {"g5_h10": 10, "g10_h30": 30}
_TARGET_X = {"g5_h10": 5, "g10_h30": 10}


def run_predict(con, date_str: str | None = None, features_path=None,
                min_median_value_mn: float = 1.0) -> dict:
    run_date = date_str or str(con.execute(
        "SELECT max(date) FROM prices").fetchone()[0])
    quality_row = con.execute(
        "SELECT score FROM data_quality WHERE date = ? AND source = 'dse_eod'",
        [run_date]).fetchone()
    quality = quality_row[0] if quality_row else 0

    active = [m for m in (loaders.active_model(con, t) for t in TARGETS)
              if m is not None]
    if not active:
        return {"date": run_date, "predictions": 0, "signals": 0, "targets": []}

    features = fengine.compute(con, out_path=features_path) if features_path \
        else fengine.compute(con)
    universe = set(tradable_universe(
        con, as_of=run_date, min_median_value_mn=min_median_value_mn))
    today = features.filter(
        (pl.col("date") == date.fromisoformat(run_date))
        & pl.col("symbol").is_in(sorted(universe)))
    categories = dict(con.execute(
        "SELECT symbol, category FROM symbols").fetchall())

    n_pred = n_sig = 0
    for model in active:
        target = model["target"]
        horizon, x_pct = _TARGET_HORIZON[target], _TARGET_X[target]
        booster, calibrator = loaders.load_artifacts(model)
        feat_names = model["features"]

        labeled = lab.make_labels(
            features, thresholds=(x_pct / 100,), horizons=(horizon,),
            continuous=True)
        idx = analogs.AnalogIndex.fit(
            labeled, feature_names=feat_names,
            label_col=f"y_g{x_pct}_h{horizon}",
            fwdmax_col=f"fwdmax_h{horizon}", fwdmin_col=f"fwdmin_h{horizon}")

        X = today.select(feat_names).to_numpy().astype(np.float64)
        raw = booster.predict(X)
        probs = M.apply_calibrator(calibrator, raw)

        preds, risks, expls = [], [], []
        for i, symbol in enumerate(today["symbol"].to_list()):
            p = float(np.clip(probs[i], 0.0, 1.0))
            stats = idx.query(X[i], k=ANALOG_K)
            row = today.row(i, named=True)
            block = risk.build(
                vol_21d=row.get("vol_21d"),
                value_mn_med_21d=row.get("value_mn_med_21d"),
                category=categories.get(symbol), analog_stats=stats)
            suppressed = _gate(p, target, block["category"], quality)
            pid = f"{run_date}_{target}_{symbol}"
            preds.append({
                "id": pid, "symbol": symbol, "date": run_date,
                "target": target, "probability": p,
                "model_id": model["model_id"], "quality_score": quality,
                "is_signal": suppressed is None,
                "suppressed_reason": suppressed,
            })
            risks.append({"prediction_id": pid, **block})
            d = shap_drivers(booster, X[i], feat_names)
            expls.append({
                "prediction_id": pid,
                "drivers": json.dumps(d),
                "analogs": json.dumps(stats),
                "rendered": explain.render(symbol, target, p, d, stats,
                                           block, quality),
            })
            n_pred += 1
            n_sig += 1 if suppressed is None else 0
        vdb.upsert(con, "predictions", preds)
        vdb.upsert(con, "risk_blocks", risks)
        vdb.upsert(con, "explanations", expls)

    return {"date": run_date, "predictions": n_pred, "signals": n_sig,
            "targets": [m["target"] for m in active]}


def _gate(p: float, target: str, category: str | None,
          quality: int) -> str | None:
    """First failing gate wins; None means the prediction is a signal."""
    if quality < MIN_QUALITY_SCORE:
        return "quality-below-floor"
    if target not in SIGNAL_THRESHOLDS:
        return "g10_h30-tail-overconfident" if target == "g10_h30" \
            else "target-not-enabled"
    if category == "Z":
        return "z-category-gate"
    if p < SIGNAL_THRESHOLDS[target]:
        return "below-probability-threshold"
    return None
