# vectora/train/trainer.py
"""Training orchestration: panel -> universe filter -> features + labels ->
walk-forward -> logistic vs LightGBM -> calibration -> report + registry.

Fold protocol: within each fold, the last 15% of training DATES are the
LightGBM early-stopping / calibration validation slice (never test data).
Out-of-sample predictions from all folds are pooled for the headline
Brier/AUC comparison. Artifacts saved for the final fold's models."""
import datetime as dt
import json
import re
import uuid
from pathlib import Path

import numpy as np
import polars as pl

from vectora import db as vdb
from vectora import labels as lab
from vectora.features import engine, registry
from vectora.settings import MODELS_DIR, REPORTS_DIR
from vectora.train import models as M
from vectora.train import walkforward as wf
from vectora.universe import tradable_universe

_TARGET_RE = re.compile(r"^g(\d+)_h(\d+)$")


def run(con, target: str = "g5_h10",
        models_dir=MODELS_DIR, reports_dir=REPORTS_DIR,
        features_path=None, min_train_days: int = 750, test_days: int = 126,
        embargo_days: int = 30, min_median_value_mn: float = 1.0) -> dict:
    m = _TARGET_RE.match(target)
    if not m:
        raise ValueError(f"bad target '{target}', expected like g5_h10")
    x_pct, horizon = int(m.group(1)), int(m.group(2))
    label_col = f"y_g{x_pct}_h{horizon}"

    feat_names = [s.name for s in registry.load()]
    df = engine.compute(con, out_path=features_path) if features_path else \
        engine.compute(con)
    df = lab.make_labels(df, thresholds=(x_pct / 100,), horizons=(horizon,))

    last_date = df["date"].max()
    universe = tradable_universe(con, as_of=str(last_date),
                                 min_median_value_mn=min_median_value_mn)
    df = df.filter(pl.col("symbol").is_in(universe)
                   & pl.col(label_col).is_not_null())

    dates = sorted(df["date"].unique().to_list())
    folds = wf.splits(dates, min_train_days=min_train_days,
                      test_days=test_days, embargo_days=embargo_days)
    if not folds:
        raise ValueError("not enough history for a single walk-forward fold")

    pooled = {"logistic": ([], []), "lgbm": ([], [])}  # (y, p)
    pooled_raw_lgbm: list = []
    last_models = {}
    for split in folds:
        tr = df.filter((pl.col("date") >= split.train_start)
                       & (pl.col("date") <= split.train_end))
        te = df.filter((pl.col("date") >= split.test_start)
                       & (pl.col("date") <= split.test_end))
        tr_dates = sorted(tr["date"].unique().to_list())
        val_cut = tr_dates[int(len(tr_dates) * 0.85)]
        fit = tr.filter(pl.col("date") < val_cut)
        val = tr.filter(pl.col("date") >= val_cut)

        def xy(frame):
            X = frame.select(feat_names).to_numpy().astype(np.float64)
            y = frame[label_col].to_numpy().astype(int)
            return X, y

        X_fit, y_fit = xy(fit)
        X_val, y_val = xy(val)
        X_te, y_te = xy(te)
        if len(np.unique(y_fit)) < 2 or len(y_te) == 0:
            continue

        logit = M.fit_logistic(np.vstack([X_fit, X_val]),
                               np.concatenate([y_fit, y_val]))
        lgbm = M.fit_lgbm(X_fit, y_fit, X_val, y_val)
        cal = M.fit_calibrator(M.predict(lgbm, X_val), y_val)

        pooled["logistic"][0].extend(y_te)
        pooled["logistic"][1].extend(M.predict(logit, X_te))
        raw_te = M.predict(lgbm, X_te)
        pooled["lgbm"][0].extend(y_te)
        pooled["lgbm"][1].extend(M.apply_calibrator(cal, raw_te))
        pooled_raw_lgbm.extend(raw_te)
        last_models = {"logistic": logit, "lgbm": lgbm, "cal": cal,
                       "train_end": split.train_end}

    metrics = {}
    for fam, (ys, ps) in pooled.items():
        ys, ps = np.asarray(ys), np.asarray(ps)
        metrics[fam] = {
            "brier": M.brier(ys, ps), "auc": M.auc(ys, ps),
            "n": int(len(ys)), "base_rate": float(ys.mean()),
            "reliability": M.reliability_table(ys, ps),
        }

    run_id = dt.date.today().isoformat()
    registered: list[tuple[str, str]] = []
    for fam in ("logistic", "lgbm"):
        model_id = f"{target}_{fam}_{run_id}_{uuid.uuid4().hex[:6]}"
        art = models_dir / model_id
        art.mkdir(parents=True, exist_ok=True)
        # registry stores a cwd-relative path: absolute local paths break on
        # any other machine (e.g. the Linux CI runner loading artifacts)
        try:
            registry_dir = str(art.relative_to(Path.cwd()))
        except ValueError:
            registry_dir = str(art)
        # POSIX separators always: a Windows-trained "models\x" path is
        # relative but unresolvable on the Linux CI runner, which silently
        # killed the predict stage for five days (2026-07-19..23)
        registry_dir = registry_dir.replace("\\", "/")
        if fam == "lgbm":
            last_models["lgbm"].booster_.save_model(str(art / "lgbm.txt"))
            deploy_cal = fit_deploy_calibrator(
                pooled_raw_lgbm, pooled["lgbm"][0])
            import pickle
            (art / "calibrator.pkl").write_bytes(pickle.dumps(deploy_cal))
        else:
            import pickle
            (art / "logistic.pkl").write_bytes(
                pickle.dumps(last_models["logistic"]))
        (art / "meta.json").write_text(json.dumps({
            "model_id": model_id, "family": fam, "target": target,
            "features": feat_names, "train_end": str(last_models["train_end"]),
            "calibration": "pooled-oos" if fam == "lgbm" else None,
            "metrics": metrics[fam] | {"reliability": None},
        }, indent=1), encoding="utf-8")
        registered.append((model_id, fam))
        vdb.upsert(con, "model_registry", [{
            "model_id": model_id, "family": fam, "target": target,
            "trained_at": dt.datetime.now().isoformat(),
            "train_end": str(last_models["train_end"]),
            "metrics": json.dumps(metrics[fam] | {"reliability": None}),
            "artifact_dir": registry_dir, "active": False,
        }])

    lgbm_id = next(mid for mid, fam in registered if fam == "lgbm")
    promoted = promote_if_better(con, target, lgbm_id,
                                 metrics["lgbm"]["brier"])

    report = _render_report(target, folds, metrics)
    deploy_probs = deploy_cal.predict(np.asarray(pooled_raw_lgbm))
    report += "\n## Reliability (deployment calibrator, in-sample on pooled OOS)\n\n"
    report += "| bin | n | predicted | realized |\n|---|---|---|---|\n"
    for b in M.reliability_table(pooled["lgbm"][0], deploy_probs):
        report += (f"| {b['bin_lo']:.1f}-{b['bin_hi']:.1f} | {b['n']} "
                   f"| {b['p_mean']:.3f} | {b['y_rate']:.3f} |\n")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"train_{target}_{run_id}.md").write_text(
        report, encoding="utf-8")

    return {"folds": len(folds), "target": target,
            "lgbm_brier": metrics["lgbm"]["brier"],
            "logistic_brier": metrics["logistic"]["brier"],
            "lgbm_auc": metrics["lgbm"]["auc"],
            "logistic_auc": metrics["logistic"]["auc"],
            "promoted": promoted}


def fit_deploy_calibrator(raw_pooled, y_pooled):
    """Deployment calibrator fit on ALL pooled OOS raw predictions —
    orders of magnitude more tail data than one fold's validation slice
    (the g10_h30 tail overconfidence root cause, training report
    2026-07-16)."""
    return M.fit_calibrator(np.asarray(raw_pooled), np.asarray(y_pooled))


def promote_if_better(con, target: str, model_id: str,
                      new_brier: float) -> bool:
    """Challenger guard (spec §17.5): activate the new lgbm model only if
    its pooled-OOS Brier is <= the incumbent's stored metric."""
    row = con.execute(
        """
        SELECT model_id, metrics FROM model_registry
        WHERE target = ? AND family = 'lgbm' AND active
        ORDER BY trained_at DESC LIMIT 1
        """, [target]).fetchone()
    if row is not None:
        incumbent_brier = json.loads(row[1]).get("brier")
        if incumbent_brier is not None and new_brier > incumbent_brier:
            return False
        con.execute(
            "UPDATE model_registry SET active = false WHERE model_id = ?",
            [row[0]])
    con.execute(
        "UPDATE model_registry SET active = true WHERE model_id = ?",
        [model_id])
    return True


def _render_report(target, folds, metrics) -> str:
    verdict = ("PASS - LightGBM beats the logistic baseline on Brier"
               if metrics["lgbm"]["brier"] < metrics["logistic"]["brier"]
               else "FAIL - baseline wins; features carry no GBM-exploitable signal yet")
    lines = [
        f"# Training report: {target}",
        "",
        f"Walk-forward folds: {len(folds)} | pooled OOS rows: "
        f"{metrics['lgbm']['n']} | base rate: {metrics['lgbm']['base_rate']:.3f}",
        "",
        "| model | Brier | AUC |",
        "|---|---|---|",
        f"| logistic (baseline) | {metrics['logistic']['brier']:.4f} "
        f"| {metrics['logistic']['auc']:.3f} |",
        f"| lightgbm (calibrated) | {metrics['lgbm']['brier']:.4f} "
        f"| {metrics['lgbm']['auc']:.3f} |",
        "",
        f"**Phase 2 exit criterion: {verdict}**",
        "",
        "## Reliability (calibrated LightGBM)",
        "",
        "| bin | n | predicted | realized |",
        "|---|---|---|---|",
    ]
    for row in metrics["lgbm"]["reliability"]:
        lines.append(f"| {row['bin_lo']:.1f}-{row['bin_hi']:.1f} | {row['n']} "
                     f"| {row['p_mean']:.3f} | {row['y_rate']:.3f} |")
    return "\n".join(lines) + "\n"
