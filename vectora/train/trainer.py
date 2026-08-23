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

    # --- production refit + common-holdout scoring -----------------------
    # Two separate models, for two separate jobs.
    #
    # Walk-forward deliberately withholds recent data so it can be tested on.
    # Shipping the last fold's model therefore ships something that has never
    # seen the most recent stretch of market: on 2026-08-20 the deployed
    # model's training data ended 2025-06-02, leaving 173 trading days unused.
    # Walk-forward is for ESTIMATING performance; the model that goes live
    # should be refit on everything.
    #
    # The promotion decision cannot use that refit, because it would be
    # scored on data it trained on. So a separate evaluation model is fit
    # strictly before a common holdout, and challenger and incumbent are
    # compared there — on the SAME rows. Comparing each model's own pooled
    # Brier is apples to oranges: a challenger can lose purely because its
    # test window was a harder market, which is what froze the live model
    # for four weeks.
    holdout_dates = dates[-test_days:]
    holdout_start = holdout_dates[0]
    eval_cut_i = max(0, len(dates) - test_days - embargo_days)
    eval_cut = dates[eval_cut_i]

    def _xy(frame):
        return (frame.select(feat_names).to_numpy().astype(np.float64),
                frame[label_col].to_numpy().astype(int))

    holdout = df.filter(pl.col("date") >= holdout_start)
    eval_tr = df.filter(pl.col("date") < eval_cut)
    X_ho, y_ho = _xy(holdout)

    eval_model = None
    if eval_tr.height and len(np.unique(_xy(eval_tr)[1])) >= 2:
        ed = sorted(eval_tr["date"].unique().to_list())
        cut = ed[int(len(ed) * 0.85)]
        f_, v_ = (eval_tr.filter(pl.col("date") < cut),
                  eval_tr.filter(pl.col("date") >= cut))
        Xf, yf = _xy(f_)
        Xv, yv = _xy(v_)
        if len(np.unique(yf)) >= 2 and len(yv):
            eval_model = M.fit_lgbm(Xf, yf, Xv, yv)
            eval_cal = M.fit_calibrator(M.predict(eval_model, Xv), yv)

    challenger_holdout_brier = None
    challenger_probs = None
    if eval_model is not None and len(y_ho):
        challenger_probs = M.apply_calibrator(
            eval_cal, M.predict(eval_model, X_ho))
        challenger_holdout_brier = M.brier(y_ho, challenger_probs)

    # the model that actually ships: every labelled row, nothing held back
    all_dates = sorted(df["date"].unique().to_list())
    prod_cut = all_dates[int(len(all_dates) * 0.85)]
    prod_fit = df.filter(pl.col("date") < prod_cut)
    prod_val = df.filter(pl.col("date") >= prod_cut)
    Xpf, ypf = _xy(prod_fit)
    Xpv, ypv = _xy(prod_val)
    prod_model = M.fit_lgbm(Xpf, ypf, Xpv, ypv)
    prod_train_end = all_dates[-1]
    last_models["lgbm"] = prod_model

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
            # calibrate the model that actually ships, on its own held-out
            # slice. The pooled-OOS calibrator maps the FOLD models' raw
            # scores; applying it to a differently-fit production model would
            # be calibrating one model with another model's curve.
            deploy_cal = fit_deploy_calibrator(
                M.predict(prod_model, Xpv), ypv)
            import pickle
            (art / "calibrator.pkl").write_bytes(pickle.dumps(deploy_cal))
        else:
            import pickle
            (art / "logistic.pkl").write_bytes(
                pickle.dumps(last_models["logistic"]))
        (art / "meta.json").write_text(json.dumps({
            "model_id": model_id, "family": fam, "target": target,
            "features": feat_names,
            "train_end": str(prod_train_end if fam == "lgbm"
                             else last_models["train_end"]),
            "calibration": "pooled-oos" if fam == "lgbm" else None,
            "holdout_brier": challenger_holdout_brier if fam == "lgbm" else None,
            "holdout_start": str(holdout_start) if fam == "lgbm" else None,
            "metrics": metrics[fam] | {"reliability": None},
        }, indent=1), encoding="utf-8")
        registered.append((model_id, fam))
        vdb.upsert(con, "model_registry", [{
            "model_id": model_id, "family": fam, "target": target,
            "trained_at": dt.datetime.now().isoformat(),
            "train_end": str(prod_train_end if fam == "lgbm"
                             else last_models["train_end"]),
            "metrics": json.dumps(
                metrics[fam] | {"reliability": None,
                                "holdout_brier": challenger_holdout_brier,
                                "holdout_start": str(holdout_start)}),
            "artifact_dir": registry_dir, "active": False,
        }])

    lgbm_id = next(mid for mid, fam in registered if fam == "lgbm")
    ho_dates = holdout["date"].to_list() if len(y_ho) else []
    promoted = promote_if_better(
        con, target, lgbm_id, metrics["lgbm"]["brier"],
        holdout=(X_ho, y_ho, feat_names, ho_dates) if len(y_ho) else None,
        challenger_holdout_brier=challenger_holdout_brier,
        challenger_probs=challenger_probs)

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
                      new_brier: float, holdout=None,
                      challenger_holdout_brier: float | None = None,
                      challenger_probs=None) -> bool:
    """Challenger guard (spec 17.5), scored on a COMMON holdout.

    The original version compared each model's own pooled-OOS Brier. Those
    numbers come from different test periods, so a challenger could lose
    purely because its window was a harder market — which is exactly what
    happened: the live g5_h10 model sat unchanged from 24 July while three
    retrains were rejected on an invalid comparison.

    When a holdout is supplied, the incumbent is loaded and scored on the
    same rows, and the two are compared with a PAIRED test across the
    holdout dates rather than on pooled Brier. Pooling rows there would
    repeat the error this guard exists to fix: 126 dates of correlated
    predictions are not 41,576 independent observations, and a 0.9%
    difference in pooled Brier can be pure noise.

    The tie-break favours the challenger. Features and hyperparameters are
    identical between vintages, so the only real difference is data
    recency — and refusing a fresher model over an insignificant gap is
    precisely how the live model came to be trained on data ending
    2024-11-21 while the market moved on for 21 months. The incumbent is
    kept only when it is SIGNIFICANTLY better.
    """
    row = con.execute(
        """
        SELECT model_id, metrics, artifact_dir FROM model_registry
        WHERE target = ? AND family = 'lgbm' AND active
        ORDER BY trained_at DESC LIMIT 1
        """, [target]).fetchone()
    if row is not None:
        incumbent_brier = json.loads(row[1]).get("brier")
        decided = False
        if holdout is not None and challenger_probs is not None:
            verdict = _paired_holdout_verdict(row[2], holdout, challenger_probs)
            if verdict is not None:
                decided = True
                if verdict["incumbent_significantly_better"]:
                    return False
        if not decided and incumbent_brier is not None \
                and challenger_holdout_brier is None \
                and new_brier > incumbent_brier:
            return False
        con.execute(
            "UPDATE model_registry SET active = false WHERE model_id = ?",
            [row[0]])
    con.execute(
        "UPDATE model_registry SET active = true WHERE model_id = ?",
        [model_id])
    return True


def _paired_holdout_verdict(artifact_dir: str, holdout,
                            challenger_probs) -> dict | None:
    """Per-date paired comparison of incumbent vs challenger.

    Returns None when the incumbent cannot be scored (missing artifact, or a
    changed feature set), which leaves the caller to fall back rather than
    guess. Otherwise reports whether the incumbent is significantly better
    across the holdout dates, one-sided at 95%.
    """
    from scipy import stats
    probs = _score_incumbent(artifact_dir, holdout, return_probs=True)
    if probs is None:
        return None
    X, y, _feat, dates = holdout
    dates = np.asarray([str(d) for d in dates])
    challenger_probs = np.asarray(challenger_probs, dtype=float)
    if len(challenger_probs) != len(y):
        return None
    deltas = []
    for d in sorted(set(dates.tolist())):
        m = dates == d
        deltas.append(float(np.mean((challenger_probs[m] - y[m]) ** 2)
                            - np.mean((probs[m] - y[m]) ** 2)))
    deltas = np.asarray(deltas)
    out = {"dates": len(deltas), "delta_mean": float(deltas.mean()),
           "dates_challenger_better": int((deltas < 0).sum()),
           "incumbent_significantly_better": False}
    if len(deltas) >= 2 and deltas.std(ddof=1) > 0:
        se = deltas.std(ddof=1) / np.sqrt(len(deltas))
        t_stat = float(deltas.mean() / se)
        crit = float(stats.t.ppf(1 - 0.05, len(deltas) - 1))   # positive
        out["t_stat"] = t_stat
        out["t_critical"] = crit
        # a POSITIVE delta means the challenger is worse; only reject when
        # that is statistically convincing, not merely true on average
        out["incumbent_significantly_better"] = bool(t_stat >= crit)
    elif len(deltas) >= 1:
        # zero variance across dates is the OPPOSITE of inconclusive: the
        # same gap on every single date is as consistent as evidence gets,
        # and treating it as "not significant" would wave through a model
        # that is uniformly worse
        out["incumbent_significantly_better"] = bool(deltas.mean() > 0)
    return out


def _score_incumbent(artifact_dir: str, holdout, return_probs: bool = False):
    """Brier (or calibrated probabilities) of the active model on the
    challenger's holdout rows."""
    import pickle
    X, y, feat_names = holdout[0], holdout[1], holdout[2]
    art = Path(artifact_dir)
    booster_path, cal_path = art / "lgbm.txt", art / "calibrator.pkl"
    if not booster_path.exists() or not cal_path.exists() or not len(y):
        return None
    try:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(booster_path))
        meta_path = art / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # a feature-set change makes the two models incomparable; fall
            # back rather than score the incumbent on the wrong columns
            if meta.get("features") and meta["features"] != list(feat_names):
                return None
        cal = pickle.loads(cal_path.read_bytes())
        probs = M.apply_calibrator(cal, booster.predict(X))
        return probs if return_probs else M.brier(y, probs)
    except Exception:      # noqa: BLE001 - never block promotion on a bad artifact
        return None


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
