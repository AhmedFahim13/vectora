"""Promotion guard: challenger and incumbent must be scored on the same rows.

The original guard compared each model's own pooled-OOS Brier. Those come
from different test periods, so a challenger could be rejected purely for
having drawn a harder market — which is what froze the live g5_h10 model
from 24 July while three retrains were turned away.
"""
import json

import numpy as np

from vectora import db as vdb
from vectora.train import trainer


def _register(con, model_id, brier, active=False, artifact_dir="models/none"):
    vdb.upsert(con, "model_registry", [{
        "model_id": model_id, "family": "lgbm", "target": "g5_h10",
        "trained_at": f"2026-08-{10 + len(model_id) % 10}",
        "train_end": "2026-08-01",
        "metrics": json.dumps({"brier": brier}),
        "artifact_dir": artifact_dir, "active": active}])


def test_first_model_is_promoted_unopposed(test_db):
    _register(test_db, "challenger", 0.21)
    assert trainer.promote_if_better(test_db, "g5_h10", "challenger", 0.21)
    row = test_db.execute(
        "SELECT active FROM model_registry WHERE model_id = 'challenger'"
    ).fetchone()
    assert row[0] is True


def test_worse_challenger_is_refused(test_db):
    _register(test_db, "incumbent", 0.20, active=True)
    _register(test_db, "challenger", 0.25)
    assert not trainer.promote_if_better(test_db, "g5_h10", "challenger", 0.25)
    still = test_db.execute(
        "SELECT model_id FROM model_registry WHERE active").fetchall()
    assert still == [("incumbent",)]


def _holdout(n_dates=40, per=25, seed=0):
    """(X, y, features, dates) with several rows on each of many dates."""
    rng = np.random.default_rng(seed)
    dates, y = [], []
    for d in range(n_dates):
        dates += [f"2026-01-{d + 1:02d}"] * per
        y += list(rng.integers(0, 2, per))
    n = len(y)
    return (np.zeros((n, 2)), np.array(y), ["a", "b"], dates)


def test_insignificant_difference_promotes_the_fresher_model(test_db,
                                                             monkeypatch):
    """The tie-break favours recency. Refusing a fresher model over noise is
    how the live model reached 21 months stale."""
    _register(test_db, "incumbent", 0.19, active=True)
    _register(test_db, "challenger", 0.21)
    ho = _holdout()
    y = ho[1]
    # both models essentially equivalent, challenger a hair worse
    inc = np.where(y == 1, 0.55, 0.45)
    monkeypatch.setattr(trainer, "_score_incumbent",
                        lambda *_a, **_k: inc)
    promoted = trainer.promote_if_better(
        test_db, "g5_h10", "challenger", 0.21, holdout=ho,
        challenger_holdout_brier=0.22, challenger_probs=inc + 0.001)
    assert promoted is True
    active = test_db.execute(
        "SELECT model_id FROM model_registry WHERE active").fetchall()
    assert active == [("challenger",)]


def test_significantly_worse_challenger_is_still_rejected(test_db,
                                                          monkeypatch):
    _register(test_db, "incumbent", 0.25, active=True)
    _register(test_db, "challenger", 0.20)
    ho = _holdout()
    y = ho[1]
    inc = np.where(y == 1, 0.9, 0.1)          # incumbent nearly perfect
    chal = np.where(y == 1, 0.1, 0.9)         # challenger inverted
    monkeypatch.setattr(trainer, "_score_incumbent", lambda *_a, **_k: inc)
    promoted = trainer.promote_if_better(
        test_db, "g5_h10", "challenger", 0.20, holdout=ho,
        challenger_holdout_brier=0.6, challenger_probs=chal)
    assert promoted is False


def test_unscorable_incumbent_falls_back_not_crashes(test_db):
    """A missing or unreadable artifact must not block the pipeline."""
    _register(test_db, "incumbent", 0.30, active=True,
              artifact_dir="models/does_not_exist")
    _register(test_db, "challenger", 0.20)
    ho = _holdout()
    promoted = trainer.promote_if_better(
        test_db, "g5_h10", "challenger", 0.20, holdout=ho,
        challenger_holdout_brier=0.22,
        challenger_probs=np.full(len(ho[1]), 0.4))
    assert promoted is True


def test_score_incumbent_returns_none_for_missing_artifact():
    holdout = (np.zeros((3, 2)), np.array([0, 1, 0]), ["a", "b"], ["d"] * 3)
    assert trainer._score_incumbent("models/nope", holdout) is None


def test_score_incumbent_refuses_a_different_feature_set(tmp_path):
    """Two models with different columns are not comparable."""
    art = tmp_path / "m"
    art.mkdir()
    (art / "lgbm.txt").write_text("stub", encoding="utf-8")
    (art / "calibrator.pkl").write_bytes(b"stub")
    (art / "meta.json").write_text(
        json.dumps({"features": ["x", "y", "z"]}), encoding="utf-8")
    holdout = (np.zeros((3, 2)), np.array([0, 1, 0]), ["a", "b"], ["d"] * 3)
    assert trainer._score_incumbent(str(art), holdout) is None


def test_verdict_is_paired_across_dates_not_pooled(tmp_path, monkeypatch):
    """The guard must not treat 1,000 rows on 40 dates as 1,000 draws."""
    ho = _holdout()
    y = ho[1]
    inc = np.where(y == 1, 0.55, 0.45)
    monkeypatch.setattr(trainer, "_score_incumbent", lambda *_a, **_k: inc)
    v = trainer._paired_holdout_verdict("x", ho, inc + 0.001)
    assert v["dates"] == 40
    assert v["incumbent_significantly_better"] is False


def test_uniform_degradation_is_maximally_significant_not_inconclusive(
        monkeypatch):
    """Zero variance across dates means the gap is identical every day.
    A naive std>0 guard treats that as 'no evidence' and promotes a model
    that is uniformly worse."""
    ho = _holdout()
    y = ho[1]
    inc = np.where(y == 1, 0.9, 0.1)
    chal = np.where(y == 1, 0.1, 0.9)          # same delta on every date
    monkeypatch.setattr(trainer, "_score_incumbent", lambda *_a, **_k: inc)
    v = trainer._paired_holdout_verdict("x", ho, chal)
    assert v["dates_challenger_better"] == 0
    assert v["incumbent_significantly_better"] is True
