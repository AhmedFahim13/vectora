# Vectora Phase 5A: Learning Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close spec §17: pooled-OOS recalibration (fixing the g10_h30 tail), event/regime features feeding retrained models, calibration accounting with error-autopsy tags and evaluation reports, and a weekly scheduled retrain with a challenger auto-promotion guard.

**Architecture:** Four surgical changes to existing machinery. (1) `trainer.run` keeps per-fold calibrators for honest OOS metrics but SAVES a deployment calibrator fit on the pooled raw OOS predictions of all 18 folds — ~660K points instead of one fold's validation slice, which is what starved the g10_h30 tail. (2) The feature engine gains three event/regime base columns (days since last material event, board-meeting-recency flag, regime code) joined leakage-safe on `post_date <= date`, registered as passthrough features. (3) A new `evaluate` module grades resolved predictions (Brier/hit-rate/reliability, segmented by regime and category), tags every miss with a rule-based autopsy cause into `outcome_tags`, and renders reports + a vault Evaluations note — gracefully empty until outcomes mature (~Jul 30). (4) `train.yml` gains a Friday cron and the trainer auto-promotes a newly trained model only if its pooled-OOS Brier beats the incumbent's stored metric.

**Tech Stack:** existing only.

**Existing contracts:** `trainer.run` (pooled dict of per-family (ys, ps) lists, per-fold `fit`/`val`/`te` split, artifacts + model_registry rows with `metrics` JSON incl. `"brier"`, `active=False`); `loaders.active_model` picks latest `active` lgbm per target; `SIGNAL_THRESHOLDS = {"g5_h10": 0.55}` in settings with the `g10_h30-tail-overconfident` gate in `predict/engine._gate`; `features/engine.compute` joins symbols meta then applies registry specs via `families.apply`; `event_labels.materiality`; `regimes` table; `outcomes(prediction_id, realized_max, realized_min, hit)`; `risk_blocks.exit_days`; `_write_machine` for vault notes. Fast tests: `uv run pytest -m "not slow"` (currently 191). Branch `phase-5a-loop` off main. Bulk-seed rule for synthetic prices.

**File structure:**

```
vectora/train/trainer.py           # pooled-OOS deploy calibrator + promotion guard
vectora/features/engine.py         # event/regime base columns
vectora/features/families.py       # col_passthrough fn
vectora/config/features.yaml       # + 3 features
vectora/evaluate/__init__.py, report.py
vectora/db.py                      # + outcome_tags table
vectora/__main__.py                # + evaluate stage
.github/workflows/train.yml        # + Friday cron + evaluate step
tests/train/test_promotion.py, tests/evaluate/…, feature test appends
```

---

### Task 1: Pooled-OOS deployment calibrator + promotion guard

**Files:**
- Modify: `vectora/train/trainer.py`
- Test: `tests/train/test_promotion.py` (new; the existing slow `test_trainer.py` end-to-end keeps passing unchanged)

- [ ] **Step 1: Write the failing tests**

```python
# tests/train/test_promotion.py
import json

import numpy as np

from vectora import db as vdb
from vectora.train import trainer


def test_promote_activates_better_challenger(test_db):
    vdb.upsert(test_db, "model_registry", [{
        "model_id": "old", "family": "lgbm", "target": "g5_h10",
        "trained_at": "2026-07-01T00:00:00", "train_end": "2026-06-01",
        "metrics": json.dumps({"brier": 0.210}), "artifact_dir": "models/old",
        "active": True}])
    promoted = trainer.promote_if_better(
        test_db, target="g5_h10", model_id="new", new_brier=0.205)
    assert promoted is True
    rows = dict(test_db.execute(
        "SELECT model_id, active FROM model_registry").fetchall())
    # old row absent from registry is impossible here; new row must be added
    # by trainer.run in production — promote_if_better only flips flags for
    # rows that exist, so seed the new row first in this unit test:


def test_promote_flags_and_demotes(test_db):
    for mid, active, brier in (("old", True, 0.210), ("new", False, 0.205)):
        vdb.upsert(test_db, "model_registry", [{
            "model_id": mid, "family": "lgbm", "target": "g5_h10",
            "trained_at": "2026-07-01T00:00:00", "train_end": "2026-06-01",
            "metrics": json.dumps({"brier": brier}),
            "artifact_dir": f"models/{mid}", "active": active}])
    assert trainer.promote_if_better(
        test_db, target="g5_h10", model_id="new", new_brier=0.205) is True
    rows = dict(test_db.execute(
        "SELECT model_id, active FROM model_registry").fetchall())
    assert rows == {"old": False, "new": True}


def test_worse_challenger_stays_inactive(test_db):
    for mid, active, brier in (("old", True, 0.200), ("new", False, 0.215)):
        vdb.upsert(test_db, "model_registry", [{
            "model_id": mid, "family": "lgbm", "target": "g5_h10",
            "trained_at": "2026-07-01T00:00:00", "train_end": "2026-06-01",
            "metrics": json.dumps({"brier": brier}),
            "artifact_dir": f"models/{mid}", "active": active}])
    assert trainer.promote_if_better(
        test_db, target="g5_h10", model_id="new", new_brier=0.215) is False
    rows = dict(test_db.execute(
        "SELECT model_id, active FROM model_registry").fetchall())
    assert rows == {"old": True, "new": False}


def test_no_incumbent_auto_promotes(test_db):
    vdb.upsert(test_db, "model_registry", [{
        "model_id": "new", "family": "lgbm", "target": "g5_h10",
        "trained_at": "2026-07-01T00:00:00", "train_end": "2026-06-01",
        "metrics": json.dumps({"brier": 0.30}), "artifact_dir": "models/new",
        "active": False}])
    assert trainer.promote_if_better(
        test_db, target="g5_h10", model_id="new", new_brier=0.30) is True


def test_pooled_calibrator_is_monotone():
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 1, 3000)
    y = (rng.uniform(size=3000) < raw ** 1.5).astype(int)  # miscalibrated
    cal = trainer.fit_deploy_calibrator(raw, y)
    grid = cal.predict(np.linspace(0, 1, 50))
    assert (np.diff(grid) >= -1e-9).all()
    assert 0 <= grid.min() and grid.max() <= 1
```

Remove the trailing comment-only assertions in the first test (`test_promote_activates_better_challenger`) — fold its scenario into `test_promote_flags_and_demotes` and DELETE the first test entirely; it exists in this plan only to show the seed-first requirement. The final test file has 4 tests.

- [ ] **Step 2: Run to verify failure** — FAIL (functions missing).

- [ ] **Step 3: Implement in `vectora/train/trainer.py`**

Add two functions (top-level, after `run`):

```python
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
```

Then modify `run`: (a) pool RAW lgbm predictions alongside calibrated ones — in the fold loop change the lgbm pooling block to:

```python
        raw_te = M.predict(lgbm, X_te)
        pooled["lgbm"][0].extend(y_te)
        pooled["lgbm"][1].extend(M.apply_calibrator(cal, raw_te))
        pooled_raw_lgbm.extend(raw_te)
```

with `pooled_raw_lgbm: list = []` initialized next to `pooled`. (b) In the artifact-saving loop, for the lgbm family replace the calibrator dump with the deployment calibrator and record the method in meta:

```python
        if fam == "lgbm":
            last_models["lgbm"].booster_.save_model(str(art / "lgbm.txt"))
            deploy_cal = fit_deploy_calibrator(
                pooled_raw_lgbm, pooled["lgbm"][0])
            import pickle
            (art / "calibrator.pkl").write_bytes(pickle.dumps(deploy_cal))
```

and add `"calibration": "pooled-oos"` to the meta.json dict. (c) After the registry upserts, promote:

```python
    lgbm_id = next(mid for mid, fam in registered if fam == "lgbm")
    promoted = promote_if_better(con, target, lgbm_id,
                                 metrics["lgbm"]["brier"])
```

where `registered` collects `(model_id, fam)` tuples in the save loop, and add `"promoted": promoted` to the returned dict. (d) Append a second reliability table to the report: apply `deploy_cal` to `pooled_raw_lgbm` and render `M.reliability_table` under the heading `## Reliability (deployment calibrator, in-sample on pooled OOS)` — labeled honestly as slightly optimistic.

- [ ] **Step 4: Run tests** — 4 passed; the slow trainer end-to-end still passes (`uv run pytest tests/train -m "" -k trainer` — allow ~15 min, or trust CI); fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/train/trainer.py tests/train/test_promotion.py
git commit -m "feat: pooled-OOS deployment calibrator and challenger promotion guard"
```

---

### Task 2: Event and regime features

**Files:**
- Modify: `vectora/features/engine.py`, `vectora/features/families.py`, `vectora/config/features.yaml`
- Test: append to `tests/features/test_engine.py`

- [ ] **Step 1: Write the failing test (append to tests/features/test_engine.py)**

```python
def test_event_and_regime_base_columns(test_db, tmp_path):
    _seed(test_db, n_days=80)
    vdb.upsert(test_db, "events", [dict(
        id="ev1", post_date="2026-04-20", symbol="AAA",
        title="AAA: Q1 Financials", body="", source="dse_news")])
    vdb.upsert(test_db, "event_labels", [dict(
        event_id="ev1", event_type="earnings_release", materiality=3)])
    vdb.upsert(test_db, "regimes", [dict(
        date="2026-05-10", regime="Bull", confidence=0.8, method="rules")])
    df = engine.compute(test_db, out_path=tmp_path / "f.parquet")
    import datetime as dt
    row = df.filter((pl.col("symbol") == "AAA")
                    & (pl.col("date") == dt.date(2026, 4, 25)))
    assert row["days_since_event"][0] == 5
    assert row["board_meeting_soon"][0] == 0
    before = df.filter((pl.col("symbol") == "AAA")
                       & (pl.col("date") == dt.date(2026, 4, 10)))
    assert before["days_since_event"][0] is None    # no event yet
    reg = df.filter(pl.col("date") == dt.date(2026, 5, 10))
    assert (reg["regime_code"] == 5).all()          # Bull -> 5
    unclass = df.filter(pl.col("date") == dt.date(2026, 4, 25))
    assert (unclass["regime_code"] == 0).all()      # unclassified -> 0
```

- [ ] **Step 2: Run to verify failure** — FAIL (columns missing).

- [ ] **Step 3: Implement**

In `vectora/features/engine.py`, replace `compute` with:

```python
REGIME_CODES = {"Panic": 1, "Bear": 2, "LowLiquidity": 3, "Sideways": 4,
                "Bull": 5, "Recovery": 6, "SpeculativeHeat": 7}


def compute(con, out_path: Path = DEFAULT_OUT,
            specs: list | None = None) -> pl.DataFrame:
    panel = base.load_panel(con)
    meta = con.execute(
        "SELECT symbol, sector, first_seen FROM symbols").pl().with_columns(
        pl.col("first_seen").cast(pl.Date))
    df = panel.join(meta, on="symbol", how="left").sort(["symbol", "date"])

    # event base columns (leakage-safe: post_date <= date by construction —
    # the last-event date is forward-filled from strictly past/same-day rows)
    events = con.execute(
        """
        SELECT e.symbol, e.post_date AS date,
               max(CASE WHEN l.materiality >= 3 THEN 1 ELSE 0 END) AS mat3,
               max(CASE WHEN l.event_type = 'board_meeting' THEN 1 ELSE 0 END)
                   AS bm
        FROM events e JOIN event_labels l ON l.event_id = e.id
        WHERE e.symbol IS NOT NULL GROUP BY 1, 2
        """).pl()
    if events.height > 0:
        df = df.join(events, on=["symbol", "date"], how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Int32).alias("mat3"),
                             pl.lit(None, dtype=pl.Int32).alias("bm"))
    last_ev = (pl.when(pl.col("mat3") == 1).then(pl.col("date"))
               .otherwise(None).forward_fill().over("symbol"))
    last_bm = (pl.when(pl.col("bm") == 1).then(pl.col("date"))
               .otherwise(None).forward_fill().over("symbol"))
    df = df.with_columns(
        (pl.col("date") - last_ev).dt.total_days()
        .alias("days_since_event"),
        ((pl.col("date") - last_bm).dt.total_days() <= 3)
        .cast(pl.Int8).fill_null(0).alias("board_meeting_soon"),
    ).drop(["mat3", "bm"])

    # regime code (market-wide, per date)
    regimes = con.execute("SELECT date, regime FROM regimes").pl()
    if regimes.height > 0:
        regimes = regimes.with_columns(
            pl.col("regime").replace_strict(REGIME_CODES, default=0)
            .alias("regime_code")).drop("regime")
        df = df.join(regimes, on="date", how="left")
        df = df.with_columns(pl.col("regime_code").fill_null(0))
    else:
        df = df.with_columns(pl.lit(0).alias("regime_code"))

    for spec in (specs or registry.load()):
        df = families.apply(df, spec.name, spec.fn, spec.params)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    return df
```

In `vectora/features/families.py` add (and register in `FNS`):

```python
def col_passthrough(df, name, col):
    """Expose an engine-prepared base column under its registry name."""
    if name == col:
        return df
    return df.with_columns(pl.col(col).alias(name))
```

Append to `vectora/config/features.yaml`:

```yaml
  # ---- event/regime (spec §8 event family; added Phase 5A with data to stand on)
  - {name: days_since_event, family: calendar, fn: col_passthrough,
     params: {col: days_since_event},
     reasoning: announcement-driven drift persists for days where information diffuses slowly}
  - {name: board_meeting_soon, family: calendar, fn: col_passthrough,
     params: {col: board_meeting_soon},
     reasoning: a board meeting notice under LR 16(1) means dividends or earnings land within days}
  - {name: regime_code, family: cross_sectional, fn: col_passthrough,
     params: {col: regime_code},
     reasoning: the same setup carries different odds in Panic versus Bull markets per spec regime gating}
```

Note `days_since_event`/`board_meeting_soon`/`regime_code` base columns share names with their features — `col_passthrough` no-ops in that case (kept as explicit registry entries so models, SHAP names, and documentation see them).

- [ ] **Step 4: Run tests** — the new engine test + leakage guard + full features suite pass; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/features vectora/config/features.yaml tests/features/test_engine.py
git commit -m "feat: event recency and regime features join the registry"
```

---

### Task 3: Evaluation module with error autopsy

**Files:**
- Modify: `vectora/db.py` (SCHEMA), `vectora/__main__.py`
- Create: `vectora/evaluate/__init__.py` (empty), `vectora/evaluate/report.py`
- Test: `tests/evaluate/__init__.py` (empty), `tests/evaluate/test_report.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/evaluate/test_report.py
from vectora import db as vdb
from vectora.evaluate import report


def _pred(symbol, d, prob, target="g5_h10"):
    return dict(id=f"{d}_{target}_{symbol}", symbol=symbol, date=d,
                target=target, probability=prob, model_id="m",
                quality_score=100, is_signal=prob >= 0.55,
                suppressed_reason=None)


def _outcome(pid, hit, rmax=0.06, rmin=-0.02):
    return dict(prediction_id=pid, realized_max=rmax, realized_min=rmin,
                hit=hit)


def _seed(con):
    vdb.upsert(con, "symbols", [
        dict(symbol=s, name=None, sector="Bank", instrument_type="Equity",
             category=c, listing_status="active", first_seen="2020-01-01",
             last_seen="2026-12-31")
        for s, c in (("AAA", "A"), ("ZZZ", "Z"))])
    vdb.upsert(con, "regimes", [dict(
        date="2026-06-01", regime="Sideways", confidence=0.5, method="rules")])
    vdb.upsert(con, "predictions", [
        _pred("AAA", "2026-06-01", 0.70), _pred("ZZZ", "2026-06-01", 0.60),
        _pred("AAA", "2026-06-02", 0.30)])
    vdb.upsert(con, "outcomes", [
        _outcome("2026-06-01_g5_h10_AAA", True),
        _outcome("2026-06-01_g5_h10_ZZZ", False, rmax=0.01),
        _outcome("2026-06-02_g5_h10_AAA", False, rmax=0.02)])
    vdb.upsert(con, "risk_blocks", [dict(
        prediction_id="2026-06-01_g5_h10_ZZZ", vol_21d=0.02, expected_up=0.05,
        expected_down=-0.03, rr_ratio=1.6, exit_days=9.0,
        analog_max_drawdown=-0.1, analog_hit_rate=0.5, analog_n=20,
        category="Z", liquidity_value_mn=0.3)])


def test_evaluate_metrics_and_report(test_db, tmp_path):
    _seed(test_db)
    result = report.evaluate(test_db, reports_dir=tmp_path,
                             vault_dir=tmp_path / "vault")
    assert result["resolved"] == 3
    assert 0 < result["targets"]["g5_h10"]["brier"] < 1
    assert abs(result["targets"]["g5_h10"]["hit_rate"] - 1 / 3) < 1e-9
    files = list(tmp_path.glob("eval_*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "g5_h10" in text and "Brier" in text and "Sideways" in text
    assert (tmp_path / "vault" / "Evaluations").exists()


def test_autopsy_tags_misses(test_db, tmp_path):
    _seed(test_db)
    # ZZZ miss: exit_days 9 -> liquidity tag; AAA 06-02 miss: no risk row,
    # no event, no regime shift -> model-error
    vdb.upsert(test_db, "events", [dict(
        id="evx", post_date="2026-06-03", symbol="AAA",
        title="AAA: Q1 Financials", body="", source="dse_news")])
    vdb.upsert(test_db, "event_labels", [dict(
        event_id="evx", event_type="earnings_release", materiality=3)])
    report.evaluate(test_db, reports_dir=tmp_path,
                    vault_dir=tmp_path / "vault")
    tags = dict(test_db.execute(
        "SELECT prediction_id, tag FROM outcome_tags").fetchall())
    assert tags["2026-06-01_g5_h10_ZZZ"] == "liquidity"
    assert tags["2026-06-02_g5_h10_AAA"] == "event-shock"  # event inside window


def test_no_outcomes_graceful(test_db, tmp_path):
    result = report.evaluate(test_db, reports_dir=tmp_path,
                             vault_dir=tmp_path / "vault")
    assert result == {"resolved": 0}
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

Append to `SCHEMA` in `vectora/db.py`:

```sql
CREATE TABLE IF NOT EXISTS outcome_tags (
    prediction_id TEXT PRIMARY KEY, tag TEXT,
    tagged_at TIMESTAMP DEFAULT current_timestamp
);
```

Create `vectora/evaluate/report.py`:

```python
"""Calibration accounting + error autopsy (spec §17.2-17.3).

Grades every resolved prediction: Brier, hit rate, reliability bins per
target, segmented by regime and category. Every miss gets a rule-based
cause tag (first match wins): event-shock (materiality-3 event inside the
horizon window), liquidity (exit_days > LIQ_DAYS), regime-shift (regime
at prediction date differs from any regime inside the window), else
model-error. Tags accumulate in outcome_tags for Phase 6 pattern notes.
"""
import datetime as dt
import re
from pathlib import Path

from vectora import db as vdb
from vectora.settings import REPORTS_DIR, VAULT_DIR
from vectora.train.models import brier as brier_score
from vectora.train.models import reliability_table
from vectora.vault.generator import _write_machine

LIQ_DAYS = 5.0
_H_RE = re.compile(r"_h(\d+)$")


def evaluate(con, reports_dir: Path = REPORTS_DIR,
             vault_dir: Path = VAULT_DIR) -> dict:
    rows = con.execute(
        """
        SELECT p.id, p.symbol, p.date, p.target, p.probability,
               o.hit, r.exit_days,
               coalesce(g.regime, 'unclassified') AS regime,
               coalesce(s.category, '?') AS category
        FROM predictions p
        JOIN outcomes o ON o.prediction_id = p.id
        LEFT JOIN risk_blocks r ON r.prediction_id = p.id
        LEFT JOIN regimes g ON g.date = p.date
        LEFT JOIN symbols s ON s.symbol = p.symbol
        """).fetchall()
    if not rows:
        return {"resolved": 0}

    targets: dict = {}
    seg_lines = []
    for tgt in sorted({r[3] for r in rows}):
        sub = [r for r in rows if r[3] == tgt]
        ys = [int(r[5]) for r in sub]
        ps = [float(r[4]) for r in sub]
        targets[tgt] = {
            "n": len(sub), "hit_rate": sum(ys) / len(ys),
            "brier": brier_score(ys, ps),
            "reliability": reliability_table(ys, ps),
        }
        for seg_idx, seg_name in ((7, "regime"), (8, "category")):
            for val in sorted({r[seg_idx] for r in sub}):
                seg = [r for r in sub if r[seg_idx] == val]
                if len(seg) < 5:
                    continue
                hr = sum(int(r[5]) for r in seg) / len(seg)
                seg_lines.append(
                    f"| {tgt} | {seg_name}={val} | {len(seg)} | {hr:.0%} |")

    tags = []
    for pid, symbol, d, tgt, _p, hit, exit_days, regime, _cat in rows:
        if hit:
            continue
        h = int(_H_RE.search(tgt).group(1)) if _H_RE.search(tgt) else 10
        end = (d + dt.timedelta(days=int(h * 1.6))).isoformat()
        ev = con.execute(
            """
            SELECT 1 FROM events e JOIN event_labels l ON l.event_id = e.id
            WHERE e.symbol = ? AND l.materiality >= 3
              AND e.post_date > ? AND e.post_date <= ? LIMIT 1
            """, [symbol, str(d), end]).fetchone()
        if ev:
            tag = "event-shock"
        elif exit_days is not None and exit_days > LIQ_DAYS:
            tag = "liquidity"
        else:
            shift = con.execute(
                "SELECT 1 FROM regimes WHERE date > ? AND date <= ? "
                "AND regime <> ? LIMIT 1", [str(d), end, regime]).fetchone()
            tag = "regime-shift" if shift else "model-error"
        tags.append({"prediction_id": pid, "tag": tag})
    if tags:
        vdb.upsert(con, "outcome_tags", tags)

    today = dt.date.today().isoformat()
    lines = [f"# Evaluation {today}", "",
             f"{len(rows)} resolved predictions", ""]
    for tgt, m in targets.items():
        lines += [f"## {tgt}", "",
                  f"n={m['n']} | hit rate {m['hit_rate']:.0%} | "
                  f"Brier {m['brier']:.4f}", "",
                  "| bin | n | predicted | realized |", "|---|---|---|---|"]
        lines += [f"| {b['bin_lo']:.1f}-{b['bin_hi']:.1f} | {b['n']} "
                  f"| {b['p_mean']:.3f} | {b['y_rate']:.3f} |"
                  for b in m["reliability"]]
        lines.append("")
    if seg_lines:
        lines += ["## Segments (n>=5)", "",
                  "| target | segment | n | hit rate |", "|---|---|---|---|"]
        lines += seg_lines
    tag_counts = con.execute(
        "SELECT tag, count(*) FROM outcome_tags GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    if tag_counts:
        lines += ["", "## Miss autopsy", ""]
        lines += [f"- {t}: {n}" for t, n in tag_counts]
    body = "\n".join(lines) + "\n"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"eval_{today}.md").write_text(body, encoding="utf-8")
    _write_machine(Path(vault_dir) / "Evaluations" / f"{today}.md", body)

    return {"resolved": len(rows), "targets": targets,
            "misses_tagged": len(tags)}
```

Add the `evaluate` CLI stage in `vectora/__main__.py` (choices gain `"evaluate"`):

```python
    if args.command == "run" and args.stage == "evaluate":
        from vectora import db as vdb
        from vectora.evaluate import report
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = report.evaluate(con)
        finally:
            con.close()
        print(json.dumps({k: v for k, v in result.items()
                          if k != "targets"} | {
            "targets": {t: {k2: v2 for k2, v2 in m.items()
                            if k2 != "reliability"}
                        for t, m in result.get("targets", {}).items()}},
              indent=1, default=str))
        return 0
```

- [ ] **Step 4: Run tests** — 3 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/db.py vectora/evaluate vectora/__main__.py tests/evaluate
git commit -m "feat: evaluation reports with per-segment calibration and miss autopsy"
```

---

### Task 4: Weekly retrain schedule + retrain both targets for real

- [ ] **Step 1: Workflow** — in `.github/workflows/train.yml`: add under `on:`:

```yaml
  schedule:
    # Friday 16:00 Dhaka = 10:00 UTC; market closed, weekly retrain (spec §17.5)
    - cron: "0 10 * * 5"
```

and give the dispatch input a default that also works for cron (the Train step already falls back): change the Train step run line to:

```yaml
        run: uv run python -m vectora run train --target "${{ github.event.inputs.target || 'g5_h10' }}"
```

then add a second step after it for the other target plus evaluation:

```yaml
      - name: Train g10_h30
        continue-on-error: true
        run: uv run python -m vectora run train --target g10_h30
      - name: Evaluate
        continue-on-error: true
        run: uv run python -m vectora run evaluate
```

- [ ] **Step 2: Fast suite + ruff + commit workflow**

```bash
uv run pytest -m "not slow" && uv run ruff check .
git add .github/workflows/train.yml
git commit -m "feat: weekly Friday retrain with dual targets and evaluation"
```

- [ ] **Step 3: REAL retrain, both targets (long — run in background, ~15 min each)**

```bash
uv run python -m vectora run train --target g5_h10
uv run python -m vectora run train --target g10_h30
```

Both should report `"promoted": true` (pooled-OOS calibrated challengers beating fold-calibrated incumbents' Brier is expected but not guaranteed — if a challenger loses, that's the guard working; report it and do NOT force-promote). Then inspect the new reports' deployment-calibrator reliability tables.

- [ ] **Step 4: g10_h30 enablement decision (pre-registered criterion)**

Enable g10_h30 signals IF its deployment-calibrator table shows |realized − predicted| ≤ 0.15 for every bin with bin_lo ≥ 0.6 and n ≥ 300. If enabled: in `vectora/settings.py` set `SIGNAL_THRESHOLDS = {"g5_h10": 0.55, "g10_h30": 0.60}`; the `g10_h30-tail-overconfident` branch in `_gate` then becomes unreachable for it (leave the code — it self-documents the history; the `target not in SIGNAL_THRESHOLDS` check is the mechanism). Run `uv run pytest -m "not slow"` — the engine test asserting g10_h30 suppression (`test_z_category_never_signals` is symbol-based, unaffected; check `tests/predict/test_engine.py` for any g10_h30-suppression assertion and update ONLY if the criterion passed, documenting the report numbers in the commit message). If the criterion fails: leave settings unchanged, record the numbers in the commit message, and the gate stays.

- [ ] **Step 5: Commit artifacts, merge, push, dispatch verification**

```bash
git add -f models reports data/vectora.duckdb vectora/settings.py
git commit -m "train: pooled-OOS recalibrated models for both targets; promotion + g10_h30 decision per criterion"
git checkout main && git pull
git merge --no-ff phase-5a-loop -m "Merge phase-5a: learning loop - recalibration, event/regime features, evaluation, weekly retrain"
git push
& "C:\Program Files\GitHub CLI\gh.exe" workflow run train --ref main
```

Watch the train workflow to green (~30-45 min on the runner; both targets + evaluate).

---

## Execution notes

- Order 1→4. Task 1's trainer changes and Task 2's features both feed Task 4's real retrain — the retrained models automatically include the new features (feat_names comes from the registry at train time) and the pooled calibrator.
- Expected fast suite ≈ 199 tests. The evaluation module runs on 0 resolved outcomes until ~Jul 30 — by design.
- After 5A, only 5B remains for full production: health.yml watchdog, README + runbook, corporate-action engine (upgrade point vectora/features/base.py).
