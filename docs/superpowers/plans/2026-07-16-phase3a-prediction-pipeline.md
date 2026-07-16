# Vectora Phase 3A: Daily Prediction Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every trading day, turn the active calibrated models into per-symbol probabilities with risk blocks, admission-gated signals, and SHAP+analog explanations — persisted to DuckDB and wired into the scheduled pipeline (spec §9.3, §14, §15).

**Architecture:** A `vectora/predict/` package orchestrated by `predict.engine.run_predict(con, date)`: recompute features (seconds), score the tradable universe with each active model from `model_registry`, attach a risk block built from analog-retrieval statistics (k-NN in standardized feature space over labeled history), extract per-prediction feature drivers via LightGBM's built-in TreeSHAP (`pred_contrib=True` — no new dependency), render a templated explanation, and upsert everything into three new tables. Signal admission applies the spec's gates: calibrated probability threshold, data-quality floor, liquidity universe membership, and a Z-category suppression. Phase 3B consumes these tables for alerts and the vault.

**Tech Stack:** existing stack only (polars, LightGBM, scikit-learn NearestNeighbors, DuckDB). No new dependencies.

**Deliberate Phase 3A decisions:**
- `g10_h30` predictions are computed and stored but **never admitted as signals** (`suppressed_reason='g10_h30-tail-overconfident'`) — its calibration tail claims ~0.9 where reality delivers ~0.5 (training report 2026-07-16). Recalibration is Phase 5's evaluation loop.
- Z-category symbols get predictions but signals are suppressed (`'z-category-gate'`); alert-tier capping arrives with alerts in 3B.
- Expected move sizes come from analog realized outcomes (median forward max-gain / max-loss of the 20 nearest historical neighbours), not conformal regression — simpler, explainable, and honest about being empirical.

**Existing contracts this plan builds on:**
- `vectora/db.py`: `connect`, `init_schema(con, backfill_parquet=None)` (creates `prices` view), `upsert`, watermarks; `model_registry(model_id, family, target, trained_at, train_end, metrics, artifact_dir, active)`.
- `vectora/features/engine.py`: `compute(con, out_path=None, specs=None) -> pl.DataFrame` — full panel with `symbol, date, ret, sector, first_seen` + all 40 registry features. ~4s on full history.
- `vectora/features/registry.py`: `load() -> list[FeatureSpec]` (`.name`, `.reasoning`).
- `vectora/labels.py`: `make_labels(panel, thresholds, horizons, downside=False)`.
- `vectora/universe.py`: `tradable_universe(con, as_of, min_median_value_mn=1.0) -> list[str]`.
- `vectora/train/models.py`: `fit_lgbm`, `fit_calibrator`, `apply_calibrator`, `predict`.
- `vectora/settings.py`: `MODELS_DIR`, `MIN_QUALITY_SCORE`, `DB_PATH`.
- Artifacts on disk: `models/g5_h10_lgbm_2026-07-16_c09374/{lgbm.txt,calibrator.pkl,meta.json}` and the g10_h30 equivalent; meta.json has `features` (ordered list) and `target`.
- `symbols.category` ('A'/'B'/'N'/'Z'/NULL); `data_quality(date, source, score, issues)`.
- Run everything with `uv run …` from repo root, branch `phase-3a-prediction` off `main` (`git checkout main && git pull && git checkout -b phase-3a-prediction`). Fast tests: `uv run pytest -m "not slow"`.

**File structure:**

```
vectora/predict/
├── __init__.py
├── loaders.py        # active_model(), load_artifacts() from model_registry
├── analogs.py        # k-NN analog retrieval + realized-outcome stats
├── risk.py           # risk block per prediction
├── explain.py        # TreeSHAP drivers + templated rendering
└── engine.py         # run_predict(): orchestrate, gate, persist
vectora/settings.py   # + SIGNAL_THRESHOLDS, ANALOG_K, POSITION_TAKA
vectora/db.py         # + predictions / risk_blocks / explanations tables
vectora/labels.py     # + continuous fwd_max/fwd_min columns option
vectora/__main__.py   # + `run predict` stage
.github/workflows/eod-pipeline.yml  # + predict step
tests/predict/…       # one test file per module
```

---

### Task 1: Prediction tables + settings + model activation

**Files:**
- Modify: `vectora/db.py` (SCHEMA), `vectora/settings.py`
- Create: `vectora/predict/__init__.py` (empty), `vectora/predict/loaders.py`
- Test: `tests/predict/__init__.py` (empty), `tests/predict/test_loaders.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/predict/test_loaders.py
import json

import pytest

from vectora import db as vdb
from vectora.predict import loaders


def test_prediction_tables_exist(test_db):
    tables = {r[0] for r in test_db.execute("SHOW TABLES").fetchall()}
    assert {"predictions", "risk_blocks", "explanations"} <= tables


def _register(con, tmp_path, target="g5_h10", active=True, features=("ret_1d",)):
    art = tmp_path / f"{target}_lgbm_test"
    art.mkdir(parents=True, exist_ok=True)
    (art / "meta.json").write_text(json.dumps(
        {"model_id": f"{target}_lgbm_test", "family": "lgbm",
         "target": target, "features": list(features)}), encoding="utf-8")
    vdb.upsert(con, "model_registry", [{
        "model_id": f"{target}_lgbm_test", "family": "lgbm", "target": target,
        "trained_at": "2026-07-16T00:00:00", "train_end": "2026-01-01",
        "metrics": "{}", "artifact_dir": str(art), "active": active,
    }])
    return art


def test_active_model_returns_meta(test_db, tmp_path):
    _register(test_db, tmp_path)
    m = loaders.active_model(test_db, "g5_h10")
    assert m["model_id"] == "g5_h10_lgbm_test"
    assert m["features"] == ["ret_1d"]
    assert m["artifact_dir"].endswith("g5_h10_lgbm_test")


def test_active_model_none_when_inactive(test_db, tmp_path):
    _register(test_db, tmp_path, active=False)
    assert loaders.active_model(test_db, "g5_h10") is None


def test_active_model_latest_wins(test_db, tmp_path):
    _register(test_db, tmp_path)
    art2 = tmp_path / "newer"
    art2.mkdir()
    (art2 / "meta.json").write_text(json.dumps(
        {"model_id": "newer", "family": "lgbm", "target": "g5_h10",
         "features": ["ret_1d", "ret_3d"]}), encoding="utf-8")
    vdb.upsert(test_db, "model_registry", [{
        "model_id": "newer", "family": "lgbm", "target": "g5_h10",
        "trained_at": "2026-07-17T00:00:00", "train_end": "2026-02-01",
        "metrics": "{}", "artifact_dir": str(art2), "active": True,
    }])
    assert loaders.active_model(test_db, "g5_h10")["model_id"] == "newer"


def test_load_artifacts_missing_dir_raises(test_db, tmp_path):
    _register(test_db, tmp_path)
    m = loaders.active_model(test_db, "g5_h10")
    with pytest.raises(FileNotFoundError):
        loaders.load_artifacts(m)  # no lgbm.txt/calibrator.pkl written
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/predict -v` → FAIL (tables/module missing). Create the empty `tests/predict/__init__.py` and `vectora/predict/__init__.py` first.

- [ ] **Step 3: Implement**

Append to `SCHEMA` in `vectora/db.py`:

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,               -- <date>_<target>_<symbol>
    created_at TIMESTAMP DEFAULT current_timestamp,
    symbol TEXT, date DATE, target TEXT,
    probability DOUBLE, model_id TEXT, quality_score INTEGER,
    is_signal BOOLEAN, suppressed_reason TEXT
);
CREATE TABLE IF NOT EXISTS risk_blocks (
    prediction_id TEXT PRIMARY KEY,
    vol_21d DOUBLE, expected_up DOUBLE, expected_down DOUBLE,
    rr_ratio DOUBLE, exit_days DOUBLE, analog_max_drawdown DOUBLE,
    analog_hit_rate DOUBLE, analog_n INTEGER,
    category TEXT, liquidity_value_mn DOUBLE
);
CREATE TABLE IF NOT EXISTS explanations (
    prediction_id TEXT PRIMARY KEY,
    drivers TEXT,      -- JSON list of {feature, contribution, value}
    analogs TEXT,      -- JSON {hit_rate, n, median_up, median_down}
    rendered TEXT
);
```

Append to `vectora/settings.py`:

```python
# Signal admission (spec §9.3). g10_h30 excluded: overconfident tail
# (training report 2026-07-16) until Phase 5 recalibration.
SIGNAL_THRESHOLDS = {"g5_h10": 0.55}
ANALOG_K = 20
POSITION_TAKA = 500_000  # assumed position size for exit-days liquidity risk
```

Create `vectora/predict/loaders.py`:

```python
"""Load active models from the registry with their on-disk artifacts."""
import json
import pickle
from pathlib import Path

import lightgbm as lgb


def active_model(con, target: str) -> dict | None:
    row = con.execute(
        """
        SELECT model_id, artifact_dir FROM model_registry
        WHERE target = ? AND family = 'lgbm' AND active
        ORDER BY trained_at DESC LIMIT 1
        """, [target]).fetchone()
    if row is None:
        return None
    meta = json.loads(
        (Path(row[1]) / "meta.json").read_text(encoding="utf-8"))
    return {"model_id": row[0], "artifact_dir": row[1],
            "target": target, "features": meta["features"]}


def load_artifacts(model: dict):
    """Returns (booster, calibrator)."""
    art = Path(model["artifact_dir"])
    booster_path, cal_path = art / "lgbm.txt", art / "calibrator.pkl"
    if not booster_path.exists() or not cal_path.exists():
        raise FileNotFoundError(f"artifacts missing in {art}")
    booster = lgb.Booster(model_file=str(booster_path))
    calibrator = pickle.loads(cal_path.read_bytes())
    return booster, calibrator
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/predict -v` → 5 passed; fast suite `uv run pytest -m "not slow"` all pass; `uv run ruff check .` clean.

- [ ] **Step 5: Activate the two real models (real-DB step)**

Run: `uv run python -c "
from vectora import db as vdb
con = vdb.connect('data/vectora.duckdb'); vdb.init_schema(con)
con.execute(\"UPDATE model_registry SET active = true WHERE family = 'lgbm'\")
print(con.execute('SELECT model_id, active FROM model_registry ORDER BY model_id').fetchall())
con.close()"`
Expected: both lgbm rows show active=True (logistic rows stay inactive).

- [ ] **Step 6: Commit**

```bash
git add vectora/db.py vectora/settings.py vectora/predict tests/predict data/vectora.duckdb
git commit -m "feat: prediction tables, signal settings, active-model loaders"
```

---

### Task 2: Continuous forward-outcome columns in labels

**Files:**
- Modify: `vectora/labels.py`
- Test: append to `tests/test_labels.py`

- [ ] **Step 1: Write the failing test (append to tests/test_labels.py)**

```python
def test_continuous_forward_outcomes():
    df = labels.make_labels(_panel(), thresholds=(0.05,), horizons=(3,),
                            continuous=True)
    row0 = df.filter(pl.col("date") == dt.date(2026, 1, 1))  # close 100
    # next 3 closes: 101,103,111 -> max +11%, min +1%
    assert abs(row0["fwdmax_h3"][0] - 0.11) < 1e-9
    assert abs(row0["fwdmin_h3"][0] - 0.01) < 1e-9
    # incomplete horizon -> null
    assert df.sort("date").tail(3)["fwdmax_h3"].null_count() == 3
```

Also add the imports the test needs if missing (`import datetime as dt`, `import polars as pl` are already at the top of the file).

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_labels.py -v` → FAIL (unexpected keyword `continuous`).

- [ ] **Step 3: Implement** — in `vectora/labels.py`, change `make_labels` signature and add the columns:

```python
def make_labels(panel: pl.DataFrame, thresholds=(0.03, 0.05, 0.10, 0.20),
                horizons=(1, 3, 5, 10, 30), downside: bool = False,
                continuous: bool = False) -> pl.DataFrame:
    df = panel.sort(["symbol", "date"])
    cols = []
    for h in horizons:
        fwd_max = _fwd_extreme(h, "max")
        fwd_min = _fwd_extreme(h, "min")
        if continuous:
            cols.append((fwd_max / pl.col("close") - 1).alias(f"fwdmax_h{h}"))
            cols.append((fwd_min / pl.col("close") - 1).alias(f"fwdmin_h{h}"))
        for x in thresholds:
            pct = round(x * 100)
            cols.append(
                (fwd_max / pl.col("close") - 1 >= x)
                .cast(pl.Int8).alias(f"y_g{pct}_h{h}"))
            if downside:
                cols.append(
                    (fwd_min / pl.col("close") - 1 <= -x)
                    .cast(pl.Int8).alias(f"y_d{pct}_h{h}"))
    return df.with_columns(cols)
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_labels.py -v` → 5 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/labels.py tests/test_labels.py
git commit -m "feat: continuous forward max/min outcome columns in label grid"
```

---

### Task 3: Analog retrieval

**Files:**
- Create: `vectora/predict/analogs.py`
- Test: `tests/predict/test_analogs.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/predict/test_analogs.py
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
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# vectora/predict/analogs.py
"""Historical-analog retrieval (spec §15): k nearest labeled situations in
standardized feature space, summarized by their realized outcomes. This is
both the explanation ingredient ("12 of 20 similar setups hit the target")
and the risk engine's empirical move-size estimate."""
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class AnalogIndex:
    def __init__(self, nn, imputer, scaler, outcomes: np.ndarray):
        self._nn = nn
        self._imputer = imputer
        self._scaler = scaler
        # outcomes columns: label, fwdmax, fwdmin
        self._outcomes = outcomes
        self.n_rows = len(outcomes)

    @classmethod
    def fit(cls, history: pl.DataFrame, feature_names: list[str],
            label_col: str, fwdmax_col: str, fwdmin_col: str) -> "AnalogIndex":
        usable = history.filter(
            pl.col(label_col).is_not_null()
            & pl.col(fwdmax_col).is_not_null()
            & pl.col(fwdmin_col).is_not_null())
        X = usable.select(feature_names).to_numpy().astype(np.float64)
        imputer = SimpleImputer(strategy="median").fit(X)
        scaler = StandardScaler().fit(imputer.transform(X))
        Xs = scaler.transform(imputer.transform(X))
        nn = NearestNeighbors(n_neighbors=50).fit(Xs)
        outcomes = usable.select(
            [label_col, fwdmax_col, fwdmin_col]).to_numpy().astype(np.float64)
        return cls(nn, imputer, scaler, outcomes)

    def query(self, x: np.ndarray, k: int = 20) -> dict:
        k = min(k, self.n_rows)
        xs = self._scaler.transform(
            self._imputer.transform(x.reshape(1, -1)))
        _, idx = self._nn.kneighbors(xs, n_neighbors=k)
        o = self._outcomes[idx[0]]
        return {
            "hit_rate": float(o[:, 0].mean()),
            "median_up": float(np.median(o[:, 1])),
            "median_down": float(np.median(o[:, 2])),
            "max_drawdown": float(o[:, 2].min()),
            "n": int(k),
        }
```

Debug note: `SimpleImputer.transform` on a query row containing NaN uses the medians learned at fit time. If a whole feature column was all-NaN at fit (possible: liquidity features in backfill era), `SimpleImputer` drops it and prints a warning — the scaler/nn were fit on the SAME transformed matrix so shapes stay consistent; the query path goes through the same imputer so it stays consistent too. Do not "fix" the warning.

- [ ] **Step 4: Run tests** — 3 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/predict/analogs.py tests/predict/test_analogs.py
git commit -m "feat: k-NN historical analog retrieval with realized-outcome stats"
```

---

### Task 4: Risk blocks

**Files:**
- Create: `vectora/predict/risk.py`
- Test: `tests/predict/test_risk.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/predict/test_risk.py
from vectora.predict import risk


def _analog_stats(hit=0.6, up=0.08, down=-0.04, mdd=-0.15, n=20):
    return {"hit_rate": hit, "median_up": up, "median_down": down,
            "max_drawdown": mdd, "n": n}


def test_risk_block_fields_and_rr():
    b = risk.build(vol_21d=0.02, value_mn_med_21d=5.0, category="A",
                   analog_stats=_analog_stats())
    assert b["expected_up"] == 0.08 and b["expected_down"] == -0.04
    assert abs(b["rr_ratio"] - 2.0) < 1e-9
    assert b["analog_max_drawdown"] == -0.15
    assert b["category"] == "A"
    # 500k position vs 20% of 5mn/day absorbable -> 0.5 days
    assert abs(b["exit_days"] - 0.5) < 1e-9


def test_rr_ratio_none_when_downside_zero():
    b = risk.build(vol_21d=0.02, value_mn_med_21d=5.0, category="A",
                   analog_stats=_analog_stats(down=0.0))
    assert b["rr_ratio"] is None


def test_illiquid_name_has_long_exit():
    b = risk.build(vol_21d=0.05, value_mn_med_21d=0.05, category="Z",
                   analog_stats=_analog_stats())
    assert b["exit_days"] == 50.0  # 500k / (0.2 * 50k/day)


def test_missing_liquidity_yields_none_exit():
    b = risk.build(vol_21d=0.02, value_mn_med_21d=None, category="B",
                   analog_stats=_analog_stats())
    assert b["exit_days"] is None
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/predict/risk.py
"""Risk block per prediction (spec §14). Empirical where possible:
expected move sizes come from analog realized outcomes; liquidity risk
assumes a POSITION_TAKA position unwound at <=20% of median daily value
(more than that and you ARE the market on a thin DSE book)."""
from vectora.settings import POSITION_TAKA

ABSORBABLE_SHARE = 0.20


def build(vol_21d: float | None, value_mn_med_21d: float | None,
          category: str | None, analog_stats: dict) -> dict:
    expected_up = analog_stats["median_up"]
    expected_down = analog_stats["median_down"]
    rr = None
    if expected_down and expected_down < 0:
        rr = expected_up / abs(expected_down)
    exit_days = None
    if value_mn_med_21d and value_mn_med_21d > 0:
        absorbable_taka_per_day = ABSORBABLE_SHARE * value_mn_med_21d * 1e6
        exit_days = POSITION_TAKA / absorbable_taka_per_day
    return {
        "vol_21d": vol_21d,
        "expected_up": expected_up,
        "expected_down": expected_down,
        "rr_ratio": rr,
        "exit_days": exit_days,
        "analog_max_drawdown": analog_stats["max_drawdown"],
        "analog_hit_rate": analog_stats["hit_rate"],
        "analog_n": analog_stats["n"],
        "category": category,
        "liquidity_value_mn": value_mn_med_21d,
    }
```

- [ ] **Step 4: Run tests** — 4 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/predict/risk.py tests/predict/test_risk.py
git commit -m "feat: risk blocks from analog outcomes and liquidity exit model"
```

---

### Task 5: Drivers + explanation rendering

**Files:**
- Create: `vectora/predict/explain.py`
- Test: `tests/predict/test_explain.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/predict/test_explain.py
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
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/predict/explain.py
"""Per-prediction explanation (spec §15): TreeSHAP drivers via LightGBM's
pred_contrib (no extra dependency), analog evidence, and templated
uncertainty warnings. Deterministic text, fully audit-traceable to numbers."""
import numpy as np

EXIT_DAYS_WARN = 3.0


def drivers(booster, x: np.ndarray, feature_names: list[str],
            top: int = 6) -> list[dict]:
    contrib = booster.predict(x.reshape(1, -1), pred_contrib=True)[0]
    # last element is the bias term; drop it
    contrib = contrib[:-1]
    order = np.argsort(-np.abs(contrib))[:top]
    return [
        {"feature": feature_names[i],
         "contribution": round(float(contrib[i]), 4),
         "value": None if np.isnan(x[i]) else round(float(x[i]), 4)}
        for i in order
    ]


def render(symbol: str, target: str, probability: float,
           driver_list: list[dict], analog_stats: dict, risk_block: dict,
           quality: int) -> str:
    lines = [
        f"{symbol}: {probability:.0%} calibrated probability of the "
        f"{target} move.",
    ]
    for d in driver_list:
        direction = "supports" if d["contribution"] > 0 else "works against"
        lines.append(
            f"- {d['feature']} = {d['value']} {direction} the setup "
            f"(contribution {d['contribution']:+.3f})")
    hits = round(analog_stats["hit_rate"] * analog_stats["n"])
    lines.append(
        f"Similar past setups: {hits} of {analog_stats['n']} hit the target; "
        f"median outcome +{analog_stats['median_up']:.1%} / "
        f"{analog_stats['median_down']:.1%}; "
        f"worst analog drawdown {analog_stats['max_drawdown']:.1%}.")
    warnings = []
    if risk_block["exit_days"] is None or risk_block["exit_days"] > EXIT_DAYS_WARN:
        warnings.append("thin book - exiting may take days and move the price")
    if risk_block["category"] == "Z":
        warnings.append("Z-category name - governance and settlement risk")
    if quality < 100:
        warnings.append(f"data quality {quality} on the underlying day")
    if warnings:
        lines.append("Warnings: " + "; ".join(warnings) + ".")
    lines.append(
        f"Downside scenario: median analog loss {analog_stats['median_down']:.1%}; "
        "this prediction can fail if market regime shifts or an "
        "unannounced corporate event lands inside the horizon.")
    return "\n".join(lines)
```

Note on `test_render_mentions_key_facts`: "13 of 20" comes from round(0.65×20); "-19.0%" from `max_drawdown:.1%` of −0.19. If a formatting detail fails, adjust the IMPLEMENTATION to satisfy the test.

- [ ] **Step 4: Run tests** — 3 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/predict/explain.py tests/predict/test_explain.py
git commit -m "feat: TreeSHAP drivers and templated explanations with risk warnings"
```

---

### Task 6: Predict engine (orchestration + gates + persistence)

**Files:**
- Create: `vectora/predict/engine.py`
- Test: `tests/predict/test_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/predict/test_engine.py
import datetime as dt
import json
import pickle

import numpy as np

from vectora import db as vdb
from vectora.predict import engine as pengine
from vectora.train import models as M


def _seed_market(con, n_days=140, n_syms=6, seed=11):
    """Enough history for 21/63d features and analog labels."""
    rng = np.random.default_rng(seed)
    vdb.upsert(con, "symbols", [
        dict(symbol=f"S{i}", name=None, sector="Bank",
             instrument_type="Equity", category="Z" if i == 5 else "A",
             listing_status="active", first_seen="2020-01-01",
             last_seen="2026-12-31")
        for i in range(n_syms)])
    rows = []
    d0 = dt.date(2026, 1, 1)
    px = {f"S{i}": 50.0 * (i + 1) for i in range(n_syms)}
    for day in range(n_days):
        d = (d0 + dt.timedelta(days=day)).isoformat()
        for sym in px:
            px[sym] *= float(np.exp(rng.normal(0.0002, 0.015)))
            p = round(max(px[sym], 1.0), 2)
            rows.append(dict(symbol=sym, date=d, open=p, high=p * 1.01,
                             low=p * 0.99, close=p, ltp=p, ycp=p,
                             trades=50, value_mn=float(rng.uniform(2, 10)),
                             volume=int(rng.integers(5000, 20000)),
                             source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)
    last = (d0 + dt.timedelta(days=n_days - 1)).isoformat()
    vdb.upsert(con, "data_quality",
               [{"date": last, "source": "dse_eod", "score": 100,
                 "issues": "[]"}])
    return last


def _train_and_register(con, tmp_path):
    """Tiny real model over the seeded market, saved like trainer does."""
    from vectora.features import engine as fengine, registry
    from vectora import labels as lab
    feat_names = [s.name for s in registry.load()]
    df = fengine.compute(con, out_path=tmp_path / "f.parquet")
    df = lab.make_labels(df, thresholds=(0.05,), horizons=(10,),
                         continuous=True)
    train = df.filter(df["y_g5_h10"].is_not_null())
    X = train.select(feat_names).to_numpy().astype(np.float64)
    y = train["y_g5_h10"].to_numpy().astype(int)
    cut = int(len(y) * 0.8)
    m = M.fit_lgbm(X[:cut], y[:cut], X[cut:], y[cut:])
    cal = M.fit_calibrator(M.predict(m, X[cut:]), y[cut:])
    art = tmp_path / "g5_h10_lgbm_test"
    art.mkdir()
    m.booster_.save_model(str(art / "lgbm.txt"))
    (art / "calibrator.pkl").write_bytes(pickle.dumps(cal))
    (art / "meta.json").write_text(json.dumps(
        {"model_id": "g5_h10_lgbm_test", "family": "lgbm",
         "target": "g5_h10", "features": feat_names}), encoding="utf-8")
    vdb.upsert(con, "model_registry", [{
        "model_id": "g5_h10_lgbm_test", "family": "lgbm", "target": "g5_h10",
        "trained_at": "2026-07-16T00:00:00", "train_end": "2026-05-01",
        "metrics": "{}", "artifact_dir": str(art), "active": True,
    }])


def test_run_predict_persists_all_three_tables(test_db, tmp_path):
    last = _seed_market(test_db)
    _train_and_register(test_db, tmp_path)
    result = pengine.run_predict(test_db, date_str=last,
                                 features_path=tmp_path / "f2.parquet",
                                 min_median_value_mn=0.1)
    assert result["predictions"] == 6            # all six symbols scored
    n_pred = test_db.execute("SELECT count(*) FROM predictions").fetchone()[0]
    n_risk = test_db.execute("SELECT count(*) FROM risk_blocks").fetchone()[0]
    n_expl = test_db.execute("SELECT count(*) FROM explanations").fetchone()[0]
    assert n_pred == n_risk == n_expl == 6
    row = test_db.execute(
        "SELECT probability, quality_score FROM predictions LIMIT 1").fetchone()
    assert 0.0 <= row[0] <= 1.0 and row[1] == 100


def test_z_category_never_signals(test_db, tmp_path):
    last = _seed_market(test_db)
    _train_and_register(test_db, tmp_path)
    pengine.run_predict(test_db, date_str=last,
                        features_path=tmp_path / "f2.parquet",
                        min_median_value_mn=0.1)
    z = test_db.execute(
        "SELECT is_signal, suppressed_reason FROM predictions "
        "WHERE symbol = 'S5'").fetchone()
    assert z[0] is False and z[1] == "z-category-gate"


def test_low_quality_day_suppresses_signals(test_db, tmp_path):
    last = _seed_market(test_db)
    vdb.upsert(test_db, "data_quality",
               [{"date": last, "source": "dse_eod", "score": 60,
                 "issues": "[]"}])
    _train_and_register(test_db, tmp_path)
    pengine.run_predict(test_db, date_str=last,
                        features_path=tmp_path / "f2.parquet",
                        min_median_value_mn=0.1)
    n_signals = test_db.execute(
        "SELECT count(*) FROM predictions WHERE is_signal").fetchone()[0]
    assert n_signals == 0
    reasons = {r[0] for r in test_db.execute(
        "SELECT DISTINCT suppressed_reason FROM predictions "
        "WHERE NOT is_signal").fetchall()}
    assert "quality-below-floor" in reasons


def test_rerun_is_idempotent(test_db, tmp_path):
    last = _seed_market(test_db)
    _train_and_register(test_db, tmp_path)
    pengine.run_predict(test_db, date_str=last,
                        features_path=tmp_path / "f2.parquet",
                        min_median_value_mn=0.1)
    pengine.run_predict(test_db, date_str=last,
                        features_path=tmp_path / "f3.parquet",
                        min_median_value_mn=0.1)
    assert test_db.execute("SELECT count(*) FROM predictions").fetchone()[0] == 6


def test_no_active_model_returns_zero(test_db, tmp_path):
    last = _seed_market(test_db)
    result = pengine.run_predict(test_db, date_str=last,
                                 features_path=tmp_path / "f.parquet",
                                 min_median_value_mn=0.1)
    assert result["predictions"] == 0 and result["targets"] == []
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing). These tests fit a real (small) LightGBM — expect the file to take ~1-2 min total, acceptable without a slow marker.

- [ ] **Step 3: Implement**

```python
# vectora/predict/engine.py
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


def _created_columns_note() -> None:
    """predictions.created_at defaults in SQL; nothing to do in Python."""
```

Remove `_created_columns_note` before committing — it is a sketch artifact (the plan includes it only to be explicit that created_at needs no Python handling).

- [ ] **Step 4: Run tests** — `uv run pytest tests/predict/test_engine.py -v` → 5 passed (~1-2 min); fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/predict/engine.py tests/predict/test_engine.py
git commit -m "feat: daily predict engine with admission gates and persistence"
```

---

### Task 7: CLI + workflow integration + first real prediction run

**Files:**
- Modify: `vectora/__main__.py`, `.github/workflows/eod-pipeline.yml`
- Test: append to `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing test (append to tests/test_orchestrator.py)**

```python
def test_cli_predict_dispatch(monkeypatch, capsys):
    from vectora.predict import engine as pengine

    def fake_predict(con, date_str=None):
        return {"date": date_str or "auto", "predictions": 12, "signals": 2,
                "targets": ["g5_h10", "g10_h30"]}

    monkeypatch.setattr(pengine, "run_predict",
                        lambda con, date_str=None: fake_predict(con, date_str))
    from vectora.__main__ import main
    rc = main(["run", "predict", "--date", "2026-07-16"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"signals": 2' in out
```

- [ ] **Step 2: Run to verify failure** — FAIL (`predict` not in stage choices).

- [ ] **Step 3: Implement the CLI stage**

In `vectora/__main__.py`: change stage choices to `["eod", "train", "predict"]` and add after the train branch:

```python
    if args.command == "run" and args.stage == "predict":
        from vectora import db as vdb
        from vectora.predict import engine as pengine
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = pengine.run_predict(con, date_str=args.date)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0
```

(Exit is 0 whenever the stage runs to completion — "no signals today" is a normal outcome, not a failure. Crashes raise and exit non-zero via the traceback path.)

- [ ] **Step 4: Add the workflow step**

In `.github/workflows/eod-pipeline.yml`, insert between the "Run EOD pipeline" step and the "Commit data" step:

```yaml
      - name: Predict
        id: predict
        continue-on-error: true
        run: uv run python -m vectora run predict
```

And extend the final failure-surfacing step's condition:

```yaml
      - name: Surface pipeline failure
        if: steps.pipeline.outcome == 'failure' || steps.predict.outcome == 'failure'
        run: |
          echo "pipeline or predict exited non-zero"
          exit 1
```

- [ ] **Step 5: Run tests + lint** — `uv run pytest -m "not slow"` all pass; `uv run ruff check .` clean.

- [ ] **Step 6: First real prediction run**

Run: `uv run python -m vectora run predict`
Expected: JSON with date = the latest collected trading day, predictions ≈ 660 (two targets × ~330 universe names), signals ≥ 0. Runtime under ~5 min (analog index fit is the slow part).

Sanity-check with:
`uv run python -c "
from vectora import db as vdb
con = vdb.connect('data/vectora.duckdb'); vdb.init_schema(con)
print(con.execute('SELECT target, count(*), sum(CASE WHEN is_signal THEN 1 ELSE 0 END) FROM predictions GROUP BY target').fetchall())
print(con.execute('SELECT suppressed_reason, count(*) FROM predictions GROUP BY suppressed_reason ORDER BY 2 DESC').fetchall())
print(con.execute('SELECT rendered FROM explanations LIMIT 1').fetchone()[0][:400])
con.close()"`
Verify: g10_h30 has zero signals (all suppressed); any g5_h10 signals have plausible probabilities (0.55–0.9); the rendered explanation reads sensibly. Paste 1-2 rendered explanations in your report.

- [ ] **Step 7: Commit**

```bash
git add vectora/__main__.py .github/workflows/eod-pipeline.yml tests/test_orchestrator.py data/vectora.duckdb
git commit -m "feat: predict stage in CLI and scheduled pipeline; first real predictions"
```

---

## Execution notes

- Strict order 1→7; suite green + separate commit per task. Fast suite: `uv run pytest -m "not slow"`.
- Task 1 Step 5 and Task 7 Step 6 touch the real database — everything else is code with isolated test DBs.
- Expected suite size at the end: ~127 tests.
- After Task 7, merge `phase-3a-prediction` into `main` (the daily 15:30 Dhaka run then predicts automatically). Phase 3B (alerts + vault + outcomes) builds on the tables this plan created.
- The g10_h30 model stays signal-suppressed by design; do not "fix" that.

