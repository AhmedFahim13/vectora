# Vectora Phase 4A: Market Regime Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify every trading day since 2012 into the spec §11 regime taxonomy from panel-derived market state, gate signals in Panic regimes, and surface the regime in the digest and journal.

**Architecture:** `regime.state` reduces the full price panel to one row per date (median cross-sectional return, a synthetic equal-weight market level with 50/200-day averages, breadth above own 50DMA, rolling volatility percentile, and an activity z-score built on total volume — volume exists in the backfill era, traded value does not). `regime.rules` maps each state row to {Panic, LowLiquidity, Recovery, SpeculativeHeat, Bull, Bear, Sideways} via ordered threshold rules (first match wins) and persists to a `regimes` table. The predict engine gains a panic gate; digest and journal display the day's regime. The HMM layer (§11 layer 2) is deliberately deferred — the rules layer delivers the full taxonomy, gating, and per-regime evaluation hooks; HMM adds change-point probabilities and belongs with Phase 5's evaluation work.

**Tech Stack:** existing only (polars, DuckDB). No new dependencies.

**Existing contracts (all on `main`):**
- `vectora/features/base.py`: `load_panel(con) -> pl.DataFrame` — symbol/date/close/volume/value_mn/`ret` (canonical daily return), sorted by symbol,date. Backfill rows (2012→2026-01) have `value_mn=None`, `volume` populated.
- `vectora/db.py`: `init_schema`, `upsert`; `vectora/predict/engine.py`: `run_predict(con, date_str=None, features_path=None, min_median_value_mn=1.0)` and `_gate(p, target, category, quality)` with gate order quality → target-enabled → Z → threshold; tests in `tests/predict/test_engine.py` build tiny real models (slow-ish, ~90s).
- `vectora/alerts/digest.py`: `build(con, date_str)`; `vectora/vault/generator.py`: `generate(con, date_str, vault_dir=VAULT_DIR)` — journal header line currently `f"{n} predictions | {len(signals)} signal(s) | quality {q}"`.
- `vectora/__main__.py`: stages eod/train/predict/digest/outcomes/vault.
- `tests/conftest.py`: `test_db`. Branch: `git checkout main && git pull && git checkout -b phase-4a-regime`. `uv run …` from repo root; fast tests `uv run pytest -m "not slow"` (currently 145).

**File structure:**

```
vectora/regime/__init__.py, state.py, rules.py
vectora/db.py                      # + regimes table
vectora/predict/engine.py          # + panic gate, regime lookup
vectora/alerts/digest.py           # + regime line
vectora/vault/generator.py         # + regime in journal header
vectora/__main__.py                # + regime stage
.github/workflows/eod-pipeline.yml # + Regime step before Predict
tests/regime/__init__.py, test_state.py, test_rules.py
```

---

### Task 1: Regimes table + market-state builder

**Files:**
- Modify: `vectora/db.py` (SCHEMA)
- Create: `vectora/regime/__init__.py` (empty), `vectora/regime/state.py`
- Test: `tests/regime/__init__.py` (empty), `tests/regime/test_state.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/regime/test_state.py
import datetime as dt

import numpy as np

from vectora import db as vdb
from vectora.regime import state


def _seed(con, n_days=300, n_syms=60, seed=4, vol_mult=1.0):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = dt.date(2025, 1, 1)
    px = {f"S{i:02d}": 100.0 for i in range(n_syms)}
    for day in range(n_days):
        d = (d0 + dt.timedelta(days=day)).isoformat()
        for sym in px:
            px[sym] *= float(np.exp(rng.normal(0.0003, 0.01 * vol_mult)))
            p = round(max(px[sym], 1.0), 2)
            rows.append(dict(symbol=sym, date=d, open=p, high=p * 1.01,
                             low=p * 0.99, close=p, ltp=p, ycp=p, trades=20,
                             value_mn=2.0, volume=int(rng.integers(500, 5000)),
                             source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)


def test_regimes_table_exists(test_db):
    tables = {r[0] for r in test_db.execute("SHOW TABLES").fetchall()}
    assert "regimes" in tables


def test_market_state_shape_and_columns(test_db):
    _seed(test_db)
    st = state.market_state(test_db)
    assert set(st.columns) >= {"date", "med_ret", "mkt_level", "ma50",
                               "ma200", "ret_21d", "vol_21d", "vol_pctile",
                               "breadth", "activity_z"}
    assert st.height == 300                      # one row per trading date
    assert st["date"].is_sorted()


def test_market_state_values_sane(test_db):
    _seed(test_db)
    st = state.market_state(test_db)
    last = st.tail(1).row(0, named=True)
    assert last["ma200"] is not None             # 300 days > 200 warmup
    assert 0.0 <= last["breadth"] <= 1.0
    assert 0.0 <= last["vol_pctile"] <= 1.0
    assert last["mkt_level"] > 0
    # med_ret of ~0.03% drift stays small
    assert abs(last["med_ret"]) < 0.05


def test_sparse_dates_are_dropped(test_db):
    _seed(test_db, n_syms=60)
    # one extra date with only 3 symbols must not produce a state row
    vdb.upsert(test_db, "prices_raw", [
        dict(symbol=f"S{i:02d}", date="2026-06-01", open=10, high=10, low=10,
             close=10, ltp=10, ycp=10, trades=1, value_mn=0.1, volume=10,
             source="dse_eod") for i in range(3)])
    st = state.market_state(test_db)
    assert str(st["date"].max()) != "2026-06-01"
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/regime -v` → FAIL (table/module missing).

- [ ] **Step 3: Implement**

Append to `SCHEMA` in `vectora/db.py`:

```sql
CREATE TABLE IF NOT EXISTS regimes (
    date DATE PRIMARY KEY, regime TEXT, confidence DOUBLE,
    method TEXT, computed_at TIMESTAMP DEFAULT current_timestamp
);
```

Create `vectora/regime/state.py`:

```python
"""Panel-derived daily market state (spec §11 inputs).

No dependence on scraped index history (which only starts 2026-07): the
market is summarized directly from the cross-section — median return, a
synthetic equal-weight level, breadth above own 50DMA, rolling volatility
percentile, and an activity z-score on TOTAL VOLUME (traded value is null
throughout the backfill era, volume is not).
"""
import polars as pl

from vectora.features import base

MIN_SYMBOLS = 30          # dates with fewer cross-sectional obs are noise
VOL_PCTL_WINDOW = 252


def market_state(con) -> pl.DataFrame:
    panel = base.load_panel(con)
    per_symbol = panel.with_columns(
        (pl.col("close") > pl.col("close").rolling_mean(50).over("symbol"))
        .cast(pl.Int8).alias("above_ma50"))
    daily = (
        per_symbol.group_by("date")
        .agg(
            pl.col("ret").median().alias("med_ret"),
            pl.col("above_ma50").mean().alias("breadth"),
            pl.col("volume").sum().alias("total_volume"),
            pl.len().alias("n_symbols"),
        )
        .filter(pl.col("n_symbols") >= MIN_SYMBOLS)
        .sort("date")
    )
    daily = daily.with_columns(
        (pl.col("med_ret").fill_null(0) + 1).cum_prod().alias("mkt_level"))
    daily = daily.with_columns(
        pl.col("mkt_level").rolling_mean(50).alias("ma50"),
        pl.col("mkt_level").rolling_mean(200).alias("ma200"),
        (pl.col("mkt_level") / pl.col("mkt_level").shift(21) - 1)
        .alias("ret_21d"),
        pl.col("med_ret").rolling_std(21).alias("vol_21d"),
        ((pl.col("total_volume")
          - pl.col("total_volume").rolling_mean(63))
         / (pl.col("total_volume").rolling_std(63) + 1e-9))
        .alias("activity_z"),
    )
    # rolling percentile rank of vol_21d within the trailing year
    daily = daily.with_columns(
        pl.col("vol_21d").rolling_map(
            lambda s: float((s < s[-1]).sum() / max(len(s) - 1, 1)),
            window_size=VOL_PCTL_WINDOW, min_samples=63)
        .alias("vol_pctile"))
    return daily
```

Debug note: `rolling_map` is slow-ish but runs once per day over ~3,300 rows — fine. If the installed polars version renamed `rolling_map`/`min_samples`, adapt the implementation (e.g. `rolling_apply`/`min_periods`), never the tests. If `vol_pctile` ends up outside [0,1] due to NaN handling, clip it in the implementation.

- [ ] **Step 4: Run tests** — `uv run pytest tests/regime -v` → 4 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/db.py vectora/regime tests/regime
git commit -m "feat: regimes table and panel-derived market state builder"
```

---

### Task 2: Rule classifier + history writer

**Files:**
- Create: `vectora/regime/rules.py`
- Test: `tests/regime/test_rules.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/regime/test_rules.py
import polars as pl

from vectora import db as vdb
from vectora.regime import rules


def _row(**kw):
    base = dict(mkt_level=1.0, ma50=1.0, ma200=0.95, ret_21d=0.01,
                vol_pctile=0.5, breadth=0.5, activity_z=0.0)
    return {**base, **kw}


def test_panic_beats_everything():
    r, c = rules.classify_row(_row(vol_pctile=0.95, ret_21d=-0.12,
                                   breadth=0.7, mkt_level=1.2))
    assert r == "Panic" and c >= 0.7


def test_low_liquidity():
    assert rules.classify_row(_row(activity_z=-2.0))[0] == "LowLiquidity"


def test_recovery_below_trend_but_rallying():
    r, _ = rules.classify_row(_row(mkt_level=0.9, ma200=1.0, ret_21d=0.08))
    assert r == "Recovery"


def test_speculative_heat():
    r, _ = rules.classify_row(_row(activity_z=2.5, vol_pctile=0.8))
    assert r == "SpeculativeHeat"


def test_bull_bear_sideways():
    assert rules.classify_row(
        _row(mkt_level=1.1, ma200=1.0, breadth=0.7))[0] == "Bull"
    assert rules.classify_row(
        _row(mkt_level=0.9, ma200=1.0, breadth=0.2))[0] == "Bear"
    assert rules.classify_row(_row())[0] == "Sideways"


def test_warmup_rows_unclassified():
    assert rules.classify_row(_row(ma200=None)) is None


def test_classify_history_writes_table(test_db, monkeypatch):
    import datetime as dt
    frame = pl.DataFrame({
        "date": [dt.date(2026, 7, d) for d in (1, 2, 3)],
        "mkt_level": [1.1, 0.9, 1.0],
        "ma50": [1.0, 1.0, 1.0],
        "ma200": [1.0, 1.0, None],          # day 3 = warmup, skipped
        "ret_21d": [0.02, -0.12, 0.0],
        "vol_pctile": [0.5, 0.95, 0.5],
        "breadth": [0.7, 0.6, 0.5],
        "activity_z": [0.0, 0.0, 0.0],
    })
    from vectora.regime import state as st
    monkeypatch.setattr(st, "market_state", lambda con: frame)
    result = rules.classify_history(test_db)
    assert result == {"classified": 2, "skipped": 1}
    rows = dict(test_db.execute(
        "SELECT date, regime FROM regimes ORDER BY date").fetchall())
    assert str(min(rows)) == "2026-07-01"
    assert list(rows.values()) == ["Bull", "Panic"]


def test_current_regime_lookup(test_db):
    vdb.upsert(test_db, "regimes", [
        {"date": "2026-07-16", "regime": "Bull", "confidence": 0.8,
         "method": "rules"}])
    assert rules.regime_on(test_db, "2026-07-16") == "Bull"
    assert rules.regime_on(test_db, "2026-07-15") is None
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/regime/rules.py
"""Ordered threshold rules mapping market state to the spec §11 taxonomy.
First match wins; thresholds are documented judgment calls, revisited in
Phase 5 once per-regime evaluation data accumulates. Warmup rows (no
200-day average yet) stay unclassified rather than guessed."""
from vectora import db as vdb
from vectora.regime import state as market_state_mod

# (regime, confidence) rules in priority order
PANIC_VOL, PANIC_RET = 0.90, -0.08
LOWLIQ_Z = -1.5
RECOVERY_RET = 0.05
HEAT_Z, HEAT_VOL = 2.0, 0.70
BULL_BREADTH, BEAR_BREADTH = 0.60, 0.35


def classify_row(r: dict) -> tuple[str, float] | None:
    if r["ma200"] is None or r["ret_21d"] is None or r["vol_pctile"] is None:
        return None
    if r["vol_pctile"] > PANIC_VOL and r["ret_21d"] < PANIC_RET:
        return "Panic", 0.8
    if r["activity_z"] is not None and r["activity_z"] < LOWLIQ_Z:
        return "LowLiquidity", 0.7
    if r["mkt_level"] < r["ma200"] and r["ret_21d"] > RECOVERY_RET:
        return "Recovery", 0.7
    if (r["activity_z"] is not None and r["activity_z"] > HEAT_Z
            and r["vol_pctile"] > HEAT_VOL):
        return "SpeculativeHeat", 0.7
    if r["mkt_level"] > r["ma200"] and r["breadth"] > BULL_BREADTH:
        return "Bull", 0.8
    if r["mkt_level"] < r["ma200"] and r["breadth"] < BEAR_BREADTH:
        return "Bear", 0.8
    return "Sideways", 0.5


def classify_history(con) -> dict:
    frame = market_state_mod.market_state(con)
    rows, skipped = [], 0
    for r in frame.iter_rows(named=True):
        result = classify_row(r)
        if result is None:
            skipped += 1
            continue
        regime, conf = result
        rows.append({"date": str(r["date"]), "regime": regime,
                     "confidence": conf, "method": "rules"})
    if rows:
        vdb.upsert(con, "regimes", rows)
    return {"classified": len(rows), "skipped": skipped}


def regime_on(con, date_str: str) -> str | None:
    row = con.execute(
        "SELECT regime FROM regimes WHERE date = ?", [date_str]).fetchone()
    return row[0] if row else None
```

Note: `classify_history` monkeypatch target in the test is `vectora.regime.state.market_state` — import the MODULE (`from vectora.regime import state as market_state_mod`) and call through it, as shown, or the monkeypatch won't take.

- [ ] **Step 4: Run tests** — 8 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/regime/rules.py tests/regime/test_rules.py
git commit -m "feat: rule-based regime classifier with history writer"
```

---

### Task 3: Panic gate + digest/journal display + CLI + workflow

**Files:**
- Modify: `vectora/predict/engine.py`, `vectora/alerts/digest.py`, `vectora/vault/generator.py`, `vectora/__main__.py`, `.github/workflows/eod-pipeline.yml`
- Test: append to `tests/predict/test_engine.py`, `tests/test_digest.py`, `tests/test_vault.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/predict/test_engine.py`:

```python
def test_panic_regime_suppresses_all_signals(test_db, tmp_path):
    last = _seed_market(test_db)
    _train_and_register(test_db, tmp_path)
    vdb.upsert(test_db, "regimes", [
        {"date": last, "regime": "Panic", "confidence": 0.8,
         "method": "rules"}])
    pengine.run_predict(test_db, date_str=last,
                        features_path=tmp_path / "f2.parquet",
                        min_median_value_mn=0.1)
    n_signals = test_db.execute(
        "SELECT count(*) FROM predictions WHERE is_signal").fetchone()[0]
    assert n_signals == 0
    reasons = {r[0] for r in test_db.execute(
        "SELECT DISTINCT suppressed_reason FROM predictions").fetchall()}
    assert "panic-regime-gate" in reasons
```

Append to `tests/test_digest.py` (inside, reuse `_seed`):

```python
def test_digest_shows_regime(test_db):
    _seed(test_db)
    vdb.upsert(test_db, "regimes", [
        {"date": "2026-07-16", "regime": "Bull", "confidence": 0.8,
         "method": "rules"}])
    body = digest.build(test_db, "2026-07-16")
    assert "regime Bull" in body
```

Append to `tests/test_vault.py`:

```python
def test_journal_shows_regime(test_db, tmp_path):
    _seed(test_db)
    vdb.upsert(test_db, "regimes", [
        {"date": "2026-07-16", "regime": "Sideways", "confidence": 0.5,
         "method": "rules"}])
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    journal = (tmp_path / "Journal" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "regime Sideways" in journal
```

- [ ] **Step 2: Run to verify failures** — the three new tests FAIL (no regime plumbing yet).

- [ ] **Step 3: Implement**

In `vectora/predict/engine.py`:
- import: `from vectora.regime import rules as regime_rules`
- in `run_predict`, after the `quality` lookup add: `regime = regime_rules.regime_on(con, run_date)`
- change the `_gate` call to `_gate(p, target, block["category"], quality, regime)`
- change `_gate` to:

```python
def _gate(p: float, target: str, category: str | None,
          quality: int, regime: str | None = None) -> str | None:
    """First failing gate wins; None means the prediction is a signal."""
    if quality < MIN_QUALITY_SCORE:
        return "quality-below-floor"
    if target not in SIGNAL_THRESHOLDS:
        return "g10_h30-tail-overconfident" if target == "g10_h30" \
            else "target-not-enabled"
    if category == "Z":
        return "z-category-gate"
    if regime == "Panic":
        return "panic-regime-gate"
    if p < SIGNAL_THRESHOLDS[target]:
        return "below-probability-threshold"
    return None
```

In `vectora/alerts/digest.py` `build()`: after the quality lookup add:

```python
    regime = con.execute(
        "SELECT regime FROM regimes WHERE date = ?", [date_str]).fetchone()
```

and change the summary line to:

```python
    reg = regime[0] if regime else "unclassified"
    lines = [
        f"# Vectora digest {date_str}",
        "",
        f"{n} predictions | {len(signals)} signal(s) | data quality {q} | "
        f"regime {reg}",
        "",
    ]
```

In `vectora/vault/generator.py` `generate()`: after the quality lookup add the same two-line regime fetch (variable `regime`), and change the journal header line to:

```python
    reg = regime[0] if regime else "unclassified"
    lines = [f"# Journal {date_str}", "",
             f"{len(preds)} predictions | {len(signals)} signal(s) | "
             f"quality {q} | regime {reg}", ""]
```

In `vectora/__main__.py`: stage choices gain `"regime"`; add before the predict branch:

```python
    if args.command == "run" and args.stage == "regime":
        from vectora import db as vdb
        from vectora.regime import rules
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = rules.classify_history(con)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0
```

In `.github/workflows/eod-pipeline.yml`, insert between the "Run EOD pipeline" step and the "Predict" step:

```yaml
      - name: Regime
        continue-on-error: true
        run: uv run python -m vectora run regime
```

- [ ] **Step 4: Run tests** — new tests pass; full fast suite (`uv run pytest -m "not slow"`) all pass (existing digest/vault tests unaffected — they assert substrings still present); ruff clean.

- [ ] **Step 5: Commit**

```bash
git add vectora/predict/engine.py vectora/alerts/digest.py vectora/vault/generator.py vectora/__main__.py .github/workflows/eod-pipeline.yml tests/predict/test_engine.py tests/test_digest.py tests/test_vault.py
git commit -m "feat: panic-regime signal gate; regime in digest, journal, CLI, pipeline"
```

---

### Task 4: Full-history classification + sanity + merge

**Files:** none new — real run, verification, merge.

- [ ] **Step 1: Classify all history**

Run: `uv run python -m vectora run regime`
Expected: `{"classified": ~3050, "skipped": ~250}` (≈13.5 years of trading days minus the 200-day warmup and sparse early dates).

- [ ] **Step 2: Sanity-check against known DSE episodes**

Run:
`uv run python -c "
from vectora import db as vdb
con = vdb.connect('data/vectora.duckdb'); vdb.init_schema(con)
print(con.execute('SELECT regime, count(*) FROM regimes GROUP BY 1 ORDER BY 2 DESC').fetchall())
print('2020 COVID window:', con.execute(\"SELECT regime, count(*) FROM regimes WHERE date BETWEEN '2020-02-15' AND '2020-04-15' GROUP BY 1 ORDER BY 2 DESC\").fetchall())
print('2022H2 floor era:', con.execute(\"SELECT regime, count(*) FROM regimes WHERE date BETWEEN '2022-08-01' AND '2023-06-30' GROUP BY 1 ORDER BY 2 DESC\").fetchall())
print('today:', con.execute('SELECT date, regime FROM regimes ORDER BY date DESC LIMIT 3').fetchall())
con.close()"`

Judgment checks (report the actual outputs):
- The 2020 Feb–Apr window should be dominated by Panic/Bear (COVID crash; DSE also closed ~2 months — fewer rows is expected).
- The 2022 H2 floor-price era should lean LowLiquidity/Sideways (frozen prices, dead turnover).
- No single regime should exceed ~60% of all days; Sideways plurality is expected. If Panic exceeds ~10% of days, thresholds are too loose — flag it, don't ship silently.

- [ ] **Step 3: Fast suite + ruff, commit data, merge, push, verify**

```bash
uv run pytest -m "not slow" && uv run ruff check .
git add data/vectora.duckdb
git commit -m "data: full-history regime classification (2012-2026)"
git checkout main && git pull
git merge --no-ff phase-4a-regime -m "Merge phase-4a: rule-based market regime detection"
git push
```

Then dispatch and watch a verification run:

```bash
& "C:\Program Files\GitHub CLI\gh.exe" workflow run eod-pipeline --ref main -f date=2026-07-16
```

Confirm all steps green including the new Regime step, and that the digest/journal for the day now carry a regime label.

---

## Execution notes

- Strict order 1→4; suite green + commit per task. Expected fast suite ≈ 156 tests.
- Thresholds in rules.py are explicit constants — Phase 5's per-regime evaluation will tune them with data; do not bikeshed them now.
- The HMM layer (spec §11 layer 2) is intentionally NOT here; it lands with Phase 5 alongside per-regime calibration accounting, where its change-point probabilities have a consumer.
- After merge, remaining Phase 4 slices: 4B event classifier + event studies, 4C Z-module + pump-phase model + pre-announcement footprints, 4D intraday scans + urgency tiers.
