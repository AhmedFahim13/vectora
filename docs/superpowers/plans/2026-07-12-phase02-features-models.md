# Vectora Phase 2: Features + Baseline Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split static backfill storage out of the daily database, then build the feature engine, label grid, walk-forward training harness, and calibrated LightGBM-vs-logistic model comparison — Phase 2 of the approved spec (`docs/superpowers/specs/2026-07-12-dse-market-intelligence-design.md` §8–11).

**Architecture:** The 1.06M-row Mendeley backfill moves to a committed ZSTD Parquet; a DuckDB view (`prices`) unions it with the live-scraped `prices_raw` so all consumers see one seamless price history while the daily-rewritten DB shrinks from 63MB to ~3MB. Features are computed by a registry-driven polars engine (full recompute per run — 1M rows × ~40 features is seconds; incremental complexity is YAGNI at this scale). Labels are the spec's P(max gain ≥ X% within H days) grid. Training uses expanding-window walk-forward with a max-horizon embargo, isotonic calibration, and Brier score as the primary metric. **Phase 2 exit criterion (spec roadmap): LightGBM beats the logistic baseline out-of-sample and the calibration plot is honest.**

**Tech Stack:** Python 3.12 + uv, DuckDB (existing), polars + pyarrow (new), LightGBM + scikit-learn (new), pytest.

**Known Phase-2 approximation (documented, deliberate):** corporate-action adjustment is crude — daily returns use `close/ycp − 1` where YCP exists (DSE's YCP is already ex-date adjusted, so scraped rows are correct across corporate actions) and band-clipped `close/prev_close − 1` (±12%) for backfill rows, which clips split/rights gaps instead of adjusting them. The full corporate-action engine is Phase 3+, fed by the events pipeline. Every consumer of returns must go through `vectora/features/base.py` so the upgrade lands in one place.

**Existing contracts this plan builds on (read before starting):**
- `vectora/db.py`: `connect(path)`, `init_schema(con)`, `upsert(con, table, rows)` (atomic, INSERT OR REPLACE), `get/set_watermark`. Tables incl. `prices_raw(symbol,date,open,high,low,close,ltp,ycp,trades,value_mn,volume,source)` PK (symbol,date,source), `symbols(symbol,…,sector,instrument_type,category,…)`, `model_registry` does NOT exist yet (this plan adds it).
- `vectora/settings.py`: `REPO_ROOT, DATA_DIR, REFERENCE_DIR, DB_PATH, MIN_QUALITY_SCORE`.
- `tests/conftest.py`: `test_db` fixture = fresh schema-initialized DuckDB per test.
- Data on hand: 1,063,452 backfill rows (source='mendeley', 2012-10-01→2026-01-22), live scraped rows (source='dse_eod') accumulating daily, 677 symbols with sector/category/instrument_type from the company sweep.
- Run everything with `uv run …` from repo root. Commit directly to the working branch (create `phase-02-features-models` off `main` first: `git checkout main && git pull && git checkout -b phase-02-features-models`).

**File structure created by this plan:**

```
vectora/
├── universe.py                  # tradable-universe filter (equities + liquidity floor)
├── labels.py                    # forward-return label grid
├── features/
│   ├── __init__.py
│   ├── base.py                  # price panel loader + canonical return column
│   ├── registry.py              # features.yaml loader + validation
│   ├── families.py              # feature computation functions (polars exprs)
│   └── engine.py                # registry -> wide feature frame -> parquet
├── train/
│   ├── __init__.py
│   ├── walkforward.py           # expanding-window splits with embargo
│   ├── models.py                # logistic baseline + LightGBM + isotonic calibration
│   └── trainer.py               # orchestrates: data -> splits -> fit -> metrics -> report
├── config/
│   └── features.yaml            # feature registry (name, family, params, reasoning)
tools/split_backfill.py          # one-time storage migration
data/reference/backfill_2012_2026.parquet   # committed static history
data/features/features.parquet   # generated, committed by train runs
models/<model_id>/…              # lgbm.txt, calibrator.pkl, meta.json
reports/train_<target>_<date>.md # comparison report
.github/workflows/train.yml      # manual-dispatch training workflow
```

---

### Task 1: Storage split — backfill to Parquet, DB slimmed, `prices` view

**Files:**
- Create: `tools/split_backfill.py`
- Modify: `vectora/settings.py` (add BACKFILL_PARQUET, FEATURES_DIR, MODELS_DIR, REPORTS_DIR)
- Modify: `vectora/db.py` (init_schema creates the `prices` view + `model_registry` table)
- Modify: `pyproject.toml` (add polars, pyarrow, lightgbm, scikit-learn)
- Test: `tests/test_storage_split.py`

- [ ] **Step 1: Add dependencies and settings**

In `pyproject.toml` `[project] dependencies`, append:

```toml
    "polars>=1.10",
    "pyarrow>=17.0",
    "lightgbm>=4.5",
    "scikit-learn>=1.5",
```

Run: `uv sync` — expect the four packages (plus transitive numpy/scipy) installed.

Append to `vectora/settings.py`:

```python
BACKFILL_PARQUET = REFERENCE_DIR / "backfill_2012_2026.parquet"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_storage_split.py
from vectora import db as vdb


def _price(symbol, d, close, source):
    return dict(symbol=symbol, date=d, open=close, high=close, low=close,
                close=close, ltp=close, ycp=close, trades=1, value_mn=1.0,
                volume=100, source=source)


def test_prices_view_without_parquet_is_prices_raw(test_db):
    vdb.upsert(test_db, "prices_raw", [_price("GP", "2026-07-09", 10.0, "dse_eod")])
    rows = test_db.execute("SELECT count(*) FROM prices").fetchone()[0]
    assert rows == 1


def test_prices_view_unions_backfill_parquet(tmp_path):
    import polars as pl
    pq = tmp_path / "backfill.parquet"
    pl.DataFrame({
        "symbol": ["GP"], "date": ["2013-01-02"], "open": [5.0], "high": [5.0],
        "low": [5.0], "close": [5.0], "ltp": [None], "ycp": [None],
        "trades": [None], "value_mn": [None], "volume": [1000],
        "source": ["mendeley"],
    }).with_columns(pl.col("date").cast(pl.Date)).write_parquet(pq)
    con = vdb.connect(tmp_path / "t.duckdb")
    vdb.init_schema(con, backfill_parquet=pq)
    vdb.upsert(con, "prices_raw", [_price("GP", "2026-07-09", 10.0, "dse_eod")])
    rows = con.execute("SELECT count(*) FROM prices").fetchone()[0]
    assert rows == 2
    srcs = {r[0] for r in con.execute("SELECT DISTINCT source FROM prices").fetchall()}
    assert srcs == {"mendeley", "dse_eod"}
    con.close()


def test_model_registry_table_exists(test_db):
    cols = {r[0] for r in test_db.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'model_registry'").fetchall()}
    assert {"model_id", "family", "target", "trained_at", "metrics", "active"} <= cols
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_storage_split.py -v`
Expected: FAIL — `init_schema() got an unexpected keyword argument 'backfill_parquet'` / no `prices` view / no `model_registry`.

- [ ] **Step 4: Implement db.py changes**

In `vectora/db.py`, append to `SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY, family TEXT, target TEXT,
    trained_at TIMESTAMP DEFAULT current_timestamp,
    train_end DATE, metrics TEXT,          -- metrics: JSON
    artifact_dir TEXT, active BOOLEAN DEFAULT false
);
```

Replace `init_schema` with:

```python
def init_schema(con: duckdb.DuckDBPyConnection,
                backfill_parquet: str | Path | None = None) -> None:
    con.execute(SCHEMA)
    if backfill_parquet is None:
        from vectora.settings import BACKFILL_PARQUET
        backfill_parquet = BACKFILL_PARQUET
    if Path(backfill_parquet).exists():
        pq = str(Path(backfill_parquet)).replace("\\", "/")
        con.execute(f"""
            CREATE OR REPLACE VIEW prices AS
            SELECT * FROM prices_raw
            UNION ALL SELECT * FROM read_parquet('{pq}')
        """)
    else:
        con.execute("CREATE OR REPLACE VIEW prices AS SELECT * FROM prices_raw")
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_storage_split.py -v` — expect 3 passed. Then full suite `uv run pytest -q` — all pass (existing tests use the no-parquet branch).

- [ ] **Step 6: Write the migration script**

```python
# tools/split_backfill.py
"""One-time Phase 2 migration: move source='mendeley' rows out of the daily
database into a committed static Parquet, then rebuild the DB file so the
freed space is actually reclaimed (DELETE alone leaves the file large).

Usage: uv run python tools/split_backfill.py
"""
import shutil
import sys

from vectora import db as vdb
from vectora.settings import BACKFILL_PARQUET, DB_PATH


def main() -> int:
    con = vdb.connect(DB_PATH)
    n = con.execute(
        "SELECT count(*) FROM prices_raw WHERE source = 'mendeley'").fetchone()[0]
    if n == 0:
        print("no mendeley rows in DB; nothing to migrate")
        con.close()
        return 0
    pq = str(BACKFILL_PARQUET).replace("\\", "/")
    con.execute(f"""
        COPY (SELECT * FROM prices_raw WHERE source = 'mendeley'
              ORDER BY symbol, date)
        TO '{pq}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.execute("DELETE FROM prices_raw WHERE source = 'mendeley'")
    con.execute("CHECKPOINT")

    # rebuild the DB file to reclaim space: copy every table to a fresh file
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_type = 'BASE TABLE'").fetchall()]
    new_path = DB_PATH.with_suffix(".duckdb.new")
    new_path.unlink(missing_ok=True)
    ncon = vdb.connect(new_path)
    vdb.init_schema(ncon)
    old = str(DB_PATH).replace("\\", "/")
    ncon.execute(f"ATTACH '{old}' AS src (READ_ONLY)")
    for t in tables:
        ncon.execute(f"INSERT INTO {t} SELECT * FROM src.{t}")
    ncon.execute("DETACH src")
    ncon.close()
    con.close()
    shutil.move(str(new_path), str(DB_PATH))

    check = vdb.connect(DB_PATH)
    vdb.init_schema(check)
    total = check.execute("SELECT count(*) FROM prices").fetchone()[0]
    raw = check.execute("SELECT count(*) FROM prices_raw").fetchone()[0]
    check.close()
    print(f"migrated {n} rows to {BACKFILL_PARQUET.name}; "
          f"prices view={total}, prices_raw={raw}, "
          f"db={DB_PATH.stat().st_size/1e6:.1f}MB, "
          f"parquet={BACKFILL_PARQUET.stat().st_size/1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Run the migration for real**

Run: `uv run python tools/split_backfill.py`
Expected output shape: `migrated 1063452 rows …; prices view=1064xxx, prices_raw≈640+ (scraped days only), db≈3-6MB, parquet≈8-15MB`.
Then verify the pipeline still works against the slim DB: `uv run python -m vectora run eod --date 2026-07-09` → exit 0, quality 100 (idempotent re-run).

- [ ] **Step 8: Ruff, full suite, commit**

```bash
uv run ruff check . && uv run pytest -q
git add pyproject.toml uv.lock vectora/settings.py vectora/db.py tools/split_backfill.py tests/test_storage_split.py data/reference/backfill_2012_2026.parquet data/vectora.duckdb
git commit -m "feat: split static backfill into committed parquet; prices view unions it (63MB DB -> ~4MB)"
```

---

### Task 2: Tradable-universe filter

**Files:**
- Create: `vectora/universe.py`
- Test: `tests/test_universe.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_universe.py
from vectora import db as vdb
from vectora.universe import tradable_universe


def _seed(con):
    vdb.upsert(con, "symbols", [
        dict(symbol="GP", name=None, sector="Telecommunication",
             instrument_type="Equity", category="A", listing_status="active",
             first_seen="2013-01-01", last_seen="2026-07-09"),
        dict(symbol="ILLIQ", name=None, sector="Bank",
             instrument_type="Equity", category="B", listing_status="active",
             first_seen="2013-01-01", last_seen="2026-07-09"),
        dict(symbol="TBOND1", name=None, sector="Govt Bond",
             instrument_type="Bond", category=None, listing_status="active",
             first_seen="2013-01-01", last_seen="2026-07-09"),
    ])
    rows = []
    for i in range(60):
        d = f"2026-{4 + i // 28:02d}-{i % 28 + 1:02d}"
        rows.append(dict(symbol="GP", date=d, open=10, high=10, low=10, close=10,
                         ltp=10, ycp=10, trades=100, value_mn=25.0, volume=9000,
                         source="dse_eod"))
        rows.append(dict(symbol="ILLIQ", date=d, open=5, high=5, low=5, close=5,
                         ltp=5, ycp=5, trades=2, value_mn=0.05, volume=100,
                         source="dse_eod"))
        rows.append(dict(symbol="TBOND1", date=d, open=100, high=100, low=100,
                         close=100, ltp=100, ycp=100, trades=50, value_mn=50.0,
                         volume=5000, source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)


def test_universe_keeps_liquid_equities_only(test_db):
    _seed(test_db)
    u = tradable_universe(test_db, as_of="2026-06-28", min_median_value_mn=1.0)
    assert "GP" in u
    assert "ILLIQ" not in u       # fails liquidity floor
    assert "TBOND1" not in u      # not an Equity


def test_universe_liquidity_floor_configurable(test_db):
    _seed(test_db)
    u = tradable_universe(test_db, as_of="2026-06-28", min_median_value_mn=0.01)
    assert {"GP", "ILLIQ"} <= set(u)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_universe.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# vectora/universe.py
"""Tradable universe: active equities passing a trailing liquidity floor.

Illiquidity is the #1 practical risk on the DSE (spec §2): signals on
names that trade a few thousand taka a day are untradable noise, so the
universe is filtered on trailing 60-trading-day median daily traded value.
"""

TRAILING_DAYS = 60


def tradable_universe(con, as_of: str, min_median_value_mn: float = 1.0) -> list[str]:
    rows = con.execute(
        """
        WITH recent AS (
            SELECT symbol, value_mn,
                   row_number() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM prices
            WHERE date <= ?
        )
        SELECT r.symbol
        FROM recent r
        JOIN symbols s USING (symbol)
        WHERE r.rn <= ?
          AND s.instrument_type = 'Equity'
          AND s.listing_status = 'active'
        GROUP BY r.symbol
        HAVING median(r.value_mn) >= ?
        ORDER BY r.symbol
        """,
        [as_of, TRAILING_DAYS, min_median_value_mn],
    ).fetchall()
    return [r[0] for r in rows]
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_universe.py -v` → 2 passed; full suite green; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add vectora/universe.py tests/test_universe.py
git commit -m "feat: tradable universe filter (equities + trailing liquidity floor)"
```

---

### Task 3: Price panel + canonical returns (`features/base.py`)

**Files:**
- Create: `vectora/features/__init__.py` (empty), `vectora/features/base.py`
- Test: `tests/features/__init__.py` (empty), `tests/features/test_base.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_base.py
import polars as pl

from vectora import db as vdb
from vectora.features import base


def _seed(con):
    rows = [
        # scraped rows: ycp is ex-date adjusted, so ret must use close/ycp.
        # 2026-07-06: close 110 vs ycp 100 -> +10%
        dict(symbol="GP", date="2026-07-06", open=100, high=111, low=99, close=110,
             ltp=110, ycp=100, trades=10, value_mn=5.0, volume=1000, source="dse_eod"),
        # backfill rows: no ycp -> close/prev_close, clipped to +/-12%
        dict(symbol="ACI", date="2026-07-05", open=10, high=10, low=10, close=10.0,
             ltp=None, ycp=None, trades=None, value_mn=None, volume=500,
             source="mendeley"),
        dict(symbol="ACI", date="2026-07-06", open=10, high=11, low=10, close=11.0,
             ltp=None, ycp=None, trades=None, value_mn=None, volume=600,
             source="mendeley"),
        # a 50% "gap" (unadjusted rights/split) must clip to the 12% band
        dict(symbol="ACI", date="2026-07-07", open=16, high=17, low=16, close=16.5,
             ltp=None, ycp=None, trades=None, value_mn=None, volume=700,
             source="mendeley"),
    ]
    vdb.upsert(con, "prices_raw", rows)


def test_panel_has_canonical_return(test_db):
    _seed(test_db)
    df = base.load_panel(test_db)
    assert isinstance(df, pl.DataFrame)
    gp = df.filter(pl.col("symbol") == "GP")
    assert abs(gp["ret"][0] - 0.10) < 1e-9  # close/ycp - 1


def test_backfill_return_uses_prev_close_and_clips(test_db):
    _seed(test_db)
    df = base.load_panel(test_db).filter(pl.col("symbol") == "ACI").sort("date")
    rets = df["ret"].to_list()
    assert rets[0] is None                     # no previous close
    assert abs(rets[1] - 0.10) < 1e-9          # 11/10 - 1
    assert abs(rets[2] - base.RET_CLIP) < 1e-9  # 16.5/11 - 1 = 50% -> clipped


def test_panel_sorted_by_symbol_date(test_db):
    _seed(test_db)
    df = base.load_panel(test_db)
    assert df["symbol"].to_list() == sorted(df["symbol"].to_list(), key=str) or True
    # per-symbol dates strictly increasing
    for _, g in df.group_by("symbol"):
        ds = g.sort("date")["date"].to_list()
        assert ds == sorted(ds)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/features/test_base.py -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/features/base.py
"""Price panel loader with the canonical daily-return column.

Return definition (Phase 2 approximation, upgrade point for the Phase 3
corporate-action engine — change it HERE and every feature inherits it):
- scraped rows (ycp present & > 0): ret = close/ycp - 1. DSE's YCP is
  already adjusted on ex-dates, so these returns are corporate-action safe.
- backfill rows (no ycp): ret = close/prev_close - 1, clipped to +/-RET_CLIP.
  Unadjusted split/rights gaps get clipped instead of adjusted; the clip
  matches the validation band (circuit ~10% + buffer).
"""
import polars as pl

RET_CLIP = 0.12

_PANEL_SQL = """
    SELECT symbol, date, open, high, low, close, ycp, trades, value_mn, volume
    FROM prices
    WHERE close IS NOT NULL AND close > 0
    ORDER BY symbol, date
"""


def load_panel(con) -> pl.DataFrame:
    df = con.execute(_PANEL_SQL).pl()
    prev_close = pl.col("close").shift(1).over("symbol")
    raw_ret = (
        pl.when(pl.col("ycp").is_not_null() & (pl.col("ycp") > 0))
        .then(pl.col("close") / pl.col("ycp") - 1)
        .otherwise(pl.col("close") / prev_close - 1)
    )
    return df.with_columns(
        raw_ret.clip(-RET_CLIP, RET_CLIP).alias("ret")
    )
```

Note: the clip applies to ycp-based returns too — harmless (they cannot exceed the circuit band except data errors, which the clip then also tames).

- [ ] **Step 4: Run tests** — 3 passed; full suite green; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add vectora/features/__init__.py vectora/features/base.py tests/features
git commit -m "feat: price panel loader with corporate-action-safe canonical returns"
```

---

### Task 4: Feature registry (`features.yaml` + loader)

**Files:**
- Create: `vectora/config/__init__.py` (empty), `vectora/config/features.yaml`, `vectora/features/registry.py`
- Modify: `pyproject.toml` (add `pyyaml>=6.0` to dependencies, run `uv sync`)
- Test: `tests/features/test_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_registry.py
import pytest

from vectora.features import registry


def test_load_registry_returns_specs():
    specs = registry.load()
    assert len(specs) >= 35
    names = [s.name for s in specs]
    assert len(names) == len(set(names))          # unique
    assert "ret_21d" in names and "amihud_21d" in names


def test_every_feature_documents_reasoning():
    for s in registry.load():
        assert len(s.reasoning) >= 20, f"{s.name} lacks documented reasoning"
        assert s.family in registry.KNOWN_FAMILIES


def test_unknown_family_rejected(tmp_path):
    bad = tmp_path / "f.yaml"
    bad.write_text(
        "features:\n  - name: x\n    family: nonsense\n"
        "    fn: ret_nd\n    params: {days: 1}\n"
        "    reasoning: twenty characters of reasoning here\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="nonsense"):
        registry.load(bad)
```

- [ ] **Step 2: Run to verify failure** — FAIL (module/yaml missing).

- [ ] **Step 3: Write `vectora/config/features.yaml`**

The full registry — 40 features. `fn` names computation functions defined in Task 5's `families.py`; `params` are passed as kwargs. (Reasoning lines are the spec §8 documentation requirement, enforced by test.)

```yaml
features:
  # ---- momentum (spec: retail herding -> short-horizon persistence, sharp reversals)
  - {name: ret_1d,  family: momentum, fn: ret_nd, params: {days: 1},
     reasoning: one-day return is the base momentum unit and mean-reversion input on a retail-driven exchange}
  - {name: ret_3d,  family: momentum, fn: ret_nd, params: {days: 3},
     reasoning: three-day compounded return captures the start of herding runs documented in DSE pump episodes}
  - {name: ret_5d,  family: momentum, fn: ret_nd, params: {days: 5},
     reasoning: one-trading-week momentum aligns with the Sun-Thu week and weekly retail cycle}
  - {name: ret_10d, family: momentum, fn: ret_nd, params: {days: 10},
     reasoning: two-week momentum is where DSE herding historically peaks before reversal risk rises}
  - {name: ret_21d, family: momentum, fn: ret_nd, params: {days: 21},
     reasoning: one-month momentum is the classic cross-sectional momentum horizon in emerging markets}
  - {name: ret_63d, family: momentum, fn: ret_nd, params: {days: 63},
     reasoning: one-quarter momentum separates persistent trends from short speculative bursts}
  - {name: rsi_14, family: momentum, fn: rsi, params: {days: 14},
     reasoning: retail participants trade RSI signals making them partly self-fulfilling on the DSE}
  - {name: dist_high_63d, family: momentum, fn: dist_from_rolling_max, params: {days: 63},
     reasoning: distance below the quarterly high measures breakout proximity which retail flows chase}
  - {name: dist_low_63d, family: momentum, fn: dist_from_rolling_min, params: {days: 63},
     reasoning: distance above the quarterly low flags capitulation levels where reversals start}
  # ---- volatility (spec: circuit bands truncate moves; limit-lock frequency beats raw vol)
  - {name: vol_21d, family: volatility, fn: ret_std, params: {days: 21},
     reasoning: one-month realized volatility is the base risk scale for position sizing and labels}
  - {name: vol_63d, family: volatility, fn: ret_std, params: {days: 63},
     reasoning: quarterly volatility anchors the regime a stock trades in versus its recent burst}
  - {name: vol_ratio_21_63, family: volatility, fn: ratio_of_stds, params: {short: 21, long: 63},
     reasoning: short-over-long volatility ratio detects fresh volatility expansion preceding large moves}
  - {name: atr_14, family: volatility, fn: atr, params: {days: 14},
     reasoning: average true range in price units feeds stop distance and expected-move estimates}
  - {name: range_pct_5d, family: volatility, fn: avg_range_pct, params: {days: 5},
     reasoning: weekly average high-low range as percent of close measures intraday heat within the band}
  - {name: limit_lock_21d, family: volatility, fn: limit_lock_count, params: {days: 21, band: 0.095},
     reasoning: count of near-circuit closes is the DSE-specific heat gauge since bands truncate raw volatility}
  # ---- liquidity (spec: illiquidity is the top practical risk and manipulation fuel)
  - {name: value_mn_med_21d, family: liquidity, fn: rolling_median_col, params: {col: value_mn, days: 21},
     reasoning: median daily traded value is the tradability floor input used in the universe filter}
  - {name: amihud_21d, family: liquidity, fn: amihud, params: {days: 21},
     reasoning: Amihud illiquidity prices the impact per taka traded which dominates execution risk on thin books}
  - {name: zero_vol_21d, family: liquidity, fn: zero_volume_days, params: {days: 21},
     reasoning: count of zero-volume days flags dormant names where any print can gap the price}
  - {name: turnover_z_21d, family: liquidity, fn: zscore_col, params: {col: value_mn, days: 21},
     reasoning: value z-score highlights unusual money inflow relative to a name's own norm}
  - {name: volume_z_21d, family: liquidity, fn: zscore_col, params: {col: volume, days: 21},
     reasoning: volume z-score is the classic accumulation signal preceding price in DSE pump patterns}
  # ---- volume/flow (spec: volume precedes price in accumulation phases)
  - {name: vol_ratio_5_21, family: volume, fn: volume_ratio, params: {short: 5, long: 21},
     reasoning: five-over-twentyone-day volume ratio measures fresh participation buildup}
  - {name: obv_slope_21d, family: volume, fn: obv_slope, params: {days: 21},
     reasoning: on-balance-volume slope captures directional flow persistence beyond raw volume}
  - {name: updown_vol_21d, family: volume, fn: updown_volume_ratio, params: {days: 21},
     reasoning: up-day versus down-day volume split separates accumulation from distribution}
  - {name: trades_z_21d, family: volume, fn: zscore_col, params: {col: trades, days: 21},
     reasoning: trade-count z-score proxies breadth of participation versus a few large prints}
  - {name: vwap_dev_5d, family: volume, fn: vwap_deviation, params: {days: 5},
     reasoning: deviation from rolling value-weighted price shows who is paying up for inventory}
  # ---- cross-sectional (spec: small market -> violent observable sector rotation)
  - {name: ret_21d_xrank, family: cross_sectional, fn: cross_rank, params: {of: ret_21d},
     reasoning: cross-sectional momentum rank is the tradable signal form robust to market-wide moves}
  - {name: vol_21d_xrank, family: cross_sectional, fn: cross_rank, params: {of: vol_21d},
     reasoning: volatility rank positions a name within the day's risk spectrum for regime-aware gating}
  - {name: turnover_xrank, family: cross_sectional, fn: cross_rank, params: {of: value_mn_med_21d},
     reasoning: liquidity rank distinguishes market darlings from dormant names in the same market state}
  - {name: sector_ret_21d, family: cross_sectional, fn: sector_mean, params: {of: ret_21d},
     reasoning: sector momentum captures the rotation flows that dominate a 22-sector market}
  - {name: ret_vs_sector_21d, family: cross_sectional, fn: minus_sector_mean, params: {of: ret_21d},
     reasoning: return relative to own sector isolates idiosyncratic strength from rotation beta}
  - {name: breadth_above_ma50, family: cross_sectional, fn: market_breadth_above_ma, params: {days: 50},
     reasoning: share of names above their 50-day average is the market-wide regime thermometer}
  # ---- calendar/structure (spec: event proximity and DSE-specific calendar)
  - {name: dow, family: calendar, fn: day_of_week, params: {},
     reasoning: Sunday and Thursday carry systematic open-of-week and pre-weekend retail flow effects}
  - {name: month, family: calendar, fn: month_of_year, params: {},
     reasoning: June-July budget season and December closing drive seasonal flows in Dhaka}
  - {name: days_listed, family: calendar, fn: days_since_first_seen, params: {},
     reasoning: newly listed names trade under different rules and speculative attention than seasoned ones}
  - {name: px_level_log, family: structure, fn: log_close, params: {},
     reasoning: low-priced shares attract disproportionate retail speculation on the DSE}
  - {name: gap_open_1d, family: structure, fn: overnight_gap, params: {},
     reasoning: open-versus-yesterday-close gap measures overnight information or manipulation pressure}
  - {name: hl_position_1d, family: structure, fn: close_in_range, params: {},
     reasoning: where the close sits in the day's range reveals end-of-session buying or selling urgency}
  - {name: ma20_dist, family: structure, fn: dist_from_sma, params: {days: 20},
     reasoning: distance from the 20-day average is the mean-reversion anchor retail chartists watch}
  - {name: ma50_dist, family: structure, fn: dist_from_sma, params: {days: 50},
     reasoning: the 50-day average is the trend line separating accumulation from markdown phases}
  - {name: ma20_above_ma50, family: structure, fn: sma_cross_state, params: {short: 20, long: 50},
     reasoning: moving-average cross state is a self-fulfilling regime flag among local technical traders}
  - {name: ycp_gap_flag, family: structure, fn: ycp_adjustment_flag, params: {},
     reasoning: days where ycp diverges from prior close mark corporate-action ex-dates the model must know about}
```

- [ ] **Step 4: Implement `vectora/features/registry.py`**

```python
# vectora/features/registry.py
"""Feature registry: every feature is declared in features.yaml with its
computation function, params, family, and documented economic reasoning
(spec §8 requires reasoning; a test enforces it)."""
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "features.yaml"

KNOWN_FAMILIES = {
    "momentum", "volatility", "liquidity", "volume",
    "cross_sectional", "calendar", "structure",
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    fn: str
    params: dict
    reasoning: str


def load(path: Path = DEFAULT_PATH) -> list[FeatureSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    specs = []
    for item in raw["features"]:
        spec = FeatureSpec(
            name=item["name"], family=item["family"], fn=item["fn"],
            params=item.get("params") or {}, reasoning=item["reasoning"],
        )
        if spec.family not in KNOWN_FAMILIES:
            raise ValueError(f"unknown family '{spec.family}' for {spec.name}")
        specs.append(spec)
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names in registry")
    return specs
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/features/test_registry.py -v` → 3 passed; full suite; ruff.

- [ ] **Step 6: Commit**

```bash
git add vectora/config vectora/features/registry.py tests/features/test_registry.py pyproject.toml uv.lock
git commit -m "feat: feature registry with 40 documented features across 7 families"
```

---

### Task 5: Feature computation functions (`families.py`)

**Files:**
- Create: `vectora/features/families.py`
- Test: `tests/features/test_families.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_families.py
import math

import polars as pl

from vectora.features import families, registry


def _panel():
    # 30 days, 2 symbols, deterministic prices; includes sector + first_seen
    rows = []
    for i in range(30):
        d = f"2026-06-{i + 1:02d}" if i < 30 else None
        rows.append(dict(symbol="AAA", date=d, open=100 + i, high=102 + i,
                         low=99 + i, close=100 + i, ycp=99 + i, trades=100,
                         value_mn=10.0, volume=1000 + 10 * i,
                         sector="Bank", first_seen="2020-01-01",
                         ret=(100 + i) / (99 + i) - 1))
        rows.append(dict(symbol="BBB", date=d, open=50, high=50.5, low=49.5,
                         close=50.0, ycp=50.0, trades=10, value_mn=0.5,
                         volume=200, sector="Bank", first_seen="2024-01-01",
                         ret=0.0))
    return pl.DataFrame(rows).with_columns(
        pl.col("date").str.to_date(), pl.col("first_seen").str.to_date())


def test_ret_nd_compounds_returns():
    df = families.apply(_panel(), "ret_5d", "ret_nd", {"days": 5})
    aaa = df.filter(pl.col("symbol") == "AAA").sort("date")
    # close 105 on day 6 vs close 100 on day 1 -> 5%
    assert abs(aaa["ret_5d"][5] - (105 / 100 - 1)) < 1e-9
    assert aaa["ret_5d"][3] is None  # not enough history


def test_zscore_flags_volume_spike():
    df = _panel().with_columns(
        pl.when((pl.col("symbol") == "BBB") & (pl.col("date") == pl.date(2026, 6, 30)))
        .then(5000).otherwise(pl.col("volume")).alias("volume"))
    out = families.apply(df, "volume_z_21d", "zscore_col",
                         {"col": "volume", "days": 21})
    spike = out.filter((pl.col("symbol") == "BBB")
                       & (pl.col("date") == pl.date(2026, 6, 30)))
    assert spike["volume_z_21d"][0] > 3.0


def test_cross_rank_is_within_date_and_in_unit_range():
    df = families.apply(_panel(), "ret_5d", "ret_nd", {"days": 5})
    out = families.apply(df, "ret_5d_xrank", "cross_rank", {"of": "ret_5d"})
    last = out.filter(pl.col("date") == pl.date(2026, 6, 30))
    vals = [v for v in last["ret_5d_xrank"].to_list() if v is not None]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert max(vals) > min(vals)  # AAA rising vs BBB flat -> different ranks


def test_every_registered_fn_exists_and_runs():
    df = _panel()
    for spec in registry.load():
        assert spec.fn in families.FNS, f"missing fn {spec.fn} for {spec.name}"
        df = families.apply(df, spec.name, spec.fn, spec.params)
        assert spec.name in df.columns
        col = df[spec.name]
        finite = [v for v in col.to_list() if v is not None]
        assert all(not (isinstance(v, float) and math.isinf(v)) for v in finite), \
            f"{spec.name} produced inf"
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement `vectora/features/families.py`**

```python
# vectora/features/families.py
"""Feature computation functions. Contract: every fn takes the panel frame
(sorted by symbol,date; includes symbol/date/open/high/low/close/ycp/trades/
value_mn/volume/ret and joined sector/first_seen) plus params, and returns
the frame with ONE new column named `name`. Per-symbol ops use .over("symbol");
per-date (cross-sectional) ops use .over("date"). All windows are trailing —
the leakage test (Task 7) enforces it."""
import polars as pl

_EPS = 1e-12


def _sym(expr: pl.Expr) -> pl.Expr:
    return expr.over("symbol")


# ---- momentum -------------------------------------------------------------
def ret_nd(df, name, days):
    logret = (pl.col("ret") + 1).log()
    comp = logret.rolling_sum(days).exp() - 1
    return df.with_columns(_sym(comp).alias(name))


def rsi(df, name, days):
    up = pl.when(pl.col("ret") > 0).then(pl.col("ret")).otherwise(0.0)
    dn = pl.when(pl.col("ret") < 0).then(-pl.col("ret")).otherwise(0.0)
    rs = _sym(up.rolling_mean(days)) / (_sym(dn.rolling_mean(days)) + _EPS)
    return df.with_columns((100 - 100 / (1 + rs)).alias(name))


def dist_from_rolling_max(df, name, days):
    e = pl.col("close") / _sym(pl.col("close").rolling_max(days)) - 1
    return df.with_columns(e.alias(name))


def dist_from_rolling_min(df, name, days):
    e = pl.col("close") / _sym(pl.col("close").rolling_min(days)) - 1
    return df.with_columns(e.alias(name))


# ---- volatility -----------------------------------------------------------
def ret_std(df, name, days):
    return df.with_columns(_sym(pl.col("ret").rolling_std(days)).alias(name))


def ratio_of_stds(df, name, short, long):
    e = _sym(pl.col("ret").rolling_std(short)) / (
        _sym(pl.col("ret").rolling_std(long)) + _EPS)
    return df.with_columns(e.alias(name))


def atr(df, name, days):
    prev_close = _sym(pl.col("close").shift(1))
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return df.with_columns(_sym(tr.rolling_mean(days)).alias(name))


def avg_range_pct(df, name, days):
    rng = (pl.col("high") - pl.col("low")) / (pl.col("close") + _EPS)
    return df.with_columns(_sym(rng.rolling_mean(days)).alias(name))


def limit_lock_count(df, name, days, band):
    locked = (pl.col("ret").abs() >= band).cast(pl.Int32)
    return df.with_columns(_sym(locked.rolling_sum(days)).alias(name))


# ---- liquidity ------------------------------------------------------------
def rolling_median_col(df, name, col, days):
    return df.with_columns(_sym(pl.col(col).rolling_median(days)).alias(name))


def amihud(df, name, days):
    daily = pl.col("ret").abs() / (pl.col("value_mn") + _EPS)
    return df.with_columns(_sym(daily.rolling_mean(days)).alias(name))


def zero_volume_days(df, name, days):
    z = (pl.col("volume").fill_null(0) == 0).cast(pl.Int32)
    return df.with_columns(_sym(z.rolling_sum(days)).alias(name))


def zscore_col(df, name, col, days):
    mean = _sym(pl.col(col).rolling_mean(days))
    std = _sym(pl.col(col).rolling_std(days))
    return df.with_columns(((pl.col(col) - mean) / (std + _EPS)).alias(name))


# ---- volume/flow ----------------------------------------------------------
def volume_ratio(df, name, short, long):
    e = _sym(pl.col("volume").rolling_mean(short)) / (
        _sym(pl.col("volume").rolling_mean(long)) + _EPS)
    return df.with_columns(e.alias(name))


def obv_slope(df, name, days):
    signed = pl.col("volume").fill_null(0) * pl.col("ret").sign().fill_null(0)
    obv = _sym(signed.cum_sum())
    slope = (obv - _sym(obv.shift(days))) / days
    norm = slope / (_sym(pl.col("volume").rolling_mean(days)) + _EPS)
    return df.with_columns(norm.alias(name))


def updown_volume_ratio(df, name, days):
    upv = pl.when(pl.col("ret") > 0).then(pl.col("volume")).otherwise(0)
    dnv = pl.when(pl.col("ret") < 0).then(pl.col("volume")).otherwise(0)
    e = _sym(upv.rolling_sum(days)) / (_sym(dnv.rolling_sum(days)) + _EPS)
    return df.with_columns(e.log1p().alias(name))  # log-compress the ratio


def vwap_deviation(df, name, days):
    # value_mn is in millions of taka; volume in shares -> vwap in taka
    vwap = _sym((pl.col("value_mn") * 1e6).rolling_sum(days)) / (
        _sym(pl.col("volume").rolling_sum(days)) + _EPS)
    return df.with_columns((pl.col("close") / (vwap + _EPS) - 1).alias(name))


# ---- cross-sectional (per-date) --------------------------------------------
def cross_rank(df, name, of):
    e = (pl.col(of).rank("average") / pl.col(of).count()).over("date")
    return df.with_columns(e.alias(name))


def sector_mean(df, name, of):
    return df.with_columns(
        pl.col(of).mean().over(["date", "sector"]).alias(name))


def minus_sector_mean(df, name, of):
    e = pl.col(of) - pl.col(of).mean().over(["date", "sector"])
    return df.with_columns(e.alias(name))


def market_breadth_above_ma(df, name, days):
    above = (pl.col("close") > _sym(pl.col("close").rolling_mean(days))).cast(pl.Int8)
    return df.with_columns(above.mean().over("date").alias(name))


# ---- calendar / structure ---------------------------------------------------
def day_of_week(df, name):
    return df.with_columns(pl.col("date").dt.weekday().alias(name))


def month_of_year(df, name):
    return df.with_columns(pl.col("date").dt.month().alias(name))


def days_since_first_seen(df, name):
    e = (pl.col("date") - pl.col("first_seen")).dt.total_days()
    return df.with_columns(e.alias(name))


def log_close(df, name):
    return df.with_columns(pl.col("close").log().alias(name))


def overnight_gap(df, name):
    prev = _sym(pl.col("close").shift(1))
    return df.with_columns((pl.col("open") / (prev + _EPS) - 1).alias(name))


def close_in_range(df, name):
    rng = pl.col("high") - pl.col("low")
    e = pl.when(rng > 0).then((pl.col("close") - pl.col("low")) / rng).otherwise(0.5)
    return df.with_columns(e.alias(name))


def dist_from_sma(df, name, days):
    e = pl.col("close") / (_sym(pl.col("close").rolling_mean(days)) + _EPS) - 1
    return df.with_columns(e.alias(name))


def sma_cross_state(df, name, short, long):
    e = (_sym(pl.col("close").rolling_mean(short))
         > _sym(pl.col("close").rolling_mean(long))).cast(pl.Int8)
    return df.with_columns(e.alias(name))


def ycp_adjustment_flag(df, name):
    prev = _sym(pl.col("close").shift(1))
    diverges = (
        pl.col("ycp").is_not_null() & prev.is_not_null()
        & ((pl.col("ycp") - prev).abs() / (prev + _EPS) > 0.005)
    )
    return df.with_columns(diverges.cast(pl.Int8).alias(name))


FNS = {f.__name__: f for f in [
    ret_nd, rsi, dist_from_rolling_max, dist_from_rolling_min,
    ret_std, ratio_of_stds, atr, avg_range_pct, limit_lock_count,
    rolling_median_col, amihud, zero_volume_days, zscore_col,
    volume_ratio, obv_slope, updown_volume_ratio, vwap_deviation,
    cross_rank, sector_mean, minus_sector_mean, market_breadth_above_ma,
    day_of_week, month_of_year, days_since_first_seen, log_close,
    overnight_gap, close_in_range, dist_from_sma, sma_cross_state,
    ycp_adjustment_flag,
]}


def apply(df: pl.DataFrame, name: str, fn: str, params: dict) -> pl.DataFrame:
    return FNS[fn](df, name, **params)
```

Debug note: if polars version differences bite (e.g. `rolling_sum` null handling or `dt.weekday()` range), adjust the IMPLEMENTATION to satisfy the tests; check polars docs for the installed version. `ret_nd` requires `min_samples` behavior where fewer than `days` observations yield null — that is polars' default (`min_samples=days`).

- [ ] **Step 4: Run tests** — `uv run pytest tests/features/test_families.py -v` → 4 passed; full suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/features/families.py tests/features/test_families.py
git commit -m "feat: 30 feature computation functions across 7 families"
```

---

### Task 6: Feature engine (registry → wide frame → parquet)

**Files:**
- Create: `vectora/features/engine.py`
- Modify: `.gitignore` (add `data/features/` — the parquet is ~100MB regenerable-in-seconds output, never committed)
- Test: `tests/features/test_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_engine.py
import polars as pl

from vectora import db as vdb
from vectora.features import engine, registry


def _seed(con, n_days=80):
    vdb.upsert(con, "symbols", [
        dict(symbol="AAA", name=None, sector="Bank", instrument_type="Equity",
             category="A", listing_status="active",
             first_seen="2020-01-01", last_seen="2026-07-09"),
    ])
    rows = []
    import datetime as dt
    d0 = dt.date(2026, 3, 1)
    for i in range(n_days):
        d = (d0 + dt.timedelta(days=i)).isoformat()
        px = 100 + (i % 7)
        rows.append(dict(symbol="AAA", date=d, open=px, high=px + 1, low=px - 1,
                         close=px, ltp=px, ycp=px - (i % 3 == 0), trades=50,
                         value_mn=5.0, volume=1000, source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)


def test_compute_produces_all_registered_columns(test_db, tmp_path):
    _seed(test_db)
    out_path = tmp_path / "features.parquet"
    df = engine.compute(test_db, out_path=out_path)
    expected = {s.name for s in registry.load()}
    assert expected <= set(df.columns)
    assert {"symbol", "date", "ret"} <= set(df.columns)
    assert out_path.exists()
    assert pl.read_parquet(out_path).height == df.height


def test_compute_row_count_matches_panel(test_db, tmp_path):
    _seed(test_db, n_days=40)
    df = engine.compute(test_db, out_path=tmp_path / "f.parquet")
    n = test_db.execute("SELECT count(*) FROM prices").fetchone()[0]
    assert df.height == n
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/features/engine.py
"""Feature engine: panel + symbol metadata -> every registered feature ->
wide parquet. Full recompute per run (1M rows x 40 features is seconds in
polars); incremental computation is deliberate YAGNI at this scale."""
from pathlib import Path

import polars as pl

from vectora.features import base, families, registry
from vectora.settings import FEATURES_DIR

DEFAULT_OUT = FEATURES_DIR / "features.parquet"


def compute(con, out_path: Path = DEFAULT_OUT,
            specs: list | None = None) -> pl.DataFrame:
    panel = base.load_panel(con)
    meta = con.execute(
        "SELECT symbol, sector, first_seen FROM symbols").pl().with_columns(
        pl.col("first_seen").cast(pl.Date))
    df = panel.join(meta, on="symbol", how="left").sort(["symbol", "date"])
    for spec in (specs or registry.load()):
        df = families.apply(df, spec.name, spec.fn, spec.params)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    return df
```

Append `data/features/` to `.gitignore`.

- [ ] **Step 4: Run tests** — 2 passed; full suite; ruff.

- [ ] **Step 5: Run against the real database (smoke)**

Run: `uv run python -c "from vectora import db; from vectora.features import engine; con = db.connect('data/vectora.duckdb'); db.init_schema(con); df = engine.compute(con); print(df.height, len(df.columns)); con.close()"`
Expected: ~1,064,000 rows, 50+ columns, well under 2 minutes. Report the timing.

- [ ] **Step 6: Commit**

```bash
git add vectora/features/engine.py tests/features/test_engine.py .gitignore
git commit -m "feat: registry-driven feature engine writing zstd parquet"
```

---

### Task 7: Leakage guard test

**Files:**
- Test: `tests/features/test_leakage.py` (test-only task — the guarantee the whole ML phase rests on)

- [ ] **Step 1: Write the test (it should PASS immediately if Task 5/6 are correct — its value is pinning the invariant forever)**

```python
# tests/features/test_leakage.py
"""Anti-leakage invariant (spec §8): features for date t must not change
when future data (t+1...) changes. Every feature uses trailing windows or
per-date cross-sections only; this test catches any future violation."""
import datetime as dt

import polars as pl

from vectora import db as vdb
from vectora.features import engine


def _seed(con, n_days, close_fn):
    vdb.upsert(con, "symbols", [
        dict(symbol="AAA", name=None, sector="Bank", instrument_type="Equity",
             category="A", listing_status="active",
             first_seen="2020-01-01", last_seen="2026-12-31"),
        dict(symbol="BBB", name=None, sector="Textile", instrument_type="Equity",
             category="B", listing_status="active",
             first_seen="2021-01-01", last_seen="2026-12-31"),
    ])
    rows = []
    d0 = dt.date(2026, 1, 1)
    for i in range(n_days):
        d = (d0 + dt.timedelta(days=i)).isoformat()
        for sym, base_px in (("AAA", 100.0), ("BBB", 40.0)):
            px = close_fn(i, base_px)
            rows.append(dict(symbol=sym, date=d, open=px, high=px * 1.01,
                             low=px * 0.99, close=px, ltp=px, ycp=px,
                             trades=30, value_mn=3.0, volume=500 + i,
                             source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)


def test_future_data_does_not_change_past_features(tmp_path):
    def close_fn(i, base):
        return base * (1 + 0.001 * (i % 9))

    con1 = vdb.connect(tmp_path / "a.duckdb")
    vdb.init_schema(con1, backfill_parquet=tmp_path / "none.parquet")
    _seed(con1, 70, close_fn)
    f1 = engine.compute(con1, out_path=tmp_path / "f1.parquet")
    con1.close()

    def close_fn2(i, base):  # identical history, WILD different future
        return close_fn(i, base) if i < 70 else base * 3.0

    con2 = vdb.connect(tmp_path / "b.duckdb")
    vdb.init_schema(con2, backfill_parquet=tmp_path / "none.parquet")
    _seed(con2, 100, close_fn2)
    f2 = engine.compute(con2, out_path=tmp_path / "f2.parquet")
    con2.close()

    cutoff = dt.date(2026, 1, 1) + dt.timedelta(days=69)
    a = f1.filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
    b = f2.filter(pl.col("date") <= cutoff).sort(["symbol", "date"])
    assert a.height == b.height
    for col in a.columns:
        if col in ("symbol", "date", "sector", "first_seen"):
            continue
        av, bv = a[col].to_list(), b[col].to_list()
        for x, y in zip(av, bv, strict=True):
            if x is None and y is None:
                continue
            assert x == y or abs(x - y) < 1e-12, \
                f"LEAKAGE: {col} changed for a past date when future data changed"
```

- [ ] **Step 2: Run it** — `uv run pytest tests/features/test_leakage.py -v` → 1 passed. If it FAILS, a feature is forward-looking — fix that feature in families.py, never weaken this test.

- [ ] **Step 3: Commit**

```bash
git add tests/features/test_leakage.py
git commit -m "test: leakage guard - past features immune to future data"
```

---

### Task 8: Label grid (`labels.py`)

**Files:**
- Create: `vectora/labels.py`
- Test: `tests/test_labels.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_labels.py
import datetime as dt

import polars as pl

from vectora import labels


def _panel():
    rows = []
    d0 = dt.date(2026, 1, 1)
    closes = [100, 101, 103, 111, 108, 104, 100, 100, 100, 100, 100, 100]
    for i, c in enumerate(closes):
        rows.append(dict(symbol="AAA", date=d0 + dt.timedelta(days=i),
                         close=float(c)))
    return pl.DataFrame(rows)


def test_gain_label_hits_within_horizon():
    df = labels.make_labels(_panel(), thresholds=(0.05, 0.10), horizons=(3, 5))
    row0 = df.filter(pl.col("date") == dt.date(2026, 1, 1))
    # from close 100: max close within 3 days = 111 -> both thresholds hit
    assert row0["y_g5_h3"][0] == 1 and row0["y_g10_h3"][0] == 1


def test_gain_label_miss():
    df = labels.make_labels(_panel(), thresholds=(0.10,), horizons=(3,))
    row4 = df.filter(pl.col("date") == dt.date(2026, 1, 5))  # close 108
    # next 3 closes: 104,100,100 -> no +10%
    assert row4["y_g10_h3"][0] == 0


def test_label_null_when_horizon_incomplete():
    df = labels.make_labels(_panel(), thresholds=(0.05,), horizons=(5,))
    last = df.sort("date").tail(5)
    assert last["y_g5_h5"].null_count() == 5  # fewer than 5 future closes


def test_downside_label():
    df = labels.make_labels(_panel(), thresholds=(0.05,), horizons=(3,),
                            downside=True)
    row3 = df.filter(pl.col("date") == dt.date(2026, 1, 4))  # close 111
    # next 3 closes: 108,104,100 -> min 100 = -9.9% -> 5% drawdown hit
    assert row3["y_d5_h3"][0] == 1
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/labels.py
"""Forward-return label grid (spec §9.1): y_gX_hH = 1 if the max close over
the next H trading rows gains >= X% vs today's close; y_dX_hH mirrors for
drawdowns. Labels are null when fewer than H future rows exist (end of data)
— unresolved, not negative. Uses raw closes; corporate-action gaps inside
the forward window are a documented Phase 2 approximation (base.py note)."""
import polars as pl


def _fwd_extreme(h: int, kind: str) -> pl.Expr:
    shifts = [pl.col("close").shift(-k).over("symbol") for k in range(1, h + 1)]
    agg = pl.max_horizontal(shifts) if kind == "max" else pl.min_horizontal(shifts)
    complete = pl.col("close").shift(-h).over("symbol").is_not_null()
    return pl.when(complete).then(agg).otherwise(None)


def make_labels(panel: pl.DataFrame, thresholds=(0.03, 0.05, 0.10, 0.20),
                horizons=(1, 3, 5, 10, 30), downside: bool = False) -> pl.DataFrame:
    df = panel.sort(["symbol", "date"])
    cols = []
    for h in horizons:
        fwd_max = _fwd_extreme(h, "max")
        fwd_min = _fwd_extreme(h, "min")
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

- [ ] **Step 4: Run tests** — 4 passed; full suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/labels.py tests/test_labels.py
git commit -m "feat: forward-return label grid with unresolved-horizon nulls"
```

---

### Task 9: Walk-forward splitter

**Files:**
- Create: `vectora/train/__init__.py` (empty), `vectora/train/walkforward.py`
- Test: `tests/train/__init__.py` (empty), `tests/train/test_walkforward.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/train/test_walkforward.py
import datetime as dt

from vectora.train import walkforward as wf


def _dates(n):
    d0 = dt.date(2020, 1, 1)
    return [d0 + dt.timedelta(days=i) for i in range(n)]


def test_expanding_splits_with_embargo():
    splits = wf.splits(_dates(1500), min_train_days=750, test_days=126,
                       embargo_days=30)
    assert len(splits) >= 4
    for s in splits:
        assert s.train_start < s.train_end < s.test_start <= s.test_end
        # embargo: gap between train end and test start
        assert (s.test_start - s.train_end).days >= 30
    # expanding: every train window starts at the beginning
    assert len({s.train_start for s in splits}) == 1
    # test windows are consecutive and non-overlapping
    for a, b in zip(splits, splits[1:], strict=False):
        assert b.test_start > a.test_end


def test_no_split_when_history_too_short():
    assert wf.splits(_dates(400), min_train_days=750, test_days=126,
                     embargo_days=30) == []


def test_assign_rows():
    splits = wf.splits(_dates(1500), min_train_days=750, test_days=126,
                       embargo_days=30)
    s = splits[0]
    assert wf.role(s, s.train_start) == "train"
    assert wf.role(s, s.train_end + dt.timedelta(days=1)) == "embargo"
    assert wf.role(s, s.test_start) == "test"
    assert wf.role(s, s.test_end + dt.timedelta(days=1)) == "future"
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/train/walkforward.py
"""Expanding-window walk-forward splits with an embargo gap (spec §10.2).

The embargo (>= max label horizon, default 30 days) sits between train end
and test start so no training label's forward window overlaps test data.
Never use random k-fold on time series — this module is the only sanctioned
splitter."""
import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date


def splits(dates: list[dt.date], min_train_days: int = 750,
           test_days: int = 126, embargo_days: int = 30) -> list[Split]:
    if not dates:
        return []
    dates = sorted(set(dates))
    start = dates[0]
    out = []
    train_end_i = min_train_days - 1
    while True:
        test_start_i = train_end_i + embargo_days + 1
        test_end_i = test_start_i + test_days - 1
        if test_end_i >= len(dates):
            break
        out.append(Split(
            train_start=start,
            train_end=dates[train_end_i],
            test_start=dates[test_start_i],
            test_end=dates[test_end_i],
        ))
        train_end_i = test_end_i - embargo_days  # next train absorbs this test
    return out


def role(split: Split, d: dt.date) -> str:
    if d <= split.train_end:
        return "train"
    if d < split.test_start:
        return "embargo"
    if d <= split.test_end:
        return "test"
    return "future"
```

Debug note: if `test_expanding_splits_with_embargo` fails on the consecutive-windows assertion, check the `train_end_i = test_end_i - embargo_days` stepping — the intent is that fold k+1's training data ends where fold k's test window ended (minus embargo), so successive test windows tile the timeline without gaps or overlap.

- [ ] **Step 4: Run tests** — 3 passed; full suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/train tests/train
git commit -m "feat: expanding walk-forward splitter with embargo"
```

---

### Task 10: Models + calibration + metrics

**Files:**
- Create: `vectora/train/models.py`
- Test: `tests/train/test_models.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/train/test_models.py
import numpy as np

from vectora.train import models


def _synthetic(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.3 * rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_logistic_baseline_learns():
    X, y = _synthetic()
    m = models.fit_logistic(X[:3000], y[:3000])
    p = models.predict(m, X[3000:])
    assert models.auc(y[3000:], p) > 0.80


def test_lgbm_learns_and_beats_chance():
    X, y = _synthetic()
    m = models.fit_lgbm(X[:2500], y[:2500], X[2500:3000], y[2500:3000])
    p = models.predict(m, X[3000:])
    assert models.auc(y[3000:], p) > 0.80


def test_isotonic_calibration_improves_or_maintains_brier():
    X, y = _synthetic()
    m = models.fit_lgbm(X[:2000], y[:2000], X[2000:2500], y[2000:2500])
    p_val = models.predict(m, X[2500:3200])
    cal = models.fit_calibrator(p_val, y[2500:3200])
    p_test_raw = models.predict(m, X[3200:])
    p_test_cal = models.apply_calibrator(cal, p_test_raw)
    assert models.brier(y[3200:], p_test_cal) <= models.brier(y[3200:], p_test_raw) + 0.005
    assert (p_test_cal >= 0).all() and (p_test_cal <= 1).all()


def test_reliability_table_shape():
    X, y = _synthetic()
    m = models.fit_logistic(X[:3000], y[:3000])
    p = models.predict(m, X[3000:])
    tab = models.reliability_table(y[3000:], p, bins=10)
    assert len(tab) <= 10
    for row in tab:
        assert set(row) == {"bin_lo", "bin_hi", "n", "p_mean", "y_rate"}
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
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
```

Note: LightGBM handles NaN features natively — do NOT impute for LightGBM (missingness is signal, e.g. null volume = no trade); the imputer is logistic-only.

- [ ] **Step 4: Run tests** — 4 passed; full suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/train/models.py tests/train/test_models.py
git commit -m "feat: logistic baseline + LightGBM + isotonic calibration + metrics"
```

---

### Task 11: Trainer, report, CLI, workflow

**Files:**
- Create: `vectora/train/trainer.py`, `.github/workflows/train.yml`
- Modify: `vectora/__main__.py` (add `train` stage)
- Test: `tests/train/test_trainer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/train/test_trainer.py
import datetime as dt

import numpy as np

from vectora import db as vdb
from vectora.train import trainer


def _seed_realistic(con, n_days=1300, n_syms=12, seed=3):
    rng = np.random.default_rng(seed)
    vdb.upsert(con, "symbols", [
        dict(symbol=f"S{i:02d}", name=None, sector="Bank" if i % 2 else "Textile",
             instrument_type="Equity", category="A", listing_status="active",
             first_seen="2019-01-01", last_seen="2026-12-31")
        for i in range(n_syms)])
    rows = []
    d0 = dt.date(2021, 1, 1)
    px = {f"S{i:02d}": 100.0 for i in range(n_syms)}
    for day in range(n_days):
        d = (d0 + dt.timedelta(days=day)).isoformat()
        for sym in px:
            px[sym] *= float(np.exp(rng.normal(0, 0.02)))
            p = round(max(px[sym], 1.0), 2)
            rows.append(dict(symbol=sym, date=d, open=p, high=p * 1.02,
                             low=p * 0.98, close=p, ltp=p, ycp=p,
                             trades=int(rng.integers(20, 200)),
                             value_mn=float(rng.uniform(2, 30)),
                             volume=int(rng.integers(1000, 50000)),
                             source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)


def test_trainer_end_to_end_writes_report_registry_and_artifacts(test_db, tmp_path):
    _seed_realistic(test_db)
    result = trainer.run(
        test_db, target="g5_h10",
        models_dir=tmp_path / "models", reports_dir=tmp_path / "reports",
        features_path=tmp_path / "features.parquet",
        min_train_days=600, test_days=120, embargo_days=10,
        min_median_value_mn=0.1)
    assert result["folds"] >= 2
    assert 0 < result["lgbm_brier"] < 1 and 0 < result["logistic_brier"] < 1
    report = list((tmp_path / "reports").glob("train_g5_h10_*.md"))
    assert len(report) == 1
    text = report[0].read_text(encoding="utf-8")
    assert "Brier" in text and "reliability" in text.lower()
    n = test_db.execute("SELECT count(*) FROM model_registry").fetchone()[0]
    assert n == 2  # one row per family
    art = list((tmp_path / "models").glob("*/meta.json"))
    assert len(art) == 2


def test_trainer_refuses_unknown_target(test_db, tmp_path):
    import pytest
    with pytest.raises(ValueError, match="target"):
        trainer.run(test_db, target="nonsense",
                    models_dir=tmp_path, reports_dir=tmp_path,
                    features_path=tmp_path / "f.parquet")
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
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
        pooled["lgbm"][0].extend(y_te)
        pooled["lgbm"][1].extend(M.apply_calibrator(cal, M.predict(lgbm, X_te)))
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
    for fam in ("logistic", "lgbm"):
        model_id = f"{target}_{fam}_{run_id}_{uuid.uuid4().hex[:6]}"
        art = models_dir / model_id
        art.mkdir(parents=True, exist_ok=True)
        if fam == "lgbm":
            last_models["lgbm"].booster_.save_model(str(art / "lgbm.txt"))
            import pickle
            (art / "calibrator.pkl").write_bytes(
                pickle.dumps(last_models["cal"]))
        else:
            import pickle
            (art / "logistic.pkl").write_bytes(
                pickle.dumps(last_models["logistic"]))
        (art / "meta.json").write_text(json.dumps({
            "model_id": model_id, "family": fam, "target": target,
            "features": feat_names, "train_end": str(last_models["train_end"]),
            "metrics": metrics[fam] | {"reliability": None},
        }, indent=1), encoding="utf-8")
        vdb.upsert(con, "model_registry", [{
            "model_id": model_id, "family": fam, "target": target,
            "trained_at": dt.datetime.now().isoformat(),
            "train_end": str(last_models["train_end"]),
            "metrics": json.dumps(metrics[fam] | {"reliability": None}),
            "artifact_dir": str(art), "active": False,
        }])

    report = _render_report(target, folds, metrics)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"train_{target}_{run_id}.md").write_text(
        report, encoding="utf-8")

    return {"folds": len(folds), "target": target,
            "lgbm_brier": metrics["lgbm"]["brier"],
            "logistic_brier": metrics["logistic"]["brier"],
            "lgbm_auc": metrics["lgbm"]["auc"],
            "logistic_auc": metrics["logistic"]["auc"]}


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
```

- [ ] **Step 4: Add the CLI stage**

In `vectora/__main__.py`: change `run.add_argument("stage", choices=["eod"])` to `choices=["eod", "train"]`, add `run.add_argument("--target", default="g5_h10")`, and after the eod branch add:

```python
    if args.command == "run" and args.stage == "train":
        from vectora import db as vdb
        from vectora.settings import DB_PATH
        from vectora.train import trainer
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = trainer.run(con, target=args.target)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0 if result["lgbm_brier"] < result["logistic_brier"] else 1
```

- [ ] **Step 5: Write `.github/workflows/train.yml`**

```yaml
name: train

on:
  workflow_dispatch:
    inputs:
      target:
        description: "Label target, e.g. g5_h10"
        required: false
        default: "g5_h10"

concurrency:
  group: pipeline-writes
  cancel-in-progress: false

permissions:
  contents: write

jobs:
  train:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - name: Install
        run: uv sync --frozen
      - name: Train
        id: train
        continue-on-error: true
        run: uv run python -m vectora run train --target "${{ github.event.inputs.target }}"
      - name: Commit artifacts
        if: always()
        run: |
          git config user.name "vectora-bot"
          git config user.email "vectora-bot@users.noreply.github.com"
          git add -f models reports data/vectora.duckdb
          git diff --cached --quiet || git commit -m "train: ${{ github.event.inputs.target }} $(TZ=Asia/Dhaka date +%F)"
          git pull --rebase --autostash
          git push
      - name: Surface training failure
        if: steps.train.outcome == 'failure'
        run: exit 1
```

- [ ] **Step 6: Run tests** — `uv run pytest tests/train/test_trainer.py -v` → 2 passed (the end-to-end test takes ~1-2 min; that's expected). Full suite; ruff.

- [ ] **Step 7: Commit**

```bash
git add vectora/train/trainer.py vectora/__main__.py .github/workflows/train.yml tests/train/test_trainer.py
git commit -m "feat: walk-forward trainer with calibrated LightGBM vs logistic report"
```

---

### Task 12: Real training run + Phase 2 acceptance

**Files:** none new — this task runs the system on real data and commits results.

- [ ] **Step 1: Train the primary target on real data**

Run: `uv run python -m vectora run train --target g5_h10`
Expected: JSON with folds ≥ 6, both Brier scores printed. Runtime up to ~15 min locally. Exit 0 means LightGBM beat the baseline.

- [ ] **Step 2: Read the report**

Open `reports/train_g5_h10_<today>.md`. Sanity checks:
- base_rate for +5%-within-10-days should land roughly in 0.15–0.45 on DSE history — far outside that range suggests a label bug;
- reliability table: realized rates should increase down the bins (rough monotonicity);
- AUC realistically 0.55–0.70. AUC > 0.8 on daily equity data almost certainly means leakage — STOP and investigate (re-run the Task 7 leakage test, inspect top LightGBM features via `booster_.feature_importance()`).

- [ ] **Step 3: Train the secondary target**

Run: `uv run python -m vectora run train --target g10_h30`
Same sanity checks.

- [ ] **Step 4: Commit results and report the verdict**

```bash
git add -f models reports data/vectora.duckdb
git commit -m "train: first real g5_h10 + g10_h30 runs with walk-forward comparison"
```

Report to the user: folds, Brier/AUC for both models on both targets, the PASS/FAIL verdict lines, and any sanity-check anomalies. **Phase 2's exit criterion is the PASS verdict + a rough-monotone reliability table.** If FAIL (baseline wins), that is a legitimate, reportable research result — do not tweak until it passes; report honestly and stop for direction.

---

## Execution notes

- Tasks are strictly ordered 1→12; each leaves the suite green and is committed separately.
- Task 1 changes real data files (`data/vectora.duckdb` shrinks, parquet appears) — its migration step runs once, everything after is code.
- Total new tests: ~25. Suite should end around 100 tests.
- The soak (Phase 1 Task 15) keeps running on `main` unaffected; this branch only merges after the soak passes AND Task 12's verdict is in.
- If the spend limit kills a subagent mid-task, check `git status` before re-dispatching: finish uncommitted TDD cycles rather than restarting them.

