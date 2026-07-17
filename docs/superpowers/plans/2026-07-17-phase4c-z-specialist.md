# Vectora Phase 4C: Z-Category Specialist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Daily pump-phase scoring across the board, a pre-announcement footprint detector trained on the 31,904-event archive, and a Z-watch surface in journal/digest plus pump warnings inside signal explanations (spec §13, §12.3).

**Architecture:** `zmod.pump` classifies each symbol-day into quiet/markup/distribution/collapse via ordered rules over existing features (21/10/63-day returns, volume ratio, OBV slope) and scores pump risk 0–100 as the product of cross-sectional percentile ranks of run-up and volume expansion, boosted for Z/B/N categories. `zmod.footprint` computes each material event's pre-announcement footprint (trailing 5-day volume z and return, measured the day BEFORE the event) into `event_footprints`, then flags current symbols whose footprint exceeds the historical 75th percentile — a purely statistical "resembles pre-announcement accumulation" warning, never an accusation (spec's public-data-only rule). `zmod.scan` runs both daily into a `zwatch` table consumed by journal, digest, and the predict engine (signals on pump-flagged names get an explanation warning; they are NOT blocked — warn, don't gate). Everything reads the existing feature engine output; no new data sources.

**Tech Stack:** existing only. No new dependencies.

**Existing contracts (all on `main`):**
- `vectora/features/engine.py`: `compute(con, out_path=None) -> pl.DataFrame` with per-symbol-day columns incl. `ret, ret_10d, ret_21d, ret_63d, vol_ratio_5_21, obv_slope_21d, volume_z_21d` (see vectora/config/features.yaml). ~4s full recompute.
- `event_labels(event_id, event_type, materiality)` — 31,904 rows; materiality 3 = price-sensitive (dividend_declared, earnings_release, rights_issue, category_change, trading_halt). `events(id, post_date, symbol, ...)`.
- `symbols.category` ('A'/'B'/'N'/'Z'/NULL). `vectora/predict/engine.py` `run_predict` (builds `expls` list of dicts with key "rendered"). `vault/generator.py` journal builder + `_write_machine`. `alerts/digest.py` `build`. `__main__.py` stages …/vault/regime/events.
- Test seeding rule: bulk-insert synthetic prices via registered polars frame, never `vdb.upsert` for >1k rows.
- Branch: `git checkout main && git pull && git checkout -b phase-4c-zmod`. Fast tests `uv run pytest -m "not slow"` (currently 169).

**File structure:**

```
vectora/zmod/__init__.py, pump.py, footprint.py, scan.py
vectora/db.py                      # + zwatch, event_footprints tables
vectora/predict/engine.py          # + pump warning appended to rendered expl
vectora/vault/generator.py         # + Z-watch journal section
vectora/alerts/digest.py           # + Z-watch digest section
vectora/__main__.py                # + zscan stage
.github/workflows/eod-pipeline.yml # + Zscan step (after Regime, before Predict)
tests/zmod/__init__.py, test_pump.py, test_footprint.py, test_scan.py
```

---

### Task 1: Pump-phase scorer

**Files:**
- Modify: `vectora/db.py` (SCHEMA)
- Create: `vectora/zmod/__init__.py` (empty), `vectora/zmod/pump.py`
- Test: `tests/zmod/__init__.py` (empty), `tests/zmod/test_pump.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/zmod/test_pump.py
import polars as pl

from vectora.zmod import pump


def _frame(rows):
    base = dict(ret_10d=0.0, ret_21d=0.0, ret_63d=0.0, vol_ratio_5_21=1.0,
                obv_slope_21d=0.1, volume_z_21d=0.0)
    return pl.DataFrame([{**base, **r} for r in rows])


def test_phases():
    df = _frame([
        {"symbol": "QUIET"},
        {"symbol": "MARKUP", "ret_21d": 0.40, "vol_ratio_5_21": 2.0},
        {"symbol": "DIST", "ret_21d": 0.40, "vol_ratio_5_21": 2.0,
         "obv_slope_21d": -0.3},
        {"symbol": "COLLAPSE", "ret_10d": -0.25, "ret_63d": 0.5},
    ])
    out = pump.phase_and_score(df, categories={})
    phases = dict(zip(out["symbol"].to_list(), out["phase"].to_list()))
    assert phases == {"QUIET": "quiet", "MARKUP": "markup",
                      "DIST": "distribution", "COLLAPSE": "collapse"}


def test_score_ranks_runners_highest_and_boosts_z():
    rows = [{"symbol": f"S{i}", "ret_21d": 0.01 * i,
             "vol_ratio_5_21": 0.9 + 0.05 * i} for i in range(20)]
    rows.append({"symbol": "ZPUMP", "ret_21d": 0.45, "vol_ratio_5_21": 3.0})
    rows.append({"symbol": "APUMP", "ret_21d": 0.45, "vol_ratio_5_21": 3.0})
    out = pump.phase_and_score(_frame(rows),
                               categories={"ZPUMP": "Z", "APUMP": "A"})
    scores = dict(zip(out["symbol"].to_list(), out["score"].to_list()))
    assert scores["ZPUMP"] > scores["APUMP"] > scores["S05"]
    assert scores["ZPUMP"] <= 100.0
    assert scores["S00"] < 20


def test_null_features_score_zero_not_crash():
    df = _frame([{"symbol": "NEWLIST", "ret_21d": None,
                  "vol_ratio_5_21": None}])
    out = pump.phase_and_score(df, categories={"NEWLIST": "Z"})
    assert out["score"][0] == 0.0
    assert out["phase"][0] == "quiet"


def test_tables_exist(test_db):
    tables = {r[0] for r in test_db.execute("SHOW TABLES").fetchall()}
    assert {"zwatch", "event_footprints"} <= tables
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/zmod -v` → FAIL.

- [ ] **Step 3: Implement**

Append to `SCHEMA` in `vectora/db.py`:

```sql
CREATE TABLE IF NOT EXISTS zwatch (
    date DATE, symbol TEXT, kind TEXT,       -- 'pump' | 'footprint'
    score DOUBLE, phase TEXT, detail TEXT,   -- detail: JSON
    PRIMARY KEY (date, symbol, kind)
);
CREATE TABLE IF NOT EXISTS event_footprints (
    event_id TEXT PRIMARY KEY,
    pre_vol_z DOUBLE, pre_ret DOUBLE,
    computed_at TIMESTAMP DEFAULT current_timestamp
);
```

Create `vectora/zmod/pump.py`:

```python
"""Pump-phase classification and scoring (spec §13).

Phases are ordered rules over one day's cross-section (first match wins):
collapse (post-run crash) > distribution (run-up, volume no longer
confirming) > markup (run-up on expanding volume) > quiet. The 0-100 score
is the product of cross-sectional percentile ranks of 21d run-up and 5/21
volume expansion — a name must be extreme on BOTH to score high — with a
1.25x boost for Z/B/N categories (documented manipulation incidence).
Purely descriptive of public data; a warning surface, never an accusation.
"""
import polars as pl

RUNUP = 0.25
VOL_EXPAND = 1.3
COLLAPSE_DROP = -0.15
COLLAPSE_PRIOR_RUN = 0.20
BOOST_CATEGORIES = {"Z", "B", "N"}
BOOST = 1.25


def phase_and_score(day: pl.DataFrame, categories: dict) -> pl.DataFrame:
    df = day.with_columns(
        pl.col("ret_10d").fill_null(0.0), pl.col("ret_21d").fill_null(0.0),
        pl.col("ret_63d").fill_null(0.0),
        pl.col("vol_ratio_5_21").fill_null(1.0),
        pl.col("obv_slope_21d").fill_null(0.0),
    )
    phase = (
        pl.when((pl.col("ret_10d") < COLLAPSE_DROP)
                & (pl.col("ret_63d") > COLLAPSE_PRIOR_RUN))
        .then(pl.lit("collapse"))
        .when((pl.col("ret_21d") > RUNUP) & (pl.col("obv_slope_21d") < 0))
        .then(pl.lit("distribution"))
        .when((pl.col("ret_21d") > RUNUP)
              & (pl.col("vol_ratio_5_21") > VOL_EXPAND))
        .then(pl.lit("markup"))
        .otherwise(pl.lit("quiet"))
    )
    n = df.height
    rank_ret = pl.col("ret_21d").rank("average") / n
    rank_vol = pl.col("vol_ratio_5_21").rank("average") / n
    df = df.with_columns(phase.alias("phase"),
                         (rank_ret * rank_vol * 100).alias("_raw"))
    boost = pl.col("symbol").map_elements(
        lambda s: BOOST if categories.get(s) in BOOST_CATEGORIES else 1.0,
        return_dtype=pl.Float64)
    df = df.with_columns(
        (pl.col("_raw") * boost).clip(0.0, 100.0).alias("score"))
    # null-heavy rows (new listings) carry no evidence: zero them
    null_mask = day["ret_21d"].is_null() & day["vol_ratio_5_21"].is_null()
    df = df.with_columns(
        pl.when(pl.Series(null_mask)).then(0.0)
        .otherwise(pl.col("score")).alias("score"))
    return df.select([c for c in df.columns if c != "_raw"])
```

- [ ] **Step 4: Run tests** — 4 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/db.py vectora/zmod tests/zmod
git commit -m "feat: pump-phase scorer with cross-sectional ranks and Z-boost"
```

---

### Task 2: Pre-announcement footprint detector

**Files:**
- Create: `vectora/zmod/footprint.py`
- Test: `tests/zmod/test_footprint.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/zmod/test_footprint.py
import datetime as dt

import polars as pl

from vectora import db as vdb
from vectora.zmod import footprint


def _features(rows):
    """Minimal feature frame: symbol, date, ret, volume_z_21d."""
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _mk(symbol, d0, days, volz=0.0, ret=0.0, spike_days=(), spike_volz=4.0):
    out = []
    for i in range(days):
        d = d0 + dt.timedelta(days=i)
        out.append(dict(symbol=symbol, date=d,
                        ret=ret, volume_z_21d=spike_volz
                        if i in spike_days else volz))
    return out


def test_event_footprint_measures_pre_event_window(test_db):
    d0 = dt.date(2026, 6, 1)
    feats = _features(_mk("GP", d0, 20, spike_days=(10, 11, 12, 13, 14)))
    # event on day 15: prior 5 days are all spiked
    vdb.upsert(test_db, "events", [dict(
        id="ev1", post_date=(d0 + dt.timedelta(days=15)).isoformat(),
        symbol="GP", title="GP: Dividend Declaration", body="",
        source="dse_news")])
    vdb.upsert(test_db, "event_labels", [dict(
        event_id="ev1", event_type="dividend_declared", materiality=3)])
    result = footprint.compute_event_footprints(test_db, feats)
    assert result == {"computed": 1}
    row = test_db.execute(
        "SELECT pre_vol_z FROM event_footprints").fetchone()
    assert abs(row[0] - 4.0) < 1e-9      # mean of the 5 spiked days


def test_incremental_skips_done_events(test_db):
    d0 = dt.date(2026, 6, 1)
    feats = _features(_mk("GP", d0, 20, spike_days=(12,)))
    vdb.upsert(test_db, "events", [dict(
        id="ev1", post_date=(d0 + dt.timedelta(days=15)).isoformat(),
        symbol="GP", title="t", body="", source="dse_news")])
    vdb.upsert(test_db, "event_labels", [dict(
        event_id="ev1", event_type="earnings_release", materiality=3)])
    footprint.compute_event_footprints(test_db, feats)
    assert footprint.compute_event_footprints(test_db, feats) == {"computed": 0}


def test_daily_watch_flags_footprint_lookalikes(test_db):
    d0 = dt.date(2026, 6, 1)
    # historical footprints: three events with pre_vol_z 1.0, 2.0, 3.0
    for i, v in enumerate((1.0, 2.0, 3.0)):
        vdb.upsert(test_db, "event_footprints", [dict(
            event_id=f"e{i}", pre_vol_z=v, pre_ret=0.02)])
    today = d0 + dt.timedelta(days=9)
    feats = _features(
        _mk("HOT", d0, 10, spike_days=(5, 6, 7, 8, 9), spike_volz=5.0,
            ret=0.01)
        + _mk("COLD", d0, 10, volz=0.1))
    flagged = footprint.daily_watch(test_db, feats, today.isoformat())
    assert [f["symbol"] for f in flagged] == ["HOT"]
    f = flagged[0]
    assert f["kind"] == "footprint" and f["score"] >= 5.0


def test_daily_watch_empty_without_history(test_db):
    d0 = dt.date(2026, 6, 1)
    feats = _features(_mk("HOT", d0, 10, spike_days=(9,), spike_volz=9.0))
    assert footprint.daily_watch(
        test_db, feats, (d0 + dt.timedelta(days=9)).isoformat()) == []
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/zmod/footprint.py
"""Pre-announcement footprint detection (spec §12.3).

For every price-sensitive event (materiality 3), record the trailing
5-day mean volume z-score and compounded return measured the day BEFORE
the announcement. The daily watch then flags symbols whose current
trailing footprint exceeds the historical 75th percentile of those
pre-event footprints with positive drift — 'this pattern statistically
resembles what public data looked like before past announcements'.
Descriptive of securities, never an accusation about persons.
"""
import json

import polars as pl

from vectora import db as vdb

WINDOW = 5
PCTL = 0.75
TOP_N = 10
MIN_HISTORY = 3   # need at least this many historical footprints to judge


def _trailing(feats: pl.DataFrame) -> pl.DataFrame:
    """Adds fp_vol_z / fp_ret = stats over the trailing WINDOW rows
    INCLUDING the current row (per symbol)."""
    f = feats.sort(["symbol", "date"])
    return f.with_columns(
        pl.col("volume_z_21d").rolling_mean(WINDOW).over("symbol")
        .alias("fp_vol_z"),
        ((pl.col("ret").fill_null(0) + 1).log().rolling_sum(WINDOW)
         .over("symbol").exp() - 1).alias("fp_ret"),
    )


def compute_event_footprints(con, feats: pl.DataFrame) -> dict:
    pending = con.execute(
        """
        SELECT e.id, e.symbol, e.post_date AS date
        FROM events e
        JOIN event_labels l ON l.event_id = e.id
        LEFT JOIN event_footprints f ON f.event_id = e.id
        WHERE l.materiality >= 3 AND e.symbol IS NOT NULL
          AND f.event_id IS NULL
        """).pl()
    if pending.height == 0:
        return {"computed": 0}
    trail = _trailing(feats).with_columns(
        # footprint the day BEFORE the event: shift trailing stats forward
        pl.col("fp_vol_z").shift(1).over("symbol"),
        pl.col("fp_ret").shift(1).over("symbol"),
    )
    joined = pending.join(
        trail.select(["symbol", "date", "fp_vol_z", "fp_ret"]),
        on=["symbol", "date"], how="inner"
    ).filter(pl.col("fp_vol_z").is_not_null())
    rows = [{"event_id": r["id"], "pre_vol_z": r["fp_vol_z"],
             "pre_ret": r["fp_ret"]}
            for r in joined.iter_rows(named=True)]
    if rows:
        vdb.upsert(con, "event_footprints", rows)
    return {"computed": len(rows)}


def daily_watch(con, feats: pl.DataFrame, date_str: str) -> list[dict]:
    hist = con.execute(
        "SELECT count(*), quantile_cont(pre_vol_z, ?) FROM event_footprints",
        [PCTL]).fetchone()
    if not hist or (hist[0] or 0) < MIN_HISTORY:
        return []
    threshold = hist[1]
    import datetime as dt
    day = dt.date.fromisoformat(date_str)
    today = _trailing(feats).filter(pl.col("date") == day)
    flagged = (today.filter((pl.col("fp_vol_z") > threshold)
                            & (pl.col("fp_ret") > 0))
               .sort("fp_vol_z", descending=True).head(TOP_N))
    return [{"date": date_str, "symbol": r["symbol"], "kind": "footprint",
             "score": round(float(r["fp_vol_z"]), 3), "phase": None,
             "detail": json.dumps({"threshold": round(float(threshold), 3),
                                   "ret_5d": round(float(r["fp_ret"]), 4)})}
            for r in flagged.iter_rows(named=True)]
```

- [ ] **Step 4: Run tests** — 4 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/zmod/footprint.py tests/zmod/test_footprint.py
git commit -m "feat: pre-announcement footprint history and daily watch"
```

---

### Task 3: Scan runner + surfaces (CLI, journal, digest, explanation warning, workflow)

**Files:**
- Create: `vectora/zmod/scan.py`
- Modify: `vectora/__main__.py`, `vectora/vault/generator.py`, `vectora/alerts/digest.py`, `vectora/predict/engine.py`, `.github/workflows/eod-pipeline.yml`
- Test: `tests/zmod/test_scan.py`, appends to `tests/test_vault.py`, `tests/test_digest.py`, `tests/predict/test_engine.py`

- [ ] **Step 1: Write the failing tests — `tests/zmod/test_scan.py`**

```python
# tests/zmod/test_scan.py
import datetime as dt

import numpy as np
import polars as pl

from vectora import db as vdb
from vectora.zmod import scan


def _seed_market(con, n_days=90, n_syms=35, seed=13):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = dt.date(2026, 4, 1)
    px = {f"S{i:02d}": 50.0 for i in range(n_syms)}
    for day in range(n_days):
        d = d0 + dt.timedelta(days=day)
        for sym in px:
            # S00 is the pump: strong drift + volume surge in the last month
            pumping = sym == "S00" and day >= n_days - 25
            drift = 0.02 if pumping else 0.0
            px[sym] *= float(np.exp(rng.normal(drift, 0.01)))
            p = round(max(px[sym], 1.0), 2)
            vol = int(rng.integers(20000, 40000)) if pumping \
                else int(rng.integers(1000, 3000))
            rows.append(dict(symbol=sym, date=d, open=p, high=p * 1.01,
                             low=p * 0.99, close=p, ltp=p, ycp=p, trades=20,
                             value_mn=2.0, volume=vol, source="dse_eod"))
    df = pl.DataFrame(rows)  # noqa: F841
    con.execute("INSERT INTO prices_raw SELECT * FROM df")
    vdb.upsert(con, "symbols", [
        dict(symbol=s, name=None, sector="Bank", instrument_type="Equity",
             category="Z" if s == "S00" else "A", listing_status="active",
             first_seen="2020-01-01", last_seen="2026-12-31")
        for s in px])
    return (d0 + dt.timedelta(days=n_days - 1)).isoformat()


def test_zscan_flags_the_pump_and_writes_zwatch(test_db, tmp_path):
    last = _seed_market(test_db)
    result = scan.run_zscan(test_db, date_str=last,
                            features_path=tmp_path / "f.parquet")
    assert result["pump_flags"] >= 1
    top = test_db.execute(
        "SELECT symbol, score, phase FROM zwatch WHERE kind='pump' "
        "ORDER BY score DESC LIMIT 1").fetchone()
    assert top[0] == "S00"
    assert top[1] > 60
    assert top[2] in ("markup", "distribution")


def test_zscan_idempotent(test_db, tmp_path):
    last = _seed_market(test_db)
    scan.run_zscan(test_db, date_str=last, features_path=tmp_path / "f.parquet")
    scan.run_zscan(test_db, date_str=last, features_path=tmp_path / "g.parquet")
    n = test_db.execute(
        "SELECT count(*) FROM (SELECT DISTINCT date, symbol, kind FROM zwatch)"
    ).fetchone()[0]
    total = test_db.execute("SELECT count(*) FROM zwatch").fetchone()[0]
    assert n == total
```

- [ ] **Step 2: Also append the surface tests**

Append to `tests/test_vault.py`:

```python
def test_journal_zwatch_section(test_db, tmp_path):
    _seed(test_db)
    vdb.upsert(test_db, "zwatch", [dict(
        date="2026-07-16", symbol="ZPUMP", kind="pump", score=88.0,
        phase="markup", detail="{}")])
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    journal = (tmp_path / "Journal" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "Z-watch" in journal and "ZPUMP" in journal and "88" in journal
```

Append to `tests/test_digest.py`:

```python
def test_digest_zwatch_section(test_db):
    _seed(test_db)
    vdb.upsert(test_db, "zwatch", [dict(
        date="2026-07-16", symbol="FOOTSYM", kind="footprint", score=3.4,
        phase=None, detail='{"threshold": 2.0, "ret_5d": 0.04}')])
    body = digest.build(test_db, "2026-07-16")
    assert "Z-watch" in body and "FOOTSYM" in body
```

Append to `tests/predict/test_engine.py`:

```python
def test_signal_on_pump_flagged_name_gets_warning(test_db, tmp_path):
    last = _seed_market(test_db)
    _train_and_register(test_db, tmp_path)
    vdb.upsert(test_db, "zwatch", [dict(
        date=last, symbol="S1", kind="pump", score=90.0,
        phase="markup", detail="{}")])
    pengine.run_predict(test_db, date_str=last,
                        features_path=tmp_path / "f2.parquet",
                        min_median_value_mn=0.1)
    rendered = test_db.execute(
        "SELECT rendered FROM explanations WHERE prediction_id LIKE ?",
        [f"%{last}_g5_h10_S1"]).fetchone()[0]
    assert "pump-phase" in rendered
```

- [ ] **Step 3: Run to verify failures** — new tests FAIL.

- [ ] **Step 4: Implement**

Create `vectora/zmod/scan.py`:

```python
"""Daily Z-scan: pump scores + footprint watch into zwatch (spec §13).
Runs after regime, before predict, so the predict engine can attach
pump warnings to any signal on a flagged name."""
import json
from datetime import date

import polars as pl

from vectora import db as vdb
from vectora.features import engine as fengine
from vectora.zmod import footprint, pump

PUMP_MIN_SCORE = 50.0
PUMP_TOP_N = 15


def run_zscan(con, date_str: str | None = None, features_path=None) -> dict:
    feats = fengine.compute(con, out_path=features_path) if features_path \
        else fengine.compute(con)
    run_date = date_str or str(feats["date"].max())
    day = feats.filter(pl.col("date") == date.fromisoformat(run_date))
    categories = dict(con.execute(
        "SELECT symbol, category FROM symbols").fetchall())

    scored = pump.phase_and_score(day, categories)
    flagged = (scored.filter(pl.col("score") >= PUMP_MIN_SCORE)
               .sort("score", descending=True).head(PUMP_TOP_N))
    pump_rows = [{"date": run_date, "symbol": r["symbol"], "kind": "pump",
                  "score": round(float(r["score"]), 1), "phase": r["phase"],
                  "detail": json.dumps({
                      "ret_21d": round(float(r["ret_21d"] or 0), 4),
                      "vol_ratio": round(float(r["vol_ratio_5_21"] or 0), 2)})}
                 for r in flagged.iter_rows(named=True)]
    if pump_rows:
        vdb.upsert(con, "zwatch", pump_rows)

    fp_result = footprint.compute_event_footprints(
        con, feats.select(["symbol", "date", "ret", "volume_z_21d"]))
    fp_rows = footprint.daily_watch(
        con, feats.select(["symbol", "date", "ret", "volume_z_21d"]), run_date)
    if fp_rows:
        vdb.upsert(con, "zwatch", fp_rows)

    return {"date": run_date, "pump_flags": len(pump_rows),
            "footprints_computed": fp_result["computed"],
            "footprint_flags": len(fp_rows)}
```

In `vectora/__main__.py`: stage choices gain `"zscan"`; add branch:

```python
    if args.command == "run" and args.stage == "zscan":
        from vectora import db as vdb
        from vectora.settings import DB_PATH
        from vectora.zmod import scan
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = scan.run_zscan(con, date_str=args.date)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0
```

In `vectora/vault/generator.py` `generate()`, after the events query add:

```python
    zwatch = con.execute(
        "SELECT symbol, kind, score, phase FROM zwatch WHERE date = ? "
        "ORDER BY kind, score DESC", [date_str]).fetchall()
```

and after the Company-events journal block add:

```python
    if zwatch:
        lines.append("")
        lines.append("## Z-watch")
        for sym, kind, score, phase in zwatch:
            tag = f"pump {score:.0f} ({phase})" if kind == "pump" \
                else f"pre-announcement footprint (vol_z {score})"
            lines.append(f"- [[{sym}]] {tag}")
```

In `vectora/alerts/digest.py` `build()`, before the Suppressions block add:

```python
    zwatch = con.execute(
        "SELECT symbol, kind, score, phase FROM zwatch WHERE date = ? "
        "ORDER BY kind, score DESC", [date_str]).fetchall()
    if zwatch:
        lines.append("")
        lines.append("## Z-watch (warnings, not signals)")
        for sym, kind, score, phase in zwatch:
            tag = f"pump {score:.0f} ({phase})" if kind == "pump" \
                else f"pre-announcement footprint (vol_z {score})"
            lines.append(f"- {sym}: {tag}")
```

In `vectora/predict/engine.py` `run_predict`, after the `categories` lookup add:

```python
    pump_flagged = {r[0] for r in con.execute(
        "SELECT symbol FROM zwatch WHERE date = ? AND kind = 'pump'",
        [run_date]).fetchall()}
```

and after building each explanation dict (right after `expls.append({...})`) add:

```python
            if symbol in pump_flagged:
                expls[-1]["rendered"] += (
                    "\nWarning: elevated pump-phase score today - treat "
                    "momentum in this name as suspect.")
```

In `.github/workflows/eod-pipeline.yml`, insert between "Regime" and "Predict":

```yaml
      - name: Zscan
        continue-on-error: true
        run: uv run python -m vectora run zscan
```

- [ ] **Step 5: Run tests** — all new tests pass; fast suite; ruff. Commit:

```bash
git add vectora/zmod vectora/__main__.py vectora/vault/generator.py vectora/alerts/digest.py vectora/predict/engine.py .github/workflows/eod-pipeline.yml tests/zmod tests/test_vault.py tests/test_digest.py tests/predict/test_engine.py
git commit -m "feat: zscan stage - pump flags and footprint watch surfaced everywhere"
```

---

### Task 4: Real run + merge + verify

- [ ] **Step 1: Real zscan**

Run: `uv run python -m vectora run zscan`
Expected: pump_flags 0–15, footprints_computed in the low thousands (first run computes all historical materiality-3 events that have feature coverage), footprint_flags 0–10. Then inspect:

`uv run python -c "
from vectora import db as vdb
con = vdb.connect('data/vectora.duckdb'); vdb.init_schema(con)
print('footprint history:', con.execute('SELECT count(*), round(avg(pre_vol_z),3), round(quantile_cont(pre_vol_z, 0.75),3) FROM event_footprints').fetchone())
print('todays zwatch:', con.execute('SELECT kind, symbol, score, phase FROM zwatch ORDER BY kind, score DESC LIMIT 15').fetchall())
con.execute('CHECKPOINT'); con.close()"`

Judgment gates: pre_vol_z 75th percentile should be modestly positive (0.3–1.5 — pre-announcement volume IS elevated on average, which is the whole thesis; ~0 means the footprint carries no signal and the watch will rarely fire — report either way, both are honest findings). Pump flags should look like actual recent runners — spot-check one against its journal/company note.

- [ ] **Step 2: Regenerate today's vault + fast suite + commit data**

```bash
uv run python -m vectora run vault
uv run pytest -m "not slow" && uv run ruff check .
git add -f data/vectora.duckdb vault
git commit -m "data: first zscan - pump flags, footprint history, Z-watch in vault"
```

- [ ] **Step 3: Merge, push, dispatch verification**

```bash
git checkout main && git pull
git merge --no-ff phase-4c-zmod -m "Merge phase-4c: Z-specialist - pump phases and pre-announcement footprints"
git push
& "C:\Program Files\GitHub CLI\gh.exe" workflow run eod-pipeline --ref main -f date=2026-07-17
```

Watch to green including the new Zscan step.

---

## Execution notes

- Order 1→4. Expected fast suite ≈ 182 tests.
- The footprint threshold (75th pctile) and pump cutoffs are explicit constants for Phase 5 tuning — don't bikeshed.
- Spec §13's "separate Z-only model heads" intentionally wait for Phase 5's retraining session (they need the learning loop's evaluation surface to prove they beat the pooled model).
- After 4C, the remaining slices: 4D intraday (4×/day chart publications), Phase 5 learning loop, production polish.
