# Vectora Phase 5B: Production Polish Implementation Plan (final slice)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the last three production requirements — a daily health watchdog, adjusted-close backfill returns replacing the ±12% clip approximation, and the README/runbook/feature docs — then sign off the roadmap's production scope.

**Architecture:** `vectora/health.py` runs five checks (price freshness vs the trading calendar, latest quality score, core-table integrity, collect watermark staleness, and live layout canaries on the two scraped pages) with a CLI stage that emails on failure and exits non-zero so the daily `health.yml` goes red loudly. The Mendeley zip's *adjusted* per-company series (on disk, never loaded until now) becomes a slim `(symbol, date, adj_close)` parquet joined by `base.load_panel`, whose return ladder becomes: YCP-based (scraped era, corporate-action safe) → adjusted-close chain (backfill era, split/rights-correct) → unadjusted chain (fallback), all still clipped as a data-error guard. Docs are generated where possible (features.md from the registry) and written once where not (README, runbook).

**Tech Stack:** existing only.

**Existing contracts:** `calendar.previous_trading_day/is_trading_day/load_holidays`; `settings.MIN_QUALITY_SCORE/REFERENCE_DIR/DSE_BASE`; `digest.send_or_save`; `PoliteSession`; `base.load_panel` current ladder (ycp else prev-close, clipped `RET_CLIP=0.12`); backfill zip at `C:\Users\hp\Downloads\Dhaka Stock Exchange End-of-Day Financial Dataset.zip` with folder `Company Separated Adjusted Data/` (487 CSVs, `Date,Open,High,Low,Close,Volume`, 00-prefixed = index series to skip); `_select_members`-style filtering in tools/backfill_mendeley.py; registry `load()` specs with `.name/.family/.reasoning`. Fast tests: `uv run pytest -m "not slow"` (currently 199). Branch `phase-5b-polish` off main. Bulk-seed rule applies.

**File structure:**

```
vectora/health.py                  # checks + result dict
vectora/features/base.py           # adjusted-chain return ladder
vectora/settings.py                # + ADJUSTED_PARQUET
vectora/__main__.py                # + health stage
tools/build_adjusted_parquet.py
tools/gen_feature_docs.py
.github/workflows/health.yml
README.md, docs/runbook.md, docs/features.md (generated)
tests/test_health.py, adjusted-return tests in tests/features/test_base.py
```

---

### Task 1: Health watchdog

**Files:**
- Create: `vectora/health.py`, `.github/workflows/health.yml`
- Modify: `vectora/__main__.py`
- Test: `tests/test_health.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_health.py
import datetime as dt

from vectora import db as vdb
from vectora import health


def _seed_fresh(con, d="2026-07-16"):
    vdb.upsert(con, "prices_raw", [dict(
        symbol="GP", date=d, open=1, high=1, low=1, close=1.0, ltp=1, ycp=1,
        trades=1, value_mn=1.0, volume=1, source="dse_eod")])
    vdb.upsert(con, "data_quality", [dict(
        date=d, source="dse_eod", score=100, issues="[]")])
    vdb.upsert(con, "regimes", [dict(
        date=d, regime="Sideways", confidence=0.5, method="rules")])
    vdb.upsert(con, "predictions", [dict(
        id=f"{d}_g5_h10_GP", symbol="GP", date=d, target="g5_h10",
        probability=0.4, model_id="m", quality_score=100, is_signal=False,
        suppressed_reason="below-probability-threshold")])
    vdb.set_watermark(con, "collect", "eod", d)


def test_all_green_when_fresh(test_db):
    _seed_fresh(test_db)
    # 2026-07-17 was Friday (weekend): last expected trading day = Thu 07-16
    result = health.check(test_db, today=dt.date(2026, 7, 18),
                          holidays=set())
    assert result["ok"] is True
    names = {c["name"] for c in result["checks"]}
    assert {"freshness", "quality", "tables", "watermark"} <= names


def test_stale_prices_fail_freshness(test_db):
    _seed_fresh(test_db, d="2026-07-09")   # a week old
    result = health.check(test_db, today=dt.date(2026, 7, 18),
                          holidays=set())
    assert result["ok"] is False
    fresh = next(c for c in result["checks"] if c["name"] == "freshness")
    assert fresh["ok"] is False and "2026-07-16" in fresh["detail"]


def test_low_quality_fails(test_db):
    _seed_fresh(test_db)
    vdb.upsert(test_db, "data_quality", [dict(
        date="2026-07-16", source="dse_eod", score=40, issues="[]")])
    result = health.check(test_db, today=dt.date(2026, 7, 18),
                          holidays=set())
    assert next(c for c in result["checks"]
                if c["name"] == "quality")["ok"] is False


def test_canaries_with_fake_session(test_db):
    _seed_fresh(test_db)

    class GoodSession:
        def get(self, url, params=None):
            return "<table class='shares-table'></table><div class='midrow'>"

    class BrokenSession:
        def get(self, url, params=None):
            return "<html>redesigned!</html>"

    ok = health.check(test_db, today=dt.date(2026, 7, 18), holidays=set(),
                      session=GoodSession())
    assert next(c for c in ok["checks"]
                if c["name"] == "canary")["ok"] is True
    bad = health.check(test_db, today=dt.date(2026, 7, 18), holidays=set(),
                       session=BrokenSession())
    assert bad["ok"] is False
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement `vectora/health.py`**

```python
"""Daily health watchdog (spec §20 monitoring).

Five checks; any failure turns the health workflow red (GitHub emails the
owner natively) and sends a [HEALTH] email when the secret is present.
The canary check needs a live session and is skipped when session=None
(unit tests, offline runs).
"""
import datetime as dt

from vectora import calendar as cal
from vectora.settings import DSE_BASE, MIN_QUALITY_SCORE


def check(con, today: dt.date | None = None,
          holidays: set | None = None, session=None) -> dict:
    today = today or dt.date.today()
    hs = cal.load_holidays() if holidays is None else holidays
    expected = today if cal.is_trading_day(today, hs) \
        else cal.previous_trading_day(today, hs)
    checks = []

    max_date = con.execute(
        "SELECT max(date) FROM prices_raw WHERE source = 'dse_eod'"
    ).fetchone()[0]
    checks.append({
        "name": "freshness",
        "ok": max_date is not None and str(max_date) >= str(expected),
        "detail": f"latest eod {max_date}, expected {expected}"})

    q = con.execute(
        "SELECT score FROM data_quality WHERE source='dse_eod' "
        "ORDER BY date DESC LIMIT 1").fetchone()
    checks.append({
        "name": "quality",
        "ok": q is not None and q[0] >= MIN_QUALITY_SCORE,
        "detail": f"latest quality {q[0] if q else 'none'}"})

    for t in ("prices_raw", "predictions", "regimes"):
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        if n == 0:
            checks.append({"name": "tables", "ok": False,
                           "detail": f"{t} is empty"})
            break
    else:
        checks.append({"name": "tables", "ok": True, "detail": "core ok"})

    from vectora import db as vdb
    wm = vdb.get_watermark(con, "collect", "eod")
    checks.append({
        "name": "watermark",
        "ok": wm is not None and wm >= str(expected),
        "detail": f"collect/eod at {wm}, expected {expected}"})

    if session is not None:
        try:
            archive = session.get(
                f"{DSE_BASE}/day_end_archive.php",
                params={"startDate": str(expected), "endDate": str(expected),
                        "inst": "All Instrument", "archive": "data"})
            home = session.get(f"{DSE_BASE}/")
            ok = "shares-table" in archive and "midrow" in home
            detail = "markers present" if ok else "LAYOUT CHANGED"
        except Exception as exc:  # noqa: BLE001 - any fetch failure is the finding
            ok, detail = False, f"fetch failed: {exc}"
        checks.append({"name": "canary", "ok": ok, "detail": detail})

    return {"ok": all(c["ok"] for c in checks), "checks": checks}
```

Add the `health` CLI stage in `vectora/__main__.py` (choices gain `"health"`):

```python
    if args.command == "run" and args.stage == "health":
        from vectora import db as vdb
        from vectora import health
        from vectora.alerts.digest import send_or_save
        from vectora.http import PoliteSession
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = health.check(con, session=PoliteSession())
        finally:
            con.close()
        print(json.dumps(result, indent=1, default=str))
        if not result["ok"]:
            failing = [c for c in result["checks"] if not c["ok"]]
            body = "\n".join(f"- {c['name']}: {c['detail']}" for c in failing)
            send_or_save(f"[HEALTH] Vectora: {len(failing)} check(s) failing",
                         body)
            return 1
        return 0
```

Create `.github/workflows/health.yml`:

```yaml
name: health

on:
  schedule:
    # 17:30 Dhaka daily = 11:30 UTC, after the eod pipeline settles
    - cron: "30 11 * * *"
  workflow_dispatch:

permissions:
  contents: read

env:
  TZ: Asia/Dhaka

jobs:
  health:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - name: Install
        run: uv sync --frozen
      - name: Health check
        env:
          GMAIL_APP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}
        run: uv run python -m vectora run health
```

- [ ] **Step 4: Run tests** — 4 passed; fast suite; ruff. Local live check: `uv run python -m vectora run health` — on a weekend with fresh Thursday data expect ok=true, canary ok (2 polite requests). Commit:

```bash
git add vectora/health.py vectora/__main__.py .github/workflows/health.yml tests/test_health.py
git commit -m "feat: daily health watchdog with freshness, quality, and layout canaries"
```

---

### Task 2: Adjusted-close backfill returns

**Files:**
- Create: `tools/build_adjusted_parquet.py`
- Modify: `vectora/settings.py`, `vectora/features/base.py`
- Test: append to `tests/features/test_base.py`

- [ ] **Step 1: Settings** — append to `vectora/settings.py`:

```python
ADJUSTED_PARQUET = REFERENCE_DIR / "backfill_adjusted_2012_2026.parquet"
```

- [ ] **Step 2: Write the failing test (append to tests/features/test_base.py)**

```python
def test_backfill_return_prefers_adjusted_chain(test_db, tmp_path, monkeypatch):
    import polars as pl as _pl  # noqa: F401 - clarity below uses pl already
    # unadjusted closes show a fake 2:1 split gap; adjusted chain is smooth
    vdb.upsert(test_db, "prices_raw", [
        dict(symbol="SPL", date="2026-07-05", open=100, high=100, low=100,
             close=100.0, ltp=None, ycp=None, trades=None, value_mn=None,
             volume=1, source="mendeley"),
        dict(symbol="SPL", date="2026-07-06", open=51, high=51, low=51,
             close=51.0, ltp=None, ycp=None, trades=None, value_mn=None,
             volume=1, source="mendeley"),
    ])
    adj = tmp_path / "adj.parquet"
    pl.DataFrame({
        "symbol": ["SPL", "SPL"],
        "date": ["2026-07-05", "2026-07-06"],
        "adj_close": [50.0, 51.0],
    }).with_columns(pl.col("date").cast(pl.Date)).write_parquet(adj)
    monkeypatch.setattr(base, "ADJUSTED_PARQUET", adj)
    df = base.load_panel(test_db).filter(pl.col("symbol") == "SPL").sort("date")
    # adjusted chain: 51/50 - 1 = +2%, NOT the clipped -12% the raw gap gives
    assert abs(df["ret"][1] - 0.02) < 1e-9


def test_backfill_without_adjusted_falls_back(test_db, tmp_path, monkeypatch):
    monkeypatch.setattr(base, "ADJUSTED_PARQUET", tmp_path / "missing.parquet")
    _seed(test_db)   # existing helper: ACI rows incl. the 50% gap day
    df = base.load_panel(test_db).filter(pl.col("symbol") == "ACI").sort("date")
    assert abs(df["ret"][2] - base.RET_CLIP) < 1e-9   # old clipped behavior
```

Fix the stray first line of the first test during implementation — `import polars as pl as _pl` is invalid syntax left as a plan artifact; DELETE that line (the file already imports polars as pl at module top).

- [ ] **Step 3: Run to verify failure** — FAIL.

- [ ] **Step 4: Implement `vectora/features/base.py`** — replace the module with:

```python
"""Price panel loader with the canonical daily-return column.

Return ladder (corporate-action correctness, best available per row):
1. scraped rows (ycp present & > 0): ret = close/ycp - 1 — DSE's YCP is
   ex-date adjusted, so these are corporate-action safe.
2. backfill rows with adjusted-close coverage: adj_close chain — the
   Mendeley adjusted series makes splits/rights invisible, replacing the
   old clip approximation for 2012-2026.
3. remainder: unadjusted close chain.
All returns clip to +/-RET_CLIP as a residual data-error guard (real DSE
moves cannot exceed the circuit band).
"""
from pathlib import Path

import polars as pl

from vectora.settings import ADJUSTED_PARQUET

RET_CLIP = 0.12

_PANEL_SQL = """
    SELECT symbol, date, open, high, low, close, ycp, trades, value_mn, volume
    FROM prices
    WHERE close IS NOT NULL AND close > 0
    ORDER BY symbol, date
"""


def load_panel(con) -> pl.DataFrame:
    df = con.execute(_PANEL_SQL).pl()
    adj_path = Path(ADJUSTED_PARQUET)
    if adj_path.exists():
        adj = pl.read_parquet(adj_path)
        df = df.join(adj, on=["symbol", "date"], how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("adj_close"))
    prev_close = pl.col("close").shift(1).over("symbol")
    prev_adj = pl.col("adj_close").shift(1).over("symbol")
    raw_ret = (
        pl.when(pl.col("ycp").is_not_null() & (pl.col("ycp") > 0))
        .then(pl.col("close") / pl.col("ycp") - 1)
        .when(pl.col("adj_close").is_not_null() & prev_adj.is_not_null()
              & (prev_adj > 0))
        .then(pl.col("adj_close") / prev_adj - 1)
        .otherwise(pl.col("close") / prev_close - 1)
    )
    return df.with_columns(
        raw_ret.clip(-RET_CLIP, RET_CLIP).alias("ret")
    ).drop("adj_close")
```

Note: `ADJUSTED_PARQUET` is imported at module level so tests can monkeypatch `base.ADJUSTED_PARQUET`.

- [ ] **Step 5: Create `tools/build_adjusted_parquet.py`**

```python
"""One-time: extract Mendeley's ADJUSTED per-company series into a slim
(symbol, date, adj_close) parquet consumed by features/base.py's return
ladder. Usage: uv run python tools/build_adjusted_parquet.py path/to/zip
"""
import sys
import tempfile
import shutil
import zipfile
from pathlib import Path

import duckdb

from vectora.settings import ADJUSTED_PARQUET


def build(zip_path: Path, out: Path = ADJUSTED_PARQUET) -> int:
    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(zip_path) as z:
        n = 0
        for m in z.namelist():
            low = m.lower()
            if not low.endswith(".csv") or "unadjust" in low \
                    or "adjust" not in low:
                continue
            stem = Path(m).stem.strip().upper()
            if not stem or stem.startswith("00") or "adjust" in stem.lower():
                continue
            with z.open(m) as src, open(Path(td) / f"{stem}.csv", "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
        glob = str(Path(td) / "*.csv").replace("\\", "/")
        con = duckdb.connect()
        outp = str(out).replace("\\", "/")
        con.execute(f"""
            COPY (
                SELECT upper(regexp_extract(filename, '([^/\\\\]+)\\.csv$', 1))
                           AS symbol,
                       try_cast("Date" AS DATE) AS date,
                       try_cast("Close" AS DOUBLE) AS adj_close
                FROM read_csv('{glob}', all_varchar=true, header=true,
                              filename=true)
                WHERE try_cast("Date" AS DATE) IS NOT NULL
                  AND try_cast("Close" AS DOUBLE) IS NOT NULL
                ORDER BY symbol, date
            ) TO '{outp}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{outp}')").fetchone()[0]
        print(f"{n} files -> {rows} rows -> {out}")
        return 0


if __name__ == "__main__":
    sys.exit(build(Path(sys.argv[1])))
```

- [ ] **Step 6: Run tests, then the real build + feature smoke**

```bash
uv run pytest tests/features -q && uv run ruff check .
uv run python tools/build_adjusted_parquet.py "C:\Users\hp\Downloads\Dhaka Stock Exchange End-of-Day Financial Dataset.zip"
uv run python -c "
from vectora import db; from vectora.features import engine
con = db.connect('data/vectora.duckdb'); db.init_schema(con)
df = engine.compute(con); print('rows', df.height, 'ret nulls', df['ret'].null_count()); con.close()"
```

Expect ~1.06M adjusted rows, parquet ~5-8MB; feature recompute still seconds; ret nulls roughly unchanged. The leakage test still passes (adjusted history is static past data).

- [ ] **Step 7: Commit**

```bash
git add vectora/settings.py vectora/features/base.py tools/build_adjusted_parquet.py tests/features/test_base.py data/reference/backfill_adjusted_2012_2026.parquet
git commit -m "feat: adjusted-close return chain for the backfill era (replaces clip approximation)"
```

Note for the commit message body: models retrain on the improved returns automatically next Friday; no manual retrain needed.

---

### Task 3: README, runbook, feature docs

**Files:**
- Create: `README.md`, `docs/runbook.md`, `tools/gen_feature_docs.py`, `docs/features.md` (generated)

- [ ] **Step 1: `tools/gen_feature_docs.py`**

```python
"""Generate docs/features.md from the feature registry (spec §22)."""
from pathlib import Path

from vectora.features import registry


def main() -> int:
    specs = registry.load()
    lines = ["# Feature registry", "",
             f"{len(specs)} features. Every feature documents its economic "
             "reasoning (enforced by test).", "",
             "| name | family | reasoning |", "|---|---|---|"]
    lines += [f"| `{s.name}` | {s.family} | {s.reasoning} |" for s in specs]
    Path("docs/features.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print(f"docs/features.md: {len(specs)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run it: `uv run python tools/gen_feature_docs.py`.

- [ ] **Step 2: `README.md`** — write exactly:

```markdown
# Vectora

Zero-cost market intelligence system for the Dhaka Stock Exchange (DSE).
Runs entirely on GitHub Actions free tier + scraped public data. No paid
APIs, no servers, no LLM in the loop — deterministic ML and statistics.

**This is a research tool, not investment advice.** Predictions are
calibrated probabilities with documented uncertainty, risk blocks, and
honest evaluation — never buy/sell recommendations.

## What it does, every trading day (all automatic)

1. **15:30 Dhaka** — scrape EOD prices/news/indices, validate (0-100
   quality score), classify announcements, update the market regime,
   run the Z-category pump/footprint scan, score ~330 liquid equities
   with calibrated LightGBM models, grade matured predictions, update
   the Obsidian vault (`vault/`), email the digest.
2. **11:00-14:00 Dhaka, hourly** — intraday snapshots at DSE's chart
   publication points; volume-surge / near-circuit urgent alerts
   (cooldowns + daily cap).
3. **Friday** — walk-forward retrain of both targets; a challenger is
   promoted only if it beats the incumbent's Brier score; evaluation
   report with per-regime calibration and miss autopsy.
4. **17:30 Dhaka** — health watchdog (freshness, quality, layout
   canaries).

## Quickstart

```bash
uv sync
uv run pytest -m "not slow"          # ~200 tests
uv run python -m vectora run eod     # collect + validate (gap-fills)
uv run python -m vectora run predict # probabilities + risk + explanations
```

Stages: `eod train predict digest outcomes vault regime events zscan
intraday evaluate health`. All idempotent; all state lives in
`data/vectora.duckdb` (+ static parquets in `data/reference/`) committed
to this repo.

## Architecture (short version)

Scrapers (polite, fixture-tested, layout-canaried) → DuckDB + immutable
gzipped raw layer → registry-driven feature engine (43 documented
features, leakage-guarded by test) → walk-forward-trained calibrated
models → admission-gated signals with SHAP explanations and analog-based
risk blocks → Obsidian vault + email. Full design: `docs/superpowers/`.

Open `vault/` in Obsidian for the human-facing knowledge base. Operational
issues: see `docs/runbook.md`. Features: `docs/features.md`.
```

- [ ] **Step 3: `docs/runbook.md`** — write exactly:

```markdown
# Vectora runbook

## A workflow went red

| Workflow | Step | Likely cause | Action |
|---|---|---|---|
| eod-pipeline | Run EOD pipeline | unlisted holiday (quality 0, "no rows") | add date to `data/reference/holidays.csv`, push; next run gap-fills |
| eod-pipeline | Run EOD pipeline | dsebd.org layout change (canary also red) | re-record fixtures (`uv run python tools/record_fixtures.py`), fix parser against them, tests are the contract |
| eod-pipeline | Predict | model artifact/path issue | check `model_registry.artifact_dir` is repo-relative; `models/` committed |
| train | Train | challenger lost (`"promoted": false`) | not an error — the guard working; nothing to do |
| health | Health check | see emailed [HEALTH] list | freshness → check eod-pipeline run; canary → layout change path above |
| intraday-scan | Intraday scan | outside trading hours/day | harmless skip, will show green |

## Routine operations

- **Add a holiday:** append `YYYY-MM-DD,description` to
  `data/reference/holidays.csv`, commit, push.
- **Rotate the Gmail app password:** revoke old at
  myaccount.google.com/apppasswords, then
  `gh secret set GMAIL_APP_PASSWORD --repo AhmedFahim13/vectora`.
  Missing/dead secret degrades safely: digests land in `reports/`.
- **Roll back a model:** `UPDATE model_registry SET active=false WHERE
  model_id='bad'; UPDATE model_registry SET active=true WHERE
  model_id='good';` via a `uv run python -c` one-liner, commit the DB.
- **DB merge conflict:** never hand-merge. `.gitattributes` sets
  `merge=ours` (local wins); run `uv run python -m vectora run eod`
  afterward — gap-fill re-ingests whatever the other side had.
  Requires once per clone: `git config merge.ours.driver true`.
- **Enable g10_h30 signals:** only when live evaluation shows the
  deployment tail holds on a HOLDOUT (see memory note about the
  tautological in-sample table); then add `"g10_h30": 0.60` to
  `SIGNAL_THRESHOLDS`.

## Known quirks (do not "fix")

- dsebd.org serves a broken TLS chain → `verify=False` in PoliteSession
  (documented, public data).
- `models/**/*.txt -text` in `.gitattributes` is load-bearing: CRLF
  corrupts LightGBM dumps on Windows.
- News archive retention starts 2024-07; older `old_news` queries return
  empty pages (real, not a bug).
- Liquidity features are null through the backfill era (no traded-value
  data before 2026-07); they activate as live history accumulates.
- Manual `run eod` during market hours (10:00-14:30 Dhaka) can mark a
  live day as no-trade; the CI schedule avoids this by design.
```

- [ ] **Step 4: Commit**

```bash
uv run ruff check . && uv run pytest -m "not slow"
git add README.md docs/runbook.md docs/features.md tools/gen_feature_docs.py
git commit -m "docs: README, operational runbook, generated feature docs"
```

---

### Task 4: Merge + verify + production sign-off

- [ ] **Step 1:** `git checkout main && git pull && git merge --no-ff phase-5b-polish -m "Merge phase-5b: health watchdog, adjusted returns, docs - production complete" && git push`
- [ ] **Step 2:** Dispatch and watch `health` workflow to green: `gh workflow run health --ref main`.
- [ ] **Step 3:** Update project memory: production scope delivered; system self-operates (daily pipeline, 4x intraday, Friday retrain, daily health); future work = Phase 6 backlog (Trends/macro collectors, dashboard, HMM, holdout deployment-calibration, LLM plug-in) driven by accumulated evaluation data.

---

## Execution notes

- Expected fast suite ≈ 205 tests. Task 2's parquet build reads the zip from Downloads — if the user deleted it, skip Task 2 gracefully (the ladder falls back) and note it.
- This is the LAST production slice; after merge the roadmap's production scope is complete.
