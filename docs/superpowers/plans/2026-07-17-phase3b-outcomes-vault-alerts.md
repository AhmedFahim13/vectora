# Vectora Phase 3B: Outcomes, Vault, Signal Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the daily loop: grade every matured prediction against reality (spec §17), mirror the day's intelligence into a human-note-safe Obsidian vault (spec §7), and make the digest signal-aware with a cooldown-logged alert trail (spec §16).

**Architecture:** Three small packages wired into the existing scheduled pipeline. `outcomes.resolver` reuses the label machinery (`make_labels(continuous=True)`) to compute realized forward max/min for predictions whose horizon has matured, writing an `outcomes` row per prediction. `vault.generator` renders Journal/Prediction/Company/Home notes from the DB via string templates, writing only between `<!-- vectora:begin/end -->` markers so human notes survive regeneration byte-identical. `alerts.log_signal_alerts` records signal alerts with a 48h per-symbol cooldown into `alerts_log`; the digest subject surfaces new-signal counts. All three get CLI stages and pipeline steps; the Commit step also commits `vault/`.

**Tech Stack:** existing only (polars, DuckDB). No new dependencies.

**Deliberate scope cuts (documented):** urgency *tiers* and intraday urgent emails wait for Phase 4's intraday scans — with one batch of predictions per day post-close, the signal-aware digest IS the urgent channel; `days_to_hit` in outcomes waits for Phase 5's evaluation loop. Z-category alert capping is inherited (Z never signals in 3A).

**Existing contracts (all on `main`, all tested):**
- `vectora/db.py`: `connect`, `init_schema(con, backfill_parquet=None)`, `upsert`; tables `predictions(id, symbol, date, target, probability, model_id, quality_score, is_signal, suppressed_reason)`, `explanations(prediction_id, drivers, analogs, rendered)`, `risk_blocks(...)`, `data_quality`, `symbols(symbol, sector, category, ...)`, `events(post_date, symbol, title, ...)`.
- `vectora/features/base.py`: `load_panel(con) -> pl.DataFrame` (symbol/date/close/... sorted).
- `vectora/labels.py`: `make_labels(panel, thresholds, horizons, downside=False, continuous=False)` — continuous adds `fwdmax_hH`/`fwdmin_hH`, null while horizon immature.
- `vectora/alerts/digest.py`: `build(con, date_str) -> str`, `send_or_save(subject, body, reports_dir=REPORTS_DIR)`.
- `vectora/__main__.py`: stages eod/train/predict/digest; digest branch computes `n_signals = body.count("### ")` and builds the subject.
- `vectora/settings.py`: `REPO_ROOT, DB_PATH, REPORTS_DIR, ...`.
- `tests/conftest.py`: `test_db` fixture. Targets look like `g5_h10` = gain ≥5% within 10 trading rows.
- Branch: `git checkout main && git pull && git checkout -b phase-3b-loop`. Run from repo root with `uv run …`; fast tests `uv run pytest -m "not slow"`. Commit per task.

**File structure:**

```
vectora/outcomes/__init__.py, resolver.py
vectora/vault/__init__.py, generator.py
vectora/alerts/signals.py            # alerts_log writer with cooldown
vectora/settings.py                  # + VAULT_DIR
vectora/db.py                        # + outcomes, alerts_log tables
vectora/__main__.py                  # + outcomes, vault stages; digest wires signals log
.github/workflows/eod-pipeline.yml   # + Outcomes, Vault steps; commit vault/
tests/test_outcomes.py, tests/test_vault.py, tests/test_signal_alerts.py
vault/                               # generated notes (committed)
```

---

### Task 1: Outcomes resolver

**Files:**
- Modify: `vectora/db.py` (SCHEMA)
- Create: `vectora/outcomes/__init__.py` (empty), `vectora/outcomes/resolver.py`
- Modify: `vectora/__main__.py` (add `outcomes` stage)
- Test: `tests/test_outcomes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_outcomes.py
import datetime as dt

from vectora import db as vdb
from vectora.outcomes import resolver


def _price(symbol, d, close):
    return dict(symbol=symbol, date=d, open=close, high=close, low=close,
                close=close, ltp=close, ycp=close, trades=10, value_mn=1.0,
                volume=100, source="dse_eod")


def _pred(symbol, d, target="g5_h3", prob=0.6):
    return dict(id=f"{d}_{target}_{symbol}", symbol=symbol, date=d,
                target=target, probability=prob, model_id="m",
                quality_score=100, is_signal=True, suppressed_reason=None)


def _seed(con, closes, symbol="GP", start="2026-07-01"):
    d0 = dt.date.fromisoformat(start)
    rows = [_price(symbol, (d0 + dt.timedelta(days=i)).isoformat(), c)
            for i, c in enumerate(closes)]
    vdb.upsert(con, "prices_raw", rows)


def test_matured_prediction_resolves_hit(test_db):
    # close 100 on day0; next 3 closes: 103, 106, 104 -> max +6% >= 5% -> hit
    _seed(test_db, [100, 103, 106, 104])
    vdb.upsert(test_db, "predictions", [_pred("GP", "2026-07-01")])
    result = resolver.resolve(test_db)
    assert result == {"resolved": 1, "pending": 0}
    row = test_db.execute(
        "SELECT hit, realized_max, realized_min FROM outcomes").fetchone()
    assert row[0] is True
    assert abs(row[1] - 0.06) < 1e-9
    assert abs(row[2] - 0.03) < 1e-9


def test_matured_prediction_resolves_miss(test_db):
    _seed(test_db, [100, 101, 102, 101])   # max +2% < 5% -> miss
    vdb.upsert(test_db, "predictions", [_pred("GP", "2026-07-01")])
    resolver.resolve(test_db)
    assert test_db.execute("SELECT hit FROM outcomes").fetchone()[0] is False


def test_immature_prediction_stays_pending(test_db):
    _seed(test_db, [100, 103])             # only 1 forward row, horizon 3
    vdb.upsert(test_db, "predictions", [_pred("GP", "2026-07-01")])
    result = resolver.resolve(test_db)
    assert result == {"resolved": 0, "pending": 1}
    assert test_db.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 0


def test_resolve_is_idempotent_and_incremental(test_db):
    _seed(test_db, [100, 103, 106, 104])
    vdb.upsert(test_db, "predictions", [_pred("GP", "2026-07-01")])
    resolver.resolve(test_db)
    result = resolver.resolve(test_db)     # nothing new to do
    assert result == {"resolved": 0, "pending": 0}
    assert test_db.execute("SELECT count(*) FROM outcomes").fetchone()[0] == 1


def test_multiple_targets_resolve_independently(test_db):
    _seed(test_db, [100, 103, 106, 104, 108, 112, 111, 115, 113, 116, 118])
    vdb.upsert(test_db, "predictions", [
        _pred("GP", "2026-07-01", target="g5_h3"),
        _pred("GP", "2026-07-01", target="g10_h10", prob=0.4),
    ])
    result = resolver.resolve(test_db)
    assert result["resolved"] == 2
    rows = dict(test_db.execute(
        "SELECT prediction_id, hit FROM outcomes").fetchall())
    assert rows["2026-07-01_g5_h3_GP"] is True     # +6% within 3
    assert rows["2026-07-01_g10_h10_GP"] is True   # +18% within 10
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_outcomes.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement**

Append to `SCHEMA` in `vectora/db.py`:

```sql
CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id TEXT PRIMARY KEY,
    resolved_at TIMESTAMP DEFAULT current_timestamp,
    realized_max DOUBLE, realized_min DOUBLE, hit BOOLEAN
);
```

Create `vectora/outcomes/resolver.py`:

```python
"""Grade matured predictions against realized prices (spec §17 step 1).

A prediction for target gX_hH matures once H trading rows exist after its
date; realized max/min forward returns come from the same label machinery
the models train on, so grading and training share one definition of
"outcome". Unresolved predictions stay pending and are retried next run.
"""
import re

import polars as pl

from vectora import db as vdb
from vectora import labels as lab
from vectora.features import base

_TARGET_RE = re.compile(r"^g(\d+)_h(\d+)$")


def resolve(con) -> dict:
    preds = con.execute(
        """
        SELECT p.id, p.symbol, p.date, p.target
        FROM predictions p
        LEFT JOIN outcomes o ON o.prediction_id = p.id
        WHERE o.prediction_id IS NULL
        """).pl()
    if preds.height == 0:
        return {"resolved": 0, "pending": 0}

    panel = base.load_panel(con).select(["symbol", "date", "close"])
    rows = []
    for target in preds["target"].unique().to_list():
        m = _TARGET_RE.match(target)
        if not m:
            continue  # unknown target format: leave pending forever
        x, h = int(m.group(1)) / 100, int(m.group(2))
        labeled = lab.make_labels(
            panel, thresholds=(x,), horizons=(h,), continuous=True)
        joined = (
            preds.filter(pl.col("target") == target)
            .join(labeled.select(["symbol", "date", f"fwdmax_h{h}",
                                  f"fwdmin_h{h}"]),
                  on=["symbol", "date"], how="left")
            .filter(pl.col(f"fwdmax_h{h}").is_not_null())
        )
        for r in joined.iter_rows(named=True):
            rows.append({
                "prediction_id": r["id"],
                "realized_max": r[f"fwdmax_h{h}"],
                "realized_min": r[f"fwdmin_h{h}"],
                "hit": bool(r[f"fwdmax_h{h}"] >= x),
            })
    if rows:
        vdb.upsert(con, "outcomes", rows)
    return {"resolved": len(rows), "pending": preds.height - len(rows)}
```

Add the CLI stage in `vectora/__main__.py`: stage choices become `["eod", "train", "predict", "digest", "outcomes"]`, and after the digest branch add:

```python
    if args.command == "run" and args.stage == "outcomes":
        from vectora import db as vdb
        from vectora.outcomes import resolver
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = resolver.resolve(con)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_outcomes.py -v` → 5 passed; fast suite `uv run pytest -m "not slow"` all pass; `uv run ruff check .` clean.

- [ ] **Step 5: Commit**

```bash
git add vectora/db.py vectora/outcomes vectora/__main__.py tests/test_outcomes.py
git commit -m "feat: outcomes resolver grades matured predictions via shared label machinery"
```

---

### Task 2: Vault generator v1

**Files:**
- Modify: `vectora/settings.py` (add `VAULT_DIR = REPO_ROOT / "vault"`)
- Create: `vectora/vault/__init__.py` (empty), `vectora/vault/generator.py`
- Modify: `vectora/__main__.py` (add `vault` stage)
- Test: `tests/test_vault.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vault.py
from vectora import db as vdb
from vectora.vault import generator as gen


def _seed(con):
    vdb.upsert(con, "symbols", [dict(
        symbol="GP", name=None, sector="Telecommunication",
        instrument_type="Equity", category="A", listing_status="active",
        first_seen="2013-01-01", last_seen="2026-07-16")])
    vdb.upsert(con, "data_quality", [
        {"date": "2026-07-16", "source": "dse_eod", "score": 100,
         "issues": "[]"}])
    vdb.upsert(con, "predictions", [
        dict(id="2026-07-16_g5_h10_GP", symbol="GP", date="2026-07-16",
             target="g5_h10", probability=0.61, model_id="m",
             quality_score=100, is_signal=True, suppressed_reason=None),
        dict(id="2026-07-16_g5_h10_ACI", symbol="ACI", date="2026-07-16",
             target="g5_h10", probability=0.31, model_id="m",
             quality_score=100, is_signal=False,
             suppressed_reason="below-probability-threshold"),
    ])
    vdb.upsert(con, "explanations", [dict(
        prediction_id="2026-07-16_g5_h10_GP", drivers="[]", analogs="{}",
        rendered="GP: 61% calibrated probability of the g5_h10 move.")])
    vdb.upsert(con, "events", [dict(
        id="e1", post_date="2026-07-16", symbol="GP",
        title="GP: Dividend Declared", body="10% cash", source="dse_news")])


def test_generate_writes_journal_prediction_company_home(test_db, tmp_path):
    _seed(test_db)
    result = gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    assert result["notes"] >= 4
    journal = (tmp_path / "Journal" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "2 predictions" in journal and "1 signal" in journal
    assert "quality 100" in journal
    pred = (tmp_path / "Predictions" / "2026-07" /
            "2026-07-16_g5_h10_GP.md").read_text(encoding="utf-8")
    assert "61%" in pred and "[[GP]]" in pred
    company = (tmp_path / "Companies" / "GP.md").read_text(encoding="utf-8")
    assert "Telecommunication" in company and "category: A" in company
    home = (tmp_path / "Home.md").read_text(encoding="utf-8")
    assert "2026-07-16" in home


def test_human_notes_survive_regeneration(test_db, tmp_path):
    _seed(test_db)
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    company = tmp_path / "Companies" / "GP.md"
    human = company.read_text(encoding="utf-8") + \
        "\n## Analyst Notes\nMy private thesis on GP.\n"
    company.write_text(human, encoding="utf-8")
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    text = company.read_text(encoding="utf-8")
    assert "My private thesis on GP." in text
    assert text.count(gen.MACHINE_BEGIN) == 1  # markers not duplicated


def test_non_signal_predictions_get_no_note(test_db, tmp_path):
    _seed(test_db)
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    assert not (tmp_path / "Predictions" / "2026-07" /
                "2026-07-16_g5_h10_ACI.md").exists()


def test_no_data_day_still_writes_journal(test_db, tmp_path):
    result = gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    assert result["notes"] >= 2   # journal + home always written
    journal = (tmp_path / "Journal" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "0 predictions" in journal
```

- [ ] **Step 2: Run to verify failure** — FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# vectora/vault/generator.py
"""Obsidian vault generator v1 (spec §7): Journal, signal Prediction notes,
Company notes, Home dashboard. Machine content lives strictly between the
markers; anything a human writes outside them survives regeneration
byte-identical (tested). Notes use [[wiki-links]] so Obsidian's graph view
is the knowledge graph."""
from pathlib import Path

from vectora.settings import VAULT_DIR

MACHINE_BEGIN = "<!-- vectora:begin -->"
MACHINE_END = "<!-- vectora:end -->"


def _write_machine(path: Path, content: str) -> None:
    block = f"{MACHINE_BEGIN}\n{content.rstrip()}\n{MACHINE_END}"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if MACHINE_BEGIN in text and MACHINE_END in text:
            pre = text.split(MACHINE_BEGIN, 1)[0]
            post = text.split(MACHINE_END, 1)[1]
            new = pre + block + post
        else:
            new = text.rstrip() + "\n\n" + block + "\n"
    else:
        new = block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new, encoding="utf-8")


def generate(con, date_str: str, vault_dir: Path = VAULT_DIR) -> dict:
    n = 0
    preds = con.execute(
        """
        SELECT p.id, p.symbol, p.target, p.probability, p.is_signal,
               p.suppressed_reason, e.rendered
        FROM predictions p LEFT JOIN explanations e ON e.prediction_id = p.id
        WHERE p.date = ? ORDER BY p.probability DESC
        """, [date_str]).fetchall()
    signals = [p for p in preds if p[4]]
    quality = con.execute(
        "SELECT score FROM data_quality WHERE date = ? AND source='dse_eod'",
        [date_str]).fetchone()
    events = con.execute(
        "SELECT symbol, title FROM events WHERE post_date = ? "
        "AND symbol IS NOT NULL LIMIT 20", [date_str]).fetchall()

    # Journal ---------------------------------------------------------------
    q = quality[0] if quality else "n/a"
    lines = [f"# Journal {date_str}", "",
             f"{len(preds)} predictions | {len(signals)} signal(s) | "
             f"quality {q}", ""]
    if signals:
        lines.append("## Signals")
        lines += [f"- [[{s[1]}]] {s[2]} at {s[3]:.0%} "
                  f"([[Predictions/{date_str[:7]}/{s[0]}|note]])"
                  for s in signals]
    if events:
        lines.append("")
        lines.append("## Company events")
        lines += [f"- [[{sym}]]: {title}" for sym, title in events]
    _write_machine(vault_dir / "Journal" / f"{date_str}.md", "\n".join(lines))
    n += 1

    # Prediction notes (signals only) ----------------------------------------
    for pid, symbol, target, prob, _sig, _rea, rendered in signals:
        body = [f"# {pid}", "",
                f"[[{symbol}]] | target {target} | {prob:.0%} calibrated", "",
                rendered or "(no explanation stored)", "",
                "## Outcome", "_pending resolution_"]
        _write_machine(
            vault_dir / "Predictions" / date_str[:7] / f"{pid}.md",
            "\n".join(body))
        n += 1

    # Company notes for symbols touched today (signals + events) -------------
    touched = sorted({s[1] for s in signals} | {e[0] for e in events})
    for symbol in touched:
        meta = con.execute(
            "SELECT sector, category FROM symbols WHERE symbol = ?",
            [symbol]).fetchone()
        stats = con.execute(
            """
            SELECT count(*),
                   sum(CASE WHEN o.hit THEN 1 ELSE 0 END)
            FROM predictions p JOIN outcomes o ON o.prediction_id = p.id
            WHERE p.symbol = ?
            """, [symbol]).fetchone()
        resolved, hits = (stats[0] or 0), (stats[1] or 0)
        sector = meta[0] if meta else None
        category = meta[1] if meta else None
        body = [f"# {symbol}", "",
                f"sector: {sector} | category: {category}", "",
                f"Prediction scorecard: {hits}/{resolved} resolved hits", ""]
        _write_machine(vault_dir / "Companies" / f"{symbol}.md",
                       "\n".join(body))
        n += 1

    # Home dashboard ----------------------------------------------------------
    total = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    resolved = con.execute("SELECT count(*) FROM outcomes").fetchone()[0]
    home = [f"# Vectora", "",
            f"Latest run: [[Journal/{date_str}|{date_str}]] | "
            f"{len(signals)} signal(s)", "",
            f"Lifetime: {total} predictions, {resolved} resolved", "",
            "_Research tool, not investment advice._"]
    _write_machine(vault_dir / "Home.md", "\n".join(home))
    n += 1
    return {"notes": n, "signals": len(signals)}
```

Add the CLI stage in `vectora/__main__.py`: stage choices gain `"vault"`; add:

```python
    if args.command == "run" and args.stage == "vault":
        from vectora import db as vdb
        from vectora.settings import DB_PATH
        from vectora.vault import generator
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            date_str = args.date or str(con.execute(
                "SELECT max(date) FROM predictions").fetchone()[0])
            result = generator.generate(con, date_str)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0
```

- [ ] **Step 4: Run tests** — 4 passed; fast suite; ruff. (If the f-string `f"# Vectora"` triggers ruff F541, drop the f-prefix.)

- [ ] **Step 5: Commit**

```bash
git add vectora/settings.py vectora/vault vectora/__main__.py tests/test_vault.py
git commit -m "feat: Obsidian vault generator v1 with human-note-safe machine sections"
```

---

### Task 3: alerts_log with cooldown + signal-aware digest

**Files:**
- Modify: `vectora/db.py` (SCHEMA)
- Create: `vectora/alerts/signals.py`
- Modify: `vectora/__main__.py` (digest branch wires the log)
- Test: `tests/test_signal_alerts.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signal_alerts.py
from vectora import db as vdb
from vectora.alerts import signals


def _signal(symbol, d):
    return dict(id=f"{d}_g5_h10_{symbol}", symbol=symbol, date=d,
                target="g5_h10", probability=0.62, model_id="m",
                quality_score=100, is_signal=True, suppressed_reason=None)


def test_new_signals_are_logged(test_db):
    vdb.upsert(test_db, "predictions",
               [_signal("GP", "2026-07-16"), _signal("ACI", "2026-07-16")])
    new = signals.log_signal_alerts(test_db, "2026-07-16")
    assert sorted(new) == ["ACI", "GP"]
    n = test_db.execute("SELECT count(*) FROM alerts_log").fetchone()[0]
    assert n == 2


def test_cooldown_suppresses_repeat_within_2_days(test_db):
    vdb.upsert(test_db, "predictions", [_signal("GP", "2026-07-16")])
    signals.log_signal_alerts(test_db, "2026-07-16")
    vdb.upsert(test_db, "predictions", [_signal("GP", "2026-07-17")])
    new = signals.log_signal_alerts(test_db, "2026-07-17")
    assert new == []   # same symbol within cooldown -> suppressed
    n = test_db.execute("SELECT count(*) FROM alerts_log").fetchone()[0]
    assert n == 1


def test_signal_after_cooldown_logs_again(test_db):
    vdb.upsert(test_db, "predictions", [_signal("GP", "2026-07-10")])
    signals.log_signal_alerts(test_db, "2026-07-10")
    vdb.upsert(test_db, "predictions", [_signal("GP", "2026-07-16")])
    new = signals.log_signal_alerts(test_db, "2026-07-16")
    assert new == ["GP"]


def test_rerun_same_day_is_idempotent(test_db):
    vdb.upsert(test_db, "predictions", [_signal("GP", "2026-07-16")])
    signals.log_signal_alerts(test_db, "2026-07-16")
    new = signals.log_signal_alerts(test_db, "2026-07-16")
    assert new == []
    assert test_db.execute("SELECT count(*) FROM alerts_log").fetchone()[0] == 1
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

Append to `SCHEMA` in `vectora/db.py`:

```sql
CREATE TABLE IF NOT EXISTS alerts_log (
    id TEXT PRIMARY KEY,               -- <date>_signal_<symbol>
    ts TIMESTAMP DEFAULT current_timestamp,
    alert_type TEXT, symbol TEXT, alert_date DATE,
    prediction_id TEXT
);
```

Create `vectora/alerts/signals.py`:

```python
"""Signal alert log with per-symbol cooldown (spec §16 anti-fatigue).

One batch of predictions lands per day post-close, so the digest is the
delivery channel; this module decides which signals count as NEW (not
alerted for the same symbol within the cooldown window) and records them.
Urgency tiers arrive with Phase 4's intraday scans.
"""
from datetime import date, timedelta

from vectora import db as vdb

COOLDOWN_DAYS = 2


def log_signal_alerts(con, date_str: str) -> list[str]:
    """Record alerts for today's signals outside cooldown; returns NEW symbols."""
    d = date.fromisoformat(date_str)
    floor = (d - timedelta(days=COOLDOWN_DAYS)).isoformat()
    rows = con.execute(
        """
        SELECT p.id, p.symbol FROM predictions p
        WHERE p.date = ? AND p.is_signal
          AND p.symbol NOT IN (
              SELECT symbol FROM alerts_log
              WHERE alert_type = 'signal' AND alert_date >= ? AND alert_date <= ?
          )
        ORDER BY p.symbol
        """, [date_str, floor, date_str]).fetchall()
    new = []
    for pred_id, symbol in rows:
        vdb.upsert(con, "alerts_log", [{
            "id": f"{date_str}_signal_{symbol}",
            "alert_type": "signal", "symbol": symbol,
            "alert_date": date_str, "prediction_id": pred_id,
        }])
        new.append(symbol)
    return new
```

Wait — `test_rerun_same_day_is_idempotent` expects the second same-day call to return `[]`. The NOT IN subquery covers `alert_date <= date_str` including today, so a re-run finds today's own alert row and suppresses — correct as written. (The upsert PK would also make it harmless, but returning `[]` matters for the digest subject.)

In `vectora/__main__.py`'s digest branch, after computing `body` and before building the subject, wire the log (inside the `try` block, while `con` is open):

```python
            from vectora.alerts import signals as sig
            new_symbols = sig.log_signal_alerts(con, date_str)
```

and change the subject line to:

```python
        n_signals = body.count("### ")
        prefix = f"[{len(new_symbols)} NEW] " if new_symbols else ""
        subject = f"{prefix}Vectora digest {date_str} - {n_signals} signal(s)"
```

- [ ] **Step 4: Run tests** — 4 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/db.py vectora/alerts/signals.py vectora/__main__.py tests/test_signal_alerts.py
git commit -m "feat: signal alert log with 48h per-symbol cooldown; digest flags NEW signals"
```

---

### Task 4: Pipeline integration + real run + merge

**Files:**
- Modify: `.github/workflows/eod-pipeline.yml`

- [ ] **Step 1: Add workflow steps**

In `.github/workflows/eod-pipeline.yml`, between the `Predict` step and the `Digest` step insert:

```yaml
      - name: Outcomes
        continue-on-error: true
        run: uv run python -m vectora run outcomes
      - name: Vault
        continue-on-error: true
        run: uv run python -m vectora run vault
```

In the `Commit data` step, extend the add line to include the vault and reports:

```yaml
          git add -f data/raw data/vectora.duckdb data/reference vault reports
```

- [ ] **Step 2: Real run (local)**

```bash
uv run python -m vectora run outcomes
uv run python -m vectora run vault
```

Expected: outcomes prints `{"resolved": 0, "pending": 664}` (yesterday's predictions need 10/30 trading days to mature — 0 resolved is correct); vault prints `{"notes": >= 2, ...}` and `vault/Journal/2026-07-16.md`, `vault/Home.md` exist (no signals yesterday → no prediction notes). Inspect the journal — it should show 664 predictions, 0 signals, quality 100, and the day's company events.

- [ ] **Step 3: Fast suite + ruff, commit, merge, push, verify**

```bash
uv run pytest -m "not slow" && uv run ruff check .
git add .github/workflows/eod-pipeline.yml vault data/vectora.duckdb
git commit -m "feat: outcomes + vault stages in scheduled pipeline; first vault build"
git checkout main && git pull
git merge --no-ff phase-3b-loop -m "Merge phase-3b: outcomes resolver, vault v1, signal alert log"
git push
```

(The `merge=ours` driver auto-resolves any DuckDB conflict; if data diverged, run `uv run python -m vectora run eod` after the merge to gap-fill, then commit.)

Then dispatch a verification run and confirm all steps green:

```bash
& "C:\Program Files\GitHub CLI\gh.exe" workflow run eod-pipeline --ref main -f date=2026-07-16
& "C:\Program Files\GitHub CLI\gh.exe" run watch --exit-status <run-id>
```

---

## Execution notes

- Strict order 1→4; suite green + commit per task. Expected final fast-suite size ≈ 145 tests.
- Task 4's real run touches the real DB and vault; everything before uses isolated test DBs.
- From the first future signal onward: it gets a Prediction note in the vault, an alerts_log row, and a `[N NEW]` digest subject; when its horizon matures the outcomes resolver grades it and the Company scorecard updates.
- After merge, spec §17's remaining items (calibration accounting, error autopsy, retrain triggers) are Phase 5, and intraday urgency tiers are Phase 4.
