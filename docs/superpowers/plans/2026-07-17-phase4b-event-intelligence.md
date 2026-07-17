# Vectora Phase 4B: Event Intelligence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify every DSE announcement into a typed taxonomy with materiality (spec §12.1), backfill the announcement archive to 2013 so event history is deep enough to study, and compute per-type event-impact statistics (spec §12.2) — wired into the daily pipeline and vault.

**Architecture:** `events.classifier` applies ordered regex rules over announcement titles (grounded in the 403 real events collected so far — the rule list below embeds the observed title shapes verbatim) writing to an append-only `event_labels` table; raw `events` rows are never mutated. `tools/backfill_news.py` walks `old_news.php` month-by-month from 2013-01 through today via the existing PoliteSession + `parse_news` (~165 polite requests), turning six days of event history into thirteen years. `events.studies` then joins labeled events to the price panel and computes market-adjusted forward returns per event type and horizon into an `event_studies` table, rendered as a vault note. The daily pipeline classifies new events each run; the journal lists typed material events instead of raw titles.

**Tech Stack:** existing only. No new dependencies.

**Existing contracts (all on `main`):**
- `events` table: `(id PK, post_date, symbol, title, body, source, scraped_at)`; ~400 rows now, populated daily by `collect_news`. Symbol is NULL for exchange-level notices; synthetic codes never appear (Trading Code field is authoritative).
- `vectora/collect/dse_news.py`: `fetch_news(session, start, end)` (old_news.php with startDate/endDate) and `parse_news(html) -> list[dict]` matching the events schema.
- `vectora/collect/raw_store.py`: `save_raw(raw_dir, source, run_date, name, payload, url=None)` (gzips).
- `vectora/http.py`: `PoliteSession` (1.5s delay). `vectora/features/base.py`: `load_panel`. `vectora/vault/generator.py`: `_write_machine(path, content)` and `MACHINE_BEGIN/END`.
- `vectora/__main__.py`: stages eod/train/predict/digest/outcomes/vault/regime. `tests/conftest.py`: `test_db`.
- Test-seeding rule (learned): bulk-insert synthetic price rows via a registered polars frame (`con.execute("INSERT INTO prices_raw SELECT * FROM df")`), never `vdb.upsert` for >1k rows.
- Branch: `git checkout main && git pull && git checkout -b phase-4b-events`. `uv run …` from repo root; fast tests `uv run pytest -m "not slow"` (currently 160).

**Observed title shapes (2026-07 collection, frequency-ordered — the classifier's ground truth):**
`Daily NAV` (168), `Record date for entitlement of coupon payment` (19), `Board Meeting schedule under LR 16(1)` (16), `DSE NEWS: Awareness Message for Investors` / `BSEC NEWS: ...` (24), `DSE NEWS: Withdrawal of Authorized Representative(s)` (12), `Resumption after Record Date` (10), `Query Response` (10), `Suspension for Record Date` (8), `Q2 Financials` / `Q1 Financials` (14), `Spot News` (7), `Dividend Disbursement` (7), `DSE NEWS: Daily Turnover of Main Board` (6), `Clarification on the news published in the online news` (6), `Credit Rating Result` (5), `Halt of trading of the company` (5), `Dividend Declaration` (4), `Declaration of Interim Dividend and Audited Q2 Financials` (3), `Price Limit Open` (3), `Record Date and key features of the rights issuance` (2).

**File structure:**

```
vectora/events/__init__.py, classifier.py, studies.py
vectora/db.py                      # + event_labels, event_studies tables
tools/backfill_news.py
vectora/__main__.py                # + events stage
vectora/vault/generator.py         # journal: typed material events
.github/workflows/eod-pipeline.yml # + Events step after eod
tests/events/__init__.py, test_classifier.py, test_studies.py
```

---

### Task 1: Event taxonomy classifier

**Files:**
- Modify: `vectora/db.py` (SCHEMA)
- Create: `vectora/events/__init__.py` (empty), `vectora/events/classifier.py`
- Modify: `vectora/__main__.py` (add `events` stage)
- Test: `tests/events/__init__.py` (empty), `tests/events/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/events/test_classifier.py
from vectora import db as vdb
from vectora.events import classifier as cls

# (title, expected_type, expected_materiality)
CASES = [
    ("GLDNJMF: Daily NAV", "daily_nav", 0),
    ("DSE NEWS: Daily Turnover of Main Board", "market_stats", 0),
    ("DSE NEWS: Awareness Message for Investors", "admin_notice", 0),
    ("BSEC NEWS: Awareness Message for Investors", "admin_notice", 0),
    ("DSE NEWS: Withdrawal of Authorized Representative", "admin_notice", 0),
    ("UNIONCAP: Board Meeting schedule under LR 16(1)", "board_meeting", 2),
    ("NATLIFEINS: Reschedule of Board Meeting under LR 16(1)", "board_meeting", 2),
    ("TB10Y0132: Record date for entitlement of coupon payment",
     "record_date", 1),
    ("PRIMELIFE: Resumption after Record Date", "trading_resume", 1),
    ("XYZ: Suspension for Record Date", "trading_suspension", 1),
    ("ABC: Halt of trading of the company", "trading_halt", 3),
    ("ABC: Price Limit Open", "price_limit_change", 2),
    ("MERCANBANK: Credit Rating Result", "credit_rating", 1),
    ("LINDEBD: Dividend Disbursement", "dividend_disbursement", 1),
    ("SQURPHARMA: Dividend Declaration", "dividend_declared", 3),
    ("ACI: Declaration of Interim Dividend and Audited Q2 Financials",
     "dividend_declared", 3),
    ("GP: Q2 Financials", "earnings_release", 3),
    ("GP: Q1 Financials", "earnings_release", 3),
    ("FIRSTFIN: Spot News", "spot_market", 2),
    ("ABC: Record Date and key features of the rights issuance",
     "rights_issue", 3),
    ("ABC: Query Response", "query_response", 2),
    ("ABC: Clarification on the news published in the online news",
     "query_response", 2),
    ("ABC: Signing of Selling & Distribution Agreement", "business_update", 2),
    ("ABC: Something entirely novel here", "unclassified", 1),
]


def test_taxonomy_on_observed_titles():
    for title, etype, mat in CASES:
        got_type, got_mat = cls.classify_title(title)
        assert got_type == etype, f"{title!r}: {got_type} != {etype}"
        assert got_mat == mat, f"{title!r}: materiality {got_mat} != {mat}"


def test_classify_new_writes_labels_and_is_incremental(test_db):
    vdb.upsert(test_db, "events", [
        dict(id="e1", post_date="2026-07-16", symbol="GP",
             title="GP: Q2 Financials", body="EPS 5.2", source="dse_news"),
        dict(id="e2", post_date="2026-07-16", symbol=None,
             title="DSE NEWS: Greetings Message", body="", source="dse_news"),
    ])
    result = cls.classify_new(test_db)
    assert result == {"classified": 2}
    rows = dict(test_db.execute(
        "SELECT event_id, event_type FROM event_labels").fetchall())
    assert rows["e1"] == "earnings_release"
    assert rows["e2"] == "admin_notice"
    assert cls.classify_new(test_db) == {"classified": 0}   # incremental


def test_event_labels_table_exists(test_db):
    tables = {r[0] for r in test_db.execute("SHOW TABLES").fetchall()}
    assert {"event_labels", "event_studies"} <= tables
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/events -v` → FAIL.

- [ ] **Step 3: Implement**

Append to `SCHEMA` in `vectora/db.py`:

```sql
CREATE TABLE IF NOT EXISTS event_labels (
    event_id TEXT PRIMARY KEY,
    event_type TEXT, materiality INTEGER,        -- 0 noise .. 3 price-sensitive
    classified_at TIMESTAMP DEFAULT current_timestamp
);
CREATE TABLE IF NOT EXISTS event_studies (
    event_type TEXT, horizon INTEGER, n INTEGER,
    mean_abn_ret DOUBLE, median_abn_ret DOUBLE, pos_share DOUBLE,
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (event_type, horizon)
);
```

Create `vectora/events/classifier.py`:

```python
"""Announcement taxonomy (spec §12.1): ordered regex rules over titles,
grounded in observed DSE title shapes (see plan doc). First match wins.
Materiality: 0 routine noise, 1 mechanical/administrative, 2 informative,
3 price-sensitive. Labels are append-only; raw events are never mutated."""
import re

from vectora import db as vdb

# (compiled regex, event_type, materiality) — ordered, first match wins.
RULES = [
    (r"daily nav", "daily_nav", 0),
    (r"daily turnover|market statistics", "market_stats", 0),
    (r"awareness message|greetings message|withdrawal of authorized|"
     r"lodging investor complaints", "admin_notice", 0),
    (r"halt of trading", "trading_halt", 3),
    (r"price limit", "price_limit_change", 2),
    (r"suspension for record date", "trading_suspension", 1),
    (r"resumption after record date", "trading_resume", 1),
    (r"record date.*rights|rights issu", "rights_issue", 3),
    (r"record date", "record_date", 1),
    (r"board meeting", "board_meeting", 2),
    (r"declaration of.*dividend|dividend declaration", "dividend_declared", 3),
    (r"dividend disbursement", "dividend_disbursement", 1),
    (r"q[1-4] financials|quarterly financ|audited financ|earnings",
     "earnings_release", 3),
    (r"credit rating", "credit_rating", 1),
    (r"spot news|spot market", "spot_market", 2),
    (r"query response|clarification on the news", "query_response", 2),
    (r"agreement|acquisition|new plant|expansion|contract",
     "business_update", 2),
    (r"agm|annual general meeting", "agm_notice", 1),
    (r"category", "category_change", 3),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), t, m) for p, t, m in RULES]

# dividend_declared beats earnings_release for combined announcements
# ("Declaration of Interim Dividend and Audited Q2 Financials") because
# the dividend rule sits earlier in RULES — order is load-bearing.


def classify_title(title: str) -> tuple[str, int]:
    text = title.strip()
    for rx, etype, mat in _COMPILED:
        if rx.search(text):
            return etype, mat
    return "unclassified", 1


def classify_new(con) -> dict:
    rows = con.execute(
        """
        SELECT e.id, e.title FROM events e
        LEFT JOIN event_labels l ON l.event_id = e.id
        WHERE l.event_id IS NULL
        """).fetchall()
    labels = []
    for event_id, title in rows:
        etype, mat = classify_title(title or "")
        labels.append({"event_id": event_id, "event_type": etype,
                       "materiality": mat})
    if labels:
        vdb.upsert(con, "event_labels", labels)
    return {"classified": len(labels)}
```

Add the CLI stage in `vectora/__main__.py`: stage choices gain `"events"`; add:

```python
    if args.command == "run" and args.stage == "events":
        from vectora import db as vdb
        from vectora.events import classifier
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = classifier.classify_new(con)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0
```

Rule-tuning note: if a CASES row fails, adjust RULES order/patterns until the observed-title table passes — the test table is the contract, built from real data.

- [ ] **Step 4: Run tests** — 3 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/db.py vectora/events vectora/__main__.py tests/events
git commit -m "feat: announcement taxonomy classifier grounded in observed DSE titles"
```

---

### Task 2: News archive backfill (2013 → today)

**Files:**
- Create: `tools/backfill_news.py`
- Test: `tests/test_backfill_news.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backfill_news.py
from vectora import db as vdb
from tools import backfill_news as bn


def test_month_windows():
    wins = bn.month_windows("2025-11-01", "2026-02-10")
    assert wins == [("2025-11-01", "2025-11-30"), ("2025-12-01", "2025-12-31"),
                    ("2026-01-01", "2026-01-31"), ("2026-02-01", "2026-02-10")]


NEWS_HTML = """
<table class="table-news">
<tr><th>Trading Code:</th><td>GP</td></tr>
<tr><th>News Title:</th><td>GP: Dividend Declaration</td></tr>
<tr><th>News:</th><td>Cash dividend 125%.</td></tr>
<tr><th>Post Date:</th><td>2025-11-05</td></tr>
</table>"""


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None):
        self.calls.append(params)
        return NEWS_HTML


def test_backfill_range_fetches_each_month_and_upserts(test_db, tmp_path):
    s = FakeSession()
    result = bn.backfill(test_db, s, "2025-11-01", "2026-01-15",
                         raw_dir=tmp_path)
    assert len(s.calls) == 3                       # three month windows
    assert s.calls[0]["startDate"] == "2025-11-01"
    assert result["months"] == 3
    # same canned HTML each month -> same event id -> one row (idempotent)
    assert test_db.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    assert len(list(tmp_path.rglob("*.html.gz"))) == 3
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# tools/backfill_news.py
"""Backfill the DSE announcement archive month-by-month via old_news.php.

Usage: uv run python tools/backfill_news.py [START [END]]
Defaults: 2013-01-01 through today. ~165 polite requests (1.5s spacing),
raw pages land in data/raw/dse_news_backfill/<month>/, parsed items upsert
into events (sha256 ids make re-runs idempotent). Event history is the
fuel for spec §12.2 event-impact studies.
"""
import calendar
import sys
from datetime import date
from pathlib import Path

from vectora import db as vdb
from vectora.collect import dse_news
from vectora.collect.raw_store import save_raw
from vectora.http import PoliteSession
from vectora.settings import DB_PATH, RAW_DIR


def month_windows(start: str, end: str) -> list[tuple[str, str]]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    cur = s
    while cur <= e:
        last_day = date(cur.year, cur.month,
                        calendar.monthrange(cur.year, cur.month)[1])
        win_end = min(last_day, e)
        out.append((cur.isoformat(), win_end.isoformat()))
        cur = last_day.replace(day=1)
        cur = (date(cur.year + 1, 1, 1) if cur.month == 12
               else date(cur.year, cur.month + 1, 1))
    return out


def backfill(con, session, start: str, end: str,
             raw_dir: Path = RAW_DIR) -> dict:
    months = items = 0
    for win_start, win_end in month_windows(start, end):
        html = dse_news.fetch_news(session, win_start, win_end)
        save_raw(raw_dir, "dse_news_backfill", win_start[:7], "old_news",
                 html, url=dse_news.URL)
        parsed = dse_news.parse_news(html)
        if parsed:
            items += vdb.upsert(con, "events", parsed)
        months += 1
        print(f"{win_start[:7]}: {len(parsed)} items", flush=True)
    return {"months": months, "items": items}


def main() -> int:
    start = sys.argv[1] if len(sys.argv) > 1 else "2013-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()
    con = vdb.connect(DB_PATH)
    try:
        vdb.init_schema(con)
        result = backfill(con, PoliteSession(), start, end)
        print(result)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests** — 2 passed; fast suite; ruff. Commit:

```bash
git add tools/backfill_news.py tests/test_backfill_news.py
git commit -m "feat: month-chunked news archive backfill tool"
```

- [ ] **Step 5: Run the real backfill (LIVE, long)**

Run in background: `uv run python tools/backfill_news.py`
Expected: ~165 month lines; total runtime 10–25 min (1.5s politeness + large months). Some months may return 0 items (archive gaps are real — note them, don't fail). Afterward run `uv run python -m vectora run events` to classify everything, then inspect:

`uv run python -c "
from vectora import db as vdb
con = vdb.connect('data/vectora.duckdb'); vdb.init_schema(con)
print('events total:', con.execute('SELECT count(*) FROM events').fetchone()[0])
print('by year:', con.execute('SELECT year(post_date), count(*) FROM events GROUP BY 1 ORDER BY 1').fetchall())
print('by type:', con.execute('SELECT event_type, count(*) FROM event_labels GROUP BY 1 ORDER BY 2 DESC LIMIT 12').fetchall())
print('unclassified share:', con.execute(\"SELECT round(avg(CASE WHEN event_type='unclassified' THEN 1.0 ELSE 0 END),3) FROM event_labels\").fetchone()[0])
con.close()"`

Judgment gates: tens of thousands of events across 13 years; unclassified share below ~0.25 (if higher, extend RULES with the top unclassified shapes and re-run `classify_new` — labels table is append-only per event, so delete-and-reclassify unclassified rows: `DELETE FROM event_labels WHERE event_type='unclassified'` then rerun). Commit the data:

```bash
git add -f data/raw/dse_news_backfill data/vectora.duckdb
git commit -m "data: announcement archive backfill 2013-2026, classified"
```

---

### Task 3: Event-impact studies

**Files:**
- Create: `vectora/events/studies.py`
- Test: `tests/events/test_studies.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/events/test_studies.py
import datetime as dt

import numpy as np
import polars as pl

from vectora import db as vdb
from vectora.events import studies


def _seed(con, n_days=40, n_syms=40, bump_sym="S00", bump_day=20,
          bump=0.05, seed=9):
    """Flat market; one symbol jumps +5% the day after its event."""
    rng = np.random.default_rng(seed)
    rows = []
    d0 = dt.date(2026, 1, 1)
    px = {f"S{i:02d}": 100.0 for i in range(n_syms)}
    for day in range(n_days):
        d = d0 + dt.timedelta(days=day)
        for sym in px:
            drift = bump if (sym == bump_sym and day == bump_day + 1) else 0.0
            px[sym] *= (1 + drift + float(rng.normal(0, 0.001)))
            p = round(px[sym], 4)
            rows.append(dict(symbol=sym, date=d, open=p, high=p, low=p,
                             close=p, ltp=p, ycp=None, trades=10,
                             value_mn=1.0, volume=100, source="mendeley"))
    df = pl.DataFrame(rows)  # noqa: F841
    con.execute("INSERT INTO prices_raw SELECT * FROM df")
    event_date = (d0 + dt.timedelta(days=bump_day)).isoformat()
    vdb.upsert(con, "events", [dict(
        id="ev1", post_date=event_date, symbol=bump_sym,
        title=f"{bump_sym}: Dividend Declaration", body="", source="dse_news")])
    vdb.upsert(con, "event_labels", [dict(
        event_id="ev1", event_type="dividend_declared", materiality=3)])


def test_event_study_detects_abnormal_return(test_db):
    _seed(test_db)
    result = studies.compute(test_db, min_events=1, horizons=(1, 3))
    assert result["types"] >= 1
    row = test_db.execute(
        """
        SELECT n, mean_abn_ret, pos_share FROM event_studies
        WHERE event_type = 'dividend_declared' AND horizon = 1
        """).fetchone()
    assert row[0] == 1
    assert row[1] > 0.03            # ~+5% vs ~flat market
    assert row[2] == 1.0


def test_min_events_threshold_excludes_thin_types(test_db):
    _seed(test_db)
    result = studies.compute(test_db, min_events=5, horizons=(1,))
    assert result["types"] == 0
    assert test_db.execute(
        "SELECT count(*) FROM event_studies").fetchone()[0] == 0


def test_vault_note_rendered(test_db, tmp_path):
    _seed(test_db)
    studies.compute(test_db, min_events=1, horizons=(1, 3))
    path = studies.write_vault_note(test_db, vault_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "dividend_declared" in text and "h1" in text
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/events/studies.py
"""Event-impact studies (spec §12.2): market-adjusted forward returns per
event type. abnormal = symbol forward return minus median market forward
return over the same window; the table answers 'what historically happens
in the H days after this kind of announcement on the DSE'."""
from pathlib import Path

import polars as pl

from vectora import db as vdb
from vectora.features import base
from vectora.settings import VAULT_DIR
from vectora.vault.generator import _write_machine

DEFAULT_HORIZONS = (1, 3, 5, 10)
MIN_EVENTS = 30


def compute(con, min_events: int = MIN_EVENTS,
            horizons: tuple = DEFAULT_HORIZONS) -> dict:
    events = con.execute(
        """
        SELECT e.symbol, e.post_date AS date, l.event_type
        FROM events e JOIN event_labels l ON l.event_id = e.id
        WHERE e.symbol IS NOT NULL AND l.materiality >= 1
        """).pl()
    if events.height == 0:
        return {"types": 0}
    panel = base.load_panel(con).select(["symbol", "date", "close"])
    med = panel.group_by("date").agg(
        pl.col("close").log().diff().alias("_")).select("date")  # placeholder
    # forward return per symbol and market median forward return per date
    out_rows = []
    frame = panel.sort(["symbol", "date"])
    for h in horizons:
        fwd = frame.with_columns(
            (pl.col("close").shift(-h).over("symbol") / pl.col("close") - 1)
            .alias("fwd"))
        mkt = fwd.group_by("date").agg(
            pl.col("fwd").median().alias("mkt_fwd"))
        joined = (events.join(fwd.select(["symbol", "date", "fwd"]),
                              on=["symbol", "date"], how="inner")
                  .join(mkt, on="date", how="left")
                  .with_columns((pl.col("fwd") - pl.col("mkt_fwd"))
                                .alias("abn"))
                  .filter(pl.col("abn").is_not_null()))
        stats = (joined.group_by("event_type")
                 .agg(pl.len().alias("n"),
                      pl.col("abn").mean().alias("mean_abn"),
                      pl.col("abn").median().alias("median_abn"),
                      (pl.col("abn") > 0).mean().alias("pos_share"))
                 .filter(pl.col("n") >= min_events))
        for r in stats.iter_rows(named=True):
            out_rows.append({
                "event_type": r["event_type"], "horizon": h,
                "n": int(r["n"]), "mean_abn_ret": r["mean_abn"],
                "median_abn_ret": r["median_abn"],
                "pos_share": r["pos_share"],
            })
    if out_rows:
        vdb.upsert(con, "event_studies", out_rows)
    return {"types": len({r["event_type"] for r in out_rows})}


def write_vault_note(con, vault_dir: Path = VAULT_DIR) -> Path:
    rows = con.execute(
        """
        SELECT event_type, horizon, n, mean_abn_ret, median_abn_ret, pos_share
        FROM event_studies ORDER BY event_type, horizon
        """).fetchall()
    lines = ["# Event impact studies", "",
             "Market-adjusted forward returns after each announcement type.",
             "", "| type | h | n | mean | median | share>0 |", "|" + "---|" * 6]
    for t, h, n, mean, median, pos in rows:
        lines.append(f"| {t} | h{h} | {n} | {mean:+.2%} | {median:+.2%} "
                     f"| {pos:.0%} |")
    path = Path(vault_dir) / "Patterns" / "event-impact.md"
    _write_machine(path, "\n".join(lines))
    return path
```

Cleanup note: remove the dead `med = ...` placeholder line before committing — it is a sketch artifact flagged here deliberately (delete it; the market median is computed inside the loop as `mkt`).

- [ ] **Step 4: Run tests** — 3 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/events/studies.py tests/events/test_studies.py
git commit -m "feat: market-adjusted event-impact studies with vault note"
```

---

### Task 4: Pipeline integration + real studies + merge

**Files:**
- Modify: `.github/workflows/eod-pipeline.yml`, `vectora/vault/generator.py`
- Test: append to `tests/test_vault.py`

- [ ] **Step 1: Journal shows typed material events — failing test (append to tests/test_vault.py)**

```python
def test_journal_types_material_events(test_db, tmp_path):
    _seed(test_db)
    vdb.upsert(test_db, "event_labels", [dict(
        event_id="e1", event_type="dividend_declared", materiality=3)])
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    journal = (tmp_path / "Journal" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "dividend_declared" in journal
```

(`_seed` in test_vault.py already inserts event `e1` "GP: Dividend Declared".)

- [ ] **Step 2: Implement** — in `vectora/vault/generator.py` `generate()`, replace the events query with a label-joined one:

```python
    events = con.execute(
        """
        SELECT e.symbol, e.title, coalesce(l.event_type, 'unclassified')
        FROM events e LEFT JOIN event_labels l ON l.event_id = e.id
        WHERE e.post_date = ? AND e.symbol IS NOT NULL
          AND coalesce(l.materiality, 1) >= 1
        ORDER BY coalesce(l.materiality, 1) DESC
        LIMIT 20
        """, [date_str]).fetchall()
```

and the journal event lines become:

```python
        lines += [f"- [[{sym}]] ({etype}): {title}"
                  for sym, title, etype in events]
```

Note the unpack order: the query returns (symbol, title, event_type) — unpack as `for sym, title, etype`.

- [ ] **Step 3: Workflow step** — in `.github/workflows/eod-pipeline.yml`, insert between "Run EOD pipeline" and "Regime":

```yaml
      - name: Events
        continue-on-error: true
        run: uv run python -m vectora run events
```

- [ ] **Step 4: Real studies run**

```bash
uv run python -m vectora run events
uv run python -c "
from vectora import db as vdb
from vectora.events import studies
con = vdb.connect('data/vectora.duckdb'); vdb.init_schema(con)
print(studies.compute(con))
print(studies.write_vault_note(con))
con.close()"
```

Inspect `vault/Patterns/event-impact.md` — report the table. Sanity gates: `dividend_declared` and `earnings_release` should have n in the hundreds-to-thousands post-backfill; mean abnormal returns within ±5% (larger means a join bug); `daily_nav` must NOT appear (materiality 0 excluded).

- [ ] **Step 5: Fast suite + ruff, commit, merge, push, dispatch verification run**

```bash
uv run pytest -m "not slow" && uv run ruff check .
git add vectora/vault/generator.py .github/workflows/eod-pipeline.yml tests/test_vault.py vault data/vectora.duckdb
git commit -m "feat: events stage in pipeline; typed journal events; real event studies"
git checkout main && git pull
git merge --no-ff phase-4b-events -m "Merge phase-4b: event taxonomy, archive backfill, impact studies"
git push
& "C:\Program Files\GitHub CLI\gh.exe" workflow run eod-pipeline --ref main -f date=2026-07-17
```

Confirm the run is green including the new Events step.

---

## Execution notes

- Order 1→4. Task 2's live backfill is long (10–25 min) — run it in the background and build Task 3 meanwhile; its data gates run before Task 4.
- The studies module recomputes from scratch (seconds) — no incremental complexity.
- Event FEATURES for the models (days-since-event etc., spec §8 event family) intentionally wait until the studies table shows which types carry signal — that's Phase 4C/5 work with data to stand on.
- Remaining Phase 4 slices after this: 4C Z-module + pump-phase + pre-announcement footprints (now feedable by labeled events), 4D intraday scans + urgency tiers.
