# Vectora Phase 4D: Intraday Scans + Urgent Tier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collect intraday snapshots 4×/day at DSE's chart-publication points, detect volume surges and near-circuit moves against 21-day baselines, and deliver them as an urgent email tier with cooldowns and a daily cap (spec §16; user fact: DSE publishes hourly chart data 4×/trading day).

**Architecture:** `collect.dse_intraday` scrapes `latest_share_price_scroll_l.php` (verified 2026-07-17: same `shares-table` markup as the day-end archive, 11 columns `# | TRADING CODE | LTP | HIGH | LOW | CLOSEP | YCP | CHANGE | TRADE | VALUE (mn) | VOLUME`) into a new `intraday_snapshots` table. `alerts.intraday` compares each snapshot row to the symbol's trailing 21-day median full-day volume (a volume already 3× a normal FULL day, mid-session, is the anomaly) and to the circuit band (|LTP/YCP−1| ≥ 8.5%), applies a 48h per-symbol cooldown via the existing `alerts_log`, caps urgent emails at 3/day (overflow folds into the evening digest which already lists zwatch/journal surfaces), and sends via the existing `digest.send_or_save` (secret-gated, reports/ fallback). A new `intraday-scan.yml` runs at 11:00/12:00/13:00/14:00 Dhaka Sun–Thu sharing the `pipeline-writes` concurrency group.

**Tech Stack:** existing only.

**Existing contracts:** `PoliteSession.get`, `save_raw(raw_dir, source, run_date, name, payload, url=None)`, `vdb.upsert` (atomic), `alerts_log(id PK, ts, alert_type, symbol, alert_date, prediction_id)`, `digest.send_or_save(subject, body, reports_dir=REPORTS_DIR) -> {"sent", "path"}`, `prices` view, `calendar.is_trading_day/load_holidays`, `tradable_universe`. `_num`/`_int` parse helpers live in `vectora/collect/dse_eod.py` (comma numbers, "-" → None). Test-seeding: bulk polars insert for >1k rows. Branch `phase-4d-intraday` off main. Fast tests: `uv run pytest -m "not slow"` (currently 182).

**File structure:**

```
vectora/collect/dse_intraday.py     # fetch + parse + collect runner
vectora/alerts/intraday.py          # anomaly detection + urgent send
vectora/db.py                       # + intraday_snapshots table
vectora/__main__.py                 # + intraday stage
.github/workflows/intraday-scan.yml
tests/collect/test_dse_intraday.py, tests/test_intraday_alerts.py
tests/fixtures/latest_share_price.html   # recorded live
```

---

### Task 1: Intraday snapshot collector

**Files:**
- Modify: `vectora/db.py` (SCHEMA)
- Create: `vectora/collect/dse_intraday.py`, `tests/fixtures/latest_share_price.html` (recorded), `tests/collect/test_dse_intraday.py`

- [ ] **Step 1: Record the fixture (live, one request)**

```bash
uv run python -c "
from vectora.http import PoliteSession
from vectora.settings import DSE_BASE
html = PoliteSession().get(f'{DSE_BASE}/latest_share_price_scroll_l.php')
open('tests/fixtures/latest_share_price.html', 'w', encoding='utf-8').write(html)
print(len(html), 'bytes')"
```

Expected ~550KB; verify `grep -c shares-table tests/fixtures/latest_share_price.html` ≥ 1.

- [ ] **Step 2: Write the failing tests**

```python
# tests/collect/test_dse_intraday.py
from vectora import db as vdb
from vectora.collect import dse_intraday


def _rows(fixtures_dir):
    html = (fixtures_dir / "latest_share_price.html").read_text(encoding="utf-8")
    return dse_intraday.parse_latest(html)


def test_parses_many_rows(fixtures_dir):
    assert len(_rows(fixtures_dir)) > 300


def test_row_shape(fixtures_dir):
    r = _rows(fixtures_dir)[0]
    assert set(r) == {"symbol", "ltp", "high", "low", "closep", "ycp",
                      "change", "trades", "value_mn", "volume"}
    assert isinstance(r["symbol"], str) and r["symbol"]
    assert r["volume"] is None or isinstance(r["volume"], int)


def test_empty_page_returns_empty():
    assert dse_intraday.parse_latest("<html></html>") == []


def test_collect_upserts_snapshots(test_db, fixtures_dir, tmp_path):
    html = (fixtures_dir / "latest_share_price.html").read_text(encoding="utf-8")

    class FakeSession:
        def get(self, url, params=None):
            return html

    n = dse_intraday.collect_intraday(
        test_db, FakeSession(), ts="2026-07-17 12:00:00", raw_dir=tmp_path)
    assert n > 300
    stored = test_db.execute(
        "SELECT count(*) FROM intraday_snapshots").fetchone()[0]
    assert stored == n
    # idempotent for the same ts
    dse_intraday.collect_intraday(
        test_db, FakeSession(), ts="2026-07-17 12:00:00", raw_dir=tmp_path)
    assert test_db.execute(
        "SELECT count(*) FROM intraday_snapshots").fetchone()[0] == n
    assert len(list(tmp_path.rglob("*.html.gz"))) == 1
```

- [ ] **Step 3: Run to verify failure** — FAIL (table/module missing).

- [ ] **Step 4: Implement**

Append to `SCHEMA` in `vectora/db.py`:

```sql
CREATE TABLE IF NOT EXISTS intraday_snapshots (
    symbol TEXT, ts TIMESTAMP, ltp DOUBLE, high DOUBLE, low DOUBLE,
    closep DOUBLE, ycp DOUBLE, change DOUBLE, trades BIGINT,
    value_mn DOUBLE, volume BIGINT,
    PRIMARY KEY (symbol, ts)
);
```

Create `vectora/collect/dse_intraday.py`:

```python
"""Intraday snapshot collector (spec §16, Phase 4D).

Endpoint (verified 2026-07-17): /latest_share_price_scroll_l.php —
same shares-table markup as the day-end archive, 11 columns:
# | TRADING CODE | LTP | HIGH | LOW | CLOSEP | YCP | CHANGE | TRADE
| VALUE (mn) | VOLUME. DSE refreshes chart data hourly, 4x per trading
day; the workflow polls exactly those publication points.
"""
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from vectora import db as vdb
from vectora.collect.dse_eod import _int, _num
from vectora.collect.raw_store import save_raw
from vectora.http import PoliteSession
from vectora.settings import DSE_BASE, RAW_DIR

URL = f"{DSE_BASE}/latest_share_price_scroll_l.php"


def fetch_latest(session: PoliteSession) -> str:
    return session.get(URL)


def parse_latest(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="shares-table")
    if table is None:
        return []
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 11:
            continue
        c = [td.get_text(strip=True) for td in tds]
        rows.append({
            "symbol": c[1], "ltp": _num(c[2]), "high": _num(c[3]),
            "low": _num(c[4]), "closep": _num(c[5]), "ycp": _num(c[6]),
            "change": _num(c[7]), "trades": _int(c[8]),
            "value_mn": _num(c[9]), "volume": _int(c[10]),
        })
    return rows


def collect_intraday(con, session, ts: str | None = None,
                     raw_dir: Path = RAW_DIR) -> int:
    stamp = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    day, hhmm = stamp[:10], stamp[11:16].replace(":", "")
    html = fetch_latest(session)
    save_raw(raw_dir, "dse_intraday", day, f"latest_{hhmm}", html, url=URL)
    rows = parse_latest(html)
    if not rows:
        return 0
    return vdb.upsert(con, "intraday_snapshots",
                      [{**r, "ts": stamp} for r in rows])
```

- [ ] **Step 5: Run tests** — 4 passed; fast suite; ruff. Commit:

```bash
git add vectora/db.py vectora/collect/dse_intraday.py tests/collect/test_dse_intraday.py tests/fixtures/latest_share_price.html
git commit -m "feat: intraday snapshot collector on DSE hourly publication points"
```

---

### Task 2: Anomaly detector + urgent tier

**Files:**
- Create: `vectora/alerts/intraday.py`
- Test: `tests/test_intraday_alerts.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_intraday_alerts.py
import polars as pl

from vectora import db as vdb
from vectora.alerts import intraday


def _seed_history(con, symbols=("NORM", "SURGE", "LIMIT"), days=30):
    import datetime as dt
    rows = []
    d0 = dt.date(2026, 6, 1)
    for i in range(days):
        d = d0 + dt.timedelta(days=i)
        for s in symbols:
            rows.append(dict(symbol=s, date=d, open=10.0, high=10.1, low=9.9,
                             close=10.0, ltp=10.0, ycp=10.0, trades=50,
                             value_mn=5.0, volume=10000, source="dse_eod"))
    df = pl.DataFrame(rows)  # noqa: F841
    con.execute("INSERT INTO prices_raw SELECT * FROM df")
    vdb.upsert(con, "symbols", [
        dict(symbol=s, name=None, sector="Bank", instrument_type="Equity",
             category="A", listing_status="active", first_seen="2020-01-01",
             last_seen="2026-12-31") for s in symbols])


def _snap(symbol, volume=10000, ltp=10.0, ycp=10.0, value_mn=5.0):
    return dict(symbol=symbol, ltp=ltp, high=ltp, low=ltp, closep=ltp,
                ycp=ycp, change=0.0, trades=100, value_mn=value_mn,
                volume=volume)


def test_volume_surge_and_limit_detected(test_db):
    _seed_history(test_db)
    snaps = [_snap("NORM"),
             _snap("SURGE", volume=40000),          # 4x median full day
             _snap("LIMIT", ltp=10.9, ycp=10.0)]    # +9%
    anomalies = intraday.detect(test_db, snaps, "2026-07-17 12:00:00")
    kinds = {(a["symbol"], a["kind"]) for a in anomalies}
    assert ("SURGE", "volume_surge") in kinds
    assert ("LIMIT", "near_circuit") in kinds
    assert not any(a["symbol"] == "NORM" for a in anomalies)


def test_illiquid_names_ignored(test_db):
    _seed_history(test_db)
    snaps = [_snap("SURGE", volume=40000, value_mn=0.05)]  # dust turnover
    assert intraday.detect(test_db, snaps, "2026-07-17 12:00:00") == []


def test_cooldown_suppresses_repeat(test_db):
    _seed_history(test_db)
    snaps = [_snap("SURGE", volume=40000)]
    first = intraday.filter_and_log(
        test_db, intraday.detect(test_db, snaps, "2026-07-17 12:00:00"),
        "2026-07-17")
    assert [a["symbol"] for a in first] == ["SURGE"]
    second = intraday.filter_and_log(
        test_db, intraday.detect(test_db, snaps, "2026-07-17 13:00:00"),
        "2026-07-17")
    assert second == []


def test_daily_email_cap(test_db):
    for i in range(3):
        vdb.upsert(test_db, "alerts_log", [dict(
            id=f"2026-07-17_intraday_email_{i}", alert_type="intraday_email",
            symbol=None, alert_date="2026-07-17", prediction_id=None)])
    assert intraday.email_allowed(test_db, "2026-07-17") is False
    assert intraday.email_allowed(test_db, "2026-07-16") is True


def test_render_body_mentions_anomalies():
    body = intraday.render(
        [{"symbol": "SURGE", "kind": "volume_surge", "ratio": 4.0,
          "detail": "40,000 vs 21d median 10,000"},
         {"symbol": "LIMIT", "kind": "near_circuit", "ratio": 0.09,
          "detail": "+9.0% vs YCP"}], "2026-07-17 12:00:00")
    assert "SURGE" in body and "volume surge" in body.lower()
    assert "LIMIT" in body and "9.0%" in body
    assert "not investment advice" in body.lower()
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/alerts/intraday.py
"""Intraday anomaly detection + urgent email tier (spec §16).

volume_surge: cumulative intraday volume already >= SURGE_X times the
symbol's trailing 21-day MEDIAN FULL-DAY volume (mid-session!), with a
turnover floor to ignore illiquid dust. near_circuit: |LTP/YCP - 1| within
a hair of the ~10% band. Cooldown: one intraday alert per symbol per
COOLDOWN_DAYS via alerts_log. Email cap: at most MAX_EMAILS_PER_DAY urgent
sends; overflow anomalies still get logged and appear in the evening digest.
"""
from datetime import date, timedelta

from vectora import db as vdb
from vectora.alerts.digest import send_or_save

SURGE_X = 3.0
MIN_VALUE_MN = 0.5
CIRCUIT_NEAR = 0.085
COOLDOWN_DAYS = 2
MAX_EMAILS_PER_DAY = 3


def detect(con, snapshots: list[dict], ts: str) -> list[dict]:
    day = ts[:10]
    baseline = dict(con.execute(
        """
        WITH recent AS (
            SELECT symbol, volume,
                   row_number() OVER (PARTITION BY symbol ORDER BY date DESC)
                   AS rn
            FROM prices WHERE date < ? AND volume IS NOT NULL
        )
        SELECT symbol, median(volume) FROM recent WHERE rn <= 21 GROUP BY symbol
        """, [day]).fetchall())
    out = []
    for s in snapshots:
        med = baseline.get(s["symbol"])
        vol, val = s.get("volume"), s.get("value_mn")
        ltp, ycp = s.get("ltp"), s.get("ycp")
        if med and vol and val and val >= MIN_VALUE_MN \
                and vol >= SURGE_X * med:
            out.append({"symbol": s["symbol"], "kind": "volume_surge",
                        "ratio": round(vol / med, 2),
                        "detail": f"{vol:,} vs 21d median {int(med):,}"})
        if ltp and ycp and ycp > 0 and abs(ltp / ycp - 1) >= CIRCUIT_NEAR:
            move = ltp / ycp - 1
            out.append({"symbol": s["symbol"], "kind": "near_circuit",
                        "ratio": round(abs(move), 4),
                        "detail": f"{move:+.1%} vs YCP"})
    return out


def filter_and_log(con, anomalies: list[dict], day: str) -> list[dict]:
    floor = (date.fromisoformat(day)
             - timedelta(days=COOLDOWN_DAYS)).isoformat()
    recent = {r[0] for r in con.execute(
        "SELECT symbol FROM alerts_log WHERE alert_type = 'intraday' "
        "AND alert_date >= ? AND alert_date <= ?", [floor, day]).fetchall()}
    fresh = [a for a in anomalies if a["symbol"] not in recent]
    seen = set()
    for a in fresh:
        if a["symbol"] in seen:
            continue
        seen.add(a["symbol"])
        vdb.upsert(con, "alerts_log", [{
            "id": f"{day}_intraday_{a['symbol']}", "alert_type": "intraday",
            "symbol": a["symbol"], "alert_date": day, "prediction_id": None}])
    return fresh


def email_allowed(con, day: str) -> bool:
    n = con.execute(
        "SELECT count(*) FROM alerts_log WHERE alert_type = 'intraday_email' "
        "AND alert_date = ?", [day]).fetchone()[0]
    return n < MAX_EMAILS_PER_DAY


def render(anomalies: list[dict], ts: str) -> str:
    lines = [f"# Vectora intraday alert {ts}", ""]
    for a in anomalies:
        label = "volume surge" if a["kind"] == "volume_surge" \
            else "near circuit"
        lines.append(f"- {a['symbol']}: {label} - {a['detail']}")
    lines += ["", "Warnings about unusual PUBLIC trading activity; "
              "not signals. _Research tool, not investment advice._", ""]
    return "\n".join(lines)


def run_intraday(con, session, ts: str | None = None) -> dict:
    from datetime import datetime

    from vectora import calendar as cal
    from vectora.collect.dse_intraday import collect_intraday, parse_latest
    from vectora.collect.dse_intraday import fetch_latest

    stamp = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    day = stamp[:10]
    if not cal.is_trading_day(date.fromisoformat(day), cal.load_holidays()):
        return {"ts": stamp, "skipped": "not a trading day"}
    html = fetch_latest(session)
    rows = parse_latest(html)
    n = collect_intraday(con, _Replay(html), ts=stamp)
    anomalies = filter_and_log(con, detect(con, rows, stamp), day)
    emailed = False
    if anomalies and email_allowed(con, day):
        subject = f"[URGENT] Vectora intraday {stamp[:16]} - " \
                  f"{len(anomalies)} anomal" \
                  f"{'y' if len(anomalies) == 1 else 'ies'}"
        send_or_save(subject, render(anomalies, stamp))
        vdb.upsert(con, "alerts_log", [{
            "id": f"{day}_intraday_email_{stamp[11:16]}",
            "alert_type": "intraday_email", "symbol": None,
            "alert_date": day, "prediction_id": None}])
        emailed = True
    return {"ts": stamp, "snapshots": n, "anomalies": len(anomalies),
            "emailed": emailed}


class _Replay:
    """Session stand-in so collect_intraday reuses the already-fetched page
    instead of hitting DSE twice per scan."""

    def __init__(self, html: str):
        self._html = html

    def get(self, url, params=None):
        return self._html
```

Note the single-fetch design: `run_intraday` fetches once, replays for storage — DSE gets one request per scan.

- [ ] **Step 4: Run tests** — 5 passed; fast suite; ruff. Commit:

```bash
git add vectora/alerts/intraday.py tests/test_intraday_alerts.py
git commit -m "feat: intraday anomaly detection with cooldowns and urgent email cap"
```

---

### Task 3: CLI + workflow + live verify + merge

- [ ] **Step 1: CLI stage** — `vectora/__main__.py`: stage choices gain `"intraday"`; add branch:

```python
    if args.command == "run" and args.stage == "intraday":
        from vectora import db as vdb
        from vectora.alerts import intraday
        from vectora.http import PoliteSession
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = intraday.run_intraday(con, PoliteSession())
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0
```

- [ ] **Step 2: Workflow** — create `.github/workflows/intraday-scan.yml`:

```yaml
name: intraday-scan

on:
  schedule:
    # DSE publishes hourly chart data 4x/day; 11:00-14:00 Dhaka = 05-08 UTC
    - cron: "0 5,6,7,8 * * 0,1,2,3,4"
  workflow_dispatch:

concurrency:
  group: pipeline-writes
  cancel-in-progress: false

permissions:
  contents: write

env:
  TZ: Asia/Dhaka

jobs:
  scan:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - name: Install
        run: uv sync --frozen
      - name: Intraday scan
        id: scan
        continue-on-error: true
        env:
          GMAIL_APP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}
        run: uv run python -m vectora run intraday
      - name: Commit data
        if: always()
        run: |
          git config user.name "vectora-bot"
          git config user.email "vectora-bot@users.noreply.github.com"
          git add -f data/raw data/vectora.duckdb reports
          git diff --cached --quiet || git commit -m "data: intraday $(date '+%F %H:%M')"
          git pull --rebase --autostash
          git push
      - name: Surface failure
        if: steps.scan.outcome == 'failure'
        run: exit 1
```

- [ ] **Step 3: Fast suite + ruff + commit; live local verify (market closed → collector stores the day's final page, detector runs, likely 0 fresh anomalies)**

```bash
uv run pytest -m "not slow" && uv run ruff check .
git add vectora/__main__.py .github/workflows/intraday-scan.yml
git commit -m "feat: intraday stage + 4x/day scan workflow on DSE publication points"
uv run python -m vectora run intraday
git add -f data/vectora.duckdb data/raw && git commit -m "data: first intraday snapshot"
```

- [ ] **Step 4: Merge, push, dispatch verification**

```bash
git checkout main && git pull
git merge --no-ff phase-4d-intraday -m "Merge phase-4d: intraday scans and urgent alert tier"
git push
& "C:\Program Files\GitHub CLI\gh.exe" workflow run intraday-scan --ref main
```

Watch to green. Tomorrow's four scheduled scans are the real soak.

---

## Execution notes

- Expected fast suite ≈ 191 tests. Thresholds (3×, 8.5%, cap 3) are Phase 5 tuning constants.
- The scan intentionally does NOT run predictions intraday — signals stay an EOD product; intraday is anomaly awareness only (spec §16 alert taxonomy).
- After 4D: Phase 5 learning loop + polish, then production-complete per roadmap.
