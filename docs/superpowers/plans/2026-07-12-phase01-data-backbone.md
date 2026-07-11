# Vectora Phase 0–1: DSE Data Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reliable, quality-scored daily EOD data pipeline for all DSE instruments — scrapers, validation, DuckDB storage, historical backfill — running unattended on GitHub Actions.

**Architecture:** Modular monolith per the spec (`docs/superpowers/specs/2026-07-12-dse-market-intelligence-design.md` §3). This plan builds: `http` (polite client) → `collect` (4 scrapers + runner) → `validate` → `db` (DuckDB) → `orchestrator` → GitHub Actions workflow. Corporate-action adjustment (`clean`), features, and models are Phase 2 — **not in this plan**.

**Tech Stack:** Python 3.12, uv, requests + BeautifulSoup4 + lxml, DuckDB, pytest + responses, ruff.

---

## Verified Ground Truth (fetched live 2026-07-12 — do not re-derive, but re-record fixtures in Task 3)

1. **TLS**: `www.dsebd.org` serves an **incomplete certificate chain**. Plain `requests.get` fails. The HTTP client uses `verify=False` with urllib3 warnings suppressed (public data, integrity risk accepted and documented).
2. **EOD archive**: `GET https://www.dsebd.org/day_end_archive.php?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&inst=All Instrument&archive=data` returns ~1.1 MB HTML. Data lives in `<table class="shares-table fixedHeader">`, header row: `# | DATE | TRADING CODE | LTP* | HIGH | LOW | OPENP* | CLOSEP* | YCP | TRADE | VALUE (mn) | VOLUME`. Symbol cell contains `<a href="displayCompany.php?name=SYMBOL">`. Volume is comma-formatted (`709,289`). ~850 rows per trading day (all instruments).
3. **News archive**: `GET https://www.dsebd.org/old_news.php?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&criteria=4&archive=news`. Each item is a small table with rows `<th>News Title:</th><td>…</td>`, `<th>News:</th><td>…</td>`, `<th>Post Date:</th><td>YYYY-MM-DD</td>`. Titles look like `MERCANBANK: Credit Rating Result` or `DSE NEWS: Daily Turnover of Main Board`.
4. **Homepage indices**: `GET https://www.dsebd.org/` has `<div class="midrow">` blocks: `m_col-1` = index name (may contain `<font>` tags: `DSE<font>X</font> Index`), `m_col-2` = value, `m_col-3` = change. Indices: DSEX, DSES, DS30, DSMEX. A later midrow pair holds Total Trade / Total Volume / Total Value (mn).
5. **Company page**: `GET https://www.dsebd.org/displayCompany.php?name=SYMBOL`. Label-keyed cells, confirmed labels include `Paid-up Capital (mn)`, `Total No. of Outstanding Securities`, `Sector`, `Market Category` (value in the **next `<td>`**), `Face/par Value`, and one-or-more `Share Holding Percentage [as on <date>]` blocks whose inner table has text like `Sponsor/Director:<br>90.00`, `Govt:…`, `Institute:…`, `Foreign:…`, `Public:…`.
6. **Trading week**: Sunday–Thursday; Friday/Saturday closed. Dhaka = UTC+6, no DST.
7. **Backfill**: Mendeley dataset `23553sm4tn` v4 (DSE EOD 2012→2026), manual one-time download from https://data.mendeley.com/datasets/23553sm4tn/4 — columns per its docs: date, open, high, low, close, volume (+ instrument identifier); loader maps headers defensively.

## File Structure (this plan creates)

```
pyproject.toml                      # deps, ruff, pytest config
.gitignore
vectora/__init__.py
vectora/__main__.py                 # CLI: python -m vectora run <stage>|backfill|bootstrap
vectora/settings.py                 # paths + constants, no logic
vectora/http.py                     # PoliteSession (delay, retries, UA, verify=False)
vectora/db.py                       # DuckDB schema, watermarks, upserts
vectora/calendar.py                 # is_trading_day()
vectora/collect/__init__.py
vectora/collect/dse_eod.py          # day_end_archive scraper+parser
vectora/collect/dse_news.py         # old_news scraper+parser
vectora/collect/dse_indices.py      # homepage index parser
vectora/collect/dse_company.py      # displayCompany parser
vectora/collect/raw_store.py        # raw payload + meta.json writer
vectora/collect/runner.py           # collect stage: fetch→raw→parse→DB
vectora/validate/__init__.py
vectora/validate/checks.py          # validation gates → data_quality score
vectora/orchestrator.py             # stage registry + run_pipeline()
tools/record_fixtures.py            # fetch live pages into tests/fixtures/
tools/backfill_mendeley.py          # load historical zip into prices_raw
tools/bootstrap_reference.py        # symbols sweep + holidays seed
data/reference/holidays.csv
tests/  (mirrors package; fixtures/ holds recorded HTML)
.github/workflows/eod-pipeline.yml
```

Conventions for every task: run commands from repo root; `uv run pytest …`; dates are ISO strings; DB path comes from `settings.DB_PATH` and every test uses a `tmp_path` DB via fixture.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `vectora/__init__.py`, `vectora/settings.py`, `tests/conftest.py`, empty package dirs

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "vectora"
version = "0.1.0"
description = "DSE market intelligence system (research tool, not investment advice)"
requires-python = ">=3.12"
dependencies = [
    "requests>=2.32",
    "beautifulsoup4>=4.12",
    "lxml>=5.2",
    "duckdb>=1.0",
]

[dependency-groups]
dev = ["pytest>=8.0", "responses>=0.25", "ruff>=0.5"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
data/raw/          # raw payloads committed ONLY by CI, not dev machines
*.duckdb.wal
```

Note: `data/vectora.duckdb` and `data/reference/` ARE tracked. `data/raw/` is ignored locally; the CI workflow force-adds it (Task 12).

- [ ] **Step 3: Write `vectora/settings.py`**

```python
"""Central paths and constants. No logic here."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
DB_PATH = DATA_DIR / "vectora.duckdb"
HOLIDAYS_CSV = REFERENCE_DIR / "holidays.csv"

DSE_BASE = "https://www.dsebd.org"
USER_AGENT = "VectoraResearch/0.1 (personal academic research)"
REQUEST_DELAY_S = 1.5
REQUEST_TIMEOUT_S = 90
MAX_RETRIES = 3

# Data-quality alert floor (spec §5.3)
MIN_QUALITY_SCORE = 80
```

- [ ] **Step 4: Create package skeleton and test conftest**

Create empty `vectora/__init__.py`, `vectora/collect/__init__.py`, `vectora/validate/__init__.py`, `tests/__init__.py` (empty files), and `tests/conftest.py`:

```python
import pytest

from vectora import db as vdb


@pytest.fixture()
def test_db(tmp_path):
    """Fresh DuckDB with full schema, isolated per test."""
    path = tmp_path / "test.duckdb"
    con = vdb.connect(path)
    vdb.init_schema(con)
    yield con
    con.close()


FIXTURES = None  # set in step below


@pytest.fixture()
def fixtures_dir():
    from pathlib import Path
    return Path(__file__).parent / "fixtures"
```

(`test_db` will fail to import until Task 2 creates `vectora/db.py` — that is expected; nothing uses it yet.)

- [ ] **Step 5: Verify environment**

Run: `uv sync && uv run python -c "import vectora, requests, bs4, duckdb; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore vectora tests
git commit -m "chore: project scaffold (uv, pytest, ruff, settings)"
```

---

### Task 2: DuckDB layer (`vectora/db.py`)

**Files:**
- Create: `vectora/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_db.py
from vectora import db as vdb


def test_schema_creates_all_tables(test_db):
    tables = {r[0] for r in test_db.execute("SHOW TABLES").fetchall()}
    assert {
        "symbols", "prices_raw", "indices", "events", "company_snapshot",
        "holdings", "data_quality", "watermarks", "no_trade_days",
    } <= tables


def test_watermark_roundtrip(test_db):
    assert vdb.get_watermark(test_db, "collect", "eod") is None
    vdb.set_watermark(test_db, "collect", "eod", "2026-07-10")
    assert vdb.get_watermark(test_db, "collect", "eod") == "2026-07-10"
    vdb.set_watermark(test_db, "collect", "eod", "2026-07-12")  # overwrite
    assert vdb.get_watermark(test_db, "collect", "eod") == "2026-07-12"


def test_upsert_prices_is_idempotent(test_db):
    row = dict(symbol="GP", date="2026-07-09", open=280.0, high=285.0, low=279.0,
               close=284.1, ltp=284.0, ycp=280.5, trades=1500, value_mn=120.5,
               volume=425000, source="dse_eod")
    vdb.upsert(test_db, "prices_raw", [row, row])
    vdb.upsert(test_db, "prices_raw", [row])
    n = test_db.execute("SELECT count(*) FROM prices_raw").fetchone()[0]
    assert n == 1


def test_upsert_replaces_on_conflict(test_db):
    r1 = dict(symbol="GP", date="2026-07-09", open=1.0, high=1.0, low=1.0, close=1.0,
              ltp=1.0, ycp=1.0, trades=1, value_mn=1.0, volume=1, source="dse_eod")
    r2 = {**r1, "close": 2.0}
    vdb.upsert(test_db, "prices_raw", [r1])
    vdb.upsert(test_db, "prices_raw", [r2])
    close = test_db.execute("SELECT close FROM prices_raw").fetchone()[0]
    assert close == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL / ERROR — `vectora.db` has no `connect` (module missing).

- [ ] **Step 3: Implement `vectora/db.py`**

```python
"""DuckDB access layer: schema, watermarks, generic upsert.

Rows are never deleted (spec: old knowledge never disappears);
conflicting rows are replaced via primary keys, superseded raw data
stays in data/raw/.
"""
from pathlib import Path

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT PRIMARY KEY, name TEXT, sector TEXT, instrument_type TEXT,
    category TEXT, listing_status TEXT DEFAULT 'active', first_seen DATE, last_seen DATE
);
CREATE TABLE IF NOT EXISTS prices_raw (
    symbol TEXT, date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    ltp DOUBLE, ycp DOUBLE, trades BIGINT, value_mn DOUBLE, volume BIGINT,
    source TEXT, PRIMARY KEY (symbol, date, source)
);
CREATE TABLE IF NOT EXISTS indices (
    index_name TEXT, date DATE, value DOUBLE, change DOUBLE,
    PRIMARY KEY (index_name, date)
);
CREATE TABLE IF NOT EXISTS market_totals (
    date DATE PRIMARY KEY, total_trades BIGINT, total_volume BIGINT, total_value_mn DOUBLE
);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,             -- sha256 of (title, post_date, body)
    post_date DATE, symbol TEXT, title TEXT, body TEXT, source TEXT,
    scraped_at TIMESTAMP DEFAULT current_timestamp
);
CREATE TABLE IF NOT EXISTS company_snapshot (
    symbol TEXT, as_of DATE, sector TEXT, category TEXT, instrument_type TEXT,
    paid_up_capital_mn DOUBLE, outstanding_shares BIGINT, face_value DOUBLE,
    market_lot INTEGER, PRIMARY KEY (symbol, as_of)
);
CREATE TABLE IF NOT EXISTS holdings (
    symbol TEXT, as_of DATE, sponsor_pct DOUBLE, govt_pct DOUBLE,
    institute_pct DOUBLE, foreign_pct DOUBLE, public_pct DOUBLE,
    PRIMARY KEY (symbol, as_of)
);
CREATE TABLE IF NOT EXISTS data_quality (
    date DATE, source TEXT, score INTEGER, issues TEXT,   -- issues: JSON list
    PRIMARY KEY (date, source)
);
CREATE TABLE IF NOT EXISTS no_trade_days (
    date DATE PRIMARY KEY, reason TEXT
);
CREATE TABLE IF NOT EXISTS watermarks (
    stage TEXT, key TEXT, value TEXT, updated_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (stage, key)
);
"""


def connect(path: str | Path) -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA)


def get_watermark(con, stage: str, key: str) -> str | None:
    row = con.execute(
        "SELECT value FROM watermarks WHERE stage = ? AND key = ?", [stage, key]
    ).fetchone()
    return row[0] if row else None


def set_watermark(con, stage: str, key: str, value: str) -> None:
    con.execute(
        "INSERT OR REPLACE INTO watermarks (stage, key, value) VALUES (?, ?, ?)",
        [stage, key, value],
    )


def upsert(con, table: str, rows: list[dict]) -> int:
    """INSERT OR REPLACE a list of same-shaped dicts. Returns row count."""
    if not rows:
        return 0
    # de-duplicate within the batch (last wins) to avoid PK clash inside one insert
    cols = list(rows[0].keys())
    seen: dict = {}
    for r in rows:
        seen[tuple(str(r[c]) for c in cols)] = r  # exact-duplicate collapse
    rows = list(seen.values())
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    con.executemany(sql, [[r[c] for c in cols] for r in rows])
    return len(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v`
Expected: 4 passed.

Note: if `test_upsert_prices_is_idempotent` fails on the duplicate-in-batch case with a PK error, the de-dup collapse above is the fix — exact duplicates collapse, and true conflicts (same PK, different values) resolve by `INSERT OR REPLACE` across calls.

- [ ] **Step 5: Commit**

```bash
git add vectora/db.py tests/test_db.py
git commit -m "feat: DuckDB layer with schema, watermarks, idempotent upsert"
```

---

### Task 3: HTTP client + fixture recorder

**Files:**
- Create: `vectora/http.py`, `tools/record_fixtures.py`
- Test: `tests/test_http.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_http.py
import responses

from vectora.http import PoliteSession


@responses.activate
def test_get_returns_text_and_sends_user_agent():
    responses.get("https://www.dsebd.org/test.php", body="<html>hi</html>")
    s = PoliteSession(delay_s=0)
    text = s.get("https://www.dsebd.org/test.php")
    assert text == "<html>hi</html>"
    assert "VectoraResearch" in responses.calls[0].request.headers["User-Agent"]


@responses.activate
def test_get_retries_on_5xx_then_succeeds():
    responses.get("https://www.dsebd.org/flaky.php", status=503)
    responses.get("https://www.dsebd.org/flaky.php", body="ok")
    s = PoliteSession(delay_s=0, backoff_s=0)
    assert s.get("https://www.dsebd.org/flaky.php") == "ok"
    assert len(responses.calls) == 2


@responses.activate
def test_get_raises_after_max_retries():
    import pytest
    import requests

    for _ in range(3):
        responses.get("https://www.dsebd.org/dead.php", status=500)
    s = PoliteSession(delay_s=0, backoff_s=0, max_retries=3)
    with pytest.raises(requests.HTTPError):
        s.get("https://www.dsebd.org/dead.php")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_http.py -v`
Expected: FAIL — `vectora.http` missing.

- [ ] **Step 3: Implement `vectora/http.py`**

```python
"""Polite HTTP client for dsebd.org.

verify=False is deliberate: dsebd.org serves an incomplete certificate
chain (verified 2026-07-12). Data is public; integrity risk is accepted
and the raw layer keeps checksummed copies of everything fetched.
"""
import time

import requests
import urllib3

from vectora import settings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PoliteSession:
    def __init__(
        self,
        delay_s: float = settings.REQUEST_DELAY_S,
        timeout_s: int = settings.REQUEST_TIMEOUT_S,
        max_retries: int = settings.MAX_RETRIES,
        backoff_s: float = 5.0,
    ):
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._last_request_ts = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = settings.USER_AGENT
        self._session.verify = False

    def get(self, url: str, params: dict | None = None) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout_s)
                resp.raise_for_status()
                return resp.text
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                time.sleep(self.backoff_s * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.delay_s:
            time.sleep(self.delay_s - elapsed)
        self._last_request_ts = time.monotonic()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_http.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write `tools/record_fixtures.py`**

```python
"""Record live dsebd.org pages as test fixtures. Run manually, rarely.

Usage: uv run python tools/record_fixtures.py [YYYY-MM-DD]
Dates default to the most recent Sun-Thu weekday before today.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

from vectora.http import PoliteSession
from vectora.settings import DSE_BASE

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def last_trading_weekday(today: date) -> date:
    d = today - timedelta(days=1)
    while d.weekday() in (4, 5):  # Fri=4, Sat=5 closed
        d -= timedelta(days=1)
    return d


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else last_trading_weekday(date.today())
    ds = d.isoformat()
    s = PoliteSession()
    pages = {
        "day_end_archive.html": (
            f"{DSE_BASE}/day_end_archive.php",
            {"startDate": ds, "endDate": ds, "inst": "All Instrument", "archive": "data"},
        ),
        "old_news.html": (
            f"{DSE_BASE}/old_news.php",
            {"startDate": ds, "endDate": ds, "criteria": "4", "archive": "news"},
        ),
        "homepage.html": (f"{DSE_BASE}/", None),
        "company_GP.html": (f"{DSE_BASE}/displayCompany.php", {"name": "GP"}),
    }
    for fname, (url, params) in pages.items():
        html = s.get(url, params=params)
        (FIXTURES / fname).write_text(html, encoding="utf-8")
        print(f"{fname}: {len(html):,} bytes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Record the fixtures (live network)**

Run: `uv run python tools/record_fixtures.py`
Expected output: four lines; `day_end_archive.html` ≈ 1,000,000+ bytes, `old_news.html` ≈ 100,000+ bytes, `homepage.html` ≈ 400,000 bytes, `company_GP.html` ≈ 330,000 bytes. If any file is < 50 KB, open it — you likely got an error page; investigate before proceeding.

- [ ] **Step 7: Commit**

```bash
git add vectora/http.py tools/record_fixtures.py tests/test_http.py tests/fixtures
git commit -m "feat: polite HTTP client + recorded live DSE fixtures"
```

---

### Task 4: EOD scraper (`vectora/collect/dse_eod.py`)

**Files:**
- Create: `vectora/collect/dse_eod.py`
- Test: `tests/collect/test_dse_eod.py` (create `tests/collect/__init__.py` too)

- [ ] **Step 1: Write the failing tests**

```python
# tests/collect/test_dse_eod.py
from vectora.collect import dse_eod


def _rows(fixtures_dir):
    html = (fixtures_dir / "day_end_archive.html").read_text(encoding="utf-8")
    return dse_eod.parse_day_end(html)


def test_parses_many_rows(fixtures_dir):
    rows = _rows(fixtures_dir)
    assert len(rows) > 300  # all instruments incl. funds/bonds ~850


def test_row_shape_and_types(fixtures_dir):
    r = _rows(fixtures_dir)[0]
    assert set(r) == {"symbol", "date", "ltp", "high", "low", "open", "close",
                      "ycp", "trades", "value_mn", "volume"}
    assert isinstance(r["symbol"], str) and r["symbol"] == r["symbol"].strip()
    assert len(r["date"]) == 10 and r["date"][4] == "-"
    assert isinstance(r["volume"], int)          # comma-formatted in HTML
    assert isinstance(r["trades"], int)
    assert r["high"] is None or r["high"] >= 0


def test_all_symbols_unique_per_date(fixtures_dir):
    rows = _rows(fixtures_dir)
    keys = [(r["symbol"], r["date"]) for r in rows]
    assert len(keys) == len(set(keys))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/collect/test_dse_eod.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `vectora/collect/dse_eod.py`**

```python
"""Day-end archive scraper.

Endpoint (verified 2026-07-12):
  /day_end_archive.php?startDate=&endDate=&inst=All Instrument&archive=data
Table: <table class="shares-table fixedHeader">
Columns: # | DATE | TRADING CODE | LTP | HIGH | LOW | OPENP | CLOSEP | YCP
         | TRADE | VALUE (mn) | VOLUME
"""
from bs4 import BeautifulSoup

from vectora.http import PoliteSession
from vectora.settings import DSE_BASE

URL = f"{DSE_BASE}/day_end_archive.php"


def fetch_day_end(session: PoliteSession, start: str, end: str) -> str:
    return session.get(URL, params={
        "startDate": start, "endDate": end,
        "inst": "All Instrument", "archive": "data",
    })


def _num(text: str) -> float | None:
    t = text.strip().replace(",", "")
    if t in ("", "-", "--", "0.00-"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _int(text: str) -> int:
    v = _num(text)
    return int(v) if v is not None else 0


def parse_day_end(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="shares-table")
    if table is None:
        return []  # holiday / empty range renders no table
    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != 12:
            continue
        cells = [td.get_text(strip=True) for td in tds]
        rows.append({
            "symbol": cells[2],
            "date": cells[1],
            "ltp": _num(cells[3]),
            "high": _num(cells[4]),
            "low": _num(cells[5]),
            "open": _num(cells[6]),
            "close": _num(cells[7]),
            "ycp": _num(cells[8]),
            "trades": _int(cells[9]),
            "value_mn": _num(cells[10]),
            "volume": _int(cells[11]),
        })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/collect/test_dse_eod.py -v`
Expected: 3 passed. If `test_parses_many_rows` finds 0 rows, your fixture was recorded on a holiday — re-record with an explicit recent trading date: `uv run python tools/record_fixtures.py 2026-07-09`.

- [ ] **Step 5: Commit**

```bash
git add vectora/collect tests/collect
git commit -m "feat: EOD day-end archive scraper and parser"
```

---

### Task 5: News scraper (`vectora/collect/dse_news.py`)

**Files:**
- Create: `vectora/collect/dse_news.py`
- Test: `tests/collect/test_dse_news.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/collect/test_dse_news.py
from vectora.collect import dse_news


def _items(fixtures_dir):
    html = (fixtures_dir / "old_news.html").read_text(encoding="utf-8")
    return dse_news.parse_news(html)


def test_parses_items(fixtures_dir):
    items = _items(fixtures_dir)
    assert len(items) >= 5


def test_item_shape(fixtures_dir):
    it = _items(fixtures_dir)[0]
    assert set(it) == {"id", "symbol", "title", "body", "post_date"}
    assert len(it["id"]) == 64  # sha256 hex
    assert len(it["post_date"]) == 10


def test_symbol_extraction():
    assert dse_news.extract_symbol("MERCANBANK: Credit Rating Result") == "MERCANBANK"
    assert dse_news.extract_symbol("DSE NEWS: Daily Turnover of Main Board") is None
    assert dse_news.extract_symbol("No colon here") is None


def test_ids_are_stable_and_unique(fixtures_dir):
    items = _items(fixtures_dir)
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    assert dse_news.parse_news(
        (fixtures_dir / "old_news.html").read_text(encoding="utf-8")
    )[0]["id"] == items[0]["id"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/collect/test_dse_news.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `vectora/collect/dse_news.py`**

```python
"""DSE news/announcement archive scraper.

Endpoint (verified 2026-07-12):
  /old_news.php?startDate=&endDate=&criteria=4&archive=news
Each item is a small table containing rows labeled
  'News Title:' / 'News:' / 'Post Date:'.
Titles are 'SYMBOL: Subject' for company news, 'DSE NEWS: ...' otherwise.
"""
import hashlib
import re

from bs4 import BeautifulSoup

from vectora.http import PoliteSession
from vectora.settings import DSE_BASE

URL = f"{DSE_BASE}/old_news.php"
_SYMBOL_RE = re.compile(r"^([A-Z0-9]{2,20}):")
_NON_COMPANY = {"DSE", "DSENEWS", "BSEC"}


def fetch_news(session: PoliteSession, start: str, end: str) -> str:
    return session.get(URL, params={
        "startDate": start, "endDate": end, "criteria": "4", "archive": "news",
    })


def extract_symbol(title: str) -> str | None:
    m = _SYMBOL_RE.match(title.strip())
    if not m:
        return None
    sym = m.group(1)
    if sym in _NON_COMPANY or title.strip().startswith("DSE NEWS"):
        return None
    return sym


def parse_news(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    items = []
    for th in soup.find_all("th", string=re.compile(r"News Title:")):
        table = th.find_parent("table")
        if table is None:
            continue
        fields: dict[str, str] = {}
        for row in table.find_all("tr"):
            h, d = row.find("th"), row.find("td")
            if h and d:
                fields[h.get_text(strip=True).rstrip(":")] = d.get_text(" ", strip=True)
        title = fields.get("News Title", "")
        body = fields.get("News", "")
        post_date = fields.get("Post Date", "")
        if not title or not post_date:
            continue
        digest = hashlib.sha256(f"{title}|{post_date}|{body}".encode()).hexdigest()
        items.append({
            "id": digest,
            "symbol": extract_symbol(title),
            "title": title,
            "body": body,
            "post_date": post_date,
        })
    # de-dup within page (same item can repeat across pagination fringes)
    return list({i["id"]: i for i in items}.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/collect/test_dse_news.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add vectora/collect/dse_news.py tests/collect/test_dse_news.py
git commit -m "feat: DSE news archive scraper with stable event ids"
```

---

### Task 6: Indices parser (`vectora/collect/dse_indices.py`)

**Files:**
- Create: `vectora/collect/dse_indices.py`
- Test: `tests/collect/test_dse_indices.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/collect/test_dse_indices.py
from vectora.collect import dse_indices


def test_parses_main_indices(fixtures_dir):
    html = (fixtures_dir / "homepage.html").read_text(encoding="utf-8")
    result = dse_indices.parse_homepage(html)
    names = {i["index_name"] for i in result["indices"]}
    assert {"DSEX", "DSES", "DS30"} <= names
    dsex = next(i for i in result["indices"] if i["index_name"] == "DSEX")
    assert dsex["value"] > 1000  # DSEX has been > 3000 since 2013
    assert isinstance(dsex["change"], float)


def test_parses_market_totals(fixtures_dir):
    html = (fixtures_dir / "homepage.html").read_text(encoding="utf-8")
    totals = dse_indices.parse_homepage(html)["totals"]
    assert totals["total_trades"] > 0
    assert totals["total_volume"] > totals["total_trades"]
    assert totals["total_value_mn"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/collect/test_dse_indices.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `vectora/collect/dse_indices.py`**

```python
"""Parse current index values + market totals from the DSE homepage.

Markup (verified 2026-07-12): <div class="midrow"> blocks with
  m_col-1 = name (may contain <font> tags: 'DSE<font>X</font> Index')
  m_col-2 = value, m_col-3 = signed change.
A later midrow pair holds Total Trade / Total Volume / Total Value (mn):
header divs (m_col-wid*) followed by a values midrow in the same order.
"""
import re

from bs4 import BeautifulSoup

from vectora.http import PoliteSession
from vectora.settings import DSE_BASE

_KNOWN = {"DSEX", "DSES", "DS30", "DSMEX"}


def fetch_homepage(session: PoliteSession) -> str:
    return session.get(f"{DSE_BASE}/")


def _clean_name(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("Index", "")).upper()


def _f(text: str) -> float | None:
    t = text.strip().replace(",", "")
    try:
        return float(t)
    except ValueError:
        return None


def parse_homepage(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    indices: list[dict] = []
    numeric_wid_rows: list[list[float]] = []
    for row in soup.find_all("div", class_="midrow"):
        c1 = row.find("div", class_="m_col-1")
        c2 = row.find("div", class_="m_col-2")
        c3 = row.find("div", class_="m_col-3")
        if c1 and c2:
            name = _clean_name(c1.get_text(strip=True))
            if name in _KNOWN and _f(c2.get_text()) is not None:
                indices.append({
                    "index_name": name,
                    "value": _f(c2.get_text()),
                    "change": _f(c3.get_text()) if c3 else None,
                })
            continue
        # totals live in m_col-wid / m_col-wid1 / m_col-wid2 divs
        wids = row.find_all("div", class_=re.compile(r"^m_col-wid"))
        vals = [_f(d.get_text()) for d in wids]
        if len(vals) >= 3 and all(v is not None for v in vals[:3]):
            numeric_wid_rows.append(vals)  # first all-numeric row = totals
    totals = {}
    if numeric_wid_rows:
        t = numeric_wid_rows[0]
        totals = {
            "total_trades": int(t[0]),
            "total_volume": int(t[1]),
            "total_value_mn": float(t[2]),
        }
    return {"indices": indices, "totals": totals}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/collect/test_dse_indices.py -v`
Expected: 2 passed. If totals parsing finds the wrong row, print all `numeric_wid_rows` in a debug run — the totals row is the one whose first value equals the day's Total Trade shown on dsebd.org; adjust the selection index accordingly and re-run.

- [ ] **Step 5: Commit**

```bash
git add vectora/collect/dse_indices.py tests/collect/test_dse_indices.py
git commit -m "feat: homepage index and market-totals parser"
```

---

### Task 7: Company page parser (`vectora/collect/dse_company.py`)

**Files:**
- Create: `vectora/collect/dse_company.py`
- Test: `tests/collect/test_dse_company.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/collect/test_dse_company.py
from vectora.collect import dse_company


def _parsed(fixtures_dir):
    html = (fixtures_dir / "company_GP.html").read_text(encoding="utf-8")
    return dse_company.parse_company(html, symbol="GP")


def test_core_fields(fixtures_dir):
    c = _parsed(fixtures_dir)["snapshot"]
    assert c["symbol"] == "GP"
    assert c["category"] in {"A", "B", "N", "Z", "G"}
    assert isinstance(c["sector"], str) and len(c["sector"]) > 2
    assert c["paid_up_capital_mn"] > 0
    assert c["outstanding_shares"] > 1_000_000
    assert c["face_value"] > 0


def test_holdings_blocks(fixtures_dir):
    blocks = _parsed(fixtures_dir)["holdings"]
    assert len(blocks) >= 1
    latest = blocks[-1]
    assert set(latest) == {"as_of", "sponsor_pct", "govt_pct", "institute_pct",
                           "foreign_pct", "public_pct"}
    total = sum(v for k, v in latest.items() if k != "as_of")
    assert 99.0 <= total <= 101.0  # percentages sum to ~100
    assert latest["as_of"].count("-") == 2  # ISO date
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/collect/test_dse_company.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `vectora/collect/dse_company.py`**

```python
"""displayCompany.php parser: static company facts + shareholding blocks.

Verified labels (2026-07-12): 'Paid-up Capital (mn)', 'Total No. of
Outstanding Securities', 'Sector', 'Market Category' (label is a <td>,
value in next <td>), 'Face/par Value', 'Market Lot', 'Type of Instrument'.
Holdings: one or more cells 'Share Holding Percentage [as on <date>]'
followed by a table with 'Sponsor/Director:<br>90.00' style cells.
"""
import re
from datetime import datetime

from bs4 import BeautifulSoup

from vectora.http import PoliteSession
from vectora.settings import DSE_BASE

_HOLD_KEYS = {
    "Sponsor/Director": "sponsor_pct", "Govt": "govt_pct",
    "Institute": "institute_pct", "Foreign": "foreign_pct", "Public": "public_pct",
}


def fetch_company(session: PoliteSession, symbol: str) -> str:
    return session.get(f"{DSE_BASE}/displayCompany.php", params={"name": symbol})


def _label_value(soup: BeautifulSoup, label: str) -> str | None:
    """Value of the cell immediately following a th/td whose text == label."""
    for cell in soup.find_all(["th", "td"]):
        if cell.get_text(strip=True).rstrip(":").strip() == label:
            nxt = cell.find_next_sibling("td")
            if nxt:
                return nxt.get_text(" ", strip=True)
    return None


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    m = re.search(r"-?[\d,]+(?:\.\d+)?", text)
    return float(m.group(0).replace(",", "")) if m else None


def _parse_as_of(header_text: str) -> str | None:
    m = re.search(r"as on\s+([A-Za-z]{3})\s+(\d{1,2}),\s*(\d{4})", header_text)
    if not m:
        return None
    dt = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y")
    return dt.date().isoformat()


def parse_company(html: str, symbol: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    out = _num(_label_value(soup, "Total No. of Outstanding Securities"))
    snapshot = {
        "symbol": symbol,
        "sector": _label_value(soup, "Sector"),
        "category": (_label_value(soup, "Market Category") or "").strip()[:1] or None,
        "instrument_type": _label_value(soup, "Type of Instrument"),
        "paid_up_capital_mn": _num(_label_value(soup, "Paid-up Capital (mn)")),
        "outstanding_shares": int(out) if out else None,
        "face_value": _num(_label_value(soup, "Face/par Value")),
        "market_lot": int(_num(_label_value(soup, "Market Lot")) or 0) or None,
    }
    holdings: list[dict] = []
    for cell in soup.find_all("td", string=re.compile(r"Share Holding Percentage")):
        as_of = _parse_as_of(cell.get_text(" ", strip=True))
        block_table = cell.find_next("table")
        if not as_of or block_table is None:
            continue
        text = block_table.get_text(" ", strip=True)
        entry: dict = {"as_of": as_of}
        for label, key in _HOLD_KEYS.items():
            m = re.search(rf"{re.escape(label)}:\s*([\d.]+)", text)
            entry[key] = float(m.group(1)) if m else None
        if all(entry[k] is not None for k in _HOLD_KEYS.values()):
            holdings.append(entry)
    holdings.sort(key=lambda h: h["as_of"])
    return {"snapshot": snapshot, "holdings": holdings}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/collect/test_dse_company.py -v`
Expected: 2 passed. Likely failure mode: the `Share Holding Percentage` label sits inside a `<td>` whose text includes nested markup so `string=` regex misses it — if `holdings` comes back empty, switch the find to `soup.find_all("td")` filtered with `"Share Holding Percentage" in td.get_text()` and take only tds with no td children (leaf cells).

- [ ] **Step 5: Commit**

```bash
git add vectora/collect/dse_company.py tests/collect/test_dse_company.py
git commit -m "feat: company page parser (facts + shareholding blocks)"
```

---

### Task 8: Trading calendar (`vectora/calendar.py`)

**Files:**
- Create: `vectora/calendar.py`, `data/reference/holidays.csv`
- Test: `tests/test_calendar.py`

- [ ] **Step 1: Seed the holidays file**

Create `data/reference/holidays.csv` with header + known upcoming closures. Populate from https://www.dsebd.org/hts.php (open it in a browser; add each listed holiday). Keep at minimum:

```csv
date,description
2026-08-15,National Mourning Day (verify against dsebd.org/hts.php)
2026-12-16,Victory Day
2026-12-25,Christmas Day
```

**Important:** the exact 2026 list (Eid days, etc.) must be copied from the DSE holidays page during execution — the three rows above are structural seeds; verify and extend them. Missing holidays are safe: the pipeline records an empty scrape as a `no_trade_days` row instead of failing (Task 9).

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_calendar.py
from datetime import date

from vectora import calendar


def test_weekend_rule():
    assert not calendar.is_trading_day(date(2026, 7, 10))  # Friday
    assert not calendar.is_trading_day(date(2026, 7, 11))  # Saturday
    assert calendar.is_trading_day(date(2026, 7, 12))      # Sunday — DSE trades
    assert calendar.is_trading_day(date(2026, 7, 9))       # Thursday


def test_holiday_rule(tmp_path):
    csv = tmp_path / "holidays.csv"
    csv.write_text("date,description\n2026-12-16,Victory Day\n")
    assert not calendar.is_trading_day(date(2026, 12, 16), holidays_csv=csv)
    assert calendar.is_trading_day(date(2026, 12, 17), holidays_csv=csv)


def test_previous_trading_day():
    # Sunday 2026-07-12 -> previous trading day is Thursday 2026-07-09
    assert calendar.previous_trading_day(date(2026, 7, 12)) == date(2026, 7, 9)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_calendar.py -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement `vectora/calendar.py`**

```python
"""DSE trading calendar: Sun-Thu, minus holidays.csv."""
import csv
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from vectora.settings import HOLIDAYS_CSV

_WEEKEND = (4, 5)  # Friday, Saturday


@lru_cache(maxsize=4)
def _holidays(path: Path) -> frozenset[date]:
    if not path.exists():
        return frozenset()
    with open(path, newline="", encoding="utf-8") as f:
        return frozenset(date.fromisoformat(r["date"]) for r in csv.DictReader(f))


def is_trading_day(d: date, holidays_csv: Path = HOLIDAYS_CSV) -> bool:
    return d.weekday() not in _WEEKEND and d not in _holidays(holidays_csv)


def previous_trading_day(d: date, holidays_csv: Path = HOLIDAYS_CSV) -> date:
    cur = d - timedelta(days=1)
    while not is_trading_day(cur, holidays_csv):
        cur -= timedelta(days=1)
    return cur
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_calendar.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add vectora/calendar.py tests/test_calendar.py data/reference/holidays.csv
git commit -m "feat: DSE trading calendar (Sun-Thu week + holidays file)"
```

---

### Task 9: Raw store + collect runner (`vectora/collect/raw_store.py`, `runner.py`)

**Files:**
- Create: `vectora/collect/raw_store.py`, `vectora/collect/runner.py`
- Test: `tests/collect/test_runner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/collect/test_runner.py
import json
from datetime import date

from vectora.collect import runner
from vectora.collect.raw_store import save_raw


def test_save_raw_writes_payload_and_meta(tmp_path):
    p = save_raw(tmp_path, "dse_eod", date(2026, 7, 9), "page.html",
                 "<html>x</html>", url="https://example/x")
    assert p.read_text(encoding="utf-8") == "<html>x</html>"
    meta = json.loads((p.parent / "page.html.meta.json").read_text())
    assert meta["url"] == "https://example/x"
    assert meta["sha256"] and meta["bytes"] == len("<html>x</html>")


class FakeFetchers:
    """Stands in for the network: returns fixture HTML."""
    def __init__(self, fixtures_dir):
        self.f = fixtures_dir

    def eod(self, start, end):
        return (self.f / "day_end_archive.html").read_text(encoding="utf-8")

    def news(self, start, end):
        return (self.f / "old_news.html").read_text(encoding="utf-8")

    def homepage(self):
        return (self.f / "homepage.html").read_text(encoding="utf-8")


def test_collect_eod_populates_db(test_db, tmp_path, fixtures_dir):
    n = runner.collect_eod(test_db, run_date=date(2026, 7, 9),
                           fetchers=FakeFetchers(fixtures_dir), raw_root=tmp_path)
    prices = test_db.execute("SELECT count(*) FROM prices_raw").fetchone()[0]
    events = test_db.execute("SELECT count(*) FROM events").fetchone()[0]
    idx = test_db.execute("SELECT count(*) FROM indices").fetchone()[0]
    assert n["prices"] == prices and prices > 300
    assert events >= 5 and idx >= 3
    from vectora import db as vdb
    assert vdb.get_watermark(test_db, "collect", "eod") is not None


def test_collect_is_idempotent(test_db, tmp_path, fixtures_dir):
    fk = FakeFetchers(fixtures_dir)
    runner.collect_eod(test_db, date(2026, 7, 9), fk, tmp_path)
    runner.collect_eod(test_db, date(2026, 7, 9), fk, tmp_path)
    assert test_db.execute(
        "SELECT count(*) FROM (SELECT symbol, date, source FROM prices_raw "
        "GROUP BY 1,2,3 HAVING count(*) > 1)"
    ).fetchone()[0] == 0


def test_empty_scrape_records_no_trade_day(test_db, tmp_path, fixtures_dir):
    class HolidayFetchers(FakeFetchers):
        def eod(self, start, end):
            return "<html><body>no table</body></html>"
    runner.collect_eod(test_db, date(2026, 7, 9), HolidayFetchers(fixtures_dir), tmp_path)
    row = test_db.execute("SELECT reason FROM no_trade_days").fetchone()
    assert row is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/collect/test_runner.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Implement `vectora/collect/raw_store.py`**

```python
"""Immutable raw payload layer: data/raw/<source>/<date>/<name> + meta.json."""
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path


def save_raw(root: Path, source: str, run_date: date, name: str,
             payload: str, url: str) -> Path:
    d = root / source / run_date.isoformat()
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(payload, encoding="utf-8")
    meta = {
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    (d / f"{name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return p
```

- [ ] **Step 4: Implement `vectora/collect/runner.py`**

```python
"""Collect stage: fetch -> raw layer -> parse -> DB -> watermark."""
import logging
from datetime import date

from vectora import db as vdb
from vectora.collect import dse_eod, dse_indices, dse_news
from vectora.collect.raw_store import save_raw
from vectora.http import PoliteSession
from vectora.settings import RAW_DIR

log = logging.getLogger(__name__)


class LiveFetchers:
    def __init__(self):
        self.session = PoliteSession()

    def eod(self, start: str, end: str) -> str:
        return dse_eod.fetch_day_end(self.session, start, end)

    def news(self, start: str, end: str) -> str:
        return dse_news.fetch_news(self.session, start, end)

    def homepage(self) -> str:
        return dse_indices.fetch_homepage(self.session)


def collect_eod(con, run_date: date, fetchers=None, raw_root=RAW_DIR) -> dict:
    """Collect EOD prices, announcements, and index values for run_date."""
    fetchers = fetchers or LiveFetchers()
    ds = run_date.isoformat()
    counts = {"prices": 0, "events": 0, "indices": 0}

    eod_html = fetchers.eod(ds, ds)
    save_raw(raw_root, "dse_eod", run_date, "day_end_archive.html", eod_html,
             url=dse_eod.URL)
    price_rows = dse_eod.parse_day_end(eod_html)
    if not price_rows:
        vdb.upsert(con, "no_trade_days",
                   [{"date": ds, "reason": "empty day_end_archive"}])
        log.info("no trade data for %s — recorded as no-trade day", ds)
    else:
        for r in price_rows:
            r["source"] = "dse_eod"
        counts["prices"] = vdb.upsert(con, "prices_raw", price_rows)

    news_html = fetchers.news(ds, ds)
    save_raw(raw_root, "dse_news", run_date, "old_news.html", news_html,
             url=dse_news.URL)
    items = dse_news.parse_news(news_html)
    for it in items:
        it["source"] = "dse_news"
    counts["events"] = vdb.upsert(con, "events", items)

    home_html = fetchers.homepage()
    save_raw(raw_root, "dse_home", run_date, "homepage.html", home_html,
             url="https://www.dsebd.org/")
    parsed = dse_indices.parse_homepage(home_html)
    idx_rows = [{**i, "date": ds} for i in parsed["indices"]]
    counts["indices"] = vdb.upsert(con, "indices", idx_rows)
    if parsed["totals"]:
        vdb.upsert(con, "market_totals", [{**parsed["totals"], "date": ds}])

    vdb.set_watermark(con, "collect", "eod", ds)
    log.info("collect_eod %s: %s", ds, counts)
    return counts
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/collect/test_runner.py -v`
Expected: 4 passed.

- [ ] **Step 6: Run the full suite and commit**

Run: `uv run pytest -v` — all tests pass.

```bash
git add vectora/collect tests/collect
git commit -m "feat: collect runner with raw layer, idempotency, no-trade-day handling"
```

---

### Task 10: Validation gates (`vectora/validate/checks.py`)

**Files:**
- Create: `vectora/validate/checks.py`
- Test: `tests/validate/test_checks.py` (create `tests/validate/__init__.py`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/validate/test_checks.py
from datetime import date

from vectora import db as vdb
from vectora.validate import checks


def _seed(con, rows):
    base = dict(date="2026-07-09", open=10.0, high=11.0, low=9.5, close=10.5,
                ltp=10.5, ycp=10.0, trades=100, value_mn=1.0, volume=10000,
                source="dse_eod")
    vdb.upsert(con, "prices_raw", [{**base, **r} for r in rows])


def test_clean_day_scores_100(test_db):
    _seed(test_db, [{"symbol": "AAA"}, {"symbol": "BBB"}])
    result = checks.validate_day(test_db, date(2026, 7, 9))
    assert result["score"] == 100
    assert result["issues"] == []


def test_bad_ohlc_relationship_penalized(test_db):
    _seed(test_db, [{"symbol": "AAA", "high": 9.0}])  # high < low
    result = checks.validate_day(test_db, date(2026, 7, 9))
    assert result["score"] < 100
    assert any("ohlc" in i for i in result["issues"])


def test_extreme_move_flagged(test_db):
    _seed(test_db, [{"symbol": "AAA", "close": 20.0, "ycp": 10.0}])  # +100% day
    result = checks.validate_day(test_db, date(2026, 7, 9))
    assert any("extreme_move" in i for i in result["issues"])


def test_missing_symbols_vs_previous_day(test_db):
    _seed(test_db, [{"symbol": "AAA", "date": "2026-07-08"},
                    {"symbol": "BBB", "date": "2026-07-08"},
                    {"symbol": "AAA", "date": "2026-07-09"}])
    result = checks.validate_day(test_db, date(2026, 7, 9),
                                 prev=date(2026, 7, 8))
    assert any("missing_symbols" in i for i in result["issues"])


def test_result_written_to_data_quality(test_db):
    _seed(test_db, [{"symbol": "AAA"}])
    checks.validate_day(test_db, date(2026, 7, 9))
    row = test_db.execute(
        "SELECT score FROM data_quality WHERE source='dse_eod'").fetchone()
    assert row is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/validate/test_checks.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `vectora/validate/checks.py`**

```python
"""Validation gates over prices_raw for one day -> data_quality score.

Score starts at 100; each issue class deducts points. Alerts and
predictions downstream are suppressed when score < settings.MIN_QUALITY_SCORE.
"""
import json
from datetime import date

from vectora import db as vdb
from vectora.calendar import previous_trading_day

_PENALTY = {"ohlc": 20, "extreme_move": 5, "missing_symbols": 10,
            "nonpositive_price": 15, "empty_day": 100}
# Hard sanity bound, deliberately wider than any DSE circuit band so only
# data errors (not big legitimate moves) trip it.
_EXTREME_MOVE = 0.50


def validate_day(con, d: date, prev: date | None = None) -> dict:
    ds = d.isoformat()
    issues: list[str] = []
    rows = con.execute(
        "SELECT symbol, open, high, low, close, ycp FROM prices_raw "
        "WHERE date = ? AND source = 'dse_eod'", [ds]).fetchall()

    if not rows:
        issues.append("empty_day")
    for sym, o, h, lo, c, ycp in rows:
        prices = [p for p in (o, h, lo, c) if p is not None]
        if any(p <= 0 for p in prices):
            issues.append(f"nonpositive_price:{sym}")
        if h is not None and lo is not None and h < lo:
            issues.append(f"ohlc:{sym}:high<low")
        if (c is not None and ycp not in (None, 0)
                and abs(c / ycp - 1) > _EXTREME_MOVE):
            issues.append(f"extreme_move:{sym}:{c/ycp - 1:+.0%}")

    prev = prev or previous_trading_day(d)
    missing = con.execute(
        "SELECT count(*) FROM (SELECT symbol FROM prices_raw WHERE date=? AND source='dse_eod' "
        "EXCEPT SELECT symbol FROM prices_raw WHERE date=? AND source='dse_eod')",
        [prev.isoformat(), ds]).fetchone()[0]
    prev_count = con.execute(
        "SELECT count(*) FROM prices_raw WHERE date=? AND source='dse_eod'",
        [prev.isoformat()]).fetchone()[0]
    if prev_count and missing > max(5, prev_count * 0.02):
        issues.append(f"missing_symbols:{missing}_of_{prev_count}")

    score = 100
    for issue in issues:
        score -= _PENALTY[issue.split(":")[0]]
    score = max(score, 0)
    vdb.upsert(con, "data_quality",
               [{"date": ds, "source": "dse_eod", "score": score,
                 "issues": json.dumps(issues[:50])}])
    return {"score": score, "issues": issues}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/validate/test_checks.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add vectora/validate tests/validate
git commit -m "feat: daily validation gates producing data_quality scores"
```

---

### Task 11: Orchestrator + CLI (`vectora/orchestrator.py`, `vectora/__main__.py`)

**Files:**
- Create: `vectora/orchestrator.py`, `vectora/__main__.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_orchestrator.py
from datetime import date

from vectora import orchestrator


def test_eod_pipeline_runs_collect_then_validate(monkeypatch, test_db):
    calls = []
    monkeypatch.setattr(orchestrator, "_stage_collect",
                        lambda con, d: calls.append("collect") or {"prices": 10})
    monkeypatch.setattr(orchestrator, "_stage_validate",
                        lambda con, d: calls.append("validate") or {"score": 100})
    rc = orchestrator.run_pipeline("eod", con=test_db, run_date=date(2026, 7, 9))
    assert rc == 0
    assert calls == ["collect", "validate"]


def test_non_trading_day_skips_and_succeeds(test_db):
    rc = orchestrator.run_pipeline("eod", con=test_db, run_date=date(2026, 7, 10))  # Friday
    assert rc == 0


def test_stage_failure_returns_nonzero(monkeypatch, test_db):
    def boom(con, d):
        raise RuntimeError("scrape failed")
    monkeypatch.setattr(orchestrator, "_stage_collect", boom)
    rc = orchestrator.run_pipeline("eod", con=test_db, run_date=date(2026, 7, 9))
    assert rc == 1


def test_unknown_pipeline_rejected(test_db):
    import pytest
    with pytest.raises(ValueError):
        orchestrator.run_pipeline("nope", con=test_db, run_date=date(2026, 7, 9))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `vectora/orchestrator.py`**

```python
"""Thin DAG runner: named pipelines as ordered stage lists.

Design rule: stages are idempotent per run_date; a failed run can simply
be re-run. Exit code 0 = success or legitimate skip, 1 = stage failure.
"""
import logging
from datetime import date

from vectora import calendar
from vectora import db as vdb
from vectora.collect import runner as collect_runner
from vectora.validate import checks

log = logging.getLogger(__name__)


def _stage_collect(con, d: date) -> dict:
    return collect_runner.collect_eod(con, d)


def _stage_validate(con, d: date) -> dict:
    return checks.validate_day(con, d)


PIPELINES: dict[str, list[str]] = {
    "eod": ["collect", "validate"],
}


def run_pipeline(name: str, con=None, run_date: date | None = None) -> int:
    if name not in PIPELINES:
        raise ValueError(f"unknown pipeline: {name!r} (have {sorted(PIPELINES)})")
    run_date = run_date or date.today()
    if not calendar.is_trading_day(run_date):
        log.info("%s is not a trading day — skipping %s pipeline", run_date, name)
        return 0
    own_con = con is None
    if own_con:
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        vdb.init_schema(con)
    try:
        for stage in PIPELINES[name]:
            fn = globals()[f"_stage_{stage}"]
            log.info("stage %s starting for %s", stage, run_date)
            result = fn(con, run_date)
            log.info("stage %s done: %s", stage, result)
        return 0
    except Exception:
        log.exception("pipeline %s failed on %s", name, run_date)
        return 1
    finally:
        if own_con:
            con.close()
```

- [ ] **Step 4: Implement `vectora/__main__.py`**

```python
"""CLI: python -m vectora run <pipeline> [--date YYYY-MM-DD]"""
import argparse
import logging
import sys
from datetime import date

from vectora.orchestrator import PIPELINES, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(prog="vectora")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a pipeline")
    run.add_argument("pipeline", choices=sorted(PIPELINES))
    run.add_argument("--date", type=date.fromisoformat, default=None,
                     help="run date (default: today, Dhaka semantics)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run_pipeline(args.pipeline, run_date=args.date)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: 4 passed.

- [ ] **Step 6: Live smoke test (network)**

Run: `uv run python -m vectora run eod --date <most recent trading day, e.g. 2026-07-09>`
Expected: log lines for both stages, exit code 0, then verify:
`uv run python -c "import duckdb; con = duckdb.connect('data/vectora.duckdb'); print(con.execute('SELECT count(*) FROM prices_raw').fetchone(), con.execute('SELECT * FROM data_quality').fetchall())"`
Expected: several hundred price rows, a data_quality row with score ≥ 80.

- [ ] **Step 7: Commit**

```bash
git add vectora/orchestrator.py vectora/__main__.py tests/test_orchestrator.py data/vectora.duckdb
git commit -m "feat: pipeline orchestrator and CLI with live-verified eod run"
```

---

### Task 12: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/eod-pipeline.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: eod-pipeline

on:
  schedule:
    # 09:30 UTC = 15:30 Dhaka (UTC+6), Sun-Thu trading days.
    # GitHub cron uses 0=Sunday. Actual trading-day check happens in code.
    - cron: "30 9 * * 0,1,2,3,4"
  workflow_dispatch: {}

concurrency:
  group: vectora-pipeline
  cancel-in-progress: false

permissions:
  contents: write

jobs:
  eod:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: uv sync --frozen

      - name: Run EOD pipeline
        run: uv run python -m vectora run eod

      - name: Commit data updates
        run: |
          git config user.name "vectora-bot"
          git config user.email "vectora-bot@users.noreply.github.com"
          git add -f data/raw data/vectora.duckdb data/reference
          if git diff --cached --quiet; then
            echo "no data changes"
          else
            git commit -m "data: eod $(date -u +%F)"
            git pull --rebase origin main
            git push
          fi
```

- [ ] **Step 2: Add a CI test workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push: {}
  pull_request: {}
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run pytest
```

- [ ] **Step 3: Create the private GitHub repo and push**

```bash
gh repo create vectora --private --source . --push
```

If `gh` is not authenticated, run `gh auth login` first (browser flow) — this needs the user's GitHub account; pause and ask if not configured.

- [ ] **Step 4: Trigger a manual run and verify**

Run: `gh workflow run eod-pipeline && sleep 90 && gh run list --workflow eod-pipeline --limit 1`
Expected: status `completed`, conclusion `success` (on a non-trading day the run still succeeds via the calendar skip). Inspect logs with `gh run view --log` if not.

- [ ] **Step 5: Commit**

```bash
git add .github
git commit -m "ci: eod pipeline schedule + test workflow"
git push
```

---

### Task 13: Mendeley historical backfill (`tools/backfill_mendeley.py`)

**Files:**
- Create: `tools/backfill_mendeley.py`
- Test: `tests/test_backfill.py`

- [ ] **Step 1: Manual download (user action, one-time)**

Download the "Dhaka Stock Exchange End-of-Day Financial Dataset" v4 archive from https://data.mendeley.com/datasets/23553sm4tn/4 ("Download all") and save the zip anywhere locally, e.g. `~/Downloads/dse-eod-mendeley.zip`. Do not commit the zip.

- [ ] **Step 2: Write the failing test (uses a synthetic fixture, not the real zip)**

```python
# tests/test_backfill.py
import io
import zipfile

from tools.backfill_mendeley import load_zip


def _fake_zip(tmp_path):
    """Mimics the dataset: per-file CSVs with flexible header spellings."""
    buf = tmp_path / "fake.zip"
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("ACI.csv",
                   "Date,Open,High,Low,Close,Volume\n"
                   "2013-01-02,150.5,152.0,149.0,151.25,120000\n"
                   "2013-01-03,151.5,153.0,150.0,150.75,90000\n")
        z.writestr("GP.csv",
                   "date,open_price,high_price,low_price,close_price,trading_volume\n"
                   "2013-01-02,180.0,181.0,178.5,180.5,500000\n")
        z.writestr("readme.txt", "not a csv")
    return buf


def test_load_zip_inserts_rows(test_db, tmp_path):
    stats = load_zip(test_db, _fake_zip(tmp_path))
    assert stats["rows"] == 3 and stats["files"] == 2
    n = test_db.execute(
        "SELECT count(*) FROM prices_raw WHERE source='mendeley'").fetchone()[0]
    assert n == 3
    row = test_db.execute(
        "SELECT open, close, volume FROM prices_raw "
        "WHERE symbol='GP' AND source='mendeley'").fetchone()
    assert row == (180.0, 180.5, 500000)


def test_load_zip_is_idempotent(test_db, tmp_path):
    z = _fake_zip(tmp_path)
    load_zip(test_db, z)
    load_zip(test_db, z)
    n = test_db.execute(
        "SELECT count(*) FROM prices_raw WHERE source='mendeley'").fetchone()[0]
    assert n == 3
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_backfill.py -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement `tools/backfill_mendeley.py`**

```python
"""Load the Mendeley DSE EOD dataset (23553sm4tn v4) into prices_raw.

The dataset ships one CSV per instrument (symbol = filename stem) with
header spellings that vary; map them defensively. Rows land with
source='mendeley'; live-scraped dse_eod rows always win downstream
because clean/features prefer source='dse_eod' when both exist.

Usage: uv run python tools/backfill_mendeley.py ~/Downloads/dse-eod-mendeley.zip
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

from vectora import db as vdb
from vectora.settings import DB_PATH

_HEADER_MAP = {
    "date": "date", "trading_date": "date",
    "open": "open", "open_price": "open", "openp": "open",
    "high": "high", "high_price": "high",
    "low": "low", "low_price": "low",
    "close": "close", "close_price": "close", "closep": "close",
    "volume": "volume", "trading_volume": "volume", "vol": "volume",
}
_REQUIRED = {"date", "open", "high", "low", "close", "volume"}


def _normalize_header(fieldnames: list[str]) -> dict[str, str] | None:
    mapping = {}
    for f in fieldnames:
        key = f.strip().lower().replace(" ", "_")
        if key in _HEADER_MAP:
            mapping[f] = _HEADER_MAP[key]
    return mapping if _REQUIRED <= set(mapping.values()) else None


def load_zip(con, zip_path: Path | str) -> dict:
    stats = {"files": 0, "rows": 0, "skipped_files": []}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            symbol = Path(name).stem.upper()
            with z.open(name) as f:
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig"))
                mapping = _normalize_header(reader.fieldnames or [])
                if mapping is None:
                    stats["skipped_files"].append(name)
                    continue
                rows = []
                for raw in reader:
                    try:
                        r = {mapping[k]: v for k, v in raw.items() if k in mapping}
                        rows.append({
                            "symbol": symbol, "date": r["date"][:10],
                            "open": float(r["open"]), "high": float(r["high"]),
                            "low": float(r["low"]), "close": float(r["close"]),
                            "ltp": None, "ycp": None, "trades": None,
                            "value_mn": None, "volume": int(float(r["volume"])),
                            "source": "mendeley",
                        })
                    except (ValueError, KeyError):
                        continue  # malformed row; dataset has some
                stats["rows"] += vdb.upsert(con, "prices_raw", rows)
                stats["files"] += 1
    return stats


if __name__ == "__main__":
    con = vdb.connect(DB_PATH)
    vdb.init_schema(con)
    result = load_zip(con, sys.argv[1])
    print(result if not result["skipped_files"]
          else {**result, "note": "some files skipped — inspect headers"})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_backfill.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run the real backfill (local, one-time)**

Run: `uv run python tools/backfill_mendeley.py <path-to-downloaded-zip>`
Expected: `files` in the hundreds, `rows` > 500,000, few/no `skipped_files`. If many files are skipped, print one file's header row and extend `_HEADER_MAP` accordingly (the map is the single point of change), re-run — idempotent.

Sanity check: `uv run python -c "import duckdb; con=duckdb.connect('data/vectora.duckdb'); print(con.execute(\"SELECT min(date), max(date), count(DISTINCT symbol) FROM prices_raw WHERE source='mendeley'\").fetchone())"`
Expected: range ≈ 2012→2026, 300+ symbols.

- [ ] **Step 7: Commit**

```bash
git add tools/backfill_mendeley.py tests/test_backfill.py data/vectora.duckdb
git commit -m "feat: Mendeley historical EOD backfill (2012-2026)"
git push
```

---

### Task 14: Reference bootstrap (`tools/bootstrap_reference.py`)

**Files:**
- Create: `tools/bootstrap_reference.py`
- Test: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bootstrap.py
from datetime import date

from vectora import db as vdb
from tools.bootstrap_reference import refresh_symbols, sweep_companies


def test_refresh_symbols_from_prices(test_db):
    vdb.upsert(test_db, "prices_raw", [
        dict(symbol="AAA", date="2026-07-09", open=1, high=1, low=1, close=1,
             ltp=1, ycp=1, trades=1, value_mn=1, volume=1, source="dse_eod"),
        dict(symbol="BBB", date="2026-07-08", open=1, high=1, low=1, close=1,
             ltp=1, ycp=1, trades=1, value_mn=1, volume=1, source="mendeley"),
    ])
    n = refresh_symbols(test_db)
    assert n == 2
    rows = dict(test_db.execute(
        "SELECT symbol, last_seen FROM symbols").fetchall())
    assert str(rows["AAA"]) == "2026-07-09"


def test_sweep_companies_writes_snapshot_and_holdings(test_db, fixtures_dir):
    vdb.upsert(test_db, "symbols", [dict(
        symbol="GP", name=None, sector=None, instrument_type=None, category=None,
        listing_status="active", first_seen="2026-01-01", last_seen="2026-07-09")])

    def fake_fetch(symbol):
        return (fixtures_dir / "company_GP.html").read_text(encoding="utf-8")

    done = sweep_companies(test_db, as_of=date(2026, 7, 12), fetch=fake_fetch)
    assert done == 1
    cat = test_db.execute("SELECT category FROM symbols WHERE symbol='GP'").fetchone()[0]
    assert cat in {"A", "B", "N", "Z", "G"}
    assert test_db.execute("SELECT count(*) FROM holdings").fetchone()[0] >= 1
    # resumable: second call skips already-swept symbol
    assert sweep_companies(test_db, as_of=date(2026, 7, 12), fetch=fake_fetch) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `tools/bootstrap_reference.py`**

```python
"""One-time / occasional reference builders.

refresh_symbols: derive the symbol master from observed price rows.
sweep_companies: fetch every active symbol's company page (polite pace,
resumable via watermarks) to fill sector/category/holdings.

Usage:
  uv run python tools/bootstrap_reference.py symbols
  uv run python tools/bootstrap_reference.py sweep      # ~650 pages, ~20 min
"""
import sys
from datetime import date

from vectora import db as vdb
from vectora.collect import dse_company
from vectora.http import PoliteSession
from vectora.settings import DB_PATH


def refresh_symbols(con) -> int:
    con.execute("""
        INSERT OR REPLACE INTO symbols (symbol, first_seen, last_seen, listing_status)
        SELECT p.symbol, min(p.date), max(p.date),
               CASE WHEN max(p.date) >= (SELECT max(date) - INTERVAL 30 DAY
                                         FROM prices_raw) THEN 'active'
                    ELSE 'inactive' END
        FROM prices_raw p GROUP BY p.symbol
    """)
    return con.execute("SELECT count(*) FROM symbols").fetchone()[0]


def sweep_companies(con, as_of: date, fetch=None) -> int:
    if fetch is None:
        session = PoliteSession()
        fetch = lambda sym: dse_company.fetch_company(session, sym)  # noqa: E731
    symbols = [r[0] for r in con.execute(
        "SELECT symbol FROM symbols WHERE listing_status='active' ORDER BY symbol"
    ).fetchall()]
    done = 0
    for sym in symbols:
        if vdb.get_watermark(con, "company_sweep", sym) == as_of.isoformat():
            continue
        try:
            parsed = dse_company.parse_company(fetch(sym), symbol=sym)
        except Exception as exc:  # keep sweeping; log the miss
            print(f"WARN {sym}: {exc}")
            continue
        snap = parsed["snapshot"]
        vdb.upsert(con, "company_snapshot", [{**snap, "as_of": as_of.isoformat()}])
        con.execute(
            "UPDATE symbols SET sector = ?, category = ?, instrument_type = ? "
            "WHERE symbol = ?",
            [snap["sector"], snap["category"], snap["instrument_type"], sym])
        if parsed["holdings"]:
            vdb.upsert(con, "holdings",
                       [{**h, "symbol": sym} for h in parsed["holdings"]])
        vdb.set_watermark(con, "company_sweep", sym, as_of.isoformat())
        done += 1
    return done


if __name__ == "__main__":
    con = vdb.connect(DB_PATH)
    vdb.init_schema(con)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "symbols"
    if cmd == "symbols":
        print(f"symbols: {refresh_symbols(con)}")
    elif cmd == "sweep":
        print(f"symbols: {refresh_symbols(con)}")
        print(f"swept: {sweep_companies(con, as_of=date.today())}")
    else:
        sys.exit(f"unknown command {cmd!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the live sweep (local, ~20 minutes)**

Run: `uv run python tools/bootstrap_reference.py sweep`
Expected: `symbols: 600+`, `swept: 300+` (equities + funds; some instruments have thin pages and will WARN — acceptable if warnings < 10%). Re-run to resume if interrupted (watermark-based).

Sanity: `uv run python -c "import duckdb; con=duckdb.connect('data/vectora.duckdb'); print(con.execute('SELECT category, count(*) FROM symbols GROUP BY 1 ORDER BY 2 DESC').fetchall())"`
Expected: A/B/N/Z counts, Z typically 50–100 — this is the Z-universe for later phases.

- [ ] **Step 6: Commit**

```bash
git add tools/bootstrap_reference.py tests/test_bootstrap.py data/vectora.duckdb
git commit -m "feat: symbol master + company sweep bootstrap"
git push
```

---

### Task 15: Acceptance — the 5-day soak

**Files:** none (operations)

- [ ] **Step 1: Confirm the schedule is live**

`gh workflow list` → `eod-pipeline` active. The cron fires 15:30 Dhaka Sun–Thu.

- [ ] **Step 2: Daily check for 5 consecutive trading days**

Each day: `gh run list --workflow eod-pipeline --limit 1` shows success, and the repo gains a `data: eod YYYY-MM-DD` commit containing that day's raw payloads + DB update with `data_quality.score ≥ 80`.

- [ ] **Step 3: Exit criterion (from spec Phase 1)**

Five consecutive green trading days ⇒ Phase 1 complete. Record the completion date in `docs/superpowers/plans/` as a note, then Phase 2 (clean/corporate-actions, features, baseline models) gets its own plan.

If any day fails: diagnose via `gh run view --log`, fix, and **restart the 5-day count only if the failure was a code defect** (infra flakes like runner outages don't reset the count — the next run backfills via `--date`).

---

## Deferred to Phase 2 (explicitly not here)

Corporate-action price adjustment (`clean` module), BSEC/Bangladesh Bank/Google Trends collectors, intraday scans, feature engine, models, alerts, and the Obsidian vault generator. The `prices_raw` + `events` + `company_snapshot`/`holdings` tables built here are their inputs.

## Self-Review Notes

- Spec coverage (Phase 0–1 scope): source verification ✔ (ground-truth section), backfill ✔ (Task 13), symbol master + calendar ✔ (Tasks 8, 14), EOD/news/company/indices scrapers ✔ (Tasks 4–7), validation + quality score ✔ (Task 10), DuckDB schema subset ✔ (Task 2), raw immutable layer ✔ (Task 9), workflow green ✔ (Tasks 12, 15). Indices *history* comes from homepage-forward collection only; historical index backfill deferred to Phase 2 (noted — DSEX history is obtainable later without blocking features on stocks).
- Types/signatures cross-checked: `PoliteSession.get(url, params)`, `vdb.upsert(con, table, rows) -> int`, `parse_day_end -> list[dict]` keys match `prices_raw` columns + `source` added by runner; `events` table columns match news item keys + `source`; `collect_eod(con, run_date, fetchers, raw_root)` matches tests.
- Known live-run risk points are called out inline with diagnosis steps (fixture recorded on holiday, holdings label nesting, totals row selection, Mendeley header variants).
