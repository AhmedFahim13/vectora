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
