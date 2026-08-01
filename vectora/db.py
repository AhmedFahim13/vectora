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
CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY, family TEXT, target TEXT,
    trained_at TIMESTAMP DEFAULT current_timestamp,
    train_end DATE, metrics TEXT,          -- metrics: JSON
    artifact_dir TEXT, active BOOLEAN DEFAULT false
);
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
CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id TEXT PRIMARY KEY,
    resolved_at TIMESTAMP DEFAULT current_timestamp,
    realized_max DOUBLE, realized_min DOUBLE, hit BOOLEAN
);
CREATE TABLE IF NOT EXISTS alerts_log (
    id TEXT PRIMARY KEY,               -- <date>_signal_<symbol>
    ts TIMESTAMP DEFAULT current_timestamp,
    alert_type TEXT, symbol TEXT, alert_date DATE,
    prediction_id TEXT
);
CREATE TABLE IF NOT EXISTS regimes (
    date DATE PRIMARY KEY, regime TEXT, confidence DOUBLE,
    method TEXT, computed_at TIMESTAMP DEFAULT current_timestamp
);
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
CREATE TABLE IF NOT EXISTS intraday_snapshots (
    symbol TEXT, ts TIMESTAMP, ltp DOUBLE, high DOUBLE, low DOUBLE,
    closep DOUBLE, ycp DOUBLE, change DOUBLE, trades BIGINT,
    value_mn DOUBLE, volume BIGINT,
    PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS outcome_tags (
    prediction_id TEXT PRIMARY KEY, tag TEXT,
    tagged_at TIMESTAMP DEFAULT current_timestamp
);
CREATE TABLE IF NOT EXISTS ta_ratings (
    date DATE, symbol TEXT, score INTEGER, band TEXT,
    votes TEXT,                     -- JSON list of {indicator, vote, reason}
    rsi DOUBLE, macd_hist DOUBLE, bb_pos DOUBLE, st_dir INTEGER,
    PRIMARY KEY (date, symbol)
);
CREATE TABLE IF NOT EXISTS ta_band_stats (
    band TEXT, horizon INTEGER, n INTEGER,
    hit_rate DOUBLE, base_rate DOUBLE, mean_fwd DOUBLE,
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (band, horizon)
);
CREATE TABLE IF NOT EXISTS ta_gauges (
    date DATE, symbol TEXT,
    ma_mean DOUBLE, ma_band TEXT, ma_buy INTEGER, ma_neutral INTEGER, ma_sell INTEGER,
    osc_mean DOUBLE, osc_band TEXT, osc_buy INTEGER, osc_neutral INTEGER, osc_sell INTEGER,
    summary_mean DOUBLE, summary_band TEXT,
    votes TEXT,                     -- JSON {ma: [...], osc: [...]}
    PRIMARY KEY (date, symbol)
);
CREATE TABLE IF NOT EXISTS ta_gauge_stats (
    gauge TEXT, band TEXT, horizon INTEGER, n INTEGER,
    hit_rate DOUBLE, base_rate DOUBLE, mean_fwd DOUBLE,
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (gauge, band, horizon)
);
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol TEXT, as_of DATE,
    market_cap_mn DOUBLE, free_float_mcap_mn DOUBLE, reserve_surplus_mn DOUBLE,
    trailing_pe DOUBLE, latest_dividend_pct DOUBLE, latest_bonus_pct DOUBLE,
    dividend_year INTEGER,
    face_value DOUBLE, listing_year INTEGER, year_end TEXT,
    PRIMARY KEY (symbol, as_of)
);
"""


def connect(path: str | Path) -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def init_schema(con: duckdb.DuckDBPyConnection,
                backfill_parquet: str | Path | None = None) -> None:
    con.execute(SCHEMA)
    if backfill_parquet is None:
        from vectora.settings import BACKFILL_PARQUET
        backfill_parquet = BACKFILL_PARQUET
    if Path(backfill_parquet).exists():
        # bake a cwd-relative path into the view when possible: the view
        # definition persists inside the committed .duckdb, and an absolute
        # local path would break any other machine that queries `prices`
        # before init_schema recreates it (all entry points run from repo root)
        p = Path(backfill_parquet)
        try:
            p = p.relative_to(Path.cwd())
        except ValueError:
            pass
        pq = str(p).replace("\\", "/")
        con.execute(f"""
            CREATE OR REPLACE VIEW prices AS
            SELECT * FROM prices_raw
            UNION ALL SELECT * FROM read_parquet('{pq}')
        """)
    else:
        con.execute("CREATE OR REPLACE VIEW prices AS SELECT * FROM prices_raw")


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
    # collapse byte-identical duplicate rows within the batch; same-PK rows with
    # different values are fine as-is (INSERT OR REPLACE handles them, last wins)
    cols = list(rows[0].keys())
    seen: dict = {}
    for r in rows:
        seen[tuple(str(r[c]) for c in cols)] = r  # exact-duplicate collapse
    rows = list(seen.values())
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    con.execute("BEGIN TRANSACTION")
    try:
        con.executemany(sql, [[r[c] for c in cols] for r in rows])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)
