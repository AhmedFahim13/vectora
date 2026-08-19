"""Keep the committed database small without losing knowledge.

The working database is committed on every pipeline run, so its size is not
a storage question but a *rate* question: a 60 MB binary re-committed daily
adds a fresh blob each time, because binaries do not delta-compress. That is
what grew .git past 450 MB.

The schema's standing rule is that rows are never deleted — old knowledge
never disappears. This module honours that rule rather than bending it, by
splitting the tables into two kinds:

REGENERABLE — ta_ratings, ta_gauges, ta_levels. Every row is a pure function
of prices, recomputable with `run ta --date YYYY-MM-DD`, and only ever read
one date at a time. Old rows are dropped outright because nothing is lost:
they can be rebuilt exactly.

ARCHIVED — intraday_snapshots and explanations. These cannot be regenerated,
so they are written to immutable monthly Parquet under data/archive/ before
being pruned. A month's file is written once and never touched again, so git
stores it a single time instead of re-committing it daily. Parquet with ZSTD
also compresses this text far better than DuckDB's row store.

Net effect: the same knowledge, one copy of each historical row in git
instead of one per day.
"""
import shutil
from pathlib import Path

from vectora.settings import DATA_DIR, DB_PATH

# regenerable via `run ta`: keep a few days for the page, drop the rest
# only the newest date is ever queried; the second is kept as a safety net
REGENERABLE = {"ta_ratings": 2, "ta_gauges": 2, "ta_levels": 2}
INTRADAY_KEEP_DAYS = 30
EXPLANATION_KEEP_DAYS = 15


def archive_dir() -> Path:
    return Path(DATA_DIR) / "archive"


def prune_regenerable(con, keep: dict | None = None) -> dict:
    """Drop old rows from tables that are a pure function of prices."""
    keep = keep or REGENERABLE
    out = {}
    for table, n_dates in keep.items():
        before = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        con.execute(
            f"""
            DELETE FROM {table} WHERE date < (
                SELECT min(d) FROM (
                    SELECT DISTINCT date AS d FROM {table}
                    ORDER BY d DESC LIMIT {int(n_dates)}))
            """)
        after = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        out[table] = before - after
    return out


def _archive_months(con, table: str, select_sql: str, ts_col: str,
                    root: Path) -> int:
    """Write the selected rows to data/archive/<table>/<YYYY-MM>.parquet.

    A month already on disk is merged and rewritten, so a late-arriving row
    cannot silently vanish; once a month is complete the file stops changing
    and git stores it once.
    """
    dest = root / table
    dest.mkdir(parents=True, exist_ok=True)
    months = [r[0] for r in con.execute(
        f"SELECT DISTINCT strftime({ts_col}, '%Y-%m') AS m "
        f"FROM ({select_sql}) ORDER BY m").fetchall()]
    written = 0
    for m in months:
        path = dest / f"{m}.parquet"
        posix = str(path).replace("\\", "/")
        rows = f"SELECT * FROM ({select_sql}) WHERE strftime({ts_col}, '%Y-%m') = '{m}'"
        if path.exists():
            # DISTINCT, not UNION ALL: intraday appends the same month from
            # several runs a day and identical snapshots must collapse
            rows = (f"SELECT DISTINCT * FROM (SELECT * FROM ({rows}) "
                    f"UNION ALL BY NAME SELECT * FROM read_parquet('{posix}'))")
        tmp = str(dest / f".{m}.tmp.parquet").replace("\\", "/")
        con.execute(f"COPY ({rows}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        Path(tmp).replace(path)
        written += 1
    return written


def archive_and_prune(con, root: Path | None = None,
                      intraday_days: int = INTRADAY_KEEP_DAYS,
                      explanation_days: int = EXPLANATION_KEEP_DAYS) -> dict:
    """Move unregenerable history to Parquet, then remove it from the DB."""
    root = Path(root) if root else archive_dir()
    out = {}

    sel = (f"SELECT * FROM intraday_snapshots WHERE ts < "
           f"current_date - INTERVAL {int(intraday_days)} DAY")
    n = con.execute(f"SELECT count(*) FROM ({sel})").fetchone()[0]
    if n:
        _archive_months(con, "intraday_snapshots", sel, "ts", root)
        con.execute(
            f"DELETE FROM intraday_snapshots WHERE ts < "
            f"current_date - INTERVAL {int(intraday_days)} DAY")
    out["intraday_snapshots"] = n

    # Explanations carry no date of their own; they age with their prediction.
    # Three conditions before one may move: the prediction must be resolved
    # (an open one is still on the dashboard), past the retention window, and
    # NOT a signal. Signals are the track record — a handful of rows — and
    # stay in the database permanently. The other ~335 explanations written
    # every day are never displayed by anything and are the single fastest
    # source of growth in the file.
    sel = (
        "SELECT e.*, p.date AS pred_date FROM explanations e "
        "JOIN predictions p ON p.id = e.prediction_id "
        "JOIN outcomes o ON o.prediction_id = e.prediction_id "
        "WHERE NOT coalesce(p.is_signal, false) "
        f"AND p.date < current_date - INTERVAL {int(explanation_days)} DAY")
    n = con.execute(f"SELECT count(*) FROM ({sel})").fetchone()[0]
    if n:
        _archive_months(con, "explanations", sel, "pred_date", root)
        con.execute(f"""
            DELETE FROM explanations WHERE prediction_id IN (
                SELECT e.prediction_id FROM explanations e
                JOIN predictions p ON p.id = e.prediction_id
                JOIN outcomes o ON o.prediction_id = e.prediction_id
                WHERE NOT coalesce(p.is_signal, false)
                  AND p.date < current_date
                      - INTERVAL {int(explanation_days)} DAY)""")
    out["explanations"] = n
    return out


MIN_COMPACT_SAVING_MB = 2.0


def compact(db_path: Path | None = None,
            min_saving_mb: float = MIN_COMPACT_SAVING_MB) -> dict:
    """Rewrite the database file compactly, but only if that actually wins.

    DuckDB never shrinks a file in place: replaced rows leave free pages
    behind. The rebuild goes to a temporary file and only replaces the
    original once it has completed, so a failure here cannot corrupt data.

    The threshold matters more than it looks. DuckDB's block layout is not
    byte-stable, so rebuilding an already-compact file produces a DIFFERENT
    42 MB file — and since this database is committed, that is a whole new
    blob in git history for zero benefit. Below `min_saving_mb` the rebuild
    is discarded and the original left untouched.
    """
    import duckdb
    src = Path(db_path or DB_PATH)
    before = src.stat().st_size
    tmp = src.with_suffix(".compact.tmp")
    if tmp.exists():
        tmp.unlink()
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{str(src).replace(chr(92), '/')}' AS old (READ_ONLY)")
        con.execute(f"ATTACH '{str(tmp).replace(chr(92), '/')}' AS new")
        con.execute("COPY FROM DATABASE old TO new")
    finally:
        con.close()
    rebuilt = tmp.stat().st_size
    saved_mb = (before - rebuilt) / 1024 / 1024
    if saved_mb < min_saving_mb:
        tmp.unlink()
        return {"before_mb": before / 1024 / 1024,
                "after_mb": before / 1024 / 1024, "saved_mb": 0.0,
                "skipped": f"rebuild would save only {saved_mb:.2f} MB"}
    shutil.move(str(tmp), str(src))
    after = src.stat().st_size
    return {"before_mb": before / 1024 / 1024, "after_mb": after / 1024 / 1024,
            "saved_mb": (before - after) / 1024 / 1024, "skipped": None}


def archive_intraday(con, root: Path | None = None) -> int:
    """Append the snapshots held in the database to the monthly archive.

    The intraday workflow runs four times a day. Committing the whole
    database each time added four ~42 MB blobs daily, which is what actually
    grew git history — the file size itself is near its floor. Snapshots go
    to immutable monthly Parquet instead, so the intraday runs commit
    kilobytes and the database is committed once, by the EOD run.
    """
    root = Path(root) if root else archive_dir()
    sel = "SELECT * FROM intraday_snapshots"
    n = con.execute(f"SELECT count(*) FROM ({sel})").fetchone()[0]
    if n:
        _archive_months(con, "intraday_snapshots", sel, "ts", root)
    return n


def archive_alerts(con, root: Path | None = None) -> int:
    """Persist alerts_log to the monthly archive."""
    root = Path(root) if root else archive_dir()
    sel = "SELECT * FROM alerts_log"
    n = con.execute(f"SELECT count(*) FROM ({sel})").fetchone()[0]
    if n:
        _archive_months(con, "alerts_log", sel, "alert_date", root)
    return n


def restore_alerts(con, date_str: str, root: Path | None = None) -> int:
    """Load one day's alerts back into the database.

    The intraday workflow no longer commits the database, so each of its four
    daily runs starts from the EOD checkout with an empty alerts_log for
    today. Without this the cooldown and dedup checks would see no prior
    alerts and re-send the same symbol every run.
    """
    root = Path(root) if root else archive_dir()
    path = root / "alerts_log" / f"{date_str[:7]}.parquet"
    if not path.exists():
        return 0
    posix = str(path).replace("\\", "/")
    rows = con.execute(
        f"SELECT * FROM read_parquet('{posix}') WHERE CAST(alert_date AS "
        f"VARCHAR) = ?", [date_str]).fetchall()
    if not rows:
        return 0
    cols = [d[0] for d in con.execute(
        f"SELECT * FROM read_parquet('{posix}') LIMIT 0").description]
    con.execute(
        f"INSERT OR REPLACE INTO alerts_log ({', '.join(cols)}) "
        f"SELECT {', '.join(cols)} FROM read_parquet('{posix}') "
        f"WHERE CAST(alert_date AS VARCHAR) = ?", [date_str])
    return len(rows)
