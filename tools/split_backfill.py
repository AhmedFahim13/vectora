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

    # rebuild the DB file to reclaim space: copy every table to a fresh file.
    # DuckDB does not allow a read-only ATTACH of a database file while another
    # connection in this process still holds it open, so `con` must be fully
    # closed before the rebuild connection attaches to it.
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_type = 'BASE TABLE'").fetchall()]
    con.close()

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
