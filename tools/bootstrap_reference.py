"""Bootstrap reference data: symbol master + company page sweep.

Usage:
  uv run python tools/bootstrap_reference.py symbols
  uv run python tools/bootstrap_reference.py sweep [limit]

`symbols` derives the symbol master from EOD price history already in the
database. `sweep` fetches every symbol's company page politely (~1.5s/page,
~20-40 min for the full board), storing snapshots, holdings history, and
sector/category on the symbol master. Both are idempotent; a partial sweep
just resumes on the next run (already-swept symbols get refreshed).
"""
import sys
from datetime import date
from pathlib import Path

from vectora import db as vdb
from vectora.collect import dse_company
from vectora.collect.raw_store import save_raw
from vectora.http import PoliteSession
from vectora.settings import DB_PATH, RAW_DIR

_SNAPSHOT_KEYS = ("symbol", "sector", "category", "instrument_type",
                  "paid_up_capital_mn", "outstanding_shares", "face_value", "market_lot")


def bootstrap_symbols(con) -> int:
    # merge, don't replace: re-runs extend first/last_seen from new price
    # history but must preserve metadata a prior company sweep filled in
    con.execute(
        """
        INSERT OR REPLACE INTO symbols
        SELECT p.symbol, s.name, s.sector, s.instrument_type, s.category,
               coalesce(s.listing_status, 'active'), p.first_seen, p.last_seen
        FROM (
            SELECT symbol, min(date) AS first_seen, max(date) AS last_seen
            FROM prices_raw GROUP BY symbol
        ) p
        LEFT JOIN symbols s USING (symbol)
        """
    )
    return con.execute("SELECT count(*) FROM symbols").fetchone()[0]


def sweep_companies(con, session, symbols: list[str], as_of: str,
                    raw_dir: Path = RAW_DIR) -> dict:
    swept, failed = 0, []
    for sym in symbols:
        try:
            html = dse_company.fetch_company(session, sym)
            save_raw(raw_dir, "dse_company", as_of, sym, html, url=dse_company.URL)
            parsed = dse_company.parse_company(html, sym)
            facts = parsed["facts"]
            if not facts:
                failed.append(sym)
                continue
            snapshot = {k: facts.get(k) for k in _SNAPSHOT_KEYS}
            vdb.upsert(con, "company_snapshot", [{**snapshot, "as_of": as_of}])
            # holdings PK is (symbol, as_of): blocks without a parseable date
            # cannot be stored
            dated_blocks = [h for h in parsed["holdings"] if h["as_of"]]
            if dated_blocks:
                vdb.upsert(con, "holdings", dated_blocks)
            # fundamentals ride along on the same page fetch (no extra request)
            fund = dse_company.parse_fundamentals(html, sym)
            vdb.upsert(con, "fundamentals", [{**fund, "as_of": as_of}])
            con.execute(
                "UPDATE symbols SET sector = ?, category = ?, instrument_type = ? "
                "WHERE symbol = ?",
                [facts.get("sector"), facts.get("category"),
                 facts.get("instrument_type"), sym],
            )
            swept += 1
        except Exception as exc:  # noqa: BLE001 - one bad page must not kill the sweep
            print(f"  FAIL {sym}: {exc}", flush=True)
            failed.append(sym)
    return {"swept": swept, "failed": failed}


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "symbols"
    con = vdb.connect(DB_PATH)
    try:
        vdb.init_schema(con)
        if cmd == "symbols":
            print(f"symbols upserted: {bootstrap_symbols(con)}")
            return 0
        if cmd == "sweep":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
            bootstrap_symbols(con)  # sweep updates symbols rows; ensure they exist
            syms = [r[0] for r in con.execute(
                "SELECT symbol FROM symbols ORDER BY symbol").fetchall()]
            if limit:
                syms = syms[:limit]
            print(f"sweeping {len(syms)} symbols...", flush=True)
            result = sweep_companies(
                con, PoliteSession(), syms, as_of=date.today().isoformat())
            print(f"swept={result['swept']} failed={len(result['failed'])}"
                  f" {result['failed'][:10]}")
            return 0
        print(f"unknown command {cmd}")
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
