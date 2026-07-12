"""Load the Mendeley DSE EOD dataset (2012-2026) into prices_raw.

Dataset: https://data.mendeley.com/datasets/23553sm4tn/4 — download the zip
manually ("Download All"), then:

  uv run python tools/backfill_mendeley.py path/to/dataset.zip

One CSV per instrument (Date,Open,High,Low,Close,Volume; symbol = filename
stem). When the zip carries both Adjusted and Unadjusted folders, only the
UNadjusted files load — prices_raw stores as-traded prices, adjustments are
Phase 2's job. Rows land under source='mendeley', coexisting with scraped
'dse_eod' rows (PK symbol,date,source). Idempotent.

Loading goes through DuckDB's native read_csv over the extracted files —
one vectorized INSERT for the whole dataset (~3M rows in seconds), not
per-row prepared statements.
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from vectora import db as vdb
from vectora.settings import DB_PATH


def _select_members(z: zipfile.ZipFile) -> list[str]:
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    # combined all-instrument dumps (UNADJUSTED.csv / ADJUSTED.csv) duplicate
    # the per-company files; only per-company files carry the symbol in the
    # filename, so dumps must never load
    csvs = [n for n in csvs if "adjust" not in Path(n).stem.lower()]
    unadjusted = [n for n in csvs if "unadjust" in n.lower()]
    if unadjusted:
        return unadjusted
    # exclude adjusted-only folders when no explicit unadjusted set exists
    plain = [n for n in csvs if "adjust" not in n.lower()]
    return plain or csvs


def load_zip(con, zip_path: Path) -> dict:
    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(zip_path) as z:
        files = 0
        for member in _select_members(z):
            symbol = Path(member).stem.strip().upper()
            # 00-prefixed files are index series (00DSEX, 00DSMEX), not
            # instruments; indices live in the indices table via the scraper
            if not symbol or symbol.startswith("00"):
                continue
            with z.open(member) as src, open(Path(td) / f"{symbol}.csv", "wb") as dst:
                shutil.copyfileobj(src, dst)
            files += 1
        if files == 0:
            return {"files": 0, "rows": 0, "skipped_rows": 0}

        glob = str(Path(td) / "*.csv").replace("\\", "/")
        src = (
            f"read_csv('{glob}', all_varchar=true, header=true, filename=true)"
        )
        total = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
        # dedup via explicit row_number subquery — QUALIFY over a derived
        # table mis-partitioned here (observed dropping ~50% of rows on
        # DuckDB 1.5.4), the two-level form has unambiguous semantics
        rows = con.execute(
            f"""
            INSERT OR REPLACE INTO prices_raw
            SELECT symbol, date, open, high, low, close, ltp, ycp, trades,
                   value_mn, volume, source
            FROM (
                SELECT p.*, row_number() OVER (PARTITION BY symbol, date) AS rn
                FROM (
                    SELECT
                        upper(regexp_extract(filename, '([^/\\\\]+)\\.csv$', 1))
                            AS symbol,
                        coalesce(
                            try_cast("Date" AS DATE),
                            try_cast(try_strptime("Date", '%d-%m-%Y') AS DATE),
                            try_cast(try_strptime("Date", '%m/%d/%Y') AS DATE)
                        ) AS date,
                        try_cast("Open" AS DOUBLE) AS open,
                        try_cast("High" AS DOUBLE) AS high,
                        try_cast("Low" AS DOUBLE) AS low,
                        try_cast("Close" AS DOUBLE) AS close,
                        NULL::DOUBLE AS ltp,
                        NULL::DOUBLE AS ycp,
                        NULL::BIGINT AS trades,
                        NULL::DOUBLE AS value_mn,
                        try_cast(try_cast("Volume" AS DOUBLE) AS BIGINT) AS volume,
                        'mendeley' AS source
                    FROM {src}
                ) p
                WHERE date IS NOT NULL AND close IS NOT NULL
            )
            WHERE rn = 1
            """
        ).fetchone()[0]
        return {"files": files, "rows": rows, "skipped_rows": total - rows}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    zip_path = Path(sys.argv[1])
    if not zip_path.exists():
        print(f"not found: {zip_path}")
        return 1
    con = vdb.connect(DB_PATH)
    try:
        vdb.init_schema(con)
        result = load_zip(con, zip_path)
        print(result)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
