"""One-time: extract Mendeley's ADJUSTED per-company series into a slim
(symbol, date, adj_close) parquet consumed by features/base.py's return
ladder. Usage: uv run python tools/build_adjusted_parquet.py path/to/zip
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import duckdb

from vectora.settings import ADJUSTED_PARQUET


def build(zip_path: Path, out: Path = ADJUSTED_PARQUET) -> int:
    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(zip_path) as z:
        n = 0
        for m in z.namelist():
            low = m.lower()
            if not low.endswith(".csv") or "unadjust" in low \
                    or "adjust" not in low:
                continue
            stem = Path(m).stem.strip().upper()
            if not stem or stem.startswith("00") or "adjust" in stem.lower():
                continue
            with z.open(m) as src, open(Path(td) / f"{stem}.csv", "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
        glob = str(Path(td) / "*.csv").replace("\\", "/")
        con = duckdb.connect()
        outp = str(out).replace("\\", "/")
        con.execute(f"""
            COPY (
                SELECT upper(regexp_extract(filename, '([^/\\\\]+)\\.csv$', 1))
                           AS symbol,
                       try_cast("Date" AS DATE) AS date,
                       try_cast("Close" AS DOUBLE) AS adj_close
                FROM read_csv('{glob}', all_varchar=true, header=true,
                              filename=true)
                WHERE try_cast("Date" AS DATE) IS NOT NULL
                  AND try_cast("Close" AS DOUBLE) IS NOT NULL
                ORDER BY symbol, date
            ) TO '{outp}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{outp}')").fetchone()[0]
        print(f"{n} files -> {rows} rows -> {out}")
        return 0


if __name__ == "__main__":
    sys.exit(build(Path(sys.argv[1])))
