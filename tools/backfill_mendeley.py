"""Load the Mendeley DSE EOD dataset (2012-2026) into prices_raw.

Dataset: https://data.mendeley.com/datasets/23553sm4tn/4 — download the zip
manually ("Download All"), then:

  uv run python tools/backfill_mendeley.py path/to/dataset.zip

One CSV per instrument (Date,Open,High,Low,Close,Volume; symbol = filename
stem). When the zip carries both Adjusted and Unadjusted folders, only the
UNadjusted files load — prices_raw stores as-traded prices, adjustments are
Phase 2's job. Rows land under source='mendeley', coexisting with scraped
'dse_eod' rows (PK symbol,date,source). Idempotent.
"""
import csv
import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path

from vectora import db as vdb
from vectora.settings import DB_PATH

_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y.%m.%d")


def _parse_date(text: str) -> str | None:
    t = (text or "").strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(t, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _num(text) -> float | None:
    t = str(text or "").strip().replace(",", "")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _select_members(z: zipfile.ZipFile) -> list[str]:
    csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
    unadjusted = [n for n in csvs if "unadjust" in n.lower()]
    if unadjusted:
        return unadjusted
    # exclude adjusted-only folders when no explicit unadjusted set exists
    plain = [n for n in csvs if "adjust" not in n.lower()]
    return plain or csvs


def load_zip(con, zip_path: Path) -> dict:
    files = rows_total = skipped = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in _select_members(z):
            symbol = Path(member).stem.strip().upper()
            # 00-prefixed files are index series (00DSEX, 00DSMEX), not
            # instruments; indices live in the indices table via the scraper
            if not symbol or symbol.startswith("00"):
                continue
            with z.open(member) as fh:
                reader = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8-sig"))
                rows = []
                for raw in reader:
                    cells = {(k or "").strip().lower(): v for k, v in raw.items()}
                    d = _parse_date(cells.get("date", ""))
                    close = _num(cells.get("close"))
                    if d is None or close is None:
                        skipped += 1
                        continue
                    rows.append({
                        "symbol": symbol, "date": d,
                        "open": _num(cells.get("open")),
                        "high": _num(cells.get("high")),
                        "low": _num(cells.get("low")),
                        "close": close,
                        "ltp": None, "ycp": None, "trades": None,
                        "value_mn": None,
                        "volume": (lambda v: int(v) if v is not None else None)(
                            _num(cells.get("volume"))),
                        "source": "mendeley",
                    })
            if rows:
                rows_total += vdb.upsert(con, "prices_raw", rows)
                files += 1
    return {"files": files, "rows": rows_total, "skipped_rows": skipped}


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
