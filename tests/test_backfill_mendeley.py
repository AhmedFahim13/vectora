# tests/test_backfill_mendeley.py
import zipfile

from tools import backfill_mendeley as bf

GP_CSV = """Date,Open,High,Low,Close,Volume
2013-01-02,180.5,182.0,179.0,181.2,250000
2013-01-03,181.0,181.5,178.5,179.0,310000
"""

BAD_ROW_CSV = """Date,Open,High,Low,Close,Volume
2013-01-02,10.0,10.5,9.5,10.2,1000
not-a-date,1,1,1,1,1
2013-01-03,,,,10.4,900
"""


def _zip(tmp_path, entries: dict[str, str]):
    p = tmp_path / "dse.zip"
    with zipfile.ZipFile(p, "w") as z:
        for name, content in entries.items():
            z.writestr(name, content)
    return p


def test_loads_unadjusted_csvs(test_db, tmp_path):
    z = _zip(tmp_path, {
        "Unadjusted Data/GP.csv": GP_CSV,
        "Adjusted Data/GP.csv": GP_CSV,  # must be ignored
    })
    result = bf.load_zip(test_db, z)
    assert result == {"files": 1, "rows": 2, "skipped_rows": 0}
    rows = test_db.execute(
        "SELECT symbol, date, close, volume, source FROM prices_raw ORDER BY date"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "GP" and str(rows[0][1]) == "2013-01-02"
    assert rows[0][2] == 181.2 and rows[0][3] == 250000
    assert rows[0][4] == "mendeley"


def test_flat_zip_without_folders_loads_everything(test_db, tmp_path):
    z = _zip(tmp_path, {"ACI.csv": GP_CSV})
    result = bf.load_zip(test_db, z)
    assert result["files"] == 1 and result["rows"] == 2
    sym = test_db.execute("SELECT DISTINCT symbol FROM prices_raw").fetchone()[0]
    assert sym == "ACI"


def test_bad_rows_skipped_not_fatal(test_db, tmp_path):
    z = _zip(tmp_path, {"XYZ.csv": BAD_ROW_CSV})
    result = bf.load_zip(test_db, z)
    assert result["rows"] == 2           # missing-OHLC row still loads close
    assert result["skipped_rows"] == 1   # unparseable date dropped
    close = test_db.execute(
        "SELECT close FROM prices_raw WHERE date = '2013-01-03'").fetchone()[0]
    assert close == 10.4


def test_idempotent_reload(test_db, tmp_path):
    z = _zip(tmp_path, {"GP.csv": GP_CSV})
    bf.load_zip(test_db, z)
    bf.load_zip(test_db, z)
    assert test_db.execute("SELECT count(*) FROM prices_raw").fetchone()[0] == 2


def test_does_not_clobber_dse_eod_source(test_db, tmp_path):
    from vectora import db as vdb
    vdb.upsert(test_db, "prices_raw", [dict(
        symbol="GP", date="2013-01-02", open=1.0, high=1.0, low=1.0, close=999.0,
        ltp=1.0, ycp=1.0, trades=1, value_mn=1.0, volume=1, source="dse_eod")])
    bf.load_zip(test_db, _zip(tmp_path, {"GP.csv": GP_CSV}))
    eod = test_db.execute(
        "SELECT close FROM prices_raw WHERE source = 'dse_eod'").fetchone()[0]
    assert eod == 999.0  # separate source rows coexist (PK symbol,date,source)
