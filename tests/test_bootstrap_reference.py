# tests/test_bootstrap_reference.py
from tools import bootstrap_reference as boot
from vectora import db as vdb

COMPANY_HTML = """
<table>
<tr><td>Sector</td><td>Telecommunication</td></tr>
<tr><td>Market Category</td><td>A</td></tr>
<tr><th>Type of Instrument</th><td>Equity</td></tr>
<tr><th>Paid-up Capital (mn)</th><td>13,503.00</td></tr>
<tr><th>Total No. of Outstanding Securities</th><td>1,350,300,022</td></tr>
<tr><th>Face/par Value</th><td>10.0</td></tr>
<tr><th>Market Lot</th><td>1</td></tr>
<tr><td>Share Holding Percentage [as on Jun 30, 2026]</td>
<td>Sponsor/Director: 90.00 Govt: 0.00 Institute: 6.76 Foreign: 0.33 Public: 2.91</td></tr>
</table>"""


class FakeSession:
    def __init__(self, html=COMPANY_HTML, fail_for=()):
        self.html = html
        self.fail_for = set(fail_for)
        self.fetched: list[str] = []

    def get(self, url, params=None):
        sym = params["name"]
        self.fetched.append(sym)
        if sym in self.fail_for:
            raise ConnectionError(f"boom {sym}")
        return self.html


def _seed_prices(con):
    rows = [dict(symbol=s, date=d, open=1.0, high=1.0, low=1.0, close=1.0, ltp=1.0,
                 ycp=1.0, trades=1, value_mn=1.0, volume=1, source="dse_eod")
            for s in ("GP", "BEXIMCO") for d in ("2026-07-08", "2026-07-09")]
    vdb.upsert(con, "prices_raw", rows)


def test_bootstrap_symbols_from_prices(test_db):
    _seed_prices(test_db)
    n = boot.bootstrap_symbols(test_db)
    assert n == 2
    rows = test_db.execute(
        "SELECT symbol, first_seen, last_seen FROM symbols ORDER BY symbol").fetchall()
    assert [r[0] for r in rows] == ["BEXIMCO", "GP"]
    assert str(rows[0][1]) == "2026-07-08" and str(rows[0][2]) == "2026-07-09"


def test_rebootstrap_preserves_swept_metadata(test_db):
    _seed_prices(test_db)
    boot.bootstrap_symbols(test_db)
    test_db.execute(
        "UPDATE symbols SET sector = 'Bank', category = 'A' WHERE symbol = 'GP'")
    # new history arrives (backfill) -> re-bootstrap must extend dates,
    # not clobber sector/category with NULLs
    vdb.upsert(test_db, "prices_raw", [dict(
        symbol="GP", date="2013-01-02", open=1.0, high=1.0, low=1.0, close=1.0,
        ltp=None, ycp=None, trades=None, value_mn=None, volume=1,
        source="mendeley")])
    boot.bootstrap_symbols(test_db)
    row = test_db.execute(
        "SELECT sector, category, first_seen, last_seen FROM symbols "
        "WHERE symbol = 'GP'").fetchone()
    assert row[0] == "Bank" and row[1] == "A"
    assert str(row[2]) == "2013-01-02" and str(row[3]) == "2026-07-09"


def test_sweep_upserts_snapshot_holdings_and_symbol_meta(test_db, tmp_path):
    _seed_prices(test_db)
    boot.bootstrap_symbols(test_db)
    result = boot.sweep_companies(
        test_db, FakeSession(), ["GP"], as_of="2026-07-12", raw_dir=tmp_path)
    assert result == {"swept": 1, "failed": []}
    snap = test_db.execute(
        "SELECT sector, category, paid_up_capital_mn FROM company_snapshot").fetchone()
    assert snap == ("Telecommunication", "A", 13503.0)
    h = test_db.execute(
        "SELECT sponsor_pct, public_pct FROM holdings WHERE symbol='GP'").fetchone()
    assert h == (90.0, 2.91)
    sym = test_db.execute(
        "SELECT sector, category FROM symbols WHERE symbol='GP'").fetchone()
    assert sym == ("Telecommunication", "A")
    assert len(list(tmp_path.rglob("*.html.gz"))) == 1


def test_sweep_continues_past_failures(test_db, tmp_path):
    _seed_prices(test_db)
    boot.bootstrap_symbols(test_db)
    result = boot.sweep_companies(
        test_db, FakeSession(fail_for={"GP"}), ["GP", "BEXIMCO"],
        as_of="2026-07-12", raw_dir=tmp_path)
    assert result["swept"] == 1
    assert result["failed"] == ["GP"]
    assert test_db.execute("SELECT count(*) FROM company_snapshot").fetchone()[0] == 1
