from vectora import db as vdb
from vectora.universe import tradable_universe


def _seed(con):
    vdb.upsert(con, "symbols", [
        dict(symbol="GP", name=None, sector="Telecommunication",
             instrument_type="Equity", category="A", listing_status="active",
             first_seen="2013-01-01", last_seen="2026-07-09"),
        dict(symbol="ILLIQ", name=None, sector="Bank",
             instrument_type="Equity", category="B", listing_status="active",
             first_seen="2013-01-01", last_seen="2026-07-09"),
        dict(symbol="TBOND1", name=None, sector="Govt Bond",
             instrument_type="Bond", category=None, listing_status="active",
             first_seen="2013-01-01", last_seen="2026-07-09"),
    ])
    rows = []
    for i in range(60):
        d = f"2026-{4 + i // 28:02d}-{i % 28 + 1:02d}"
        rows.append(dict(symbol="GP", date=d, open=10, high=10, low=10, close=10,
                         ltp=10, ycp=10, trades=100, value_mn=25.0, volume=9000,
                         source="dse_eod"))
        rows.append(dict(symbol="ILLIQ", date=d, open=5, high=5, low=5, close=5,
                         ltp=5, ycp=5, trades=2, value_mn=0.05, volume=100,
                         source="dse_eod"))
        rows.append(dict(symbol="TBOND1", date=d, open=100, high=100, low=100,
                         close=100, ltp=100, ycp=100, trades=50, value_mn=50.0,
                         volume=5000, source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)


def test_universe_keeps_liquid_equities_only(test_db):
    _seed(test_db)
    u = tradable_universe(test_db, as_of="2026-06-28", min_median_value_mn=1.0)
    assert "GP" in u
    assert "ILLIQ" not in u       # fails liquidity floor
    assert "TBOND1" not in u      # not an Equity


def test_universe_liquidity_floor_configurable(test_db):
    _seed(test_db)
    u = tradable_universe(test_db, as_of="2026-06-28", min_median_value_mn=0.01)
    assert {"GP", "ILLIQ"} <= set(u)
