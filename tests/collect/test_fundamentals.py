# tests/collect/test_fundamentals.py
from vectora.collect import dse_company as dc


def _html(fixtures_dir):
    return (fixtures_dir / "company_GP.html").read_text(encoding="utf-8")


def test_extracts_headline_fundamentals(fixtures_dir):
    f = dc.parse_fundamentals(_html(fixtures_dir), "GP")
    assert f["symbol"] == "GP"
    assert abs(f["market_cap_mn"] - 350267.826) < 0.01
    assert abs(f["free_float_mcap_mn"] - 35051.544) < 0.01
    assert abs(f["trailing_pe"] - 11.68) < 0.01
    assert abs(f["reserve_surplus_mn"] - 35167.4) < 0.1
    assert f["listing_year"] == 2009
    assert f["year_end"] == "31-Dec"


def test_dividend_parsed_with_year(fixtures_dir):
    f = dc.parse_fundamentals(_html(fixtures_dir), "GP")
    assert abs(f["latest_dividend_pct"] - 215.0) < 0.01
    assert f["dividend_year"] == 2025


def test_derived_metrics_use_dse_conventions(fixtures_dir):
    """DSE quotes dividends as % of FACE value (10 tk), not of price."""
    f = dc.parse_fundamentals(_html(fixtures_dir), "GP")
    d = dc.derive_metrics(f, close=256.70)
    # 215% of 10 tk face = 21.50 tk per share
    assert abs(d["dividend_per_share"] - 21.50) < 0.01
    assert abs(d["dividend_yield"] - 21.50 / 256.70) < 1e-6
    # trailing EPS implied by price / trailing P/E
    assert abs(d["eps_trailing"] - 256.70 / 11.68) < 0.01


def test_missing_fields_are_none_not_crash():
    f = dc.parse_fundamentals("<html><body>nothing</body></html>", "XYZ")
    assert f["symbol"] == "XYZ"
    assert f["market_cap_mn"] is None and f["trailing_pe"] is None
    d = dc.derive_metrics(f, close=None)
    assert d["dividend_yield"] is None and d["eps_trailing"] is None
