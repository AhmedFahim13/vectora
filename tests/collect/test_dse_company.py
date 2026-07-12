# tests/collect/test_dse_company.py
from vectora.collect import dse_company

_FACT_KEYS = {
    "symbol",
    "paid_up_capital_mn",
    "face_value",
    "outstanding_shares",
    "instrument_type",
    "market_lot",
    "sector",
    "category",
}


def _parsed(fixtures_dir):
    html = (fixtures_dir / "company_GP.html").read_text(encoding="utf-8")
    return dse_company.parse_company(html, "GP")


def test_facts(fixtures_dir):
    facts = _parsed(fixtures_dir)["facts"]
    assert facts["symbol"] == "GP"
    assert facts["category"] == "A"
    assert facts["sector"] == "Telecommunication"
    assert facts["instrument_type"]          # non-empty, e.g. "Equity"
    assert facts["paid_up_capital_mn"] and facts["paid_up_capital_mn"] > 1000
    assert facts["outstanding_shares"] and facts["outstanding_shares"] > 1_000_000
    assert facts["face_value"] == 10.0
    assert isinstance(facts["market_lot"], int)


def test_holdings_all_blocks_sorted(fixtures_dir):
    # The GP fixture carries THREE shareholding blocks (Dec 31, 2025
    # "year ended"; May 31, 2026; Jun 30, 2026). All valid blocks are
    # returned, sorted ascending by as_of — the page is a free historical
    # time series and the holdings table PK is (symbol, as_of).
    holdings = _parsed(fixtures_dir)["holdings"]
    assert len(holdings) == 3
    assert [h["as_of"] for h in holdings] == [
        "2025-12-31", "2026-05-31", "2026-06-30",
    ]
    latest = holdings[-1]
    assert latest["sponsor_pct"] == 90.0
    assert latest["govt_pct"] == 0.0
    for k in ("institute_pct", "foreign_pct", "public_pct"):
        assert latest[k] is None or 0.0 <= latest[k] <= 100.0
    assert all(h["symbol"] == "GP" for h in holdings)


def test_holdings_sum_plausible(fixtures_dir):
    h = _parsed(fixtures_dir)["holdings"][-1]
    total = sum(v for k, v in h.items() if k.endswith("_pct") and v is not None)
    assert 99.0 < total < 101.0


def test_empty_page():
    assert dse_company.parse_company("<html></html>", "GP") == {
        "facts": {}, "holdings": [],
    }


def test_sparse_page_all_fact_keys_present():
    html = "<table><tr><td>Sector</td><td>Bank</td></tr></table>"
    parsed = dse_company.parse_company(html, "XB")
    facts = parsed["facts"]
    assert set(facts) == _FACT_KEYS
    assert facts["symbol"] == "XB"
    assert facts["sector"] == "Bank"
    for key in _FACT_KEYS - {"symbol", "sector"}:
        assert facts[key] is None
    assert parsed["holdings"] == []
