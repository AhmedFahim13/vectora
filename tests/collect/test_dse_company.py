# tests/collect/test_dse_company.py
from vectora.collect import dse_company


def _parsed(fixtures_dir):
    html = (fixtures_dir / "company_GP.html").read_text(encoding="utf-8")
    return dse_company.parse_company(html)


def test_facts(fixtures_dir):
    facts = _parsed(fixtures_dir)["facts"]
    assert facts["category"] == "A"
    assert facts["sector"] == "Telecommunication"
    assert facts["instrument_type"]          # non-empty, e.g. "Equity"
    assert facts["paid_up_capital_mn"] and facts["paid_up_capital_mn"] > 1000
    assert facts["outstanding_shares"] and facts["outstanding_shares"] > 1_000_000
    assert facts["face_value"] == 10.0
    assert isinstance(facts["market_lot"], int)


def test_holdings_latest_block_wins(fixtures_dir):
    # NOTE: the fixture actually contains THREE shareholding blocks (Dec 31,
    # 2025 "year ended"; May 31, 2026; Jun 30, 2026) rather than the two
    # originally assumed in the task spec. Jun 30, 2026 is the genuinely
    # latest parseable block, so that's the one that must win here.
    h = _parsed(fixtures_dir)["holdings"]
    assert h["as_of"] == "2026-06-30"      # latest block, not year-end or mid-2026
    assert h["sponsor_pct"] == 90.0
    assert h["govt_pct"] == 0.0
    for k in ("institute_pct", "foreign_pct", "public_pct"):
        assert h[k] is None or 0.0 <= h[k] <= 100.0


def test_holdings_sum_plausible(fixtures_dir):
    h = _parsed(fixtures_dir)["holdings"]
    total = sum(v for k, v in h.items() if k.endswith("_pct") and v is not None)
    assert 99.0 < total < 101.0


def test_empty_page():
    assert dse_company.parse_company("<html></html>") == {"facts": {}, "holdings": None}
