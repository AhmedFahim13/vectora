"""Fundamental overlay screens."""
from vectora.ta import overlay


def _entry(symbol="X", band="Buy", **fund) -> dict:
    return {"symbol": symbol, "band": band, "fundamentals": fund}


def test_value_screen_requires_positive_earnings():
    cheap = overlay._screen(_entry(trailing_pe=8.0))
    assert {s["screen"] for s in cheap} == {"Value"}
    # a negative P/E is a loss, not a bargain
    assert overlay._screen(_entry(trailing_pe=-4.0)) == []
    assert overlay._screen(_entry(trailing_pe=40.0)) == []


def test_income_screen_uses_cash_yield():
    flags = overlay._screen(_entry(dividend_yield=0.062))
    assert flags[0]["screen"] == "Income"
    assert "6.2%" in flags[0]["detail"]
    assert overlay._screen(_entry(dividend_yield=0.01)) == []


def test_quality_and_impairment_are_distinct():
    strong = overlay._screen(_entry(reserve_surplus_mn=5000.0,
                                    paid_up_capital_mn=1000.0))
    assert strong[0]["screen"] == "Quality"
    broke = overlay._screen(_entry(reserve_surplus_mn=-41405.0,
                                   paid_up_capital_mn=1000.0))
    assert broke[0]["screen"] == "Impaired"


def test_quality_bar_is_above_the_exchange_median():
    """Reserves merely exceeding capital is the DSE median (1.08x) — half
    the board. A screen that selects half of everything selects nothing."""
    median_ish = overlay._screen(_entry(reserve_surplus_mn=1100.0,
                                        paid_up_capital_mn=1000.0))
    assert median_ish == []
    top_quartile = overlay._screen(_entry(reserve_surplus_mn=3200.0,
                                          paid_up_capital_mn=1000.0))
    assert top_quartile[0]["screen"] == "Quality"


def test_thin_float_is_reported_as_risk():
    flags = overlay._screen(_entry(free_float_mcap_mn=100.0,
                                   market_cap_mn=1000.0))
    assert flags[0]["screen"] == "Thin float"
    assert "trap" in flags[0]["detail"]


def test_missing_fundamentals_produce_no_flags():
    assert overlay._screen({"symbol": "X"}) == []
    assert overlay._screen(_entry()) == []


def test_confluence_requires_bullish_plus_support():
    entries = overlay.annotate(
        [{"symbol": "GOOD", "band": "Strong Buy"},
         {"symbol": "CHEAPBEAR", "band": "Sell"},
         {"symbol": "PRICEY", "band": "Strong Buy"}],
        {"GOOD": {"trailing_pe": 7.0}, "CHEAPBEAR": {"trailing_pe": 7.0},
         "PRICEY": {"trailing_pe": 90.0}})
    picked = [e["symbol"] for e in overlay.confluence(entries, {})]
    assert picked == ["GOOD"]


def test_confluence_excludes_impaired_and_thin_float():
    """Technically strong and cheap, but hollow underneath — not a pick."""
    entries = overlay.annotate(
        [{"symbol": "TRAP", "band": "Strong Buy"}],
        {"TRAP": {"trailing_pe": 6.0, "free_float_mcap_mn": 50.0,
                  "market_cap_mn": 1000.0}})
    assert overlay.confluence(entries, {}) == []


def test_confluence_accepts_the_gauge_verdict_too():
    entries = overlay.annotate([{"symbol": "G", "band": "Hold"}],
                               {"G": {"trailing_pe": 9.0}})
    assert overlay.confluence(entries, {}) == []
    assert [e["symbol"] for e in
            overlay.confluence(entries, {"G": {"summary_band": "Strong Buy"}})
            ] == ["G"]


def test_counts_tallies_each_screen():
    entries = overlay.annotate(
        [{"symbol": "A"}, {"symbol": "B"}],
        {"A": {"trailing_pe": 5.0, "dividend_yield": 0.08},
         "B": {"trailing_pe": 6.0}})
    assert overlay.counts(entries) == {"Value": 2, "Income": 1}


def test_unreported_free_float_is_not_a_thin_float_flag():
    """DSE reports free float 0 for all 35 mutual funds and 22 bonds — that
    means 'not applicable', not 'nothing can trade'. Flagging them produced
    the line 'only 0% of the company can trade'."""
    assert overlay._screen(_entry(free_float_mcap_mn=0.0,
                                  market_cap_mn=1130.7)) == []
    assert overlay._screen(_entry(free_float_mcap_mn=None,
                                  market_cap_mn=1130.7)) == []
    # a genuinely thin equity still flags
    flags = overlay._screen(_entry(free_float_mcap_mn=35038.0,
                                   market_cap_mn=350133.0))
    assert flags[0]["screen"] == "Thin float"
    assert "10%" in flags[0]["detail"]
