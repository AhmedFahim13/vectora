# tests/collect/test_dse_indices.py
from vectora.collect import dse_indices


def _parsed(fixtures_dir):
    html = (fixtures_dir / "homepage.html").read_text(encoding="utf-8")
    return dse_indices.parse_homepage(html)


def test_parses_three_main_indices(fixtures_dir):
    idx = _parsed(fixtures_dir)["indices"]
    names = {i["index_name"] for i in idx}
    assert {"DSEX", "DSES", "DS30"} <= names


def test_index_values_plausible(fixtures_dir):
    idx = {i["index_name"]: i for i in _parsed(fixtures_dir)["indices"]}
    dsex = idx["DSEX"]
    assert 1000 < dsex["value"] < 20000
    assert isinstance(dsex["change"], float)


def test_one_entry_per_index_and_current_day_wins(fixtures_dir):
    idx = _parsed(fixtures_dir)["indices"]
    names = [i["index_name"] for i in idx]
    assert len(names) == len(set(names))  # no duplicates
    # first occurrence (current day, +33.78998) must win over the later
    # "Preceding Trade Date" section (-11.00069)
    dsex = next(i for i in idx if i["index_name"] == "DSEX")
    assert dsex["change"] == 33.78998


def test_zero_index_is_dropped_not_replaced_by_the_prior_day():
    """old.dsebd.org sometimes renders a live index as 0.00 (seen 2026-11-03,
    one fetch in five). No index is ever really 0, and the prior-day value
    further down the page must not stand in for today's."""
    row = ('<div class="midrow"><div class="m_col-1">{0} Index</div>'
           '<div class="m_col-2">{1}</div><div class="m_col-3">{2}</div></div>')
    html = (row.format("DSEX", "0.00", "0.00") + row.format("DS30", "2016.59", "-25.5")
            + row.format("DSEX", "5333.58", "-11.0"))  # "Preceding Trade Date"
    idx = dse_indices.parse_homepage(html)["indices"]
    assert idx == [{"index_name": "DS30", "value": 2016.59, "change": -25.5}]


def test_signed_change_parsing():
    # DSMEX-style negative change (comment-wrapped in the fixture, so pinned
    # here with a synthetic midrow instead)
    html = (
        '<div class="midrow"><div class="m_col-1"><font size="+1">D</font>SME'
        '<font size="+1">X</font> Index</div><div class="m_col-2"> 1318.86324 '
        '</div><div class="m_col-3">-23.05349 </div></div>'
    )
    idx = dse_indices.parse_homepage(html)["indices"]
    assert idx == [{"index_name": "DSMEX", "value": 1318.86324, "change": -23.05349}]


def test_market_totals(fixtures_dir):
    t = _parsed(fixtures_dir)["totals"]
    assert t["total_trades"] > 10000
    assert t["total_volume"] > 1_000_000
    assert t["total_value_mn"] > 100.0


def test_empty_page():
    assert dse_indices.parse_homepage("<html></html>") == {"indices": [], "totals": None}


def test_totals_anchored_on_header_labels_not_position():
    # An Advanced/Declined/Unchanged-style numeric row BEFORE the labeled
    # totals header must not be mistaken for market totals
    html = (
        '<div class="midrow"><div class="m_col-wid">214</div>'
        '<div class="m_col-wid1">116</div><div class="m_col-wid2">62</div></div>'
        '<div class="midrow"><div class="m_col-wid">Total Trade</div>'
        '<div class="m_col-wid1">Total Volume</div>'
        '<div class="m_col-wid2">Total Value in Taka (mn)</div></div>'
        '<div class="midrow"><div class="m_col-wid">324776</div>'
        '<div class="m_col-wid1">428342460</div>'
        '<div class="m_col-wid2">14284.684</div></div>'
    )
    t = dse_indices.parse_homepage(html)["totals"]
    assert t == {"total_trades": 324776, "total_volume": 428342460,
                 "total_value_mn": 14284.684}
