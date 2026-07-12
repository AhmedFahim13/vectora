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
