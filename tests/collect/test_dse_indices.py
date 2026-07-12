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


def test_negative_change_preserved(fixtures_dir):
    idx = {i["index_name"]: i for i in _parsed(fixtures_dir)["indices"]}
    assert any(i["change"] < 0 for i in idx.values())  # DSMEX is negative in fixture


def test_market_totals(fixtures_dir):
    t = _parsed(fixtures_dir)["totals"]
    assert t["total_trades"] > 10000
    assert t["total_volume"] > 1_000_000
    assert t["total_value_mn"] > 100.0


def test_empty_page():
    assert dse_indices.parse_homepage("<html></html>") == {"indices": [], "totals": None}
