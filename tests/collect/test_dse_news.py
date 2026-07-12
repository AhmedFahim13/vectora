# tests/collect/test_dse_news.py
from vectora.collect import dse_news


def _items(fixtures_dir):
    html = (fixtures_dir / "old_news.html").read_text(encoding="utf-8")
    return dse_news.parse_news(html)


def test_parses_many_items(fixtures_dir):
    # exact pin against the checked-in fixture; update if fixtures are re-recorded
    assert len(_items(fixtures_dir)) == 57


def test_empty_page_yields_no_items():
    assert dse_news.parse_news("<html></html>") == []


def test_item_shape(fixtures_dir):
    it = _items(fixtures_dir)[0]
    assert set(it) == {"id", "post_date", "symbol", "title", "body", "source"}
    assert it["source"] == "dse_news"
    assert len(it["id"]) == 64  # sha256 hex
    assert len(it["post_date"]) == 10


def test_symbol_extraction():
    assert dse_news.symbol_from_title("MERCANBANK: Credit Rating Result") == "MERCANBANK"
    assert dse_news.symbol_from_title("DSE NEWS: Daily Turnover of Main Board") is None
    assert dse_news.symbol_from_title("No colon here") is None
    assert dse_news.symbol_from_title("lowercase: not a ticker") is None
    assert dse_news.symbol_from_title("DSENEWS: Withdrawal of Authorized Representatives") is None


def test_ids_are_stable_and_unique(fixtures_dir):
    a = _items(fixtures_dir)
    b = _items(fixtures_dir)
    assert [x["id"] for x in a] == [x["id"] for x in b]  # deterministic
    assert len({x["id"] for x in a}) == len(a)           # unique in fixture


def test_company_news_has_symbol(fixtures_dir):
    items = _items(fixtures_dir)
    with_symbol = [i for i in items if i["symbol"]]
    assert len(with_symbol) > 10


def test_symbol_comes_from_trading_code(fixtures_dir):
    items = _items(fixtures_dir)
    # synthetic bucket codes and title-regex leaks must never surface as symbols
    assert not any(i["symbol"] in {"DSENEWS", "EXCH", "REGL"} for i in items)
    # Trading Code recovers company-scoped "DSE NEWS:" withdrawal notices
    # (OSL/KDS/PIS/ASE/IBB) that the title regex misses
    with_symbol = [i for i in items if i["symbol"]]
    assert len(with_symbol) == 49  # title-regex baseline was 45; Trading Code adds 4
    withdrawals = [i for i in items if i["title"].startswith("DSE NEWS: Withdrawal")]
    assert any(i["symbol"] is not None for i in withdrawals)
