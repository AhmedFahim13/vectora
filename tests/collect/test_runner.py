# tests/collect/test_runner.py
from datetime import date

import pytest

from vectora import db as vdb
from vectora.collect import runner


class FakeSession:
    """Returns canned HTML per URL substring; records calls. No network."""

    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str, params: dict | None = None) -> str:
        self.calls.append(url)
        for key, html in self.pages.items():
            if key in url:
                return html
        raise AssertionError(f"unexpected URL {url}")


EOD_HTML = """
<table class="shares-table"><thead><tr><th>#</th></tr></thead><tbody>
<tr><td>1</td><td>2026-07-09</td><td><a href="x">GP</a></td><td>284.0</td>
<td>285.0</td><td>279.0</td><td>280.0</td><td>284.1</td><td>280.5</td>
<td>1,500</td><td>120.5</td><td>425,000</td></tr>
</tbody></table>"""

NEWS_HTML = """
<table class="table-news">
<tr><th>Trading Code:</th><td>GP</td></tr>
<tr><th>News Title:</th><td>GP: Dividend Declared</td></tr>
<tr><th>News:</th><td>10% cash dividend declared.</td></tr>
<tr><th>Post Date:</th><td>2026-07-09</td></tr>
</table>"""

HOME_HTML = """
<div class="midrow"><div class="m_col-1">DSEX Index</div>
<div class="m_col-2">5804.06</div><div class="m_col-3">33.79</div></div>
<div class="midrow"><div class="m_col-1">DS30 Index</div>
<div class="m_col-2">2177.76</div><div class="m_col-3">8.52</div></div>
<div class="midrow"><div class="m_col-wid">Total Trade</div>
<div class="m_col-wid1">Total Volume</div>
<div class="m_col-wid2">Total Value in Taka (mn)</div></div>
<div class="midrow"><div class="m_col-wid">324776</div>
<div class="m_col-wid1">428342460</div><div class="m_col-wid2">14284.684</div></div>"""

EMPTY_HTML = "<html><body>No data</body></html>"


def _session():
    return FakeSession({
        "day_end_archive": EOD_HTML,
        "old_news": NEWS_HTML,
        "dsebd.org/": HOME_HTML,
    })


def test_collect_eod_upserts_and_advances_watermark(test_db, tmp_path):
    n = runner.collect_eod(test_db, _session(), date(2026, 7, 9), raw_dir=tmp_path)
    assert n == 1
    row = test_db.execute(
        "SELECT symbol, close, volume, source FROM prices_raw").fetchone()
    assert row == ("GP", 284.1, 425000, "dse_eod")
    assert vdb.get_watermark(test_db, "collect", "eod") == "2026-07-09"


def test_collect_eod_is_idempotent(test_db, tmp_path):
    s = _session()
    runner.collect_eod(test_db, s, date(2026, 7, 9), raw_dir=tmp_path)
    runner.collect_eod(test_db, s, date(2026, 7, 9), raw_dir=tmp_path)
    assert test_db.execute("SELECT count(*) FROM prices_raw").fetchone()[0] == 1


def test_collect_eod_empty_day_records_no_trade(test_db, tmp_path):
    s = FakeSession({"day_end_archive": EMPTY_HTML})
    n = runner.collect_eod(test_db, s, date(2026, 7, 9), raw_dir=tmp_path)
    assert n == 0
    assert test_db.execute("SELECT count(*) FROM no_trade_days").fetchone()[0] == 1
    # watermark must NOT advance on an empty day
    assert vdb.get_watermark(test_db, "collect", "eod") is None


def test_collect_eod_writes_raw_file(test_db, tmp_path):
    runner.collect_eod(test_db, _session(), date(2026, 7, 9), raw_dir=tmp_path)
    saved = list(tmp_path.rglob("*.html.gz"))
    assert len(saved) == 1
    assert "dse_eod" in str(saved[0])
    meta = list(tmp_path.rglob("*.meta.json"))
    assert len(meta) == 1


def test_collect_eod_parse_crash_leaves_raw_and_no_watermark(
        test_db, tmp_path, monkeypatch):
    def boom(html):
        raise ValueError("markup changed")

    monkeypatch.setattr(runner.dse_eod, "parse_day_end", boom)
    with pytest.raises(ValueError):
        runner.collect_eod(test_db, _session(), date(2026, 7, 9), raw_dir=tmp_path)
    # raw payload landed on disk before the parser ran
    assert len(list(tmp_path.rglob("*.html.gz"))) == 1
    assert vdb.get_watermark(test_db, "collect", "eod") is None


def test_collect_news_upserts_events(test_db, tmp_path):
    n = runner.collect_news(test_db, _session(), date(2026, 7, 9), raw_dir=tmp_path)
    assert n == 1
    row = test_db.execute("SELECT symbol, title FROM events").fetchone()
    assert row == ("GP", "GP: Dividend Declared")
    assert vdb.get_watermark(test_db, "collect", "news") == "2026-07-09"


def test_collect_indices_upserts_values_and_totals(test_db, tmp_path):
    n = runner.collect_indices(test_db, _session(), date(2026, 7, 9), raw_dir=tmp_path)
    assert n == 2
    idx = test_db.execute(
        "SELECT index_name, value, change FROM indices ORDER BY index_name").fetchall()
    assert idx == [("DS30", 2177.76, 8.52), ("DSEX", 5804.06, 33.79)]
    tot = test_db.execute(
        "SELECT total_trades, total_volume, total_value_mn FROM market_totals").fetchone()
    assert tot == (324776, 428342460, 14284.684)
