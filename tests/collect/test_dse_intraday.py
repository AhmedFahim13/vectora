# tests/collect/test_dse_intraday.py
from vectora.collect import dse_intraday


def _rows(fixtures_dir):
    html = (fixtures_dir / "latest_share_price.html").read_text(encoding="utf-8")
    return dse_intraday.parse_latest(html)


def test_parses_many_rows(fixtures_dir):
    assert len(_rows(fixtures_dir)) > 300


def test_row_shape(fixtures_dir):
    r = _rows(fixtures_dir)[0]
    assert set(r) == {"symbol", "ltp", "high", "low", "closep", "ycp",
                      "change", "trades", "value_mn", "volume"}
    assert isinstance(r["symbol"], str) and r["symbol"]
    assert r["volume"] is None or isinstance(r["volume"], int)


def test_empty_page_returns_empty():
    assert dse_intraday.parse_latest("<html></html>") == []


def test_collect_upserts_snapshots(test_db, fixtures_dir, tmp_path):
    html = (fixtures_dir / "latest_share_price.html").read_text(encoding="utf-8")

    class FakeSession:
        def get(self, url, params=None):
            return html

    n = dse_intraday.collect_intraday(
        test_db, FakeSession(), ts="2026-07-17 12:00:00", raw_dir=tmp_path)
    assert n > 300
    stored = test_db.execute(
        "SELECT count(*) FROM intraday_snapshots").fetchone()[0]
    assert stored == n
    # idempotent for the same ts
    dse_intraday.collect_intraday(
        test_db, FakeSession(), ts="2026-07-17 12:00:00", raw_dir=tmp_path)
    assert test_db.execute(
        "SELECT count(*) FROM intraday_snapshots").fetchone()[0] == n
    assert len(list(tmp_path.rglob("*.html.gz"))) == 1
