# tests/collect/test_dse_eod.py
from vectora.collect import dse_eod


def _rows(fixtures_dir):
    html = (fixtures_dir / "day_end_archive.html").read_text(encoding="utf-8")
    return dse_eod.parse_day_end(html)


def test_parses_many_rows(fixtures_dir):
    rows = _rows(fixtures_dir)
    assert len(rows) > 300  # all instruments incl. funds/bonds ~850


def test_row_shape_and_types(fixtures_dir):
    r = _rows(fixtures_dir)[0]
    assert set(r) == {"symbol", "date", "ltp", "high", "low", "open", "close",
                      "ycp", "trades", "value_mn", "volume"}
    assert isinstance(r["symbol"], str) and r["symbol"] == r["symbol"].strip()
    assert len(r["date"]) == 10 and r["date"][4] == "-"
    assert isinstance(r["volume"], int)          # comma-formatted in HTML
    assert isinstance(r["trades"], int)
    assert r["high"] is None or r["high"] >= 0


def test_all_symbols_unique_per_date(fixtures_dir):
    rows = _rows(fixtures_dir)
    keys = [(r["symbol"], r["date"]) for r in rows]
    assert len(keys) == len(set(keys))
