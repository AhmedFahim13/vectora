# tests/test_orchestrator.py
from datetime import date

from vectora import orchestrator
from vectora.collect import runner  # noqa: F401  (fake session mirrors its contract)


class FakeSession:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages

    def get(self, url: str, params: dict | None = None) -> str:
        for key, html in self.pages.items():
            if key in url:
                return html
        raise AssertionError(f"unexpected URL {url}")


EOD_HTML = """
<table class="shares-table"><tbody>
<tr><td>1</td><td>2026-07-09</td><td><a href="x">GP</a></td><td>284.0</td>
<td>285.0</td><td>279.0</td><td>280.0</td><td>284.1</td><td>280.5</td>
<td>1,500</td><td>120.5</td><td>425,000</td></tr>
</tbody></table>"""

NEWS_HTML = """
<table class="table-news">
<tr><th>Trading Code:</th><td>GP</td></tr>
<tr><th>News Title:</th><td>GP: Dividend Declared</td></tr>
<tr><th>News:</th><td>10% cash.</td></tr>
<tr><th>Post Date:</th><td>2026-07-09</td></tr>
</table>"""

HOME_HTML = """
<div class="midrow"><div class="m_col-1">DSEX Index</div>
<div class="m_col-2">5804.06</div><div class="m_col-3">33.79</div></div>"""


def _session():
    return FakeSession({
        "day_end_archive": EOD_HTML,
        "old_news": NEWS_HTML,
        "dsebd.org/": HOME_HTML,
    })


def test_run_eod_full_cycle(test_db, tmp_path):
    summary = orchestrator.run_eod(
        test_db, _session(), date(2026, 7, 9), raw_dir=tmp_path, holidays=set())
    assert summary == {
        "date": "2026-07-09",
        "skipped": None,
        "eod_rows": 1,
        "news_items": 1,
        "index_rows": 1,
        "quality_score": 100,
    }
    # all four stages actually wrote
    for table in ("prices_raw", "events", "indices", "data_quality"):
        assert test_db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] >= 1


def test_run_eod_skips_non_trading_day(test_db, tmp_path):
    summary = orchestrator.run_eod(
        test_db, _session(), date(2026, 7, 10), raw_dir=tmp_path, holidays=set())
    assert summary["skipped"] == "not a trading day"
    assert test_db.execute("SELECT count(*) FROM prices_raw").fetchone()[0] == 0


def test_run_eod_continues_past_stage_failure(test_db, tmp_path, monkeypatch):
    # news blows up; EOD and indices must still land, failure reported
    def boom(con, session, run_date, raw_dir):
        raise ValueError("news markup changed")

    monkeypatch.setattr(orchestrator.runner, "collect_news", boom)
    summary = orchestrator.run_eod(
        test_db, _session(), date(2026, 7, 9), raw_dir=tmp_path, holidays=set())
    assert summary["eod_rows"] == 1
    assert summary["news_items"] is None
    assert "news" in summary["errors"][0]
    assert test_db.execute("SELECT count(*) FROM prices_raw").fetchone()[0] == 1


def test_dates_to_run_gap_fills_from_watermark():
    # watermark Sunday 07-05, target Thursday 07-09 -> Mon/Tue/Wed/Thu
    dates = orchestrator._dates_to_run(
        "2026-07-05", date(2026, 7, 9), holidays=set(), explicit=False)
    assert [d.isoformat() for d in dates] == [
        "2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"]


def test_dates_to_run_skips_weekend_and_holiday():
    dates = orchestrator._dates_to_run(
        "2026-07-08", date(2026, 7, 12),  # Thu watermark -> Sun target
        holidays={date(2026, 7, 12)}, explicit=False)
    assert dates == [date(2026, 7, 9)]  # Fri/Sat weekend, Sun holiday


def test_dates_to_run_nothing_new():
    assert orchestrator._dates_to_run(
        "2026-07-09", date(2026, 7, 9), holidays=set(), explicit=False) == []


def test_dates_to_run_explicit_date_bypasses_gap_fill():
    assert orchestrator._dates_to_run(
        "2026-07-05", date(2026, 7, 9), holidays=set(), explicit=True
    ) == [date(2026, 7, 9)]


def _summary(**overrides):
    base = {"date": "2026-07-09", "skipped": None, "quality_score": 100,
            "eod_rows": 5, "news_items": 2, "index_rows": 3}
    return {**base, **overrides}


def test_cli_dispatch(monkeypatch, capsys):
    calls = {}

    def fake_run(date_str):
        calls["date"] = date_str
        return [_summary(date=date_str or "auto")]

    monkeypatch.setattr(orchestrator, "run_eod_live", fake_run)
    from vectora.__main__ import main
    rc = main(["run", "eod", "--date", "2026-07-09"])
    assert rc == 0
    assert calls["date"] == "2026-07-09"
    assert "2026-07-09" in capsys.readouterr().out


def test_cli_fails_on_unusable_day(monkeypatch):
    monkeypatch.setattr(orchestrator, "run_eod_live",
                        lambda d: [_summary(quality_score=0, eod_rows=0,
                                            errors=["eod: no rows"])])
    from vectora.__main__ import main
    assert main(["run", "eod", "--date", "2026-07-09"]) == 1


def test_cli_fails_below_quality_floor(monkeypatch):
    monkeypatch.setattr(orchestrator, "run_eod_live",
                        lambda d: [_summary(quality_score=79)])
    from vectora.__main__ import main
    assert main(["run", "eod", "--date", "2026-07-09"]) == 1


def test_cli_fails_on_stage_errors_even_with_good_quality(monkeypatch):
    monkeypatch.setattr(orchestrator, "run_eod_live",
                        lambda d: [_summary(errors=["news: markup changed"])])
    from vectora.__main__ import main
    assert main(["run", "eod", "--date", "2026-07-09"]) == 1


def test_cli_fails_when_validation_crashed(monkeypatch):
    monkeypatch.setattr(orchestrator, "run_eod_live",
                        lambda d: [_summary(quality_score=None)])
    from vectora.__main__ import main
    assert main(["run", "eod", "--date", "2026-07-09"]) == 1


def test_cli_ok_on_clean_skip_and_worst_run_wins(monkeypatch):
    monkeypatch.setattr(orchestrator, "run_eod_live",
                        lambda d: [_summary(skipped="not a trading day",
                                            quality_score=None)])
    from vectora.__main__ import main
    assert main(["run", "eod"]) == 0

    monkeypatch.setattr(orchestrator, "run_eod_live",
                        lambda d: [_summary(), _summary(quality_score=40)])
    assert main(["run", "eod"]) == 1
