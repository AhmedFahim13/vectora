import polars as pl

from vectora.backtest import runner


def test_backtest_report_is_named_by_run_date(test_db, tmp_path, monkeypatch):
    """A replayed Friday files under its own date, not the day it was replayed."""
    monkeypatch.setattr(runner, "load_panel", lambda con: pl.DataFrame())
    monkeypatch.setattr(runner, "load_entries",
                        lambda con, min_prob: pl.DataFrame({"symbol": ["GP"]}))
    monkeypatch.setattr(runner.simulator, "run", lambda *a, **k: None)
    monkeypatch.setattr(runner.metrics, "summarize", lambda res: {})
    monkeypatch.setattr(runner.metrics, "render", lambda name, m: name)
    out = runner.run(test_db, reports_dir=tmp_path, run_date="2026-10-02")
    path = tmp_path / "backtest_2026-10-02.md"
    assert out["_report"] == str(path)
    assert path.read_text(encoding="utf-8").startswith("# Backtest 2026-10-02")
