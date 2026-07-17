# tests/evaluate/test_report.py
from vectora import db as vdb
from vectora.evaluate import report


def _pred(symbol, d, prob, target="g5_h10"):
    return dict(id=f"{d}_{target}_{symbol}", symbol=symbol, date=d,
                target=target, probability=prob, model_id="m",
                quality_score=100, is_signal=prob >= 0.55,
                suppressed_reason=None)


def _outcome(pid, hit, rmax=0.06, rmin=-0.02):
    return dict(prediction_id=pid, realized_max=rmax, realized_min=rmin,
                hit=hit)


def _seed(con):
    vdb.upsert(con, "symbols", [
        dict(symbol=s, name=None, sector="Bank", instrument_type="Equity",
             category=c, listing_status="active", first_seen="2020-01-01",
             last_seen="2026-12-31")
        for s, c in (("AAA", "A"), ("ZZZ", "Z"))])
    vdb.upsert(con, "regimes", [dict(
        date="2026-06-01", regime="Sideways", confidence=0.5, method="rules")])
    vdb.upsert(con, "predictions", [
        _pred("AAA", "2026-06-01", 0.70), _pred("ZZZ", "2026-06-01", 0.60),
        _pred("AAA", "2026-06-02", 0.30)])
    vdb.upsert(con, "outcomes", [
        _outcome("2026-06-01_g5_h10_AAA", True),
        _outcome("2026-06-01_g5_h10_ZZZ", False, rmax=0.01),
        _outcome("2026-06-02_g5_h10_AAA", False, rmax=0.02)])
    vdb.upsert(con, "risk_blocks", [dict(
        prediction_id="2026-06-01_g5_h10_ZZZ", vol_21d=0.02, expected_up=0.05,
        expected_down=-0.03, rr_ratio=1.6, exit_days=9.0,
        analog_max_drawdown=-0.1, analog_hit_rate=0.5, analog_n=20,
        category="Z", liquidity_value_mn=0.3)])


def test_evaluate_metrics_and_report(test_db, tmp_path):
    _seed(test_db)
    result = report.evaluate(test_db, reports_dir=tmp_path,
                             vault_dir=tmp_path / "vault", seg_min=2)
    assert result["resolved"] == 3
    assert 0 < result["targets"]["g5_h10"]["brier"] < 1
    assert abs(result["targets"]["g5_h10"]["hit_rate"] - 1 / 3) < 1e-9
    files = list(tmp_path.glob("eval_*.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert "g5_h10" in text and "Brier" in text and "Sideways" in text
    assert (tmp_path / "vault" / "Evaluations").exists()


def test_autopsy_tags_misses(test_db, tmp_path):
    _seed(test_db)
    # ZZZ miss: exit_days 9 -> liquidity tag; AAA 06-02 miss: no risk row,
    # no event, no regime shift -> model-error
    vdb.upsert(test_db, "events", [dict(
        id="evx", post_date="2026-06-03", symbol="AAA",
        title="AAA: Q1 Financials", body="", source="dse_news")])
    vdb.upsert(test_db, "event_labels", [dict(
        event_id="evx", event_type="earnings_release", materiality=3)])
    report.evaluate(test_db, reports_dir=tmp_path,
                    vault_dir=tmp_path / "vault")
    tags = dict(test_db.execute(
        "SELECT prediction_id, tag FROM outcome_tags").fetchall())
    assert tags["2026-06-01_g5_h10_ZZZ"] == "liquidity"
    assert tags["2026-06-02_g5_h10_AAA"] == "event-shock"  # event inside window


def test_no_outcomes_graceful(test_db, tmp_path):
    result = report.evaluate(test_db, reports_dir=tmp_path,
                             vault_dir=tmp_path / "vault", seg_min=2)
    assert result == {"resolved": 0}
