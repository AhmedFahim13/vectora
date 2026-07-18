# tests/test_dashboard.py
from vectora import dashboard
from vectora import db as vdb


def _seed(con):
    vdb.upsert(con, "symbols", [dict(
        symbol="GP", name=None, sector="Telecom", instrument_type="Equity",
        category="A", listing_status="active", first_seen="2013-01-01",
        last_seen="2026-07-16")])
    vdb.upsert(con, "data_quality", [dict(
        date="2026-07-16", source="dse_eod", score=100, issues="[]")])
    vdb.upsert(con, "regimes", [dict(
        date="2026-07-16", regime="Sideways", confidence=0.5, method="rules")])
    vdb.upsert(con, "model_registry", [dict(
        model_id="m", family="lgbm", target="g5_h10",
        trained_at="2026-07-17T00:00:00", train_end="2026-05-01",
        metrics='{"brier": 0.2035}', artifact_dir="models/m", active=True)])
    vdb.upsert(con, "predictions", [
        dict(id="2026-07-16_g5_h10_GP", symbol="GP", date="2026-07-16",
             target="g5_h10", probability=0.61, model_id="m",
             quality_score=100, is_signal=True, suppressed_reason=None),
        dict(id="2026-07-16_g5_h10_ACI", symbol="ACI", date="2026-07-16",
             target="g5_h10", probability=0.30, model_id="m",
             quality_score=100, is_signal=False,
             suppressed_reason="below-probability-threshold")])
    vdb.upsert(con, "risk_blocks", [dict(
        prediction_id="2026-07-16_g5_h10_GP", vol_21d=0.02, expected_up=0.08,
        expected_down=-0.03, rr_ratio=2.6, exit_days=1.2,
        analog_max_drawdown=-0.1, analog_hit_rate=0.6, analog_n=20,
        category="A", liquidity_value_mn=5.0)])
    vdb.upsert(con, "explanations", [dict(
        prediction_id="2026-07-16_g5_h10_GP", drivers="[]", analogs="{}",
        rendered="GP: 61% calibrated probability of the g5_h10 move.")])
    vdb.upsert(con, "zwatch", [dict(
        date="2026-07-16", symbol="ATLASBANG", kind="pump", score=100.0,
        phase="markup", detail="{}")])
    vdb.upsert(con, "event_studies", [dict(
        event_type="category_change", horizon=10, n=174, mean_abn_ret=-0.024,
        median_abn_ret=-0.016, pos_share=0.39)])


def test_dashboard_renders_core_content(test_db):
    _seed(test_db)
    html = dashboard.build_html(test_db, "2026-07-16")
    assert html.startswith("<!doctype html>")
    assert "PAPER" in html and "not investment advice" in html
    assert "2026-07-30" in html               # track-record banner
    assert "Sideways" in html                  # regime
    assert "GP" in html and "61" in html       # top signal + prob
    assert "ATLASBANG" in html                 # z-watch
    assert "category_change" in html           # event study
    assert "0.203" in html or "0.204" in html  # model Brier


def test_dashboard_quiet_market_shows_top_setups(test_db):
    _seed(test_db)
    # demote the only signal so 0 signals remain
    test_db.execute("UPDATE predictions SET is_signal=false")
    html = dashboard.build_html(test_db, "2026-07-16")
    assert "Top-ranked setups" in html
    assert "No setup cleared the signal bar" in html


def test_dashboard_is_self_contained(test_db):
    _seed(test_db)
    html = dashboard.build_html(test_db, "2026-07-16")
    # no external asset references (works offline / on a static host)
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html   # no JS needed for the static view
