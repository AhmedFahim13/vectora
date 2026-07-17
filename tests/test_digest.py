# tests/test_digest.py
from vectora import db as vdb
from vectora.alerts import digest


def _seed(con):
    vdb.upsert(con, "data_quality", [
        {"date": "2026-07-16", "source": "dse_eod", "score": 100, "issues": "[]"}])
    vdb.upsert(con, "predictions", [
        dict(id="2026-07-16_g5_h10_GP", symbol="GP", date="2026-07-16",
             target="g5_h10", probability=0.61, model_id="m", quality_score=100,
             is_signal=True, suppressed_reason=None),
        dict(id="2026-07-16_g5_h10_ACI", symbol="ACI", date="2026-07-16",
             target="g5_h10", probability=0.31, model_id="m", quality_score=100,
             is_signal=False, suppressed_reason="below-probability-threshold"),
    ])
    vdb.upsert(con, "explanations", [
        dict(prediction_id="2026-07-16_g5_h10_GP", drivers="[]", analogs="{}",
             rendered="GP: 61% calibrated probability of the g5_h10 move.")])


def test_digest_contains_signals_and_counts(test_db):
    _seed(test_db)
    body = digest.build(test_db, "2026-07-16")
    assert "2026-07-16" in body
    assert "GP" in body and "61%" in body           # the signal with rendering
    assert "1 signal" in body
    assert "quality 100" in body
    assert "below-probability-threshold: 1" in body  # suppression breakdown


def test_digest_no_predictions_day(test_db):
    body = digest.build(test_db, "2026-07-16")
    assert "no predictions" in body.lower()


def test_send_without_password_saves_to_reports(test_db, tmp_path, monkeypatch):
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    _seed(test_db)
    result = digest.send_or_save("subject", "body text", reports_dir=tmp_path)
    assert result["sent"] is False
    saved = list(tmp_path.glob("digest_*.md"))
    assert len(saved) == 1 and saved[0].read_text(encoding="utf-8") == "body text"


def test_send_with_password_uses_smtp(monkeypatch, tmp_path):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port):
            sent["host"] = host

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def login(self, user, pw):
            sent["user"], sent["pw"] = user, pw

        def send_message(self, msg):
            sent["subject"] = msg["Subject"]

    monkeypatch.setenv("GMAIL_APP_PASSWORD", "test-pw")
    monkeypatch.setattr(digest.smtplib, "SMTP_SSL", FakeSMTP)
    result = digest.send_or_save("Vectora digest", "body", reports_dir=tmp_path)
    assert result["sent"] is True
    assert sent["subject"] == "Vectora digest"
    assert sent["pw"] == "test-pw"
    assert sent["host"] == "smtp.gmail.com"


def test_digest_shows_regime(test_db):
    _seed(test_db)
    vdb.upsert(test_db, "regimes", [
        {"date": "2026-07-16", "regime": "Bull", "confidence": 0.8,
         "method": "rules"}])
    body = digest.build(test_db, "2026-07-16")
    assert "regime Bull" in body
