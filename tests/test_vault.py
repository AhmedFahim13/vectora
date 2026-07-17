from vectora import db as vdb
from vectora.vault import generator as gen


def _seed(con):
    vdb.upsert(con, "symbols", [dict(
        symbol="GP", name=None, sector="Telecommunication",
        instrument_type="Equity", category="A", listing_status="active",
        first_seen="2013-01-01", last_seen="2026-07-16")])
    vdb.upsert(con, "data_quality", [
        {"date": "2026-07-16", "source": "dse_eod", "score": 100,
         "issues": "[]"}])
    vdb.upsert(con, "predictions", [
        dict(id="2026-07-16_g5_h10_GP", symbol="GP", date="2026-07-16",
             target="g5_h10", probability=0.61, model_id="m",
             quality_score=100, is_signal=True, suppressed_reason=None),
        dict(id="2026-07-16_g5_h10_ACI", symbol="ACI", date="2026-07-16",
             target="g5_h10", probability=0.31, model_id="m",
             quality_score=100, is_signal=False,
             suppressed_reason="below-probability-threshold"),
    ])
    vdb.upsert(con, "explanations", [dict(
        prediction_id="2026-07-16_g5_h10_GP", drivers="[]", analogs="{}",
        rendered="GP: 61% calibrated probability of the g5_h10 move.")])
    vdb.upsert(con, "events", [dict(
        id="e1", post_date="2026-07-16", symbol="GP",
        title="GP: Dividend Declared", body="10% cash", source="dse_news")])


def test_generate_writes_journal_prediction_company_home(test_db, tmp_path):
    _seed(test_db)
    result = gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    assert result["notes"] >= 4
    journal = (tmp_path / "Journal" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "2 predictions" in journal and "1 signal" in journal
    assert "quality 100" in journal
    pred = (tmp_path / "Predictions" / "2026-07" /
            "2026-07-16_g5_h10_GP.md").read_text(encoding="utf-8")
    assert "61%" in pred and "[[GP]]" in pred
    company = (tmp_path / "Companies" / "GP.md").read_text(encoding="utf-8")
    assert "Telecommunication" in company and "category: A" in company
    home = (tmp_path / "Home.md").read_text(encoding="utf-8")
    assert "2026-07-16" in home


def test_human_notes_survive_regeneration(test_db, tmp_path):
    _seed(test_db)
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    company = tmp_path / "Companies" / "GP.md"
    human = company.read_text(encoding="utf-8") + \
        "\n## Analyst Notes\nMy private thesis on GP.\n"
    company.write_text(human, encoding="utf-8")
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    text = company.read_text(encoding="utf-8")
    assert "My private thesis on GP." in text
    assert text.count(gen.MACHINE_BEGIN) == 1  # markers not duplicated


def test_non_signal_predictions_get_no_note(test_db, tmp_path):
    _seed(test_db)
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    assert not (tmp_path / "Predictions" / "2026-07" /
                "2026-07-16_g5_h10_ACI.md").exists()


def test_no_data_day_still_writes_journal(test_db, tmp_path):
    result = gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    assert result["notes"] >= 2   # journal + home always written
    journal = (tmp_path / "Journal" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "0 predictions" in journal


def test_journal_shows_regime(test_db, tmp_path):
    _seed(test_db)
    vdb.upsert(test_db, "regimes", [
        {"date": "2026-07-16", "regime": "Sideways", "confidence": 0.5,
         "method": "rules"}])
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    journal = (tmp_path / "Journal" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "regime Sideways" in journal


def test_journal_types_material_events(test_db, tmp_path):
    _seed(test_db)
    vdb.upsert(test_db, "event_labels", [dict(
        event_id="e1", event_type="dividend_declared", materiality=3)])
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    journal = (tmp_path / "Journal" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "dividend_declared" in journal
