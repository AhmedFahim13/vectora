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
    assert "Telecommunication" in company and "category A" in company
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


def test_journal_zwatch_section(test_db, tmp_path):
    _seed(test_db)
    vdb.upsert(test_db, "zwatch", [dict(
        date="2026-07-16", symbol="ZPUMP", kind="pump", score=88.0,
        phase="markup", detail="{}")])
    gen.generate(test_db, "2026-07-16", vault_dir=tmp_path)
    journal = (tmp_path / "Journal" / "2026-07-16.md").read_text(encoding="utf-8")
    assert "Z-watch" in journal and "ZPUMP" in journal and "88" in journal


def test_every_rated_symbol_gets_a_company_note(test_db, tmp_path):
    """The screener analyses 400+ securities a day. Writing notes only for
    the handful that signalled threw that work away."""
    from vectora import db as vdb
    from vectora.vault import generator
    d = "2026-08-23"
    vdb.upsert(test_db, "symbols", [
        {"symbol": s, "sector": "Bank", "category": "A"}
        for s in ("AAA", "BBB", "CCC")])
    vdb.upsert(test_db, "ta_ratings", [
        {"date": d, "symbol": s, "score": i, "band": "Buy", "votes": "[]",
         "rsi": 55.0, "macd_hist": 0.1, "bb_pos": 0.5, "st_dir": 1}
        for i, s in enumerate(("AAA", "BBB", "CCC"))])
    generator.generate(test_db, d, vault_dir=tmp_path)
    notes = sorted(p.name for p in (tmp_path / "Companies").glob("*.md"))
    assert notes == ["AAA.md", "BBB.md", "CCC.md"]
    body = (tmp_path / "Companies" / "AAA.md").read_text(encoding="utf-8")
    assert "Technical posture" in body
    assert "Sectors/Bank" in body          # links into the sector note


def test_sector_notes_link_their_members(test_db, tmp_path):
    from vectora import db as vdb
    from vectora.vault import generator
    d = "2026-08-23"
    vdb.upsert(test_db, "symbols", [
        {"symbol": "AAA", "sector": "Bank", "category": "A"}])
    vdb.upsert(test_db, "ta_ratings", [
        {"date": d, "symbol": "AAA", "score": 3, "band": "Buy", "votes": "[]",
         "rsi": 55.0, "macd_hist": 0.1, "bb_pos": 0.5, "st_dir": 1}])
    vdb.upsert(test_db, "sector_rs", [
        {"date": d, "sector": "Bank", "n_symbols": 1, "ret_5d": 0.01,
         "ret_21d": 0.05, "ret_63d": 0.1, "ret_126d": 0.2, "rs_21d": 0.02,
         "rs_63d": 0.03, "rs_momentum": 0.01, "quadrant": "Leading"}])
    generator.generate(test_db, d, vault_dir=tmp_path)
    note = (tmp_path / "Sectors" / "Bank.md").read_text(encoding="utf-8")
    assert "Leading" in note
    assert "[[AAA]]" in note


def test_company_note_survives_human_edits(test_db, tmp_path):
    """A note the user has annotated must keep their text byte-identical."""
    from vectora import db as vdb
    from vectora.vault import generator
    d = "2026-08-23"
    vdb.upsert(test_db, "symbols", [
        {"symbol": "AAA", "sector": "Bank", "category": "A"}])
    vdb.upsert(test_db, "ta_ratings", [
        {"date": d, "symbol": "AAA", "score": 1, "band": "Buy", "votes": "[]",
         "rsi": 50.0, "macd_hist": 0.0, "bb_pos": 0.5, "st_dir": 1}])
    generator.generate(test_db, d, vault_dir=tmp_path)
    path = tmp_path / "Companies" / "AAA.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nMy own note.\n",
                    encoding="utf-8")
    generator.generate(test_db, d, vault_dir=tmp_path)
    assert "My own note." in path.read_text(encoding="utf-8")


def test_backfilling_an_old_journal_does_not_rewrite_current_notes(test_db,
                                                                   tmp_path):
    """Company notes describe where a stock stands NOW. Regenerating a
    journal from six weeks ago must not overwrite them with that day's
    posture — the vault would silently go stale."""
    from vectora import db as vdb
    from vectora.vault import generator
    vdb.upsert(test_db, "symbols", [
        {"symbol": "AAA", "sector": "Bank", "category": "A"}])
    vdb.upsert(test_db, "ta_ratings", [
        {"date": "2026-08-23", "symbol": "AAA", "score": 5, "band": "Buy",
         "votes": "[]", "rsi": 60.0, "macd_hist": 0.2, "bb_pos": 0.8,
         "st_dir": 1}])
    generator.generate(test_db, "2026-08-23", vault_dir=tmp_path)
    fresh = (tmp_path / "Companies" / "AAA.md").read_text(encoding="utf-8")
    assert "+5" in fresh

    generator.generate(test_db, "2026-07-21", vault_dir=tmp_path)
    assert (tmp_path / "Journal" / "2026-07-21.md").exists()
    after = (tmp_path / "Companies" / "AAA.md").read_text(encoding="utf-8")
    assert after == fresh, "backfill rewrote a current-state note"
