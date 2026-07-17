# tests/regime/test_rules.py
import polars as pl

from vectora import db as vdb
from vectora.regime import rules


def _row(**kw):
    base = dict(mkt_level=1.0, ma50=1.0, ma200=0.95, ret_21d=0.01,
                vol_pctile=0.5, breadth=0.5, activity_z=0.0)
    return {**base, **kw}


def test_panic_beats_everything():
    r, c = rules.classify_row(_row(vol_pctile=0.95, ret_21d=-0.12,
                                   breadth=0.7, mkt_level=1.2))
    assert r == "Panic" and c >= 0.7


def test_low_liquidity():
    assert rules.classify_row(_row(activity_z=-2.0))[0] == "LowLiquidity"


def test_recovery_below_trend_but_rallying():
    r, _ = rules.classify_row(_row(mkt_level=0.9, ma200=1.0, ret_21d=0.08))
    assert r == "Recovery"


def test_speculative_heat():
    r, _ = rules.classify_row(_row(activity_z=2.5, vol_pctile=0.8))
    assert r == "SpeculativeHeat"


def test_bull_bear_sideways():
    assert rules.classify_row(
        _row(mkt_level=1.1, ma200=1.0, breadth=0.7))[0] == "Bull"
    assert rules.classify_row(
        _row(mkt_level=0.9, ma200=1.0, breadth=0.2))[0] == "Bear"
    assert rules.classify_row(_row())[0] == "Sideways"


def test_warmup_rows_unclassified():
    assert rules.classify_row(_row(ma200=None)) is None


def test_classify_history_writes_table(test_db, monkeypatch):
    import datetime as dt
    frame = pl.DataFrame({
        "date": [dt.date(2026, 7, d) for d in (1, 2, 3)],
        "mkt_level": [1.1, 0.9, 1.0],
        "ma50": [1.0, 1.0, 1.0],
        "ma200": [1.0, 1.0, None],          # day 3 = warmup, skipped
        "ret_21d": [0.02, -0.12, 0.0],
        "vol_pctile": [0.5, 0.95, 0.5],
        "breadth": [0.7, 0.6, 0.5],
        "activity_z": [0.0, 0.0, 0.0],
    })
    from vectora.regime import state as st
    monkeypatch.setattr(st, "market_state", lambda con: frame)
    result = rules.classify_history(test_db)
    assert result == {"classified": 2, "skipped": 1}
    rows = dict(test_db.execute(
        "SELECT date, regime FROM regimes ORDER BY date").fetchall())
    assert str(min(rows)) == "2026-07-01"
    assert list(rows.values()) == ["Bull", "Panic"]


def test_current_regime_lookup(test_db):
    vdb.upsert(test_db, "regimes", [
        {"date": "2026-07-16", "regime": "Bull", "confidence": 0.8,
         "method": "rules"}])
    assert rules.regime_on(test_db, "2026-07-16") == "Bull"
    assert rules.regime_on(test_db, "2026-07-15") is None
