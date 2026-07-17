# tests/events/test_studies.py
import datetime as dt

import numpy as np
import polars as pl

from vectora import db as vdb
from vectora.events import studies


def _seed(con, n_days=40, n_syms=40, bump_sym="S00", bump_day=20,
          bump=0.05, seed=9):
    """Flat market; one symbol jumps +5% the day after its event."""
    rng = np.random.default_rng(seed)
    rows = []
    d0 = dt.date(2026, 1, 1)
    px = {f"S{i:02d}": 100.0 for i in range(n_syms)}
    for day in range(n_days):
        d = d0 + dt.timedelta(days=day)
        for sym in px:
            drift = bump if (sym == bump_sym and day == bump_day + 1) else 0.0
            px[sym] *= (1 + drift + float(rng.normal(0, 0.001)))
            p = round(px[sym], 4)
            rows.append(dict(symbol=sym, date=d, open=p, high=p, low=p,
                             close=p, ltp=p, ycp=None, trades=10,
                             value_mn=1.0, volume=100, source="mendeley"))
    df = pl.DataFrame(rows)  # noqa: F841
    con.execute("INSERT INTO prices_raw SELECT * FROM df")
    event_date = (d0 + dt.timedelta(days=bump_day)).isoformat()
    vdb.upsert(con, "events", [dict(
        id="ev1", post_date=event_date, symbol=bump_sym,
        title=f"{bump_sym}: Dividend Declaration", body="", source="dse_news")])
    vdb.upsert(con, "event_labels", [dict(
        event_id="ev1", event_type="dividend_declared", materiality=3)])


def test_event_study_detects_abnormal_return(test_db):
    _seed(test_db)
    result = studies.compute(test_db, min_events=1, horizons=(1, 3))
    assert result["types"] >= 1
    row = test_db.execute(
        """
        SELECT n, mean_abn_ret, pos_share FROM event_studies
        WHERE event_type = 'dividend_declared' AND horizon = 1
        """).fetchone()
    assert row[0] == 1
    assert row[1] > 0.03            # ~+5% vs ~flat market
    assert row[2] == 1.0


def test_min_events_threshold_excludes_thin_types(test_db):
    _seed(test_db)
    result = studies.compute(test_db, min_events=5, horizons=(1,))
    assert result["types"] == 0
    assert test_db.execute(
        "SELECT count(*) FROM event_studies").fetchone()[0] == 0


def test_vault_note_rendered(test_db, tmp_path):
    _seed(test_db)
    studies.compute(test_db, min_events=1, horizons=(1, 3))
    path = studies.write_vault_note(test_db, vault_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "dividend_declared" in text and "h1" in text
