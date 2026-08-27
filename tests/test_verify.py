"""Per-stock verification: what we said, and what the price then did."""
import datetime as dt

import numpy as np

from vectora import db as vdb
from vectora import verify


def _seed(con, symbol="AAA", n=200, seed=3):
    rng = np.random.default_rng(seed)
    px = 100.0
    d0 = dt.date(2025, 1, 1)
    rows = []
    for i in range(n):
        px *= 1 + rng.normal(0.0005, 0.012)
        rows.append(dict(
            symbol=symbol, date=(d0 + dt.timedelta(days=i)).isoformat(),
            open=px, high=px * 1.01, low=px * 0.99, close=px, ltp=None,
            ycp=None, trades=10, value_mn=5.0, volume=1000, source="dse_eod"))
    vdb.upsert(con, "prices_raw", rows)


def test_history_returns_a_row_per_recent_day(test_db):
    _seed(test_db)
    rows = verify.history(test_db, "AAA", days=12)
    assert len(rows) == 12
    r = rows[0]
    assert r["summary"] in ("Strong Sell", "Sell", "Hold", "Buy", "Strong Buy")
    assert "RSI(14)" in r["readings"]
    assert r["close"] > 0


def test_recent_rows_say_pending_rather_than_borrowing_a_number(test_db):
    """The last few days have no completed 10-day forward window. Filling
    them with a partial return would overstate what is known."""
    _seed(test_db)
    rows = verify.history(test_db, "AAA", days=12)
    assert rows[-1]["ret_10d"] is None
    assert rows[0]["ret_10d"] is not None


def test_forward_return_is_measured_from_the_row_s_own_close(test_db):
    _seed(test_db)
    rows = verify.history(test_db, "AAA", days=30)
    graded = [r for r in rows if r["ret_5d"] is not None]
    closes = {r["date"]: r["close"] for r in rows}
    dates = [r["date"] for r in rows]
    r = graded[0]
    i = dates.index(r["date"])
    expected = closes[dates[i + 5]] / r["close"] - 1
    assert abs(r["ret_5d"] - expected) < 1e-9


def test_scorecard_counts_only_graded_rows(test_db):
    _seed(test_db)
    rows = verify.history(test_db, "AAA", days=15)
    card = verify.scorecard(rows, horizon=10)
    graded = [r for r in rows if r["ret_10d"] is not None]
    assert card["n"] == len(graded)
    assert card["bullish_n"] + card["bearish_n"] <= card["n"]


def test_short_history_returns_nothing_rather_than_noise(test_db):
    _seed(test_db, n=30)
    assert verify.history(test_db, "AAA") == []


def test_unknown_symbol_is_empty(test_db):
    _seed(test_db)
    assert verify.history(test_db, "NOPE") == []


def test_shared_panel_matches_a_solo_load(test_db):
    """The panel is loaded once for a loop over many symbols; that must not
    change a single symbol's answer."""
    _seed(test_db)
    solo = verify.history(test_db, "AAA", days=10)
    shared = verify.history(test_db, "AAA", days=10,
                            panel=verify.load_panel(test_db))
    assert [r["summary"] for r in solo] == [r["summary"] for r in shared]
    assert [r["ret_5d"] for r in solo] == [r["ret_5d"] for r in shared]
