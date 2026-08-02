"""Pivot / support-resistance tests, including the lookahead guard."""
import datetime as dt

import polars as pl

from vectora.ta import levels


def _frame(rows) -> pl.DataFrame:
    return pl.DataFrame(
        {"symbol": [r[0] for r in rows],
         "date": [r[1] for r in rows],
         "high": [float(r[2]) for r in rows],
         "low": [float(r[3]) for r in rows],
         "close": [float(r[4]) for r in rows]})


def test_classic_pivot_formula():
    """January H/L/C = 120/80/100 -> P=100, R1=120, S1=80 for February."""
    rows = [("A", dt.date(2026, 1, 5), 120, 80, 90),
            ("A", dt.date(2026, 1, 20), 110, 90, 100),
            ("A", dt.date(2026, 2, 3), 105, 95, 100)]
    out = levels.add_pivots(_frame(rows)).sort("date")
    feb = out.filter(pl.col("date") == dt.date(2026, 2, 3))
    assert abs(feb["pivot_point"][0] - 100.0) < 1e-9
    assert abs(feb["r1"][0] - 120.0) < 1e-9      # 2*100 - 80
    assert abs(feb["s1"][0] - 80.0) < 1e-9       # 2*100 - 120
    assert abs(feb["r2"][0] - 140.0) < 1e-9      # 100 + (120-80)
    assert abs(feb["s2"][0] - 60.0) < 1e-9


def test_first_month_has_no_pivot():
    """Nothing precedes January, so its levels must be null, not invented."""
    rows = [("A", dt.date(2026, 1, 5), 120, 80, 90)]
    out = levels.add_pivots(_frame(rows))
    assert out["pivot_point"][0] is None


def test_pivots_never_use_the_current_month():
    """A violent February must not move February's own pivot."""
    base = [("A", dt.date(2026, 1, 5), 120, 80, 100),
            ("A", dt.date(2026, 2, 3), 105, 95, 100)]
    spike = [*base, ("A", dt.date(2026, 2, 20), 900, 5, 400)]
    p_base = levels.add_pivots(_frame(base)).filter(
        pl.col("date") == dt.date(2026, 2, 3))["pivot_point"][0]
    p_spike = levels.add_pivots(_frame(spike)).filter(
        pl.col("date") == dt.date(2026, 2, 3))["pivot_point"][0]
    assert abs(p_base - p_spike) < 1e-9


def test_pivots_are_per_symbol():
    rows = [("A", dt.date(2026, 1, 5), 120, 80, 100),
            ("B", dt.date(2026, 1, 5), 12, 8, 10),
            ("A", dt.date(2026, 2, 3), 105, 95, 100),
            ("B", dt.date(2026, 2, 3), 11, 9, 10)]
    out = levels.add_pivots(_frame(rows))
    b = out.filter((pl.col("symbol") == "B")
                   & (pl.col("date") == dt.date(2026, 2, 3)))
    assert abs(b["pivot_point"][0] - 10.0) < 1e-9      # (12+8+10)/3, not A's 100


def test_swing_high_excludes_today():
    """A new high today must not be reported as today's own resistance."""
    days = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(30)]
    rows = [("A", d, 100, 90, 95) for d in days[:-1]]
    rows.append(("A", days[-1], 500, 90, 480))      # today spikes
    out = levels.add_swing_levels(_frame(rows)).sort("date")
    assert abs(out["hi_20d"][-1] - 100.0) < 1e-9   # not 500


def test_nearest_levels_bracket_the_price():
    rows = [("A", dt.date(2026, 1, 5), 120, 80, 100),
            ("A", dt.date(2026, 2, 3), 105, 95, 100)]
    out = levels.add_all(_frame(rows)).sort("date")
    row = out.filter(pl.col("date") == dt.date(2026, 2, 3))
    res, sup, close = row["nearest_res"][0], row["nearest_sup"][0], 100.0
    if res is not None:
        assert res > close
    if sup is not None:
        assert sup < close


def test_room_metrics_are_fractions_of_price():
    rows = [("A", dt.date(2026, 1, 5), 120, 80, 100),
            ("A", dt.date(2026, 2, 3), 105, 95, 100)]
    out = levels.add_all(_frame(rows)).filter(
        pl.col("date") == dt.date(2026, 2, 3))
    up = out["room_up"][0]
    if up is not None:
        assert 0 < up < 1        # R1 at 120 vs close 100 -> 0.20
        assert abs(up - 0.20) < 1e-9
