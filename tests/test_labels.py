# tests/test_labels.py
import datetime as dt

import polars as pl

from vectora import labels


def _panel():
    rows = []
    d0 = dt.date(2026, 1, 1)
    closes = [100, 101, 103, 111, 108, 104, 100, 100, 100, 100, 100, 100]
    for i, c in enumerate(closes):
        rows.append(dict(symbol="AAA", date=d0 + dt.timedelta(days=i),
                         close=float(c)))
    return pl.DataFrame(rows)


def test_gain_label_hits_within_horizon():
    df = labels.make_labels(_panel(), thresholds=(0.05, 0.10), horizons=(3, 5))
    row0 = df.filter(pl.col("date") == dt.date(2026, 1, 1))
    # from close 100: max close within 3 days = 111 -> both thresholds hit
    assert row0["y_g5_h3"][0] == 1 and row0["y_g10_h3"][0] == 1


def test_gain_label_miss():
    df = labels.make_labels(_panel(), thresholds=(0.10,), horizons=(3,))
    row4 = df.filter(pl.col("date") == dt.date(2026, 1, 5))  # close 108
    # next 3 closes: 104,100,100 -> no +10%
    assert row4["y_g10_h3"][0] == 0


def test_label_null_when_horizon_incomplete():
    df = labels.make_labels(_panel(), thresholds=(0.05,), horizons=(5,))
    last = df.sort("date").tail(5)
    assert last["y_g5_h5"].null_count() == 5  # fewer than 5 future closes


def test_downside_label():
    df = labels.make_labels(_panel(), thresholds=(0.05,), horizons=(3,),
                            downside=True)
    row3 = df.filter(pl.col("date") == dt.date(2026, 1, 4))  # close 111
    # next 3 closes: 108,104,100 -> min 100 = -9.9% -> 5% drawdown hit
    assert row3["y_d5_h3"][0] == 1


def test_continuous_forward_outcomes():
    df = labels.make_labels(_panel(), thresholds=(0.05,), horizons=(3,),
                            continuous=True)
    row0 = df.filter(pl.col("date") == dt.date(2026, 1, 1))  # close 100
    # next 3 closes: 101,103,111 -> max +11%, min +1%
    assert abs(row0["fwdmax_h3"][0] - 0.11) < 1e-9
    assert abs(row0["fwdmin_h3"][0] - 0.01) < 1e-9
    # incomplete horizon -> null
    assert df.sort("date").tail(3)["fwdmax_h3"].null_count() == 3
