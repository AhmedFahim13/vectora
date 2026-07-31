# tests/ta/test_indicators.py
import datetime as dt

import polars as pl

from vectora.ta import indicators as ind


def _series(closes, highs=None, lows=None, opens=None):
    n = len(closes)
    return pl.DataFrame({
        "symbol": ["X"] * n,
        "date": [dt.date(2026, 1, 1) + dt.timedelta(days=i)
                 for i in range(n)],
        "open": opens or closes,
        "high": highs or [c * 1.01 for c in closes],
        "low": lows or [c * 0.99 for c in closes],
        "close": closes,
    })


def test_macd_positive_on_uptrend():
    up = [100 * (1.01 ** i) for i in range(60)]
    out = ind.add_all(_series(up))
    last = out.tail(1).row(0, named=True)
    assert last["macd_hist"] > 0        # fast EMA above slow, rising
    assert last["macd"] > last["macd_signal"]


def test_macd_negative_on_downtrend():
    down = [100 * (0.99 ** i) for i in range(60)]
    last = ind.add_all(_series(down)).tail(1).row(0, named=True)
    # the MACD LEVEL carries trend direction. The histogram is deliberately
    # not asserted here: a constant-percentage decline shrinks in absolute
    # terms, so MACD rises toward zero from below and the histogram turns
    # positive — correct behaviour (decelerating downtrend), not a bug.
    assert last["macd"] < 0


def test_rsi_extremes():
    up = [100 * (1.02 ** i) for i in range(40)]
    down = [100 * (0.98 ** i) for i in range(40)]
    assert ind.add_all(_series(up)).tail(1).row(0, named=True)["rsi14"] > 70
    assert ind.add_all(_series(down)).tail(1).row(0, named=True)["rsi14"] < 30


def test_bollinger_position_and_width():
    flat = [100.0] * 30 + [130.0]
    out = ind.add_all(_series(flat))
    last = out.tail(1).row(0, named=True)
    assert last["bb_pos"] > 1.0          # closed above the upper band
    assert last["bb_width"] >= 0


def test_ma_cross_state_and_recency():
    # 60 flat days (both averages warm and equal) then a sharp rally, so the
    # crossover happens with a real "before" state rather than during warmup
    closes = [100.0] * 60 + [100 * (1.03 ** i) for i in range(1, 30)]
    out = ind.add_all(_series(closes))
    last = out.tail(1).row(0, named=True)
    assert last["ma_fast"] > last["ma_slow"]
    # the flag marks a FRESH cross (5-day window), so it fires during the
    # crossover and goes quiet again once the trend is established
    assert max(out["ma_cross_up"].fill_null(0).to_list()) == 1
    assert last["ma_cross_up"] == 0


def test_supertrend_flips_direction():
    closes = [100 * (1.02 ** i) for i in range(40)] + \
             [100 * (1.02 ** 39) * (0.97 ** i) for i in range(1, 30)]
    out = ind.add_all(_series(closes))
    dirs = out["st_dir"].to_list()
    assert dirs[38] == 1                 # uptrend during the rally
    assert dirs[-1] == -1                # flipped down after the slide


def test_candles_detect_bullish_engulfing():
    # prior red candle, then a larger green candle engulfing it
    o = [100, 99, 96]
    c = [99, 96, 101]
    out = ind.add_all(_series(c, opens=o,
                              highs=[max(a, b) * 1.005 for a, b in zip(o, c, strict=True)],
                              lows=[min(a, b) * 0.995 for a, b in zip(o, c, strict=True)]))
    assert out.tail(1).row(0, named=True)["candle_bull"] == 1


def test_no_lookahead_in_any_indicator():
    """Indicator values for early rows must not change when later rows differ."""
    base = [100 + i * 0.5 for i in range(60)]
    a = ind.add_all(_series(base + [100.0] * 20))
    b = ind.add_all(_series(base + [400.0] * 20))
    cols = [c for c in a.columns if c not in ("symbol", "date", "open",
                                              "high", "low", "close")]
    ha, hb = a.head(60), b.head(60)
    for col in cols:
        va, vb = ha[col].to_list(), hb[col].to_list()
        for x, y in zip(va, vb, strict=True):
            if x is None and y is None:
                continue
            assert x == y or abs(x - y) < 1e-9, f"LOOKAHEAD in {col}"
