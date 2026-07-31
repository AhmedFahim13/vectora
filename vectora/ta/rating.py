"""Six indicator families -> votes -> 5-band technical posture.

Each family returns a vote in [-2, +2] with a plain-English reason; the
sum maps to a band. This is a MECHANICAL SUMMARY of indicator states, not
a forecast and not advice — vectora/ta/validate.py measures what each band
historically did, and the UI always shows that base rate alongside.

RSI is scored the way practitioners actually read it: mid-range strength
is constructive, but >70 is exhaustion (bearish) and <30 is washed-out
(bullish) — the opposite of the naive 'high RSI = strong' reading.
"""
import json

import polars as pl

BANDS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
# score thresholds: <=-5 SS, -4..-2 S, -1..1 H, 2..4 B, >=5 SB
_CUTS = [(-99, -5, "Strong Sell"), (-5, -1, "Sell"), (-1, 2, "Hold"),
         (2, 5, "Buy"), (5, 99, "Strong Buy")]


def band_for(score: int) -> str:
    for lo, hi, name in _CUTS:
        if lo <= score < hi if name != "Strong Sell" else score < hi:
            return name
    return "Strong Buy"


def _macd(r) -> dict:
    h = r.get("macd_hist") or 0.0
    if h > 0:
        v, why = (2 if (r.get("macd") or 0) > 0 else 1), \
            "MACD is above its signal line and the histogram is positive"
    elif h < 0:
        v, why = (-2 if (r.get("macd") or 0) < 0 else -1), \
            "MACD is below its signal line and the histogram is negative"
    else:
        v, why = 0, "MACD is sitting on its signal line"
    return {"indicator": "MACD", "vote": v, "reason": why}


def _rsi(r) -> dict:
    x = r.get("rsi14")
    if x is None:
        return {"indicator": "RSI", "vote": 0, "reason": "RSI not yet available"}
    if x >= 70:
        return {"indicator": "RSI", "vote": -1,
                "reason": f"RSI {x:.0f} is overbought — stretched, prone to mean reversion"}
    if x <= 30:
        return {"indicator": "RSI", "vote": 1,
                "reason": f"RSI {x:.0f} is oversold — washed out, prone to a bounce"}
    if x >= 55:
        return {"indicator": "RSI", "vote": 1,
                "reason": f"RSI {x:.0f} shows constructive momentum without being stretched"}
    if x <= 45:
        return {"indicator": "RSI", "vote": -1,
                "reason": f"RSI {x:.0f} shows fading momentum"}
    return {"indicator": "RSI", "vote": 0, "reason": f"RSI {x:.0f} is neutral"}


def _bollinger(r) -> dict:
    p = r.get("bb_pos")
    if p is None:
        return {"indicator": "Bollinger", "vote": 0,
                "reason": "Bollinger bands not yet available"}
    if p > 1:
        return {"indicator": "Bollinger", "vote": -1,
                "reason": "price closed above the upper band — extended"}
    if p < 0:
        return {"indicator": "Bollinger", "vote": 1,
                "reason": "price closed below the lower band — capitulation zone"}
    if p > 0.65:
        return {"indicator": "Bollinger", "vote": 1,
                "reason": "price is riding the upper half of the band"}
    if p < 0.35:
        return {"indicator": "Bollinger", "vote": -1,
                "reason": "price is pinned to the lower half of the band"}
    return {"indicator": "Bollinger", "vote": 0,
            "reason": "price is mid-band, no stretch either way"}


def _ma(r) -> dict:
    f, s = r.get("ma_fast"), r.get("ma_slow")
    if f is None or s is None:
        return {"indicator": "MA cross", "vote": 0,
                "reason": "moving averages not yet available"}
    if r.get("ma_cross_up"):
        return {"indicator": "MA cross", "vote": 2,
                "reason": "the 20-day average has just crossed above the 50-day (golden cross)"}
    if r.get("ma_cross_dn"):
        return {"indicator": "MA cross", "vote": -2,
                "reason": "the 20-day average has just crossed below the 50-day (death cross)"}
    if f > s:
        return {"indicator": "MA cross", "vote": 1,
                "reason": "the 20-day average holds above the 50-day"}
    if f < s:
        return {"indicator": "MA cross", "vote": -1,
                "reason": "the 20-day average sits below the 50-day"}
    return {"indicator": "MA cross", "vote": 0, "reason": "moving averages are entwined"}


def _supertrend(r) -> dict:
    d = r.get("st_dir") or 0
    if d > 0:
        return {"indicator": "SuperTrend", "vote": 2,
                "reason": "SuperTrend is in an up-phase; its stop sits below price"}
    if d < 0:
        return {"indicator": "SuperTrend", "vote": -2,
                "reason": "SuperTrend is in a down-phase; its stop sits above price"}
    return {"indicator": "SuperTrend", "vote": 0,
            "reason": "SuperTrend has not established a direction"}


def _candle(r) -> dict:
    if r.get("candle_bull"):
        return {"indicator": "Candlestick", "vote": 1,
                "reason": "a bullish reversal candle printed (engulfing or hammer)"}
    if r.get("candle_bear"):
        return {"indicator": "Candlestick", "vote": -1,
                "reason": "a bearish reversal candle printed (engulfing or shooting star)"}
    return {"indicator": "Candlestick", "vote": 0,
            "reason": "no notable candlestick pattern"}


_FAMILIES = (_macd, _rsi, _bollinger, _ma, _supertrend, _candle)


def score_row(r: dict) -> dict:
    votes = [f(r) for f in _FAMILIES]
    score = sum(v["vote"] for v in votes)
    return {"score": score, "band": band_for(score), "votes": votes}


def rate_frame(df: pl.DataFrame) -> pl.DataFrame:
    results = [score_row(r) for r in df.iter_rows(named=True)]
    return df.with_columns(
        pl.Series("ta_score", [x["score"] for x in results]),
        pl.Series("ta_band", [x["band"] for x in results]),
        pl.Series("ta_votes", [json.dumps(x["votes"]) for x in results]))
