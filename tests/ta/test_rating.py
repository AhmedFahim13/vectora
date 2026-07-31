# tests/ta/test_rating.py
import polars as pl

from vectora.ta import rating


def _row(**kw):
    base = dict(macd=0.0, macd_signal=0.0, macd_hist=0.0, rsi14=50.0,
                bb_pos=0.5, bb_width=0.1, ma_fast=100.0, ma_slow=100.0,
                ma_cross_up=0, ma_cross_dn=0, st_dir=0, candle_bull=0,
                candle_bear=0, close=100.0)
    return {**base, **kw}


def test_all_bullish_gives_strong_buy():
    r = rating.score_row(_row(macd_hist=1.0, macd=1.0, rsi14=58,
                              ma_fast=110, ma_slow=100, ma_cross_up=1,
                              st_dir=1, candle_bull=1, bb_pos=0.7))
    assert r["band"] == "Strong Buy"
    assert r["score"] > 0
    assert len(r["votes"]) == 6            # one per indicator family


def test_all_bearish_gives_strong_sell():
    r = rating.score_row(_row(macd_hist=-1.0, macd=-1.0, rsi14=42,
                              ma_fast=90, ma_slow=100, ma_cross_dn=1,
                              st_dir=-1, candle_bear=1, bb_pos=0.3))
    assert r["band"] == "Strong Sell"
    assert r["score"] < 0


def test_neutral_gives_hold():
    assert rating.score_row(_row())["band"] == "Hold"


def test_overbought_rsi_counts_bearish_not_bullish():
    """RSI > 70 is exhaustion, not strength — the classic novice error."""
    hot = rating.score_row(_row(rsi14=82))
    rsi_vote = next(v for v in hot["votes"] if v["indicator"] == "RSI")
    assert rsi_vote["vote"] < 0
    assert "overbought" in rsi_vote["reason"].lower()


def test_votes_carry_human_readable_reasons():
    r = rating.score_row(_row(macd_hist=0.8, macd=0.5, st_dir=1))
    for v in r["votes"]:
        assert set(v) == {"indicator", "vote", "reason"}
        assert -2 <= v["vote"] <= 2
        assert len(v["reason"]) > 12
    macd = next(v for v in r["votes"] if v["indicator"] == "MACD")
    assert "signal line" in macd["reason"].lower()


def test_bands_are_ordered_and_exhaustive():
    scores = [rating.band_for(s) for s in range(-8, 9)]
    assert scores[0] == "Strong Sell" and scores[-1] == "Strong Buy"
    assert set(scores) == {"Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"}
    # monotone: never improves as score falls
    order = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
    idx = [order.index(b) for b in scores]
    assert idx == sorted(idx)


def test_rate_frame_adds_columns():
    df = pl.DataFrame([_row(macd_hist=1.0, st_dir=1) | {"symbol": "A"},
                       _row(macd_hist=-1.0, st_dir=-1) | {"symbol": "B"}])
    out = rating.rate_frame(df)
    assert {"ta_score", "ta_band", "ta_votes"} <= set(out.columns)
    assert out.filter(pl.col("symbol") == "A")["ta_score"][0] > 0
