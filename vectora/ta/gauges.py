"""Two technical gauges: moving averages, and oscillators.

Started from TradingView's 26-component rating and pruned on 2026-08-26 to
the components the client actually reads (see indicators.MA_PERIODS).

The split into two gauges is the point: trend-followers and oscillators are
designed to disagree at turning points. A stock above every average with
its oscillators screaming overbought is a different animal from one below
its averages with oscillators washed out, even though a single blended
number can rate them identically.

Each component votes exactly -1 / 0 / +1 (TradingView's own convention), the
gauge is the mean vote, and the summary is the mean of the two gauge means.
Bands follow TradingView's cut points: |mean| >= 0.5 is "strong".

This module deliberately does NOT replace rating.py. That six-family score
has a measured edge on this exchange (validate.py); this one is a second,
independent read whose own edge is measured the same way before it is
trusted. Neither is advice.
"""
import json

import polars as pl

from vectora.ta.indicators import CCI_PERIODS, MA_PERIODS, VWMA_PERIODS

BANDS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
_STRONG, _WEAK = 0.5, 0.1

# columns whose direction (rising/falling) a TradingView rule depends on
PREV_COLS = ("rsi14", "cci10", "cci20", "ao", "stochrsi", "macd_hist")


def add_prev(df: pl.DataFrame) -> pl.DataFrame:
    """Adds prev_<col> for every direction-sensitive component."""
    return df.with_columns([
        pl.col(c).shift(1).over("symbol").alias(f"prev_{c}")
        for c in PREV_COLS if c in df.columns])


def band_for(mean_vote: float) -> str:
    if mean_vote >= _STRONG:
        return "Strong Buy"
    if mean_vote >= _WEAK:
        return "Buy"
    if mean_vote > -_WEAK:
        return "Hold"
    if mean_vote > -_STRONG:
        return "Sell"
    return "Strong Sell"


def _rising(r, col) -> bool | None:
    now, prev = r.get(col), r.get(f"prev_{col}")
    if now is None or prev is None:
        return None
    return now > prev


def _idle(value: float, lo: float, hi: float, label: str) -> str:
    """Why a component abstained.

    These rules are contrarian: an oscillator only votes when it is at an
    extreme AND turning back. Saying "mid-range" when the reading is 8 out
    of 100 is simply wrong, and a reader checking the number against the
    prose loses trust in everything else on the page.
    """
    if value < lo:
        return f"{label} {value:.0f} is oversold but still falling - no signal yet"
    if value > hi:
        return f"{label} {value:.0f} is overbought but still rising - no signal yet"
    return f"{label} {value:.0f} is mid-range"


def _vote(indicator: str, vote: int, reason: str) -> dict:
    return {"indicator": indicator, "vote": vote, "reason": reason}


# --- moving averages -------------------------------------------------------
# Rule is uniform and deliberately dumb: price above the average is a buy
# vote, below is a sell. Simplicity is what makes the gauge readable.
def ma_votes(r: dict) -> list[dict]:
    close = r.get("close")
    out: list[dict] = []
    if close is None:
        return out
    for kind in ("sma", "ema"):
        label = "SMA" if kind == "sma" else "EMA"
        for n in MA_PERIODS:
            v = r.get(f"{kind}{n}")
            if v is None:
                out.append(_vote(f"{label}({n})", 0, "not enough history yet"))
                continue
            side = "above" if close > v else ("below" if close < v else "on")
            out.append(_vote(
                f"{label}({n})", 1 if close > v else (-1 if close < v else 0),
                f"{side} it ({v:,.2f})"))
    for n in VWMA_PERIODS:
        col, label = f"vwma{n}", f"VWMA({n})"
        v = r.get(col)
        if v is None:
            out.append(_vote(label, 0, "not enough history yet"))
        else:
            side = "above" if close > v else ("below" if close < v else "on")
            out.append(_vote(label, 1 if close > v else (-1 if close < v else 0),
                             f"{side} it ({v:,.2f})"))
    # Ichimoku: position relative to the cloud, the way the system is read
    a, b = r.get("ichi_a"), r.get("ichi_b")
    if a is None or b is None:
        out.append(_vote("Ichimoku Cloud", 0, "cloud not yet formed"))
    else:
        top, bot = max(a, b), min(a, b)
        if close > top:
            out.append(_vote("Ichimoku Cloud", 1,
                             f"price is above the cloud ({bot:,.2f}-{top:,.2f})"))
        elif close < bot:
            out.append(_vote("Ichimoku Cloud", -1,
                             f"price is below the cloud ({bot:,.2f}-{top:,.2f})"))
        else:
            out.append(_vote("Ichimoku Cloud", 0,
                             "price is inside the cloud — no trend agreement"))
    return out


# --- oscillators -----------------------------------------------------------
# These follow TradingView's published rules, which are contrarian at the
# extremes: an oscillator only votes BUY when it is oversold AND turning up.
def osc_votes(r: dict) -> list[dict]:
    out: list[dict] = []

    x = r.get("rsi14")
    if x is None:
        out.append(_vote("RSI(14)", 0, "not available"))
    elif x < 30 and _rising(r, "rsi14"):
        out.append(_vote("RSI(14)", 1, f"RSI {x:.0f} is oversold and turning up"))
    elif x > 70 and _rising(r, "rsi14") is False:
        out.append(_vote("RSI(14)", -1, f"RSI {x:.0f} is overbought and rolling over"))
    else:
        out.append(_vote("RSI(14)", 0, f"RSI {x:.0f} is not at an actionable extreme"))

    for n in CCI_PERIODS:
        c = r.get(f"cci{n}")
        label = f"CCI({n})"
        if c is None:
            out.append(_vote(label, 0, "not available"))
        elif c < -100 and _rising(r, f"cci{n}"):
            out.append(_vote(label, 1,
                             f"CCI {c:.0f} is deeply negative and rising"))
        elif c > 100 and _rising(r, f"cci{n}") is False:
            out.append(_vote(label, -1,
                             f"CCI {c:.0f} is stretched high and falling"))
        else:
            out.append(_vote(label, 0, _idle(c, -100, 100, "CCI")))

    adx, dp, dm = r.get("adx14"), r.get("di_plus"), r.get("di_minus")
    if adx is None or dp is None or dm is None:
        out.append(_vote("ADX(14)", 0, "not available"))
    elif adx < 20:
        out.append(_vote("ADX(14)", 0,
                         f"ADX {adx:.0f} — no trend strong enough to follow"))
    elif dp > dm:
        out.append(_vote("ADX(14)", 1,
                         f"ADX {adx:.0f} with +DI above -DI — trending up"))
    else:
        out.append(_vote("ADX(14)", -1,
                         f"ADX {adx:.0f} with -DI above +DI — trending down"))

    ao = r.get("ao")
    if ao is None:
        out.append(_vote("Awesome Oscillator", 0, "not available"))
    elif ao > 0 and _rising(r, "ao"):
        out.append(_vote("Awesome Oscillator", 1, "AO is positive and expanding"))
    elif ao < 0 and _rising(r, "ao") is False:
        out.append(_vote("Awesome Oscillator", -1, "AO is negative and deepening"))
    else:
        out.append(_vote("Awesome Oscillator", 0, "AO gives no confirmed signal"))

    h = r.get("macd_hist")
    if h is None:
        out.append(_vote("MACD(12,26)", 0, "not available"))
    else:
        out.append(_vote("MACD(12,26)", 1 if h > 0 else (-1 if h < 0 else 0),
                         "MACD is above its signal line" if h > 0 else
                         ("MACD is below its signal line" if h < 0
                          else "MACD is on its signal line")))

    sr = r.get("stochrsi")
    if sr is None:
        out.append(_vote("Stochastic RSI(14)", 0, "not available"))
    elif sr < 20 and _rising(r, "stochrsi"):
        out.append(_vote("Stochastic RSI(14)", 1,
                         f"StochRSI {sr:.0f} is oversold and turning up"))
    elif sr > 80 and _rising(r, "stochrsi") is False:
        out.append(_vote("Stochastic RSI(14)", -1,
                         f"StochRSI {sr:.0f} is overbought and turning down"))
    else:
        out.append(_vote("Stochastic RSI(14)", 0, _idle(sr, 20, 80, "StochRSI")))

    uo = r.get("uo")
    if uo is None:
        out.append(_vote("Ultimate Oscillator", 0, "not available"))
    elif uo > 70:
        out.append(_vote("Ultimate Oscillator", 1, f"UO {uo:.0f} confirms buying pressure"))
    elif uo < 30:
        out.append(_vote("Ultimate Oscillator", -1, f"UO {uo:.0f} confirms selling pressure"))
    else:
        out.append(_vote("Ultimate Oscillator", 0, f"UO {uo:.0f} is neutral"))

    return out


def gauge(votes: list[dict]) -> dict:
    if not votes:
        return {"buy": 0, "neutral": 0, "sell": 0, "mean": 0.0, "band": "Hold"}
    buy = sum(1 for v in votes if v["vote"] > 0)
    sell = sum(1 for v in votes if v["vote"] < 0)
    mean = (buy - sell) / len(votes)
    return {"buy": buy, "neutral": len(votes) - buy - sell, "sell": sell,
            "mean": mean, "band": band_for(mean)}


def rate_row(r: dict) -> dict:
    mav, ov = ma_votes(r), osc_votes(r)
    ma_g, osc_g = gauge(mav), gauge(ov)
    # summary = mean of the two gauge means, TradingView's own aggregation:
    # it gives the 11 oscillators equal weight to the 15 averages instead of
    # letting the larger group dominate
    summary_mean = (ma_g["mean"] + osc_g["mean"]) / 2
    return {"ma": ma_g, "osc": osc_g, "ma_votes": mav, "osc_votes": ov,
            "summary_mean": summary_mean, "summary_band": band_for(summary_mean)}


# --- vectorised path -------------------------------------------------------
# rate_row() builds a plain-English reason for all 26 components, which is
# what the reader wants but costs ~26 f-strings per row — unusable across the
# 1M+ row history that validate.py replays. vote_frame() computes the same
# votes as pure polars expressions and drops the prose. A test pins the two
# against each other so the duplicated rules cannot silently drift apart.
def _sign_vs(col: str) -> pl.Expr:
    c = pl.col(col)
    return (pl.when(c.is_null()).then(0)
            .when(pl.col("close") > c).then(1)
            .when(pl.col("close") < c).then(-1).otherwise(0))


def _extreme(col: str, lo: float, hi: float, *, low_is_buy: bool = True) -> pl.Expr:
    """Contrarian oscillator rule: oversold+turning-up buys, overbought+
    turning-down sells; everything else abstains."""
    c, p = pl.col(col), pl.col(f"prev_{col}")
    buy, sell = ((c < lo) & (c > p), (c > hi) & (c < p)) if low_is_buy else \
                ((c > hi) & (c > p), (c < lo) & (c < p))
    return (pl.when(buy.fill_null(False)).then(1)  # noqa: FBT003
            .when(sell.fill_null(False)).then(-1).otherwise(0))


def _direction(col: str) -> pl.Expr:
    c, p = pl.col(col), pl.col(f"prev_{col}")
    return (pl.when((c > p).fill_null(False)).then(1)   # noqa: FBT003
            .when((c < p).fill_null(False)).then(-1).otherwise(0))


def _confirmed(col: str) -> pl.Expr:
    """Signed level that must also be moving the same way (AO, Bull Bear)."""
    c, p = pl.col(col), pl.col(f"prev_{col}")
    return (pl.when(((c > 0) & (c > p)).fill_null(False)).then(1)  # noqa: FBT003
            .when(((c < 0) & (c < p)).fill_null(False)).then(-1).otherwise(0))


def vote_frame(df: pl.DataFrame) -> pl.DataFrame:
    """ma_mean / osc_mean / summary_mean + bands, without the reason text."""
    d = add_prev(df)
    ma = [_sign_vs(f"{k}{n}") for k in ("sma", "ema") for n in MA_PERIODS]
    ma += [_sign_vs(f"vwma{n}") for n in VWMA_PERIODS]
    a, b = pl.col("ichi_a"), pl.col("ichi_b")
    top, bot = pl.max_horizontal(a, b), pl.min_horizontal(a, b)
    ma.append(pl.when(a.is_null() | b.is_null()).then(0)
              .when(pl.col("close") > top).then(1)
              .when(pl.col("close") < bot).then(-1).otherwise(0))

    adx, dp, dm = pl.col("adx14"), pl.col("di_plus"), pl.col("di_minus")
    h = pl.col("macd_hist")
    uo = pl.col("uo")
    osc = [_extreme("rsi14", 30, 70)]
    osc += [_extreme(f"cci{n}", -100, 100) for n in CCI_PERIODS]
    osc += [
        (pl.when(adx.is_null() | dp.is_null() | dm.is_null() | (adx < 20)).then(0)
         .when(dp > dm).then(1).otherwise(-1)),
        _confirmed("ao"),
        pl.when(h.is_null()).then(0).when(h > 0).then(1)
        .when(h < 0).then(-1).otherwise(0),
        _extreme("stochrsi", 20, 80),
        pl.when(uo.is_null()).then(0).when(uo > 70).then(1)
        .when(uo < 30).then(-1).otherwise(0),
    ]
    d = d.with_columns(
        (sum(ma) / len(ma)).cast(pl.Float64).alias("ma_mean"),
        (sum(osc) / len(osc)).cast(pl.Float64).alias("osc_mean"))
    d = d.with_columns(
        ((pl.col("ma_mean") + pl.col("osc_mean")) / 2).alias("summary_mean"))
    return d.with_columns([
        (pl.when(pl.col(c) >= _STRONG).then(pl.lit("Strong Buy"))
         .when(pl.col(c) >= _WEAK).then(pl.lit("Buy"))
         .when(pl.col(c) > -_WEAK).then(pl.lit("Hold"))
         .when(pl.col(c) > -_STRONG).then(pl.lit("Sell"))
         .otherwise(pl.lit("Strong Sell"))).alias(c.replace("_mean", "_band"))
        for c in ("ma_mean", "osc_mean", "summary_mean")])


def rate_frame(df: pl.DataFrame) -> pl.DataFrame:
    res = [rate_row(r) for r in add_prev(df).iter_rows(named=True)]
    return df.with_columns(
        pl.Series("ma_mean", [x["ma"]["mean"] for x in res]),
        pl.Series("ma_band", [x["ma"]["band"] for x in res]),
        pl.Series("osc_mean", [x["osc"]["mean"] for x in res]),
        pl.Series("osc_band", [x["osc"]["band"] for x in res]),
        pl.Series("summary_mean", [x["summary_mean"] for x in res]),
        pl.Series("summary_band", [x["summary_band"] for x in res]),
        pl.Series("gauge_votes", [
            json.dumps({"ma": x["ma_votes"], "osc": x["osc_votes"]}) for x in res]),
    )
