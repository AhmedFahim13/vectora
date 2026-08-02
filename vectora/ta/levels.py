"""Pivot points and swing support/resistance (spec: Phase 6D).

Two independent notions of "level", because traders use both:

1. PIVOTS — arithmetic levels derived from the previous completed period's
   high/low/close. They are not discovered from price action; they are
   computed, which is exactly why so many participants watch the same
   numbers. TradingView's daily-chart default derives them from the
   previous MONTH, and that is what is used here.

2. SWING LEVELS — the actual highest high and lowest low over trailing
   windows (20d, 60d, 252d). These are where price genuinely turned.

Both are trailing-only. Pivots use the previous month, never the current
one, and swing windows exclude today so a fresh high cannot be reported as
its own resistance.
"""
import polars as pl

SWING_WINDOWS = (20, 60, 252)


def add_pivots(df: pl.DataFrame) -> pl.DataFrame:
    """Classic and Fibonacci pivots from the previous calendar month."""
    d = df.sort(["symbol", "date"]).with_columns(
        pl.col("date").dt.truncate("1mo").alias("_month"))
    monthly = (d.group_by(["symbol", "_month"])
               .agg(pl.col("high").max().alias("_mh"),
                    pl.col("low").min().alias("_ml"),
                    pl.col("close").last().alias("_mc"))
               .sort(["symbol", "_month"]))
    # shift within symbol: the month a row belongs to must use the month
    # BEFORE it, otherwise the level is computed from candles that had not
    # printed yet
    monthly = monthly.with_columns(
        pl.col("_mh").shift(1).over("symbol").alias("ph"),
        pl.col("_ml").shift(1).over("symbol").alias("plo"),
        pl.col("_mc").shift(1).over("symbol").alias("pc"))
    d = d.join(monthly.select(["symbol", "_month", "ph", "plo", "pc"]),
               on=["symbol", "_month"], how="left")

    ph, plo, pc = pl.col("ph"), pl.col("plo"), pl.col("pc")
    rng = ph - plo
    pivot = (ph + plo + pc) / 3
    d = d.with_columns(pivot.alias("pivot_point"))
    p = pl.col("pivot_point")
    return d.with_columns(
        (2 * p - plo).alias("r1"), (p + rng).alias("r2"),
        (ph + 2 * (p - plo)).alias("r3"),
        (2 * p - ph).alias("s1"), (p - rng).alias("s2"),
        (plo - 2 * (ph - p)).alias("s3"),
        # Fibonacci variants share the pivot but scale the range
        (p + 0.382 * rng).alias("fib_r1"), (p + 0.618 * rng).alias("fib_r2"),
        (p - 0.382 * rng).alias("fib_s1"), (p - 0.618 * rng).alias("fib_s2"),
    ).drop(["_month", "ph", "plo", "pc"])


def add_swing_levels(df: pl.DataFrame) -> pl.DataFrame:
    """Trailing swing highs/lows. Today is excluded from its own window."""
    d = df.sort(["symbol", "date"])
    exprs = []
    for n in SWING_WINDOWS:
        exprs += [
            pl.col("high").shift(1).rolling_max(n).over("symbol")
            .alias(f"hi_{n}d"),
            pl.col("low").shift(1).rolling_min(n).over("symbol")
            .alias(f"lo_{n}d")]
    return d.with_columns(exprs)


def add_level_distances(df: pl.DataFrame) -> pl.DataFrame:
    """Distance to the nearest level above and below, as a fraction of price.

    This is the number that actually matters for position sizing: how much
    room there is before price meets a level, in percent, rather than the
    raw level itself.
    """
    close = pl.col("close")
    above = ["r1", "r2", "r3", "hi_20d", "hi_60d", "hi_252d", "pivot_point"]
    below = ["s1", "s2", "s3", "lo_20d", "lo_60d", "lo_252d", "pivot_point"]
    # a level only counts as resistance while it sits above price
    res = pl.min_horizontal([
        pl.when(pl.col(c) > close).then(pl.col(c)) for c in above])
    sup = pl.max_horizontal([
        pl.when(pl.col(c) < close).then(pl.col(c)) for c in below])
    return df.with_columns(
        res.alias("nearest_res"), sup.alias("nearest_sup")
    ).with_columns(
        ((pl.col("nearest_res") - close) / close).alias("room_up"),
        ((close - pl.col("nearest_sup")) / close).alias("room_dn"))


def add_all(df: pl.DataFrame) -> pl.DataFrame:
    return add_level_distances(add_swing_levels(add_pivots(df)))
