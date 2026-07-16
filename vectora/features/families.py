# vectora/features/families.py
"""Feature computation functions. Contract: every fn takes the panel frame
(sorted by symbol,date; includes symbol/date/open/high/low/close/ycp/trades/
value_mn/volume/ret and joined sector/first_seen) plus params, and returns
the frame with ONE new column named `name`. Per-symbol ops use .over("symbol");
per-date (cross-sectional) ops use .over("date"). All windows are trailing —
the leakage test (Task 7) enforces it."""
import polars as pl

_EPS = 1e-12


def _sym(expr: pl.Expr) -> pl.Expr:
    return expr.over("symbol")


# ---- momentum -------------------------------------------------------------
def ret_nd(df, name, days):
    logret = (pl.col("ret") + 1).log()
    comp = logret.rolling_sum(days).exp() - 1
    return df.with_columns(_sym(comp).alias(name))


def rsi(df, name, days):
    up = pl.when(pl.col("ret") > 0).then(pl.col("ret")).otherwise(0.0)
    dn = pl.when(pl.col("ret") < 0).then(-pl.col("ret")).otherwise(0.0)
    rs = _sym(up.rolling_mean(days)) / (_sym(dn.rolling_mean(days)) + _EPS)
    return df.with_columns((100 - 100 / (1 + rs)).alias(name))


def dist_from_rolling_max(df, name, days):
    e = pl.col("close") / _sym(pl.col("close").rolling_max(days)) - 1
    return df.with_columns(e.alias(name))


def dist_from_rolling_min(df, name, days):
    e = pl.col("close") / _sym(pl.col("close").rolling_min(days)) - 1
    return df.with_columns(e.alias(name))


# ---- volatility -----------------------------------------------------------
def ret_std(df, name, days):
    return df.with_columns(_sym(pl.col("ret").rolling_std(days)).alias(name))


def ratio_of_stds(df, name, short, long):
    e = _sym(pl.col("ret").rolling_std(short)) / (
        _sym(pl.col("ret").rolling_std(long)) + _EPS)
    return df.with_columns(e.alias(name))


def atr(df, name, days):
    prev_close = _sym(pl.col("close").shift(1))
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return df.with_columns(_sym(tr.rolling_mean(days)).alias(name))


def avg_range_pct(df, name, days):
    rng = (pl.col("high") - pl.col("low")) / (pl.col("close") + _EPS)
    return df.with_columns(_sym(rng.rolling_mean(days)).alias(name))


def limit_lock_count(df, name, days, band):
    locked = (pl.col("ret").abs() >= band).cast(pl.Int32)
    return df.with_columns(_sym(locked.rolling_sum(days)).alias(name))


# ---- liquidity ------------------------------------------------------------
def rolling_median_col(df, name, col, days):
    return df.with_columns(_sym(pl.col(col).rolling_median(days)).alias(name))


def amihud(df, name, days):
    daily = pl.col("ret").abs() / (pl.col("value_mn") + _EPS)
    return df.with_columns(_sym(daily.rolling_mean(days)).alias(name))


def zero_volume_days(df, name, days):
    z = (pl.col("volume").fill_null(0) == 0).cast(pl.Int32)
    return df.with_columns(_sym(z.rolling_sum(days)).alias(name))


def zscore_col(df, name, col, days):
    mean = _sym(pl.col(col).rolling_mean(days))
    std = _sym(pl.col(col).rolling_std(days))
    return df.with_columns(((pl.col(col) - mean) / (std + _EPS)).alias(name))


# ---- volume/flow ----------------------------------------------------------
def volume_ratio(df, name, short, long):
    e = _sym(pl.col("volume").rolling_mean(short)) / (
        _sym(pl.col("volume").rolling_mean(long)) + _EPS)
    return df.with_columns(e.alias(name))


def obv_slope(df, name, days):
    signed = pl.col("volume").fill_null(0) * pl.col("ret").sign().fill_null(0)
    obv = _sym(signed.cum_sum())
    slope = (obv - _sym(obv.shift(days))) / days
    norm = slope / (_sym(pl.col("volume").rolling_mean(days)) + _EPS)
    return df.with_columns(norm.alias(name))


def updown_volume_ratio(df, name, days):
    upv = pl.when(pl.col("ret") > 0).then(pl.col("volume")).otherwise(0)
    dnv = pl.when(pl.col("ret") < 0).then(pl.col("volume")).otherwise(0)
    e = _sym(upv.rolling_sum(days)) / (_sym(dnv.rolling_sum(days)) + _EPS)
    return df.with_columns(e.log1p().alias(name))  # log-compress the ratio


def vwap_deviation(df, name, days):
    # value_mn is in millions of taka; volume in shares -> vwap in taka
    vwap = _sym((pl.col("value_mn") * 1e6).rolling_sum(days)) / (
        _sym(pl.col("volume").rolling_sum(days)) + _EPS)
    return df.with_columns((pl.col("close") / (vwap + _EPS) - 1).alias(name))


# ---- cross-sectional (per-date) --------------------------------------------
def cross_rank(df, name, of):
    e = (pl.col(of).rank("average") / pl.col(of).count()).over("date")
    return df.with_columns(e.alias(name))


def sector_mean(df, name, of):
    return df.with_columns(
        pl.col(of).mean().over(["date", "sector"]).alias(name))


def minus_sector_mean(df, name, of):
    e = pl.col(of) - pl.col(of).mean().over(["date", "sector"])
    return df.with_columns(e.alias(name))


def market_breadth_above_ma(df, name, days):
    above = (pl.col("close") > _sym(pl.col("close").rolling_mean(days))).cast(pl.Int8)
    return df.with_columns(above.mean().over("date").alias(name))


# ---- calendar / structure ---------------------------------------------------
def day_of_week(df, name):
    return df.with_columns(pl.col("date").dt.weekday().alias(name))


def month_of_year(df, name):
    return df.with_columns(pl.col("date").dt.month().alias(name))


def days_since_first_seen(df, name):
    e = (pl.col("date") - pl.col("first_seen")).dt.total_days()
    return df.with_columns(e.alias(name))


def log_close(df, name):
    return df.with_columns(pl.col("close").log().alias(name))


def overnight_gap(df, name):
    prev = _sym(pl.col("close").shift(1))
    return df.with_columns((pl.col("open") / (prev + _EPS) - 1).alias(name))


def close_in_range(df, name):
    rng = pl.col("high") - pl.col("low")
    e = pl.when(rng > 0).then((pl.col("close") - pl.col("low")) / rng).otherwise(0.5)
    return df.with_columns(e.alias(name))


def dist_from_sma(df, name, days):
    e = pl.col("close") / (_sym(pl.col("close").rolling_mean(days)) + _EPS) - 1
    return df.with_columns(e.alias(name))


def sma_cross_state(df, name, short, long):
    e = (_sym(pl.col("close").rolling_mean(short))
         > _sym(pl.col("close").rolling_mean(long))).cast(pl.Int8)
    return df.with_columns(e.alias(name))


def ycp_adjustment_flag(df, name):
    prev = _sym(pl.col("close").shift(1))
    diverges = (
        pl.col("ycp").is_not_null() & prev.is_not_null()
        & ((pl.col("ycp") - prev).abs() / (prev + _EPS) > 0.005)
    )
    return df.with_columns(diverges.cast(pl.Int8).alias(name))


FNS = {f.__name__: f for f in [
    ret_nd, rsi, dist_from_rolling_max, dist_from_rolling_min,
    ret_std, ratio_of_stds, atr, avg_range_pct, limit_lock_count,
    rolling_median_col, amihud, zero_volume_days, zscore_col,
    volume_ratio, obv_slope, updown_volume_ratio, vwap_deviation,
    cross_rank, sector_mean, minus_sector_mean, market_breadth_above_ma,
    day_of_week, month_of_year, days_since_first_seen, log_close,
    overnight_gap, close_in_range, dist_from_sma, sma_cross_state,
    ycp_adjustment_flag,
]}


def apply(df: pl.DataFrame, name: str, fn: str, params: dict) -> pl.DataFrame:
    return FNS[fn](df, name, **params)
