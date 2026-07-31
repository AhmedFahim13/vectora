"""Classical technical indicators as polars expressions (spec: Phase 6B).

Every indicator is trailing-only — a value at row t uses rows <= t, which
the lookahead test pins. No TA-Lib: a C dependency would break the
zero-cost CI, and each of these is a handful of lines.

Columns added by add_all():
  macd, macd_signal, macd_hist      MACD(12,26,9) on close
  rsi14                             Wilder-style RSI, 14
  bb_mid, bb_up, bb_lo, bb_pos,     Bollinger(20, 2sd); bb_pos is 0 at the
  bb_width                          lower band, 1 at the upper
  ma_fast, ma_slow, ma_cross_up,    SMA(20)/SMA(50) + fresh-cross flags
  ma_cross_dn
  st_dir, st_line                   SuperTrend(10, 3) direction and stop
  candle_bull, candle_bear          engulfing / hammer / shooting-star flags
"""
import numpy as np
import polars as pl

MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
RSI_N = 14
BB_N, BB_SD = 20, 2.0
MA_FAST, MA_SLOW = 20, 50
ST_N, ST_MULT = 10, 3.0
CROSS_LOOKBACK = 5      # a cross counts as "fresh" for this many days


def _ema(col: str, span: int) -> pl.Expr:
    return pl.col(col).ewm_mean(span=span, adjust=False).over("symbol")


def add_all(df: pl.DataFrame) -> pl.DataFrame:
    d = df.sort(["symbol", "date"])

    # --- MACD ---
    d = d.with_columns(
        (_ema("close", MACD_FAST) - _ema("close", MACD_SLOW)).alias("macd"))
    d = d.with_columns(
        pl.col("macd").ewm_mean(span=MACD_SIG, adjust=False).over("symbol")
        .alias("macd_signal"))
    d = d.with_columns((pl.col("macd") - pl.col("macd_signal"))
                       .alias("macd_hist"))

    # --- RSI (Wilder smoothing via ewm alpha=1/n) ---
    chg = pl.col("close").diff().over("symbol")
    gain = pl.when(chg > 0).then(chg).otherwise(0.0)
    loss = pl.when(chg < 0).then(-chg).otherwise(0.0)
    d = d.with_columns(
        gain.ewm_mean(alpha=1 / RSI_N, adjust=False).over("symbol").alias("_g"),
        loss.ewm_mean(alpha=1 / RSI_N, adjust=False).over("symbol").alias("_l"))
    d = d.with_columns(
        (100 - 100 / (1 + pl.col("_g") / (pl.col("_l") + 1e-12))).alias("rsi14"))

    # --- Bollinger ---
    mid = pl.col("close").rolling_mean(BB_N).over("symbol")
    sd = pl.col("close").rolling_std(BB_N).over("symbol")
    d = d.with_columns(mid.alias("bb_mid"), sd.alias("_sd"))
    d = d.with_columns(
        (pl.col("bb_mid") + BB_SD * pl.col("_sd")).alias("bb_up"),
        (pl.col("bb_mid") - BB_SD * pl.col("_sd")).alias("bb_lo"))
    d = d.with_columns(
        ((pl.col("close") - pl.col("bb_lo"))
         / (pl.col("bb_up") - pl.col("bb_lo") + 1e-12)).alias("bb_pos"),
        ((pl.col("bb_up") - pl.col("bb_lo"))
         / (pl.col("bb_mid") + 1e-12)).alias("bb_width"))

    # --- MA cross ---
    d = d.with_columns(
        pl.col("close").rolling_mean(MA_FAST).over("symbol").alias("ma_fast"),
        pl.col("close").rolling_mean(MA_SLOW).over("symbol").alias("ma_slow"))
    above = (pl.col("ma_fast") > pl.col("ma_slow")).cast(pl.Int8)
    d = d.with_columns(above.alias("_above"))
    prev = pl.col("_above").shift(1).over("symbol")
    d = d.with_columns(
        ((pl.col("_above") == 1) & (prev == 0)).cast(pl.Int8).alias("_xup"),
        ((pl.col("_above") == 0) & (prev == 1)).cast(pl.Int8).alias("_xdn"))
    d = d.with_columns(
        (pl.col("_xup").rolling_sum(CROSS_LOOKBACK).over("symbol") > 0)
        .cast(pl.Int8).alias("ma_cross_up"),
        (pl.col("_xdn").rolling_sum(CROSS_LOOKBACK).over("symbol") > 0)
        .cast(pl.Int8).alias("ma_cross_dn"))

    # --- SuperTrend(10,3): true recursive final bands ---
    prev_close = pl.col("close").shift(1).over("symbol")
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs())
    d = d.with_columns(
        tr.ewm_mean(alpha=1 / ST_N, adjust=False).over("symbol").alias("_atr"))
    hl2 = (pl.col("high") + pl.col("low")) / 2
    d = d.with_columns(
        (hl2 - ST_MULT * pl.col("_atr")).alias("_basic_lo"),
        (hl2 + ST_MULT * pl.col("_atr")).alias("_basic_up"))
    # SuperTrend needs its recursion: the final bands ratchet (the lower band
    # only rises while price holds above it, the upper only falls) and the
    # direction flips when a final band is breached. There is no faithful
    # vectorised form, so this runs per symbol over numpy arrays.
    dirs, lines = [], []
    for (_sym,), g in d.group_by(["symbol"], maintain_order=True):
        lo = g["_basic_lo"].to_numpy()
        up = g["_basic_up"].to_numpy()
        close = g["close"].to_numpy()
        n = len(close)
        f_lo = np.empty(n)
        f_up = np.empty(n)
        dir_ = np.ones(n, dtype=np.int8)
        for i in range(n):
            if i == 0 or np.isnan(lo[i]) or np.isnan(up[i]):
                f_lo[i] = lo[i] if not np.isnan(lo[i]) else close[i]
                f_up[i] = up[i] if not np.isnan(up[i]) else close[i]
                dir_[i] = 1
                continue
            f_lo[i] = (max(lo[i], f_lo[i - 1])
                       if close[i - 1] > f_lo[i - 1] else lo[i])
            f_up[i] = (min(up[i], f_up[i - 1])
                       if close[i - 1] < f_up[i - 1] else up[i])
            if close[i] > f_up[i - 1]:
                dir_[i] = 1
            elif close[i] < f_lo[i - 1]:
                dir_[i] = -1
            else:
                dir_[i] = dir_[i - 1]
        dirs.append(dir_)
        lines.append(np.where(dir_ == 1, f_lo, f_up))
    d = d.with_columns(
        pl.Series("st_dir", np.concatenate(dirs)).cast(pl.Int8),
        pl.Series("st_line", np.concatenate(lines)))

    # --- candlesticks (engulfing, hammer, shooting star) ---
    body = (pl.col("close") - pl.col("open"))
    rng = (pl.col("high") - pl.col("low")) + 1e-12
    p_open = pl.col("open").shift(1).over("symbol")
    p_close = pl.col("close").shift(1).over("symbol")
    bull_engulf = ((p_close < p_open) & (pl.col("close") > pl.col("open"))
                   & (pl.col("close") >= p_open) & (pl.col("open") <= p_close))
    bear_engulf = ((p_close > p_open) & (pl.col("close") < pl.col("open"))
                   & (pl.col("close") <= p_open) & (pl.col("open") >= p_close))
    lower_wick = pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low")
    upper_wick = pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close"))
    hammer = (lower_wick > 2 * body.abs()) & (upper_wick < body.abs()) & (body > 0)
    shooting = (upper_wick > 2 * body.abs()) & (lower_wick < body.abs()) & (body < 0)
    d = d.with_columns(
        ((bull_engulf | hammer).cast(pl.Int8)).alias("candle_bull"),
        ((bear_engulf | shooting).cast(pl.Int8)).alias("candle_bear"),
        (body / rng).alias("candle_body_frac"))

    drop = [c for c in d.columns if c.startswith("_")]
    return d.drop(drop)
