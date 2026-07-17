"""Panel-derived daily market state (spec §11 inputs).

No dependence on scraped index history (which only starts 2026-07): the
market is summarized directly from the cross-section — median return, a
synthetic equal-weight level, breadth above own 50DMA, rolling volatility
percentile, and an activity z-score on TOTAL VOLUME (traded value is null
throughout the backfill era, volume is not).
"""
import polars as pl

from vectora.features import base

MIN_SYMBOLS = 30          # dates with fewer cross-sectional obs are noise
VOL_PCTL_WINDOW = 252


def market_state(con) -> pl.DataFrame:
    panel = base.load_panel(con)
    per_symbol = panel.with_columns(
        (pl.col("close") > pl.col("close").rolling_mean(50).over("symbol"))
        .cast(pl.Int8).alias("above_ma50"))
    daily = (
        per_symbol.group_by("date")
        .agg(
            pl.col("ret").median().alias("med_ret"),
            pl.col("above_ma50").mean().alias("breadth"),
            pl.col("volume").sum().alias("total_volume"),
            pl.len().alias("n_symbols"),
        )
        .filter(pl.col("n_symbols") >= MIN_SYMBOLS)
        .sort("date")
    )
    daily = daily.with_columns(
        (pl.col("med_ret").fill_null(0) + 1).cum_prod().alias("mkt_level"))
    daily = daily.with_columns(
        pl.col("mkt_level").rolling_mean(50).alias("ma50"),
        pl.col("mkt_level").rolling_mean(200).alias("ma200"),
        (pl.col("mkt_level") / pl.col("mkt_level").shift(21) - 1)
        .alias("ret_21d"),
        pl.col("med_ret").rolling_std(21).alias("vol_21d"),
        ((pl.col("total_volume")
          - pl.col("total_volume").rolling_mean(63))
         / (pl.col("total_volume").rolling_std(63) + 1e-9))
        .alias("activity_z"),
    )
    # rolling percentile rank of vol_21d within the trailing year
    daily = daily.with_columns(
        pl.col("vol_21d").rolling_map(
            lambda s: float((s < s[-1]).sum() / max(len(s) - 1, 1)),
            window_size=VOL_PCTL_WINDOW, min_samples=63)
        .alias("vol_pctile"))
    return daily
