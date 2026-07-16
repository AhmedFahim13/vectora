"""Price panel loader with the canonical daily-return column.

Return definition (Phase 2 approximation, upgrade point for the Phase 3
corporate-action engine — change it HERE and every feature inherits it):
- scraped rows (ycp present & > 0): ret = close/ycp - 1. DSE's YCP is
  already adjusted on ex-dates, so these returns are corporate-action safe.
- backfill rows (no ycp): ret = close/prev_close - 1, clipped to +/-RET_CLIP.
  Unadjusted split/rights gaps get clipped instead of adjusted; the clip
  matches the validation band (circuit ~10% + buffer).
"""
import polars as pl

RET_CLIP = 0.12

_PANEL_SQL = """
    SELECT symbol, date, open, high, low, close, ycp, trades, value_mn, volume
    FROM prices
    WHERE close IS NOT NULL AND close > 0
    ORDER BY symbol, date
"""


def load_panel(con) -> pl.DataFrame:
    df = con.execute(_PANEL_SQL).pl()
    prev_close = pl.col("close").shift(1).over("symbol")
    raw_ret = (
        pl.when(pl.col("ycp").is_not_null() & (pl.col("ycp") > 0))
        .then(pl.col("close") / pl.col("ycp") - 1)
        .otherwise(pl.col("close") / prev_close - 1)
    )
    return df.with_columns(
        raw_ret.clip(-RET_CLIP, RET_CLIP).alias("ret")
    )
