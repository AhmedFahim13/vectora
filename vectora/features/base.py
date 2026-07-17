"""Price panel loader with the canonical daily-return column.

Return ladder (corporate-action correctness, best available per row):
1. scraped rows (ycp present & > 0): ret = close/ycp - 1 — DSE's YCP is
   ex-date adjusted, so these are corporate-action safe.
2. backfill rows with adjusted-close coverage: adj_close chain — the
   Mendeley adjusted series makes splits/rights invisible, replacing the
   old clip approximation for 2012-2026.
3. remainder: unadjusted close chain.
All returns clip to +/-RET_CLIP as a residual data-error guard (real DSE
moves cannot exceed the circuit band).
"""
from pathlib import Path

import polars as pl

from vectora.settings import ADJUSTED_PARQUET

RET_CLIP = 0.12

_PANEL_SQL = """
    SELECT symbol, date, open, high, low, close, ycp, trades, value_mn, volume
    FROM prices
    WHERE close IS NOT NULL AND close > 0
    ORDER BY symbol, date
"""


def load_panel(con) -> pl.DataFrame:
    df = con.execute(_PANEL_SQL).pl()
    adj_path = Path(ADJUSTED_PARQUET)
    if adj_path.exists():
        adj = pl.read_parquet(adj_path)
        df = df.join(adj, on=["symbol", "date"], how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("adj_close"))
    prev_close = pl.col("close").shift(1).over("symbol")
    prev_adj = pl.col("adj_close").shift(1).over("symbol")
    raw_ret = (
        pl.when(pl.col("ycp").is_not_null() & (pl.col("ycp") > 0))
        .then(pl.col("close") / pl.col("ycp") - 1)
        .when(pl.col("adj_close").is_not_null() & prev_adj.is_not_null()
              & (prev_adj > 0))
        .then(pl.col("adj_close") / prev_adj - 1)
        .otherwise(pl.col("close") / prev_close - 1)
    )
    return df.with_columns(
        raw_ret.clip(-RET_CLIP, RET_CLIP).alias("ret")
    ).drop("adj_close")
