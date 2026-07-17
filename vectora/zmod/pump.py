"""Pump-phase classification and scoring (spec §13).

Phases are ordered rules over one day's cross-section (first match wins):
collapse (post-run crash) > distribution (run-up, volume no longer
confirming) > markup (run-up on expanding volume) > quiet. The 0-100 score
is the product of cross-sectional percentile ranks of 21d run-up and 5/21
volume expansion — a name must be extreme on BOTH to score high — with a
1.25x boost for Z/B/N categories (documented manipulation incidence).
Purely descriptive of public data; a warning surface, never an accusation.
"""
import polars as pl

RUNUP = 0.25
FAST_RUNUP = 0.20   # 10-day; DSE pumps often complete inside two weeks
VOL_EXPAND = 1.3
COLLAPSE_DROP = -0.15
COLLAPSE_PRIOR_RUN = 0.20
BOOST_CATEGORIES = {"Z", "B", "N"}
BOOST = 1.25


def phase_and_score(day: pl.DataFrame, categories: dict) -> pl.DataFrame:
    df = day.with_columns(
        pl.col("ret_10d").fill_null(0.0), pl.col("ret_21d").fill_null(0.0),
        pl.col("ret_63d").fill_null(0.0),
        pl.col("vol_ratio_5_21").fill_null(1.0),
        pl.col("obv_slope_21d").fill_null(0.0),
    )
    phase = (
        pl.when((pl.col("ret_10d") < COLLAPSE_DROP)
                & (pl.col("ret_63d") > COLLAPSE_PRIOR_RUN))
        .then(pl.lit("collapse"))
        .when(((pl.col("ret_21d") > RUNUP)
               | (pl.col("ret_10d") > FAST_RUNUP))
              & (pl.col("obv_slope_21d") < 0))
        .then(pl.lit("distribution"))
        .when(((pl.col("ret_21d") > RUNUP)
               | (pl.col("ret_10d") > FAST_RUNUP))
              & (pl.col("vol_ratio_5_21") > VOL_EXPAND))
        .then(pl.lit("markup"))
        .otherwise(pl.lit("quiet"))
    )
    n = df.height
    rank_ret = pl.col("ret_21d").rank("average") / n
    rank_vol = pl.col("vol_ratio_5_21").rank("average") / n
    df = df.with_columns(phase.alias("phase"),
                         (rank_ret * rank_vol * 100).alias("_raw"))
    boost = pl.col("symbol").map_elements(
        lambda s: BOOST if categories.get(s) in BOOST_CATEGORIES else 1.0,
        return_dtype=pl.Float64)
    df = df.with_columns(
        (pl.col("_raw") * boost).clip(0.0, 100.0).alias("score"))
    # null-heavy rows (new listings) carry no evidence: zero them
    null_mask = (day["ret_21d"].is_null()
                 & day["vol_ratio_5_21"].is_null()).rename("_nullrow")
    df = df.with_columns(null_mask).with_columns(
        pl.when(pl.col("_nullrow")).then(0.0)
        .otherwise(pl.col("score")).alias("score"))
    return df.select([c for c in df.columns if c not in ("_raw", "_nullrow")])
