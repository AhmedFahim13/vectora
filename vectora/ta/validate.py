"""Measure what each technical band HISTORICALLY did (spec: Phase 6B).

Replays the rating across all available history, joins each rating to the
same forward outcome the models are trained on (did price gain X% within H
trading rows), and records hit rate per band beside the market-wide base
rate. This is the honesty layer: a band is only worth showing if it beats
the base rate, and the page prints both numbers so the reader can judge.

Both views are measured:
- band     : the absolute TradingView-style posture
- rank     : cross-sectional decile of the score within each day, which
             asks the more useful question "is this stock better placed
             than its peers TODAY" rather than "is the market rising"
"""
import polars as pl

from vectora import db as vdb
from vectora import labels as lab
from vectora.features import base
from vectora.ta import indicators, rating

BANDS = ("Strong Sell", "Sell", "Hold", "Buy", "Strong Buy")


def run(con, horizons=(5, 10), threshold: float = 0.05) -> dict:
    panel = base.load_panel(con).select(
        ["symbol", "date", "open", "high", "low", "close"])
    ind = indicators.add_all(panel).filter(pl.col("ma_slow").is_not_null())
    rated = rating.rate_frame(ind).select(["symbol", "date", "ta_score",
                                           "ta_band"])
    # cross-sectional decile within each day: 0 = worst placed, 9 = best
    rated = rated.with_columns(
        ((pl.col("ta_score").rank("average") / pl.len() * 10)
         .floor().clip(0, 9).cast(pl.Int8)).over("date").alias("ta_decile"))

    labeled = lab.make_labels(panel, thresholds=(threshold,),
                              horizons=tuple(horizons), continuous=True)
    pct = round(threshold * 100)

    rows = []
    for h in horizons:
        col = f"y_g{pct}_h{h}"
        joined = (rated.join(labeled.select(["symbol", "date", col]),
                             on=["symbol", "date"], how="inner")
                  .filter(pl.col(col).is_not_null()))
        if joined.height == 0:
            continue
        base_rate = float(joined[col].mean())
        for band in BANDS:
            sub = joined.filter(pl.col("ta_band") == band)
            if sub.height == 0:
                continue
            rows.append({
                "band": band, "horizon": h, "n": int(sub.height),
                "hit_rate": float(sub[col].mean()), "base_rate": base_rate,
                "mean_fwd": float(sub[col].mean()) - base_rate,
            })
        # top and bottom decile of the cross-section, stored as pseudo-bands
        for label, dec in (("Top decile", 9), ("Bottom decile", 0)):
            sub = joined.filter(pl.col("ta_decile") == dec)
            if sub.height == 0:
                continue
            rows.append({
                "band": label, "horizon": h, "n": int(sub.height),
                "hit_rate": float(sub[col].mean()), "base_rate": base_rate,
                "mean_fwd": float(sub[col].mean()) - base_rate,
            })
    if rows:
        vdb.upsert(con, "ta_band_stats", rows)
    return {"rows_scored": rated.height, "bands": len(rows)}
