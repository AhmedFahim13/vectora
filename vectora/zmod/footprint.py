# vectora/zmod/footprint.py
"""Pre-announcement footprint detection (spec §12.3).

For every price-sensitive event (materiality 3), record the trailing
5-day mean volume z-score and compounded return measured the day BEFORE
the announcement. The daily watch then flags symbols whose current
trailing footprint exceeds the historical 75th percentile of those
pre-event footprints with positive drift — 'this pattern statistically
resembles what public data looked like before past announcements'.
Descriptive of securities, never an accusation about persons.
"""
import json

import polars as pl

from vectora import db as vdb

WINDOW = 5
PCTL = 0.75
TOP_N = 10
MIN_HISTORY = 3   # need at least this many historical footprints to judge


def _trailing(feats: pl.DataFrame) -> pl.DataFrame:
    """Adds fp_vol_z / fp_ret = stats over the trailing WINDOW rows
    INCLUDING the current row (per symbol)."""
    f = feats.sort(["symbol", "date"])
    return f.with_columns(
        pl.col("volume_z_21d").rolling_mean(WINDOW).over("symbol")
        .alias("fp_vol_z"),
        ((pl.col("ret").fill_null(0) + 1).log().rolling_sum(WINDOW)
         .over("symbol").exp() - 1).alias("fp_ret"),
    )


def compute_event_footprints(con, feats: pl.DataFrame) -> dict:
    pending = con.execute(
        """
        SELECT e.id, e.symbol, e.post_date AS date
        FROM events e
        JOIN event_labels l ON l.event_id = e.id
        LEFT JOIN event_footprints f ON f.event_id = e.id
        WHERE l.materiality >= 3 AND e.symbol IS NOT NULL
          AND f.event_id IS NULL
        """).pl()
    if pending.height == 0:
        return {"computed": 0}
    trail = _trailing(feats).with_columns(
        # footprint the day BEFORE the event: shift trailing stats forward
        pl.col("fp_vol_z").shift(1).over("symbol"),
        pl.col("fp_ret").shift(1).over("symbol"),
    )
    joined = pending.join(
        trail.select(["symbol", "date", "fp_vol_z", "fp_ret"]),
        on=["symbol", "date"], how="inner"
    ).filter(pl.col("fp_vol_z").is_not_null())
    rows = [{"event_id": r["id"], "pre_vol_z": r["fp_vol_z"],
             "pre_ret": r["fp_ret"]}
            for r in joined.iter_rows(named=True)]
    if rows:
        vdb.upsert(con, "event_footprints", rows)
    return {"computed": len(rows)}


def daily_watch(con, feats: pl.DataFrame, date_str: str) -> list[dict]:
    hist = con.execute(
        "SELECT count(*), quantile_cont(pre_vol_z, ?) FROM event_footprints",
        [PCTL]).fetchone()
    if not hist or (hist[0] or 0) < MIN_HISTORY:
        return []
    threshold = hist[1]
    import datetime as dt
    day = dt.date.fromisoformat(date_str)
    today = _trailing(feats).filter(pl.col("date") == day)
    flagged = (today.filter((pl.col("fp_vol_z") > threshold)
                            & (pl.col("fp_ret") > 0))
               .sort("fp_vol_z", descending=True).head(TOP_N))
    return [{"date": date_str, "symbol": r["symbol"], "kind": "footprint",
             "score": round(float(r["fp_vol_z"]), 3), "phase": None,
             "detail": json.dumps({"threshold": round(float(threshold), 3),
                                   "ret_5d": round(float(r["fp_ret"]), 4)})}
            for r in flagged.iter_rows(named=True)]
