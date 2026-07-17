"""Daily Z-scan: pump scores + footprint watch into zwatch (spec §13).
Runs after regime, before predict, so the predict engine can attach
pump warnings to any signal on a flagged name."""
import json
from datetime import date

import polars as pl

from vectora import db as vdb
from vectora.features import engine as fengine
from vectora.zmod import footprint, pump

PUMP_MIN_SCORE = 50.0
PUMP_TOP_N = 15


def run_zscan(con, date_str: str | None = None, features_path=None) -> dict:
    feats = fengine.compute(con, out_path=features_path) if features_path \
        else fengine.compute(con)
    run_date = date_str or str(feats["date"].max())
    day = feats.filter(pl.col("date") == date.fromisoformat(run_date))
    categories = dict(con.execute(
        "SELECT symbol, category FROM symbols").fetchall())

    scored = pump.phase_and_score(day, categories)
    flagged = (scored.filter(pl.col("score") >= PUMP_MIN_SCORE)
               .sort("score", descending=True).head(PUMP_TOP_N))
    pump_rows = [{"date": run_date, "symbol": r["symbol"], "kind": "pump",
                  "score": round(float(r["score"]), 1), "phase": r["phase"],
                  "detail": json.dumps({
                      "ret_21d": round(float(r["ret_21d"] or 0), 4),
                      "vol_ratio": round(float(r["vol_ratio_5_21"] or 0), 2)})}
                 for r in flagged.iter_rows(named=True)]
    if pump_rows:
        vdb.upsert(con, "zwatch", pump_rows)

    fp_result = footprint.compute_event_footprints(
        con, feats.select(["symbol", "date", "ret", "volume_z_21d"]))
    fp_rows = footprint.daily_watch(
        con, feats.select(["symbol", "date", "ret", "volume_z_21d"]), run_date)
    if fp_rows:
        vdb.upsert(con, "zwatch", fp_rows)

    return {"date": run_date, "pump_flags": len(pump_rows),
            "footprints_computed": fp_result["computed"],
            "footprint_flags": len(fp_rows)}
