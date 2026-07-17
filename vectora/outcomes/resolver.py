"""Grade matured predictions against realized prices (spec §17 step 1).

A prediction for target gX_hH matures once H trading rows exist after its
date; realized max/min forward returns come from the same label machinery
the models train on, so grading and training share one definition of
"outcome". Unresolved predictions stay pending and are retried next run.
"""
import re

import polars as pl

from vectora import db as vdb
from vectora import labels as lab
from vectora.features import base

_TARGET_RE = re.compile(r"^g(\d+)_h(\d+)$")


def resolve(con) -> dict:
    preds = con.execute(
        """
        SELECT p.id, p.symbol, p.date, p.target
        FROM predictions p
        LEFT JOIN outcomes o ON o.prediction_id = p.id
        WHERE o.prediction_id IS NULL
        """).pl()
    if preds.height == 0:
        return {"resolved": 0, "pending": 0}

    panel = base.load_panel(con).select(["symbol", "date", "close"])
    rows = []
    for target in preds["target"].unique().to_list():
        m = _TARGET_RE.match(target)
        if not m:
            continue  # unknown target format: leave pending forever
        x, h = int(m.group(1)) / 100, int(m.group(2))
        labeled = lab.make_labels(
            panel, thresholds=(x,), horizons=(h,), continuous=True)
        joined = (
            preds.filter(pl.col("target") == target)
            .join(labeled.select(["symbol", "date", f"fwdmax_h{h}",
                                  f"fwdmin_h{h}"]),
                  on=["symbol", "date"], how="left")
            .filter(pl.col(f"fwdmax_h{h}").is_not_null())
        )
        for r in joined.iter_rows(named=True):
            rows.append({
                "prediction_id": r["id"],
                "realized_max": r[f"fwdmax_h{h}"],
                "realized_min": r[f"fwdmin_h{h}"],
                "hit": bool(r[f"fwdmax_h{h}"] >= x),
            })
    if rows:
        vdb.upsert(con, "outcomes", rows)
    return {"resolved": len(rows), "pending": preds.height - len(rows)}
