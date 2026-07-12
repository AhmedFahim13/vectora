"""Data-quality gates for a collected day (spec §5.3).

validate_day scores one EOD date 0-100 and records the result in
data_quality. Alerts and model runs downstream treat scores below
settings.MIN_QUALITY_SCORE as untrustworthy; a score of 0 means the day
is unusable (no rows at all).

Deductions:
- no rows at all ................................. score 0
- row count < 70% of previous trading day ........ -30
- per bad row (nonpositive close, high<low,
  negative volume, |close/ycp-1| > band) ......... -2 each, capped at -60
"""
import json
from datetime import date

from vectora import db as vdb

ROW_DROP_THRESHOLD = 0.70
# DSE circuit breaker is ~10%; the buffer absorbs corporate-action gaps
# until adjustment logic lands in Phase 2
MOVE_BAND = 0.12
PER_ROW_DEDUCTION = 2
PER_ROW_CAP = 60


def validate_day(con, d: date, prev: date) -> dict:
    ds, ps = d.isoformat(), prev.isoformat()
    issues: list[str] = []

    n_today = con.execute(
        "SELECT count(*) FROM prices_raw WHERE date = ? AND source = 'dse_eod'", [ds]
    ).fetchone()[0]
    if n_today == 0:
        issues.append(f"no rows for {ds}")
        _record(con, ds, 0, issues)
        return {"score": 0, "issues": issues}

    n_prev = con.execute(
        "SELECT count(*) FROM prices_raw WHERE date = ? AND source = 'dse_eod'", [ps]
    ).fetchone()[0]

    deductions = 0
    if n_prev > 0 and n_today < n_prev * ROW_DROP_THRESHOLD:
        deductions += 30
        issues.append(f"row count dropped {n_prev} -> {n_today}")

    bad_rows = con.execute(
        """
        SELECT
            count(*) FILTER (WHERE close IS NOT NULL AND close <= 0) AS bad_close,
            count(*) FILTER (WHERE high IS NOT NULL AND low IS NOT NULL
                             AND high < low) AS bad_hl,
            count(*) FILTER (WHERE volume IS NOT NULL AND volume < 0) AS bad_vol,
            count(*) FILTER (WHERE close IS NOT NULL AND ycp IS NOT NULL AND ycp > 0
                             AND abs(close / ycp - 1) > ?) AS bad_move
        FROM prices_raw WHERE date = ? AND source = 'dse_eod'
        """,
        [MOVE_BAND, ds],
    ).fetchone()
    labels = ("nonpositive close", "high<low", "negative volume",
              f"move outside +/-{MOVE_BAND:.0%} band")
    n_bad = 0
    for count, label in zip(bad_rows, labels, strict=True):
        if count:
            n_bad += count
            issues.append(f"{count} rows: {label}")
    deductions += min(n_bad * PER_ROW_DEDUCTION, PER_ROW_CAP)

    score = max(0, 100 - deductions)
    _record(con, ds, score, issues)
    return {"score": score, "issues": issues}


def _record(con, ds: str, score: int, issues: list[str]) -> None:
    vdb.upsert(con, "data_quality", [{
        "date": ds, "source": "dse_eod", "score": score, "issues": json.dumps(issues),
    }])
