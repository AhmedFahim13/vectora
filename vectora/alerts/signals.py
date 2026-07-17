"""Signal alert log with per-symbol cooldown (spec §16 anti-fatigue).

One batch of predictions lands per day post-close, so the digest is the
delivery channel; this module decides which signals count as NEW (not
alerted for the same symbol within the cooldown window) and records them.
Urgency tiers arrive with Phase 4's intraday scans.
"""
from datetime import date, timedelta

from vectora import db as vdb

COOLDOWN_DAYS = 2


def log_signal_alerts(con, date_str: str) -> list[str]:
    """Record alerts for today's signals outside cooldown; returns NEW symbols."""
    d = date.fromisoformat(date_str)
    floor = (d - timedelta(days=COOLDOWN_DAYS)).isoformat()
    rows = con.execute(
        """
        SELECT p.id, p.symbol FROM predictions p
        WHERE p.date = ? AND p.is_signal
          AND p.symbol NOT IN (
              SELECT symbol FROM alerts_log
              WHERE alert_type = 'signal' AND alert_date >= ? AND alert_date <= ?
          )
        ORDER BY p.symbol
        """, [date_str, floor, date_str]).fetchall()
    new = []
    for pred_id, symbol in rows:
        vdb.upsert(con, "alerts_log", [{
            "id": f"{date_str}_signal_{symbol}",
            "alert_type": "signal", "symbol": symbol,
            "alert_date": date_str, "prediction_id": pred_id,
        }])
        new.append(symbol)
    return new
