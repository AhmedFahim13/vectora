# vectora/alerts/intraday.py
"""Intraday anomaly detection + urgent email tier (spec §16).

volume_surge: cumulative intraday volume already >= SURGE_X times the
symbol's trailing 21-day MEDIAN FULL-DAY volume (mid-session!), with a
turnover floor to ignore illiquid dust. near_circuit: |LTP/YCP - 1| within
a hair of the ~10% band. Cooldown: one intraday alert per symbol per
COOLDOWN_DAYS via alerts_log. Email cap: at most MAX_EMAILS_PER_DAY urgent
sends; overflow anomalies still get logged and appear in the evening digest.
"""
from datetime import date, timedelta

from vectora import db as vdb
from vectora.alerts.digest import send_or_save

SURGE_X = 3.0
MIN_VALUE_MN = 0.5
CIRCUIT_NEAR = 0.085
COOLDOWN_DAYS = 2
MAX_EMAILS_PER_DAY = 3


def detect(con, snapshots: list[dict], ts: str) -> list[dict]:
    day = ts[:10]
    baseline = dict(con.execute(
        """
        WITH recent AS (
            SELECT symbol, volume,
                   row_number() OVER (PARTITION BY symbol ORDER BY date DESC)
                   AS rn
            FROM prices WHERE date < ? AND volume IS NOT NULL
        )
        SELECT symbol, median(volume) FROM recent WHERE rn <= 21 GROUP BY symbol
        """, [day]).fetchall())
    out = []
    for s in snapshots:
        med = baseline.get(s["symbol"])
        vol, val = s.get("volume"), s.get("value_mn")
        ltp, ycp = s.get("ltp"), s.get("ycp")
        if med and vol and val and val >= MIN_VALUE_MN \
                and vol >= SURGE_X * med:
            out.append({"symbol": s["symbol"], "kind": "volume_surge",
                        "ratio": round(vol / med, 2),
                        "detail": f"{vol:,} vs 21d median {int(med):,}"})
        if ltp and ycp and ycp > 0 and abs(ltp / ycp - 1) >= CIRCUIT_NEAR:
            move = ltp / ycp - 1
            out.append({"symbol": s["symbol"], "kind": "near_circuit",
                        "ratio": round(abs(move), 4),
                        "detail": f"{move:+.1%} vs YCP"})
    return out


def filter_and_log(con, anomalies: list[dict], day: str) -> list[dict]:
    floor = (date.fromisoformat(day)
             - timedelta(days=COOLDOWN_DAYS)).isoformat()
    recent = {r[0] for r in con.execute(
        "SELECT symbol FROM alerts_log WHERE alert_type = 'intraday' "
        "AND alert_date >= ? AND alert_date <= ?", [floor, day]).fetchall()}
    fresh = [a for a in anomalies if a["symbol"] not in recent]
    seen = set()
    for a in fresh:
        if a["symbol"] in seen:
            continue
        seen.add(a["symbol"])
        vdb.upsert(con, "alerts_log", [{
            "id": f"{day}_intraday_{a['symbol']}", "alert_type": "intraday",
            "symbol": a["symbol"], "alert_date": day, "prediction_id": None}])
    return fresh


def email_allowed(con, day: str) -> bool:
    n = con.execute(
        "SELECT count(*) FROM alerts_log WHERE alert_type = 'intraday_email' "
        "AND alert_date = ?", [day]).fetchone()[0]
    return n < MAX_EMAILS_PER_DAY


def render(anomalies: list[dict], ts: str) -> str:
    lines = [f"# Vectora intraday alert {ts}", ""]
    for a in anomalies:
        label = "volume surge" if a["kind"] == "volume_surge" \
            else "near circuit"
        lines.append(f"- {a['symbol']}: {label} - {a['detail']}")
    lines += ["", "Warnings about unusual PUBLIC trading activity; "
              "not signals. _Research tool, not investment advice._", ""]
    return "\n".join(lines)


def run_intraday(con, session, ts: str | None = None) -> dict:
    from datetime import datetime

    from vectora import calendar as cal
    from vectora.collect.dse_intraday import collect_intraday, fetch_latest, parse_latest

    stamp = ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    day = stamp[:10]
    if not cal.is_trading_day(date.fromisoformat(day), cal.load_holidays()):
        return {"ts": stamp, "skipped": "not a trading day"}
    html = fetch_latest(session)
    rows = parse_latest(html)
    n = collect_intraday(con, _Replay(html), ts=stamp)
    anomalies = filter_and_log(con, detect(con, rows, stamp), day)
    emailed = False
    if anomalies and email_allowed(con, day):
        subject = f"[URGENT] Vectora intraday {stamp[:16]} - " \
                  f"{len(anomalies)} anomal" \
                  f"{'y' if len(anomalies) == 1 else 'ies'}"
        send_or_save(subject, render(anomalies, stamp))
        vdb.upsert(con, "alerts_log", [{
            "id": f"{day}_intraday_email_{stamp[11:16]}",
            "alert_type": "intraday_email", "symbol": None,
            "alert_date": day, "prediction_id": None}])
        emailed = True
    return {"ts": stamp, "snapshots": n, "anomalies": len(anomalies),
            "emailed": emailed}


class _Replay:
    """Session stand-in so collect_intraday reuses the already-fetched page
    instead of hitting DSE twice per scan."""

    def __init__(self, html: str):
        self._html = html

    def get(self, url, params=None):
        return self._html
