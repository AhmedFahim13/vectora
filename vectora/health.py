"""Daily health watchdog (spec §20 monitoring).

Five checks; any failure turns the health workflow red (GitHub emails the
owner natively) and sends a [HEALTH] email when the secret is present.
The canary check needs a live session and is skipped when session=None
(unit tests, offline runs).
"""
import datetime as dt

from vectora import calendar as cal
from vectora.settings import DSE_BASE, MIN_QUALITY_SCORE


def check(con, today: dt.date | None = None,
          holidays: set | None = None, session=None) -> dict:
    today = today or dt.date.today()
    hs = cal.load_holidays() if holidays is None else holidays
    expected = today if cal.is_trading_day(today, hs) \
        else cal.previous_trading_day(today, hs)
    checks = []

    max_date = con.execute(
        "SELECT max(date) FROM prices_raw WHERE source = 'dse_eod'"
    ).fetchone()[0]
    checks.append({
        "name": "freshness",
        "ok": max_date is not None and str(max_date) >= str(expected),
        "detail": f"latest eod {max_date}, expected {expected}"})

    q = con.execute(
        "SELECT score FROM data_quality WHERE source='dse_eod' "
        "ORDER BY date DESC LIMIT 1").fetchone()
    checks.append({
        "name": "quality",
        "ok": q is not None and q[0] >= MIN_QUALITY_SCORE,
        "detail": f"latest quality {q[0] if q else 'none'}"})

    for t in ("prices_raw", "predictions", "regimes"):
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        if n == 0:
            checks.append({"name": "tables", "ok": False,
                           "detail": f"{t} is empty"})
            break
    else:
        checks.append({"name": "tables", "ok": True, "detail": "core ok"})

    from vectora import db as vdb
    wm = vdb.get_watermark(con, "collect", "eod")
    checks.append({
        "name": "watermark",
        "ok": wm is not None and wm >= str(expected),
        "detail": f"collect/eod at {wm}, expected {expected}"})

    if session is not None:
        try:
            archive = session.get(
                f"{DSE_BASE}/day_end_archive.php",
                params={"startDate": str(expected), "endDate": str(expected),
                        "inst": "All Instrument", "archive": "data"})
            home = session.get(f"{DSE_BASE}/")
            ok = "shares-table" in archive and "midrow" in home
            detail = "markers present" if ok else "LAYOUT CHANGED"
        except Exception as exc:  # noqa: BLE001 - any fetch failure is the finding
            ok, detail = False, f"fetch failed: {exc}"
        checks.append({"name": "canary", "ok": ok, "detail": detail})

    return {"ok": all(c["ok"] for c in checks), "checks": checks}
