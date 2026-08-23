"""Daily health watchdog (spec §20 monitoring).

Five checks; any failure turns the health workflow red (GitHub emails the
owner natively) and sends a [HEALTH] email when the secret is present.
The canary check needs a live session and is skipped when session=None
(unit tests, offline runs).
"""
import datetime as dt

from vectora import calendar as cal
from vectora.settings import DSE_BASE, MIN_QUALITY_SCORE

MODEL_STALE_DAYS = 120   # weekly retrain; a quarter with no change is wrong


def check(con, today: dt.date | None = None,
          holidays: set | None = None, session=None,
          state_root=None) -> dict:
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

    # predictions must keep pace with prices: the predict stage failing is
    # invisible to a freshness check on prices alone (real outage 2026-07-19)
    pred_max = con.execute("SELECT max(date) FROM predictions").fetchone()[0]
    checks.append({
        "name": "predictions",
        "ok": pred_max is not None and str(pred_max) >= str(expected),
        "detail": f"latest prediction {pred_max}, expected {expected}"})

    # the active model must be trained on recent data. Weekly retraining can
    # run for a month and change nothing — a promotion guard that refuses
    # every challenger, or a lost registry row, both leave a stale model
    # serving predictions while every other check stays green. On 2026-08-23
    # the live model's training data ended 2024-11-21, 21 months earlier.
    # every target, not just the headline one: pinning this to g5_h10 hid
    # that g10_h30 was still serving a model trained through 2024-11-21
    rows = con.execute(
        """
        SELECT target, model_id, train_end FROM model_registry
        WHERE family = 'lgbm' AND active ORDER BY target
        """).fetchall()
    if not rows:
        checks.append({"name": "model_freshness", "ok": False,
                       "detail": "no active model for any target"})
    else:
        stale = []
        for target, _mid, train_end in rows:
            age = (today - train_end).days if train_end else None
            if age is None or age > MODEL_STALE_DAYS:
                stale.append(f"{target} trained through {train_end} "
                             f"({age} days ago)")
        checks.append({
            "name": "model_freshness",
            "ok": not stale,
            "detail": "; ".join(stale) if stale
            else f"{len(rows)} active model(s) within "
                 f"{MODEL_STALE_DAYS} days"})

    # the mirror in data/state must agree with the database. A discarded
    # binary merge shows up here and nowhere else — every other check stays
    # green while an older model quietly serves predictions.
    try:
        from vectora import state as vstate
        problems = vstate.divergence(con, state_root)
    except Exception as exc:  # noqa: BLE001 - a broken mirror IS the finding
        problems = [f"state mirror unreadable: {exc}"]
    checks.append({
        "name": "state_mirror",
        "ok": not problems,
        "detail": "; ".join(problems) if problems else "database matches mirror"})

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
