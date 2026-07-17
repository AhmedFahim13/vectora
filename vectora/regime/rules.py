# vectora/regime/rules.py
"""Ordered threshold rules mapping market state to the spec §11 taxonomy.
First match wins; thresholds are documented judgment calls, revisited in
Phase 5 once per-regime evaluation data accumulates. Warmup rows (no
200-day average yet) stay unclassified rather than guessed."""
from vectora import db as vdb
from vectora.regime import state as market_state_mod

# (regime, confidence) rules in priority order
PANIC_VOL, PANIC_RET = 0.90, -0.08
LOWLIQ_Z = -1.5
RECOVERY_RET = 0.05
HEAT_Z, HEAT_VOL = 2.0, 0.70
BULL_BREADTH, BEAR_BREADTH = 0.60, 0.35


def classify_row(r: dict) -> tuple[str, float] | None:
    if r["ma200"] is None or r["ret_21d"] is None or r["vol_pctile"] is None:
        return None
    if r["vol_pctile"] > PANIC_VOL and r["ret_21d"] < PANIC_RET:
        return "Panic", 0.8
    if r["activity_z"] is not None and r["activity_z"] < LOWLIQ_Z:
        return "LowLiquidity", 0.7
    if r["mkt_level"] < r["ma200"] and r["ret_21d"] > RECOVERY_RET:
        return "Recovery", 0.7
    if (r["activity_z"] is not None and r["activity_z"] > HEAT_Z
            and r["vol_pctile"] > HEAT_VOL):
        return "SpeculativeHeat", 0.7
    if r["mkt_level"] > r["ma200"] and r["breadth"] > BULL_BREADTH:
        return "Bull", 0.8
    if r["mkt_level"] < r["ma200"] and r["breadth"] < BEAR_BREADTH:
        return "Bear", 0.8
    return "Sideways", 0.5


def classify_history(con) -> dict:
    frame = market_state_mod.market_state(con)
    rows, skipped = [], 0
    for r in frame.iter_rows(named=True):
        result = classify_row(r)
        if result is None:
            skipped += 1
            continue
        regime, conf = result
        rows.append({"date": str(r["date"]), "regime": regime,
                     "confidence": conf, "method": "rules"})
    if rows:
        vdb.upsert(con, "regimes", rows)
    return {"classified": len(rows), "skipped": skipped}


def regime_on(con, date_str: str) -> str | None:
    row = con.execute(
        "SELECT regime FROM regimes WHERE date = ?", [date_str]).fetchone()
    return row[0] if row else None
