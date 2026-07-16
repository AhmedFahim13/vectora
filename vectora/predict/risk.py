"""Risk block per prediction (spec §14). Empirical where possible:
expected move sizes come from analog realized outcomes; liquidity risk
assumes a POSITION_TAKA position unwound at <=20% of median daily value
(more than that and you ARE the market on a thin DSE book)."""
from vectora.settings import POSITION_TAKA

ABSORBABLE_SHARE = 0.20


def build(vol_21d: float | None, value_mn_med_21d: float | None,
          category: str | None, analog_stats: dict) -> dict:
    expected_up = analog_stats["median_up"]
    expected_down = analog_stats["median_down"]
    rr = None
    if expected_down and expected_down < 0:
        rr = expected_up / abs(expected_down)
    exit_days = None
    if value_mn_med_21d and value_mn_med_21d > 0:
        absorbable_taka_per_day = ABSORBABLE_SHARE * (value_mn_med_21d * 1e6)
        exit_days = POSITION_TAKA / absorbable_taka_per_day
    return {
        "vol_21d": vol_21d,
        "expected_up": expected_up,
        "expected_down": expected_down,
        "rr_ratio": rr,
        "exit_days": exit_days,
        "analog_max_drawdown": analog_stats["max_drawdown"],
        "analog_hit_rate": analog_stats["hit_rate"],
        "analog_n": analog_stats["n"],
        "category": category,
        "liquidity_value_mn": value_mn_med_21d,
    }
