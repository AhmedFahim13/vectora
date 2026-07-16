"""Per-prediction explanation (spec §15): TreeSHAP drivers via LightGBM's
pred_contrib (no extra dependency), analog evidence, and templated
uncertainty warnings. Deterministic text, fully audit-traceable to numbers."""
import numpy as np

EXIT_DAYS_WARN = 3.0


def drivers(booster, x: np.ndarray, feature_names: list[str],
            top: int = 6) -> list[dict]:
    contrib = booster.predict(x.reshape(1, -1), pred_contrib=True)[0]
    # last element is the bias term; drop it
    contrib = contrib[:-1]
    order = np.argsort(-np.abs(contrib))[:top]
    return [
        {"feature": feature_names[i],
         "contribution": round(float(contrib[i]), 4),
         "value": None if np.isnan(x[i]) else round(float(x[i]), 4)}
        for i in order
    ]


def render(symbol: str, target: str, probability: float,
           driver_list: list[dict], analog_stats: dict, risk_block: dict,
           quality: int) -> str:
    lines = [
        f"{symbol}: {probability:.0%} calibrated probability of the "
        f"{target} move.",
    ]
    for d in driver_list:
        direction = "supports" if d["contribution"] > 0 else "works against"
        lines.append(
            f"- {d['feature']} = {d['value']} {direction} the setup "
            f"(contribution {d['contribution']:+.3f})")
    hits = round(analog_stats["hit_rate"] * analog_stats["n"])
    lines.append(
        f"Similar past setups: {hits} of {analog_stats['n']} hit the target; "
        f"median outcome +{analog_stats['median_up']:.1%} / "
        f"{analog_stats['median_down']:.1%}; "
        f"worst analog drawdown {analog_stats['max_drawdown']:.1%}.")
    warnings = []
    if risk_block["exit_days"] is None or risk_block["exit_days"] > EXIT_DAYS_WARN:
        warnings.append("thin book - exiting may take days and move the price")
    if risk_block["category"] == "Z":
        warnings.append("Z-category name - governance and settlement risk")
    if quality < 100:
        warnings.append(f"data quality {quality} on the underlying day")
    if warnings:
        lines.append("Warnings: " + "; ".join(warnings) + ".")
    lines.append(
        f"Downside scenario: median analog loss {analog_stats['median_down']:.1%}; "
        "this prediction can fail if market regime shifts or an "
        "unannounced corporate event lands inside the horizon.")
    return "\n".join(lines)
