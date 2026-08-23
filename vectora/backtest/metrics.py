"""Portfolio metrics. Expectancy is the one that decides everything.

Win rate is the number people ask for and the least useful of these: a
strategy winning 51% of the time loses money when gains cap at +5% and
losses run, which is exactly what today's measurement showed. Expectancy —
average net return per trade — is the number that says whether to trade at
all, and profit factor says how it is earned.
"""
import math

TRADING_DAYS = 250


def summarize(result: dict, rf: float = 0.0) -> dict:
    trades = result["trades"]
    start = result["start_capital"]
    final = result["final_equity"]
    n = len(trades)
    if n == 0:
        return {"trades": 0, "expectancy": None, "total_return": 0.0,
                "win_rate": None, "profit_factor": None, "max_drawdown": 0.0,
                "sharpe": None, "final_equity": final}
    nets = [t["net_ret"] for t in trades]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)

    curve = result["equity_curve"]
    peak, mdd = -math.inf, 0.0
    for point in curve:
        peak = max(peak, point["equity"])
        if peak > 0:
            mdd = max(mdd, (peak - point["equity"]) / peak)

    rets = []
    for a, b in zip(curve, curve[1:], strict=False):
        if a["equity"] > 0:
            rets.append(b["equity"] / a["equity"] - 1)
    sharpe = None
    if len(rets) > 2:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sd = var ** 0.5
        if sd > 0:
            sharpe = (mean - rf / TRADING_DAYS) / sd * math.sqrt(TRADING_DAYS)

    reasons: dict = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1

    return {
        "trades": n,
        "expectancy": sum(nets) / n,            # the headline number
        "win_rate": len(wins) / n,
        "avg_win": (gross_win / len(wins)) if wins else None,
        "avg_loss": (-gross_loss / len(losses)) if losses else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
        "total_return": final / start - 1 if start else None,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "final_equity": final,
        "exit_reasons": reasons,
        "avg_cost": sum(t["cost"] for t in trades) / n,
    }


def render(name: str, m: dict) -> str:
    """Markdown table for the weekly report and the dashboard."""
    if not m["trades"]:
        return f"**{name}** - no trades taken.\n"

    def pct(x, dp=2):
        return "n/a" if x is None else f"{x * 100:+.{dp}f}%"

    def num(x, dp=2):
        return "n/a" if x is None else f"{x:.{dp}f}"

    rows = [
        ("expectancy per trade", f"**{pct(m['expectancy'])}**"),
        ("win rate", f"{m['win_rate']:.1%}"),
        ("average win", pct(m["avg_win"])),
        ("average loss", pct(m["avg_loss"])),
        ("profit factor", num(m["profit_factor"])),
        ("total return", pct(m["total_return"])),
        ("max drawdown", f"{m['max_drawdown']:.1%}"),
        ("Sharpe", num(m["sharpe"])),
        ("average round-trip cost", f"{m['avg_cost'] * 100:.2f}%"),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (f"**{name}** - {m['trades']} trades\n\n"
            "| metric | value |\n|---|---|\n" + body + "\n")


