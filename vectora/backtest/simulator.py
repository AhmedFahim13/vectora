"""Event-driven portfolio simulation over the real price path.

Three choices here decide whether the output means anything:

ENTRY TIMING. Predictions are made from end-of-day data, so the earliest
price you could actually transact at is the NEXT session's open. Entering at
the close you just predicted from is a quiet lookahead that flatters results,
so `next_open` is the default.

INTRABAR ORDER. When a day's high clears the target and its low breaks the
stop, daily bars cannot say which came first. Assuming the target won is the
single most common way backtests lie. The default assumes the STOP hit
first — pessimistic, and the only assumption that cannot overstate results.

CAPITAL IS FINITE. Signals are not free to take: with a position cap, taking
one means declining another. The loop respects slots and ranks same-day
candidates by score, so the simulation answers "what could I have run",
not "what if I had unlimited money".
"""
from dataclasses import dataclass, field

import polars as pl

from vectora.backtest.costs import CostModel


@dataclass(frozen=True)
class Rules:
    target_pct: float = 0.05
    stop_pct: float | None = None        # None = no stop, ride to the horizon
    max_days: int = 10
    entry: str = "next_open"             # or "close"
    pessimistic_intrabar: bool = True


@dataclass
class Portfolio:
    capital: float = 1_000_000.0         # BDT
    max_positions: int = 10
    size_pct: float = 0.10               # fraction of equity per position
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


def _paths(panel: pl.DataFrame) -> dict:
    """symbol -> (dates, open, high, low, close, adv_mn) as plain lists."""
    out = {}
    for (sym,), g in panel.sort(["symbol", "date"]).group_by(
            ["symbol"], maintain_order=True):
        out[sym] = {
            "date": g["date"].to_list(), "open": g["open"].to_list(),
            "high": g["high"].to_list(), "low": g["low"].to_list(),
            "close": g["close"].to_list(),
            "adv": g["adv_mn"].to_list() if "adv_mn" in g.columns
            else [None] * g.height,
            "idx": {d: i for i, d in enumerate(g["date"].to_list())},
        }
    return out


def _exit(path: dict, i0: int, entry_px: float, rules: Rules) -> tuple:
    """(exit_index, exit_price, reason) walking the real bars forward."""
    target = entry_px * (1 + rules.target_pct)
    stop = entry_px * (1 + rules.stop_pct) if rules.stop_pct else None
    last = min(i0 + rules.max_days, len(path["date"]) - 1)
    for i in range(i0, last + 1):
        hi, lo = path["high"][i], path["low"][i]
        hit_t = hi is not None and hi >= target
        hit_s = stop is not None and lo is not None and lo <= stop
        if hit_t and hit_s:
            # both touched in one bar; daily data cannot order them
            return ((i, stop, "stop") if rules.pessimistic_intrabar
                    else (i, target, "target"))
        if hit_s:
            return (i, stop, "stop")
        if hit_t:
            return (i, target, "target")
    return (last, path["close"][last], "time")


def run(panel: pl.DataFrame, entries: pl.DataFrame, rules: Rules,
        costs: CostModel, portfolio: Portfolio | None = None) -> dict:
    """Simulate taking `entries` under `rules`, paying `costs`."""
    pf = portfolio or Portfolio()
    paths = _paths(panel)
    ent = entries.sort(["date", "score"], descending=[False, True])

    open_pos: list = []          # dicts with exit_date
    equity = pf.capital
    by_date: dict = {}
    for r in ent.iter_rows(named=True):
        by_date.setdefault(r["date"], []).append(r)

    for day in sorted({d for d in panel["date"].to_list()}):
        # close anything due today
        still = []
        for pos in open_pos:
            if pos["exit_date"] <= day:
                equity += pos["pnl"]
                pf.trades.append(pos)
            else:
                still.append(pos)
        open_pos = still

        for cand in by_date.get(day, []):
            if len(open_pos) >= pf.max_positions:
                break
            p = paths.get(cand["symbol"])
            if p is None or day not in p["idx"]:
                continue
            i = p["idx"][day]
            i0 = i + 1 if rules.entry == "next_open" else i
            if i0 >= len(p["date"]):
                continue
            entry_px = (p["open"][i0] if rules.entry == "next_open"
                        else p["close"][i])
            if not entry_px or entry_px <= 0:
                continue
            j, exit_px, reason = _exit(p, i0, entry_px, rules)
            notional = equity * pf.size_pct
            adv = p["adv"][i]
            side = costs.side_cost(notional / 1_000_000, adv)
            gross = exit_px / entry_px - 1
            net = gross - 2 * side
            open_pos.append({
                "symbol": cand["symbol"], "entry_date": p["date"][i0],
                "exit_date": p["date"][j], "entry_px": entry_px,
                "exit_px": exit_px, "gross_ret": gross, "cost": 2 * side,
                "net_ret": net, "reason": reason, "pnl": notional * net,
                "notional": notional,
            })
        pf.equity_curve.append(
            {"date": day,
             "equity": equity + sum(p["pnl"] for p in open_pos)})

    for pos in open_pos:                   # mark out anything still open
        equity += pos["pnl"]
        pf.trades.append(pos)
    return {"trades": pf.trades, "equity_curve": pf.equity_curve,
            "final_equity": equity, "start_capital": pf.capital}
