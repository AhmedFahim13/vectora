"""Run the backtest over live predictions and write the report.

The point of this module is to make profitability a first-class metric that
runs on a schedule, beside Brier. A better-calibrated model that loses money
is not a better model, and until this ran there was no number in the system
that could tell the difference.
"""
import datetime as dt
from pathlib import Path

import polars as pl

from vectora.backtest import metrics, simulator
from vectora.backtest.costs import CostModel
from vectora.backtest.simulator import Portfolio, Rules
from vectora.features import base
from vectora.settings import REPORTS_DIR

# strategies worth separating: the current rule, the same rule with the
# downside capped, and a version that lets winners run
STRATEGIES = {
    "current (+5% target, no stop, 10d)":
        Rules(target_pct=0.05, stop_pct=None, max_days=10),
    "+5% target with -3% stop":
        Rules(target_pct=0.05, stop_pct=-0.03, max_days=10),
    "+10% target with -5% stop":
        Rules(target_pct=0.10, stop_pct=-0.05, max_days=10),
    "+15% target with -5% stop, 30d":
        Rules(target_pct=0.15, stop_pct=-0.05, max_days=30),
}


def load_panel(con) -> pl.DataFrame:
    """Price path plus 21-day median turnover, for the impact model.

    Only 18,512 of 1,081,964 rows carry the scraped `value_mn` column, but
    every row has volume — and where both exist, volume x close reproduces
    the reported turnover to within 0.5% median error. So liquidity is
    recoverable across the whole 13 years rather than only the scraped era,
    which is what lets the impact model apply to deep history instead of
    charging every old trade the unknown-liquidity cap.
    """
    p = base.load_panel(con).select(
        ["symbol", "date", "open", "high", "low", "close", "value_mn",
         "volume"])
    turnover = pl.coalesce(
        pl.col("value_mn"), pl.col("volume") * pl.col("close") / 1_000_000)
    return p.with_columns(turnover.alias("_turnover")).with_columns(
        pl.col("_turnover").rolling_median(21).over("symbol").alias("adv_mn"))


def load_entries(con, min_prob: float = 0.0,
                 signals_only: bool = False) -> pl.DataFrame:
    q = ("SELECT symbol, CAST(date AS VARCHAR) AS d, probability AS score "
         "FROM predictions WHERE target = 'g5_h10' AND probability >= ?")
    if signals_only:
        q += " AND is_signal"
    rows = con.execute(q, [min_prob]).pl()
    if rows.height == 0:
        return rows
    return rows.with_columns(pl.col("d").str.to_date().alias("date")).drop("d")


def run(con, costs: CostModel | None = None,
        reports_dir: Path = REPORTS_DIR, min_prob: float = 0.40) -> dict:
    costs = costs or CostModel()
    panel = load_panel(con)
    entries = load_entries(con, min_prob=min_prob)
    out: dict = {}
    if entries.height == 0:
        return {"entries": 0}

    lines = [f"# Backtest {dt.date.today().isoformat()}", "",
             f"{entries.height} candidate entries at probability >= "
             f"{min_prob:.2f}, priced on the real path.", "",
             "Entry is the NEXT session's open, never the close the "
             "prediction was made from. When a single day touches both the "
             "target and the stop, the stop is assumed to have hit first: "
             "daily bars cannot order them, and the opposite assumption is "
             "the most common way a backtest overstates itself.", "",
             f"Costs: {costs.commission * 100:.2f}% commission and "
             f"{costs.regulatory * 100:.2f}% regulatory per side, "
             f"{costs.half_spread * 100:.2f}% half-spread, plus impact "
             f"scaled by share of daily turnover. **These are estimates — "
             f"replace them with the rates on a real contract note.**", ""]

    for name, rules in STRATEGIES.items():
        res = simulator.run(panel, entries, rules, costs,
                            Portfolio(max_positions=10, size_pct=0.10))
        m = metrics.summarize(res)
        out[name] = m
        lines.append(metrics.render(name, m))
        lines.append("")

    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"backtest_{dt.date.today().isoformat()}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    out["_report"] = str(path)
    out["entries"] = entries.height
    return out
