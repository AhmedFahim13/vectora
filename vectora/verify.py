"""Per-stock verification: what we said, and what the stock then did.

The aggregate table ("Strong Buy hits 40.2% against a 32.1% base rate") is
the right way to measure a system and the wrong way to convince a person of
it. The client put it plainly: a tilt across 133,592 cases tells her nothing
she can check. She wants to look at one stock, see what each indicator read
on a given day, and then see what the price actually did afterwards — so she
can go to the market and confirm it herself.

That is a fair demand and this module answers it. For any symbol it replays
the indicators over history, records the posture on each of the last N
trading days, and joins the return that actually followed. Rows too recent
for the horizon to have finished say "pending" rather than borrowing a
number that does not exist yet.

Nothing here is a trade instruction. There is no entry, no stop and no size
— the output is a direction and how far price went, which is what the tool
is for.
"""
import polars as pl

from vectora.features import base
from vectora.ta import gauges, indicators

# what the client reads, in her order
INDICATOR_COLS = [
    ("rsi14", "RSI(14)", "{:.0f}"),
    ("macd_hist", "MACD hist", "{:+.2f}"),
    ("cci20", "CCI(20)", "{:.0f}"),
    ("adx14", "ADX(14)", "{:.0f}"),
    ("stochrsi", "StochRSI", "{:.0f}"),
    ("uo", "Ultimate", "{:.0f}"),
    ("sma20", "SMA(20)", "{:,.2f}"),
    ("sma50", "SMA(50)", "{:,.2f}"),
    ("sma200", "SMA(200)", "{:,.2f}"),
]
HORIZONS = (5, 10)


def load_panel(con) -> pl.DataFrame:
    """Load the price panel once, for callers looping over many symbols.

    base.load_panel reads every symbol and 1.05M rows. Calling history()
    per symbol without this would re-read all of it 62 times.
    """
    cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    p = base.load_panel(con)
    if "value_mn" in p.columns:
        cols.append("value_mn")
    return p.select(cols).sort(["symbol", "date"])


def history(con, symbol: str, days: int = 15,
            horizons: tuple = HORIZONS, panel: pl.DataFrame | None = None
            ) -> list[dict]:
    """One row per recent trading day: posture, readings, what followed.

    Recomputed from prices rather than read from ta_gauges, because those
    tables are pruned to the last couple of dates — they are a pure function
    of price and cheap to rebuild for a single symbol.
    """
    source = panel if panel is not None else load_panel(con)
    panel = source.filter(pl.col("symbol") == symbol).sort("date")
    if panel.height < 60:
        return []

    ind = indicators.add_tradingview_set(indicators.add_all(panel))
    voted = gauges.vote_frame(ind)

    # realised forward moves, from the close on the row's own day
    close = pl.col("close")
    for h in horizons:
        fwd_close = close.shift(-h).over("symbol")
        fwd_max = close.shift(-1).rolling_max(h, min_periods=1).over("symbol")
        voted = voted.with_columns(
            (fwd_close / close - 1).alias(f"ret_{h}d"),
            (fwd_max.shift(-(h - 1)).over("symbol") / close - 1)
            .alias(f"peak_{h}d"))

    # median turnover, so "was it tradable" is answerable alongside
    turnover = pl.coalesce(
        pl.col("value_mn") if "value_mn" in voted.columns else pl.lit(None),
        close * pl.col("volume") / 1_000_000)
    voted = voted.with_columns(turnover.alias("turnover_mn"))

    rows = voted.tail(days).iter_rows(named=True)
    out = []
    for r in rows:
        rec = {
            "date": str(r["date"]),
            "close": r["close"],
            "summary": r["summary_band"],
            "ma_band": r["ma_band"],
            "osc_band": r["osc_band"],
            "turnover_mn": r.get("turnover_mn"),
            "readings": {label: (None if r.get(col) is None
                                 else spec.format(r[col]))
                         for col, label, spec in INDICATOR_COLS},
        }
        for h in horizons:
            rec[f"ret_{h}d"] = r.get(f"ret_{h}d")
            rec[f"peak_{h}d"] = r.get(f"peak_{h}d")
        out.append(rec)
    return out


def scorecard(rows: list[dict], horizon: int = 10) -> dict:
    """Did a bullish posture actually precede a rise, for THIS stock?

    Small samples by construction — fifteen days is not evidence, it is a
    spot check. The count is returned so nobody mistakes it for one.
    """
    key = f"ret_{horizon}d"
    graded = [r for r in rows if r.get(key) is not None]
    if not graded:
        return {"n": 0, "bullish_n": 0, "bullish_up": None,
                "bearish_n": 0, "bearish_down": None, "horizon": horizon}
    bull = [r for r in graded if r["summary"] in ("Buy", "Strong Buy")]
    bear = [r for r in graded if r["summary"] in ("Sell", "Strong Sell")]
    return {
        "n": len(graded),
        "horizon": horizon,
        "bullish_n": len(bull),
        "bullish_up": (sum(1 for r in bull if r[key] > 0) / len(bull)
                       if bull else None),
        "bearish_n": len(bear),
        "bearish_down": (sum(1 for r in bear if r[key] < 0) / len(bear)
                         if bear else None),
        "mean_after_bullish": (sum(r[key] for r in bull) / len(bull)
                               if bull else None),
        "mean_after_bearish": (sum(r[key] for r in bear) / len(bear)
                               if bear else None),
    }
