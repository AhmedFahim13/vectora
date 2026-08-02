"""Sector rotation from relative strength (spec: Phase 6D).

The question a rotation view answers is not "which sector went up" — in a
rising market almost all of them did. It is "which sector is beating the
market, and is that lead still growing". Those are two different numbers,
and plotting them against each other is the Relative Rotation Graph idea:

    relative strength > 0, momentum > 0   Leading
    relative strength > 0, momentum < 0   Weakening   (lead is decaying)
    relative strength < 0, momentum < 0   Lagging
    relative strength < 0, momentum > 0   Improving   (turn may be starting)

Sectors are equal-weighted across their members, not capitalisation
weighted. On the DSE a handful of giants (GP, SQURPHARMA, BATBC) would
otherwise BE their sector, and the view would just restate those tickers.

The benchmark is the equal-weighted mean of the same universe rather than
DSEX, so sector and benchmark are computed identically and the difference
between them is not an artifact of two different weighting schemes.

One consequence worth knowing when reading the numbers: relative strength
across sectors does not sum to zero. Compounding is convex, so the mean of
compounded sector returns sits slightly above the compounded mean daily
return. The gap is small next to the dispersion it measures, but it is a
real property of the arithmetic, not a rounding artifact.
"""
import polars as pl

from vectora import db as vdb
from vectora.features import base

# The DSE symbol master carries 243 debt instruments — 219 government
# T-bonds, 16 corporate bonds, 8 debentures — whose MEDIAN daily traded value
# is exactly zero. Left in, they were 40% of an equal-weighted benchmark while
# contributing no real price discovery, which made every equity sector's
# relative strength a measurement of bond staleness.
EXCLUDED_TYPES = ("Debt", "Corporate Bond", "Debenture")
LOOKBACKS = (5, 21, 63, 126)
MOMENTUM_LOOKBACK = 21
_RS_WINDOW = 21


def _cum(col: str, n: int) -> pl.Expr:
    """Compounded return over n rows, via logs — mean daily returns are not
    additive over a window and summing them overstates trends."""
    return ((pl.col(col).log1p().rolling_sum(n).over("sector")).exp() - 1)


def compute(con) -> pl.DataFrame:
    panel = base.load_panel(con).select(["symbol", "date", "ret", "value_mn"])
    secs = con.execute(
        "SELECT symbol, sector FROM symbols WHERE sector IS NOT NULL "
        "AND coalesce(instrument_type, '') NOT IN "
        f"({', '.join(repr(t) for t in EXCLUDED_TYPES)})").pl()
    joined = panel.join(secs, on="symbol", how="inner").filter(
        pl.col("ret").is_not_null()
        # a stale quote is not a return: an instrument that did not trade
        # contributes ret=0 every day and quietly damps the benchmark.
        # Null is "unknown", not "zero" — the 1.06M-row Mendeley backfill
        # carries no turnover column at all, and testing `> 0` alone would
        # discard thirteen years of history without saying so.
        & (pl.col("value_mn").is_null() | (pl.col("value_mn") > 0)))
    if joined.height == 0:
        return pl.DataFrame()

    market = (joined.group_by("date").agg(pl.col("ret").mean().alias("mkt"))
              .sort("date"))
    sector = (joined.group_by(["sector", "date"])
              .agg(pl.col("ret").mean().alias("ret"),
                   pl.len().alias("n_symbols"))
              .sort(["sector", "date"]))

    # market cumulative returns, computed once and broadcast to every sector
    mkt = market.with_columns([
        ((pl.col("mkt").log1p().rolling_sum(n)).exp() - 1).alias(f"mkt_{n}")
        for n in LOOKBACKS])
    d = sector.join(mkt.drop("mkt"), on="date", how="left").sort(
        ["sector", "date"])
    d = d.with_columns([_cum("ret", n).alias(f"ret_{n}") for n in LOOKBACKS])
    d = d.with_columns([
        (pl.col(f"ret_{n}") - pl.col(f"mkt_{n}")).alias(f"rs_{n}")
        for n in LOOKBACKS])
    # is the lead growing? compare relative strength to itself 21 rows back
    d = d.with_columns(
        (pl.col(f"rs_{_RS_WINDOW}")
         - pl.col(f"rs_{_RS_WINDOW}").shift(MOMENTUM_LOOKBACK).over("sector"))
        .alias("rs_momentum"))
    rs, mom = pl.col(f"rs_{_RS_WINDOW}"), pl.col("rs_momentum")
    return d.with_columns(
        pl.when(rs.is_null() | mom.is_null()).then(pl.lit("Insufficient data"))
        .when((rs > 0) & (mom > 0)).then(pl.lit("Leading"))
        .when((rs > 0) & (mom <= 0)).then(pl.lit("Weakening"))
        .when((rs <= 0) & (mom > 0)).then(pl.lit("Improving"))
        .otherwise(pl.lit("Lagging")).alias("quadrant"))


def run(con, date_str: str | None = None) -> dict:
    d = compute(con)
    if d.height == 0:
        return {"date": date_str, "sectors": 0}
    run_date = date_str or str(d["date"].max())
    today = d.filter(pl.col("date") == pl.lit(run_date).str.to_date())
    rows = [{
        "date": run_date, "sector": r["sector"], "n_symbols": int(r["n_symbols"]),
        "ret_5d": r["ret_5"], "ret_21d": r["ret_21"], "ret_63d": r["ret_63"],
        "ret_126d": r["ret_126"], "rs_21d": r["rs_21"], "rs_63d": r["rs_63"],
        "rs_momentum": r["rs_momentum"], "quadrant": r["quadrant"],
    } for r in today.iter_rows(named=True)]
    # replace the date wholesale: an upsert alone leaves rows for sectors that
    # have since been excluded from the universe sitting in the table forever
    con.execute("DELETE FROM sector_rs WHERE date = ?", [run_date])
    vdb.upsert(con, "sector_rs", rows)
    quads: dict = {}
    for r in rows:
        quads[r["quadrant"]] = quads.get(r["quadrant"], 0) + 1
    return {"date": run_date, "sectors": len(rows), "quadrants": quads}


def load(con, date_str: str) -> list[dict]:
    cols = ("sector", "n_symbols", "ret_5d", "ret_21d", "ret_63d", "ret_126d",
            "rs_21d", "rs_63d", "rs_momentum", "quadrant")
    rows = con.execute(
        f"SELECT {', '.join(cols)} FROM sector_rs WHERE date = ? "
        "ORDER BY rs_21d DESC", [date_str]).fetchall()
    return [dict(zip(cols, r, strict=True)) for r in rows]
