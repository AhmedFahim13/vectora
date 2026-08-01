"""Rate the whole board for one date and persist to ta_ratings.

Stores both the absolute band (the TradingView-style posture the reader
recognises) and the cross-sectional decile within the day. Validation
measures both: the absolute band carries the larger measured edge, the
decile isolates stock selection from market direction.
"""
import json
from pathlib import Path

import polars as pl
import yaml

from vectora import db as vdb
from vectora.features import base
from vectora.ta import gauges, indicators, rating

WATCHLIST_PATH = (Path(__file__).resolve().parent.parent / "config"
                  / "watchlist.yaml")


def load_watchlist(path: Path = WATCHLIST_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["groups"]


def run(con, date_str: str | None = None) -> dict:
    panel = base.load_panel(con).select(
        ["symbol", "date", "open", "high", "low", "close", "volume"])
    ind = indicators.add_tradingview_set(indicators.add_all(panel))
    ind = ind.filter(pl.col("ma_slow").is_not_null())
    run_date = date_str or str(ind["date"].max())
    # the gauge rules need yesterday's oscillator readings, so shift BEFORE
    # slicing to the run date rather than after (a one-row slice has no prior)
    ind = gauges.add_prev(ind)
    today = ind.filter(pl.col("date") == pl.lit(run_date).str.to_date())
    if today.height == 0:
        return {"date": run_date, "rated": 0}
    _store_gauges(con, today, run_date)
    rated = rating.rate_frame(today)
    rated = rated.with_columns(
        ((pl.col("ta_score").rank("average") / pl.len() * 10)
         .floor().clip(0, 9).cast(pl.Int8)).alias("ta_decile"))
    rows = [{
        "date": run_date, "symbol": r["symbol"], "score": int(r["ta_score"]),
        "band": r["ta_band"], "votes": r["ta_votes"],
        "rsi": r.get("rsi14"), "macd_hist": r.get("macd_hist"),
        "bb_pos": r.get("bb_pos"), "st_dir": int(r.get("st_dir") or 0),
    } for r in rated.iter_rows(named=True)]
    vdb.upsert(con, "ta_ratings", rows)
    bands: dict = {}
    for r in rows:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
    return {"date": run_date, "rated": len(rows), "bands": bands,
            "watchlist_groups": len(load_watchlist())}


def _store_gauges(con, today: pl.DataFrame, run_date: str) -> int:
    """Persist the 26-component TradingView gauges alongside the 6-family score."""
    rows = []
    for r in today.iter_rows(named=True):
        g = gauges.rate_row(r)
        rows.append({
            "date": run_date, "symbol": r["symbol"],
            "ma_mean": g["ma"]["mean"], "ma_band": g["ma"]["band"],
            "ma_buy": g["ma"]["buy"], "ma_neutral": g["ma"]["neutral"],
            "ma_sell": g["ma"]["sell"],
            "osc_mean": g["osc"]["mean"], "osc_band": g["osc"]["band"],
            "osc_buy": g["osc"]["buy"], "osc_neutral": g["osc"]["neutral"],
            "osc_sell": g["osc"]["sell"],
            "summary_mean": g["summary_mean"], "summary_band": g["summary_band"],
            "votes": json.dumps({"ma": g["ma_votes"], "osc": g["osc_votes"]}),
        })
    return vdb.upsert(con, "ta_gauges", rows)


def gauges_for(con, date_str: str, symbols: list[str] | None = None) -> dict:
    """{symbol: gauge dict} for one date — used by the screener page."""
    q = ("SELECT symbol, ma_mean, ma_band, ma_buy, ma_neutral, ma_sell, "
         "osc_mean, osc_band, osc_buy, osc_neutral, osc_sell, "
         "summary_mean, summary_band, votes FROM ta_gauges WHERE date = ?")
    params: list = [date_str]
    if symbols:
        q += " AND symbol IN (" + ",".join("?" * len(symbols)) + ")"
        params += symbols
    cols = ("ma_mean", "ma_band", "ma_buy", "ma_neutral", "ma_sell",
            "osc_mean", "osc_band", "osc_buy", "osc_neutral", "osc_sell",
            "summary_mean", "summary_band")
    out = {}
    for row in con.execute(q, params).fetchall():
        d = dict(zip(cols, row[1:-1], strict=True))
        d["votes"] = json.loads(row[-1])
        out[row[0]] = d
    return out


def ranked(con, date_str: str, symbols: list[str] | None = None,
           limit: int = 50, ascending: bool = False) -> list[dict]:
    q = ("SELECT r.symbol, r.score, r.band, r.votes, r.rsi, r.macd_hist, "
         "r.bb_pos, r.st_dir, coalesce(s.category,'?'), coalesce(s.sector,'?') "
         "FROM ta_ratings r LEFT JOIN symbols s ON s.symbol = r.symbol "
         "WHERE r.date = ?")
    params: list = [date_str]
    if symbols:
        q += " AND r.symbol IN (" + ",".join("?" * len(symbols)) + ")"
        params += symbols
    q += f" ORDER BY r.score {'ASC' if ascending else 'DESC'}, r.symbol LIMIT ?"
    params.append(limit)
    return [{"symbol": s, "score": sc, "band": b, "votes": json.loads(v),
             "rsi": rsi, "macd_hist": mh, "bb_pos": bp, "st_dir": sd,
             "category": cat, "sector": sec}
            for s, sc, b, v, rsi, mh, bp, sd, cat, sec
            in con.execute(q, params).fetchall()]
