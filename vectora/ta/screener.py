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
from vectora.ta import indicators, rating

WATCHLIST_PATH = (Path(__file__).resolve().parent.parent / "config"
                  / "watchlist.yaml")


def load_watchlist(path: Path = WATCHLIST_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["groups"]


def run(con, date_str: str | None = None) -> dict:
    panel = base.load_panel(con).select(
        ["symbol", "date", "open", "high", "low", "close"])
    ind = indicators.add_all(panel).filter(pl.col("ma_slow").is_not_null())
    run_date = date_str or str(ind["date"].max())
    today = ind.filter(pl.col("date") == pl.lit(run_date).str.to_date())
    if today.height == 0:
        return {"date": run_date, "rated": 0}
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
