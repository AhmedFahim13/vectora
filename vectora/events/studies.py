# vectora/events/studies.py
"""Event-impact studies (spec §12.2): market-adjusted forward returns per
event type. abnormal = symbol forward return minus median market forward
return over the same window; the table answers 'what historically happens
in the H days after this kind of announcement on the DSE'."""
from pathlib import Path

import polars as pl

from vectora import db as vdb
from vectora.features import base
from vectora.settings import VAULT_DIR
from vectora.vault.generator import _write_machine

DEFAULT_HORIZONS = (1, 3, 5, 10)
MIN_EVENTS = 30


def compute(con, min_events: int = MIN_EVENTS,
            horizons: tuple = DEFAULT_HORIZONS) -> dict:
    events = con.execute(
        """
        SELECT e.symbol, e.post_date AS date, l.event_type
        FROM events e JOIN event_labels l ON l.event_id = e.id
        WHERE e.symbol IS NOT NULL AND l.materiality >= 1
        """).pl()
    if events.height == 0:
        return {"types": 0}
    panel = base.load_panel(con).select(["symbol", "date", "close"])
    # forward return per symbol and market median forward return per date
    out_rows = []
    frame = panel.sort(["symbol", "date"])
    for h in horizons:
        fwd = frame.with_columns(
            (pl.col("close").shift(-h).over("symbol") / pl.col("close") - 1)
            .alias("fwd"))
        mkt = fwd.group_by("date").agg(
            pl.col("fwd").median().alias("mkt_fwd"))
        joined = (events.join(fwd.select(["symbol", "date", "fwd"]),
                              on=["symbol", "date"], how="inner")
                  .join(mkt, on="date", how="left")
                  .with_columns((pl.col("fwd") - pl.col("mkt_fwd"))
                                .alias("abn"))
                  .filter(pl.col("abn").is_not_null()))
        stats = (joined.group_by("event_type")
                 .agg(pl.len().alias("n"),
                      pl.col("abn").mean().alias("mean_abn"),
                      pl.col("abn").median().alias("median_abn"),
                      (pl.col("abn") > 0).mean().alias("pos_share"))
                 .filter(pl.col("n") >= min_events))
        for r in stats.iter_rows(named=True):
            out_rows.append({
                "event_type": r["event_type"], "horizon": h,
                "n": int(r["n"]), "mean_abn_ret": r["mean_abn"],
                "median_abn_ret": r["median_abn"],
                "pos_share": r["pos_share"],
            })
    if out_rows:
        vdb.upsert(con, "event_studies", out_rows)
    return {"types": len({r["event_type"] for r in out_rows})}


def write_vault_note(con, vault_dir: Path = VAULT_DIR) -> Path:
    rows = con.execute(
        """
        SELECT event_type, horizon, n, mean_abn_ret, median_abn_ret, pos_share
        FROM event_studies ORDER BY event_type, horizon
        """).fetchall()
    lines = ["# Event impact studies", "",
             "Market-adjusted forward returns after each announcement type.",
             "", "| type | h | n | mean | median | share>0 |", "|" + "---|" * 6]
    for t, h, n, mean, median, pos in rows:
        lines.append(f"| {t} | h{h} | {n} | {mean:+.2%} | {median:+.2%} "
                     f"| {pos:.0%} |")
    path = Path(vault_dir) / "Patterns" / "event-impact.md"
    _write_machine(path, "\n".join(lines))
    return path
