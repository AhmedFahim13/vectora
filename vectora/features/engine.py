# vectora/features/engine.py
"""Feature engine: panel + symbol metadata -> every registered feature ->
wide parquet. Full recompute per run (1M rows x 40 features is seconds in
polars); incremental computation is deliberate YAGNI at this scale."""
from pathlib import Path

import polars as pl

from vectora.features import base, families, registry
from vectora.settings import FEATURES_DIR

DEFAULT_OUT = FEATURES_DIR / "features.parquet"


REGIME_CODES = {"Panic": 1, "Bear": 2, "LowLiquidity": 3, "Sideways": 4,
                "Bull": 5, "Recovery": 6, "SpeculativeHeat": 7}


def compute(con, out_path: Path = DEFAULT_OUT,
            specs: list | None = None) -> pl.DataFrame:
    panel = base.load_panel(con)
    meta = con.execute(
        "SELECT symbol, sector, first_seen FROM symbols").pl().with_columns(
        pl.col("first_seen").cast(pl.Date))
    df = panel.join(meta, on="symbol", how="left").sort(["symbol", "date"])

    # event base columns (leakage-safe: post_date <= date by construction —
    # the last-event date is forward-filled from strictly past/same-day rows)
    events = con.execute(
        """
        SELECT e.symbol, e.post_date AS date,
               max(CASE WHEN l.materiality >= 3 THEN 1 ELSE 0 END) AS mat3,
               max(CASE WHEN l.event_type = 'board_meeting' THEN 1 ELSE 0 END)
                   AS bm
        FROM events e JOIN event_labels l ON l.event_id = e.id
        WHERE e.symbol IS NOT NULL GROUP BY 1, 2
        """).pl()
    if events.height > 0:
        df = df.join(events, on=["symbol", "date"], how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Int32).alias("mat3"),
                             pl.lit(None, dtype=pl.Int32).alias("bm"))
    last_ev = (pl.when(pl.col("mat3") == 1).then(pl.col("date"))
               .otherwise(None).forward_fill().over("symbol"))
    last_bm = (pl.when(pl.col("bm") == 1).then(pl.col("date"))
               .otherwise(None).forward_fill().over("symbol"))
    df = df.with_columns(
        (pl.col("date") - last_ev).dt.total_days()
        .alias("days_since_event"),
        ((pl.col("date") - last_bm).dt.total_days() <= 3)
        .cast(pl.Int8).fill_null(0).alias("board_meeting_soon"),
    ).drop(["mat3", "bm"])

    # regime code (market-wide, per date)
    regimes = con.execute("SELECT date, regime FROM regimes").pl()
    if regimes.height > 0:
        regimes = regimes.with_columns(
            pl.col("regime").replace_strict(REGIME_CODES, default=0)
            .alias("regime_code")).drop("regime")
        df = df.join(regimes, on="date", how="left")
        df = df.with_columns(pl.col("regime_code").fill_null(0))
    else:
        df = df.with_columns(pl.lit(0).alias("regime_code"))

    for spec in (specs or registry.load()):
        df = families.apply(df, spec.name, spec.fn, spec.params)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    return df
