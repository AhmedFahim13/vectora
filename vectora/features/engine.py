# vectora/features/engine.py
"""Feature engine: panel + symbol metadata -> every registered feature ->
wide parquet. Full recompute per run (1M rows x 40 features is seconds in
polars); incremental computation is deliberate YAGNI at this scale."""
from pathlib import Path

import polars as pl

from vectora.features import base, families, registry
from vectora.settings import FEATURES_DIR

DEFAULT_OUT = FEATURES_DIR / "features.parquet"


def compute(con, out_path: Path = DEFAULT_OUT,
            specs: list | None = None) -> pl.DataFrame:
    panel = base.load_panel(con)
    meta = con.execute(
        "SELECT symbol, sector, first_seen FROM symbols").pl().with_columns(
        pl.col("first_seen").cast(pl.Date))
    df = panel.join(meta, on="symbol", how="left").sort(["symbol", "date"])
    for spec in (specs or registry.load()):
        df = families.apply(df, spec.name, spec.fn, spec.params)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    return df
