# vectora/labels.py
"""Forward-return label grid (spec §9.1): y_gX_hH = 1 if the max close over
the next H trading rows gains >= X% vs today's close; y_dX_hH mirrors for
drawdowns. Labels are null when fewer than H future rows exist (end of data)
— unresolved, not negative. Uses raw closes; corporate-action gaps inside
the forward window are a documented Phase 2 approximation (base.py note)."""
import polars as pl


def _fwd_extreme(h: int, kind: str) -> pl.Expr:
    shifts = [pl.col("close").shift(-k).over("symbol") for k in range(1, h + 1)]
    agg = pl.max_horizontal(shifts) if kind == "max" else pl.min_horizontal(shifts)
    complete = pl.col("close").shift(-h).over("symbol").is_not_null()
    return pl.when(complete).then(agg).otherwise(None)


def make_labels(panel: pl.DataFrame, thresholds=(0.03, 0.05, 0.10, 0.20),
                horizons=(1, 3, 5, 10, 30), downside: bool = False,
                continuous: bool = False) -> pl.DataFrame:
    df = panel.sort(["symbol", "date"])
    cols = []
    for h in horizons:
        fwd_max = _fwd_extreme(h, "max")
        fwd_min = _fwd_extreme(h, "min")
        if continuous:
            cols.append((fwd_max / pl.col("close") - 1).alias(f"fwdmax_h{h}"))
            cols.append((fwd_min / pl.col("close") - 1).alias(f"fwdmin_h{h}"))
        for x in thresholds:
            pct = round(x * 100)
            cols.append(
                (fwd_max / pl.col("close") - 1 >= x)
                .cast(pl.Int8).alias(f"y_g{pct}_h{h}"))
            if downside:
                cols.append(
                    (fwd_min / pl.col("close") - 1 <= -x)
                    .cast(pl.Int8).alias(f"y_d{pct}_h{h}"))
    return df.with_columns(cols)
