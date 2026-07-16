# tests/features/test_families.py
import math

import polars as pl

from vectora.features import families, registry


def _panel():
    # 30 days, 2 symbols, deterministic prices; includes sector + first_seen
    rows = []
    for i in range(30):
        d = f"2026-06-{i + 1:02d}" if i < 30 else None
        rows.append(dict(symbol="AAA", date=d, open=100 + i, high=102 + i,
                         low=99 + i, close=100 + i, ycp=99 + i, trades=100,
                         value_mn=10.0, volume=1000 + 10 * i,
                         sector="Bank", first_seen="2020-01-01",
                         ret=(100 + i) / (99 + i) - 1))
        rows.append(dict(symbol="BBB", date=d, open=50, high=50.5, low=49.5,
                         close=50.0, ycp=50.0, trades=10, value_mn=0.5,
                         volume=200, sector="Bank", first_seen="2024-01-01",
                         ret=0.0))
    return pl.DataFrame(rows).with_columns(
        pl.col("date").str.to_date(), pl.col("first_seen").str.to_date())


def test_ret_nd_compounds_returns():
    df = families.apply(_panel(), "ret_5d", "ret_nd", {"days": 5})
    aaa = df.filter(pl.col("symbol") == "AAA").sort("date")
    # close 105 on day 6 vs close 100 on day 1 -> 5%
    assert abs(aaa["ret_5d"][5] - (105 / 100 - 1)) < 1e-9
    assert aaa["ret_5d"][3] is None  # not enough history


def test_zscore_flags_volume_spike():
    df = _panel().with_columns(
        pl.when((pl.col("symbol") == "BBB") & (pl.col("date") == pl.date(2026, 6, 30)))
        .then(5000).otherwise(pl.col("volume")).alias("volume"))
    out = families.apply(df, "volume_z_21d", "zscore_col",
                         {"col": "volume", "days": 21})
    spike = out.filter((pl.col("symbol") == "BBB")
                       & (pl.col("date") == pl.date(2026, 6, 30)))
    assert spike["volume_z_21d"][0] > 3.0


def test_cross_rank_is_within_date_and_in_unit_range():
    df = families.apply(_panel(), "ret_5d", "ret_nd", {"days": 5})
    out = families.apply(df, "ret_5d_xrank", "cross_rank", {"of": "ret_5d"})
    last = out.filter(pl.col("date") == pl.date(2026, 6, 30))
    vals = [v for v in last["ret_5d_xrank"].to_list() if v is not None]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert max(vals) > min(vals)  # AAA rising vs BBB flat -> different ranks


def test_every_registered_fn_exists_and_runs():
    df = _panel()
    for spec in registry.load():
        assert spec.fn in families.FNS, f"missing fn {spec.fn} for {spec.name}"
        df = families.apply(df, spec.name, spec.fn, spec.params)
        assert spec.name in df.columns
        col = df[spec.name]
        finite = [v for v in col.to_list() if v is not None]
        assert all(not (isinstance(v, float) and math.isinf(v)) for v in finite), \
            f"{spec.name} produced inf"
