# tests/zmod/test_pump.py
import polars as pl

from vectora.zmod import pump


def _frame(rows):
    base = dict(ret_10d=0.0, ret_21d=0.0, ret_63d=0.0, vol_ratio_5_21=1.0,
                obv_slope_21d=0.1, volume_z_21d=0.0)
    return pl.DataFrame([{**base, **r} for r in rows])


def test_phases():
    df = _frame([
        {"symbol": "QUIET"},
        {"symbol": "MARKUP", "ret_21d": 0.40, "vol_ratio_5_21": 2.0},
        {"symbol": "DIST", "ret_21d": 0.40, "vol_ratio_5_21": 2.0,
         "obv_slope_21d": -0.3},
        {"symbol": "COLLAPSE", "ret_10d": -0.25, "ret_63d": 0.5},
    ])
    out = pump.phase_and_score(df, categories={})
    phases = dict(zip(out["symbol"].to_list(), out["phase"].to_list(), strict=True))
    assert phases == {"QUIET": "quiet", "MARKUP": "markup",
                      "DIST": "distribution", "COLLAPSE": "collapse"}


def test_score_ranks_runners_highest_and_boosts_z():
    rows = [{"symbol": f"S{i}", "ret_21d": 0.01 * i,
             "vol_ratio_5_21": 0.9 + 0.05 * i} for i in range(20)]
    rows.append({"symbol": "ZPUMP", "ret_21d": 0.45, "vol_ratio_5_21": 3.0})
    rows.append({"symbol": "APUMP", "ret_21d": 0.45, "vol_ratio_5_21": 3.0})
    out = pump.phase_and_score(_frame(rows),
                               categories={"ZPUMP": "Z", "APUMP": "A"})
    scores = dict(zip(out["symbol"].to_list(), out["score"].to_list(), strict=True))
    assert scores["ZPUMP"] > scores["APUMP"] > scores["S5"]
    assert scores["ZPUMP"] <= 100.0
    assert scores["S0"] < 20


def test_null_features_score_zero_not_crash():
    df = _frame([{"symbol": "NEWLIST", "ret_21d": None,
                  "vol_ratio_5_21": None}])
    out = pump.phase_and_score(df, categories={"NEWLIST": "Z"})
    assert out["score"][0] == 0.0
    assert out["phase"][0] == "quiet"


def test_tables_exist(test_db):
    tables = {r[0] for r in test_db.execute("SHOW TABLES").fetchall()}
    assert {"zwatch", "event_footprints"} <= tables
