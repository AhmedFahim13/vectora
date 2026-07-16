from vectora.predict import risk


def _analog_stats(hit=0.6, up=0.08, down=-0.04, mdd=-0.15, n=20):
    return {"hit_rate": hit, "median_up": up, "median_down": down,
            "max_drawdown": mdd, "n": n}


def test_risk_block_fields_and_rr():
    b = risk.build(vol_21d=0.02, value_mn_med_21d=5.0, category="A",
                   analog_stats=_analog_stats())
    assert b["expected_up"] == 0.08 and b["expected_down"] == -0.04
    assert abs(b["rr_ratio"] - 2.0) < 1e-9
    assert b["analog_max_drawdown"] == -0.15
    assert b["category"] == "A"
    # 500k position vs 20% of 5mn/day absorbable -> 0.5 days
    assert abs(b["exit_days"] - 0.5) < 1e-9


def test_rr_ratio_none_when_downside_zero():
    b = risk.build(vol_21d=0.02, value_mn_med_21d=5.0, category="A",
                   analog_stats=_analog_stats(down=0.0))
    assert b["rr_ratio"] is None


def test_illiquid_name_has_long_exit():
    b = risk.build(vol_21d=0.05, value_mn_med_21d=0.05, category="Z",
                   analog_stats=_analog_stats())
    assert b["exit_days"] == 50.0  # 500k / (0.2 * 50k/day)


def test_missing_liquidity_yields_none_exit():
    b = risk.build(vol_21d=0.02, value_mn_med_21d=None, category="B",
                   analog_stats=_analog_stats())
    assert b["exit_days"] is None
