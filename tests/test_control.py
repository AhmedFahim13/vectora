"""Controls must catch a confound dressed up as a signal.

The live system reported +7.9pp for months while ranking by trailing
volatility alone scored +10.8pp. These tests pin the checks that would have
caught that on day one: a score which is secretly the confound must fail,
and a genuine edge must survive holding the confound constant.
"""
import numpy as np

from vectora import control


def _world(n=20000, seed=0, signal_strength=0.25, confound_strength=0.6):
    """Outcomes driven by BOTH a real signal and a confound.

    `confound` raises the hit rate without any directional information --
    exactly how volatility inflates a "gained 5%" label.
    """
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    conf = rng.uniform(0.2, 2.0, size=n)
    p = 0.2 + signal_strength * (sig > 0.8) + confound_strength * 0.1 * conf
    y = (rng.uniform(size=n) < np.clip(p, 0, 1)).astype(int)
    return sig, y, conf


def test_confound_alone_does_not_beat_the_control():
    """A score that IS the confound cannot claim to beat the confound."""
    sig, y, conf = _world()
    cmp_ = control.compare(conf, y, conf)
    assert cmp_["beats_control"] is False
    assert abs(cmp_["margin_pp"]) < 1e-9


def test_real_signal_beats_the_control():
    sig, y, conf = _world(signal_strength=0.35, confound_strength=0.1)
    cmp_ = control.compare(sig, y, conf)
    assert cmp_["beats_control"] is True
    assert cmp_["margin_pp"] > 0


def test_random_score_has_no_lift():
    sig, y, conf = _world()
    rng = np.random.default_rng(1)
    assert abs(control.naive_lift(rng.uniform(size=len(y)), y)) < 2.0


def test_matched_lift_strips_the_confound():
    """A pure confound must show ~no lift once the confound is held fixed."""
    sig, y, conf = _world(signal_strength=0.0, confound_strength=1.0)
    naive = control.naive_lift(conf, y)
    matched = control.matched_lift(conf, y, conf)
    assert naive > 3.0, "the confound should look good naively"
    assert matched["lift_pp"] < naive / 2, "and much weaker once matched"


def test_matched_lift_keeps_a_real_edge_and_is_consistent():
    sig, y, conf = _world(signal_strength=0.35)
    m = control.matched_lift(sig, y, conf)
    assert m["lift_pp"] > 3.0
    assert m["consistent"] is True, "a real edge shows up in every stratum"
    assert len(m["per_stratum"]) == control.N_STRATA


def test_strata_split_evenly():
    v = np.arange(10000, dtype=float)
    s = control._strata(v)
    counts = np.bincount(s, minlength=control.N_STRATA)
    assert counts.min() > 900 and counts.max() < 1100


def test_strata_survive_nan():
    v = np.arange(1000, dtype=float)
    v[::10] = np.nan
    s = control._strata(v)
    assert len(s) == 1000
    assert not np.isnan(s).any()


def test_render_never_shows_naive_lift_alone():
    sig, y, conf = _world()
    text = control.render(control.compare(sig, y, conf),
                          control.matched_lift(sig, y, conf))
    assert "control" in text.lower()
    assert "constant" in text.lower(), "the matched number must be present"
    assert "Both numbers, always." in text


def test_render_warns_loudly_when_the_control_wins():
    sig, y, conf = _world(signal_strength=0.0, confound_strength=1.0)
    text = control.render(control.compare(conf * 0.999, y, conf),
                          control.matched_lift(conf, y, conf))
    assert "control wins" in text.lower()
    assert "not evidence" in text.lower()


def test_inconsistent_edge_is_called_unproven():
    """An edge living in one stratum must not be sold as a finding."""
    n = 20000
    rng = np.random.default_rng(3)
    conf = rng.uniform(0, 1, n)
    sig = rng.normal(size=n)
    y = np.zeros(n, dtype=int)
    top = conf > 0.9                       # only the top stratum has an edge
    y[top] = (sig[top] > 0).astype(int)
    m = control.matched_lift(sig, y, conf)
    assert m["consistent"] is False
    assert "unproven" in control.render(
        control.compare(sig, y, conf), m).lower()


def test_consistency_bar_is_eight_of_ten_not_unanimity():
    """One unlucky stratum must not condemn an otherwise even edge.

    The live model scored +8.1pp positive in 9 of 10 strata. Demanding all
    ten would have printed 'unproven' there, and a flag that fires on a good
    result is a flag nobody reads.
    """
    assert control.CONSISTENT_SHARE == 0.8
    ten = [{"lift_pp": 1.0} for _ in range(10)]

    def verdict(rows):
        return (sum(1 for r in rows if r["lift_pp"] > 0)
                >= control.CONSISTENT_SHARE * len(rows))

    nine = [dict(r) for r in ten]
    nine[0]["lift_pp"] = -0.5
    eight = [dict(r) for r in nine]
    eight[1]["lift_pp"] = -0.5
    seven = [dict(r) for r in eight]
    seven[2]["lift_pp"] = -0.5
    assert verdict(ten) and verdict(nine) and verdict(eight)
    assert not verdict(seven), "7/10 is coin-flip territory"


def test_empty_input_is_safe():
    assert control.naive_lift([], []) == 0.0
    m = control.matched_lift([], [], [])
    assert m["lift_pp"] == 0.0 and m["per_stratum"] == []
    assert m["consistent"] is False
