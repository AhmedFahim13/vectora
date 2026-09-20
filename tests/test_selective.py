"""Selective prediction: the acted-on rate must never travel without coverage.

Deferring hard enough makes any model look excellent on the few calls it
still makes. These tests pin the properties that stop that number lying:
coverage is always reported, and a slice drawn from one lucky day is flagged
rather than celebrated.
"""
import numpy as np

from vectora import selective


def _ranked(n=1000, cohorts=20, seed=0):
    """Scores that genuinely rank: higher score, higher chance of a hit."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < (0.1 + 0.5 * s)).astype(int)
    c = [f"d{i % cohorts}" for i in range(n)]
    return s, y, c


def test_acting_on_fewer_calls_raises_the_rate():
    s, y, c = _ranked()
    rows = selective.sweep(s, y, c)
    full = next(r for r in rows if r["coverage"] == 1.0)
    tight = next(r for r in rows if r["coverage"] == 0.05)
    assert full["acted_hit_rate"] < tight["acted_hit_rate"]
    assert full["lift_pp"] == 0.0            # covering everything IS the base


def test_coverage_and_deferral_always_present():
    """The number that makes the headline honest."""
    s, y, c = _ranked()
    for r in selective.sweep(s, y, c):
        assert 0 < r["coverage"] <= 1
        assert abs(r["coverage"] + r["deferral"] - 1.0) < 1e-9
        assert r["n"] >= 1


def test_a_slice_drawn_from_one_day_is_not_credited():
    """A brilliant rate from a single date is a lucky day, not a finding."""
    n = 400
    s = np.zeros(n)
    y = np.zeros(n, dtype=int)
    c = [f"d{i % 20}" for i in range(n)]
    # the top-scoring rows all sit on one date and all hit
    for i in range(20):
        s[i] = 10.0
        y[i] = 1
        c[i] = "lucky-day"
    rows = selective.sweep(s, y, c, coverages=(0.05,))
    r = rows[0]
    assert r["acted_hit_rate"] == 1.0
    assert r["cohorts"] == 1
    assert r["max_cohort_share"] == 1.0
    assert r["beats_base"] is False, "a one-date slice must not count"


def test_spread_slice_that_beats_the_base_is_credited():
    s, y, c = _ranked(n=4000, cohorts=30)
    rows = selective.sweep(s, y, c, coverages=(0.10,))
    r = rows[0]
    assert r["cohorts"] >= 25
    assert r["max_cohort_share"] < 0.2
    assert r["beats_base"] is True


def test_threshold_for_target_rate():
    s, y, c = _ranked(n=3000)
    got = selective.threshold_for(s, y, target_rate=0.45)
    assert got is not None
    assert got["acted_hit_rate"] >= 0.45
    assert 0 < got["coverage"] < 1


def test_unreachable_target_returns_none_rather_than_deferring_to_nothing():
    """Refusing is the honest answer; shrinking to three rows is not."""
    s, y, c = _ranked()
    assert selective.threshold_for(s, y, target_rate=0.99) is None


def test_render_states_the_caveat():
    s, y, c = _ranked()
    text = selective.render(selective.sweep(s, y, c))
    assert "coverage" in text
    assert "not a result" in text


def test_empty_input_is_safe():
    assert selective.sweep([], [], []) == []
    assert selective.threshold_for([], [], 0.5) is None
    assert "no resolved" in selective.render([])
