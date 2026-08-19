"""Cohort-aware evidence accounting.

Predictions made on the same day share the same market. Counting them as
independent observations is what makes a track record look far stronger
than it is, so the report must measure in cohorts, not rows.
"""
from vectora.evaluate import report


def test_single_cohort_has_no_usable_spread():
    """One date is one observation however many rows it carries."""
    rows = [("2026-07-16", 1)] * 86 + [("2026-07-16", 0)] * 246
    c = report.cohort_stats(rows)
    assert c["cohorts"] == 1
    assert c["n"] == 332
    assert c["cohort_se"] is None          # a single point has no spread
    assert c["ci95"] is None


def test_cohort_se_uses_the_spread_between_dates():
    rows = ([("d1", 1)] * 20 + [("d1", 0)] * 80        # 20%
            + [("d2", 1)] * 40 + [("d2", 0)] * 60      # 40%
            + [("d3", 1)] * 30 + [("d3", 0)] * 70)     # 30%
    c = report.cohort_stats(rows)
    assert c["cohorts"] == 3
    assert abs(c["cohort_mean"] - 0.30) < 1e-9
    # sample std of (.2,.4,.3) is .1 -> SE = .1/sqrt(3)
    assert abs(c["cohort_se"] - 0.1 / 3 ** 0.5) < 1e-9


def test_naive_interval_is_reported_as_too_narrow():
    """The whole point: the row-wise interval understates uncertainty."""
    rows = ([("d1", 1)] * 20 + [("d1", 0)] * 80
            + [("d2", 1)] * 40 + [("d2", 0)] * 60
            + [("d3", 1)] * 30 + [("d3", 0)] * 70)
    c = report.cohort_stats(rows)
    assert c["naive_se"] < c["cohort_se"]
    assert c["se_inflation"] > 1.0


def test_identical_cohorts_collapse_the_inflation():
    """If every date behaves the same, cohorting costs nothing."""
    rows = [(f"d{i}", 1) for i in range(3) for _ in range(30)]
    rows += [(f"d{i}", 0) for i in range(3) for _ in range(70)]
    c = report.cohort_stats(rows)
    assert c["cohort_se"] == 0.0
    assert c["se_inflation"] == 0.0


def test_empty_input():
    c = report.cohort_stats([])
    assert c["cohorts"] == 0 and c["n"] == 0
