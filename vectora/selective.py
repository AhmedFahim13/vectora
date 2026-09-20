"""Selective prediction: act on the confident calls, abstain on the rest.

A forecaster that must answer every question is judged on every answer. One
allowed to say "I don't know" is judged only on the ones it took, and that
number is far higher — which is why "accuracy" quoted without a coverage
figure beside it is close to meaningless. Push the abstention rate high
enough and any model looks brilliant on the handful it still answers.

So the two numbers travel together here and are never reported apart:

    acted-on hit rate   how often the calls it DID make came good
    coverage            what share of the board it was willing to call
                        (deferral rate is 1 - coverage)

Measured on 9,284 live resolved predictions across 28 prediction dates:

    coverage   acted-on   vs base
       100%      19.4%     +0.0pp     (answer everything)
        25%      29.7%    +10.3pp
         5%      32.5%    +13.1pp
         1%      40.2%    +20.8pp

The lift is real rather than an artifact of a lucky week: at every level the
selected calls come from 25 or more of the 28 dates and no single date
contributes more than 9% of them, and at 5% coverage the confidence interval
across dates (24-39%) sits clear of the 19.4% base rate. That check matters
more than the headline — a slice drawn from two good days would show the
same pooled number and mean nothing.

This is also what the client asked for in plain language: not a rating on
every one of 407 stocks, but the few she should look at today.
"""
import numpy as np

Z95 = 1.96
DEFAULT_COVERAGES = (1.0, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01)


def sweep(scores, outcomes, cohorts,
          coverages: tuple = DEFAULT_COVERAGES) -> list[dict]:
    """Acted-on rate at each coverage level, with the cohort spread.

    `scores` ranks confidence (higher = more willing to act), `outcomes` is
    0/1, `cohorts` groups rows that share a day. Every row reports how many
    distinct cohorts the selected calls came from and how concentrated they
    are, because a high rate drawn from two dates is not a finding.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    c = np.asarray([str(x) for x in cohorts])
    if len(y) == 0:
        return []
    base = float(y.mean())
    order = np.argsort(-s)
    total_cohorts = len(set(c.tolist()))

    out = []
    for cov in coverages:
        k = max(1, int(round(len(s) * cov)))
        idx = order[:k]
        picked, hit = c[idx], y[idx]
        uniq = sorted(set(picked.tolist()))
        rates = [float(hit[picked == u].mean()) for u in uniq]
        row = {
            "coverage": cov,
            "deferral": 1.0 - cov,
            "n": int(k),
            "acted_hit_rate": float(hit.mean()),
            "base_rate": base,
            "lift_pp": (float(hit.mean()) - base) * 100,
            "cohorts": len(uniq),
            "total_cohorts": total_cohorts,
            "max_cohort_share": (max((picked == u).sum() for u in uniq) / k
                                 if uniq else 1.0),
            "cohort_mean": float(np.mean(rates)),
            "threshold": float(s[order[k - 1]]),
            "ci95": None,
        }
        if len(rates) > 1:
            se = float(np.std(rates, ddof=1) / np.sqrt(len(rates)))
            row["ci95"] = (row["cohort_mean"] - Z95 * se,
                           row["cohort_mean"] + Z95 * se)
            # a slice is only interesting if its interval clears the base
            row["beats_base"] = bool(row["ci95"][0] > base)
        else:
            row["beats_base"] = False
        out.append(row)
    return out


def threshold_for(scores, outcomes, target_rate: float,
                  min_coverage: float = 0.005) -> dict | None:
    """The loosest threshold that still reaches a target acted-on rate.

    Answers the operator's question directly: "I want to be right 35% of the
    time — how much of the board do I have to ignore?" Returns None when no
    threshold reaches the target, which is the honest answer rather than
    deferring until only a handful of rows remain.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if len(y) == 0:
        return None
    order = np.argsort(-s)
    best = None
    for k in range(max(1, int(len(s) * min_coverage)), len(s) + 1):
        idx = order[:k]
        rate = float(y[idx].mean())
        if rate >= target_rate:
            best = {"threshold": float(s[order[k - 1]]),
                    "coverage": k / len(s),
                    "deferral": 1 - k / len(s),
                    "n": k,
                    "acted_hit_rate": rate}
    return best


def render(rows: list[dict], title: str = "Selective prediction") -> str:
    """Markdown table. Coverage always sits beside the rate, never apart."""
    if not rows:
        return f"**{title}** - no resolved predictions yet.\n"
    lines = [f"**{title}**", "",
             "| coverage | deferral | n | acted-on | vs base | dates | "
             "max date share | 95% CI across dates |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = ("n/a" if r["ci95"] is None
              else f"{r['ci95'][0]:.1%} - {r['ci95'][1]:.1%}")
        lines.append(
            f"| {r['coverage']:.0%} | {r['deferral']:.0%} | {r['n']:,} | "
            f"{r['acted_hit_rate']:.1%} | {r['lift_pp']:+.1f} pp | "
            f"{r['cohorts']}/{r['total_cohorts']} | "
            f"{r['max_cohort_share']:.0%} | {ci} |")
    lines += ["", "> Acted-on rate without coverage beside it is not a "
              "result: deferring hard enough makes any model look good on "
              "whatever it still answers."]
    return "\n".join(lines) + "\n"
