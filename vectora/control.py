"""Controls: is a reported edge the signal, or something dumber?

Every hit-rate number this system has ever published came with a base rate
beside it, which answers "better than nothing?" but not "better than the
obvious cheat?". For a target shaped like "did price gain 5% within 10 days"
there is a very obvious cheat: pick volatile stocks. A wild stock brushes
+5% on noise alone. A sleepy one never does. Rank by trailing volatility --
a number that knows nothing whatsoever about direction -- and the hit rate
climbs.

Measured on 788,370 rows of DSE history against the production label:

    ranked by                    top 10%    lift
    the gauge (our signal)        41.0%    +7.9pp
    trailing volatility ONLY      43.9%   +10.8pp   <-- the cheat wins
    random                        33.2%    +0.1pp

So the gauge's headline edge was not mostly signal. It was mostly volatility,
and a one-line control outperformed a rule set built over months. That was
invisible for as long as the only comparison was the base rate.

The machine-learned model is a different story, and the distinction matters:
evaluated walk-forward over 155,435 test rows it scores +19.7pp against the
control's +9.5pp, and holds +11.8pp with volatility fixed in all ten strata.
It has real directional skill. The control is not here to condemn anything --
it is here so that the two cases can be told apart, which the base rate alone
could never do.

The signal is not worthless, though. Hold volatility constant -- compare a
stock only against others with similar volatility -- and a real edge remains:

    volatility-matched lift       +4.6pp (production touch label)
                                  +3.4pp on the close-to-close label, where
                                  every one of the ten deciles was positive
                                  and the range was only +2.9 to +4.1pp

Spread across the deciles is what makes it credible; an artifact concentrates
in the wild names instead of sitting evenly. Hence `consistent` below, which
asks how many strata agree rather than trusting the pooled figure.

So two numbers are reported from here on and neither travels alone:

    naive lift              vs the base rate       (flattering, contaminated)
    volatility-matched      vs volatility peers    (smaller, real)

and the control's own lift, so that a future edge which the control matches
is caught the way this one should have been.
"""
import numpy as np

N_STRATA = 10
CONSISTENT_SHARE = 0.8


def _strata(values, n: int = N_STRATA) -> np.ndarray:
    """Assign each row to a quantile bucket of `values`."""
    v = np.asarray(values, dtype=float)
    v = np.where(np.isnan(v), np.nanmedian(v), v)
    edges = np.quantile(v, np.linspace(0, 1, n + 1))[1:-1]
    return np.clip(np.searchsorted(edges, v), 0, n - 1)


def naive_lift(scores, outcomes, coverage: float = 0.10) -> float:
    """Hit rate of the top `coverage` by score, minus the overall base rate.

    This is the number the system reported before controls existed. Kept so
    the two can be printed side by side and the gap can be seen.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if len(y) == 0:
        return 0.0
    k = max(1, int(len(s) * coverage))
    return float(y[np.argsort(-s)[:k]].mean() - y.mean()) * 100


def matched_lift(scores, outcomes, confound, coverage: float = 0.20,
                 n_strata: int = N_STRATA, min_rows: int = 500) -> dict:
    """Lift with `confound` held constant, plus the per-stratum detail.

    Within each stratum the top `coverage` by score is compared against that
    stratum's own base rate, so a score cannot win by preferring high-
    confound rows. `per_stratum` is returned because an edge that appears in
    one bucket and not the others is a fluke wearing a suit.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if len(y) == 0:
        return {"lift_pp": 0.0, "base_rate": 0.0, "acted_rate": 0.0,
                "per_stratum": [], "consistent": False, "n": 0}
    strat = _strata(confound, n_strata)

    hits = picked = base_hits = base_n = 0
    rows = []
    for i in range(n_strata):
        m = strat == i
        if m.sum() < min_rows:
            continue
        ys = y[m]
        k = max(1, int(m.sum() * coverage))
        top = ys[np.argsort(-s[m])[:k]]
        hits += int(top.sum())
        picked += k
        base_hits += int(ys.sum())
        base_n += int(m.sum())
        rows.append({"stratum": i, "n": int(m.sum()),
                     "base_rate": float(ys.mean()),
                     "acted_rate": float(top.mean()),
                     "lift_pp": float(top.mean() - ys.mean()) * 100})
    if not picked:
        return {"lift_pp": 0.0, "base_rate": 0.0, "acted_rate": 0.0,
                "per_stratum": [], "consistent": False, "n": 0}
    acted, base = hits / picked, base_hits / base_n
    return {
        "lift_pp": float(acted - base) * 100,
        "base_rate": float(base),
        "acted_rate": float(acted),
        "coverage": coverage,
        "n": int(base_n),
        "per_stratum": rows,
        # Most strata pointing the same way is the point of the test. The
        # bar is 80% rather than unanimity: with ten noisy buckets even a
        # real edge loses one, and a rule that cries wolf gets ignored.
        # Against a coin (p=0.5 per stratum) 8-of-10 is roughly a 5% test.
        "consistent": bool(rows) and (
            sum(1 for r in rows if r["lift_pp"] > 0)
            >= CONSISTENT_SHARE * len(rows)),
    }


def compare(scores, outcomes, confound, coverage: float = 0.10) -> dict:
    """Signal vs the confound used directly as a score vs random.

    `beats_control` is the question that matters: if ranking by the confound
    alone does as well, the signal has not been shown to add anything.
    """
    y = np.asarray(outcomes, dtype=int)
    c = np.asarray(confound, dtype=float)
    c = np.where(np.isnan(c), np.nanmedian(c), c)
    rng = np.random.default_rng(0)
    sig = naive_lift(scores, y, coverage)
    ctl = naive_lift(c, y, coverage)
    rnd = naive_lift(rng.uniform(size=len(y)), y, coverage)
    return {"coverage": coverage, "signal_lift_pp": sig,
            "control_lift_pp": ctl, "random_lift_pp": rnd,
            "beats_control": bool(sig > ctl),
            "margin_pp": float(sig - ctl)}


def render(cmp_: dict, matched: dict, confound_name: str = "volatility",
           title: str = "Controls") -> str:
    """Markdown. The naive number never appears without the matched one."""
    lines = [f"**{title}** - is the edge the signal, or {confound_name}?", "",
             f"| ranked by | lift at {cmp_['coverage']:.0%} coverage |",
             "|---|---|",
             f"| the signal | {cmp_['signal_lift_pp']:+.1f} pp |",
             f"| {confound_name} alone (control) "
             f"| {cmp_['control_lift_pp']:+.1f} pp |",
             f"| random | {cmp_['random_lift_pp']:+.1f} pp |", ""]
    if cmp_["beats_control"]:
        lines.append(f"Signal beats the control by "
                     f"{cmp_['margin_pp']:+.1f} pp.")
    else:
        lines.append(f"> **The control wins by "
                     f"{-cmp_['margin_pp']:.1f} pp.** Ranking by "
                     f"{confound_name} alone -- which knows nothing about "
                     f"direction -- scores at least as well, so the naive "
                     f"lift is not evidence of a directional edge.")
    if matched["per_stratum"]:
        good = sum(1 for r in matched["per_stratum"] if r["lift_pp"] > 0)
        lines += ["", f"Holding {confound_name} constant "
                  f"({matched['coverage']:.0%} coverage within each of "
                  f"{len(matched['per_stratum'])} strata, n={matched['n']:,}):",
                  "", f"- base {matched['base_rate']:.1%} -> acted "
                  f"{matched['acted_rate']:.1%} "
                  f"(**{matched['lift_pp']:+.1f} pp**)",
                  f"- positive in {good}/{len(matched['per_stratum'])} strata"
                  + (" - consistent, which is what makes it credible"
                     if matched["consistent"] else
                     " - **not consistent**, so treat it as unproven")]
    lines += ["", "> A lift measured only against the base rate cannot tell "
              "a signal from a confound. Both numbers, always."]
    return "\n".join(lines) + "\n"
