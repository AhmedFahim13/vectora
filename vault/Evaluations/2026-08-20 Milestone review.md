---
type: milestone-review
date: 2026-08-20
---

# The 20 August gate

On 2 August I wrote that the live track record could not be judged, because
all 332 resolved predictions came from a single prediction date — one
observation wearing 332 hats. The gate was: wait until roughly 20 August,
when 15–20 genuinely independent windows would have matured, and only then
read the calibration as evidence.

That date is today. **9 independent prediction dates, 3,009 resolved
predictions.** Fewer cohorts than I projected, because the pipeline resolves
a cohort only after the full 10-trading-day horizon and DSE trades Sunday to
Thursday. The verdict is still readable.

## Did it hold up?

| | backtest | live |
|---|---|---|
| Brier | 0.203 | **0.2050** |
| AUC | — | **0.644** |
| Skill vs always-predict-base-rate | — | **+0.045** |

**The model did not degrade out of sample.** Live Brier of 0.2050 against a
walk-forward backtest of 0.203 is the single most important result here: the
walk-forward protocol with an embargo was not fooling itself. AUC of 0.644
is real discrimination, not a coin flip.

The skill score is the sobering half. Beating a constant "always predict the
base rate" forecast by 4.5% is a genuine edge and a small one.

## Where it is wrong: overconfidence

Mean predicted probability **36.6%** against **31.2%** realized. The model is
systematically overconfident by about 5 percentage points, and it is
concentrated at the bottom of the range:

| bin | predicted | realized | gap |
|---|---|---|---|
| 0.1–0.2 | 15.4% | 3.0% | **−12.4pp** |
| 0.2–0.3 | 25.1% | 12.4% | **−12.7pp** |
| 0.3–0.4 | 34.9% | 24.7% | −10.2pp |
| 0.4–0.5 | 41.4% | 39.8% | −1.6pp |
| 0.5–0.6 | 54.3% | 52.4% | −1.9pp |

The high bins are well calibrated. The low bins are badly wrong — when this
model says 15%, the truth is nearer 3%. It does not know how to say
"unlikely". For a tool whose job is partly to warn you off, that matters.

## Probability compression

Ranked into deciles, hit rate rises cleanly from 4.7% to 46.5% across the
bottom seven, then flattens:

D1 4.7% · D2 15.6% · D3 23.3% · D4 28.6% · D5 38.2% · D6 38.2% ·
D7 46.5% · D8 37.9% · D9 36.5% · D10 42.7%

Deciles 5 through 9 all carry a mean predicted probability of ~40.7%. More
than half of all predictions (1,688 of 3,009) land in the single 0.4–0.5
bucket. **The model rarely commits**, so the top of the ranking is noise
rather than conviction. Ranking is trustworthy in the lower range and not at
the top — the opposite of what a user would assume.

## The interval, stated honestly

Per-date hit rates: 25.9 · 52.0 · 38.6 · 28.4 · 28.7 · 31.6 · 27.8 · 27.7 ·
20.3 percent.

Mean **31.2%**, 95% CI **25.2% to 37.2%**. The row-wise interval would have
been ±1.7% — **3.6x too narrow**. The 26 July cohort hit 52% and the 4 August
cohort hit 20%; averaging rows across those pretends they were independent
draws when they were two market days.

`cohort_stats()` now computes this, and both the weekly report and the
dashboard quote it. See [[2026-08-20]] for the generated evaluation.

## What this justifies changing

The gate existed to prevent tuning on one cohort. It has now passed, so
recalibration is defensible — but the target is specific:

1. **Refit the deployment calibrator against live outcomes**, not only pooled
   out-of-sample backtest predictions. The overconfidence is concentrated
   below 0.4, which pooled-OOS isotonic did not catch.
2. **Investigate the compression at ~0.41.** A model that puts half its mass
   in one decile band is not discriminating where it matters most.
3. Leave the walk-forward protocol alone. It predicted live Brier to within
   0.002 and needs no change.

## What it does not justify

Claiming the system is validated. Nine cohorts in a single *Sideways* regime
is not evidence about Bull, Bear, Panic or Recovery — every segment row in
today's report reads `regime=Sideways`. The regime taxonomy is still
completely untested live.

---

## Recalibration attempt (same day)

Acting on point 1 above: fit an isotonic correction `g` on live outcomes and
compose it after the existing calibrator, `p_final = g(calibrator(raw))`.
Composing rather than refitting keeps the backtest calibrator — thirteen
years, every regime — and corrects only the deployment bias on top. Both
maps are monotone, so this cannot reorder the book. It addresses the
calibration failure and does nothing for the compression failure.

**The guard refused it.**

Leave-one-cohort-out across all 9 dates:

| | |
|---|---|
| pooled Brier | 0.2050 → 0.2036 |
| cohorts improved | 7 of 9 |
| per-cohort delta | −0.00140 (SE 0.00458) |
| 95% CI | −0.0104 to **+0.0076** |
| t | **−0.31** vs critical −1.86 |
| worst cohort | **+0.0344** |

Pooled Brier improves, so a naive check would have installed this. The
paired test across cohorts says it is noise. The shape is worse than the
average suggests: seven ordinary days improve by about 0.007 each, and the
26 July cohort — the anomalous day that hit 52% — degrades by 0.034, roughly
five times the typical gain. The correction pulls probabilities down, so it
is most wrong exactly when the market delivers more than expected.

Small consistent gains against rare large losses is the payoff shape of
something that eventually blows up. Not installed.

I had to fix my own guard to reach this answer: `cross_validate` originally
returned `improved=True` off pooled Brier, which is the same row-pooling
error the weekly report was making this morning. The verdict is now a paired
t-test across cohorts.

The machinery ships anyway. `vectora/train/recalibrate.py` re-runs on every
`evaluate`, and installs itself the moment the evidence clears the bar
without anyone deciding to trust it. If installed, it applies only in the
regimes it was fitted under — a correction learned entirely in Sideways
would be actively harmful in a Panic, so it is skipped rather than
extrapolated.

Every attempt is recorded in `calibration_log`, refusals included.
