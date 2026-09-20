# What the accuracy number actually measures

**20 September 2026.** A campaign to improve Vectora's headline numbers by any
means. It did not improve them. It found that a large part of the number being
improved was not what it claimed to be, which is the more useful outcome, and
the controls that establish this now run inside the weekly evaluation.

Every figure below is reproducible from the committed database.

---

## 0. The short version

Two components were tested against the same control. They gave opposite
answers, and keeping them apart is the whole point:

| component | naive lift | volatility control | verdict |
|---|---|---|---|
| **technical gauge** (rules, 788k rows) | +7.9pp | **+10.8pp** | control wins — **contaminated** |
| **ML model** (walk-forward, 155k rows) | +19.7pp | +9.5pp | signal wins by +10.2pp — **genuine** |

The machine-learned model has real directional skill. The hand-built technical
gauge, on this label, mostly does not. Everything below is the detail.

## 1. The finding

The technical gauge's bullish readings hit a +5%/10-day target **+7.9pp** more
often than the base rate across 788,370 rows of history.

Ranking stocks by **trailing 20-day volatility alone** — a number that contains
no directional information whatsoever — scores **+10.8pp** on the same rows.

| ranked by | top 10%, production label | lift |
|---|---|---|
| the gauge (the signal) | 41.0% | +7.9pp |
| trailing volatility only (control) | 43.9% | **+10.8pp** |
| random | 33.2% | +0.1pp |

The control beats the signal. **For the gauge**, the reported edge was not
mostly signal.

On the 9,284 resolved live predictions the ML model looked contaminated too
(signal +11.9pp, control +12.3pp) — but that is 28 dates inside a single
regime, scored on the model that had already drifted. Evaluated properly
walk-forward (§3a) it beats the control comfortably. The small live sample was
misleading, and it is worth saying so plainly rather than leaving the harsher
reading standing.

### Why

The label asks whether price *touched* +5% within ten days. A volatile stock
brushes +5% on noise; a quiet one never does. Any threshold expressed in fixed
percentage points rewards movement, and movement is not direction. The effect
is visible directly in the bands:

| band | touched +5% | fell below −5% | median 10d return | volatility |
|---|---|---|---|---|
| Strong Sell | 18.7% | 20.1% | 0.00% | 8.5% |
| Sell | 17.0% | 21.7% | −0.66% | 8.4% |
| Buy | 20.5% | 25.6% | −1.14% | 9.9% |
| Strong Buy | 24.1% | **28.9%** | **−1.20%** | **11.5%** |

Strong Buy has a *fatter downside tail than upside* and a *lower* median return
than Strong Sell. What rises monotonically with the rating is volatility.

## 2. What survives

Compare each stock only against others of similar volatility — ten strata, top
20% within each — and a real edge remains:

| label | volatility-matched lift | strata positive |
|---|---|---|
| production (touch +5%/10d) | **+4.6pp** | 9/10 |
| close-to-close +5%/10d | **+3.4pp** | 10/10, range +2.9 to +4.1pp |
| volatility-normalised +1σ | **+3.7pp** | — |

The close-to-close row is the credible one: an edge present in all ten
volatility deciles within a narrow band is hard to explain as an artifact,
because artifacts concentrate in the wild names.

**So the honest number is roughly +3 to +5pp, not +8 to +12pp.**

One reassurance: the touch label is not pathological. Of rows that touched
+5%, 84.3% finished the ten days up, mean finish +7.9%, and only 4.5% finished
below −5%. Touching +5% does usually mean a genuine move.

## 3a. The model, judged the same way

Same features, same walk-forward splits (last 3 folds, 155,435 test rows),
same algorithm — only the question changes:

| target | base | naive lift | control | volatility-matched | strata positive |
|---|---|---|---|---|---|
| absolute +5%/10d (production) | 34.6% | +19.7pp | +9.5pp | **+11.8pp** | 10/10 |
| volatility-normalised +1σ | 14.8% | +10.2pp | **−8.2pp** | +6.6pp | 10/10 |

Two conclusions:

1. **The ML model is not contaminated.** It beats the volatility control by
   +10.2pp and holds +11.8pp with volatility fixed, in all ten strata. It is
   doing something the gauge is not.
2. **Switching targets is not an upgrade.** The volatility-normalised target is
   *cleaner* — its control scores −8.2pp, so none of its lift can come from
   volatility — but its matched lift is lower (+6.6pp against +11.8pp). Purity
   is not the same as strength. **Keep the current target and keep the
   control beside it.**

## 3b. A cleaner target, measured on the gauge

"+1 standard deviation in 10 days" cannot be won by picking volatile stocks,
because each stock is measured against its own volatility.

| target | base | gauge lift @10% | volatility-only lift @10% |
|---|---|---|---|
| absolute +5%/10d | 19.5% | +5.0pp | **+5.1pp** (control matches) |
| direction, up at all | 41.4% | +0.7pp | −2.2pp |
| vol-normalised +1σ | 15.0% | **+3.1pp** | **−5.9pp** (control fails) |
| vol-normalised +0.5σ | 24.9% | +2.9pp | −6.5pp |

On the absolute target the control fully explains the result. On the
volatility-normalised target the control points the *wrong way*, so the lift
there cannot come from anywhere but direction. It is the better-specified
question even though its headline number is smaller.

It is also somewhat more stable across regimes — year-to-year standard
deviation of the base rate falls from 4.1pp to 3.3pp — though not immune.

## 4. Levels drift, lift does not

Across the 28 live prediction dates the base rate ranged from **4.9% to 52.0%**.
Splitting the live predictions in half by time:

| half | base rate | acted-on @10% | lift |
|---|---|---|---|
| early (before 13 Aug) | 26.9% | 40.6% | +13.7pp |
| late (from 13 Aug) | 11.8% | 23.5% | +11.7pp |

The base rate more than halved; the lift barely moved. **Report lift against
the contemporaneous base rate, never the absolute hit rate** — the latter mostly
reports what the market did.

This is also the mechanism behind the −15.3pp calibration drift found in August.

## 5. Selective prediction is genuine stock-picking

Acting only on the most confident calls could gain by picking good *days*
rather than good *stocks*. It does not:

| coverage | acted-on | day effect | stock effect |
|---|---|---|---|
| 25% | 29.7% | +1.6pp | +8.6pp |
| 10% | 31.4% | +0.7pp | +11.3pp |
| 5% | 32.5% | +0.2pp | **+13.0pp** |
| 1% | 40.2% | +0.1pp | **+20.7pp** |

Essentially all of it is stock selection. (I predicted the opposite before
running it and was wrong.)

## 6. Things that did not work

Recorded because negative results are the expensive part and are otherwise
re-attempted.

| attempt | result |
|---|---|
| Ensemble: ML probability + technical gauge | Looked like +4.8pp. Vanished entirely under causal within-date standardisation, and **lost** out-of-time (−0.9pp early, −2.4pp late). The apparent gain was lookahead in my own scaling. Rejected. |
| Cross-sectional target ("beat the median stock") | Base rate 49.0–49.7% in every regime — perfectly drift-proof. But the gauge has **no edge** on it: 45.6% at 5% coverage against a 49.3% base, i.e. worse than random. Rejected. |
| Confidence × analog hit rate | 34.5% vs 32.5% at 5% coverage, but on 23 of 28 dates rather than 28, and the intervals overlap heavily. Not conclusive. |
| Liquidity weighting | 23.0% at 5% coverage, far worse. Rejected. |
| Momentum / quality-score / ADX weighting | Neutral to slightly worse. Rejected. |
| Market-condition gate (act only in strong weeks) | Worth +10pp in hindsight, but the week is barely forecastable: best predictor correlates 0.27, breadth quintiles are non-monotonic (31.7, 26.8, 29.0, 33.9, 38.7), and the strongest predictor is *again* volatility. Rejected. |
| Volatility-decile ranking of the probability | Headline falls (32.5% → 31.5%) but the CI floor rises (24.1% → 25.2%) with all 28 dates retained. Marginal; not adopted. |

## 7. What shipped

`vectora/control.py`, wired into the weekly evaluation report, which now prints
for every target:

- the naive lift (what was reported before),
- the same lift for the confound used directly as a score,
- the volatility-matched lift with its per-stratum breakdown,
- and a loud warning when the control wins, which on current live data it does.

The naive number can no longer appear without the matched one beside it.

The parallel rule already in `vectora/selective.py` — that an acted-on rate
never travels without its coverage — came from the same principle.

## 8. Honest summary

No change made the accuracy number go up. What the campaign produced instead is
a correct account of which part of it was ever real:

- The **ML model** is genuinely skilled: **+11.8pp** with volatility held
  constant, positive in all ten strata, beating the control by +10.2pp. Its
  headline (+19.7pp naive) overstates it, but the substance is there.
- The **technical gauge** is largely a volatility detector on this label: a
  ten-line control outperforms it. Its Strong Buy band has a fatter downside
  tail than upside and a lower median return than Strong Sell.
- The **honest gauge edge** is +3 to +5pp, not +8 to +12pp.
- **Levels drift, lift does not.** Report lift against the contemporaneous base
  rate.

None of this was visible for months, because every number was only ever
compared against the base rate. The system is now built so it cannot happen
again silently — that is the deliverable, and it is worth more than the
percentage point or two the experiments were chasing.

### Open leads

- Cross-sectional features (same-day rank, sector-relative) are in the database
  but are not model inputs. Untested, and the one structural idea not yet
  eliminated.
- The cached `data/features/features.parquet` (24 Aug) has
  `breadth_above_ma50` 100% null and four liquidity features ~99% null. The
  production path recomputes, so this may be stale cache only — but it is worth
  confirming the live feature matrix does not share it.
