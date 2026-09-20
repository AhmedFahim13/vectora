---
tags: [signal, method]
---
# The volatility confound

*Hand-written. 20 September 2026.*

The label asks whether price **touched** +5% within ten days. That rewards any
stock which *moves*, and movement is not direction. So a large part of every
edge this system has reported was volatility wearing a signal's clothes.

## The control that should have existed from day one

| ranked by | lift, top 10% |
|---|---|
| the gauge (our signal) | +7.9pp |
| trailing volatility alone | **+10.8pp** |
| random | +0.1pp |

Ranking by trailing 20-day volatility — a number containing no directional
information at all — beat the gauge.

**The ML model is the opposite case.** Walk-forward over 155,435 test rows it
scores **+19.7pp** against the control's **+9.5pp** and holds **+11.8pp** with
volatility fixed, in all ten strata. It has genuine directional skill. (On the
9,284 live rows it looked contaminated too — but that is 28 dates in one
regime, scored on the drifted model. The small sample lied.)

So: the model earns its keep; the hand-built gauge, on this label, mostly
does not.

## What is actually there

Compare each stock only against others of **similar volatility** and a real
edge survives:

- **+4.6pp** on the production touch label
- **+3.4pp** close-to-close, positive in all ten volatility deciles, range
  only +2.9 to +4.1pp

That narrow, even spread is what makes it credible. Artifacts concentrate in
the wild names; this does not. **Read +3 to +5pp as the honest edge.**

## The tell, visible in the bands

| band | touched +5% | fell below −5% | median 10d | volatility |
|---|---|---|---|---|
| Strong Sell | 18.7% | 20.1% | 0.00% | 8.5% |
| Strong Buy | 24.1% | **28.9%** | **−1.20%** | **11.5%** |

Strong Buy has a *fatter downside tail than upside* and a *lower* median
return than Strong Sell. What climbs with the rating is volatility.

This is why the client's framing — trend indication, never buy/sell — is the
correct one, and why [[backtest]] found the edge smaller than round-trip costs.

## Consequence

`vectora/control.py` now runs inside the weekly evaluation. Every lift prints
beside the same lift for the control and the volatility-matched version, and
warns loudly when the control wins — which, on today's live data, it does.

The same principle as [[selective-prediction]]: an acted-on rate never travels
without its coverage, and a lift never travels without its control.

Full write-up: `docs/accuracy-investigation.md`.
