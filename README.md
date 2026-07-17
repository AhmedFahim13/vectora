# Vectora

Zero-cost market intelligence system for the Dhaka Stock Exchange (DSE).
Runs entirely on GitHub Actions free tier + scraped public data. No paid
APIs, no servers, no LLM in the loop — deterministic ML and statistics.

**This is a research tool, not investment advice.** Predictions are
calibrated probabilities with documented uncertainty, risk blocks, and
honest evaluation — never buy/sell recommendations.

## What it does, every trading day (all automatic)

1. **15:30 Dhaka** — scrape EOD prices/news/indices, validate (0-100
   quality score), classify announcements, update the market regime,
   run the Z-category pump/footprint scan, score ~330 liquid equities
   with calibrated LightGBM models, grade matured predictions, update
   the Obsidian vault (`vault/`), email the digest.
2. **11:00-14:00 Dhaka, hourly** — intraday snapshots at DSE's chart
   publication points; volume-surge / near-circuit urgent alerts
   (cooldowns + daily cap).
3. **Friday** — walk-forward retrain of both targets; a challenger is
   promoted only if it beats the incumbent's Brier score; evaluation
   report with per-regime calibration and miss autopsy.
4. **17:30 Dhaka** — health watchdog (freshness, quality, layout
   canaries).

## Quickstart

```bash
uv sync
uv run pytest -m "not slow"          # ~200 tests
uv run python -m vectora run eod     # collect + validate (gap-fills)
uv run python -m vectora run predict # probabilities + risk + explanations
