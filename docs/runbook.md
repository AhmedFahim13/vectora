# Vectora runbook

## A workflow went red

| Workflow | Step | Likely cause | Action |
|---|---|---|---|
| eod-pipeline | Run EOD pipeline | unlisted holiday (quality 0, "no rows") | add date to `data/reference/holidays.csv`, push; next run gap-fills |
| eod-pipeline | Run EOD pipeline | dsebd.org layout change (canary also red) | re-record fixtures (`uv run python tools/record_fixtures.py`), fix parser against them, tests are the contract |
| eod-pipeline | Predict | model artifact/path issue | check `model_registry.artifact_dir` is repo-relative; `models/` committed |
| train | Train | challenger lost (`"promoted": false`) | not an error — the guard working; nothing to do |
| health | Health check | see emailed [HEALTH] list | freshness → check eod-pipeline run; canary → layout change path above |
| intraday-scan | Intraday scan | outside trading hours/day | harmless skip, will show green |

## Routine operations

- **Add a holiday:** append `YYYY-MM-DD,description` to
  `data/reference/holidays.csv`, commit, push.
- **Rotate the Gmail app password:** revoke old at
  myaccount.google.com/apppasswords, then
  `gh secret set GMAIL_APP_PASSWORD --repo AhmedFahim13/vectora`.
  Missing/dead secret degrades safely: digests land in `reports/`.
- **Roll back a model:** `UPDATE model_registry SET active=false WHERE
  model_id='bad'; UPDATE model_registry SET active=true WHERE
  model_id='good';` via a `uv run python -c` one-liner, commit the DB.
- **DB merge conflict:** never hand-merge. `.gitattributes` sets
  `merge=ours` (local wins); run `uv run python -m vectora run eod`
  afterward — gap-fill re-ingests whatever the other side had.
  Requires once per clone: `git config merge.ours.driver true`.
- **Enable g10_h30 signals:** only when live evaluation shows the
  deployment tail holds on a HOLDOUT (see memory note about the
  tautological in-sample table); then add `"g10_h30": 0.60` to
  `SIGNAL_THRESHOLDS`.

## Known quirks (do not "fix")

- dsebd.org serves a broken TLS chain → `verify=False` in PoliteSession
  (documented, public data).
- `models/**/*.txt -text` in `.gitattributes` is load-bearing: CRLF
  corrupts LightGBM dumps on Windows.
- News archive retention starts 2024-07; older `old_news` queries return
  empty pages (real, not a bug).
- Liquidity features are null through the backfill era (no traded-value
  data before 2026-07); they activate as live history accumulates.
- Manual `run eod` during market hours (10:00-14:30 Dhaka) can mark a
  live day as no-trade; the CI schedule avoids this by design.
