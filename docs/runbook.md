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

## Sharing the dashboard with a client

`docs/dashboard/index.html` is regenerated and committed by the eod pipeline
every trading day. It is one self-contained file — no external assets, no
server, works offline. Three ways to get it in front of someone:

| Path | Effort | Stays current? | Use when |
|---|---|---|---|
| Email/send the file | none | no (snapshot) | one-off look, or a client who wants a copy |
| claude.ai artifact | none | no (re-publish to refresh) | quick shareable link today |
| Cloudflare Pages | ~5 min setup | **yes, auto** | the real client-facing URL |

**Cloudflare Pages (free, works with this private repo):**

1. dash.cloudflare.com → Workers & Pages → Create → Pages → Connect to Git.
2. Authorize GitHub, select `AhmedFahim13/vectora`.
3. Build settings: framework preset **None**, build command **empty**,
   build output directory **`docs/dashboard`**.
4. Deploy. You get a `*.pages.dev` URL.

Every bot commit to `main` (one per trading day) triggers a redeploy, so the
client URL is never stale. To restrict it to the client only, add Cloudflare
Access (free tier) with their email — they get a one-time code to view.

GitHub Pages is NOT an option here: it requires a public repo on the free
plan, and this repo holds scraped market data that should stay private.

## Vercel deployment (live client site)

Project **vectora** → https://vectora-amber.vercel.app (public; SSO disabled
2026-07-31 so clients can open it without a Vercel account).

### Connect the repo for daily auto-deploys (one-time, ~1 minute)

1. vercel.com → project **vectora** → **Settings → Git**.
2. **Connect Git Repository** → GitHub → `AhmedFahim13/vectora`
   (authorize Vercel for the private repo if prompted).
3. **Settings → Build & Deployment**:
   - Framework preset: **Other**
   - Build command: *(leave empty)*
   - Output directory: **`docs/dashboard`**
   - Install command: *(leave empty)*
4. Save. Vercel now redeploys on every push to `main`.

Why this is the right cadence: the eod pipeline commits once per trading day
after the market closes, so the site refreshes exactly once per day, on a
real data change — event-driven, no polling, no scheduled rebuilds burning
minutes. Intraday scans do not touch `docs/dashboard`, so they do not
trigger deploys.

### Locking it down later

Free plan has no password protection. Options when the demo is over:
- Re-enable **Settings → Deployment Protection → Vercel Authentication**
  (only your Vercel account can view).
- Vercel **Pro**: password-protect the production URL.
- Custom domain behind **Cloudflare Access** (free): email-code login for
  named client addresses.
