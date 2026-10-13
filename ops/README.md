# EdgeFinder daily ops

> **Status 2026-10-13:** the assists and rebounds models, their dashboards and this
> repo's daily pipeline are retired. The only site kept active is
> edgefinder.cognivaultlabs.com (Vercel project `edgefinder-pro`, backed by the
> `edgefinder-server` Cloudflare worker). The launchd job below does not need to be
> installed; the sections are kept for reference.

Local launchd cron that runs the assists + rebounds predict & resolve pipelines
once a day at noon local time. Replaces the GitHub Actions `daily-refresh.yml`
cron (which was killed 2026-04-25 by stats.nba.com WAF rate-limiting GHA IPs).

## One-time install

```bash
mkdir -p ops/logs
chmod +x ops/run_daily.sh

# Install plist into LaunchAgents and load it
cp ops/com.edgefinder.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.edgefinder.daily.plist
```

Verify:
```bash
launchctl list | grep edgefinder
# expect: <pid>  0  com.edgefinder.daily   (or - 0 com.edgefinder.daily when not running)
```

## Run on demand

```bash
launchctl start com.edgefinder.daily
tail -f ops/logs/$(date +%Y-%m-%d).log
```

## Uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.edgefinder.daily.plist
rm ~/Library/LaunchAgents/com.edgefinder.daily.plist
```

## Why local instead of GHA

stats.nba.com fronts a Cloudflare WAF that flags GitHub Actions runner IPs.
Every cron run since 2026-04-17 was cancelled at 30→45 min timeout. The library
fetches succeed instantly from a residential IP (this Mac), so the cleanest fix
for a personal-use tool is to run the pipeline here.

If the Mac is asleep at noon, the run is skipped that day. The dashboard's
status pill turns yellow to indicate stale data; resolution catches up the next
time the Mac is awake at noon.

## Worker + NBA app: finishing the key rotation

Code for both is pushed (`Khalidinter/edgefinder-server`, `Khalidinter/edgefinder-nba-pro`,
both private). These dashboard steps remain, in this order:

1. **Supabase secret key for the worker** — Supabase -> project `jnirimzrhunjdtyvkhtt` ->
   Settings -> API Keys -> Secret keys -> create one named `edgefinder-server` and copy it
   (it is shown once).
2. **Cloudflare API token** — Cloudflare dashboard -> My Profile -> API Tokens -> Create
   Token -> template "Edit Cloudflare Workers" -> copy it. The Account ID is on the
   Workers & Pages overview page (right-hand column).
3. **GitHub Actions secrets** — github.com/Khalidinter/edgefinder-server -> Settings ->
   Secrets and variables -> Actions -> New repository secret, five times:
   - `CLOUDFLARE_API_TOKEN` (step 2)
   - `CLOUDFLARE_ACCOUNT_ID` (step 2)
   - `SUPABASE_SECRET_KEY` (step 1)
   - `WORKER_AUTH_TOKEN` — the `workerToken` value in
     `edgefinder-nba-pro/src/utils/constants.js` (64 hex characters)
   - `ODDS_API_KEY` — the the-odds-api.com key
4. **Deploy the worker** — repo -> Actions -> "Deploy worker" -> Run workflow (the run that
   fired on the first push failed only because the secrets did not exist yet). A green run
   means the new code and the three secrets are live. Check with the new token:
   ```bash
   curl -s -H "Authorization: Bearer $WORKER_AUTH_TOKEN" https://edgefinder-server.edgefinder.workers.dev/status
   ```
   The reply should contain `"scanProfile":"budget"` and, after the first game-day run,
   `oddsUsage` with `used`, `calls` and `remaining`.
5. **NBA app** — the Vercel project `edgefinder-nba-pro` was redeployed from the repo's
   `main` (new token, free schedule poll). If a later change is needed, link the project to
   the GitHub repo in Vercel -> Project -> Settings -> Git, then every push deploys.
   The site stays behind Vercel login (Deployment Protection) until you turn it off.
6. **Dashboards** — the Vercel projects `edgefinder-assists` and `edgefinder-rebounds` now
   deploy from this repo (`sites/edgefinder-assists`, `sites/edgefinder-rebounds` as the
   project root directories) and were redeployed with the publishable key on 2026-10-13.
   To redeploy later: push, then Vercel -> project -> Deployments -> Redeploy, or point the
   project at `main` once this branch is merged.
7. **MLB app** — `edgefinder.cognivaultlabs.com` (Vercel project `edgefinder-pro`) was deployed
   on 2026-05-21 from an uncommitted working copy of the `edgefinder-pro` repo and still
   embeds the legacy JWT. Before step 8, restore it with `ops/restore_vercel_deployment.py`
   (deployment `dpl_Ahcq9PTSE4S2Kn5vGMc1GhQYswuR`), swap the key for the publishable one,
   and redeploy. Skipping this means that site stops loading data at step 8.
8. **Retire the legacy JWTs** — once the worker deploy is green and this repo's `.env`
   uses `sb_publishable_` / `sb_secret_` keys: Supabase -> Settings -> API Keys ->
   Legacy tab -> Disable. The leaked service_role JWT stops working at that moment.
9. **Rotate the Odds API key** if the old one was ever pasted into a public page; the new
   one goes to the GitHub secret, this repo's `.env`, and the NBA app's settings drawer
   (it is stored in the browser only).

## Season-start checklist (2026-27 tips off 2026-10-20)

1. **Odds API key** — the pipeline calls `api.the-odds-api.com/v4` (the-odds-api.com).
   The 20K-credit plan (US$30/mo) covers it: each run costs 1 credit per upcoming
   event per market, roughly 30-90 credits/day during the season. `theoddsapi.com`
   is a different product and its key will not work here. Put the new key in `.env`
   (`ODDS_API_KEY`) and in the GitHub Actions secret of the same name, then verify:
   ```bash
   curl -sI "https://api.the-odds-api.com/v4/sports/?apiKey=$ODDS_API_KEY" | grep -i x-requests
   ```
2. **Supabase** — project `jnirimzrhunjdtyvkhtt` must be ACTIVE. `.env` needs
   `SUPABASE_KEY` = the publishable key (`sb_publishable_...`) and
   `SUPABASE_SERVICE_ROLE_KEY` = a secret key (`sb_secret_...`); both come from
   Supabase -> Settings -> API Keys. The legacy anon / service_role JWTs are being
   retired because the service_role JWT was found in public front-end bundles.
3. **Dependencies** — `pip install -r requirements.txt` (pandas is now pinned to 2.x and pyarrow is explicit).
4. **Scheduler** — `launchctl list | grep edgefinder`; if missing, follow the one-time install above.
5. **Smoke test** on a game day:
   ```bash
   python3 scripts/paper_trade.py --threshold 0.60
   python3 scripts/rebound_paper_trade.py
   ```
   Expect a `pipeline_runs` row per model and rows on the dashboards. Players need 5
   regular-season games before they get a prediction, so the first ~2 weeks of the
   season produce few or no rows — that is expected, not a failure.
6. **Retrain** once there are a few weeks of games: `python3 scripts/retrain.py`.
   It now re-fetches the in-progress season and rebuilds the training set from every
   season on disk (it previously targeted 2025-26 and could shrink the training set).
