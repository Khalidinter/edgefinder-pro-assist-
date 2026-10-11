# EdgeFinder daily ops

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
