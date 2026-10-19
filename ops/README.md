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
6. **Dashboards** — retired on 2026-10-14: the Vercel projects `edgefinder-assists` and
   `edgefinder-rebounds` were rebuilt with the publishable key but stay parked behind the
   Vercel login. Nothing to do unless you bring them back.
7. **edgefinder.cognivaultlabs.com** (Vercel project `edgefinder-pro`, the only live site) —
   done on 2026-10-15. Its source is restored in the private repo
   `Khalidinter/edgefinder-pro-live` (commit `59697a1`: publishable key, new worker token,
   `bookmakers=` list, injuries via `inj.edgefinder.workers.dev`, LIVE data kept across
   reloads). The Vercel project is linked to that repo since 2026-10-15 (the earlier
   account-configuration refusal is resolved), so every push to its `main` deploys with
   the default install command. The interim build-time patch route
   (`inj.edgefinder.workers.dev/patch.sh`) is retired and the install command is cleared.
8. **Retire the legacy JWTs** — once the worker deploy is green and this repo's `.env`
   uses `sb_publishable_` / `sb_secret_` keys: Supabase -> Settings -> API Keys ->
   Legacy tab -> Disable. The leaked service_role JWT stops working at that moment.
9. **Rotate the Odds API key** if the old one was ever pasted into a public page; the new
   one goes to the GitHub secret, this repo's `.env`, and the NBA app's settings drawer
   (it is stored in the browser only).

## Grading, closing snapshot and regrade (worker, 2026-10-15)

`Khalidinter/edgefinder-server` now grades and prices the way the review asked for:

- **Did not play = void.** A player with ESPN's did-not-play flag, an empty stat line
  or zero minutes is a push with `dnp=true` on `edges` and `bets`; players are matched
  inside their own game (team nickname in the "Away @ Home" label), an exact name beats
  a loose one and an ambiguous name stays pending. The old previous-day box-score merge
  is gone. Accented names are transliterated before matching.
- **Closing snapshot.** Every props fetch rewrites the game's rows in `closing_prices`
  (ET date + `player|stat|line`) with the raw per-book prices (`books`), the weighted
  devigged consensus (`consensus_over/under`), the average vigged implied price per side
  (`implied_over/under`), `minutes_to_tip` and the Odds API `event_id`; the scan loop
  forces one fetch inside 6 minutes of tip, so the surviving row is the true close.
- **CLV on one basis.** At resolve, `edges.clv` = vigged closing price of the same side
  (same book when it posted one, else the average across books) minus the bet's vigged
  break-even, in points; positive means you beat the close. `closing_sharp_prob` is the
  devigged consensus at close (compare with `raw_tp`); `closing_implied`,
  `closing_book_price` and `closing_mtt` say what it was measured against.
- **Tracked bets too.** The bets resolver voids DNPs the same way and fills
  `bets.clv_prob / clv_value / clv_ts` from the closing snapshot; the app no longer
  freezes its own estimate as a closing line and no longer writes closing fields to
  `edges`.
- **Fair price = market consensus (2026-10-19).** A prop is priced against the
  weighted mean of every fresh two-sided quote except the book being priced
  (`consensusFromEntries`): Pinnacle weight 2 when it posts the line, FanDuel 1,
  Caesars 0.9, bet365 0.8, DraftKings 0.7, BetMGM 0.6, BetRivers 0.5, Fanatics and
  theScore 0.4, offshore and sweepstakes books 0.2. It needs three books and 1.8 total
  weight, drops quotes older than 45 minutes, decays younger ones with a 20-minute
  time constant, and never uses one-way quotes. A book is flagged only when it
  disagrees with the consensus by 2.5 points and the raw EV is 4% or more; the old
  1.10/0.76 calibration is gone from the gate (kept as `edges.tp_cal`). Each edge
  records `anchor_type`, `anchor_n`, `anchor_weight`, `quote_age_s` and Pinnacle's
  prices; `/status` shows the live rules. Re-estimate the weights from
  `closing_prices.books` after three or four weeks (see
  ops/sharp-anchor-research-2026-10.md).
- **Rebounds only.** `PROP_MARKETS = player_rebounds` in the worker, and the app's
  PROPS button fetches `player_rebounds` for pre-game events only: one credit per
  event call instead of four.
- **Dedup from the database.** Fingerprints for recent game dates are read from
  Supabase at the start of every tick, so a fresh isolate cannot re-insert an edge.

To re-grade a date range (after a grading change, or if ESPN was down):
GitHub -> `edgefinder-server` -> Actions -> "Regrade edges and bets" -> Run workflow
with `from`, `to` (ET game dates) and `tables` (`edges,bets`). It loops over the
worker's `/regrade` endpoint one date at a time and prints per-date counts
(`changed`, `dnp`, `pending`, `byChange`). `pending` rows are games ESPN did not
report as finished or names it could not match; rerun later or fix the name.
Tests: `npm test` in the worker repo.

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
