# Odds API credit audit — edgefinder-server worker (2026-10-11)

Scope: the Cloudflare Worker `edgefinder-server` (source lives only in `~/edgefinder-server`
on the Mac; not reviewed here) plus the two browser apps and this repo's pipeline, all of
which will share one the-odds-api.com key for the 2026-27 season. Numbers come from the
worker's own run log (`server_runs.data.apiRemaining`, the `x-requests-remaining`
header it records) in Supabase project `jnirimzrhunjdtyvkhtt`.

## What the data says

| Period | Avg games on slate | Credits per scan | Credits per scan per game | Credits per day |
|---|---|---|---|---|
| Apr 13 – May 17 | 5.7 | ~200 | ~42 | 23K – 46K |
| May 18 – Jun 3 | 2.1 | ~325 (290 – 490) | ~188 | 58K – 99K |

- Cadence: ~204 scan runs + 36 resolve runs per day, i.e. a scan roughly every 5 minutes
  for about 17 hours. Still running every ~40 minutes today in the off-season (no games,
  so no credits spent).
- The key reset to 5,000,000 on Apr 12, May 1 and Jun 1: that is the **5M plan
  (US$119/month)**. Use was ~1.4M/month in April with a near-full slate and ~1.5M in May
  with playoff slates; June ran at a ~2M/month pace with ONE game a day.
- **Credits per scan do not track the slate.** Within each period the correlation between
  credits per scan and games, props scanned, or closing-price updates is ~0. The worker
  does a fixed amount of paid work per scan, and that fixed amount roughly doubled on
  May 18 (a worker deploy; `kv_store.worker-mode` was rewritten May 24). In the Finals the
  pattern alternated: off-days ~380 per scan, game days ~175, so the cost is driven by
  how many *upcoming* events the worker re-prices all day, not by the day's games.
- Per upcoming event the worker spends ~150 – 190 credits per scan after May 18. The
  theoretical minimum for one event is markets × regions: 4 prop markets from a
  `bookmakers=` list of ≤10 books is **4 credits**.

## Why this matters before buying

If the May pattern meets a regular-season slate of 12 games, each scan costs
1,800 – 2,300 credits, so 200 scans burn 360K – 460K per day and a 5M plan is
exhausted in about 12 days. Even the April pattern (42 per game per scan) gives
~100K per day, ~3M per month, which fits the 5M plan with little margin and does
not fit the 100K plan at all.

Target after the fixes below: ≤10 credits per event per scan and ≤150 scans a day,
which is ≤18K per day and ≤540K per month on a 12-game slate. That fits the 5M plan
with 9× headroom. The 100K plan only works if scans drop to ~40 a day with one
market set.

## What to change in `~/edgefinder-server/worker.js`

1. **Schedule from the free endpoint.** `GET /v4/sports/basketball_nba/events` costs 0.
   Use it for the games list and never the paid `/odds?markets=h2h` call. The browser app
   (`~/edgefinder-nba-pro/src/api/adapter.js`) does the same paid call every 60 s while a
   tab is open: switch it too.
2. **One region.** Replace any `regions=us,us2,uk,eu` style parameter with an explicit
   `bookmakers=` list of at most 10 books (DraftKings, FanDuel, BetMGM, Caesars, bet365,
   BetRivers, Fanatics, ESPN BET, Pinnacle). Up to 10 books count as one region, so this
   alone divides the per-call cost by the number of regions used today.
3. **Only the markets the scoring reads.** Request `player_points,player_rebounds,
   player_assists,player_threes` (plus steals/blocks/PRA only if they are scored).
   Alternate-line markets double the bill; drop them unless an engine uses them.
4. **Only imminent events.** Price games tipping within the next ~24 h and skip games
   that have started (the app already filters pre-game only). Do not re-price tomorrow's
   slate all day.
5. **Closing-price / CLV refresh per event, not per edge.** One event-odds call returns
   every prop in that game; refreshing per edge multiplies the cost by the number of open
   edges. Also limit it to the last hour before tip-off.
6. **Cadence.** 15-minute scans as the baseline, 5-minute only inside the two hours before
   the first tip-off, and nothing between 02:00 and 10:00 ET.
7. **Budget guard.** Log `x-requests-remaining` on every scan (today it is logged only when
   the games list came from the API) and stop scanning for the day when the daily burn
   exceeds plan ÷ 30.

## Other consumers of the same key

- Browser app (edgefinder-nba-pro): schedule poll ~1 credit per minute per open tab,
  line-shop button up to ~25 credits per click.
- This repo's pipeline: 30 – 90 credits per day (see `lib/odds_utils.py`).

## Code review of the worker (2026-10-13)

The worker source was recovered from Cloudflare and now lives in the private repo
`Khalidinter/edgefinder-server`. What the deployed code was doing per scan:

- Games list: `GET /odds?regions=us,us2,uk,eu&markets=h2h,spreads,totals`, 12 credits,
  refreshed every 30 minutes. The bookmaker prices in that response were never read.
- Props: `GET /events/{id}/odds?regions=us,us2,uk,eu&markets=player_points,
  player_rebounds,player_assists,player_threes`, 16 credits per event, for every event the
  API lists (several days ahead), throttled by a `propsFetchedAt` map that lived only in
  the isolate's memory. `saveGameCache` never wrote it back, so every new isolate (each
  deploy, each idle restart) re-priced every listed event on its first run.
- The quota header was recorded only when the games list came from the API, so
  `server_runs.data.apiRemaining` was blank on most runs.

That code bills ~16 credits per listed event per cold start plus ~12 per 30 minutes, which
is far below the 150-190 credits per event per scan measured in May-June. The remaining
count is per key, so those deltas also include the MLB worker and app that shared the key
at the time, the browser line shop and the pipeline. The May 18 jump therefore most likely
came from the MLB side, not from an NBA worker change. Either way the NBA worker alone could
not have been checked against a budget, because nothing in it capped spend.

### What changed (edgefinder-server commit "Cut Odds API spend to fit the 20K plan")

1. Games list from the free `/events` endpoint (0 credits).
2. Props with `bookmakers=fanduel,bet365,draftkings,betmgm,williamhill_us,betrivers,fanatics,
   espnbet,pinnacle` (9 books = 1 region): 4 credits per event fetch instead of 16. Caesars is
   keyed `williamhill_us` on The Odds API, so it was added to the legal-book set; the old
   `caesars` key never matched anything.
3. `propsFetchedAt` is persisted in `kv_store` (`game-cache`) and merged on load, so the
   throttle survives restarts and concurrent isolates.
4. Scan profiles, selected with the `SCAN_PROFILE` var in `wrangler.jsonc`:

   | Profile | Lookahead | Cadence by minutes to tip | Daily cap | Stop when key below |
   |---|---|---|---|---|
   | `budget` (default) | 6 h | >240: 60 min, 60-240: 30 min, <60: 15 min | 500 | 1,500 |
   | `standard` | 24 h | >480: 60, 120-480: 30, 30-120: 10, <30: every run | 20,000 | 5,000 |

   Overrides without a redeploy: `ODDS_LOOKAHEAD_MIN`, `ODDS_DAILY_BUDGET`,
   `ODDS_MIN_REMAINING`, `ODDS_BOOKMAKERS` (dashboard vars are kept on deploy).
5. Every Odds API response's `x-requests-last` / `x-requests-remaining` is added to
   `kv_store.odds-usage` (`{date, used, calls, remaining}`), returned by `/status` as
   `oddsUsage`, written to `server_runs.data` (`oddsCalls`, `oddsUsedToday`,
   `oddsBudgetStop`) and printed once per run to Workers Logs as `[EF] scan ...`.
   Imminent games are priced first, so a cap skips the late slate, not the next tip-off.
6. Deploys run from GitHub Actions (`.github/workflows/deploy.yml`) with
   `cloudflare/wrangler-action`, which also sets the rotated `SUPABASE_KEY`, `AUTH_TOKEN`
   and `ODDS_API_KEY` secrets from repository secrets.

### Expected spend on the 20K plan

With the `budget` profile a game first seen 6 h before tip is priced about 12 times
(2 + 6 + 4), i.e. ~48 credits per game. An average slate of 7-8 games is ~350-400
credits per day, about 11K per month; a 12-game slate hits the 500 cap. Together with the
pipeline (1-3K per month) that stays under 20K. The `standard` profile is the old cadence
and needs the 100K plan.

The NBA app's schedule poll now also uses `/events` (it cost 1 credit per minute per open
tab) and a line-shop click is at most 1 credit per event instead of 2.
