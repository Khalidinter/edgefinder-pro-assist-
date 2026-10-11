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

## Not done here

The worker source is not in any repository, so the exact lines behind the May 18 jump
could not be identified. Push `~/edgefinder-server` to a private repo (exclude `.env`,
`.wrangler/`, `node_modules/`) and the call-by-call review can be finished.
