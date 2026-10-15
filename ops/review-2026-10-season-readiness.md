# EdgeFinder review before the 2026-27 season (2026-10-15)

Scope: the Cloudflare worker (`Khalidinter/edgefinder-server`), the live app
(`Khalidinter/edgefinder-pro-live`, edgefinder.cognivaultlabs.com), the Python
assists/rebounds pipelines in this repo, and every row they wrote to Supabase last
season. Question asked: what did last season's "rebounds under only" edge really
consist of, and what should change before tip-off on 2026-10-20.

## 1. What the data says about last season

All numbers from Supabase project `jnirimzrhunjdtyvkhtt` on 2026-10-15. "profit" is
per 1 unit staked at the flagged price.

### 1.1 Worker edges (table `edges`, game dates 2026-03-16 .. 2026-06-03)

| stat / side | rows | win % ex push | avg price | profit / unit |
|---|---|---|---|---|
| rebounds Under | 15,734 | 58.8 | -57 | +0.094 |
| threes Under | 6,654 | 54.2 | -85 | -0.040 |
| points Under | 1,496 | 52.6 | -44 | +0.021 |
| points Over | 1,229 | 49.2 | -58 | -0.058 |
| rebounds Over | 788 | 43.3 | +17 | -0.117 |
| assists Over | 731 | 45.8 | +16 | -0.071 |
| assists Under | 660 | 45.6 | +14 | -0.075 |
| pra Under | 211 | 41.6 | -111 | -0.189 |

That headline table is what made rebounds-under look like the edge. Four corrections
shrink it to roughly nothing:

1. **Duplicates.** 15,759 rebounds-under rows are 1,811 player-games. `seen_count` is 1
   on every row; every 5-minute tick re-inserted the same edge. One row per
   player-game: 1,808 games, 54.4% win, +0.047/unit.
2. **DNPs graded as wins.** 720 rows (101 player-games) are "won" with 0 rebounds:
   Paul Reed Jr 132 rows, Keldon Johnson 53, Mikal Bridges 37, Brandon Ingram 36,
   Andrew Wiggins 34, Myles Turner 33, Dean Wade 31. Those players did not play; a
   sportsbook voids the bet. The worker parses an empty box-score line as 0 rebounds
   (`worker.js` ~2318-2331) and grades it (~2556-2567); only a player missing from the
   box score becomes a push. Excluding zero-rebound games, one row per player-game:

   | month | player-games | win % | profit / unit |
   |---|---|---|---|
   | 2026-03 | 726 | 52.5 | +0.032 |
   | 2026-04 | 705 | 52.6 | +0.002 |
   | 2026-05 | 100 | 48.5 | -0.098 |

3. **Closing-line value is negative.** `closing_prices` holds 27,641 devigged closing
   probabilities from 2026-04-08 (the `edges.closing_sharp_prob` column was never
   filled). Joining 786 rebounds-under edges from that date: the closing fair
   under-probability averaged 1.9 points below the bet's break-even, 6.4 points below
   the worker's flag-time "true prob", and only 26% of flagged unders beat the close.
   Every segment (line size, price, minutes) had negative CLV. The market moved against
   these bets after they were flagged, which is what a stale-quote or over-calibrated
   signal looks like, not a real edge.
4. **The wins that remained were minutes, not rebounding.** Where minutes are recorded
   (2,669 rows, almost all April): under 15 minutes played 95% win (+0.55/unit), 15-24
   minutes 69% (+0.31), 25-31 minutes 55% (+0.04), 32+ minutes 44.5% (-0.17). In April
   29.5% of flagged player-games played under 25 minutes and carried all the profit.
   Late-season rest, shutdowns and blowouts on tanking teams drove it; the NBA's
   2026-27 anti-tanking package (lottery reform passed 2026-05-28, fines for sitting
   healthy players) targets exactly that behaviour.

Other slices worth keeping in mind: lines 7.5-9.5 were the best bucket (69.5% win,
+0.25/unit raw; still +0.21 in April, flat in May) and lines 10.5+ the worst (32% win,
-0.42); "morning" flags lost (-0.04 on 1,009 rows); `es_tier` is not predictive (A+ was
the worst tier at 50.2% win, -0.07/unit on 915 rows; D was +0.10); `ev` buckets are not
monotonic.

### 1.2 Your real bets (table `bets`, 209 rows, 2026-03-10 .. 2026-04-25)

rebounds Under: 93 bets, 50-33-10, +38.52 on 330.55 staked (+11.7%), avg price -33.
Only 3 of the 50 wins were zero-rebound DNPs (+2.50), so the record is genuine, but 93
bets at that ROI is about one standard error from zero. Every other stat/side was a
handful of bets and flat to negative.

### 1.3 Python pipelines (tables `rb_paper_trades`, `am_paper_trades`, `pipeline_runs`)

- Rebounds model: ran 2026-04-14 .. 2026-06-05 only (play-in and playoffs). 825
  predictions, 39 placed bets, 38.5% hit, -794.69 PnL. It claimed an average 10.4-point
  edge (model over-prob 0.372 vs DraftKings 0.545) that did not exist. Brier score was
  never logged for rebounds.
- Assists model: 17 placed bets, small loss. Logged Brier 0.2502 raw / 0.2485
  calibrated, i.e. a coin flip (0.25).
- `am_backtest_runs` is empty; no stored backtest supports either model.

### 1.4 Bottom line

Last season produced no demonstrated, market-beating edge. The rebounds-under result
was a grading bug plus a late-season minutes effect, flagged by a signal that the
closing market disagreed with. Your own 93 bets are consistent with that story and
with ordinary variance. That is the starting point for 2026-27, and it is useful: it
says where the measurement has to be fixed before any model or market can be judged.

## 2. Worker (`edgefinder-server/src/worker.js`) findings

Line numbers are approximate to the 2026-10-15 main branch.

1. Grading: DNP parsed as 0 (2318-2331, 2556-2567); previous day's box scores merged
   into the same list when fewer than 8 games finished (2495-2502) so a back-to-back
   player can be graded on the wrong game; `normName` strips non-ASCII (2278) so
   accented names can fall to a DNP push.
2. "EV" is mostly a constant: `CALIB` multiplies the devigged FanDuel/bet365 probability
   by 1.10 for rebounds Under and 0.76 for rebounds Over (149-163) before the 5% EV gate.
   At -110 any FanDuel under-lean clears the gate; assists/points almost never do, yet
   `player_points` and `player_assists` still cost half of every credit (1310, 1329).
3. No sharp anchor: `PROP_ANCHOR_BOOKS = ["fanduel","bet365"]`; `sharpAnchored` is
   always false; Pinnacle is requested but only counts toward the 2-book minimum.
4. CLV: `closing_prices` is overwritten with the last-seen devigged anchor probability
   per (UTC date, player|stat|line), compared against the vigged break-even (2057,
   2596-2599); `closing_sharp_prob`, `closing_ts`, `resolved_by` are dropped by
   `toSbResolvePatch` (1157-1167).
5. State lives in the isolate: dedup and edge-timing persist at most every 30 minutes
   (1210-1264); a fresh isolate re-inserts and re-pushes; `computePlayerSets` only runs
   at UTC hour 10 (1747) but the crons skip 07-10, so position sets and the blacklist
   are frozen on 2025-26 names (14-189).
6. Budget: tiers 60/30/15 minutes before tip, 4 markets x 9 books = 4 credits per call,
   about 50 credits per game, daily cap 500; on 10+ game slates the latest tips lose
   their pre-tip fetches. Hours 5, 11 and 14 UTC are resolve-only and hit ESPN 12x/hour.
7. Hygiene: `edges`, `closing_prices` and `server_runs` are never pruned; the resolver
   reads only the newest 5,000 edges; `/run` can overlap a cron tick.

## 3. App (`edgefinder-pro-live/src`) findings

1. CLV falls back to the bet's own estimate: `closing = sharpProb || latestSharpProb ||
   openSharp` where `openSharp` is the calibrated tp (`EdgeFinder.jsx` ~805-812). Bases
   differ (vigged implied vs devigged anchor vs calibrated tp). The tracker's AVG CLV
   echoes the app's own EV.
2. Push handling adds phantom EV on integer lines: `evWithPush` keeps pWin at the
   devigged value and only shrinks pLose (`utils/math.js` 38-41), so a zero-edge integer
   line shows about +3 points of EV; same bias in `kellyWithPush`. Correct form is
   pWin(1-push) and (1-pWin)(1-push), with a per-line push table rather than a flat 3%.
3. Outlier removal deletes the candidate: entries beyond 2 SD are filtered out of the
   same list the EV scan iterates (`utils/engines.js` ~96-104, ~180-200), so with 6+
   books the single most mispriced quote, the one worth betting, disappears.
4. The same `CALIB` constants as the worker (`utils/players.js` 57-66) manufacture prop
   edges: at price parity with FanDuel every rebounds Under elsewhere shows about +5%.
5. Only Pinnacle of the three "sharp" books is requested (`ODDS_BOOKMAKERS`), so the
   sharp consensus is Pinnacle alone and only where it posts the exact point; soft
   quotes at other points are graded against soft consensus.
6. Steam detection cannot see line moves (matching on identical point, so `pointDiff`
   is always 0) and measures price moves in American odds, so -105 to +105 reads as 210.
7. Cost: `fetchProps` requests all four prop markets for every game up to 3 hours after
   tip; only rebounds matters.
8. Default staking is root-unit (bankroll x 2% / sqrt(dec-1)), not the quarter-Kelly the
   README describes; grades: A+ is also the default feed filter, and A+ was the worst
   tier in last season's data.

## 4. Python pipelines (`lib/`, `scripts/`) findings

1. Tonight's opponent and venue come from the previous game. `rebound_paper_trade.py`
   405-417 takes `is_home` and the opponent abbreviation from `logs.iloc[-1]` (the last
   game log); the event's `home_team`/`away_team` are carried at 345-346 but never used,
   and `lib/odds_team_map.py` is imported nowhere. Six features (is_home, opp_pace,
   opp_reb_allowed, opp_fga_l10, opp_fg_pct_l10, opp_oreb_rate_l10) describe the last
   opponent. The assists path (`paper_trade.py` 380-399) has the same bug.
2. Current-season lines never reach training. Daily DraftKings lines are written only to
   Supabase; training merges `data/lines/all_historical_lines.parquet`, which only
   `fetch_historical_lines.py` fills at about 40 credits per event (roughly 50K credits
   a season, impossible on the 20K plan). Weekly retrains add game logs but no 2026-27
   lines.
3. Season-start defaults and silent drops. Live `team_pace`/`opp_pace` default to 100 and
   `opp_reb_allowed` to 44 until the 2026-27 keys exist (419-421), which needs
   `data_pipeline.py --refresh-current` and `fetch_opp_rebounds.py`, neither called by
   `retrain.py`. The feature builder drops any row with a NaN core feature
   (`rebound_feature_engineering.py` 393-399), so the new season is excluded from
   retraining until those run.
4. Under-only is a rule, not a finding. `bet_placed = edge_under >= 8.0` (52, 785) against
   DraftKings' vigged implied probability; overs are scored but never bet; there is no
   EV or Kelly gate, the stored Kelly uses the vig-inclusive edge, stakes are a flat
   $100, and a push counts as a loss. The rule dates from the pipeline's first commit
   (2026-04-11), so the live record is two months of season tail plus playoffs.
5. Train/serve skew. Training lines are a 20:00 UTC snapshot, live lines are pulled at
   noon up to 48 hours ahead; training rolling windows cross season boundaries while
   live uses current-season logs only; `opp_*_l10` is refreshed only at retrain. No
   calibration runs in production (the isotonic and NB layers exist only in the research
   backtest). `audit_lookahead.py` checks only the assists matrix.
6. The model itself is real: XGBoost classifier, 500 trees, depth 5, 25 features,
   walk-forward retrain every 30 days (`models/rebound_binary_classifier.json`, 919 KB).
   In the committed feature set (79,563 rows over four seasons, 34,203 with DK lines)
   blanket unders return -1.9% and decay by season (-0.2%, -2.3%, -3.1%); DraftKings
   already shades unders (average under price -61 vs over -45). The naive signal bucket
   went +2.0% / +0.4% / -1.0% by season. A plausible structural tilt that is being priced
   away, and the tooling cannot tell it from noise (no bootstrap CI, no per-season
   breakdown for the 8% rule).
7. Fragility: hundreds of `PlayerGameLogs` calls a day, a bundled static player list
   that will miss 2026 rookies, GitHub Actions already blocked by the NBA WAF; assists
   still runs daily in `ops/run_daily.sh` and spends half the credit budget on a retired
   market; per-player exceptions are swallowed.

## 5. What to change, in order

### Before 2026-10-20

1. **Fix grading and re-grade last season.** DNP and minutes check in the worker
   resolver (void, not win), scope the previous-day merge by team, transliterate names,
   then null the outcomes and rerun `/reconcile`. Every "where is the edge" question
   depends on this.
2. **Capture one true closing snapshot per game.** Last worker fetch in the final 5
   minutes before tip, keyed by event id, stored as raw per-book prices; compute CLV
   devigged-vs-devigged; persist `closing_sharp_prob` and `closing_ts`. The app must stop
   falling back to its own tp and must tag `clv_source`. Zero extra credits.
3. **Scan only rebounds** (`markets=player_rebounds`) in the worker and in `fetchProps`,
   pre-game only. Cuts prop cost 4x; spend the savings on the closing snapshot and
   alternate lines.
4. **Retire the 1.10 / 0.76 constants.** Show raw and calibrated EV side by side; re-fit
   calibration only on re-graded, out-of-sample 2025-26 data, by stat, side and
   probability bucket.
5. **Fix the push math** (both EV and Kelly, per-line push table) and stop deleting
   outliers from the candidate list (exclude them from the consensus only).
6. **Worker plumbing:** write dedup/timing state through every tick, move
   `computePlayerSets` into a cron hour, refresh 2026-27 rosters and position sets,
   resolve once per window instead of 12x per hour, give each game a credit allowance
   from the slate size, prune `closing_prices` and `server_runs` after 30 days.

Python pipeline, if it keeps running: derive opponent and venue from the event
(`lib/odds_team_map.py`), append each day's DK lines to `data/lines/` and merge them in
training, run `fetch_opp_rebounds.py` and the current-season refresh inside
`retrain.py`, drop assists from `ops/run_daily.sh`, move the predict run to late
afternoon ET, de-vig before computing edge, count pushes as pushes, and add a bootstrap
CI per season for the 8% under rule before trusting it.

### First month of the season (paper, not money)

7. **Treat minutes as the product.** The only thing that paid last season was players
   playing fewer minutes than the line assumed. Use the injuries proxy already deployed
   (`inj.edgefinder.workers.dev`) to skip OUT/doubtful players, flag absent starters and
   back-to-backs, and add a simple minutes projection (last-5 minutes, starter flag,
   blowout risk from the spread) to both the worker score and the Python features.
   Expect the late-season version of this effect to shrink under the anti-tanking rules.
8. **Alternate lines.** The Odds API lists `player_rebounds_alternate` next to
   `player_rebounds` (per-event endpoint, +1 credit per call). Price your modeled number
   at the best available line instead of only the main line.
9. **Consensus anchor.** Leave-one-out weighted average of devigged probabilities across
   all two-sided books (the weight table exists), require the target to beat the best
   available price, translate Pinnacle spreads/totals across points.
10. **Decide with CLV, not win rate.** Bet only segments that show positive CLV over at
    least 300 player-games with the fixed grading; keep EdgeHealth's rolling-100 ROI
    gate. Only then pull the Python model probability into the app.

### Do not

- Do not widen to assists, points or threes; nothing in the data supports them.
- Do not use `es_tier`, A+ or the confidence labels as a filter; they are cosmetic.
- Do not size stakes off EV% until calibration is measured.
- Do not bet "morning" flags; they lost.

## 6. New since last year

- The Odds API: alternate player-prop ladders (`player_rebounds_alternate`), player props
  via `/events/{id}/odds`, a `bookmakers=` list of up to 10 books billed as one region
  (already in use), free `/events`.
- NBA 2026-27: anti-tanking lottery reform (14 to 16 lottery teams, relegation zone,
  flattened odds) and fines for resting healthy players (Jazz, Pacers in 2025-26).
  Officiating emphasis on push-offs and screens. Emirates NBA Cup semifinals at the
  higher seed's arena.
- Books: `espnbet` in the Odds API now means theScore Bet (ESPN BET was rebranded);
  it still returned prices through June 2026. Confirm it is still the Ohio book you use.

Sources: https://the-odds-api.com/sports-odds-data/betting-markets.html ,
https://www.espn.com/story/_/id/50079138 ,
https://pr.nba.com/nba-cup-semifinal-host-sites-coachs-challenge-rules
