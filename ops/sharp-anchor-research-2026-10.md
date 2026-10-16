# An effective "sharp anchor" for NBA rebounds props (research, 2026-10-16)

Question: the worker and the app price every prop against a "sharp anchor", but last
season that anchor was almost never a Vegas or Pinnacle number. What should the fair
price be built from instead, and what do other people use?

## 1. What our own data says

`closing_prices` (last season, rebounds rows, keyed player|rebounds|line) records which
book ended up as the anchor at close. The anchor fell back to whatever two-sided quote
had the highest weight when FanDuel and bet365 had no two-sided price:

| anchor at close | share of rows |
|---|---|
| Bovada | 41.0% |
| FanDuel | 22.8% |
| ESPN BET (theScore) | 16.8% |
| Pinnacle | 4.3% |
| BetMGM | 3.1% |
| Fliff, Hard Rock, BetOnline, BetRivers, DraftKings, Caesars, others | under 3% each |

So Pinnacle posted the exact rebounds line about one time in twenty-five, FanDuel was
one-sided or absent for most lines, and the single most common "sharp anchor" was
Bovada, a soft offshore book. Edges flagged against each anchor (one row per
player-game-book, DNP voids excluded):

| flag-time anchor | rows | win % | return / unit |
|---|---|---|---|
| FanDuel | 2,109 | 52.7 | -2.4% |
| Pinnacle | 154 | 60.4 | +18.2% |
| Bovada | 75 | 39.7 | -6.3% |
| BetMGM | 23 | 30.4 | -35% |

Small samples apart from FanDuel, but the direction is the one the industry describes:
Pinnacle when it exists, otherwise a single soft book is worse than nothing.

## 2. What the market and the tools say

- Pinnacle offers far fewer props than US books, and vendor coverage tables list it
  for milestone props (double-doubles, first basket) but not for rebounds or assists;
  rebounds come from DraftKings, FanDuel, BetMGM, Caesars, bet365, PrizePicks, Underdog
  and Novig
  ([SportsGameOdds NBA coverage](https://sportsgameodds.com/blog/nba-betting-markets-api),
  [OddsPapi](https://oddspapi.io/blog/?p=2520)). When Pinnacle does price a prop, that
  line is the sharpest benchmark available.
- The books most people call sharpest, because they take the highest limits, are
  Pinnacle, Betcris and Circa; among US books the standouts are FanDuel and Caesars.
  FanDuel originates its own prop numbers and runs them through price discovery at
  modest limits; DraftKings, Caesars and BetMGM take their numbers from a common source
  ([Establish The Run](https://establishtherun.com/understanding-the-current-ecosystem-of-nfl-player-props/)).
- Props went "from one of the easiest markets to beat to an efficient market" as handle
  grew; FanDuel's prop lines are efficient enough to serve as a de-vig reference, but it
  often posts one-way markets, which hides the hold
  ([BetSmart](https://betsmart.beehiiv.com/p/efficient-prop-markets)).
- The +EV tools do not rely on one book for props. OddsJam compares an outlier to "a
  sharp book (like Pinnacle) or a consensus market average"
  ([RotoWire review](https://www.rotowire.com/news/best-betting-strategy-for-the-long-term-get-a-7-day-free-trial-for-sharp-money-97444));
  Unabated sells a proprietary vig-free consensus "derived from the sharpest
  sportsbooks" ([Unabated](https://app.dealroom.co/companies/unabated_products),
  [SportsGameOdds comparison](https://sportsgameodds.com/compare/unabated)); one tool
  claims to learn which book is sharpest per market
  ([4for4 EV tool](https://www.4for4.com/sharpstack/evtool)).
- The academic version of the same idea: Kaunitz, Zhong and Kreiner used the average of
  all bookmakers' odds as the probability estimate and bet the outliers against it,
  returning about +3.5% over ten years of football odds in simulation and staying
  profitable with real money until the books limited them
  ([arXiv 1710.02824](https://arxiv.org/abs/1710.02824),
  [MIT Technology Review](https://www.technologyreview.com/2017/10/19/67760/the-secret-betting-strategy-that-beats-online-bookmakers/),
  [code](https://github.com/alannesta/BeatTheBookie/blob/master/README.md)).
- De-vig choice matters less than the inputs: multiplicative, power and Shin differ
  little on a two-way prop with 4-6% hold; power or Shin pull more margin off the
  longshot side ([OddsPapi no-vig methods](https://oddspapi.io/blog/?p=3089),
  [consensus odds](https://oddspapi.io/blog/?p=2880)).
- Exchanges: Novig has CFTC approval and ProphetX runs player-performance prediction
  markets, but liquidity is thin away from marquee games and neither appears in The Odds
  API's US bookmaker lists
  ([DefiRate](https://defirate.com/?p=13815),
  [Gambling.com ProphetX review](https://www.gambling.com/us/prediction-markets/prophetx),
  [The Odds API bookmakers](https://the-odds-api.com/sports-odds-data/bookmaker-apis.html)).
- The Odds API: `us` has betonlineag, betmgm, betrivers, betus, bovada, williamhill_us,
  draftkings, fanatics, fanduel, lowvig, mybookieag; `us2` has ballybet, betparx,
  espnbet (theScore), fliff, hardrockbet, rebet; Pinnacle sits in the EU group;
  `us_dfs` (PrizePicks, Underdog, Pick6) is "indicative only"; every 10 bookmakers named
  count as one region for credits.

## 3. Recommended anchor (fair price) for rebounds

Build the fair price per player and line from the whole market, never from one soft
book, and let the data set the weights.

1. **Weighted consensus, leave-one-out.** De-vig every two-sided quote (power method,
   as now) and take the weighted mean of the de-vigged under-probabilities, excluding the
   book being priced. Starting weights: Pinnacle 2.0, FanDuel 1.0, Caesars 0.9,
   bet365 0.8, DraftKings 0.7, BetMGM 0.6, BetRivers 0.5, Fanatics 0.4, theScore 0.4.
   Require at least three two-sided books and total weight 1.8 or more; otherwise there
   is no fair price and no edge. One-way quotes never enter the fair price (they can
   still be targets).
2. **Pinnacle is an input, not the anchor.** With weight 2.0 it dominates when present
   (4% of lines) without letting a stale or low-limit Pinnacle quote stand alone.
3. **Freshness.** Weight each quote by exp(-age / 20 min) using the market's
   `last_update`; drop quotes older than 45 minutes from the fair price. A stale quote is
   something to bet into, not evidence about the fair price.
4. **Different lines.** Today only same-line quotes are compared. Add
   `player_rebounds_alternate` (one more credit per event) and fit a per-player count
   distribution (negative binomial) to the ladder, so every book's quote can be
   translated to the target line and the consensus uses the whole market.
5. **EV against the raw consensus, no calibration constant.** Flag when the target's
   de-vigged price beats the consensus by 2.5 points or more and EV is 4% or more, and
   record `anchor_type`, `anchor_n`, `anchor_weight` and the quote ages on the edge.
6. **Learn the weights from the closing snapshot.** After three to four weeks of
   `closing_prices.books`, score each book: Brier of its de-vigged probability against
   outcomes, how often its moves lead the consensus (5-minute ticks), and the
   closing-line value of edges by anchor type. Set weights to normalised inverse Brier
   (floor 0.2), re-estimate monthly, and drop any anchor type whose edges do not beat
   the close.
7. **Later inputs.** Exchange prices (Novig, ProphetX) if a feed appears, weighted by
   liquidity; PrizePicks and Underdog lines as a line-position cross-check only.

Cost: the current bookmakers list is exactly ten books (one region), Pinnacle included.
Adding BetOnline, Bovada or Hard Rock as inputs would make it two regions and double the
per-call price; they are not worth that.

## 4. Implementation notes for the worker

- Replace the `PROP_ANCHOR_BOOKS` block in the edge loop with a `fairProbForLine(entries,
  targetBook, nowMs)` that reuses `closingSnapshotRow` (already computes the weighted
  consensus), so the flag-time fair price and the closing fair price share one basis.
- Revise `BOOK_WEIGHTS` to the starting weights above; remove `calibrateProb` from the
  EV gate (keep the raw and calibrated numbers side by side for a few weeks if useful).
- Migration: `edges.anchor_type text`, `anchor_n integer`, `anchor_weight real`,
  `quote_age_s integer`.
- Record each quote's `last_update` from the Odds API payload (the parser drops it now).
- Tests: consensus with and without Pinnacle, leave-one-out exclusion, the minimum-book
  rule, freshness decay, one-way quotes excluded.
