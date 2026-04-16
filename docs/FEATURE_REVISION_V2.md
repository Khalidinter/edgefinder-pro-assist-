# Feature Revision V2 — Approved Changes

## Source
User audit of V1 feature sets. All changes approved. Implement in next session.

## Assists Regression: 15 → 19 features

### Keep (15):
1. proj_minutes
2. ast_per_min_l5
3. ast_per_min_l10
4. ast_per_min_season
5. ast_std_l10
6. fga_per_min_l5
7. pts_per_min_l5 (test dropping — weakly related to assists)
8. tov_per_min_l5
9. team_pace
10. opp_pace
11. opp_ast_allowed (KEEP season-level alongside new L10)
12. rest_days
13. is_home
14. b2b_flag
15. games_played_season (consider replacing with early_season_flag)

### Add (4):
16. opp_ast_allowed_l10 — rolling opponent assists allowed L10 (replaces season-level as primary)
17. game_total — Vegas over/under for this specific game (from Odds API)
18. spread_abs — absolute value of game spread (blowout risk indicator)
19. min_trend_l5 — slope of minutes over last 5 games (role trajectory)

### Data sources for new features:
- opp_ast_allowed_l10: Derivable from existing game logs (opponent team's AST allowed per game, rolling L10)
- game_total: Available in Odds API events data (already fetched — check if stored)
- spread_abs: Available in Odds API events data (h2h_spread market)
- min_trend_l5: Derivable from existing game logs (linear regression slope of MIN over L5)

## Assists Binary Classifier: 18 → 23 features

All 19 regression features PLUS:
20. line_value — DK assist line
21. pred_minus_line — XGBoost REGRESSOR prediction minus line (NOT crude linear estimate)
22. dk_implied_over_prob — implied probability from DK over price
23. dk_over_price — raw DK American odds

### Critical fix: pred_minus_line must use the actual XGBoost regressor output, not ast_per_min_season × proj_minutes.

## Rebounds Regression: 21 → 22 features

### Remove (1):
- dreb_share_l5 (redundant — perfectly inversely correlated with oreb_share_l5)

### Keep (20):
1. proj_minutes
2. reb_per_min_l5
3. reb_per_min_l10
4. reb_per_min_season
5. reb_std_l10
6. oreb_share_l5
7. fga_per_min_l5
8. pts_per_min_l5 (consider dropping — minimal rebounding relevance)
9. tov_per_min_l5
10. team_pace
11. opp_pace
12. opp_reb_allowed
13. opp_fga_l10
14. opp_fg_pct_l10
15. opp_oreb_rate_l10
16. team_fga_l5
17. rest_days
18. is_home
19. b2b_flag
20. games_played_season

### Add (2):
21. game_total — Vegas over/under
22. spread_abs — blowout risk

## Rebounds Binary Classifier: 25 → 26 features

All 22 regression features PLUS existing 4 line features.

## Implementation Priority

1. **game_total** — highest leverage, captures pace/possessions for THIS game
2. **opp_ast_allowed_l10** — #1 feature in binary classifier, making it rolling is obvious upgrade
3. **min_trend_l5** — captures role trajectory, exploits slow line adjustment
4. **spread_abs** — blowout risk affects minutes which affects everything
5. **dk_implied_over_prob + dk_over_price** for assists binary (already in rebounds)
6. Fix pred_minus_line to use actual XGBoost prediction
7. Drop dreb_share_l5 from rebounds
8. Test dropping pts_per_min_l5 from both models

## Data Requirements

### game_total and spread_abs:
Need to fetch h2h and totals markets alongside player props in historical line fetcher.
Current fetch_historical_lines.py only fetches player_assists + player_rebounds.
Need to add: markets = "player_assists,player_rebounds,totals,spreads"
Cost impact: +20 quota per event (2 more markets × 10 each)
Total: ~40 quota per event instead of 20
With 5M quota remaining: completely unconstrained

### opp_ast_allowed_l10:
Derivable from existing all_game_logs.parquet — aggregate team AST at game level, 
compute per-opponent rolling L10. No new API calls needed.

### min_trend_l5:
Derivable from existing game logs. numpy.polyfit(range(5), last_5_minutes, 1)[0]
No new API calls needed.
