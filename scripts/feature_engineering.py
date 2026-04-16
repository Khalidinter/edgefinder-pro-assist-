#!/usr/bin/env python3
"""
Feature matrix builder — strict anti-lookahead discipline.
Every rolling/expanding feature uses .shift(1) within the player group.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

from lib.config import logger
from lib.backtest_utils import (
    FEATURE_COLS, TARGET_COL, MIN_MINUTES_FILTER,
    MIN_GAMES_FOR_FEATURES, get_project_root, ensure_dirs,
)


def weighted_l5_minutes(series: pd.Series) -> pd.Series:
    """Weighted L5 mean with weights [0.35, 0.25, 0.20, 0.12, 0.08] (most recent first)."""
    weights = np.array([0.08, 0.12, 0.20, 0.25, 0.35])  # oldest to newest

    def _wm(x):
        if len(x) < 3:
            return np.nan
        w = weights[-len(x):]
        w = w / w.sum()
        return np.dot(x, w)

    return series.rolling(5, min_periods=3).apply(_wm, raw=True)


def ratio_of_sums(num: pd.Series, den: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Compute rolling sum(num) / rolling sum(den) to avoid low-minute bias."""
    num_sum = num.rolling(window, min_periods=min_periods).sum()
    den_sum = den.rolling(window, min_periods=min_periods).sum()
    return (num_sum / den_sum).replace([np.inf, -np.inf], np.nan)


def build_feature_matrix(
    data_dir: str = None,
    output_dir: str = None,
    min_minutes: float = MIN_MINUTES_FILTER,
    min_games: int = MIN_GAMES_FOR_FEATURES,
) -> pd.DataFrame:
    """
    Build the full feature matrix from raw parquet data.
    Anti-lookahead: ALL per-player features are .shift(1) within player group.
    """
    ensure_dirs()
    root = get_project_root()
    data_dir = Path(data_dir) if data_dir else root / "data" / "raw"
    output_dir = Path(output_dir) if output_dir else root / "data" / "features"

    # Load raw data
    logs = pd.read_parquet(data_dir / "all_game_logs.parquet")
    teams = pd.read_parquet(data_dir / "all_team_stats.parquet")

    logger.info("Raw data loaded: %d player-games, %d team-seasons", len(logs), len(teams))

    # Ensure types
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs["AST"] = pd.to_numeric(logs["AST"], errors="coerce").fillna(0).astype(float)
    logs["MIN_FLOAT"] = pd.to_numeric(logs["MIN_FLOAT"], errors="coerce").fillna(0).astype(float)
    logs["PTS"] = pd.to_numeric(logs["PTS"], errors="coerce").fillna(0).astype(float)
    logs["FGA"] = pd.to_numeric(logs["FGA"], errors="coerce").fillna(0).astype(float)
    logs["TOV"] = pd.to_numeric(logs["TOV"], errors="coerce").fillna(0).astype(float)

    # Filter out DNPs (0 minutes)
    logs = logs[logs["MIN_FLOAT"] > 0].copy()

    # Sort by player + date (critical for rolling)
    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    logger.info("After filtering: %d player-games with MIN > 0", len(logs))

    # ── GROUP BY PLAYER for rolling features ──
    grp = logs.groupby("PLAYER_ID")

    # All per-player features: SHIFT(1) first, then rolling
    # This ensures game N's features use only games 1..N-1

    shifted_ast = grp["AST"].shift(1)
    shifted_min = grp["MIN_FLOAT"].shift(1)
    shifted_pts = grp["PTS"].shift(1)
    shifted_fga = grp["FGA"].shift(1)
    shifted_tov = grp["TOV"].shift(1)

    # Feature 1: proj_minutes — weighted L5 of shifted minutes
    logs["proj_minutes"] = grp["MIN_FLOAT"].transform(
        lambda x: weighted_l5_minutes(x.shift(1))
    )

    # Features 2-3: ast_per_min L5 and L10 — ratio of shifted sums
    logs["ast_per_min_l5"] = grp.apply(
        lambda g: ratio_of_sums(g["AST"].shift(1), g["MIN_FLOAT"].shift(1), 5, 3),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    logs["ast_per_min_l10"] = grp.apply(
        lambda g: ratio_of_sums(g["AST"].shift(1), g["MIN_FLOAT"].shift(1), 10, 5),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Feature 4: ast_per_min_season — expanding ratio within (player, season)
    def expanding_rate(g):
        s_ast = g["AST"].shift(1).expanding(min_periods=3).sum()
        s_min = g["MIN_FLOAT"].shift(1).expanding(min_periods=3).sum()
        return (s_ast / s_min).replace([np.inf, -np.inf], np.nan)

    logs["ast_per_min_season"] = logs.groupby(["PLAYER_ID", "SEASON"]).apply(
        expanding_rate, include_groups=False,
    ).reset_index(level=[0, 1], drop=True)

    # Feature 5: ast_std_l10 — std dev of shifted assists
    logs["ast_std_l10"] = grp["AST"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).std()
    )

    # Features 6-8: fga/pts/tov per min L5
    logs["fga_per_min_l5"] = grp.apply(
        lambda g: ratio_of_sums(g["FGA"].shift(1), g["MIN_FLOAT"].shift(1), 5, 3),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    logs["pts_per_min_l5"] = grp.apply(
        lambda g: ratio_of_sums(g["PTS"].shift(1), g["MIN_FLOAT"].shift(1), 5, 3),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    logs["tov_per_min_l5"] = grp.apply(
        lambda g: ratio_of_sums(g["TOV"].shift(1), g["MIN_FLOAT"].shift(1), 5, 3),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Features 9-11: team context (season-level, joined)
    team_pace_map = teams.set_index(["team_abbr", "season"])["pace"].to_dict()
    team_opp_ast_map = teams.set_index(["team_abbr", "season"])["opp_ast_allowed"].to_dict()

    logs["team_pace"] = logs.apply(
        lambda r: team_pace_map.get((r["TEAM_ABBREVIATION"], r["SEASON"]), 100.0), axis=1
    )
    logs["opp_pace"] = logs.apply(
        lambda r: team_pace_map.get((r["OPP_TEAM_ABBR"], r["SEASON"]), 100.0), axis=1
    )
    logs["opp_ast_allowed"] = logs.apply(
        lambda r: team_opp_ast_map.get((r["OPP_TEAM_ABBR"], r["SEASON"]), 25.0), axis=1
    )

    # ── V2 Feature 16: opp_ast_allowed_l10 — rolling opponent AST allowed ──
    # "AST allowed by team X" = assists scored BY X's opponent in each game.
    # Step 1: aggregate AST per team per game
    # Step 2: self-join to find what opponents scored against each team
    # Step 3: rolling L10 with shift(1) per team
    logger.info("Computing opp_ast_allowed_l10 from game logs...")
    team_game_ast = logs.groupby(
        ["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]
    ).agg(T_AST=("AST", "sum")).reset_index()

    # Self-join on GAME_ID to pair each team with its opponent's AST
    ast_allowed = team_game_ast.merge(
        team_game_ast[["GAME_ID", "TEAM_ABBREVIATION", "T_AST"]],
        on="GAME_ID", suffixes=("", "_opp"),
    )
    ast_allowed = ast_allowed[
        ast_allowed["TEAM_ABBREVIATION"] != ast_allowed["TEAM_ABBREVIATION_opp"]
    ].copy()
    # T_AST_opp = assists scored by the other team = assists ALLOWED by this team
    ast_allowed = ast_allowed.rename(columns={"T_AST_opp": "AST_ALLOWED"})
    ast_allowed = ast_allowed.sort_values(
        ["TEAM_ABBREVIATION", "GAME_DATE"]
    ).reset_index(drop=True)

    ast_allowed["_ast_allowed_l10"] = ast_allowed.groupby(
        "TEAM_ABBREVIATION"
    )["AST_ALLOWED"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean()
    )

    opp_ast_l10_lookup = ast_allowed.drop_duplicates(
        subset=["TEAM_ABBREVIATION", "GAME_DATE"]
    ).set_index(
        ["TEAM_ABBREVIATION", "GAME_DATE"]
    )["_ast_allowed_l10"].to_dict()

    # Lookup: for each player row, get opponent team's AST allowed L10
    logs["opp_ast_allowed_l10"] = logs.apply(
        lambda r: opp_ast_l10_lookup.get(
            (r["OPP_TEAM_ABBR"], r["GAME_DATE"]), np.nan
        ), axis=1
    )
    logger.info("  opp_ast_allowed_l10 computed.")

    # ── V2 Feature 19: min_trend_l5 — slope of minutes over last 5 games ──
    def minutes_slope_l5(series):
        """Rolling polyfit slope of minutes over L5 (shifted)."""
        def _slope(x):
            if len(x) < 3:
                return np.nan
            return np.polyfit(range(len(x)), x, 1)[0]
        return series.rolling(5, min_periods=3).apply(_slope, raw=True)

    logs["min_trend_l5"] = grp["MIN_FLOAT"].transform(
        lambda x: minutes_slope_l5(x.shift(1))
    )

    # Feature 12: rest_days
    logs["rest_days"] = grp["GAME_DATE"].diff().dt.days.fillna(7).astype(int)

    # Feature 13: is_home (already computed in data_pipeline)
    if "IS_HOME" in logs.columns:
        logs["is_home"] = logs["IS_HOME"].astype(int)
    else:
        logs["is_home"] = (~logs["MATCHUP"].astype(str).str.contains("@")).astype(int)

    # Feature 14: b2b_flag
    logs["b2b_flag"] = (logs["rest_days"] <= 1).astype(int)

    # Feature 15: games_played_season — cumcount BEFORE current game
    logs["games_played_season"] = logs.groupby(["PLAYER_ID", "SEASON"]).cumcount()

    # Target column
    logs["actual_ast"] = logs["AST"].astype(int)
    logs["actual_min"] = logs["MIN_FLOAT"]

    # ── V2 Features 17-18: game_total and spread_abs from historical lines ──
    lines_dir = root / "data" / "lines"
    all_lines_path = lines_dir / "all_historical_lines.parquet"
    if all_lines_path.exists():
        logger.info("Merging game-level Vegas data (game_total, spread_abs)...")
        all_lines = pd.read_parquet(all_lines_path)

        # Build event_id → game mapping from player prop lines
        # Each player prop line has event_id + game_date, and we know TEAM_ABBREVIATION
        # from the feature matrix. Bridge via player name match.
        prop_lines = all_lines[all_lines["market"].isin(["player_assists", "player_rebounds"])].copy()
        prop_lines["game_date"] = pd.to_datetime(prop_lines["game_date"])
        event_games = prop_lines.drop_duplicates(subset=["event_id"])[
            ["event_id", "game_date", "home_team", "away_team"]
        ].copy()

        # game_total from totals market
        totals = all_lines[all_lines["market"] == "totals"].copy()
        if not totals.empty:
            totals["game_date"] = pd.to_datetime(totals["game_date"])
            totals_dedup = totals.drop_duplicates(
                subset=["event_id"], keep="first"
            )[["event_id", "game_date", "line"]].rename(columns={"line": "game_total"})

            # Merge totals to event_games by event_id
            event_games = event_games.merge(
                totals_dedup[["event_id", "game_total"]], on="event_id", how="left"
            )
        else:
            event_games["game_total"] = np.nan
            logger.warning("  No totals market data — game_total will be NaN")

        # spread_abs from spreads market
        spreads = all_lines[all_lines["market"] == "spreads"].copy()
        if not spreads.empty:
            spreads["game_date"] = pd.to_datetime(spreads["game_date"])
            spreads["spread_abs"] = spreads["line"].abs()
            spreads_dedup = spreads.drop_duplicates(
                subset=["event_id"], keep="first"
            )[["event_id", "spread_abs"]]

            event_games = event_games.merge(
                spreads_dedup, on="event_id", how="left"
            )
        else:
            event_games["spread_abs"] = np.nan
            logger.warning("  No spreads market data — spread_abs will be NaN")

        # Now match event_games to feature matrix rows.
        # Use ODDS_TO_NBA mapping to bridge team names.
        from lib.odds_team_map import ODDS_TO_NBA_ABBR
        event_games["home_abbr"] = event_games["home_team"].map(ODDS_TO_NBA_ABBR)
        event_games["away_abbr"] = event_games["away_team"].map(ODDS_TO_NBA_ABBR)

        # Build lookup: (game_date, team_abbr) → (game_total, spread_abs)
        vegas_lookup = {}
        for _, row in event_games.iterrows():
            gd = row["game_date"]
            gt = row.get("game_total", np.nan)
            sa = row.get("spread_abs", np.nan)
            if pd.notna(row.get("home_abbr")):
                vegas_lookup[(gd, row["home_abbr"])] = (gt, sa)
            if pd.notna(row.get("away_abbr")):
                vegas_lookup[(gd, row["away_abbr"])] = (gt, sa)

        def _lookup_vegas(r):
            v = vegas_lookup.get((r["GAME_DATE"], r["TEAM_ABBREVIATION"]))
            if v:
                return pd.Series({"game_total": v[0], "spread_abs": v[1]})
            return pd.Series({"game_total": np.nan, "spread_abs": np.nan})

        vegas_df = logs.apply(_lookup_vegas, axis=1)
        logs["game_total"] = vegas_df["game_total"]
        logs["spread_abs"] = vegas_df["spread_abs"]

        gt_filled = logs["game_total"].notna().sum()
        sp_filled = logs["spread_abs"].notna().sum()
        logger.info("  game_total matched: %d / %d (%.1f%%)", gt_filled, len(logs), gt_filled / len(logs) * 100)
        logger.info("  spread_abs matched: %d / %d (%.1f%%)", sp_filled, len(logs), sp_filled / len(logs) * 100)
    else:
        logs["game_total"] = np.nan
        logs["spread_abs"] = np.nan
        logger.warning("No historical lines file found — game_total/spread_abs will be NaN")

    # ── Filter ──
    # Drop rows with insufficient history
    logs = logs[logs["games_played_season"] >= min_games].copy()

    # Drop rows with NaN in core features (first ~5 games per player per season)
    # Allow Vegas features to be NaN (populated after data fetch)
    optional_features = ("game_total", "spread_abs")
    core_features = [c for c in FEATURE_COLS if c not in optional_features]
    feature_mask = logs[core_features].notna().all(axis=1)
    before_drop = len(logs)
    logs = logs[feature_mask].copy()
    logger.info("Dropped %d rows with NaN features (%d remaining)", before_drop - len(logs), len(logs))

    # Minutes filter: only players averaging ≥12 min in their L5 window
    logs = logs[logs["proj_minutes"] >= min_minutes].copy()
    logger.info("After min-minutes filter (≥%.0f): %d rows", min_minutes, len(logs))

    # ── Select output columns ──
    output_cols = [
        "PLAYER_NAME", "PLAYER_ID", "SEASON", "GAME_DATE",
        "TEAM_ABBREVIATION", "OPP_TEAM_ABBR", "GAME_ID",
    ] + FEATURE_COLS + [TARGET_COL, "actual_min"]

    result = logs[output_cols].copy()
    result = result.rename(columns={
        "PLAYER_NAME": "player_name",
        "PLAYER_ID": "player_id",
        "SEASON": "season",
        "GAME_DATE": "game_date",
        "TEAM_ABBREVIATION": "team_abbr",
        "OPP_TEAM_ABBR": "opp_team_abbr",
        "GAME_ID": "game_id",
    })

    # Save
    output_path = output_dir / "feature_matrix.parquet"
    result.to_parquet(output_path, index=False)
    logger.info("Feature matrix saved: %s (%d rows, %d features)",
                output_path, len(result), len(FEATURE_COLS))

    # Summary stats
    logger.info("Feature matrix summary:")
    logger.info("  Seasons: %s", sorted(result["season"].unique()))
    logger.info("  Players: %d unique", result["player_id"].nunique())
    logger.info("  Date range: %s to %s", result["game_date"].min(), result["game_date"].max())
    logger.info("  Target (AST) mean=%.2f, std=%.2f",
                result[TARGET_COL].mean(), result[TARGET_COL].std())

    return result


if __name__ == "__main__":
    build_feature_matrix()
