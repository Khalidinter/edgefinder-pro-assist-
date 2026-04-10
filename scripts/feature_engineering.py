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

    # ── Filter ──
    # Drop rows with insufficient history
    logs = logs[logs["games_played_season"] >= min_games].copy()

    # Drop rows with NaN features (first ~5 games per player per season)
    feature_mask = logs[FEATURE_COLS].notna().all(axis=1)
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
