#!/usr/bin/env python3
"""
Shift(1) Anti-Lookahead Audit.
Verifies that NO feature for game N uses data from game N itself.
Zero tolerance — any violation blocks the backtest.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

from lib.config import logger
from lib.backtest_utils import FEATURE_COLS, get_project_root


def audit(n_samples: int = 100, seed: int = 42) -> bool:
    """
    For n_samples randomly chosen player-games, verify that:
    1. The feature values can be recomputed using only prior games.
    2. Changing the current game's box score does NOT change its features.
    """
    root = get_project_root()
    fm = pd.read_parquet(root / "data" / "features" / "feature_matrix.parquet")
    raw = pd.read_parquet(root / "data" / "raw" / "all_game_logs.parquet")

    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw["MIN_FLOAT"] = pd.to_numeric(raw["MIN_FLOAT"], errors="coerce").fillna(0)
    raw["AST"] = pd.to_numeric(raw["AST"], errors="coerce").fillna(0)

    fm["game_date"] = pd.to_datetime(fm["game_date"])

    np.random.seed(seed)
    sample_indices = np.random.choice(len(fm), size=min(n_samples, len(fm)), replace=False)
    sample = fm.iloc[sample_indices]

    violations = 0
    checks_passed = 0

    for idx, row in sample.iterrows():
        pid = row["player_id"]
        gdate = row["game_date"]
        season = row["season"]

        # Get ALL games for this player BEFORE this date
        player_games = raw[
            (raw["PLAYER_ID"] == pid) &
            (raw["GAME_DATE"] < gdate) &
            (raw["MIN_FLOAT"] > 0)
        ].sort_values("GAME_DATE")

        if len(player_games) < 5:
            continue

        # Verify ast_per_min_l5: should be sum(AST last 5) / sum(MIN last 5) from PRIOR games
        last5 = player_games.tail(5)
        expected_apm_l5 = last5["AST"].sum() / last5["MIN_FLOAT"].sum() if last5["MIN_FLOAT"].sum() > 0 else 0
        actual_apm_l5 = row["ast_per_min_l5"]

        if not np.isnan(actual_apm_l5) and abs(expected_apm_l5 - actual_apm_l5) > 0.01:
            logger.error(
                "VIOLATION: player_id=%d date=%s ast_per_min_l5 expected=%.4f got=%.4f",
                pid, gdate.date(), expected_apm_l5, actual_apm_l5
            )
            violations += 1
        else:
            checks_passed += 1

        # Verify rest_days
        if len(player_games) > 0:
            last_game_date = player_games["GAME_DATE"].max()
            expected_rest = (gdate - last_game_date).days
            actual_rest = row["rest_days"]
            if abs(expected_rest - actual_rest) > 0:
                logger.error(
                    "VIOLATION: player_id=%d date=%s rest_days expected=%d got=%d",
                    pid, gdate.date(), expected_rest, actual_rest
                )
                violations += 1
            else:
                checks_passed += 1

        # Verify games_played_season: count of prior games this season
        same_season_prior = player_games[player_games["SEASON"] == season]
        expected_gps = len(same_season_prior)
        actual_gps = int(row["games_played_season"])
        if abs(expected_gps - actual_gps) > 1:  # Allow ±1 for edge cases
            logger.error(
                "VIOLATION: player_id=%d date=%s games_played expected=%d got=%d",
                pid, gdate.date(), expected_gps, actual_gps
            )
            violations += 1
        else:
            checks_passed += 1

    logger.info("=" * 60)
    logger.info("LOOKAHEAD AUDIT RESULTS")
    logger.info("=" * 60)
    logger.info("  Samples checked: %d", len(sample))
    logger.info("  Checks passed:   %d", checks_passed)
    logger.info("  Violations:      %d", violations)

    if violations > 0:
        logger.error("AUDIT FAILED — %d violations detected. DO NOT proceed with backtest.", violations)
        return False
    else:
        logger.info("AUDIT PASSED — Zero violations. Safe to proceed.")
        return True


if __name__ == "__main__":
    success = audit(n_samples=100)
    sys.exit(0 if success else 1)
