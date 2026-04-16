"""
Shared constants and helpers for the XGBoost backtest framework.
"""
import os
from pathlib import Path
from typing import List

# ── Feature columns (strict order, V2: 19 features) ──
FEATURE_COLS: List[str] = [
    "proj_minutes",
    "ast_per_min_l5",
    "ast_per_min_l10",
    "ast_per_min_season",
    "ast_std_l10",
    "fga_per_min_l5",
    "pts_per_min_l5",
    "tov_per_min_l5",
    "team_pace",
    "opp_pace",
    "opp_ast_allowed",
    "rest_days",
    "is_home",
    "b2b_flag",
    "games_played_season",
    # V2 additions
    "opp_ast_allowed_l10",
    "game_total",
    "spread_abs",
    "min_trend_l5",
]

TARGET_COL = "actual_ast"
MIN_MINUTES_FILTER = 12.0   # Rolling L5 avg minutes threshold
MIN_GAMES_FOR_FEATURES = 5  # Minimum prior games to generate features

# ── Season date ranges (regular season approximate) ──
SEASONS_MAP = {
    "2022-23": {"start": "2022-10-18", "end": "2023-04-09"},
    "2023-24": {"start": "2023-10-24", "end": "2024-04-14"},
    "2024-25": {"start": "2024-10-22", "end": "2025-04-13"},
    "2025-26": {"start": "2025-10-21", "end": "2026-04-12"},
}

# ── Backtest config defaults ──
TUNING_CUTOFF = "2023-07-01"
BACKTEST_START = "2023-07-01"
BACKTEST_END = "2026-04-01"
RETRAIN_FREQUENCY_DAYS = 30
EDGE_THRESHOLD_PCT = 3.0
FLAT_BET_SIZE = 100.0
STANDARD_ODDS = -110  # Assumed juice both sides

# ── Hyperparameter search space ──
PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "n_estimators": [200, 400, 600, 800, 1000],
    "min_child_weight": [3, 5, 10, 20, 30],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0],
    "gamma": [0, 0.1, 0.3, 0.5],
}

# ── Paths ──
def get_project_root() -> Path:
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_dirs() -> None:
    root = get_project_root()
    for d in ["data/raw", "data/features", "data/reports", "models"]:
        (root / d).mkdir(parents=True, exist_ok=True)


# ── Odds helpers ──
def american_to_implied(odds: float) -> float:
    if odds == 0:
        return 0.0
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    return 100 / (odds + 100)


def american_to_decimal(odds: float) -> float:
    if odds == 0:
        return 1.0
    if odds < 0:
        return 1.0 + (100.0 / abs(odds))
    return 1.0 + (odds / 100.0)


def payout_at_odds(odds: float, stake: float, won: bool) -> float:
    """Net PnL for a bet at American odds."""
    if not won:
        return -stake
    dec = american_to_decimal(odds)
    return stake * (dec - 1)
