"""
Rebound-specific constants for the XGBoost rebound prediction framework.
Shared utilities (paths, odds helpers) are imported from backtest_utils.
"""
from typing import List

# ── Rebound feature columns (strict order, V2: 22 features) ──
REBOUND_FEATURE_COLS: List[str] = [
    "proj_minutes",
    "reb_per_min_l5",
    "reb_per_min_l10",
    "reb_per_min_season",
    "reb_std_l10",
    "oreb_share_l5",
    # dreb_share_l5 removed V2 — perfectly inversely correlated with oreb_share_l5
    "fga_per_min_l5",
    "pts_per_min_l5",
    "tov_per_min_l5",
    "team_pace",
    "opp_pace",
    "opp_reb_allowed",
    "opp_fga_l10",
    "opp_fg_pct_l10",
    "opp_oreb_rate_l10",
    "team_fga_l5",
    "rest_days",
    "is_home",
    "b2b_flag",
    "games_played_season",
    # V2 additions
    "game_total",
    "spread_abs",
]

# Binary classifier adds DK line features (26 total)
BINARY_FEATURE_COLS: List[str] = REBOUND_FEATURE_COLS + [
    "dk_line",
    "pred_minus_line",
    "dk_implied_over_prob",
    "dk_over_price",
]

REBOUND_TARGET_COL = "actual_reb"

# ── Backtest config (tuned on 2023-24, walk-forward on 2024-25 + 2025-26) ──
REBOUND_TUNING_CUTOFF = "2024-07-01"    # Tune hyperparams on 2023-24 only
REBOUND_BACKTEST_START = "2024-10-22"   # 2024-25 season start
REBOUND_BACKTEST_END = "2026-04-12"     # 2025-26 season end
REBOUND_RETRAIN_DAYS = 30
REBOUND_EDGE_THRESHOLD = 3.0           # Minimum edge % to place a bet
REBOUND_MIN_ISOTONIC_SAMPLES = 200     # Min OOS predictions before isotonic kicks in
