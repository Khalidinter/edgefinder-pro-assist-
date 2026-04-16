#!/usr/bin/env python3
"""
Rebound Feature Matrix Builder — strict anti-lookahead discipline.
Every rolling/expanding feature uses .shift(1) within the player group.

Mirrors the assists feature_engineering.py pattern but with rebound-specific
features: OREB/DREB share, opponent rebounding opportunity, team FGA volume.

Usage:
    python scripts/rebound_feature_engineering.py [--skip-audit]
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unicodedata

import numpy as np
import pandas as pd
from pathlib import Path

from lib.config import logger
from lib.backtest_utils import (
    MIN_MINUTES_FILTER, MIN_GAMES_FOR_FEATURES,
    get_project_root, ensure_dirs,
)

# ── Rebound-specific constants ──

REBOUND_FEATURE_COLS = [
    # Core rebound features
    "proj_minutes",
    "reb_per_min_l5",
    "reb_per_min_l10",
    "reb_per_min_season",
    "reb_std_l10",
    "oreb_share_l5",
    # dreb_share_l5 removed V2 — perfectly inversely correlated with oreb_share_l5
    # Scoring/usage context
    "fga_per_min_l5",
    "pts_per_min_l5",
    "tov_per_min_l5",
    # Team context
    "team_pace",
    "opp_pace",
    # Rebound opportunity features
    "opp_reb_allowed",
    "opp_fga_l10",
    "opp_fg_pct_l10",
    "opp_oreb_rate_l10",
    "team_fga_l5",
    # Game context
    "rest_days",
    "is_home",
    "b2b_flag",
    "games_played_season",
    # V2 additions — Vegas game-level context
    "game_total",
    "spread_abs",
]

# ── DK-to-NBA player name alias dictionary ──
# Maps DK (Odds API) lowercased names to NBA API lowercased names
# Covers: accented chars, Jr/Sr suffixes, numeric suffixes, initials, name variants

DK_TO_NBA_NAME_ALIAS = {
    # Accented characters (ASCII -> Unicode)
    "nikola jokic": "nikola jokić",
    "nikola vucevic": "nikola vučević",
    "luka doncic": "luka dončić",
    "jusuf nurkic": "jusuf nurkić",
    "jonas valanciunas": "jonas valančiūnas",
    "kristaps porzingis": "kristaps porziņģis",
    "dennis schroder": "dennis schröder",
    "moussa diabate": "moussa diabaté",
    "bogdan bogdanovic": "bogdan bogdanović",
    "egor demin": "egor dëmin",
    "nikola jovic": "nikola jović",
    "vit krejci": "vít krejčí",
    "tidjane salaun": "tidjane salaün",
    "hugo gonzalez": "hugo gonzález",
    "kasparas jakucionis": "kasparas jakučionis",
    "yanic konan niederhauser": "yanic konan niederhäuser",
    # Jr/Sr suffix: DK missing period
    "wendell carter jr": "wendell carter jr.",
    "jabari smith jr": "jabari smith jr.",
    "michael porter jr": "michael porter jr.",
    "jaime jaquez jr": "jaime jaquez jr.",
    "kelly oubre jr": "kelly oubre jr.",
    "jaren jackson jr": "jaren jackson jr.",
    "tim hardaway jr": "tim hardaway jr.",
    "gary trent jr": "gary trent jr.",
    "larry nance jr": "larry nance jr.",
    "scotty pippen jr": "scotty pippen jr.",
    "craig porter jr": "craig porter jr.",
    "troy brown jr": "troy brown jr.",
    "andre jackson jr": "andre jackson jr.",
    # Jr/Sr suffix: DK adds or drops Jr/Sr
    "derrick jones": "derrick jones jr.",
    "paul reed jr": "paul reed",
    "bruce brown jr": "bruce brown",
    "dennis smith": "dennis smith jr.",
    "marcus morris": "marcus morris sr.",
    "reggie bullock": "reggie bullock jr.",
    # Numeric suffixes (II/III): DK drops or adds them
    "jimmy butler": "jimmy butler iii",
    "isaiah stewart ii": "isaiah stewart",
    "ron holland": "ronald holland ii",
    "robert williams": "robert williams iii",
    "marvin bagley": "marvin bagley iii",
    "kevin knox": "kevin knox ii",
    "trey jemison": "trey jemison iii",
    # Period/dot in initials (C.J. vs CJ)
    "c.j. mccollum": "cj mccollum",
    "r.j. barrett": "rj barrett",
    "a.j. green": "aj green",
    "g.g. jackson": "gg jackson",
    # Shortened/different first names
    "nicolas claxton": "nic claxton",
    "herb jones": "herbert jones",
    "cameron thomas": "cam thomas",
    "moe wagner": "moritz wagner",
    "mohamed bamba": "mo bamba",
    "gregory jackson": "gg jackson",
    "vincent williams jr": "vince williams jr.",
    # Completely different name
    "carlton carrington": "bub carrington",
}

REBOUND_TARGET_COL = "actual_reb"


# ── Helpers ──

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


def american_to_implied_prob(odds):
    """Convert American odds to implied probability."""
    if pd.isna(odds):
        return np.nan
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


# ── Main builder ──

def build_rebound_feature_matrix(
    data_dir: str = None,
    output_dir: str = None,
    min_minutes: float = MIN_MINUTES_FILTER,
    min_games: int = MIN_GAMES_FOR_FEATURES,
    run_audit: bool = True,
) -> pd.DataFrame:
    """
    Build the full rebound feature matrix from raw parquet data.
    Anti-lookahead: ALL per-player features are .shift(1) within player group.
    """
    ensure_dirs()
    root = get_project_root()
    data_dir = Path(data_dir) if data_dir else root / "data" / "raw"
    output_dir = Path(output_dir) if output_dir else root / "data" / "features"
    lines_dir = root / "data" / "lines"

    # ── Load raw data ──
    logs = pd.read_parquet(data_dir / "all_game_logs.parquet")
    teams = pd.read_parquet(data_dir / "all_team_stats.parquet")

    logger.info("Raw data loaded: %d player-games, %d team-seasons", len(logs), len(teams))

    # Ensure types
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    for col in ["MIN_FLOAT", "REB", "OREB", "DREB", "AST", "PTS", "FGA", "FGM", "TOV"]:
        if col in logs.columns:
            logs[col] = pd.to_numeric(logs[col], errors="coerce").fillna(0).astype(float)

    has_oreb = "OREB" in logs.columns and (logs["OREB"] > 0).any()
    has_dreb = "DREB" in logs.columns and (logs["DREB"] > 0).any()
    if not has_oreb:
        logger.warning("OREB data unavailable — oreb_share_l5 will be NaN")

    # Filter out DNPs (0 minutes)
    logs = logs[logs["MIN_FLOAT"] > 0].copy()

    # Sort by player + date (critical for rolling)
    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    logger.info("After filtering: %d player-games with MIN > 0", len(logs))

    # ═══════════════════════════════════════════════════════════════════
    # PLAYER-LEVEL ROLLING FEATURES (all shift(1) within player group)
    # ═══════════════════════════════════════════════════════════════════
    grp = logs.groupby("PLAYER_ID")

    # Feature 1: proj_minutes — weighted L5 of shifted minutes
    logs["proj_minutes"] = grp["MIN_FLOAT"].transform(
        lambda x: weighted_l5_minutes(x.shift(1))
    )

    # Features 2-3: reb_per_min L5 and L10 — ratio of shifted sums
    logs["reb_per_min_l5"] = grp.apply(
        lambda g: ratio_of_sums(g["REB"].shift(1), g["MIN_FLOAT"].shift(1), 5, 3),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    logs["reb_per_min_l10"] = grp.apply(
        lambda g: ratio_of_sums(g["REB"].shift(1), g["MIN_FLOAT"].shift(1), 10, 5),
        include_groups=False,
    ).reset_index(level=0, drop=True)

    # Feature 4: reb_per_min_season — expanding ratio within (player, season)
    def expanding_reb_rate(g):
        s_reb = g["REB"].shift(1).expanding(min_periods=3).sum()
        s_min = g["MIN_FLOAT"].shift(1).expanding(min_periods=3).sum()
        return (s_reb / s_min).replace([np.inf, -np.inf], np.nan)

    logs["reb_per_min_season"] = logs.groupby(["PLAYER_ID", "SEASON"]).apply(
        expanding_reb_rate, include_groups=False,
    ).reset_index(level=[0, 1], drop=True)

    # Feature 5: reb_std_l10 — volatility of rebounds
    logs["reb_std_l10"] = grp["REB"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).std()
    )

    # Feature 6: OREB share L5 (dreb_share_l5 removed V2 — perfectly correlated)
    if has_oreb:
        logs["oreb_share_l5"] = grp.apply(
            lambda g: ratio_of_sums(g["OREB"].shift(1), g["REB"].shift(1), 5, 3),
            include_groups=False,
        ).reset_index(level=0, drop=True)
    else:
        logs["oreb_share_l5"] = np.nan

    # Features 8-10: scoring/usage context per min L5
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

    # ═══════════════════════════════════════════════════════════════════
    # TEAM CONTEXT (season-level, dict lookup — no merge)
    # ═══════════════════════════════════════════════════════════════════
    team_pace_map = teams.set_index(["team_abbr", "season"])["pace"].to_dict()

    logs["team_pace"] = logs.apply(
        lambda r: team_pace_map.get((r["TEAM_ABBREVIATION"], r["SEASON"]), 100.0), axis=1
    )
    logs["opp_pace"] = logs.apply(
        lambda r: team_pace_map.get((r["OPP_TEAM_ABBR"], r["SEASON"]), 100.0), axis=1
    )

    # Opponent rebounds allowed per game (season-level)
    if "opp_reb_allowed" in teams.columns:
        opp_reb_map = teams.set_index(["team_abbr", "season"])["opp_reb_allowed"].to_dict()
        logs["opp_reb_allowed"] = logs.apply(
            lambda r: opp_reb_map.get((r["OPP_TEAM_ABBR"], r["SEASON"]), np.nan), axis=1
        )
    else:
        logs["opp_reb_allowed"] = np.nan
        logger.warning("  opp_reb_allowed not in team stats — using NaN")

    # ═══════════════════════════════════════════════════════════════════
    # OPPONENT ROLLING FEATURES (computed from game logs, shift(1) per team)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("Computing opponent rolling features from game logs...")

    # Aggregate player stats to team-game totals
    team_game = logs.groupby(
        ["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]
    ).agg(
        T_FGA=("FGA", "sum"),
        T_FGM=("FGM", "sum"),
        T_REB=("REB", "sum"),
        T_OREB=("OREB", "sum"),
    ).reset_index()

    team_game["T_FG_PCT"] = team_game["T_FGM"] / team_game["T_FGA"].replace(0, np.nan)
    team_game["T_OREB_RATE"] = team_game["T_OREB"] / team_game["T_REB"].replace(0, np.nan)

    team_game = team_game.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    # Rolling L10 with shift(1) within each team
    tg = team_game.groupby("TEAM_ABBREVIATION")

    team_game["_fga_l10"] = tg["T_FGA"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean()
    )
    team_game["_fg_pct_l10"] = tg["T_FG_PCT"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean()
    )
    team_game["_oreb_rate_l10"] = tg["T_OREB_RATE"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean()
    )

    # Team FGA L5 (own team's shot volume for offensive rebound opportunity)
    team_game["_team_fga_l5"] = tg["T_FGA"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )

    # Build lookup dicts keyed by (team, game_date) — opponent features merge on OPP_TEAM
    opp_feat_df = team_game[["TEAM_ABBREVIATION", "GAME_DATE", "_fga_l10", "_fg_pct_l10", "_oreb_rate_l10"]].copy()
    opp_feat_df = opp_feat_df.drop_duplicates(subset=["TEAM_ABBREVIATION", "GAME_DATE"])
    opp_lookup = opp_feat_df.set_index(["TEAM_ABBREVIATION", "GAME_DATE"]).to_dict("index")

    # Own team FGA L5 — merge on TEAM_ABBREVIATION
    own_fga_df = team_game[["TEAM_ABBREVIATION", "GAME_DATE", "_team_fga_l5"]].copy()
    own_fga_df = own_fga_df.drop_duplicates(subset=["TEAM_ABBREVIATION", "GAME_DATE"])
    own_fga_lookup = own_fga_df.set_index(["TEAM_ABBREVIATION", "GAME_DATE"])["_team_fga_l5"].to_dict()

    logs["opp_fga_l10"] = logs.apply(
        lambda r: opp_lookup.get((r["OPP_TEAM_ABBR"], r["GAME_DATE"]), {}).get("_fga_l10", np.nan), axis=1
    )
    logs["opp_fg_pct_l10"] = logs.apply(
        lambda r: opp_lookup.get((r["OPP_TEAM_ABBR"], r["GAME_DATE"]), {}).get("_fg_pct_l10", np.nan), axis=1
    )
    logs["opp_oreb_rate_l10"] = logs.apply(
        lambda r: opp_lookup.get((r["OPP_TEAM_ABBR"], r["GAME_DATE"]), {}).get("_oreb_rate_l10", np.nan), axis=1
    )
    logs["team_fga_l5"] = logs.apply(
        lambda r: own_fga_lookup.get((r["TEAM_ABBREVIATION"], r["GAME_DATE"]), np.nan), axis=1
    )

    logger.info("  Opponent rolling features computed.")

    # ═══════════════════════════════════════════════════════════════════
    # GAME CONTEXT FEATURES
    # ═══════════════════════════════════════════════════════════════════

    # rest_days
    logs["rest_days"] = grp["GAME_DATE"].diff().dt.days.fillna(7).astype(int)

    # is_home (already precomputed in data_pipeline)
    if "IS_HOME" in logs.columns:
        logs["is_home"] = logs["IS_HOME"].astype(int)
    else:
        logs["is_home"] = (~logs["MATCHUP"].astype(str).str.contains("@")).astype(int)

    # b2b_flag
    logs["b2b_flag"] = (logs["rest_days"] <= 1).astype(int)

    # games_played_season — cumcount BEFORE current game
    logs["games_played_season"] = logs.groupby(["PLAYER_ID", "SEASON"]).cumcount()

    # Target columns
    logs["actual_reb"] = logs["REB"].astype(int)
    logs["actual_min"] = logs["MIN_FLOAT"]

    # ═══════════════════════════════════════════════════════════════════
    # V2: GAME-LEVEL VEGAS FEATURES (game_total, spread_abs)
    # ═══════════════════════════════════════════════════════════════════
    all_lines_path = lines_dir / "all_historical_lines.parquet"
    if all_lines_path.exists():
        logger.info("Merging game-level Vegas data (game_total, spread_abs)...")
        all_lines_full = pd.read_parquet(all_lines_path)

        # Build event → game mapping from player prop lines
        prop_lines = all_lines_full[
            all_lines_full["market"].isin(["player_assists", "player_rebounds"])
        ].copy()
        prop_lines["game_date"] = pd.to_datetime(prop_lines["game_date"])
        event_games = prop_lines.drop_duplicates(subset=["event_id"])[
            ["event_id", "game_date", "home_team", "away_team"]
        ].copy()

        # game_total
        totals = all_lines_full[all_lines_full["market"] == "totals"].copy()
        if not totals.empty:
            totals["game_date"] = pd.to_datetime(totals["game_date"])
            totals_dedup = totals.drop_duplicates(subset=["event_id"], keep="first")[
                ["event_id", "line"]
            ].rename(columns={"line": "game_total"})
            event_games = event_games.merge(totals_dedup, on="event_id", how="left")
        else:
            event_games["game_total"] = np.nan

        # spread_abs
        spreads = all_lines_full[all_lines_full["market"] == "spreads"].copy()
        if not spreads.empty:
            spreads["spread_abs"] = spreads["line"].abs()
            spreads_dedup = spreads.drop_duplicates(subset=["event_id"], keep="first")[
                ["event_id", "spread_abs"]
            ]
            event_games = event_games.merge(spreads_dedup, on="event_id", how="left")
        else:
            event_games["spread_abs"] = np.nan

        # Map Odds API team names → NBA abbreviations
        from lib.odds_team_map import ODDS_TO_NBA_ABBR
        event_games["home_abbr"] = event_games["home_team"].map(ODDS_TO_NBA_ABBR)
        event_games["away_abbr"] = event_games["away_team"].map(ODDS_TO_NBA_ABBR)

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
        logger.warning("No historical lines file — game_total/spread_abs will be NaN")

    # ═══════════════════════════════════════════════════════════════════
    # LOOKAHEAD AUDIT (must run BEFORE filtering — audit needs full game history)
    # ═══════════════════════════════════════════════════════════════════
    if run_audit:
        passed = audit_lookahead(logs)
        if not passed:
            logger.error("LOOKAHEAD AUDIT FAILED — do not use this feature matrix")
            sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # FILTERS
    # ═══════════════════════════════════════════════════════════════════

    # Drop rows with insufficient history
    logs = logs[logs["games_played_season"] >= min_games].copy()

    # Drop rows with NaN in core features (first ~5 games per player)
    # Allow optional features to be NaN: oreb_share, game_total, spread_abs
    optional_features = ("oreb_share_l5", "game_total", "spread_abs")
    core_features = [c for c in REBOUND_FEATURE_COLS if c not in optional_features]
    feature_mask = logs[core_features].notna().all(axis=1)
    before_drop = len(logs)
    logs = logs[feature_mask].copy()
    logger.info("Dropped %d rows with NaN features (%d remaining)", before_drop - len(logs), len(logs))

    # Minutes filter: only players averaging >= 12 min in their L5 window
    logs = logs[logs["proj_minutes"] >= min_minutes].copy()
    logger.info("After min-minutes filter (>=%.0f): %d rows", min_minutes, len(logs))

    # ═══════════════════════════════════════════════════════════════════
    # MERGE DK REBOUND LINES
    # ═══════════════════════════════════════════════════════════════════
    dk_line_path = lines_dir / "all_historical_lines.parquet"
    if dk_line_path.exists():
        logger.info("Merging DK rebound lines...")
        all_lines = pd.read_parquet(dk_line_path)
        dk = all_lines[all_lines["market"] == "player_rebounds"].copy()
        logger.info("  %d rebound lines loaded (from %d total)", len(dk), len(all_lines))

        dk["game_date"] = pd.to_datetime(dk["game_date"])
        dk["_name_lower"] = dk["player"].str.lower().str.strip()

        # Apply alias dictionary + accent normalization to DK names
        def normalize_dk_name(name):
            """Resolve DK name to NBA API name via alias dict, then accent-strip fallback."""
            resolved = DK_TO_NBA_NAME_ALIAS.get(name, name)
            return resolved

        dk["_name_lower"] = dk["_name_lower"].map(normalize_dk_name)

        # For game log names, also build an accent-stripped reverse lookup
        logs["_name_lower"] = logs["PLAYER_NAME"].str.lower().str.strip()

        dk_deduped = dk.drop_duplicates(subset=["_name_lower", "game_date"], keep="first")
        dk_merge = dk_deduped[["_name_lower", "game_date", "line", "over_price", "under_price"]].rename(
            columns={"game_date": "GAME_DATE", "line": "dk_line", "over_price": "dk_over_price", "under_price": "dk_under_price"}
        )

        logs = logs.merge(dk_merge, left_on=["_name_lower", "GAME_DATE"], right_on=["_name_lower", "GAME_DATE"], how="left")
        logs.drop(columns=["_name_lower"], inplace=True)

        matched = logs["dk_line"].notna().sum()
        logger.info("  Matched %d / %d rows with DK lines (%.1f%%)", matched, len(logs), matched / len(logs) * 100)
    else:
        logger.warning("No DK lines file found at %s — skipping", dk_line_path)
        logs["dk_line"] = np.nan
        logs["dk_over_price"] = np.nan
        logs["dk_under_price"] = np.nan

    # ═══════════════════════════════════════════════════════════════════
    # DERIVED FEATURES (for binary classifier)
    # ═══════════════════════════════════════════════════════════════════

    # Predicted rebounds (simple: reb_per_min_l5 * proj_minutes)
    logs["predicted_reb"] = logs["reb_per_min_l5"] * logs["proj_minutes"]

    # Prediction minus line
    logs["pred_minus_line"] = logs["predicted_reb"] - logs["dk_line"]

    # DK implied probabilities from American odds
    logs["dk_implied_over_prob"] = logs["dk_over_price"].apply(american_to_implied_prob)
    logs["dk_implied_under_prob"] = logs["dk_under_price"].apply(american_to_implied_prob)

    # Binary target: did the over hit?
    logs["over_hit"] = np.where(
        logs["dk_line"].isna(), np.nan,
        np.where(logs["REB"] > logs["dk_line"], 1.0,
                 np.where(logs["REB"] == logs["dk_line"], np.nan, 0.0))
    )

    # ═══════════════════════════════════════════════════════════════════
    # SELECT & SAVE
    # ═══════════════════════════════════════════════════════════════════
    output_cols = (
        ["PLAYER_NAME", "PLAYER_ID", "SEASON", "GAME_DATE",
         "TEAM_ABBREVIATION", "OPP_TEAM_ABBR", "GAME_ID"]
        + REBOUND_FEATURE_COLS
        + [REBOUND_TARGET_COL, "actual_min",
           "dk_line", "dk_over_price", "dk_under_price",
           "predicted_reb", "pred_minus_line",
           "dk_implied_over_prob", "dk_implied_under_prob", "over_hit"]
    )

    result = logs[[c for c in output_cols if c in logs.columns]].copy()
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
    output_path = output_dir / "rebound_features.parquet"
    result.to_parquet(output_path, index=False)
    logger.info("Rebound feature matrix saved: %s (%d rows, %d features)",
                output_path, len(result), len(REBOUND_FEATURE_COLS))

    # Summary stats
    print_summary(result)

    return result


# ── Lookahead Audit ──

def audit_lookahead(df: pd.DataFrame, n_samples: int = 100) -> bool:
    """
    Spot-check that reb_per_min_l5 at game N uses only data from games 1..N-1.
    """
    logger.info("Running lookahead audit on %d samples...", n_samples)
    violations = 0

    eligible = df[df["games_played_season"] >= 10].copy()
    if len(eligible) < n_samples:
        n_samples = len(eligible)
    if n_samples == 0:
        logger.warning("  No eligible rows for audit — skipping")
        return True

    sample = eligible.sample(n=n_samples, random_state=42)

    for _, row in sample.iterrows():
        pid = row["PLAYER_ID"]
        gdate = row["GAME_DATE"]

        prior = df[(df["PLAYER_ID"] == pid) & (df["GAME_DATE"] < gdate)].sort_values("GAME_DATE")
        if len(prior) < 5:
            continue

        last5 = prior.tail(5)
        expected_rpm = last5["REB"].sum() / last5["MIN_FLOAT"].sum()
        stored_rpm = row["reb_per_min_l5"]

        if pd.notna(stored_rpm) and pd.notna(expected_rpm):
            if abs(stored_rpm - expected_rpm) > 0.01:
                violations += 1
                logger.warning("  VIOLATION: Player %s on %s: stored=%.4f, expected=%.4f",
                               pid, gdate, stored_rpm, expected_rpm)

    if violations == 0:
        logger.info("  PASS — %d samples checked, zero violations", n_samples)
        return True
    else:
        logger.error("  FAIL — %d violations in %d samples", violations, n_samples)
        return False


# ── Summary ──

def print_summary(result: pd.DataFrame):
    """Print feature matrix summary."""
    logger.info("=" * 60)
    logger.info("REBOUND FEATURE MATRIX SUMMARY")
    logger.info("=" * 60)
    logger.info("Total rows:     %s", f"{len(result):,}")
    logger.info("Unique players: %d", result["player_id"].nunique())
    logger.info("Date range:     %s to %s", result["game_date"].min().date(), result["game_date"].max().date())

    for season in sorted(result["season"].unique()):
        sdf = result[result["season"] == season]
        logger.info("  %s: %s rows, %d players", season, f"{len(sdf):,}", sdf["player_id"].nunique())

    logger.info("\nFeature coverage (non-null %%):")
    for col in REBOUND_FEATURE_COLS:
        if col in result.columns:
            pct = result[col].notna().mean() * 100
            logger.info("  %-28s %6.1f%%", col, pct)

    if "dk_line" in result.columns:
        matched = result["dk_line"].notna().sum()
        logger.info("\nDK line matches: %s / %s (%.1f%%)",
                    f"{matched:,}", f"{len(result):,}", matched / len(result) * 100)

    logger.info("\nTarget (REB) distribution:")
    t = result[REBOUND_TARGET_COL]
    logger.info("  Mean:   %.2f", t.mean())
    logger.info("  Median: %.1f", t.median())
    logger.info("  Std:    %.2f", t.std())

    if "over_hit" in result.columns:
        valid = result["over_hit"].dropna()
        if len(valid) > 0:
            logger.info("\nOver hit rate: %.1f%% (%s games with DK lines)",
                        valid.mean() * 100, f"{len(valid):,}")

    logger.info("=" * 60)


# ── CLI ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build rebound feature matrix")
    parser.add_argument("--skip-audit", action="store_true", help="Skip lookahead audit")
    args = parser.parse_args()

    build_rebound_feature_matrix(run_audit=not args.skip_audit)
