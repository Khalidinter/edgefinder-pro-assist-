#!/usr/bin/env python3
"""
Bulk NBA API data fetcher.
Uses LeagueGameLog (ALL players per season in ONE call) + LeagueDashTeamStats.
Total: ~12 API calls for 4 seasons of ~160,000 player-games.
"""
import sys, os, time, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path

from nba_api.stats.endpoints import leaguegamelog, leaguedashteamstats
from lib.config import logger
from lib.backtest_utils import ensure_dirs, get_project_root, SEASONS_MAP

# ── Retry wrapper ──
def api_call_with_retry(fn, max_retries=3, base_delay=3.0, max_delay=30.0, **kwargs):
    """Call NBA API function with exponential backoff + jitter."""
    for attempt in range(max_retries + 1):
        try:
            time.sleep(1.0)  # Base rate limit
            result = fn(**kwargs)
            return result
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
            logger.warning("API call failed (attempt %d/%d): %s. Retrying in %.1fs",
                          attempt + 1, max_retries, str(e)[:200], delay)
            time.sleep(delay)


def safe_minutes(val) -> float:
    """Parse MIN column (could be float, int, or 'MM:SS' string)."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if ":" in s:
        try:
            parts = s.split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


# ── Fetch all player game logs for a season ──
def fetch_season_game_logs(season: str) -> pd.DataFrame:
    """
    One API call returns ALL ~40,000 player-game rows for a season.
    Uses LeagueGameLog with player_or_team_abbreviation="P".
    """
    logger.info("Fetching game logs for season %s (all players)...", season)

    endpoint = api_call_with_retry(
        leaguegamelog.LeagueGameLog,
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star="Regular Season",
    )
    df = endpoint.get_data_frames()[0]

    if df.empty:
        logger.warning("No game logs returned for season %s", season)
        return pd.DataFrame()

    # Normalize columns
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["MIN_FLOAT"] = df["MIN"].apply(safe_minutes)
    df["SEASON"] = season

    # Parse team abbreviation from TEAM_ABBREVIATION column
    if "TEAM_ABBREVIATION" not in df.columns and "TEAM_ID" in df.columns:
        df["TEAM_ABBREVIATION"] = ""

    # Parse opponent from MATCHUP (e.g., "LAL vs. GSW" or "LAL @ GSW")
    df["IS_HOME"] = ~df["MATCHUP"].astype(str).str.contains("@")
    df["OPP_TEAM_ABBR"] = df["MATCHUP"].apply(_parse_opponent)

    logger.info("  Season %s: %d player-games loaded", season, len(df))
    return df


def _parse_opponent(matchup: str) -> str:
    """Parse opponent abbreviation from MATCHUP string."""
    m = str(matchup).strip()
    if " vs. " in m:
        return m.split(" vs. ")[1].strip()
    elif " @ " in m:
        return m.split(" @ ")[1].strip()
    return ""


# ── Fetch team stats ──
def fetch_team_stats(season: str) -> pd.DataFrame:
    """
    Two API calls per season: Advanced (PACE) + Opponent (AST allowed).
    Returns DataFrame with team_abbr, pace, opp_ast_allowed.
    """
    logger.info("Fetching team stats for season %s...", season)

    # Advanced stats (for PACE)
    adv = api_call_with_retry(
        leaguedashteamstats.LeagueDashTeamStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    adv_df = adv.get_data_frames()[0]

    # Opponent stats (for AST allowed)
    time.sleep(1.5)
    opp = api_call_with_retry(
        leaguedashteamstats.LeagueDashTeamStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Opponent",
        per_mode_detailed="PerGame",
    )
    opp_df = opp.get_data_frames()[0]

    # Extract relevant columns — indexed by team name
    pace_map = adv_df.set_index("TEAM_NAME")["PACE"].to_dict()

    ast_col = "AST" if "AST" in opp_df.columns else "OPP_AST"
    opp_ast_map = opp_df.set_index("TEAM_NAME")[ast_col].to_dict()

    # Build name-to-abbreviation lookup from adv_df
    if "TEAM_ABBREVIATION" in adv_df.columns:
        name_to_abbr = adv_df.set_index("TEAM_NAME")["TEAM_ABBREVIATION"].to_dict()
    else:
        # Fallback: reverse the model's TEAM_ABBR_TO_NAME map
        from lib.model import TEAM_ABBR_TO_NAME
        name_to_abbr = {v: k for k, v in TEAM_ABBR_TO_NAME.items()}

    # Combine
    team_names = sorted(set(pace_map.keys()) | set(opp_ast_map.keys()))
    rows = []
    for name in team_names:
        abbr = name_to_abbr.get(name, name[:3].upper())
        rows.append({
            "team_abbr": abbr,
            "team_name": name,
            "pace": float(pace_map.get(name, 100.0)),
            "opp_ast_allowed": float(opp_ast_map.get(name, 25.0)),
            "season": season,
        })

    result = pd.DataFrame(rows)
    logger.info("  Season %s: %d teams loaded (pace + opp_ast)", season, len(result))
    return result


# ── Main pipeline ──
def run_pipeline(seasons: list = None, data_dir: str = None) -> None:
    ensure_dirs()
    root = get_project_root()
    data_dir = Path(data_dir) if data_dir else root / "data" / "raw"

    if seasons is None:
        seasons = list(SEASONS_MAP.keys())

    all_logs = []
    all_teams = []

    for season in seasons:
        logs_path = data_dir / f"game_logs_{season}.parquet"
        teams_path = data_dir / f"team_stats_{season}.parquet"

        # Fetch game logs
        if logs_path.exists():
            logger.info("Loading cached game logs for %s", season)
            logs = pd.read_parquet(logs_path)
        else:
            logs = fetch_season_game_logs(season)
            if not logs.empty:
                logs.to_parquet(logs_path, index=False)
                logger.info("  Saved to %s", logs_path)

        # Fetch team stats
        if teams_path.exists():
            logger.info("Loading cached team stats for %s", season)
            teams = pd.read_parquet(teams_path)
        else:
            teams = fetch_team_stats(season)
            if not teams.empty:
                teams.to_parquet(teams_path, index=False)
                logger.info("  Saved to %s", teams_path)

        all_logs.append(logs)
        all_teams.append(teams)

    # Combine all seasons
    combined_logs = pd.concat(all_logs, ignore_index=True)
    combined_teams = pd.concat(all_teams, ignore_index=True)

    combined_logs.to_parquet(data_dir / "all_game_logs.parquet", index=False)
    combined_teams.to_parquet(data_dir / "all_team_stats.parquet", index=False)

    logger.info("Pipeline complete. Total: %d player-games, %d team-seasons",
                len(combined_logs), len(combined_teams))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA game data")
    parser.add_argument("--seasons", nargs="+", default=list(SEASONS_MAP.keys()))
    args = parser.parse_args()
    run_pipeline(seasons=args.seasons)
