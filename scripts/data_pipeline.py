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
from lib.backtest_utils import ensure_dirs, get_project_root, SEASONS_MAP, current_nba_season

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
def _fetch_or_keep(fetch_fn, season: str, path: Path, label: str) -> None:
    """Re-fetch one per-season parquet. If the API returns nothing (season not
    started yet) or fails, keep the existing snapshot instead of dropping it."""
    try:
        df = fetch_fn(season)
    except Exception as e:  # noqa: BLE001 — surface, don't abort the whole retrain
        df = pd.DataFrame()
        logger.warning("%s fetch for %s failed: %s", label, season, str(e)[:200])
    if not df.empty:
        df.to_parquet(path, index=False)
        logger.info("  Saved %s to %s", label, path)
    elif path.exists():
        logger.warning("No %s returned for %s — keeping cached snapshot %s", label, season, path.name)
    else:
        logger.warning("No %s returned for %s and no cached snapshot — SKIPPING season", label, season)


def run_pipeline(seasons: list = None, data_dir: str = None,
                 refresh_seasons: list = None) -> None:
    """Make sure per-season raw parquets exist, then rebuild the combined files.

    seasons:         seasons that must exist locally (default: all of SEASONS_MAP).
                     Cached per-season files are reused.
    refresh_seasons: seasons to re-fetch from the NBA API even when cached — pass
                     the in-progress season so new games are picked up. An empty
                     or failed refetch keeps the previous snapshot.

    all_game_logs.parquet / all_team_stats.parquet are rebuilt from EVERY
    per-season file in data_dir, not just `seasons`. Previously a refresh of a
    single season overwrote the combined files with that one season, which
    silently shrank the training set used by the weekly retrain.
    """
    ensure_dirs()
    root = get_project_root()
    data_dir = Path(data_dir) if data_dir else root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    if seasons is None:
        seasons = list(SEASONS_MAP.keys())
    refresh = set(refresh_seasons or [])

    for season in seasons:
        logs_path = data_dir / f"game_logs_{season}.parquet"
        teams_path = data_dir / f"team_stats_{season}.parquet"

        if logs_path.exists() and season not in refresh:
            logger.info("Using cached game logs for %s", season)
        else:
            _fetch_or_keep(fetch_season_game_logs, season, logs_path, "game logs")

        if teams_path.exists() and season not in refresh:
            logger.info("Using cached team stats for %s", season)
        else:
            _fetch_or_keep(fetch_team_stats, season, teams_path, "team stats")

    # Combine every per-season file on disk, oldest season first.
    log_files = sorted(data_dir.glob("game_logs_*.parquet"))
    team_files = sorted(data_dir.glob("team_stats_*.parquet"))
    if not log_files:
        logger.error("No per-season game log parquets found in %s — nothing to combine", data_dir)
        return

    combined_logs = pd.concat([pd.read_parquet(p) for p in log_files], ignore_index=True)
    combined_teams = (pd.concat([pd.read_parquet(p) for p in team_files], ignore_index=True)
                      if team_files else pd.DataFrame())

    # fetch_opp_rebounds.py adds opp_reb_allowed/opp_oreb/opp_dreb to the combined
    # team file only. Carry those columns over so a rebuild doesn't drop them.
    all_teams_path = data_dir / "all_team_stats.parquet"
    if all_teams_path.exists() and not combined_teams.empty:
        try:
            prev = pd.read_parquet(all_teams_path)
            extra = [c for c in prev.columns if c not in combined_teams.columns]
            if extra and {"team_abbr", "season"} <= set(prev.columns):
                carry = prev[["team_abbr", "season", *extra]].drop_duplicates(["team_abbr", "season"])
                combined_teams = combined_teams.merge(carry, on=["team_abbr", "season"], how="left")
                logger.info("Preserved extra team-stat columns from previous combined file: %s", extra)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not preserve extra team-stat columns: %s", e)

    combined_logs.to_parquet(data_dir / "all_game_logs.parquet", index=False)
    combined_teams.to_parquet(all_teams_path, index=False)

    logger.info("Pipeline complete. Total: %d player-games, %d team-seasons across %d season file(s)",
                len(combined_logs), len(combined_teams), len(log_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch NBA game data")
    parser.add_argument("--seasons", nargs="+", default=list(SEASONS_MAP.keys()))
    parser.add_argument("--refresh-current", action="store_true",
                        help="Re-fetch the in-progress season even if a cached parquet exists")
    args = parser.parse_args()
    refresh = [current_nba_season()] if args.refresh_current else None
    run_pipeline(seasons=args.seasons, refresh_seasons=refresh)
