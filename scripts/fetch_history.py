#!/usr/bin/env python3
"""
Fetch historical game logs for 3 seasons and save to Supabase.
Run once: python scripts/fetch_history.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import logger
from lib.model import (
    get_player_logs_df, find_player_id, safe_minutes_to_float,
    BACKTEST_FALLBACK_PLAYERS,
)
from lib.db import save_game_logs, get_cached_game_logs

SEASONS = ["2023-24", "2024-25", "2025-26"]


def main():
    players = BACKTEST_FALLBACK_PLAYERS
    fetched, skipped, errors = 0, 0, 0

    for player_name in players:
        for season in SEASONS:
            cached = get_cached_game_logs(player_name, season)
            if cached and len(cached) > 10:
                skipped += 1
                continue

            result = get_player_logs_df(player_name, season)
            if result is None:
                errors += 1
                time.sleep(0.6)
                continue

            df, player_id, team_id = result
            records = []
            for _, row in df.iterrows():
                records.append({
                    "player_name": player_name,
                    "player_id": player_id,
                    "team_id": team_id,
                    "team_abbr": str(row.get("TEAM_ABBREVIATION", "")),
                    "season": season,
                    "game_date": str(row["GAME_DATE"].date()),
                    "game_id": str(row.get("GAME_ID", "")),
                    "matchup": str(row.get("MATCHUP", "")),
                    "minutes": round(safe_minutes_to_float(row.get("MIN", 0)), 1),
                    "assists": int(row.get("AST", 0)),
                    "points": int(row.get("PTS", 0)),
                    "rebounds": int(row.get("REB", 0)),
                    "turnovers": int(row.get("TOV", 0)),
                    "fga": int(row.get("FGA", 0)),
                    "fgm": int(row.get("FGM", 0)),
                })

            save_game_logs(records)
            fetched += 1
            logger.info("Fetched %s %s: %d games", player_name, season, len(records))
            time.sleep(0.6)

    logger.info("Done. fetched=%d skipped=%d errors=%d", fetched, skipped, errors)


if __name__ == "__main__":
    main()
