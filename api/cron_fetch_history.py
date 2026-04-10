"""Fetch and store historical game logs for 3 seasons. Run manually or via cron."""
import time
from flask import Flask, request, jsonify
from lib.config import logger
from lib.model import (
    get_multi_season_logs, find_player_id, safe_minutes_to_float,
    BACKTEST_FALLBACK_PLAYERS,
)
from lib.db import save_game_logs, get_cached_game_logs

app = Flask(__name__)

TARGET_SEASONS = ["2023-24", "2024-25", "2025-26"]


@app.route("/api/cron/fetch-history")
def fetch_history():
    player_list = request.args.get("players", "").split(",")
    player_list = [p.strip() for p in player_list if p.strip()]
    if not player_list:
        player_list = BACKTEST_FALLBACK_PLAYERS

    seasons = request.args.get("seasons", ",".join(TARGET_SEASONS)).split(",")
    fetched = 0
    skipped = 0
    errors = 0

    for player_name in player_list:
        for season in seasons:
            # Check if already cached
            cached = get_cached_game_logs(player_name, season)
            if cached and len(cached) > 10:
                skipped += 1
                continue

            pid = find_player_id(player_name)
            if pid is None:
                errors += 1
                continue

            from lib.model import get_player_logs_df
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
                    "team_abbr": str(row.get("TEAM_ABBREVIATION", "")) if "TEAM_ABBREVIATION" in df.columns else "",
                    "season": season,
                    "game_date": str(row["GAME_DATE"].date()) if hasattr(row["GAME_DATE"], "date") else str(row["GAME_DATE"]),
                    "game_id": str(row.get("GAME_ID", "")) if "GAME_ID" in df.columns else "",
                    "matchup": str(row.get("MATCHUP", "")) if "MATCHUP" in df.columns else "",
                    "minutes": round(safe_minutes_to_float(row.get("MIN", 0)), 1),
                    "assists": int(row.get("AST", 0)),
                    "points": int(row.get("PTS", 0)) if "PTS" in df.columns else 0,
                    "rebounds": int(row.get("REB", 0)) if "REB" in df.columns else 0,
                    "turnovers": int(row.get("TOV", 0)) if "TOV" in df.columns else 0,
                    "fga": int(row.get("FGA", 0)) if "FGA" in df.columns else 0,
                    "fgm": int(row.get("FGM", 0)) if "FGM" in df.columns else 0,
                })

            save_game_logs(records)
            fetched += 1
            logger.info("Fetched %s %s: %d games", player_name, season, len(records))
            time.sleep(0.6)

    return jsonify({
        "status": "ok",
        "players": len(player_list),
        "seasons": seasons,
        "fetched": fetched,
        "skipped_cached": skipped,
        "errors": errors,
    })
