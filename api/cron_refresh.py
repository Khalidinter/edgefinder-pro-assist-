"""Vercel Cron: daily cache refresh. Fetches live market + builds projections."""
import os
from flask import Flask, request, jsonify
from lib.config import logger
from lib.model import fetch_live_market, build_projection_rows, current_nba_season_string
from lib.db import save_projections

app = Flask(__name__)


@app.route("/api/cron/refresh")
def cron_refresh():
    # Verify cron secret in production
    auth = request.headers.get("Authorization", "")
    cron_secret = os.getenv("CRON_SECRET")
    if cron_secret and auth != f"Bearer {cron_secret}":
        # Allow unauthenticated in dev
        if os.getenv("VERCEL_ENV"):
            return jsonify({"error": "Unauthorized"}), 401

    logger.info("Cron refresh triggered")

    market, err, fetched_at = fetch_live_market()
    if market is None:
        logger.warning("Market fetch failed: %s", err)
        return jsonify({"error": err, "fetched_at": fetched_at}), 500

    season = current_nba_season_string()
    rows, summary, metrics, gen_at = build_projection_rows(market, season=season)

    save_projections(rows, summary, metrics)

    return jsonify({
        "status": "ok",
        "players_projected": len(rows),
        "market_players": metrics["market_players"],
        "fetched_at": fetched_at,
        "generated_at": gen_at,
    })
