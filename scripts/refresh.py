#!/usr/bin/env python3
"""
Daily refresh: fetch live market, build projections, save to Supabase.
Run via GitHub Actions or locally: python scripts/refresh.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import logger
from lib.model import fetch_live_market, build_projection_rows, current_nba_season_string
from lib.db import save_projections


def main():
    logger.info("Starting daily refresh...")

    market, err, fetched_at = fetch_live_market()
    if market is None:
        logger.error("Market fetch failed: %s", err)
        sys.exit(1)

    season = current_nba_season_string()
    rows, summary, metrics, gen_at = build_projection_rows(market, season=season)

    logger.info("Projections built: %d players", len(rows))
    save_projections(rows, summary, metrics)
    logger.info("Saved to Supabase. Done.")


if __name__ == "__main__":
    main()
