#!/usr/bin/env python3
"""
Fetch historical player prop lines from The Odds API.
Pulls player_assists + player_rebounds for all NBA games.
Stores results as parquet files in data/lines/.

Cost: 20 quota per event (2 markets × 10 per market × 1 region)
      + 1 per events list call
Expected: ~170 game days × (1 + 8×20) = ~27,370 quota per season
"""
import sys, os, time, argparse, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from lib.config import logger

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "f39e8cadbb6880ac4d4cd0c5bf2f481c")
SPORT = "basketball_nba"
MARKETS = "player_assists,player_rebounds"
REGIONS = "us"
BOOKMAKERS = "draftkings"  # Focus on DK for consistency
BASE = "https://api.the-odds-api.com/v4"

# NBA regular season approximate date ranges
SEASON_DATES = {
    "2023-24": ("2023-10-24", "2024-04-14"),
    "2024-25": ("2024-10-22", "2025-04-13"),
    "2025-26": ("2025-10-21", "2026-04-12"),
}


def api_get(url, params, max_retries=3):
    """GET with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            import requests
            r = requests.get(url, params=params, timeout=20)
            remaining = r.headers.get("x-requests-remaining", "?")
            if r.status_code == 200:
                return r.json(), remaining
            elif r.status_code == 429:
                delay = min(10 * (2 ** attempt) + random.uniform(0, 2), 60)
                logger.warning("Rate limited. Waiting %.0fs...", delay)
                time.sleep(delay)
                continue
            elif r.status_code == 422:
                return {"error": r.json().get("message", ""), "status": 422}, remaining
            else:
                logger.warning("API error %d: %s", r.status_code, r.text[:200])
                if attempt < max_retries:
                    time.sleep(3 * (attempt + 1))
                    continue
                return None, remaining
        except Exception as e:
            if attempt < max_retries:
                time.sleep(3 * (attempt + 1))
                continue
            logger.error("API call failed: %s", e)
            return None, "?"
    return None, "?"


def fetch_events_for_date(date_str: str) -> list:
    """Get all NBA events at a specific historical timestamp."""
    data, _ = api_get(
        f"{BASE}/historical/sports/{SPORT}/events",
        {"apiKey": ODDS_API_KEY, "date": f"{date_str}T20:00:00Z"},
    )
    if data and "data" in data:
        return data["data"]
    return []


def fetch_event_odds(event_id: str, date_str: str) -> dict:
    """Get player_assists + player_rebounds odds for one event."""
    data, remaining = api_get(
        f"{BASE}/historical/sports/{SPORT}/events/{event_id}/odds",
        {
            "apiKey": ODDS_API_KEY,
            "date": f"{date_str}T20:00:00Z",
            "regions": REGIONS,
            "markets": MARKETS,
            "oddsFormat": "american",
            "bookmakers": BOOKMAKERS,
        },
    )
    return data, remaining


def parse_odds_to_rows(odds_data: dict, event_info: dict, date_str: str) -> list:
    """Parse API response into flat rows."""
    rows = []
    if not odds_data or "data" not in odds_data:
        return rows

    event = odds_data.get("data", {})
    home_team = event.get("home_team", event_info.get("home_team", ""))
    away_team = event.get("away_team", event_info.get("away_team", ""))
    event_id = event_info.get("id", "")
    commence = event_info.get("commence_time", "")

    for book in event.get("bookmakers", []):
        book_key = book.get("key", "")
        book_title = book.get("title", "")

        for market in book.get("markets", []):
            market_key = market.get("key", "")  # player_assists or player_rebounds

            # Group outcomes by player+line
            player_lines = {}
            for outcome in market.get("outcomes", []):
                player = outcome.get("description", "")
                side = outcome.get("name", "")  # Over or Under
                line = outcome.get("point")
                price = outcome.get("price")

                if not player or side not in ("Over", "Under") or line is None:
                    continue

                key = (player, line)
                if key not in player_lines:
                    player_lines[key] = {
                        "player": player,
                        "market": market_key,
                        "line": line,
                        "book": book_title,
                        "book_key": book_key,
                        "event_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "game_date": date_str,
                        "commence_time": commence,
                    }

                if side == "Over":
                    player_lines[key]["over_price"] = price
                elif side == "Under":
                    player_lines[key]["under_price"] = price

            for pl in player_lines.values():
                if "over_price" in pl and "under_price" in pl:
                    rows.append(pl)

    return rows


def fetch_season(season: str, output_dir: Path, resume: bool = True) -> pd.DataFrame:
    """Fetch all historical lines for one NBA season."""
    start_str, end_str = SEASON_DATES[season]
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    output_path = output_dir / f"lines_{season}.parquet"
    checkpoint_path = output_dir / f"lines_{season}_checkpoint.json"

    # Resume from checkpoint
    all_rows = []
    processed_dates = set()
    if resume and output_path.exists():
        existing = pd.read_parquet(output_path)
        all_rows = existing.to_dict("records")
        processed_dates = set(existing["game_date"].unique())
        logger.info("Resuming %s: %d rows, %d dates already done", season, len(all_rows), len(processed_dates))

    current = start
    day_count = 0
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        current += timedelta(days=1)

        if date_str in processed_dates:
            continue

        # Skip known non-game days (rough filter — API returns empty for off-days)
        events = fetch_events_for_date(date_str)
        if not events:
            continue

        day_count += 1
        day_rows = 0

        for event in events:
            eid = event.get("id")
            if not eid:
                continue

            odds_data, remaining = fetch_event_odds(eid, date_str)

            if odds_data and isinstance(odds_data, dict) and "error" not in odds_data:
                rows = parse_odds_to_rows(odds_data, event, date_str)
                all_rows.extend(rows)
                day_rows += len(rows)

            time.sleep(0.3)  # Be gentle

        logger.info("  %s: %d events, %d lines fetched (quota: %s)",
                    date_str, len(events), day_rows, remaining)

        # Checkpoint every 5 game days
        if day_count % 5 == 0:
            df = pd.DataFrame(all_rows)
            if not df.empty:
                df.to_parquet(output_path, index=False)
                logger.info("  Checkpoint saved: %d total rows", len(df))

    # Final save
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df.to_parquet(output_path, index=False)

    logger.info("Season %s complete: %d total rows, %d game days", season, len(df), day_count)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch historical NBA player prop lines")
    parser.add_argument("--seasons", nargs="+", default=["2023-24", "2024-25", "2025-26"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = Path(args.output_dir) if args.output_dir else root / "data" / "lines"
    output_dir.mkdir(parents=True, exist_ok=True)

    import requests
    r = requests.get(f"{BASE}/sports", params={"apiKey": ODDS_API_KEY}, timeout=10)
    logger.info("Starting historical line fetch. Quota: %s", r.headers.get("x-requests-remaining"))

    all_dfs = []
    for season in args.seasons:
        if season not in SEASON_DATES:
            logger.warning("Unknown season %s, skipping", season)
            continue
        df = fetch_season(season, output_dir, resume=not args.no_resume)
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_parquet(output_dir / "all_historical_lines.parquet", index=False)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("HISTORICAL LINES FETCH COMPLETE")
        logger.info("=" * 60)
        logger.info("Total rows: %d", len(combined))
        for market in combined["market"].unique():
            m = combined[combined["market"] == market]
            logger.info("  %s: %d lines, %d unique players",
                       market, len(m), m["player"].nunique())
        logger.info("Date range: %s to %s",
                    combined["game_date"].min(), combined["game_date"].max())

    # Check remaining quota
    r2 = requests.get(f"{BASE}/sports", params={"apiKey": ODDS_API_KEY}, timeout=10)
    logger.info("Quota remaining: %s", r2.headers.get("x-requests-remaining"))


if __name__ == "__main__":
    main()
