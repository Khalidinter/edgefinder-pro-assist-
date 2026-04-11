#!/usr/bin/env python3
"""
Fetch opponent rebounding stats (OPP_REB, OPP_OREB, OPP_DREB) from NBA API
and merge into the existing all_team_stats.parquet file.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import leaguedashteamstats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "raw" / "all_team_stats.parquet"

SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]


def api_call_with_retry(fn, max_retries=3, base_delay=3.0, **kwargs):
    """Call NBA API with exponential backoff (mirrors data_pipeline.py pattern)."""
    import random
    for attempt in range(max_retries + 1):
        try:
            time.sleep(1.5)  # Rate limit
            result = fn(**kwargs)
            return result
        except Exception as e:
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), 30.0)
            print(f"  API call failed (attempt {attempt+1}/{max_retries}): {str(e)[:150]}. Retrying in {delay:.1f}s")
            time.sleep(delay)


def fetch_opp_rebound_stats(season: str) -> pd.DataFrame:
    """
    Fetch Opponent stats for a season and extract rebounding columns.
    Returns DataFrame with: team_name, opp_reb_allowed, opp_oreb_allowed, opp_dreb_allowed, season
    """
    print(f"Fetching opponent rebound stats for {season}...")

    opp = api_call_with_retry(
        leaguedashteamstats.LeagueDashTeamStats,
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Opponent",
        per_mode_detailed="PerGame",
    )
    opp_df = opp.get_data_frames()[0]

    if opp_df.empty:
        print(f"  WARNING: No data returned for {season}")
        return pd.DataFrame()

    # Extract rebound columns -- the Opponent endpoint uses OPP_ prefix
    rows = []
    for _, row in opp_df.iterrows():
        rows.append({
            "team_name": row["TEAM_NAME"],
            "opp_reb_allowed": float(row["OPP_REB"]),
            "opp_oreb_allowed": float(row["OPP_OREB"]),
            "opp_dreb_allowed": float(row["OPP_DREB"]),
            "season": season,
        })

    result = pd.DataFrame(rows)
    print(f"  {season}: {len(result)} teams fetched (OPP_REB, OPP_OREB, OPP_DREB)")
    return result


def main():
    # Load existing parquet
    if not PARQUET_PATH.exists():
        print(f"ERROR: {PARQUET_PATH} not found")
        sys.exit(1)

    existing = pd.read_parquet(PARQUET_PATH)
    print(f"Existing file: {existing.shape[0]} rows, columns: {list(existing.columns)}")

    # Check if columns already exist (idempotent)
    new_cols = ["opp_reb_allowed", "opp_oreb_allowed", "opp_dreb_allowed"]
    if all(c in existing.columns for c in new_cols):
        print("Opponent rebound columns already exist. Will overwrite with fresh data.")
        existing = existing.drop(columns=new_cols)

    # Fetch opponent rebound data for each season
    all_opp = []
    for season in SEASONS:
        opp_df = fetch_opp_rebound_stats(season)
        if not opp_df.empty:
            all_opp.append(opp_df)
        time.sleep(2.0)  # Extra delay between seasons

    if not all_opp:
        print("ERROR: No opponent data fetched from any season")
        sys.exit(1)

    opp_combined = pd.concat(all_opp, ignore_index=True)
    print(f"\nTotal opponent rebound rows fetched: {opp_combined.shape[0]}")

    # Merge on team_name + season (both DataFrames have these columns)
    merged = existing.merge(
        opp_combined,
        on=["team_name", "season"],
        how="left",
    )

    # Check for any teams that didn't match
    missing = merged[merged["opp_reb_allowed"].isna()]
    if len(missing) > 0:
        print(f"\nWARNING: {len(missing)} rows had no opponent rebound match:")
        print(missing[["team_abbr", "team_name", "season"]].to_string(index=False))

    # Save back
    merged.to_parquet(PARQUET_PATH, index=False)
    print(f"\nSaved updated file to: {PARQUET_PATH}")

    # Verify
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    print(f"Shape: {merged.shape}")
    print(f"Columns: {list(merged.columns)}")
    print(f"\nSample rows:")
    print(merged[["team_abbr", "team_name", "pace", "opp_ast_allowed",
                   "opp_reb_allowed", "opp_oreb_allowed", "opp_dreb_allowed",
                   "season"]].head(10).to_string(index=False))
    print(f"\nRebound stats summary:")
    for col in new_cols:
        vals = merged[col].dropna()
        print(f"  {col}: min={vals.min():.1f}, max={vals.max():.1f}, mean={vals.mean():.1f}, nulls={merged[col].isna().sum()}")


if __name__ == "__main__":
    main()
