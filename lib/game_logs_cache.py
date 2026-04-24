"""Player game-logs cache (S5 — NBA API outage fallback).

Strategy: every time the Refresh pipeline pulls a player's game logs from
stats.nba.com we persist them to a shared parquet file, keyed by
(player_id, season). When the NBA API is unreachable, the caller loads the
most-recent snapshot instead — rolling features only consume the last ~20 games
per player, and yesterday's cache is typically missing just one row (last
night's box score), so predictions stay viable.

Design rules:
  • Cache is **only** for predict/refresh paths. Resolve still needs fresh
    data from last night's games, which the cache almost certainly lacks.
  • A single parquet at data/cache/game_logs_latest.parquet holds every
    player's logs stacked into one frame. Each row carries _cache_player_id,
    _cache_season, and _cache_fetched_at so we can upsert a specific player
    without rewriting the entire cache's non-player rows.
  • Freshness is judged by the oldest-stored row's _cache_fetched_at, not
    the newest — a 30-hour-old row for Player X is stale even if Player Y's
    row is fresh.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from lib.backtest_utils import get_project_root
from lib.config import logger

ROOT = get_project_root()
CACHE_DIR = ROOT / "data" / "cache"
CACHE_PATH = CACHE_DIR / "game_logs_latest.parquet"
MAX_CACHE_AGE = timedelta(hours=24)

_SENTINELS = ("_cache_player_id", "_cache_season", "_cache_fetched_at")


def save_player_logs(player_id: int, season: str, logs: pd.DataFrame) -> None:
    """Upsert a single player's logs into the shared cache.

    Silent no-op on empty frames so a failed or short-history fetch never
    wipes out a good cached snapshot.
    """
    if logs is None or logs.empty:
        return
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df = logs.copy()
        # Normalize sentinel columns — overwrite if the player frame happened
        # to carry them in already, e.g. from a reload.
        df["_cache_player_id"] = int(player_id)
        df["_cache_season"] = str(season)
        df["_cache_fetched_at"] = datetime.now(timezone.utc).isoformat()

        if CACHE_PATH.exists():
            try:
                existing = pd.read_parquet(CACHE_PATH)
                existing = existing[~(
                    (existing["_cache_player_id"] == int(player_id))
                    & (existing["_cache_season"] == str(season))
                )]
                df = pd.concat([existing, df], ignore_index=True, sort=False)
            except Exception as e:
                # Corrupt file — log once and rewrite cleanly. Better to lose
                # a day of cache than block today's refresh.
                logger.warning("Cache read failed (%s) — rewriting from scratch", e)

        df.to_parquet(CACHE_PATH, index=False)
    except Exception as e:
        # Cache is best-effort. Don't let a parquet hiccup kill a live refresh.
        logger.warning("Failed to cache logs for player %s: %s", player_id, e)


def load_player_logs(player_id: int, season: str) -> Optional[pd.DataFrame]:
    """Return cached logs for this (player_id, season), or None if absent."""
    if not CACHE_PATH.exists():
        return None
    try:
        df = pd.read_parquet(CACHE_PATH)
    except Exception as e:
        logger.warning("Cache read failed: %s", e)
        return None
    if df.empty:
        return None
    mask = (df["_cache_player_id"] == int(player_id)) & (df["_cache_season"] == str(season))
    hit = df[mask]
    if hit.empty:
        return None
    # Strip sentinels so the returned frame looks exactly like a fresh API fetch.
    return hit.drop(columns=[c for c in _SENTINELS if c in hit.columns]).reset_index(drop=True)


def cache_age() -> Optional[timedelta]:
    """Age of the **oldest** stored snapshot.

    Using the oldest (min) rather than newest (max) is intentional — it treats
    the cache as only as fresh as its staleest player row, which matches how
    the downstream features consume it (every player, not a sampled subset).
    Returns None when the cache is missing or malformed.
    """
    if not CACHE_PATH.exists():
        return None
    try:
        df = pd.read_parquet(CACHE_PATH, columns=["_cache_fetched_at"])
    except Exception:
        return None
    if df.empty or "_cache_fetched_at" not in df.columns:
        return None
    try:
        oldest = pd.to_datetime(df["_cache_fetched_at"], utc=True).min()
        if pd.isna(oldest):
            return None
        return datetime.now(timezone.utc) - oldest.to_pydatetime()
    except Exception:
        return None


def cache_is_fresh(max_age: timedelta = MAX_CACHE_AGE) -> bool:
    age = cache_age()
    return age is not None and age < max_age


def cached_player_count() -> int:
    if not CACHE_PATH.exists():
        return 0
    try:
        df = pd.read_parquet(CACHE_PATH, columns=["_cache_player_id"])
        return int(df["_cache_player_id"].nunique())
    except Exception:
        return 0
