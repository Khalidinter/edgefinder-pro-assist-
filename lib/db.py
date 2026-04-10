"""
Supabase cache layer for the assist model.
All tables live in the `assist_model` schema.
"""
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from lib.config import get_supabase, SCHEMA, logger


def _table(name: str):
    return get_supabase().schema(SCHEMA).table(name)


# ── Model Projections ──
def get_cached_projections() -> List[Dict]:
    now = datetime.now(timezone.utc).isoformat()
    res = _table("model_projections").select("*").gte("expires_at", now).execute()
    if res.data:
        return [json.loads(r["model_data"]) if isinstance(r["model_data"], str) else r["model_data"] for r in res.data]
    return []


def save_projections(rows: List[Dict], summary: Dict, metrics: Dict) -> None:
    # Clear expired
    now = datetime.now(timezone.utc).isoformat()
    try:
        _table("model_projections").delete().lt("expires_at", now).execute()
    except Exception:
        pass

    expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    records = []
    for r in rows:
        records.append({
            "player": r.get("player"),
            "team": r.get("team"),
            "opponent": r.get("opponent"),
            "venue": r.get("venue"),
            "market_line": r.get("market_line"),
            "expected_assists": r.get("expected_assists"),
            "over_prob": r.get("over_prob"),
            "best_edge": r.get("best_edge"),
            "best_ev": r.get("best_ev"),
            "best_side": r.get("best_side"),
            "kelly_pct": r.get("kelly_pct"),
            "confidence": r.get("confidence"),
            "model_data": json.dumps(r),
            "season": r.get("season", ""),
            "expires_at": expires,
        })

    if records:
        # Batch insert in chunks of 50
        for i in range(0, len(records), 50):
            _table("model_projections").insert(records[i:i+50]).execute()
    logger.info("Saved %d projections to Supabase", len(records))


# ── Historical Game Logs ──
def get_cached_game_logs(player_name: str, season: str) -> Optional[List[Dict]]:
    res = (_table("historical_game_logs")
           .select("*")
           .eq("player_name", player_name)
           .eq("season", season)
           .order("game_date")
           .execute())
    return res.data if res.data else None


def save_game_logs(logs: List[Dict]) -> None:
    if not logs:
        return
    for i in range(0, len(logs), 50):
        try:
            _table("historical_game_logs").upsert(
                logs[i:i+50], on_conflict="player_id,game_date"
            ).execute()
        except Exception as e:
            logger.warning("Failed to save game logs batch: %s", e)


# ── Player Tracking ──
def get_cached_tracking(player_id: int, season: str) -> Optional[Dict]:
    now = datetime.now(timezone.utc).isoformat()
    res = (_table("player_tracking")
           .select("*")
           .eq("player_id", player_id)
           .eq("season", season)
           .gte("expires_at", now)
           .limit(1)
           .execute())
    return res.data[0] if res.data else None


def save_tracking(player_id: int, team_id: int, season: str, data: Dict) -> None:
    expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    _table("player_tracking").upsert({
        "player_id": player_id,
        "team_id": team_id,
        "season": season,
        "pa_ratio": data.get("pa_ratio"),
        "tracking_conversion": data.get("tracking_conversion"),
        "total_ast": data.get("total_ast"),
        "potential_ast": data.get("potential_ast"),
        "expires_at": expires,
    }, on_conflict="player_id,season").execute()


# ── Backtest Results ──
def save_backtest_run(run_summary: Dict, results: List[Dict]) -> str:
    run = _table("backtest_runs").insert({
        "seasons": run_summary.get("seasons", []),
        "players_tested": run_summary.get("players_tested", 0),
        "total_predictions": run_summary.get("total_predictions", 0),
        "correct": run_summary.get("correct", 0),
        "hit_rate": run_summary.get("hit_rate", 0),
        "mae": run_summary.get("mae", 0),
        "avg_error": run_summary.get("avg_error", 0),
        "calibration": run_summary.get("calibration", {}),
        "line_source": run_summary.get("line_source", "synthetic"),
    }).execute()

    run_id = run.data[0]["id"] if run.data else "unknown"

    for r in results:
        r["run_id"] = run_id

    for i in range(0, len(results), 50):
        _table("backtest_results").insert(results[i:i+50]).execute()

    logger.info("Saved backtest run %s with %d results", run_id, len(results))
    return run_id


def get_backtest_runs() -> List[Dict]:
    res = _table("backtest_runs").select("*").order("created_at", desc=True).limit(20).execute()
    return res.data or []
