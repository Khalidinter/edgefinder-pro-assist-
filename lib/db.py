"""
Supabase REST API wrapper for the assist model.
Uses requests directly instead of the supabase Python client.
All tables live in the `assist_model` schema.
"""
import json
import requests
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from lib.config import SUPABASE_URL, SUPABASE_KEY, SCHEMA, logger

BASE = f"{SUPABASE_URL}/rest/v1"
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Accept-Profile": "public",
    "Content-Profile": "public",
    "Prefer": "return=representation",
}


def _get(table: str, params: str = "") -> List[Dict]:
    url = f"{BASE}/{table}?{params}" if params else f"{BASE}/{table}"
    res = requests.get(url, headers=HEADERS, timeout=15)
    res.raise_for_status()
    return res.json()


def _post(table: str, data: List[Dict], upsert_conflict: str = None) -> None:
    h = dict(HEADERS)
    if upsert_conflict:
        h["Prefer"] = "resolution=merge-duplicates,return=minimal"
    else:
        h["Prefer"] = "return=minimal"
    url = f"{BASE}/{table}"
    res = requests.post(url, headers=h, json=data, timeout=30)
    if not res.ok:
        logger.error("Supabase POST %s failed: %s %s", table, res.status_code, res.text[:500])
    res.raise_for_status()


def _delete(table: str, params: str) -> None:
    url = f"{BASE}/{table}?{params}"
    res = requests.delete(url, headers=HEADERS, timeout=15)
    res.raise_for_status()


# ── Model Projections ──
def get_cached_projections() -> List[Dict]:
    now = datetime.now(timezone.utc).isoformat()
    data = _get("am_projections", f"expires_at=gte.{now}&select=model_data&order=created_at.desc&limit=50")
    return [json.loads(r["model_data"]) if isinstance(r["model_data"], str) else r["model_data"] for r in data]


def save_projections(rows: List[Dict], summary: Dict, metrics: Dict) -> None:
    # Clear expired
    now = datetime.now(timezone.utc).isoformat()
    try:
        _delete("am_projections", f"expires_at=lt.{now}")
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

    for i in range(0, len(records), 50):
        _post("am_projections", records[i:i+50])
    logger.info("Saved %d projections to Supabase", len(records))


# ── Historical Game Logs ──
def get_cached_game_logs(player_name: str, season: str) -> Optional[List[Dict]]:
    data = _get("am_game_logs", f"player_name=eq.{player_name}&season=eq.{season}&order=game_date")
    return data if data else None


def save_game_logs(logs: List[Dict]) -> None:
    if not logs:
        return
    for i in range(0, len(logs), 50):
        try:
            _post("am_game_logs", logs[i:i+50], upsert_conflict="player_id,game_date")
        except Exception as e:
            logger.warning("Failed to save game logs batch: %s", e)


# ── Rebound Projections ──
def save_rebound_projections(rows: List[Dict]) -> None:
    if not rows:
        return
    now = datetime.now(timezone.utc).isoformat()
    try:
        _delete("rb_projections", f"expires_at=lt.{now}")
    except Exception:
        pass
    expires = (datetime.now(timezone.utc) + timedelta(hours=48)).isoformat()
    records = []
    for r in rows:
        records.append({
            "player": r.get("player"),
            "team": r.get("home_team", ""),
            "opponent": r.get("away_team", ""),
            "market_line": r.get("line"),
            "expected_reb": r.get("expected_reb"),
            "over_prob": r.get("over_prob"),
            "best_edge": r.get("best_edge"),
            "best_ev": r.get("best_ev"),
            "best_side": r.get("best_side"),
            "kelly_pct": r.get("kelly_pct"),
            "edge_under": r.get("edge_under"),
            "edge_over": r.get("edge_over"),
            "model_data": json.dumps(r),
            "expires_at": expires,
        })
    for i in range(0, len(records), 50):
        _post("rb_projections", records[i:i+50])
    logger.info("Saved %d rebound projections to Supabase", len(records))


# ── Backtest Results ──
def save_backtest_run(run_summary: Dict, results: List[Dict]) -> str:
    h = dict(HEADERS)
    h["Prefer"] = "return=representation"
    url = f"{BASE}/am_backtest_runs"
    res = requests.post(url, headers=h, json=[{
        "seasons": run_summary.get("seasons", []),
        "players_tested": run_summary.get("players_tested", 0),
        "total_predictions": run_summary.get("total_predictions", 0),
        "correct": run_summary.get("correct", 0),
        "hit_rate": run_summary.get("hit_rate", 0),
        "mae": run_summary.get("mae", 0),
        "avg_error": run_summary.get("avg_error", 0),
        "calibration": run_summary.get("calibration", {}),
        "line_source": run_summary.get("line_source", "synthetic"),
    }], timeout=15)
    res.raise_for_status()
    run_data = res.json()
    run_id = run_data[0]["id"] if run_data else "unknown"

    for r in results:
        r["run_id"] = run_id

    for i in range(0, len(results), 50):
        _post("am_backtest_results", results[i:i+50])

    logger.info("Saved backtest run %s with %d results", run_id, len(results))
    return run_id
