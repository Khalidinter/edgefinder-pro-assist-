#!/usr/bin/env python3
"""
Rebound paper trading — under-only at 8% edge threshold.
Runs daily: fetches today's DK rebound lines, generates predictions,
logs them to Supabase rb_paper_trades. No real bets.

Usage:
  python scripts/rebound_paper_trade.py              # Generate today's predictions
  python scripts/rebound_paper_trade.py --resolve    # Resolve yesterday's predictions
  python scripts/rebound_paper_trade.py --report     # Print running performance
"""
import sys, os, argparse, json, re, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

from lib.config import logger
from lib.backtest_utils import (
    get_project_root, ensure_dirs,
    american_to_implied, american_to_decimal, payout_at_odds,
)
from lib.db import save_run_log
from lib.rebound_config import BINARY_FEATURE_COLS
from lib.game_logs_cache import (
    save_player_logs, load_player_logs, cache_age,
    cached_player_count, MAX_CACHE_AGE,
)
from scripts.rebound_feature_engineering import DK_TO_NBA_NAME_ALIAS

ROOT = get_project_root()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jnirimzrhunjdtyvkhtt.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "SUPABASE_SERVICE_ROLE_KEY is not set. Required for rb_paper_trades writes. "
        "Set it in .env (local) or GitHub Actions secrets (CI). "
        "Get the key from Supabase dashboard → Settings → API → service_role."
    )
SB_HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json", "Prefer": "return=minimal",
}

EDGE_THRESHOLD = 8.0  # Under-only, 8% edge minimum

# Reverse alias: NBA API name → DK name (for matching)
NBA_TO_DK_ALIAS = {v: k for k, v in DK_TO_NBA_NAME_ALIAS.items()}


def _current_season():
    now = datetime.now()
    if now.month >= 10:
        return f"{now.year}-{(now.year + 1) % 100:02d}"
    return f"{now.year - 1}-{now.year % 100:02d}"


def _safe_min(v):
    if pd.isna(v):
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if ":" in s:
        try:
            parts = s.split(":")
            return float(parts[0]) + float(parts[1]) / 60
        except:
            return 0.0
    try:
        return float(s)
    except:
        return 0.0


def norm_name(n):
    """Normalize player name for matching (mirrors paper_trade.norm)."""
    if not n:
        return ""
    s = str(n).lower().strip()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def build_opp_team_lookup():
    """Aggregate historical game logs to team-game totals, compute rolling L10 with shift(1).

    Mirrors the training logic in scripts/rebound_feature_engineering.py so the
    live inference features match what the classifier was trained on.

    Returns dict keyed by team_abbr -> latest (fga_l10, fg_pct_l10, oreb_rate_l10, team_fga_l5).
    Falls back to None if the game logs parquet is missing.
    """
    gl_path = ROOT / "data" / "raw" / "all_game_logs.parquet"
    if not gl_path.exists():
        logger.warning("all_game_logs.parquet missing - opp_*_l10 features will fall back to defaults")
        return None

    logs = pd.read_parquet(gl_path)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])

    team_game = logs.groupby(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]).agg(
        T_FGA=("FGA", "sum"),
        T_FGM=("FGM", "sum"),
        T_REB=("REB", "sum"),
        T_OREB=("OREB", "sum"),
    ).reset_index()

    team_game["T_FG_PCT"] = team_game["T_FGM"] / team_game["T_FGA"].replace(0, np.nan)
    team_game["T_OREB_RATE"] = team_game["T_OREB"] / team_game["T_REB"].replace(0, np.nan)
    team_game = team_game.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    tg = team_game.groupby("TEAM_ABBREVIATION")
    team_game["_fga_l10"] = tg["T_FGA"].transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    team_game["_fg_pct_l10"] = tg["T_FG_PCT"].transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    team_game["_oreb_rate_l10"] = tg["T_OREB_RATE"].transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    team_game["_team_fga_l5"] = tg["T_FGA"].transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())

    # Latest available row per team (most-recent game)
    latest = team_game.sort_values("GAME_DATE").groupby("TEAM_ABBREVIATION").tail(1)
    lookup = {}
    for _, r in latest.iterrows():
        lookup[r["TEAM_ABBREVIATION"]] = {
            "opp_fga_l10": float(r["_fga_l10"]) if pd.notna(r["_fga_l10"]) else 85.0,
            "opp_fg_pct_l10": float(r["_fg_pct_l10"]) if pd.notna(r["_fg_pct_l10"]) else 0.46,
            "opp_oreb_rate_l10": float(r["_oreb_rate_l10"]) if pd.notna(r["_oreb_rate_l10"]) else 0.25,
            "team_fga_l5": float(r["_team_fga_l5"]) if pd.notna(r["_team_fga_l5"]) else 85.0,
        }
    logger.info("Built opponent L10 lookup for %d teams", len(lookup))
    return lookup


# ── NBA API Reachability Probe ──
def _nba_api_reachable(timeout: int = 10) -> bool:
    """Fast probe of stats.nba.com before starting a full refresh (S4 fix).

    Uses the ``nba_api`` library directly so the probe's header set matches the
    real downstream calls. A hand-rolled ``requests.get`` with a subset of
    headers trips the WAF (silent read timeout) even when the library itself
    would succeed — false negative, aborting the whole pipeline for nothing.

    If the downstream API really is unreachable, ``playergamelogs`` calls will
    hang for minutes per player — aborting early with a clean error beats
    spinning for 20 min. Returns True if the server answered within ``timeout``.
    """
    try:
        from nba_api.stats.endpoints import commonallplayers
        df = commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
            league_id="00",
            season=_current_season(),
            timeout=timeout,
        ).get_data_frames()[0]
        return not df.empty
    except Exception:
        return False


# ── Load Model ──

def load_model():
    model_path = ROOT / "models" / "rebound_binary_classifier.json"
    if not model_path.exists():
        logger.error("No rebound binary classifier found. Run rebound_binary_classifier.py first.")
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


# ── Fetch Today's Lines ──

def fetch_todays_rebound_lines() -> pd.DataFrame:
    """Fetch current DK player_rebounds lines from live Odds API."""
    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY not set")
        return pd.DataFrame()

    r = requests.get(
        "https://api.the-odds-api.com/v4/sports/basketball_nba/events",
        params={"apiKey": ODDS_API_KEY}, timeout=10)
    if r.status_code != 200:
        logger.error("Events fetch failed: %s", r.text[:200])
        return pd.DataFrame()

    events = r.json()
    logger.info("Found %d NBA events", len(events))

    rows = []
    for event in events:
        eid = event.get("id")
        if not eid:
            continue

        r2 = requests.get(
            f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{eid}/odds",
            params={
                "apiKey": ODDS_API_KEY, "regions": "us",
                "markets": "player_rebounds", "bookmakers": "draftkings",
                "oddsFormat": "american",
            }, timeout=10)

        if r2.status_code != 200:
            continue

        data = r2.json()
        home = data.get("home_team", "")
        away = data.get("away_team", "")

        for book in data.get("bookmakers", []):
            for market in book.get("markets", []):
                player_lines = {}
                for outcome in market.get("outcomes", []):
                    desc = outcome.get("description", "")
                    side = outcome.get("name", "")
                    line = outcome.get("point")
                    price = outcome.get("price")
                    if not desc or line is None:
                        continue
                    key = (desc, line)
                    if key not in player_lines:
                        player_lines[key] = {
                            "player": desc, "line": line,
                            "home_team": home, "away_team": away,
                            "event_id": eid,
                            "game_time": event.get("commence_time"),
                        }
                    if side == "Over":
                        player_lines[key]["over_price"] = price
                    elif side == "Under":
                        player_lines[key]["under_price"] = price

                for pl in player_lines.values():
                    if "over_price" in pl and "under_price" in pl:
                        rows.append(pl)

        time.sleep(0.3)

    df = pd.DataFrame(rows)
    logger.info("Fetched %d player rebound lines", len(df))
    return df


# ── Build Features for Today ──

def build_today_features(lines_df: pd.DataFrame, api_up: bool = True) -> pd.DataFrame:
    """Build 25-feature vector for each player using nba_api game logs.

    api_up: when False, skip live fetches and read from the local game-logs
    cache instead (S5 outage path).
    """
    from nba_api.stats.endpoints import playergamelogs
    from nba_api.stats.static import players

    season = _current_season()

    # Load team stats for pace/opp lookups
    teams = pd.read_parquet(ROOT / "data" / "raw" / "all_team_stats.parquet")
    pace_map = teams.set_index(["team_abbr", "season"])["pace"].to_dict()
    opp_reb_map = teams.set_index(["team_abbr", "season"])["opp_reb_allowed"].to_dict() \
        if "opp_reb_allowed" in teams.columns else {}

    # Real opponent rolling L10 features (fixes S1-4: previously used player's own stats)
    opp_team_lookup = build_opp_team_lookup() or {}

    results = []

    for _, row in lines_df.iterrows():
        dk_name = row["player"]
        # Resolve DK name → NBA API name
        dk_lower = norm_name(dk_name)
        nba_name = DK_TO_NBA_NAME_ALIAS.get(dk_lower, dk_name)

        try:
            matches = players.find_players_by_full_name(nba_name)
            if not matches:
                # Try original DK name
                matches = players.find_players_by_full_name(dk_name)
            if not matches:
                continue
            pid = matches[0]["id"]

            if api_up:
                time.sleep(0.6)
                logs = playergamelogs.PlayerGameLogs(
                    player_id_nullable=pid, season_nullable=season,
                    timeout=10,
                ).get_data_frames()[0]
                if not logs.empty:
                    save_player_logs(pid, season, logs)
            else:
                cached = load_player_logs(pid, season)
                if cached is None or cached.empty:
                    continue
                logs = cached

            if logs.empty or len(logs) < 5:
                continue

            logs = logs.copy()
            logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
            logs["MIN_FLOAT"] = logs["MIN"].apply(_safe_min)
            for col in ["REB", "OREB", "DREB", "PTS", "FGA", "FGM", "TOV"]:
                if col in logs.columns:
                    logs[col] = pd.to_numeric(logs[col], errors="coerce").fillna(0)
            logs = logs[logs["MIN_FLOAT"] > 0].sort_values("GAME_DATE")

            if len(logs) < 5:
                continue

            features = _compute_rebound_features(
                logs, row["line"], row.get("over_price", -110),
                row.get("under_price", -110), season, pace_map, opp_reb_map,
                opp_team_lookup,
            )
            if features is None:
                continue

            features["player"] = dk_name
            features["player_id"] = pid
            features["line"] = row["line"]
            features["over_price"] = row.get("over_price")
            features["under_price"] = row.get("under_price")
            features["home_team"] = row.get("home_team", "")
            features["away_team"] = row.get("away_team", "")
            features["game_time"] = row.get("game_time")
            # Confidence for rebounds uses REB logs + projected count
            projected_reb = float(features.get("reb_per_min_season", 0.0)) * float(features.get("proj_minutes", 0.0))
            features["projected_reb"] = round(projected_reb, 2)
            features["confidence"] = _compute_rebound_confidence(logs, projected_reb, row["line"])
            results.append(features)

        except Exception as e:
            logger.warning("Failed for %s: %s", dk_name, str(e)[:100])
            continue

    return pd.DataFrame(results)


def _compute_rebound_features(
    logs: pd.DataFrame, line: float,
    over_price: float, under_price: float,
    season: str, pace_map: dict, opp_reb_map: dict,
    opp_team_lookup: dict = None,
) -> dict:
    """Compute all 25 binary classifier features from game logs (prior games only)."""
    if len(logs) < 5:
        return None

    reb = logs["REB"].values
    oreb = logs["OREB"].values if "OREB" in logs.columns else np.zeros(len(logs))
    dreb = logs["DREB"].values if "DREB" in logs.columns else np.zeros(len(logs))
    mins = logs["MIN_FLOAT"].values
    pts = logs["PTS"].values
    fga = logs["FGA"].values
    fgm = logs["FGM"].values if "FGM" in logs.columns else np.zeros(len(logs))
    tov = logs["TOV"].values

    # Weighted L5 minutes
    l5_min = mins[-5:]
    w = np.array([0.08, 0.12, 0.20, 0.25, 0.35])[-len(l5_min):]
    w = w / w.sum()
    proj_min = float(np.dot(l5_min, w))

    def rate(num, den, n):
        n_arr, d_arr = num[-n:], den[-n:]
        return float(n_arr.sum() / d_arr.sum()) if d_arr.sum() > 0 else 0.0

    reb_pm_l5 = rate(reb, mins, 5)
    reb_pm_l10 = rate(reb, mins, min(10, len(reb)))
    reb_pm_season = float(reb.sum() / mins.sum()) if mins.sum() > 0 else 0.0
    reb_std = float(np.std(reb[-10:], ddof=1)) if len(reb) >= 5 else 0.0

    # OREB/DREB share
    l5_reb_sum = float(reb[-5:].sum())
    oreb_share = float(oreb[-5:].sum() / l5_reb_sum) if l5_reb_sum > 0 else 0.0
    dreb_share = 1.0 - oreb_share

    fga_pm_l5 = rate(fga, mins, 5)
    pts_pm_l5 = rate(pts, mins, 5)
    tov_pm_l5 = rate(tov, mins, 5)

    # Team context
    last_row = logs.iloc[-1]
    team_abbr = str(last_row.get("TEAM_ABBREVIATION", ""))
    matchup = str(last_row.get("MATCHUP", ""))
    is_home = 0 if "@" in matchup else 1

    # Extract opponent
    parts = matchup.replace(" ", "").split("@" if "@" in matchup else "vs.")
    opp_abbr = ""
    for p in parts:
        p = p.strip()
        if p and p != team_abbr:
            opp_abbr = p
            break

    team_pace = pace_map.get((team_abbr, season), 100.0)
    opp_pace = pace_map.get((opp_abbr, season), 100.0)
    opp_reb_allowed = opp_reb_map.get((opp_abbr, season), 44.0)

    # Opponent rolling features — look up the OPPONENT team's trailing-L10,
    # not the player's own stats (S1-4 fix). Mirrors training pipeline.
    opp_stats = (opp_team_lookup or {}).get(opp_abbr, {})
    opp_fga_l10 = opp_stats.get("opp_fga_l10", 85.0)
    opp_fg_pct_l10 = opp_stats.get("opp_fg_pct_l10", 0.46)
    opp_oreb_rate_l10 = opp_stats.get("opp_oreb_rate_l10", 0.25)

    # team_fga_l5 is the player's OWN team's trailing-L5 FGA (team-level)
    own_stats = (opp_team_lookup or {}).get(team_abbr, {})
    team_fga_l5 = own_stats.get("team_fga_l5", 85.0)

    # Rest days
    dates = logs["GAME_DATE"].values
    if len(dates) >= 1:
        rest = (pd.Timestamp.now() - pd.Timestamp(dates[-1])).days
    else:
        rest = 7

    # DK-derived features
    dk_implied_over = american_to_implied(over_price) if over_price else 0.5
    pred_minus_line = reb_pm_l5 * proj_min - line

    return {
        "proj_minutes": round(proj_min, 1),
        "reb_per_min_l5": round(reb_pm_l5, 4),
        "reb_per_min_l10": round(reb_pm_l10, 4),
        "reb_per_min_season": round(reb_pm_season, 4),
        "reb_std_l10": round(reb_std, 2),
        "oreb_share_l5": round(oreb_share, 4),
        "dreb_share_l5": round(dreb_share, 4),
        "fga_per_min_l5": round(fga_pm_l5, 4),
        "pts_per_min_l5": round(pts_pm_l5, 4),
        "tov_per_min_l5": round(tov_pm_l5, 4),
        "team_pace": team_pace,
        "opp_pace": opp_pace,
        "opp_reb_allowed": opp_reb_allowed,
        "opp_fga_l10": round(opp_fga_l10, 1),
        "opp_fg_pct_l10": round(opp_fg_pct_l10, 4),
        "opp_oreb_rate_l10": round(opp_oreb_rate_l10, 4),
        "team_fga_l5": round(team_fga_l5, 1),
        "rest_days": min(rest, 7),
        "is_home": is_home,
        "b2b_flag": 1 if rest <= 1 else 0,
        "games_played_season": len(logs),
        "dk_line": line,
        "pred_minus_line": round(pred_minus_line, 2),
        "dk_implied_over_prob": round(dk_implied_over, 4),
        "dk_over_price": float(over_price) if over_price else -110.0,
        # Non-feature passthroughs (not used by the classifier, but consumed
        # by save_rebound_projections so away players get the right team/opp).
        "_team_abbr": team_abbr,
        "_opp_abbr": opp_abbr,
        # Enrichment for the unified rb_paper_trades row (S3-b fix)
        "team": team_abbr,
    }


def _compute_rebound_confidence(logs: pd.DataFrame, projected: float, line: float) -> str:
    """Confidence grade ported from lib/model.py:compute_confidence_grade, REB variant.

    Scores CV of REB, games played, recent-minutes stability, and line distance.
    Returns 'A' / 'B' / 'C' / 'D'.

    Defensive guards (S4 audit):
      - n < 1 → D
      - Missing REB / MIN_FLOAT column → D
      - NaN mean/std fall through to fallback branches (NaN > 0 is False)
    """
    n = len(logs)
    if n < 1:
        return "D"
    if "REB" not in logs.columns or "MIN_FLOAT" not in logs.columns:
        return "D"
    m = float(logs["REB"].mean())
    s = float(logs["REB"].std(ddof=1)) if n > 1 else m
    # NaN mean lands here via the (m > 0) guard → cv = 1.0 fallback.
    cv = (s / m) if (m > 0) else 1.0
    score = 0.0
    score += 35 if cv <= 0.25 else 28 if cv <= 0.40 else 18 if cv <= 0.55 else 10 if cv <= 0.70 else 5
    score += 25 if n >= 50 else 20 if n >= 30 else 12 if n >= 15 else 8 if n >= 10 else 3
    last10 = logs.tail(min(10, n))
    mm = float(last10["MIN_FLOAT"].mean()) if len(last10) > 0 else 0.0
    ms = float(last10["MIN_FLOAT"].std(ddof=1)) if len(last10) > 1 else 0.0
    mcv = (ms / mm) if mm > 0 else 1.0
    score += 20 if mcv <= 0.08 else 15 if mcv <= 0.15 else 10 if mcv <= 0.25 else 4
    dist = abs(projected - line)
    score += 15 if dist <= 0.5 else 12 if dist <= 1.0 else 7 if dist <= 2.0 else 3
    if score >= 75: return "A"
    if score >= 55: return "B"
    if score >= 35: return "C"
    return "D"


# ── Supabase I/O ──

def save_predictions(predictions: list) -> None:
    """Upsert predictions on (prediction_date, player_id).

    Idempotent — re-running the same day produces no duplicate rows, provided
    the rb_paper_trades table has a UNIQUE(prediction_date, player_id) constraint.
    """
    if not predictions:
        return
    # Merge-duplicates header turns POST into an upsert keyed by the table's
    # unique constraint. Without the DB-side constraint this still behaves as
    # a plain insert, so this change is safe to ship ahead of the migration.
    upsert_headers = {**SB_HEADERS, "Prefer": "resolution=merge-duplicates,return=minimal"}
    for i in range(0, len(predictions), 50):
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/rb_paper_trades",
            headers=upsert_headers, json=predictions[i:i + 50], timeout=15)
        if not r.ok:
            logger.error("Failed to save predictions: %s %s", r.status_code, r.text[:200])


def resolve_predictions() -> None:
    """Resolve ALL unresolved predictions with prediction_date <= yesterday.

    Previously this only checked `eq.yesterday`, so any bet missed by the cron
    (DNP, NBA API hiccup, workflow gap) stayed stuck forever. Now it sweeps
    every unresolved row on or before yesterday.

    DNP handling: if the player has game logs but none for the bet's
    prediction_date, mark the row resolved with `void_reason='dnp'`, pnl=0,
    under_hit=NULL (stake returned).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/rb_paper_trades",
        params={"prediction_date": f"lte.{cutoff}", "resolved": "eq.false",
                "select": "*", "order": "prediction_date.asc"},
        headers=SB_HEADERS, timeout=15)
    if not r.ok or not r.json():
        logger.info("No unresolved rebound predictions with prediction_date <= %s", cutoff)
        save_run_log("rebounds", "resolve", cutoff, predictions_resolved=0)
        return

    preds = r.json()
    logger.info("Resolving %d rebound predictions (prediction_date <= %s)", len(preds), cutoff)

    resolved_count = 0
    voided_count = 0

    # Cache NBA API game-log fetches per player — same player can have bets on
    # multiple prediction_dates and we don't want to re-hit the API per bet.
    logs_cache: dict = {}

    from nba_api.stats.endpoints import playergamelogs
    for pred in preds:
        try:
            pid = pred.get("player_id")
            if not pid:
                continue
            pred_date = pred["prediction_date"]

            if pid not in logs_cache:
                # Pull Regular Season + PlayIn + Playoffs and concat so bets in
                # any phase of the season can resolve. Without this, anything
                # after the regular-season end date looks like a DNP.
                frames = []
                for st in ("Regular Season", "PlayIn", "Playoffs"):
                    try:
                        time.sleep(0.6)
                        df = playergamelogs.PlayerGameLogs(
                            player_id_nullable=pid,
                            season_nullable=_current_season(),
                            season_type_nullable=st,
                            timeout=10,
                        ).get_data_frames()[0]
                        if not df.empty:
                            frames.append(df)
                    except Exception as e:
                        logger.warning("PlayerGameLogs %s failed for %s: %s", st, pid, e)
                logs_cache[pid] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            logs = logs_cache[pid]

            if logs.empty:
                # No logs at all for this player this season — probably a
                # roster/ID issue, not a DNP. Skip for manual review.
                continue

            logs = logs.copy()
            logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
            game = logs[logs["GAME_DATE"].dt.strftime("%Y-%m-%d") == pred_date]
            if game.empty:
                # DNP: player has logs but none on pred_date. Void the bet
                # (stake returned, pnl=0, hit=null) so it doesn't stay stuck.
                update = {
                    "actual_reb": None,
                    "under_hit": None,
                    "pnl": 0.0,
                    "resolved": True,
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                    "void_reason": "dnp",
                }
                pr = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/rb_paper_trades",
                    params={"id": f"eq.{pred['id']}"},
                    headers={**SB_HEADERS, "Prefer": "return=minimal"},
                    json=update, timeout=15)
                if not pr.ok:
                    logger.error("Void PATCH failed for %s (%s): %s %s",
                                 pred.get("player"), pred_date, pr.status_code, pr.text[:200])
                else:
                    resolved_count += 1
                    voided_count += 1
                continue

            actual_reb = int(game.iloc[0]["REB"])
            line = pred["line"]
            under_hit = actual_reb < line
            under_price = pred.get("under_price", -110)
            # Only compute PnL for rows we actually "bet"; non-bets stay at 0.0
            pnl = payout_at_odds(under_price, 100, under_hit) if pred.get("bet_placed") else 0.0

            update = {
                "actual_reb": actual_reb,
                "under_hit": under_hit,
                "pnl": round(pnl, 2),
                "resolved": True,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
            pr = requests.patch(
                f"{SUPABASE_URL}/rest/v1/rb_paper_trades",
                params={"id": f"eq.{pred['id']}"},
                headers={**SB_HEADERS, "Prefer": "return=minimal"},
                json=update, timeout=15)
            if not pr.ok:
                logger.error("PATCH failed for %s: %s %s", pred.get("player"), pr.status_code, pr.text[:200])
            else:
                resolved_count += 1

        except Exception as e:
            logger.warning("Failed to resolve %s: %s", pred.get("player"), str(e)[:100])

    logger.info("Rebound resolution complete (<= %s): %d resolved (%d voided) of %d",
                cutoff, resolved_count, voided_count, len(preds))
    save_run_log("rebounds", "resolve", cutoff,
                 predictions_resolved=resolved_count)


def print_report() -> None:
    """Print running rebound paper trade performance."""
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/rb_paper_trades",
        params={"resolved": "eq.true", "bet_placed": "eq.true", "select": "*",
                "order": "prediction_date"},
        headers=SB_HEADERS, timeout=15)

    if not r.ok or not r.json():
        print("No resolved rebound paper trades yet.")
        return

    trades = pd.DataFrame(r.json())
    n = len(trades)
    wins = trades["under_hit"].sum()
    hr = wins / n * 100 if n > 0 else 0
    total_pnl = trades["pnl"].sum()
    roi = total_pnl / (n * 100) * 100 if n > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  REBOUND PAPER TRADE REPORT (Under-only, 8% edge)")
    print(f"{'=' * 60}")
    print(f"  Days active:     {trades['prediction_date'].nunique()}")
    print(f"  Bets placed:     {n}")
    print(f"  Hit rate:        {hr:.1f}%")
    print(f"  Total PnL:       ${total_pnl:+.2f}")
    print(f"  ROI:             {roi:+.1f}%")
    print(f"  Date range:      {trades['prediction_date'].min()} to {trades['prediction_date'].max()}")
    print(f"{'=' * 60}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Rebound paper trading (under-only, 8% edge)")
    parser.add_argument("--resolve", action="store_true", help="Resolve yesterday's predictions")
    parser.add_argument("--report", action="store_true", help="Print performance report")
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD)
    args = parser.parse_args()

    ensure_dirs()

    if args.report:
        print_report()
        return

    if args.resolve:
        resolve_predictions()
        return

    # ── Generate today's predictions ──
    logger.info("Rebound paper trading — generating predictions for today")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Fast pre-check: if stats.nba.com is unreachable, fall back to cached game
    # logs rather than abort entirely (S4 fix, S5 fallback). Refresh stays
    # viable — predictions just miss last night's box scores. Resolve still
    # requires live API.
    api_up = _nba_api_reachable()
    if not api_up:
        age = cache_age()
        if age is None:
            logger.error("NBA API unreachable and no game-logs cache exists — aborting")
            save_run_log("rebounds", "predict", today, status="error",
                         error_msg="NBA API unreachable; no cache")
            sys.exit(2)
        if age > MAX_CACHE_AGE:
            hours = age.total_seconds() / 3600
            logger.error("NBA API unreachable and cache is %.1fh old (max %.0fh) — aborting",
                         hours, MAX_CACHE_AGE.total_seconds() / 3600)
            save_run_log("rebounds", "predict", today, status="error",
                         error_msg=f"NBA API unreachable; cache stale ({hours:.1f}h)")
            sys.exit(2)
        hours = age.total_seconds() / 3600
        logger.warning(
            "NBA API unreachable — using cached game logs (%.1fh old, %d players). "
            "Predictions will miss any games played in the last %.1fh.",
            hours, cached_player_count(), hours,
        )

    model = load_model()
    if model is None:
        save_run_log("rebounds", "predict", today,
                     status="error", error_msg="Rebound binary classifier not found")
        return

    lines = fetch_todays_rebound_lines()
    num_events = int(lines["event_id"].nunique()) if not lines.empty else 0
    if lines.empty:
        logger.info("No rebound lines available (no games today?)")
        save_run_log("rebounds", "predict", today,
                     events_found=num_events, lines_fetched=0,
                     predictions_saved=0, bets_placed=0)
        return

    features = build_today_features(lines, api_up=api_up)
    if features.empty:
        logger.info("No features built (all players failed)")
        save_run_log("rebounds", "predict", today,
                     events_found=num_events, lines_fetched=len(lines),
                     predictions_saved=0, bets_placed=0,
                     status="error", error_msg="All player feature builds failed")
        return

    # Predict P(over)
    X = features[BINARY_FEATURE_COLS].values
    over_probs = model.predict_proba(X)[:, 1]
    under_probs = 1.0 - over_probs

    predictions = []

    for i, (_, row) in enumerate(features.iterrows()):
        over_prob = float(over_probs[i])
        under_prob = 1.0 - over_prob
        over_price = row.get("over_price")
        under_price = row.get("under_price")
        dk_implied_over = american_to_implied(over_price) if over_price else 0.5
        dk_implied_under = american_to_implied(under_price) if under_price else 0.5
        edge_over = (over_prob - dk_implied_over) * 100
        edge_under = (under_prob - dk_implied_under) * 100
        bet_placed = edge_under >= args.edge_threshold

        # ── Unified enrichment: Kelly + best-side + model EV (S3-b fix) ──
        d_over = american_to_decimal(over_price) if over_price else 1.91
        d_under = american_to_decimal(under_price) if under_price else 1.91
        ev_over = (over_prob * d_over - 1) * 100
        ev_under = (under_prob * d_under - 1) * 100

        if edge_over >= edge_under:
            best_side, best_edge, best_ev, best_decimal = "OVER", edge_over, ev_over, d_over
        else:
            best_side, best_edge, best_ev, best_decimal = "UNDER", edge_under, ev_under, d_under

        # Quarter-Kelly, capped at 5% of bankroll
        kelly_raw = (best_edge / 100.0) / (best_decimal - 1) if best_decimal > 1 else 0.0
        kelly_pct = round(max(0.0, min(kelly_raw / 4.0, 0.05)) * 100, 2)

        gt = row.get("game_time")
        if gt is not None and not (isinstance(gt, float) and math.isnan(gt)):
            game_time = str(gt)
        else:
            game_time = None

        # Stamp prediction_date with the actual game date (in NBA's ET clock),
        # not "today". The odds API publishes lines a day or two early; a
        # prediction generated today may be for a game tomorrow. Resolution
        # matches on prediction_date == GAME_DATE (which stats.nba.com reports
        # in ET), so stamping `today` falsely VOIDs off-day predictions as
        # DNP (root cause of the 04-20/04-21 voids).
        pred_date_for_row = today
        if game_time:
            try:
                from zoneinfo import ZoneInfo
                pred_date_for_row = datetime.fromisoformat(
                    game_time.replace("Z", "+00:00")
                ).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            except Exception:
                pass

        predictions.append({
            "prediction_date": pred_date_for_row,
            "player": row["player"],
            "player_id": int(row["player_id"]),
            "line": float(row["line"]),
            "over_price": int(over_price) if over_price else None,
            "under_price": int(under_price) if under_price else None,
            "model_over_prob": round(over_prob, 4),
            "dk_implied_over": round(dk_implied_over, 4),
            "edge_under": round(edge_under, 2),
            "bet_placed": bet_placed,
            "resolved": False,
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            # Unified fields (populated by add_projection_fields_to_paper_trades migration)
            "team": row.get("team") or None,
            "game_time": game_time,
            "best_side": best_side,
            "kelly_pct": kelly_pct,
            "confidence": row.get("confidence") or "C",
            "over_prob": round(over_prob, 4),
            "model_ev": round(float(best_ev), 2),
        })

    bets = [p for p in predictions if p["bet_placed"]]
    logger.info("Predictions: %d total, %d bets (under edge >= %.0f%%)",
                len(predictions), len(bets), args.edge_threshold)

    for b in bets:
        logger.info("  BET: %s Under %.1f @ %s (edge: %.1f%%)",
                    b["player"], b["line"], b.get("under_price", "?"), b["edge_under"])

    save_predictions(predictions)
    logger.info("Saved %d predictions to Supabase rb_paper_trades", len(predictions))

    save_run_log("rebounds", "predict", today,
                 events_found=num_events,
                 lines_fetched=len(lines),
                 predictions_saved=len(predictions),
                 bets_placed=len(bets))


if __name__ == "__main__":
    main()
