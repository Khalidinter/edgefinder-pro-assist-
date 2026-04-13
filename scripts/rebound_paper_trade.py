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
import sys, os, argparse, json, re, time
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
from lib.rebound_config import BINARY_FEATURE_COLS
from scripts.rebound_feature_engineering import DK_TO_NBA_NAME_ALIAS

ROOT = get_project_root()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jnirimzrhunjdtyvkhtt.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SB_HEADERS = {
    "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
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
    """Normalize player name for matching."""
    if not n:
        return ""
    return str(n).lower().strip()


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

def build_today_features(lines_df: pd.DataFrame) -> pd.DataFrame:
    """Build 25-feature vector for each player using nba_api game logs."""
    from nba_api.stats.endpoints import playergamelogs
    from nba_api.stats.static import players

    season = _current_season()

    # Load team stats for pace/opp lookups
    teams = pd.read_parquet(ROOT / "data" / "raw" / "all_team_stats.parquet")
    pace_map = teams.set_index(["team_abbr", "season"])["pace"].to_dict()
    opp_reb_map = teams.set_index(["team_abbr", "season"])["opp_reb_allowed"].to_dict() \
        if "opp_reb_allowed" in teams.columns else {}

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

            time.sleep(0.6)
            logs = playergamelogs.PlayerGameLogs(
                player_id_nullable=pid, season_nullable=season,
                timeout=10,
            ).get_data_frames()[0]

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
                row.get("under_price", -110), season, pace_map, opp_reb_map
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
            results.append(features)

        except Exception as e:
            logger.warning("Failed for %s: %s", dk_name, str(e)[:100])
            continue

    return pd.DataFrame(results)


def _compute_rebound_features(
    logs: pd.DataFrame, line: float,
    over_price: float, under_price: float,
    season: str, pace_map: dict, opp_reb_map: dict,
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

    # Opponent rolling features (approximate from logs if available)
    opp_fga_l10 = float(fga[-10:].mean()) if len(fga) >= 5 else 85.0
    opp_fg_pct_l10 = float(fgm[-10:].sum() / fga[-10:].sum()) if fga[-10:].sum() > 0 else 0.46
    opp_oreb_rate_l10 = float(oreb[-10:].sum() / reb[-10:].sum()) if reb[-10:].sum() > 0 else 0.25
    team_fga_l5 = float(fga[-5:].mean()) if len(fga) >= 3 else 85.0

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
    }


# ── Supabase I/O ──

def save_predictions(predictions: list) -> None:
    if not predictions:
        return
    for i in range(0, len(predictions), 50):
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/rb_paper_trades",
            headers=SB_HEADERS, json=predictions[i:i + 50], timeout=15)
        if not r.ok:
            logger.error("Failed to save predictions: %s", r.text[:200])


def resolve_predictions() -> None:
    """Match yesterday's predictions with actual rebound results."""
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/rb_paper_trades",
        params={"prediction_date": f"eq.{yesterday}", "resolved": "eq.false", "select": "*"},
        headers=SB_HEADERS, timeout=15)
    if not r.ok or not r.json():
        logger.info("No unresolved rebound predictions for %s", yesterday)
        return

    preds = r.json()
    logger.info("Resolving %d rebound predictions for %s", len(preds), yesterday)

    from nba_api.stats.endpoints import playergamelogs
    for pred in preds:
        try:
            pid = pred.get("player_id")
            if not pid:
                continue

            time.sleep(0.6)
            logs = playergamelogs.PlayerGameLogs(
                player_id_nullable=pid, season_nullable=_current_season(),
                timeout=10,
            ).get_data_frames()[0]

            if logs.empty:
                continue

            logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
            game = logs[logs["GAME_DATE"].dt.strftime("%Y-%m-%d") == yesterday]
            if game.empty:
                continue

            actual_reb = int(game.iloc[0]["REB"])
            line = pred["line"]
            under_hit = actual_reb < line
            under_price = pred.get("under_price", -110)
            pnl = payout_at_odds(under_price, 100, under_hit) if pred.get("bet_placed") else 0.0

            update = {
                "actual_reb": actual_reb,
                "under_hit": under_hit,
                "pnl": round(pnl, 2),
                "resolved": True,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/rb_paper_trades",
                params={"id": f"eq.{pred['id']}"},
                headers={**SB_HEADERS, "Prefer": "return=minimal"},
                json=update, timeout=15)

        except Exception as e:
            logger.warning("Failed to resolve %s: %s", pred.get("player"), str(e)[:100])

    logger.info("Rebound resolution complete for %s", yesterday)


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

    model = load_model()
    if model is None:
        return

    lines = fetch_todays_rebound_lines()
    if lines.empty:
        logger.info("No rebound lines available (no games today?)")
        return

    features = build_today_features(lines)
    if features.empty:
        logger.info("No features built (all players failed)")
        return

    # Predict P(over)
    X = features[BINARY_FEATURE_COLS].values
    over_probs = model.predict_proba(X)[:, 1]
    under_probs = 1.0 - over_probs

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    predictions = []

    for i, (_, row) in enumerate(features.iterrows()):
        over_prob = float(over_probs[i])
        dk_implied_over = american_to_implied(row["over_price"]) if row.get("over_price") else 0.5
        dk_implied_under = american_to_implied(row["under_price"]) if row.get("under_price") else 0.5
        edge_under = ((1.0 - over_prob) - dk_implied_under) * 100
        bet_placed = edge_under >= args.edge_threshold

        predictions.append({
            "prediction_date": today,
            "player": row["player"],
            "player_id": int(row["player_id"]),
            "line": float(row["line"]),
            "over_price": int(row["over_price"]) if row.get("over_price") else None,
            "under_price": int(row["under_price"]) if row.get("under_price") else None,
            "model_over_prob": round(over_prob, 4),
            "dk_implied_over": round(dk_implied_over, 4),
            "edge_under": round(edge_under, 2),
            "bet_placed": bet_placed,
            "resolved": False,
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
        })

    bets = [p for p in predictions if p["bet_placed"]]
    logger.info("Predictions: %d total, %d bets (under edge >= %.0f%%)",
                len(predictions), len(bets), args.edge_threshold)

    for b in bets:
        logger.info("  BET: %s Under %.1f @ %s (edge: %.1f%%)",
                    b["player"], b["line"], b.get("under_price", "?"), b["edge_under"])

    save_predictions(predictions)
    logger.info("Saved %d predictions to Supabase rb_paper_trades", len(predictions))


if __name__ == "__main__":
    main()
