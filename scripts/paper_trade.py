#!/usr/bin/env python3
"""
Paper trading system for the standalone under signal.
Runs daily: fetches today's DK assist lines, generates predictions,
logs them to Supabase. No real bets — just tracking predictions vs outcomes.

Includes isotonic calibration on historical walk-forward predictions.

Usage:
  python scripts/paper_trade.py              # Generate today's predictions
  python scripts/paper_trade.py --resolve    # Resolve yesterday's predictions with actuals
  python scripts/paper_trade.py --report     # Print running performance report
"""
import sys, os, argparse, json, re, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from sklearn.isotonic import IsotonicRegression

from lib.config import logger
from lib.backtest_utils import (
    FEATURE_COLS, get_project_root, ensure_dirs,
    american_to_implied, american_to_decimal, payout_at_odds, STANDARD_ODDS,
)
from lib.db import save_run_log

ROOT = get_project_root()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jnirimzrhunjdtyvkhtt.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SB_HEADERS = {
    "apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json", "Prefer": "return=minimal",
}

BINARY_FEATURES = FEATURE_COLS + ["line_value", "pred_minus_line", "line_minus_l10"]


def norm(n):
    if not n: return ""
    s = str(n).lower().strip()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


# ── Isotonic Calibration ──
def build_isotonic_calibrator():
    """Train isotonic regression on historical walk-forward predictions.
    Returns (IsotonicRegression, brier_raw, brier_calibrated) or (None, None, None)."""
    path = ROOT / "data" / "reports" / "under_signal_predictions.parquet"
    if not path.exists():
        logger.warning("No historical predictions for calibration. Using raw probabilities.")
        return None, None, None

    df = pd.read_parquet(path)
    # Use first 70% for calibration training
    df = df.sort_values("game_date").reset_index(drop=True)
    split = int(len(df) * 0.7)
    train = df.iloc[:split]

    # Support both column name variants
    line_col = "synthetic_line" if "synthetic_line" in df.columns else "line"

    raw_under_probs = train["prob_under"].values
    actual_under = (train["actual_ast"] < train[line_col]).astype(float).values

    iso = IsotonicRegression(y_min=0.01, y_max=0.99)
    iso.fit(raw_under_probs, actual_under)

    # Verify calibration improvement on validation set
    val = df.iloc[split:]
    raw_brier = float(((val["prob_under"] - (val["actual_ast"] < val[line_col]).astype(float)) ** 2).mean())
    cal_probs = iso.predict(val["prob_under"].values)
    cal_brier = float(((cal_probs - (val["actual_ast"] < val[line_col]).astype(float)) ** 2).mean())
    logger.info("Isotonic calibration: raw Brier=%.4f → calibrated Brier=%.4f", raw_brier, cal_brier)

    return iso, raw_brier, cal_brier


# ── Load Model ──
def load_model():
    """Load the trained binary classifier."""
    model_path = ROOT / "models" / "binary_classifier.json"
    if not model_path.exists():
        logger.error("No binary classifier found. Run binary_classifier.py first.")
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


# ── Fetch Today's Lines ──
def fetch_todays_lines():
    """Fetch current DK player_assists lines from live Odds API.
    Returns (DataFrame, num_events)."""
    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY not set")
        return pd.DataFrame(), 0

    # Get events
    r = requests.get(
        "https://api.the-odds-api.com/v4/sports/basketball_nba/events",
        params={"apiKey": ODDS_API_KEY}, timeout=10)
    if r.status_code != 200:
        logger.error("Events fetch failed: %s", r.text[:200])
        return pd.DataFrame(), 0

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
                "markets": "player_assists", "bookmakers": "draftkings",
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
    logger.info("Fetched %d player assist lines", len(df))
    return df, len(events)


# ── Build Features for Today ──
def build_today_features(lines_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature vector for each player using nba_api game logs."""
    from nba_api.stats.endpoints import playergamelogs, leaguedashteamstats
    from nba_api.stats.static import players

    season = _current_season()
    results = []

    for _, row in lines_df.iterrows():
        player_name = row["player"]
        try:
            matches = players.find_players_by_full_name(player_name)
            if not matches:
                continue
            pid = matches[0]["id"]

            time.sleep(0.6)
            logs = playergamelogs.PlayerGameLogs(
                player_id_nullable=pid, season_nullable=season
            ).get_data_frames()[0]

            if logs.empty or len(logs) < 5:
                continue

            logs = logs.copy()
            logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
            logs["MIN_FLOAT"] = logs["MIN"].apply(_safe_min)
            logs["AST"] = pd.to_numeric(logs["AST"], errors="coerce").fillna(0)
            logs = logs[logs["MIN_FLOAT"] > 0].sort_values("GAME_DATE")

            if len(logs) < 5:
                continue

            # Compute features from PRIOR games only
            features = _compute_features(logs, row["line"])
            if features is None:
                continue

            features["player"] = player_name
            features["player_id"] = pid
            features["line"] = row["line"]
            features["over_price"] = row.get("over_price")
            features["under_price"] = row.get("under_price")
            features["home_team"] = row.get("home_team", "")
            features["away_team"] = row.get("away_team", "")
            results.append(features)

        except Exception as e:
            logger.warning("Failed for %s: %s", player_name, str(e)[:100])
            continue

    return pd.DataFrame(results)


def _current_season():
    now = datetime.now()
    if now.month >= 10:
        return f"{now.year}-{(now.year+1)%100:02d}"
    return f"{now.year-1}-{now.year%100:02d}"


def _safe_min(v):
    if pd.isna(v): return 0.0
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip()
    if ":" in s:
        try:
            parts = s.split(":")
            return float(parts[0]) + float(parts[1]) / 60
        except: return 0.0
    try: return float(s)
    except: return 0.0


def _compute_features(logs: pd.DataFrame, line: float,
                      xgb_regressor=None) -> dict:
    """Compute all 18 features from game logs (prior games only)."""
    if len(logs) < 5:
        return None

    ast = logs["AST"].values
    mins = logs["MIN_FLOAT"].values
    pts = pd.to_numeric(logs.get("PTS", 0), errors="coerce").fillna(0).values
    fga = pd.to_numeric(logs.get("FGA", 0), errors="coerce").fillna(0).values
    tov = pd.to_numeric(logs.get("TOV", 0), errors="coerce").fillna(0).values

    # Weighted L5 minutes
    l5_min = mins[-5:]
    w = np.array([0.08, 0.12, 0.20, 0.25, 0.35])[-len(l5_min):]
    w = w / w.sum()
    proj_min = np.dot(l5_min, w)

    def rate(num, den, n):
        n_arr, d_arr = num[-n:], den[-n:]
        return n_arr.sum() / d_arr.sum() if d_arr.sum() > 0 else 0

    ast_pm_l5 = rate(ast, mins, 5)
    ast_pm_l10 = rate(ast, mins, min(10, len(ast)))
    ast_pm_season = ast.sum() / mins.sum() if mins.sum() > 0 else 0
    ast_std = np.std(ast[-10:], ddof=1) if len(ast) >= 5 else 0
    fga_pm_l5 = rate(fga, mins, 5)
    pts_pm_l5 = rate(pts, mins, 5)
    tov_pm_l5 = rate(tov, mins, 5)

    # Rest days
    dates = logs["GAME_DATE"].values
    if len(dates) >= 2:
        rest = (pd.Timestamp.now() - pd.Timestamp(dates[-1])).days
    else:
        rest = 7

    matchup = str(logs.iloc[-1].get("MATCHUP", ""))
    is_home = 0 if "@" in matchup else 1

    # Build regression features first (needed for pred_minus_line)
    reg_features = {
        "proj_minutes": round(proj_min, 1),
        "ast_per_min_l5": round(ast_pm_l5, 4),
        "ast_per_min_l10": round(ast_pm_l10, 4),
        "ast_per_min_season": round(ast_pm_season, 4),
        "ast_std_l10": round(ast_std, 2),
        "fga_per_min_l5": round(fga_pm_l5, 4),
        "pts_per_min_l5": round(pts_pm_l5, 4),
        "tov_per_min_l5": round(tov_pm_l5, 4),
        "team_pace": 100.0,
        "opp_pace": 100.0,
        "opp_ast_allowed": 25.0,
        "rest_days": rest,
        "is_home": is_home,
        "b2b_flag": 1 if rest <= 1 else 0,
        "games_played_season": len(logs),
    }

    # pred_minus_line: use XGBoost regressor if available, else linear estimate
    if xgb_regressor is not None:
        try:
            X_reg = np.array([[reg_features[c] for c in FEATURE_COLS]])
            xgb_pred = xgb_regressor.predict(X_reg)[0]
            pred_minus = xgb_pred - line
        except Exception:
            pred_minus = ast_pm_season * proj_min - line
    else:
        pred_minus = ast_pm_season * proj_min - line

    reg_features.update({
        "line_value": line,
        "pred_minus_line": round(pred_minus, 3),
        "line_minus_l10": line - ast_pm_l10 * proj_min,
    })

    return reg_features


# ── Save Predictions to Supabase ──
def save_predictions(predictions: list) -> None:
    """Save paper trade predictions to am_paper_trades table."""
    if not predictions:
        return
    for i in range(0, len(predictions), 50):
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/am_paper_trades",
            headers=SB_HEADERS, json=predictions[i:i+50], timeout=15)
        if not r.ok:
            logger.error("Failed to save predictions: %s", r.text[:200])


def resolve_predictions() -> None:
    """Match yesterday's predictions with actual game results."""
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch unresolved predictions
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/am_paper_trades",
        params={"prediction_date": f"eq.{yesterday}", "resolved": "eq.false",
                "select": "*"},
        headers=SB_HEADERS, timeout=15)
    if not r.ok or not r.json():
        logger.info("No unresolved predictions for %s", yesterday)
        save_run_log("assists", "resolve", yesterday, predictions_resolved=0)
        return

    preds = r.json()
    logger.info("Resolving %d predictions for %s", len(preds), yesterday)

    # Fetch actual results from nba_api
    from nba_api.stats.endpoints import playergamelogs
    for pred in preds:
        try:
            pid = pred.get("player_id")
            if not pid:
                continue

            time.sleep(0.6)
            logs = playergamelogs.PlayerGameLogs(
                player_id_nullable=pid, season_nullable=_current_season()
            ).get_data_frames()[0]

            if logs.empty:
                continue

            logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
            game = logs[logs["GAME_DATE"].dt.strftime("%Y-%m-%d") == yesterday]

            if game.empty:
                continue

            actual_ast = int(game.iloc[0]["AST"])
            line = pred["line"]
            under_hit = actual_ast < line
            under_price = pred.get("under_price", -110)
            dec = american_to_decimal(under_price) if under_price else 1.91
            pnl = 100 * (dec - 1) if under_hit else -100

            # Update in Supabase
            update = {
                "actual_ast": actual_ast,
                "under_hit": under_hit,
                "pnl": round(pnl, 2),
                "resolved": True,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
            requests.patch(
                f"{SUPABASE_URL}/rest/v1/am_paper_trades",
                params={"id": f"eq.{pred['id']}"},
                headers={**SB_HEADERS, "Prefer": "return=minimal"},
                json=update, timeout=15)

        except Exception as e:
            logger.warning("Failed to resolve %s: %s", pred.get("player"), str(e)[:100])

    logger.info("Resolution complete for %s", yesterday)
    save_run_log("assists", "resolve", yesterday, predictions_resolved=len(preds))


def print_report() -> None:
    """Print running paper trade performance."""
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/am_paper_trades",
        params={"resolved": "eq.true", "bet_placed": "eq.true", "select": "*",
                "order": "prediction_date"},
        headers=SB_HEADERS, timeout=15)

    if not r.ok or not r.json():
        print("No resolved paper trades yet.")
        return

    trades = pd.DataFrame(r.json())
    n = len(trades)
    wins = trades["under_hit"].sum()
    hr = wins / n * 100
    total_pnl = trades["pnl"].sum()
    roi = total_pnl / (n * 100) * 100

    print(f"\n{'='*60}")
    print(f"  PAPER TRADE REPORT")
    print(f"{'='*60}")
    print(f"  Predictions resolved: {n}")
    print(f"  Bets placed (under 60%+): {(trades['bet_placed']).sum()}")
    print(f"  Hit rate: {hr:.1f}%")
    print(f"  Total PnL: ${total_pnl:.2f}")
    print(f"  ROI: {roi:+.1f}%")
    print(f"  Date range: {trades['prediction_date'].min()} to {trades['prediction_date'].max()}")
    print(f"  Days active: {trades['prediction_date'].nunique()}")


# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolve", action="store_true", help="Resolve yesterday's predictions")
    parser.add_argument("--report", action="store_true", help="Print performance report")
    parser.add_argument("--threshold", type=float, default=0.60, help="Under confidence threshold")
    args = parser.parse_args()

    ensure_dirs()

    if args.report:
        print_report()
        return

    if args.resolve:
        resolve_predictions()
        return

    # ── Generate today's predictions ──
    logger.info("Paper trading — generating predictions for today")

    model = load_model()
    if model is None:
        return

    iso, brier_raw, brier_cal = build_isotonic_calibrator()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines, num_events = fetch_todays_lines()
    if lines.empty:
        logger.info("No lines available (no games today?)")
        save_run_log("assists", "predict", today, events_found=num_events, lines_fetched=0,
                     predictions_saved=0, bets_placed=0)
        return

    features = build_today_features(lines)
    if features.empty:
        logger.info("No features built (all players failed)")
        save_run_log("assists", "predict", today, events_found=num_events,
                     lines_fetched=len(lines), predictions_saved=0, bets_placed=0,
                     status="error", error_msg="All player feature builds failed")
        return

    # Predict
    X = features[BINARY_FEATURES].values
    raw_probs = model.predict_proba(X)
    raw_over = raw_probs[:, 1]
    raw_under = 1 - raw_over

    # Isotonic calibration
    if iso is not None:
        cal_under = iso.predict(raw_under)
    else:
        cal_under = raw_under

    features["raw_prob_under"] = raw_under
    features["calibrated_prob_under"] = cal_under
    features["bet_placed"] = cal_under >= args.threshold

    predictions = []
    for _, row in features.iterrows():
        under_price = row.get("under_price")
        dk_implied = american_to_implied(under_price) if under_price else 0.5
        cal_prob = float(row["calibrated_prob_under"])
        edge = (cal_prob - dk_implied) * 100

        predictions.append({
            "prediction_date": today,
            "player": row["player"],
            "player_id": int(row["player_id"]),
            "line": float(row["line"]),
            "over_price": row.get("over_price"),
            "under_price": under_price,
            "raw_prob_under": round(float(row["raw_prob_under"]), 4),
            "calibrated_prob_under": round(cal_prob, 4),
            "dk_implied_under": round(float(dk_implied), 4),
            "edge_under": round(edge, 2),
            "bet_placed": bool(row["bet_placed"]),
            "resolved": False,
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
        })

    bets = [p for p in predictions if p["bet_placed"]]
    logger.info("Predictions: %d total, %d bets (under ≥%.0f%%)",
                len(predictions), len(bets), args.threshold * 100)

    for b in bets:
        logger.info("  BET: %s Under %.1f @ %s (prob: %.1f%% raw, %.1f%% calibrated)",
                    b["player"], b["line"], b.get("under_price", "?"),
                    b["raw_prob_under"] * 100, b["calibrated_prob_under"] * 100)

    save_predictions(predictions)
    logger.info("Saved %d predictions to Supabase", len(predictions))

    max_prob = max(p["calibrated_prob_under"] for p in predictions) if predictions else None
    save_run_log("assists", "predict", today,
                 events_found=num_events,
                 lines_fetched=len(lines),
                 predictions_saved=len(predictions),
                 bets_placed=len(bets),
                 max_prob_under=round(max_prob, 4) if max_prob else None,
                 brier_raw=round(brier_raw, 4) if brier_raw else None,
                 brier_calibrated=round(brier_cal, 4) if brier_cal else None)


if __name__ == "__main__":
    main()
