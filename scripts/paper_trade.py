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
from lib.game_logs_cache import (
    save_player_logs, load_player_logs, cache_age, cache_is_fresh,
    cached_player_count, MAX_CACHE_AGE,
)

ROOT = get_project_root()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jnirimzrhunjdtyvkhtt.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "SUPABASE_SERVICE_ROLE_KEY is not set. Required for am_paper_trades writes. "
        "Set it in .env (local) or GitHub Actions secrets (CI). "
        "Get the key from Supabase dashboard → Settings → API → service_role."
    )
SB_HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
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


# ── Isotonic Calibration ──
def build_isotonic_calibrator():
    """Train isotonic regression on historical walk-forward predictions.

    Fits on ALL historical walk-forward predictions so production inference
    uses the maximum signal available. A separate 80/20 split is used only
    for logging Brier-score improvement (not for fitting the production model).

    Returns (IsotonicRegression, brier_raw, brier_calibrated) or (None, None, None).
    """
    path = ROOT / "data" / "reports" / "under_signal_predictions.parquet"
    if not path.exists():
        logger.warning("No historical predictions for calibration. Using raw probabilities.")
        return None, None, None

    df = pd.read_parquet(path).sort_values("game_date").reset_index(drop=True)
    # Support both column name variants
    line_col = "synthetic_line" if "synthetic_line" in df.columns else "line"
    actual_under_all = (df["actual_ast"] < df[line_col]).astype(float).values
    raw_all = df["prob_under"].values

    # Production calibrator: fit on all data
    iso = IsotonicRegression(y_min=0.01, y_max=0.99)
    iso.fit(raw_all, actual_under_all)

    # Diagnostic Brier scores on a hold-out split (fit-on-train, score-on-val)
    # so we can log whether calibration actually helps. This diag-only fit is
    # NOT the calibrator returned for production use.
    split = int(len(df) * 0.8)
    if split >= 100 and len(df) - split >= 50:
        iso_diag = IsotonicRegression(y_min=0.01, y_max=0.99).fit(
            raw_all[:split], actual_under_all[:split]
        )
        raw_brier = float(((raw_all[split:] - actual_under_all[split:]) ** 2).mean())
        cal_brier = float(((iso_diag.predict(raw_all[split:]) - actual_under_all[split:]) ** 2).mean())
        logger.info("Isotonic calibration diag: raw Brier=%.4f → calibrated Brier=%.4f", raw_brier, cal_brier)
    else:
        # Too little data for a meaningful split — report in-sample scores only.
        raw_brier = float(((raw_all - actual_under_all) ** 2).mean())
        cal_brier = float(((iso.predict(raw_all) - actual_under_all) ** 2).mean())
        logger.info("Isotonic calibration (in-sample): raw Brier=%.4f → calibrated Brier=%.4f", raw_brier, cal_brier)

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
    logger.info("Fetched %d player assist lines", len(df))
    return df, len(events)


# ── Build Features for Today ──
def build_today_features(lines_df: pd.DataFrame, api_up: bool = True) -> pd.DataFrame:
    """Build feature vector for each player using nba_api game logs.

    api_up: when False, the NBA API was judged unreachable upstream — skip live
    fetches and read from the local game-logs cache instead (S5 outage path).
    """
    from nba_api.stats.endpoints import playergamelogs, leaguedashteamstats
    from nba_api.stats.static import players

    season = _current_season()

    # Load real team pace / opponent assists allowed so the classifier sees
    # the same per-team context it was trained on. Falls back to league-average
    # constants if the parquet is missing (S1-3 fix).
    team_pace_map, opp_ast_map = {}, {}
    try:
        teams = pd.read_parquet(ROOT / "data" / "raw" / "all_team_stats.parquet")
        team_pace_map = teams.set_index(["team_abbr", "season"])["pace"].to_dict()
        if "opp_ast_allowed" in teams.columns:
            opp_ast_map = teams.set_index(["team_abbr", "season"])["opp_ast_allowed"].to_dict()
    except Exception as e:
        logger.warning("Failed to load team stats for pace lookup: %s", e)

    results = []

    for _, row in lines_df.iterrows():
        player_name = row["player"]
        try:
            matches = players.find_players_by_full_name(player_name)
            if not matches:
                continue
            pid = matches[0]["id"]

            if api_up:
                time.sleep(0.6)
                logs = playergamelogs.PlayerGameLogs(
                    player_id_nullable=pid, season_nullable=season
                ).get_data_frames()[0]
                # Cache every successful, non-trivial fetch so a future outage
                # has something to fall back on (S5).
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
            logs["AST"] = pd.to_numeric(logs["AST"], errors="coerce").fillna(0)
            logs = logs[logs["MIN_FLOAT"] > 0].sort_values("GAME_DATE")

            if len(logs) < 5:
                continue

            # Compute features from PRIOR games only
            features = _compute_features(
                logs, row["line"], season=season,
                team_pace_map=team_pace_map, opp_ast_map=opp_ast_map,
            )
            if features is None:
                continue

            features["player"] = player_name
            features["player_id"] = pid
            features["line"] = row["line"]
            features["over_price"] = row.get("over_price")
            features["under_price"] = row.get("under_price")
            features["home_team"] = row.get("home_team", "")
            features["away_team"] = row.get("away_team", "")
            features["game_time"] = row.get("game_time")
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
                      xgb_regressor=None,
                      season: str = None,
                      team_pace_map: dict = None,
                      opp_ast_map: dict = None) -> dict:
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

    # Resolve team / opponent abbreviations from the most recent game row so we
    # can look up real pace & opp_ast_allowed (S1-3 fix).
    last_row = logs.iloc[-1]
    team_abbr = str(last_row.get("TEAM_ABBREVIATION", ""))
    parts = matchup.replace(" ", "").split("@" if "@" in matchup else "vs.")
    opp_abbr = ""
    for p in parts:
        p = p.strip()
        if p and p != team_abbr:
            opp_abbr = p
            break

    tp_map = team_pace_map or {}
    oa_map = opp_ast_map or {}
    team_pace = tp_map.get((team_abbr, season), 100.0) if season else 100.0
    opp_pace = tp_map.get((opp_abbr, season), 100.0) if season else 100.0
    opp_ast_allowed = oa_map.get((opp_abbr, season), 25.0) if season else 25.0

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
        "team_pace": float(team_pace),
        "opp_pace": float(opp_pace),
        "opp_ast_allowed": float(opp_ast_allowed),
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

    # Enrichment fields for the unified paper_trades row (S3-a fix):
    # team abbreviation + projected count used for confidence grading
    projected_ast = float(ast_pm_season * proj_min)
    reg_features["team"] = team_abbr
    reg_features["projected_ast"] = round(projected_ast, 2)
    reg_features["confidence"] = _compute_confidence(logs, projected_ast, line)

    return reg_features


def _compute_confidence(logs: pd.DataFrame, projected: float, line: float) -> str:
    """Simplified confidence grade ported from lib/model.py:compute_confidence_grade.

    Scores CV of AST, games played, recent-minutes stability, and line distance.
    Returns 'A' / 'B' / 'C' / 'D'.

    Defensive guards (S4 audit):
      - n < 1 → D
      - Missing AST / MIN_FLOAT column → D
      - NaN mean/std fall through to the fallback branches (NaN > 0 is False)
    """
    n = len(logs)
    if n < 1:
        return "D"
    if "AST" not in logs.columns or "MIN_FLOAT" not in logs.columns:
        return "D"
    m = float(logs["AST"].mean())
    s = float(logs["AST"].std(ddof=1)) if n > 1 else m
    # Guard against pandas returning NaN from empty/all-NaN slices — NaN > 0 is
    # False in Python, so the fallback branch fires and we land on cv = 1.0.
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


# ── Save Predictions to Supabase ──
def save_predictions(predictions: list) -> None:
    """Upsert paper trade predictions to am_paper_trades.

    Idempotent — re-running the same day produces no duplicate rows, provided
    the am_paper_trades table has a UNIQUE(prediction_date, player_id) constraint.
    """
    if not predictions:
        return
    # Merge-duplicates header turns POST into an upsert keyed by the table's
    # unique constraint. Safe to ship ahead of the DB migration (behaves as
    # plain insert until the constraint is added).
    upsert_headers = {**SB_HEADERS, "Prefer": "resolution=merge-duplicates,return=minimal"}
    for i in range(0, len(predictions), 50):
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/am_paper_trades",
            headers=upsert_headers, json=predictions[i:i+50], timeout=15)
        if not r.ok:
            logger.error("Failed to save predictions: %s %s", r.status_code, r.text[:200])


def resolve_predictions() -> None:
    """Resolve ALL unresolved predictions with prediction_date <= yesterday.

    Previously this only checked `eq.yesterday`, so any bet missed by the cron
    (DNP, NBA API hiccup, workflow gap) stayed stuck forever. Now it sweeps
    every unresolved row on or before yesterday.

    DNP handling: if the player has game logs but none for the bet's
    prediction_date, mark the row resolved with `void_reason='dnp'`, pnl=0,
    under_hit=NULL (stake returned). Previously these fell through a silent
    `continue` and stayed unresolved indefinitely.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch ALL unresolved predictions with prediction_date <= cutoff (was eq.)
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/am_paper_trades",
        params={"prediction_date": f"lte.{cutoff}", "resolved": "eq.false",
                "select": "*", "order": "prediction_date.asc"},
        headers=SB_HEADERS, timeout=15)
    if not r.ok or not r.json():
        logger.info("No unresolved predictions with prediction_date <= %s", cutoff)
        save_run_log("assists", "resolve", cutoff, predictions_resolved=0)
        return

    preds = r.json()
    logger.info("Resolving %d predictions (prediction_date <= %s)", len(preds), cutoff)
    resolved_count = 0
    voided_count = 0

    # Cache NBA API game-log fetches per player — same player can have bets on
    # multiple prediction_dates and we don't want to re-hit the API per bet.
    logs_cache: dict = {}

    # Fetch actual results from nba_api
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
                    "actual_ast": None,
                    "under_hit": None,
                    "pnl": 0.0,
                    "resolved": True,
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                    "void_reason": "dnp",
                }
                pr = requests.patch(
                    f"{SUPABASE_URL}/rest/v1/am_paper_trades",
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

            actual_ast = int(game.iloc[0]["AST"])
            line = pred["line"]
            under_hit = actual_ast < line
            under_price = pred.get("under_price", -110)
            # Only compute PnL for actual bets; non-bet rows stay at 0 so the
            # dashboard's cumulative PnL doesn't double-count skipped picks.
            if pred.get("bet_placed"):
                dec = american_to_decimal(under_price) if under_price else 1.91
                pnl = 100 * (dec - 1) if under_hit else -100
            else:
                pnl = 0.0

            # Update in Supabase
            update = {
                "actual_ast": actual_ast,
                "under_hit": under_hit,
                "pnl": round(pnl, 2),
                "resolved": True,
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            }
            pr = requests.patch(
                f"{SUPABASE_URL}/rest/v1/am_paper_trades",
                params={"id": f"eq.{pred['id']}"},
                headers={**SB_HEADERS, "Prefer": "return=minimal"},
                json=update, timeout=15)
            if not pr.ok:
                logger.error("PATCH failed for %s: %s %s", pred.get("player"), pr.status_code, pr.text[:200])
            else:
                resolved_count += 1

        except Exception as e:
            logger.warning("Failed to resolve %s: %s", pred.get("player"), str(e)[:100])

    logger.info("Resolution complete (<= %s): %d resolved (%d voided) of %d",
                cutoff, resolved_count, voided_count, len(preds))
    save_run_log("assists", "resolve", cutoff,
                 predictions_resolved=resolved_count)


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
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Fast pre-check: if stats.nba.com is unreachable, fall back to yesterday's
    # cached logs instead of timing out per-player (S4 fix, S5 fallback). We
    # only fall back for predict/refresh — resolve still requires fresh data.
    api_up = _nba_api_reachable()
    if not api_up:
        age = cache_age()
        if age is None:
            logger.error("NBA API unreachable and no game-logs cache exists — aborting")
            save_run_log("assists", "predict", today, status="error",
                         error_msg="NBA API unreachable; no cache")
            sys.exit(2)
        if age > MAX_CACHE_AGE:
            hours = age.total_seconds() / 3600
            logger.error("NBA API unreachable and cache is %.1fh old (max %.0fh) — aborting",
                         hours, MAX_CACHE_AGE.total_seconds() / 3600)
            save_run_log("assists", "predict", today, status="error",
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
        return

    iso, brier_raw, brier_cal = build_isotonic_calibrator()

    # `today` is already set above in the probe block.
    lines, num_events = fetch_todays_lines()
    if lines.empty:
        logger.info("No lines available (no games today?)")
        save_run_log("assists", "predict", today, events_found=num_events, lines_fetched=0,
                     predictions_saved=0, bets_placed=0)
        return

    features = build_today_features(lines, api_up=api_up)
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
        over_price = row.get("over_price")
        dk_implied = american_to_implied(under_price) if under_price else 0.5
        cal_prob = float(row["calibrated_prob_under"])
        edge = (cal_prob - dk_implied) * 100

        # ── Unified enrichment: Kelly + best-side + model EV (S3-a fix) ──
        # Mirrors lib/model.py:build_projection_rows so paper_trades carries the
        # same best_side / kelly_pct fields the projections tab used to own.
        prob_over = 1.0 - cal_prob
        dk_implied_over = american_to_implied(over_price) if over_price else 0.5
        edge_over = (prob_over - dk_implied_over) * 100
        edge_under = edge  # already computed above
        d_over = american_to_decimal(over_price) if over_price else 1.91
        d_under = american_to_decimal(under_price) if under_price else 1.91
        ev_over = (prob_over * d_over - 1) * 100
        ev_under = (cal_prob * d_under - 1) * 100

        if edge_over >= edge_under:
            best_side, best_edge, best_ev, best_decimal = "OVER", edge_over, ev_over, d_over
        else:
            best_side, best_edge, best_ev, best_decimal = "UNDER", edge_under, ev_under, d_under

        # Quarter-Kelly, capped at 5% of bankroll
        kelly_raw = (best_edge / 100.0) / (best_decimal - 1) if best_decimal > 1 else 0.0
        kelly_pct = round(max(0.0, min(kelly_raw / 4.0, 0.05)) * 100, 2)

        # Game tip-off — normalize to ISO string for timestamptz insert
        gt = row.get("game_time")
        if gt is not None and not (isinstance(gt, float) and math.isnan(gt)):
            game_time = str(gt)
        else:
            game_time = None

        # Stamp prediction_date with the actual game date (in NBA's ET clock),
        # not "today". The odds API publishes lines a day or two early, so a
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
            "over_price": over_price,
            "under_price": under_price,
            "raw_prob_under": round(float(row["raw_prob_under"]), 4),
            "calibrated_prob_under": round(cal_prob, 4),
            "dk_implied_under": round(float(dk_implied), 4),
            "edge_under": round(edge, 2),
            "bet_placed": bool(row["bet_placed"]),
            "resolved": False,
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            # New unified fields (populated by migration add_projection_fields_to_paper_trades):
            "team": row.get("team") or None,
            "game_time": game_time,
            "best_side": best_side,
            "kelly_pct": kelly_pct,
            "confidence": row.get("confidence") or "C",
            "over_prob": round(float(prob_over), 4),
            "model_ev": round(float(best_ev), 2),
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
