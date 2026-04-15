#!/usr/bin/env python3
"""
Rebound projection refresh — generates daily rebound projections using the
trained XGBoost rebound binary classifier and saves them to Supabase rb_projections.

Usage:
    python scripts/rebound_refresh.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone

from lib.config import logger
from lib.backtest_utils import (
    get_project_root, ensure_dirs,
    american_to_implied, american_to_decimal,
)
from lib.rebound_config import BINARY_FEATURE_COLS
from lib.db import save_rebound_projections, save_run_log
from scripts.rebound_paper_trade import (
    fetch_todays_rebound_lines,
    build_today_features,
    _compute_rebound_features,
)

ROOT = get_project_root()
EDGE_THRESHOLD = 3.0  # Show all players with at least 3% edge


def load_model():
    model_path = ROOT / "models" / "rebound_binary_classifier.json"
    if not model_path.exists():
        logger.error("No rebound binary classifier found at %s", model_path)
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    logger.info("Loaded rebound XGB model")
    return model


def main():
    ensure_dirs()
    logger.info("Rebound refresh — generating rebound projections for today")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    model = load_model()
    if model is None:
        return

    lines = fetch_todays_rebound_lines()
    if lines.empty:
        logger.info("No rebound lines available — no games today or props not posted yet")
        save_run_log("rebounds", "refresh", today, events_found=0, predictions_saved=0)
        return

    features = build_today_features(lines)
    if features.empty:
        logger.info("No features built — all players failed NBA API lookups")
        save_run_log("rebounds", "refresh", today, lines_fetched=len(lines),
                     predictions_saved=0, status="error",
                     error_msg="All player feature builds failed")
        return

    X = features[BINARY_FEATURE_COLS].values
    over_probs = model.predict_proba(X)[:, 1]

    rows = []
    for i, (_, feat_row) in enumerate(features.iterrows()):
        over_prob = float(over_probs[i])
        under_prob = 1.0 - over_prob

        over_price = feat_row.get("over_price")
        under_price = feat_row.get("under_price")

        dk_implied_over = american_to_implied(over_price) if over_price else 0.5
        dk_implied_under = american_to_implied(under_price) if under_price else 0.5
        ft = dk_implied_over + dk_implied_under
        if ft > 0:
            dk_implied_over /= ft
            dk_implied_under /= ft

        edge_over = round((over_prob - dk_implied_over) * 100, 2)
        edge_under = round((under_prob - dk_implied_under) * 100, 2)

        do_ = american_to_decimal(over_price) if over_price else 1.91
        du_ = american_to_decimal(under_price) if under_price else 1.91
        ev_over = round((over_prob * do_ - 1) * 100, 2)
        ev_under = round((under_prob * du_ - 1) * 100, 2)

        if edge_over >= edge_under:
            best_side, best_edge, best_ev = "OVER", edge_over, ev_over
        else:
            best_side, best_edge, best_ev = "UNDER", edge_under, ev_under

        bd = do_ if best_side == "OVER" else du_
        kr = (best_edge / 100) / (bd - 1) if bd > 1 else 0
        kelly_pct = round(max(0.0, min(kr / 4, 0.05)) * 100, 2)

        # Expected rebounds from the model features
        proj_min = float(feat_row.get("proj_minutes", 30))
        reb_pm = float(feat_row.get("reb_per_min_l5", 0))
        expected_reb = round(proj_min * reb_pm, 2)

        rows.append({
            "player": feat_row["player"],
            "player_id": int(feat_row.get("player_id", 0)),
            "line": float(feat_row["line"]),
            "over_price": int(over_price) if over_price else None,
            "under_price": int(under_price) if under_price else None,
            "home_team": feat_row.get("home_team", ""),
            "away_team": feat_row.get("away_team", ""),
            "over_prob": round(over_prob * 100, 1),
            "dk_implied_over": round(dk_implied_over * 100, 1),
            "edge_over": edge_over,
            "edge_under": edge_under,
            "ev_over": ev_over,
            "ev_under": ev_under,
            "best_side": best_side,
            "best_edge": best_edge,
            "best_ev": best_ev,
            "kelly_pct": kelly_pct,
            "expected_reb": expected_reb,
            "proj_minutes": proj_min,
        })

    rows.sort(key=lambda x: abs(x["best_edge"]), reverse=True)
    logger.info("Generated %d rebound projections", len(rows))

    save_rebound_projections(rows)

    save_run_log("rebounds", "refresh", today,
                 lines_fetched=len(lines),
                 predictions_saved=len(rows))


if __name__ == "__main__":
    main()
