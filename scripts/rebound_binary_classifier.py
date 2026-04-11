#!/usr/bin/env python3
"""
Rebound binary classifier — predicts P(over) directly with DK line as feature.

25 features: 21 base rebound features + dk_line + pred_minus_line +
             dk_implied_over_prob + dk_over_price (juice asymmetry signal)

Walk-forward: train on expanding window, retrain every 30 days.
ROI uses actual DK juice. Both directions evaluated independently.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

from lib.config import logger
from lib.backtest_utils import (
    get_project_root, ensure_dirs, FLAT_BET_SIZE,
    american_to_implied, american_to_decimal, payout_at_odds,
)
from lib.rebound_config import (
    BINARY_FEATURE_COLS, REBOUND_BACKTEST_START, REBOUND_BACKTEST_END,
    REBOUND_RETRAIN_DAYS, REBOUND_EDGE_THRESHOLD,
)


def walk_forward_binary(
    df: pd.DataFrame,
    start_date: str = REBOUND_BACKTEST_START,
    end_date: str = REBOUND_BACKTEST_END,
    retrain_days: int = REBOUND_RETRAIN_DAYS,
    edge_threshold: float = REBOUND_EDGE_THRESHOLD,
) -> pd.DataFrame:
    """
    Walk-forward binary classifier backtest with real DK lines.
    """
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Filter to rows with DK lines + valid over_hit target
    df = df[df["dk_line"].notna() & df["over_hit"].notna()].copy()
    df = df.sort_values("game_date").reset_index(drop=True)

    logger.info("Binary classifier data: %d rows with DK lines and valid target", len(df))

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    test_dates = sorted(df[df["game_date"].between(start, end)]["game_date"].unique())
    logger.info("Walk-forward: %d test dates from %s to %s", len(test_dates), start_date, end_date)

    model = None
    last_retrain_date = None
    retrain_count = 0
    results = []

    for current_date in test_dates:
        needs_retrain = (
            model is None or
            last_retrain_date is None or
            (current_date - last_retrain_date).days >= retrain_days
        )

        if needs_retrain:
            train = df[df["game_date"] < current_date].copy()
            if len(train) < 2000:
                continue

            X_train = train[BINARY_FEATURE_COLS].values
            y_train = train["over_hit"].values

            # Adaptive hyperparams based on training set size
            depth = 3 if len(train) < 5000 else 4 if len(train) < 20000 else 5
            n_est = 100 if len(train) < 5000 else 300 if len(train) < 20000 else 500
            mcw = max(10, len(train) // 200)

            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                max_depth=depth,
                n_estimators=n_est,
                learning_rate=0.05,
                min_child_weight=mcw,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=1.0,
                reg_lambda=5.0,
                gamma=0.1,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_train, y_train)

            last_retrain_date = current_date
            retrain_count += 1
            logger.info("  Retrain #%d at %s (train=%d, depth=%d, n_est=%d)",
                        retrain_count, current_date.date(), len(train), depth, n_est)

        # Predict today's games
        today = df[df["game_date"] == current_date].copy()
        if today.empty:
            continue

        X_today = today[BINARY_FEATURE_COLS].values
        proba = model.predict_proba(X_today)[:, 1]  # P(over)

        for i, (idx, row) in enumerate(today.iterrows()):
            over_prob = float(proba[i])
            under_prob = 1.0 - over_prob
            actual_reb = int(row["actual_reb"])
            dk_line = float(row["dk_line"])
            dk_over_price = float(row["dk_over_price"])
            dk_under_price = float(row["dk_under_price"])

            dk_implied_over = american_to_implied(dk_over_price)
            dk_implied_under = american_to_implied(dk_under_price)

            edge_over = (over_prob - dk_implied_over) * 100
            edge_under = (under_prob - dk_implied_under) * 100

            # Bet decisions — independent evaluation
            over_bet = edge_over >= edge_threshold
            under_bet = edge_under >= edge_threshold

            if over_bet and under_bet:
                if edge_over >= edge_under:
                    under_bet = False
                else:
                    over_bet = False

            bet_placed = over_bet or under_bet
            if over_bet:
                bet_side = "OVER"
                bet_won = actual_reb > dk_line
                pnl = payout_at_odds(dk_over_price, FLAT_BET_SIZE, bet_won)
            elif under_bet:
                bet_side = "UNDER"
                bet_won = actual_reb < dk_line
                pnl = payout_at_odds(dk_under_price, FLAT_BET_SIZE, bet_won)
            else:
                bet_side = None
                bet_won = None
                pnl = 0.0

            # Push
            if bet_placed and actual_reb == dk_line:
                pnl = 0.0
                bet_won = None

            results.append({
                "player_name": row["player_name"],
                "player_id": int(row["player_id"]),
                "season": row["season"],
                "game_date": current_date,
                "team_abbr": row.get("team_abbr", ""),
                "opp_team_abbr": row.get("opp_team_abbr", ""),
                "actual_reb": actual_reb,
                "dk_line": dk_line,
                "dk_over_price": dk_over_price,
                "dk_under_price": dk_under_price,
                "model_over_prob": round(over_prob, 4),
                "dk_implied_over": round(dk_implied_over, 4),
                "edge_over": round(edge_over, 2),
                "edge_under": round(edge_under, 2),
                "bet_placed": bet_placed,
                "bet_side": bet_side,
                "bet_won": bet_won,
                "pnl": round(pnl, 2),
            })

    results_df = pd.DataFrame(results)

    # ── Summary ──
    bets = results_df[results_df["bet_placed"]].copy()

    logger.info("=" * 60)
    logger.info("REBOUND BINARY CLASSIFIER WALK-FORWARD SUMMARY")
    logger.info("=" * 60)
    logger.info("Total predictions: %d", len(results_df))
    logger.info("Retrains: %d", retrain_count)

    if len(bets) > 0:
        total_pnl = bets["pnl"].sum()
        total_risked = len(bets) * FLAT_BET_SIZE
        roi = total_pnl / total_risked * 100

        logger.info("\nBetting (edge >= %.1f%%):", edge_threshold)
        logger.info("  Bets: %d", len(bets))
        logger.info("  PnL:  $%.2f", total_pnl)
        logger.info("  ROI:  %.2f%%", roi)

        for side in ["OVER", "UNDER"]:
            sb = bets[bets["bet_side"] == side]
            if len(sb) > 0:
                s_pnl = sb["pnl"].sum()
                s_won = sb["bet_won"].dropna()
                s_roi = s_pnl / (len(sb) * FLAT_BET_SIZE) * 100
                logger.info("  %s: %d bets, %.1f%% hit, ROI %.2f%%",
                            side, len(sb),
                            s_won.mean() * 100 if len(s_won) > 0 else 0, s_roi)

        for season in sorted(bets["season"].unique()):
            sb = bets[bets["season"] == season]
            s_roi = sb["pnl"].sum() / (len(sb) * FLAT_BET_SIZE) * 100
            logger.info("  %s: %d bets, ROI %.2f%%", season, len(sb), s_roi)

    # Feature importance
    if model is not None:
        importance = dict(zip(BINARY_FEATURE_COLS, model.feature_importances_))
        top_5 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("\nTop 5 features:")
        for name, imp in top_5:
            logger.info("  %-25s %.4f", name, imp)

    logger.info("=" * 60)

    # Save
    root = get_project_root()
    output_path = root / "data" / "reports" / "rebound_binary_results.parquet"
    results_df.to_parquet(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebound binary classifier walk-forward")
    parser.add_argument("--start", default=REBOUND_BACKTEST_START)
    parser.add_argument("--end", default=REBOUND_BACKTEST_END)
    parser.add_argument("--retrain-days", type=int, default=REBOUND_RETRAIN_DAYS)
    parser.add_argument("--edge-threshold", type=float, default=REBOUND_EDGE_THRESHOLD)
    args = parser.parse_args()

    root = get_project_root()
    df = pd.read_parquet(root / "data" / "features" / "rebound_features.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])

    results = walk_forward_binary(
        df,
        start_date=args.start, end_date=args.end,
        retrain_days=args.retrain_days, edge_threshold=args.edge_threshold,
    )
