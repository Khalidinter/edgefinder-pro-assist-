#!/usr/bin/env python3
"""
Rebound walk-forward backtest with REAL DK lines.

Key differences from assists backtest:
- Uses real DK lines (not synthetic) → edge computed vs DK implied prob
- ROI uses actual DK juice (not flat -110)
- Isotonic calibration from day one (trained on OOS predictions only)
- Evaluates both over AND under directions independently
- No directional assumption — model must find conditional edges
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from pathlib import Path

from lib.config import logger
from lib.backtest_utils import (
    get_project_root, ensure_dirs,
    american_to_implied, american_to_decimal, payout_at_odds, FLAT_BET_SIZE,
)
from lib.rebound_config import (
    REBOUND_FEATURE_COLS, REBOUND_TARGET_COL,
    REBOUND_BACKTEST_START, REBOUND_BACKTEST_END,
    REBOUND_RETRAIN_DAYS, REBOUND_EDGE_THRESHOLD,
    REBOUND_MIN_ISOTONIC_SAMPLES,
)
from scripts.nb_layer import NBProbabilityLayer
from scripts.rebound_train_model import train_rebound_model


def walk_forward_rebound_backtest(
    feature_matrix: pd.DataFrame,
    frozen_params: dict,
    start_date: str = REBOUND_BACKTEST_START,
    end_date: str = REBOUND_BACKTEST_END,
    retrain_days: int = REBOUND_RETRAIN_DAYS,
    edge_threshold: float = REBOUND_EDGE_THRESHOLD,
) -> pd.DataFrame:
    """
    Walk-forward backtest against real DK rebound lines.
    Isotonic calibration uses only genuinely out-of-sample predictions.
    """
    df = feature_matrix.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    test_dates = sorted(df[df["game_date"].between(start, end)]["game_date"].unique())
    logger.info("Walk-forward: %d test dates from %s to %s", len(test_dates), start_date, end_date)

    model = None
    nb_layer = NBProbabilityLayer()
    isotonic = None
    last_retrain_date = None
    retrain_count = 0

    # Accumulate genuinely OOS predictions for isotonic training
    # Each entry: (nb_over_prob, actual_over_hit)
    oos_predictions = []

    results = []

    for current_date in test_dates:
        # ── Retrain check ──
        needs_retrain = (
            model is None or
            last_retrain_date is None or
            (current_date - last_retrain_date).days >= retrain_days
        )

        if needs_retrain:
            train_data = df[df["game_date"] < current_date].copy()
            if len(train_data) < 5000:
                continue

            X_train = train_data[REBOUND_FEATURE_COLS].values
            y_train = train_data[REBOUND_TARGET_COL].values

            model = train_rebound_model(X_train, y_train, frozen_params)

            # Fit NB layer for alpha estimation (uses in-sample — only for alpha, not probs)
            train_preds = model.predict(X_train)
            nb_layer.fit(
                actuals=pd.Series(y_train),
                predictions=pd.Series(train_preds),
                player_ids=train_data["player_id"].reset_index(drop=True),
            )

            # Fit isotonic on genuinely OOS predictions (from previous test windows)
            if len(oos_predictions) >= REBOUND_MIN_ISOTONIC_SAMPLES:
                oos_probs = np.array([p[0] for p in oos_predictions])
                oos_actuals = np.array([p[1] for p in oos_predictions])
                isotonic = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
                isotonic.fit(oos_probs, oos_actuals)
                logger.info("  Isotonic calibration fitted on %d OOS predictions", len(oos_predictions))
            else:
                isotonic = None

            last_retrain_date = current_date
            retrain_count += 1
            logger.info("  Retrain #%d at %s (train=%d, pop_alpha=%.3f, isotonic=%s)",
                        retrain_count, current_date.date(), len(train_data),
                        nb_layer.population_alpha,
                        f"{len(oos_predictions)} samples" if isotonic else "not yet")

        # ── Predict today's games ──
        today = df[df["game_date"] == current_date].copy()
        if today.empty:
            continue

        X_today = today[REBOUND_FEATURE_COLS].values
        predictions = model.predict(X_today)

        for i, (idx, row) in enumerate(today.iterrows()):
            pred_reb = float(predictions[i])
            actual_reb = int(row[REBOUND_TARGET_COL])
            dk_line = row.get("dk_line")

            # Only bet on rows with real DK lines
            has_dk = pd.notna(dk_line) and dk_line > 0

            if has_dk:
                dk_over_price = float(row["dk_over_price"])
                dk_under_price = float(row["dk_under_price"])
                dk_implied_over = american_to_implied(dk_over_price)
                dk_implied_under = american_to_implied(dk_under_price)
            else:
                dk_over_price = dk_under_price = np.nan
                dk_implied_over = dk_implied_under = np.nan

            # NB probability
            line_for_nb = dk_line if has_dk else pred_reb  # fallback for non-DK rows
            proba = nb_layer.predict_proba(pred_reb, line_for_nb, int(row["player_id"]))
            raw_over_prob = proba["over_prob"]
            alpha = proba["alpha"]

            # Isotonic calibration (only if fitted from OOS data)
            if isotonic is not None:
                cal_over_prob = float(isotonic.predict([raw_over_prob])[0])
            else:
                cal_over_prob = raw_over_prob
            cal_under_prob = 1.0 - cal_over_prob

            # Accumulate this genuinely OOS prediction for future isotonic training
            if has_dk:
                actual_over = 1 if actual_reb > dk_line else 0
                if actual_reb != dk_line:  # Exclude pushes
                    oos_predictions.append((raw_over_prob, actual_over))

            # Edge computation (vs DK implied, not vs 50%)
            if has_dk:
                edge_over = (cal_over_prob - dk_implied_over) * 100
                edge_under = (cal_under_prob - dk_implied_under) * 100

                # Bet decisions — evaluate each direction independently
                over_bet = edge_over >= edge_threshold
                under_bet = edge_under >= edge_threshold

                # If both qualify, take the higher edge
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

                # Push handling
                if bet_placed and actual_reb == dk_line:
                    pnl = 0.0
                    bet_won = None
            else:
                edge_over = edge_under = np.nan
                bet_placed = False
                bet_side = None
                bet_won = None
                pnl = 0.0

            results.append({
                "player_name": row["player_name"],
                "player_id": int(row["player_id"]),
                "season": row["season"],
                "game_date": current_date,
                "team_abbr": row.get("team_abbr", ""),
                "opp_team_abbr": row.get("opp_team_abbr", ""),
                "predicted_reb": round(pred_reb, 2),
                "actual_reb": actual_reb,
                "actual_min": float(row.get("actual_min", 0)),
                "dk_line": dk_line if has_dk else np.nan,
                "dk_over_price": dk_over_price if has_dk else np.nan,
                "dk_under_price": dk_under_price if has_dk else np.nan,
                "dk_implied_over": round(dk_implied_over, 4) if has_dk else np.nan,
                "raw_over_prob": round(raw_over_prob, 4),
                "cal_over_prob": round(cal_over_prob, 4),
                "alpha": round(alpha, 4),
                "edge_over": round(edge_over, 2) if has_dk else np.nan,
                "edge_under": round(edge_under, 2) if has_dk else np.nan,
                "bet_placed": bet_placed,
                "bet_side": bet_side,
                "bet_won": bet_won,
                "pnl": round(pnl, 2),
                "error": round(pred_reb - actual_reb, 2),
            })

    results_df = pd.DataFrame(results)

    # ── Summary ──
    bets = results_df[results_df["bet_placed"]].copy()
    dk_rows = results_df[results_df["dk_line"].notna()]

    logger.info("=" * 60)
    logger.info("REBOUND WALK-FORWARD BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info("Total predictions: %d (%d with DK lines)", len(results_df), len(dk_rows))
    logger.info("Retrains: %d", retrain_count)
    logger.info("Isotonic OOS samples: %d", len(oos_predictions))

    # Prediction accuracy
    logger.info("\nPrediction accuracy (all rows):")
    errors = results_df["predicted_reb"] - results_df["actual_reb"]
    logger.info("  MAE:  %.3f (%.1f%% of mean)", errors.abs().mean(),
                errors.abs().mean() / results_df["actual_reb"].mean() * 100)
    logger.info("  RMSE: %.3f", np.sqrt((errors ** 2).mean()))

    if len(bets) > 0:
        total_pnl = bets["pnl"].sum()
        total_risked = len(bets) * FLAT_BET_SIZE
        roi = total_pnl / total_risked * 100 if total_risked > 0 else 0

        logger.info("\nBetting results (edge >= %.1f%%):", edge_threshold)
        logger.info("  Bets placed: %d", len(bets))
        logger.info("  Total PnL:   $%.2f", total_pnl)
        logger.info("  ROI:         %.2f%%", roi)

        # By direction
        for side in ["OVER", "UNDER"]:
            side_bets = bets[bets["bet_side"] == side]
            if len(side_bets) > 0:
                side_pnl = side_bets["pnl"].sum()
                side_won = side_bets["bet_won"].dropna()
                side_roi = side_pnl / (len(side_bets) * FLAT_BET_SIZE) * 100
                logger.info("  %s: %d bets, %.1f%% hit rate, ROI %.2f%%",
                           side, len(side_bets),
                           side_won.mean() * 100 if len(side_won) > 0 else 0,
                           side_roi)

        # By season
        for season in sorted(bets["season"].unique()):
            s_bets = bets[bets["season"] == season]
            s_pnl = s_bets["pnl"].sum()
            s_roi = s_pnl / (len(s_bets) * FLAT_BET_SIZE) * 100
            logger.info("  %s: %d bets, ROI %.2f%%", season, len(s_bets), s_roi)

    logger.info("=" * 60)

    # Save
    root = get_project_root()
    output_path = root / "data" / "reports" / "rebound_backtest_results.parquet"
    results_df.to_parquet(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebound walk-forward backtest")
    parser.add_argument("--start", default=REBOUND_BACKTEST_START)
    parser.add_argument("--end", default=REBOUND_BACKTEST_END)
    parser.add_argument("--retrain-days", type=int, default=REBOUND_RETRAIN_DAYS)
    parser.add_argument("--edge-threshold", type=float, default=REBOUND_EDGE_THRESHOLD)
    args = parser.parse_args()

    root = get_project_root()
    df = pd.read_parquet(root / "data" / "features" / "rebound_features.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])

    params_path = root / "models" / "rebound_frozen_params.json"
    if not params_path.exists():
        logger.error("No frozen params found. Run rebound_train_model.py first.")
        sys.exit(1)

    with open(params_path) as f:
        frozen_params = json.load(f)

    results = walk_forward_rebound_backtest(
        df, frozen_params,
        start_date=args.start, end_date=args.end,
        retrain_days=args.retrain_days, edge_threshold=args.edge_threshold,
    )
