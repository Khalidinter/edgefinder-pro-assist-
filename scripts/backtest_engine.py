#!/usr/bin/env python3
"""
Walk-forward backtest engine.
Retrains every N days with expanding window. Frozen hyperparameters.
Generates synthetic lines and computes NB probabilities with shrinkage alpha.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

from lib.config import logger
from lib.backtest_utils import (
    FEATURE_COLS, TARGET_COL, BACKTEST_START, BACKTEST_END,
    RETRAIN_FREQUENCY_DAYS, EDGE_THRESHOLD_PCT, FLAT_BET_SIZE,
    STANDARD_ODDS, get_project_root, ensure_dirs,
    american_to_implied, american_to_decimal, payout_at_odds,
)
from scripts.nb_layer import NBProbabilityLayer
from scripts.train_model import train_model, get_feature_importance


def generate_synthetic_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    4 synthetic line types — all use shift(1) to prevent lookahead.
    Primary: rolling 20-game mean, rounded to 0.5.
    """
    df = df.copy()
    grp = df.groupby("player_id")

    # 1. Rolling 20-game mean (primary)
    df["line_rolling_mean"] = grp["actual_ast"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).mean()
    ).round(1)
    # Round to nearest 0.5
    df["line_rolling_mean"] = (df["line_rolling_mean"] * 2).round() / 2

    # 2. Rolling 20-game median
    df["line_rolling_median"] = grp["actual_ast"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).median()
    )
    df["line_rolling_median"] = (df["line_rolling_median"] * 2).round() / 2

    # 3. Minutes-adjusted mean: (rolling AST/MIN) * proj_minutes
    df["line_min_adjusted"] = df["ast_per_min_l10"] * df["proj_minutes"]
    df["line_min_adjusted"] = (df["line_min_adjusted"] * 2).round() / 2

    # 4. Exponentially weighted mean (halflife=10 games)
    df["line_ewm"] = grp["actual_ast"].transform(
        lambda x: x.shift(1).ewm(halflife=10, min_periods=10).mean()
    )
    df["line_ewm"] = (df["line_ewm"] * 2).round() / 2

    # Primary line
    df["synthetic_line"] = df["line_rolling_mean"]

    return df


def walk_forward_backtest(
    feature_matrix: pd.DataFrame,
    frozen_params: dict,
    start_date: str = BACKTEST_START,
    end_date: str = BACKTEST_END,
    retrain_days: int = RETRAIN_FREQUENCY_DAYS,
    edge_threshold: float = EDGE_THRESHOLD_PCT,
) -> pd.DataFrame:
    """
    Walk-forward simulation with monthly retraining.
    Returns DataFrame of all predictions with bet outcomes.
    """
    df = feature_matrix.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    # Generate synthetic lines
    df = generate_synthetic_lines(df)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Filter to test period only for predictions, but training uses all prior data
    test_dates = sorted(df[df["game_date"].between(start, end)]["game_date"].unique())
    logger.info("Walk-forward: %d test dates from %s to %s", len(test_dates), start_date, end_date)

    model = None
    nb_layer = NBProbabilityLayer()
    last_retrain_date = None
    results = []
    retrain_count = 0

    for current_date in test_dates:
        # Check if retrain needed
        needs_retrain = (
            model is None or
            last_retrain_date is None or
            (current_date - last_retrain_date).days >= retrain_days
        )

        if needs_retrain:
            train_data = df[df["game_date"] < current_date].copy()
            if len(train_data) < 5000:
                continue

            X_train = train_data[FEATURE_COLS].values
            y_train = train_data[TARGET_COL].values

            model = train_model(X_train, y_train, frozen_params)

            # Fit NB layer: predict on training data to estimate alphas
            train_preds = model.predict(X_train)
            nb_layer.fit(
                actuals=pd.Series(y_train),
                predictions=pd.Series(train_preds),
                player_ids=train_data["player_id"].reset_index(drop=True),
            )

            last_retrain_date = current_date
            retrain_count += 1
            logger.info("  Retrain #%d at %s (train size: %d, pop_alpha: %.3f)",
                       retrain_count, current_date.date(), len(train_data), nb_layer.population_alpha)

        # Predict today's games
        today = df[df["game_date"] == current_date].copy()
        if today.empty:
            continue

        X_today = today[FEATURE_COLS].values
        predictions = model.predict(X_today)

        for i, (idx, row) in enumerate(today.iterrows()):
            pred_ast = float(predictions[i])
            actual_ast = int(row[TARGET_COL])
            line = row["synthetic_line"]

            if pd.isna(line) or line <= 0:
                continue

            # NB probability
            proba = nb_layer.predict_proba(pred_ast, line, int(row["player_id"]))
            over_prob = proba["over_prob"]
            under_prob = proba["under_prob"]
            alpha = proba["alpha"]

            # Edge vs 50% (synthetic line assumes -110 both sides = 50% fair)
            edge_over = (over_prob - 0.50) * 100
            edge_under = (under_prob - 0.50) * 100

            # Determine best side
            if edge_over >= edge_under:
                best_side = "OVER"
                best_edge = edge_over
                bet_won = actual_ast > line
            else:
                best_side = "UNDER"
                best_edge = edge_under
                bet_won = actual_ast < line

            # Bet decision
            bet_placed = abs(best_edge) >= edge_threshold
            pnl = payout_at_odds(STANDARD_ODDS, FLAT_BET_SIZE, bet_won) if bet_placed else 0.0

            results.append({
                "player_name": row["player_name"],
                "player_id": int(row["player_id"]),
                "season": row["season"],
                "game_date": current_date,
                "team_abbr": row.get("team_abbr", ""),
                "opp_team_abbr": row.get("opp_team_abbr", ""),
                "predicted_ast": round(pred_ast, 2),
                "actual_ast": actual_ast,
                "actual_min": float(row.get("actual_min", 0)),
                "synthetic_line": line,
                "over_prob": round(over_prob, 4),
                "under_prob": round(under_prob, 4),
                "alpha": round(alpha, 4),
                "edge_over": round(edge_over, 2),
                "edge_under": round(edge_under, 2),
                "best_side": best_side,
                "best_edge": round(best_edge, 2),
                "bet_placed": bet_placed,
                "bet_won": bet_won if bet_placed else None,
                "pnl": round(pnl, 2),
                "error": round(pred_ast - actual_ast, 2),
            })

    results_df = pd.DataFrame(results)
    logger.info("Walk-forward complete: %d predictions, %d bets placed, %d retrains",
                len(results_df),
                results_df["bet_placed"].sum(),
                retrain_count)

    # Save results
    root = get_project_root()
    results_df.to_parquet(root / "data" / "reports" / "backtest_results.parquet", index=False)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=BACKTEST_START)
    parser.add_argument("--end", default=BACKTEST_END)
    parser.add_argument("--retrain-days", type=int, default=RETRAIN_FREQUENCY_DAYS)
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD_PCT)
    args = parser.parse_args()

    root = get_project_root()
    df = pd.read_parquet(root / "data" / "features" / "feature_matrix.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])

    params_path = root / "models" / "frozen_params.json"
    if not params_path.exists():
        logger.error("No frozen params found. Run train_model.py --tune first.")
        sys.exit(1)

    with open(params_path) as f:
        frozen_params = json.load(f)

    results = walk_forward_backtest(
        df, frozen_params,
        start_date=args.start, end_date=args.end,
        retrain_days=args.retrain_days, edge_threshold=args.edge_threshold,
    )
