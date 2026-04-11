#!/usr/bin/env python3
"""
Rebound XGBoost training with TimeSeriesSplit hyperparameter tuning.
Hyperparameters are tuned ONCE on the pre-backtest era (2023-24) and frozen.
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from pathlib import Path

from lib.config import logger
from lib.backtest_utils import (
    PARAM_SPACE, get_project_root, ensure_dirs,
)
from lib.rebound_config import (
    REBOUND_FEATURE_COLS, REBOUND_TARGET_COL, REBOUND_TUNING_CUTOFF,
)


def custom_time_series_split(df: pd.DataFrame, n_splits: int = 5):
    """TimeSeriesSplit that keeps all rows on the same game_date together."""
    dates = np.sort(df["game_date"].unique())
    n_dates = len(dates)
    fold_size = n_dates // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_end = fold_size * (i + 2) if i < n_splits - 1 else n_dates

        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]

        train_idx = df[df["game_date"].isin(train_dates)].index.values
        val_idx = df[df["game_date"].isin(val_dates)].index.values

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits


def tune_hyperparameters(
    df: pd.DataFrame,
    cutoff_date: str = REBOUND_TUNING_CUTOFF,
    n_splits: int = 5,
    n_iter: int = 50,
    objectives: list = None,
) -> dict:
    """
    Tune on pre-backtest era (before cutoff_date).
    Tests 3 objectives, picks best MAE. Returns frozen params.
    """
    ensure_dirs()
    root = get_project_root()

    if objectives is None:
        objectives = ["reg:absoluteerror", "reg:squarederror", "count:poisson"]

    tune_df = df[df["game_date"] < pd.Timestamp(cutoff_date)].copy().reset_index(drop=True)
    logger.info("Tuning era: %d rows (before %s)", len(tune_df), cutoff_date)

    if len(tune_df) < 5000:
        logger.warning("Small tuning set (%d rows). Results may be unreliable.", len(tune_df))

    X = tune_df[REBOUND_FEATURE_COLS].values
    y = tune_df[REBOUND_TARGET_COL].values

    splits = custom_time_series_split(tune_df, n_splits=n_splits)
    cv_indices = [(train, val) for train, val in splits]

    best_overall = None
    best_objective = None
    all_results = {}

    for obj in objectives:
        logger.info("Testing objective: %s", obj)

        model = xgb.XGBRegressor(
            objective=obj,
            eval_metric="mae",
            tree_method="hist",
            random_state=42,
            verbosity=0,
        )

        search = RandomizedSearchCV(
            model, PARAM_SPACE, n_iter=n_iter,
            cv=cv_indices,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )

        search.fit(X, y)
        mae = -search.best_score_

        logger.info("  %s: best CV MAE = %.4f (%.1f%% of mean)", obj, mae, mae / y.mean() * 100)
        all_results[obj] = {
            "mae": round(mae, 4),
            "pct_of_mean": round(mae / y.mean() * 100, 1),
            "params": search.best_params_,
        }

        if best_overall is None or mae < best_overall:
            best_overall = mae
            best_objective = obj
            best_params = {**search.best_params_, "objective": obj}

    logger.info("Best objective: %s (MAE=%.4f, %.1f%% of mean)",
                best_objective, best_overall, best_overall / y.mean() * 100)

    # Save frozen params
    best_params["eval_metric"] = "mae"
    best_params["tree_method"] = "hist"
    best_params["random_state"] = 42
    best_params["verbosity"] = 0

    params_path = root / "models" / "rebound_frozen_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    logger.info("Frozen params saved to %s", params_path)

    results_path = root / "data" / "reports" / "rebound_cv_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return best_params


def train_rebound_model(
    X: np.ndarray, y: np.ndarray, params: dict
) -> xgb.XGBRegressor:
    """Train XGBoost regressor with frozen hyperparameters."""
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def get_feature_importance(model: xgb.XGBRegressor) -> dict:
    """Extract gain-based feature importance."""
    importance = model.feature_importances_
    return {name: round(float(imp), 4)
            for name, imp in zip(REBOUND_FEATURE_COLS, importance)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune rebound XGBoost hyperparameters")
    parser.add_argument("--cutoff", default=REBOUND_TUNING_CUTOFF)
    parser.add_argument("--n-iter", type=int, default=50)
    args = parser.parse_args()

    root = get_project_root()
    df = pd.read_parquet(root / "data" / "features" / "rebound_features.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])

    best = tune_hyperparameters(df, cutoff_date=args.cutoff, n_iter=args.n_iter)
    logger.info("Tuning complete. Best params:\n%s", json.dumps(best, indent=2, default=str))
