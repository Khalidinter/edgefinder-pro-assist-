#!/usr/bin/env python3
"""
Full model retrain pipeline — run weekly to keep models current.

Steps:
  1. Refresh current season game logs via data_pipeline.py
  2. Rebuild assist feature matrix via feature_engineering.py
  3. Rebuild rebound feature matrix via rebound_feature_engineering.py
  4. Retrain assist binary classifier (binary_classifier.py)
  5. Retrain rebound binary classifier (rebound_binary_classifier.py)

Usage:
    python scripts/retrain.py
    python scripts/retrain.py --skip-data   # Skip data fetch, just retrain
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import logger
from lib.backtest_utils import get_project_root, ensure_dirs, SEASONS_MAP


def run_step(name: str, fn, *args, **kwargs):
    logger.info("=" * 60)
    logger.info("STEP: %s", name)
    logger.info("=" * 60)
    try:
        fn(*args, **kwargs)
        logger.info("✓ %s complete", name)
    except Exception as e:
        logger.error("✗ %s FAILED: %s", name, e)
        raise


def main():
    parser = argparse.ArgumentParser(description="Full model retrain pipeline")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data fetch step (use existing raw parquets)")
    args = parser.parse_args()

    ensure_dirs()
    current_season = sorted(SEASONS_MAP.keys())[-1]

    # ── Step 1: Refresh current season game data ──
    if not args.skip_data:
        from scripts.data_pipeline import run_pipeline
        run_step("Data pipeline — current season", run_pipeline,
                 seasons=[current_season])

    # ── Step 2: Rebuild assist feature matrix ──
    from scripts.feature_engineering import build_feature_matrix
    run_step("Assist feature matrix rebuild", build_feature_matrix)

    # ── Step 3: Rebuild rebound feature matrix ──
    from scripts.rebound_feature_engineering import build_rebound_feature_matrix
    run_step("Rebound feature matrix rebuild", build_rebound_feature_matrix)

    # ── Step 4: Retrain assist binary classifier ──
    # Note: train_and_evaluate already saves the model internally to
    # models/binary_classifier.json and returns a results dict (no "model" key).
    from scripts.binary_classifier import load_and_merge, train_and_evaluate
    def retrain_assist():
        df = load_and_merge()
        result = train_and_evaluate(df)
        logger.info("Assist model saved (test_size=%d, AUC=%.4f)",
                    result.get("test_size", 0), result.get("auc", 0))
    run_step("Assist binary classifier retrain", retrain_assist)

    # ── Step 5: Retrain rebound binary classifier ──
    import pandas as pd
    from scripts.rebound_binary_classifier import walk_forward_binary
    from lib.rebound_config import REBOUND_BACKTEST_START, REBOUND_BACKTEST_END
    def retrain_rebound():
        root = get_project_root()
        df = pd.read_parquet(root / "data" / "features" / "rebound_features.parquet")
        df["game_date"] = pd.to_datetime(df["game_date"])
        walk_forward_binary(df,
                            start_date=REBOUND_BACKTEST_START,
                            end_date=REBOUND_BACKTEST_END)
        # model is saved inside walk_forward_binary (we fixed that)
    run_step("Rebound binary classifier retrain", retrain_rebound)

    logger.info("=" * 60)
    logger.info("All retrain steps complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
