#!/usr/bin/env python3
"""
End-to-end backtest orchestrator.
Runs: data_pipeline → feature_engineering → audit → tune → walk-forward → evaluate.
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from lib.config import logger
from lib.backtest_utils import (
    BACKTEST_START, BACKTEST_END, RETRAIN_FREQUENCY_DAYS,
    EDGE_THRESHOLD_PCT, TUNING_CUTOFF, get_project_root, ensure_dirs,
    FEATURE_COLS,
)


def main():
    parser = argparse.ArgumentParser(description="EdgeFinder Assist PRO — Full V1 Backtest")
    parser.add_argument("--seasons", nargs="+", default=["2022-23", "2023-24", "2024-25", "2025-26"])
    parser.add_argument("--start-date", default=BACKTEST_START)
    parser.add_argument("--end-date", default=BACKTEST_END)
    parser.add_argument("--retrain-days", type=int, default=RETRAIN_FREQUENCY_DAYS)
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD_PCT)
    parser.add_argument("--tuning-cutoff", default=TUNING_CUTOFF)
    parser.add_argument("--tuning-iter", type=int, default=50)
    parser.add_argument("--skip-fetch", action="store_true", help="Skip API data fetch")
    parser.add_argument("--skip-tune", action="store_true", help="Skip hyperparameter tuning")
    parser.add_argument("--skip-audit", action="store_true", help="Skip lookahead audit")
    args = parser.parse_args()

    root = get_project_root()
    ensure_dirs()
    start_time = time.time()

    # ── Step 1: Data Pipeline ──
    if not args.skip_fetch:
        logger.info("=" * 60)
        logger.info("STEP 1: Data Pipeline — Fetching NBA data")
        logger.info("=" * 60)
        from scripts.data_pipeline import run_pipeline
        run_pipeline(seasons=args.seasons)
    else:
        logger.info("Skipping data fetch (--skip-fetch)")

    # ── Step 2: Feature Engineering ──
    logger.info("=" * 60)
    logger.info("STEP 2: Feature Engineering — Building feature matrix")
    logger.info("=" * 60)
    from scripts.feature_engineering import build_feature_matrix
    feature_df = build_feature_matrix()

    # ── Step 3: Lookahead Audit ──
    if not args.skip_audit:
        logger.info("=" * 60)
        logger.info("STEP 3: Lookahead Audit — Verifying shift(1) discipline")
        logger.info("=" * 60)
        from scripts.audit_lookahead import audit
        passed = audit(n_samples=100)
        if not passed:
            logger.error("AUDIT FAILED. Aborting backtest.")
            sys.exit(1)
    else:
        logger.info("Skipping audit (--skip-audit)")

    # ── Step 4: Hyperparameter Tuning ──
    params_path = root / "models" / "frozen_params.json"
    if not args.skip_tune or not params_path.exists():
        logger.info("=" * 60)
        logger.info("STEP 4: Hyperparameter Tuning — Pre-backtest era")
        logger.info("=" * 60)
        from scripts.train_model import tune_hyperparameters
        import pandas as pd
        df = pd.read_parquet(root / "data" / "features" / "feature_matrix.parquet")
        df["game_date"] = pd.to_datetime(df["game_date"])
        frozen_params = tune_hyperparameters(df, cutoff_date=args.tuning_cutoff, n_iter=args.tuning_iter)
    else:
        logger.info("Using existing frozen params from %s", params_path)
        with open(params_path) as f:
            frozen_params = json.load(f)

    # ── Step 5: Walk-Forward Backtest ──
    logger.info("=" * 60)
    logger.info("STEP 5: Walk-Forward Backtest")
    logger.info("=" * 60)
    import pandas as pd
    from scripts.backtest_engine import walk_forward_backtest
    df = pd.read_parquet(root / "data" / "features" / "feature_matrix.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])

    results = walk_forward_backtest(
        df, frozen_params,
        start_date=args.start_date, end_date=args.end_date,
        retrain_days=args.retrain_days, edge_threshold=args.edge_threshold,
    )

    # ── Step 6: Evaluation ──
    logger.info("=" * 60)
    logger.info("STEP 6: Evaluation — Full Statistical Report")
    logger.info("=" * 60)
    from scripts.evaluation import generate_report, print_report
    from scripts.train_model import get_feature_importance

    # Get feature importance from final model
    import xgboost as xgb
    final_model_path = root / "models" / "final_model.json"
    fi = {}
    if final_model_path.exists():
        model = xgb.XGBRegressor()
        model.load_model(str(final_model_path))
        fi = get_feature_importance(model)

    report = generate_report(results, feature_importance=fi)
    print_report(report)

    # Save report
    report_path = root / "data" / "reports" / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start_time
    logger.info("Total runtime: %.1f minutes", elapsed / 60)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
