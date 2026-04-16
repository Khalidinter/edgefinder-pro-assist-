#!/usr/bin/env python3
"""
Binary classification model: Given player + matchup + REAL line, does the over hit?

Architecture:
  - Target: over_hit (binary) = actual_ast > line
  - Features: 15 regression + line_value + pred_minus_line + line_minus_l10 (18 total)
  - pred_minus_line uses actual XGBoost regressor output (walk-forward, retrained every 30 days)
  - Model: XGBoost with log-loss
  - No NB layer, no alpha estimation, no distributional assumptions
  - Calibrated probabilities directly from XGBoost

Requires real historical lines from fetch_historical_lines.py.
"""
import sys, os, argparse, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    log_loss, accuracy_score, roc_auc_score, brier_score_loss
)
from pathlib import Path

from lib.config import logger
from lib.backtest_utils import (
    FEATURE_COLS, get_project_root, ensure_dirs,
    american_to_implied, american_to_decimal, payout_at_odds,
)


def normalize_name(name):
    if not name:
        return ""
    s = str(name).lower().strip()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_and_merge(market: str = "player_assists") -> pd.DataFrame:
    """
    Merge real historical lines with the feature matrix.
    Returns DataFrame with features + real line + target.
    """
    root = get_project_root()

    # Load feature matrix
    fm = pd.read_parquet(root / "data" / "features" / "feature_matrix.parquet")
    fm["game_date"] = pd.to_datetime(fm["game_date"])
    fm["norm_player"] = fm["player_name"].apply(normalize_name)

    # Load historical lines (combine all available files)
    lines_dir = root / "data" / "lines"
    line_files = list(lines_dir.glob("lines_*.parquet"))
    if not line_files:
        logger.error("No historical line files found in %s", lines_dir)
        sys.exit(1)

    lines_dfs = [pd.read_parquet(f) for f in line_files]
    lines = pd.concat(lines_dfs, ignore_index=True)

    # Filter to target market
    lines = lines[lines["market"] == market].copy()
    lines["game_date"] = pd.to_datetime(lines["game_date"])
    lines["norm_player"] = lines["player"].apply(normalize_name)
    lines["line"] = pd.to_numeric(lines["line"], errors="coerce")
    lines["over_price"] = pd.to_numeric(lines["over_price"], errors="coerce")
    lines["under_price"] = pd.to_numeric(lines["under_price"], errors="coerce")

    # Dedup: one line per player per date (take DraftKings, latest if multiple)
    lines = lines.sort_values("game_date").drop_duplicates(
        subset=["norm_player", "game_date", "line"], keep="last"
    )

    logger.info("Lines loaded: %d %s lines, %d unique players, dates %s to %s",
                len(lines), market, lines["norm_player"].nunique(),
                lines["game_date"].min().date(), lines["game_date"].max().date())

    # Merge on normalized player name + game date
    merged = lines.merge(
        fm, on=["norm_player", "game_date"], how="inner",
        suffixes=("_line", "_feat"),
    )

    # Get actual assists from the feature matrix (actual_ast column)
    merged["actual"] = merged["actual_ast"]
    merged["over_hit"] = (merged["actual"] > merged["line"]).astype(int)

    # Additional features derived from the real line
    merged["line_value"] = merged["line"]

    # pred_minus_line: XGBoost regressor prediction minus line
    # Load the trained regression model and predict, falling back to linear estimate
    regressor_path = root / "models" / "frozen_params.json"
    xgb_pred = None
    if regressor_path.exists():
        try:
            import json as _json
            with open(regressor_path) as f:
                frozen_params = _json.load(f)
            # Strip verbosity if present (we set it explicitly when constructing)
            frozen_params.pop("verbosity", None)

            # Train a regressor on the same training data to generate predictions
            # Use the feature matrix directly (all rows sorted by date)
            fm_sorted = fm.sort_values("game_date").copy()

            # Walk-forward in 30-day windows: train once per window, predict all
            # rows in that window in one batch. This is ~100x faster than per-row.
            logger.info("Computing pred_minus_line with XGBoost regressor (batched walk-forward)...")
            merged = merged.sort_values("game_date").reset_index(drop=True)
            preds = np.full(len(merged), np.nan)

            # Define 30-day window boundaries from first to last merged date
            min_date = merged["game_date"].min()
            max_date = merged["game_date"].max()
            window_start = min_date
            n_retrains = 0

            while window_start <= max_date:
                window_end = window_start + pd.Timedelta(days=30)

                # Train on all feature matrix data BEFORE this window
                train_fm = fm_sorted[fm_sorted["game_date"] < window_start]
                if len(train_fm) < 500:
                    window_start = window_end
                    continue

                X_tr = train_fm[FEATURE_COLS].values
                y_tr = train_fm["actual_ast"].values
                model_w = xgb.XGBRegressor(**frozen_params, verbosity=0)
                model_w.fit(X_tr, y_tr)
                n_retrains += 1

                # Predict all merged rows in this window in one batch
                in_window = (merged["game_date"] >= window_start) & (merged["game_date"] < window_end)
                if in_window.any():
                    X_pred = merged.loc[in_window, FEATURE_COLS].values
                    preds[in_window.values] = model_w.predict(X_pred)

                window_start = window_end

            merged["xgb_reg_pred"] = preds
            merged["pred_minus_line"] = merged["xgb_reg_pred"] - merged["line"]
            valid_preds = np.isfinite(preds).sum()
            logger.info("  XGBoost regressor: %d retrains, %d / %d rows predicted",
                        n_retrains, valid_preds, len(merged))
        except Exception as e:
            logger.warning("XGBoost regressor failed (%s), using linear estimate", e)
            merged["pred_minus_line"] = merged["ast_per_min_season"] * merged["proj_minutes"] - merged["line"]
    else:
        logger.warning("No frozen_params.json — using linear estimate for pred_minus_line")
        merged["pred_minus_line"] = merged["ast_per_min_season"] * merged["proj_minutes"] - merged["line"]

    # line_minus_l10 — line minus L10 expected assists
    merged["line_minus_l10"] = merged["line"] - (merged["ast_per_min_l10"] * merged["proj_minutes"])

    logger.info("Merged dataset: %d rows, %d unique players, over_hit rate: %.1f%%",
                len(merged), merged["norm_player"].nunique(),
                merged["over_hit"].mean() * 100)

    return merged


# Binary feature set: original 15 + 3 line-derived = 18
BINARY_FEATURES = FEATURE_COLS + ["line_value", "pred_minus_line", "line_minus_l10"]


def train_and_evaluate(
    df: pd.DataFrame,
    train_end_date: str = None,
    min_train_rows: int = 2000,
) -> dict:
    """
    Time-based train/test split. Train on earlier dates, test on later.
    If train_end_date not specified, uses 70/30 chronological split.
    """
    df = df.sort_values("game_date").reset_index(drop=True)

    if train_end_date:
        train_mask = df["game_date"] < pd.Timestamp(train_end_date)
    else:
        split = int(len(df) * 0.7)
        train_mask = pd.Series([True] * split + [False] * (len(df) - split))

    train = df[train_mask].copy()
    test = df[~train_mask].copy()

    if len(train) < min_train_rows:
        logger.warning("Only %d training rows (need %d). Results will be noisy.", len(train), min_train_rows)

    X_train = train[BINARY_FEATURES].values
    y_train = train["over_hit"].values
    X_test = test[BINARY_FEATURES].values
    y_test = test["over_hit"].values

    logger.info("Train: %d rows (%s to %s), Test: %d rows (%s to %s)",
                len(train), train["game_date"].min().date(), train["game_date"].max().date(),
                len(test), test["game_date"].min().date(), test["game_date"].max().date())
    logger.info("Train over rate: %.1f%%, Test over rate: %.1f%%",
                y_train.mean() * 100, y_test.mean() * 100)

    # Conservative regularization — scale with data size
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

    # Predict
    proba = model.predict_proba(X_test)[:, 1]
    test["prob_over"] = proba
    test["pred_over"] = (proba >= 0.5).astype(int)

    # ── Metrics ──
    accuracy = accuracy_score(y_test, test["pred_over"])
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0
    ll = log_loss(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    logger.info("\nBINARY CLASSIFIER RESULTS (test set):")
    logger.info("  Accuracy: %.1f%%", accuracy * 100)
    logger.info("  AUC-ROC:  %.4f", auc)
    logger.info("  Log Loss: %.4f", ll)
    logger.info("  Brier:    %.4f", brier)

    # ── Betting sim at various thresholds ──
    results = {"test_size": len(test), "accuracy": accuracy, "auc": auc,
               "log_loss": ll, "brier": brier, "thresholds": {}}

    print(f"\n{'='*70}")
    print(f"  BETTING SIMULATION — OVER ONLY (real DK lines + odds)")
    print(f"{'='*70}")
    print(f"  {'Thresh':>8} {'Bets':>6} {'Hit%':>6} {'ROI%':>7} {'PnL':>9} {'PF':>5}")
    print(f"  {'-'*48}")

    for thresh in [0.50, 0.52, 0.54, 0.55, 0.57, 0.60, 0.65, 0.70]:
        bets = test[test["prob_over"] >= thresh].copy()
        if len(bets) < 5:
            continue

        bets["won"] = bets["actual"] > bets["line"]
        n = len(bets)
        hr = bets["won"].mean() * 100

        # PnL using real DK odds
        pnl = 0
        for _, r in bets.iterrows():
            odds = r.get("over_price", -110)
            if pd.isna(odds):
                odds = -110
            dec = american_to_decimal(odds)
            pnl += 100 * (dec - 1) if r["won"] else -100
        roi = pnl / (n * 100) * 100
        gw = sum(100 * (american_to_decimal(r.get("over_price", -110)) - 1)
                 for _, r in bets[bets["won"]].iterrows())
        gl = (~bets["won"]).sum() * 100
        pf = gw / gl if gl > 0 else 0

        print(f"  {thresh:>7.0%} {n:>6} {hr:>5.1f}% {roi:>+6.1f}% ${pnl:>8.0f} {pf:>5.2f}")
        results["thresholds"][str(thresh)] = {
            "bets": n, "hit_rate": round(hr, 1),
            "roi": round(roi, 1), "pnl": round(pnl, 2),
        }

    # ── UNDER bets too ──
    print(f"\n  UNDER ONLY:")
    print(f"  {'Thresh':>8} {'Bets':>6} {'Hit%':>6} {'ROI%':>7}")
    print(f"  {'-'*35}")
    for thresh in [0.50, 0.55, 0.60, 0.65]:
        prob_under = 1 - test["prob_over"]
        bets = test[prob_under >= thresh].copy()
        if len(bets) < 5:
            continue
        bets["won"] = bets["actual"] < bets["line"]
        hr = bets["won"].mean() * 100
        pnl = 0
        for _, r in bets.iterrows():
            odds = r.get("under_price", -110)
            if pd.isna(odds):
                odds = -110
            dec = american_to_decimal(odds)
            pnl += 100 * (dec - 1) if r["won"] else -100
        roi = pnl / (len(bets) * 100) * 100
        print(f"  {thresh:>7.0%} {len(bets):>6} {hr:>5.1f}% {roi:>+6.1f}%")

    # ── Feature importance ──
    fi = dict(zip(BINARY_FEATURES, model.feature_importances_))
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  FEATURE IMPORTANCE:")
    for name, imp in fi_sorted[:12]:
        bar = "█" * int(imp * 100)
        print(f"    {name:<22} {imp:.3f} {bar}")

    results["feature_importance"] = {k: round(v, 4) for k, v in fi_sorted}

    # ── Calibration ──
    print(f"\n  CALIBRATION:")
    for lo, hi in [(0.3, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        mask = (test["prob_over"] >= lo) & (test["prob_over"] < hi)
        s = test[mask]
        if len(s) < 10:
            continue
        actual_rate = s["over_hit"].mean() * 100
        pred_rate = s["prob_over"].mean() * 100
        print(f"    Predicted {lo*100:.0f}-{hi*100:.0f}%: actual={actual_rate:.1f}%, predicted={pred_rate:.1f}% (n={len(s)})")

    # ── Bootstrap CI on best threshold ──
    best_thresh = 0.55  # default
    bets_best = test[test["prob_over"] >= best_thresh].copy()
    if len(bets_best) >= 20:
        bets_best["won"] = bets_best["actual"] > bets_best["line"]
        rng = np.random.RandomState(42)
        rois = []
        for _ in range(1000):
            s = bets_best.sample(n=len(bets_best), replace=True, random_state=rng)
            p = 0
            for _, r in s.iterrows():
                odds = r.get("over_price", -110)
                if pd.isna(odds): odds = -110
                dec = american_to_decimal(odds)
                p += 100 * (dec - 1) if r["won"] else -100
            rois.append(p / (len(s) * 100) * 100)
        ci_lo, ci_hi = np.percentile(rois, [2.5, 97.5])
        print(f"\n  Bootstrap 95% CI for ROI at {best_thresh:.0%} threshold:")
        print(f"    [{ci_lo:.2f}%, {ci_hi:.2f}%] (n={len(bets_best)})")
        results["bootstrap_ci"] = {"lower": round(ci_lo, 2), "upper": round(ci_hi, 2)}

    # Save model and results
    root = get_project_root()
    model.save_model(str(root / "models" / "binary_classifier.json"))
    with open(root / "data" / "reports" / "binary_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    test.to_parquet(root / "data" / "reports" / "binary_test_predictions.parquet", index=False)

    logger.info("\nModel saved to models/binary_classifier.json")
    logger.info("Results saved to data/reports/binary_results.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="player_assists",
                        choices=["player_assists", "player_rebounds"])
    parser.add_argument("--train-end", default=None,
                        help="Date to split train/test (default: 70/30)")
    args = parser.parse_args()

    ensure_dirs()
    df = load_and_merge(market=args.market)

    if len(df) < 100:
        logger.warning("Only %d rows. Fetch more historical lines first.", len(df))
        logger.info("Run: python scripts/fetch_historical_lines.py")

    results = train_and_evaluate(df, train_end_date=args.train_end)
