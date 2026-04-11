#!/usr/bin/env python3
"""
Four probability layer approaches — comparative evaluation.

1. Raw NB (V1 baseline)
2. Isotonic-calibrated NB
3. Direct binary classifier (walk-forward)
4. NGBoost distributional model (walk-forward)

All evaluated on the same game set with standardized metrics.
"""
import sys, os, json, math, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.stats import nbinom
from pathlib import Path

from lib.config import logger
from lib.backtest_utils import (
    FEATURE_COLS, TARGET_COL, get_project_root, ensure_dirs,
    payout_at_odds, STANDARD_ODDS,
)

ROOT = get_project_root()


# ═══════════════════════════════════════════════════════════
# STANDARDIZED EVALUATION
# ═══════════════════════════════════════════════════════════
def evaluate_probs(
    name: str,
    over_probs: np.ndarray,
    actual_over: np.ndarray,
    actual_ast: np.ndarray,
    lines: np.ndarray,
    edge_thresholds: list = [3, 5, 8, 10, 15],
) -> dict:
    """Standardized evaluation for any probability vector."""

    result = {"name": name, "n": len(over_probs)}

    # Core metrics
    valid = np.isfinite(over_probs) & np.isfinite(actual_over)
    op, ao = over_probs[valid], actual_over[valid]

    result["brier"] = round(brier_score_loss(ao, op), 5)
    result["log_loss"] = round(log_loss(ao, np.clip(op, 1e-6, 1 - 1e-6)), 5)
    result["auc"] = round(roc_auc_score(ao, op), 4) if len(np.unique(ao)) > 1 else 0

    # Probability distribution stats
    result["prob_mean"] = round(np.mean(op), 4)
    result["prob_std"] = round(np.std(op), 4)
    result["prob_p10"] = round(np.percentile(op, 10), 4)
    result["prob_p90"] = round(np.percentile(op, 90), 4)

    # Calibration by bucket
    calibration = {}
    for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                   (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        mask = (op >= lo) & (op < hi)
        if mask.sum() >= 10:
            calibration[f"{int(lo*100)}-{int(hi*100)}%"] = {
                "n": int(mask.sum()),
                "predicted": round(op[mask].mean() * 100, 1),
                "actual": round(ao[mask].mean() * 100, 1),
            }
    result["calibration"] = calibration

    # Betting simulation at each threshold (OVER only + UNDER only + combined)
    result["betting"] = {}
    for thresh in edge_thresholds:
        over_edge = (op - 0.5) * 100
        under_edge = ((1 - op) - 0.5) * 100

        # OVER bets
        over_mask = over_edge >= thresh
        if over_mask.sum() >= 5:
            over_won = ao[over_mask].astype(bool)
            over_pnl = sum(payout_at_odds(STANDARD_ODDS, 100, w) for w in over_won)
            over_n = int(over_mask.sum())
            result["betting"][f"over_{thresh}%"] = {
                "bets": over_n,
                "hit_rate": round(over_won.mean() * 100, 1),
                "roi": round(over_pnl / (over_n * 100) * 100, 1),
                "pnl": round(over_pnl, 0),
            }

        # UNDER bets
        under_mask = under_edge >= thresh
        if under_mask.sum() >= 5:
            under_won = (~ao[under_mask].astype(bool))
            under_pnl = sum(payout_at_odds(STANDARD_ODDS, 100, w) for w in under_won)
            under_n = int(under_mask.sum())
            result["betting"][f"under_{thresh}%"] = {
                "bets": under_n,
                "hit_rate": round(under_won.mean() * 100, 1),
                "roi": round(under_pnl / (under_n * 100) * 100, 1),
                "pnl": round(under_pnl, 0),
            }

    # Edge monotonicity (OVER only)
    over_edge = (op - 0.5) * 100
    mono = {}
    for lo, hi in [(3, 5), (5, 8), (8, 12), (12, 15), (15, 20), (20, 100)]:
        mask = (over_edge >= lo) & (over_edge < hi)
        if mask.sum() >= 10:
            mono[f"{lo}-{hi}%"] = {
                "n": int(mask.sum()),
                "hit_rate": round(ao[mask].mean() * 100, 1),
            }
    result["edge_monotonicity"] = mono

    return result


def print_comparison(results: list) -> None:
    """Side-by-side comparison of all probability layers."""

    print(f"\n{'='*80}")
    print(f"  PROBABILITY LAYER COMPARISON")
    print(f"{'='*80}")

    # Header
    names = [r["name"] for r in results]
    print(f"\n  {'Metric':<25}", end="")
    for n in names:
        print(f" {n:>14}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in names:
        print(f" {'-'*14}", end="")
    print()

    # Core metrics
    for metric, label in [("brier", "Brier Score"),
                          ("log_loss", "Log Loss"),
                          ("auc", "AUC-ROC"),
                          ("prob_mean", "Mean Prob"),
                          ("prob_std", "Prob Std Dev"),
                          ("prob_p10", "P10"),
                          ("prob_p90", "P90")]:
        print(f"  {label:<25}", end="")
        vals = [r.get(metric, 0) for r in results]
        best = min(vals) if metric in ("brier", "log_loss") else max(vals)
        for v in vals:
            marker = " *" if v == best and metric in ("brier", "log_loss", "auc") else "  "
            print(f" {v:>12.4f}{marker}", end="")
        print()

    # Betting at 10% edge
    print(f"\n  OVER BETS @ 10% Edge:")
    for metric, label in [("bets", "Bets"), ("hit_rate", "Hit %"), ("roi", "ROI %")]:
        print(f"  {label:<25}", end="")
        for r in results:
            v = r.get("betting", {}).get("over_10%", {}).get(metric, "--")
            if isinstance(v, (int, float)):
                print(f" {v:>14}", end="")
            else:
                print(f" {'--':>14}", end="")
        print()

    # Edge monotonicity
    print(f"\n  OVER Edge Monotonicity:")
    all_buckets = set()
    for r in results:
        all_buckets.update(r.get("edge_monotonicity", {}).keys())
    for bucket in sorted(all_buckets):
        print(f"  {bucket:<25}", end="")
        for r in results:
            data = r.get("edge_monotonicity", {}).get(bucket, {})
            hr = data.get("hit_rate", "--")
            n = data.get("n", 0)
            if isinstance(hr, (int, float)):
                print(f" {hr:>8.1f}% n={n:<3}", end="")
            else:
                print(f" {'--':>14}", end="")
        print()

    # Calibration
    print(f"\n  Calibration (predicted → actual over %):")
    all_cal_buckets = set()
    for r in results:
        all_cal_buckets.update(r.get("calibration", {}).keys())
    for bucket in sorted(all_cal_buckets):
        print(f"  {bucket:<25}", end="")
        for r in results:
            data = r.get("calibration", {}).get(bucket, {})
            if data:
                print(f" {data['actual']:>5.1f}% (n={data['n']:<4})", end="")
            else:
                print(f" {'--':>14}", end="")
        print()


# ═══════════════════════════════════════════════════════════
# APPROACH 1: RAW NB (V1 BASELINE)
# ═══════════════════════════════════════════════════════════
def approach_raw_nb(df: pd.DataFrame, test_mask: np.ndarray) -> np.ndarray:
    """Use the existing NB probabilities from the V1 backtest."""
    return df.loc[test_mask, "over_prob"].values


# ═══════════════════════════════════════════════════════════
# APPROACH 2: ISOTONIC-CALIBRATED NB
# ═══════════════════════════════════════════════════════════
def approach_isotonic(df: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray) -> np.ndarray:
    """Train isotonic regression on NB probs, apply to test."""
    train_probs = df.loc[train_mask, "over_prob"].values
    train_hits = df.loc[train_mask, "over_hit"].values

    iso = IsotonicRegression(y_min=0.01, y_max=0.99)
    iso.fit(train_probs, train_hits)

    test_probs = df.loc[test_mask, "over_prob"].values
    calibrated = iso.predict(test_probs)
    return calibrated


# ═══════════════════════════════════════════════════════════
# APPROACH 3: DIRECT BINARY CLASSIFIER (walk-forward)
# ═══════════════════════════════════════════════════════════
def approach_binary_direct(
    df: pd.DataFrame, test_mask: np.ndarray,
    retrain_days: int = 30,
) -> np.ndarray:
    """Walk-forward binary classifier with synthetic line as feature."""

    binary_features = FEATURE_COLS + ["synthetic_line", "pred_minus_line"]
    df = df.copy()
    df["pred_minus_line"] = df["predicted_ast"] - df["synthetic_line"]

    test_df = df[test_mask].copy()
    test_dates = sorted(test_df["game_date"].unique())

    probs = np.full(test_mask.sum(), 0.5)
    model = None
    last_retrain = None
    retrain_count = 0

    for current_date in test_dates:
        needs_retrain = (
            model is None or
            last_retrain is None or
            (current_date - last_retrain).days >= retrain_days
        )

        if needs_retrain:
            train = df[df["game_date"] < current_date]
            if len(train) < 3000:
                continue

            X_tr = train[binary_features].values
            y_tr = train["over_hit"].values

            depth = 3 if len(train) < 10000 else 4 if len(train) < 30000 else 5
            model = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                max_depth=depth, n_estimators=300, learning_rate=0.05,
                min_child_weight=max(10, len(train) // 300),
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=5.0, gamma=0.1,
                random_state=42, verbosity=0,
            )
            model.fit(X_tr, y_tr)
            last_retrain = current_date
            retrain_count += 1

        if model is None:
            continue

        today_mask = test_df["game_date"] == current_date
        today_idx = test_df[today_mask].index
        if len(today_idx) == 0:
            continue

        X_today = test_df.loc[today_idx, binary_features].values
        p = model.predict_proba(X_today)[:, 1]

        # Map back to the probs array
        for i, idx in enumerate(today_idx):
            pos = test_df.index.get_loc(idx)
            probs[pos] = p[i]

    logger.info("  Binary classifier: %d retrains", retrain_count)
    return probs


# ═══════════════════════════════════════════════════════════
# APPROACH 4: NGBOOST
# ═══════════════════════════════════════════════════════════
def approach_ngboost(
    df: pd.DataFrame, test_mask: np.ndarray,
    retrain_days: int = 30,
) -> np.ndarray:
    """Walk-forward NGBoost with learned distribution."""
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Poisson
    except ImportError:
        logger.warning("NGBoost not installed. Skipping. Install with: pip install ngboost")
        return None

    test_df = df[test_mask].copy()
    test_dates = sorted(test_df["game_date"].unique())
    lines = test_df["synthetic_line"].values

    probs = np.full(test_mask.sum(), 0.5)
    model = None
    last_retrain = None
    retrain_count = 0

    for current_date in test_dates:
        needs_retrain = (
            model is None or
            last_retrain is None or
            (current_date - last_retrain).days >= retrain_days
        )

        if needs_retrain:
            train = df[df["game_date"] < current_date]
            if len(train) < 3000:
                continue

            X_tr = train[FEATURE_COLS].values
            y_tr = train[TARGET_COL].values.astype(float)

            model = NGBRegressor(
                Dist=Poisson,
                n_estimators=300,
                learning_rate=0.05,
                minibatch_frac=0.8,
                verbose=False,
                random_state=42,
            )
            model.fit(X_tr, y_tr)
            last_retrain = current_date
            retrain_count += 1

        if model is None:
            continue

        today_mask = test_df["game_date"] == current_date
        today_idx = test_df[today_mask].index
        if len(today_idx) == 0:
            continue

        X_today = test_df.loc[today_idx, FEATURE_COLS].values
        dist = model.pred_dist(X_today)

        for i, idx in enumerate(today_idx):
            pos = test_df.index.get_loc(idx)
            line = test_df.loc[idx, "synthetic_line"]
            # P(X > line) = 1 - CDF(floor(line))
            probs[pos] = 1.0 - dist[i].cdf(int(math.floor(line)))

    logger.info("  NGBoost: %d retrains", retrain_count)
    return probs


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    ensure_dirs()

    # Load V1 backtest results (has predicted_ast, over_prob, synthetic_line, actual_ast)
    bt = pd.read_parquet(ROOT / "data" / "reports" / "backtest_results.parquet")
    bt["game_date"] = pd.to_datetime(bt["game_date"])
    bt = bt.sort_values("game_date").reset_index(drop=True)

    # Apply strict bias correction to predicted_ast
    bt_by_player = bt.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    grp = bt_by_player.groupby("player_id")
    bt_by_player["pbias"] = grp["error"].transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
    bt_by_date = bt_by_player.sort_values("game_date").reset_index(drop=True)
    bt_by_date["gbias"] = bt_by_date["error"].shift(1).expanding(min_periods=100).mean()
    bt_by_date["corrected_ast"] = bt_by_date["predicted_ast"] - bt_by_date["pbias"].fillna(bt_by_date["gbias"])
    bt = bt_by_date.copy()

    # Recompute NB over_prob with bias-corrected predictions + capped alpha
    def cap_a(a, p):
        m, v = p.mean(), a.var(ddof=1)
        if v <= m or m <= 0: return 0.5
        return max(0.5, min((m ** 2) / (v - m), 8.0))

    pa = {pid: cap_a(g["actual_ast"], g["corrected_ast"])
          for pid, g in bt.groupby("player_id") if len(g) >= 20}
    pop_a = np.median(list(pa.values()))

    def nb_op(exp, line, alpha):
        if exp <= 0 or alpha <= 0: return 0.5
        return max(0, min(1, 1 - nbinom.cdf(int(math.floor(line)), alpha, alpha / (alpha + exp))))

    bt["over_prob"] = [
        nb_op(r["corrected_ast"], r["synthetic_line"], pa.get(int(r["player_id"]), pop_a))
        if pd.notna(r["corrected_ast"]) else 0.5
        for _, r in bt.iterrows()
    ]
    bt["over_hit"] = (bt["actual_ast"] > bt["synthetic_line"]).astype(int)

    # Also load feature matrix for binary classifier and NGBoost
    fm = pd.read_parquet(ROOT / "data" / "features" / "feature_matrix.parquet")
    fm["game_date"] = pd.to_datetime(fm["game_date"])

    # Merge features into bt
    import re
    def norm(n):
        if not n: return ""
        s = str(n).lower().strip()
        s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
        s = s.replace("-", " ")
        s = re.sub(r"[^a-z\s]", "", s)
        return re.sub(r"\s+", " ", s).strip()

    bt["np"] = bt["player_name"].apply(norm)
    fm["np"] = fm["player_name"].apply(norm)

    merged = bt.merge(
        fm[["np", "game_date"] + FEATURE_COLS + [TARGET_COL]],
        on=["np", "game_date"], how="inner", suffixes=("", "_fm"),
    )
    # Use feature matrix target if available
    if f"{TARGET_COL}_fm" in merged.columns:
        merged[TARGET_COL] = merged[f"{TARGET_COL}_fm"].fillna(merged[TARGET_COL])

    logger.info("Merged bt+features: %d rows", len(merged))

    # Time-based 70/30 split
    split = int(len(merged) * 0.7)
    train_mask = pd.Series([True] * split + [False] * (len(merged) - split))
    test_mask = ~train_mask

    test_dates = merged.loc[test_mask, "game_date"]
    logger.info("Test period: %s to %s (%d rows)",
                test_dates.min().date(), test_dates.max().date(), test_mask.sum())

    # Common test targets
    test_over_hit = merged.loc[test_mask, "over_hit"].values.astype(float)
    test_actual_ast = merged.loc[test_mask, "actual_ast"].values
    test_lines = merged.loc[test_mask, "synthetic_line"].values

    all_results = []

    # ── Approach 1: Raw NB ──
    logger.info("\n--- Approach 1: Raw NB (bias-corrected, capped alpha) ---")
    raw_probs = approach_raw_nb(merged, test_mask)
    r1 = evaluate_probs("Raw NB", raw_probs, test_over_hit, test_actual_ast, test_lines)
    all_results.append(r1)

    # ── Approach 2: Isotonic-calibrated NB ──
    logger.info("\n--- Approach 2: Isotonic Calibration ---")
    iso_probs = approach_isotonic(merged, train_mask.values, test_mask.values)
    r2 = evaluate_probs("Isotonic NB", iso_probs, test_over_hit, test_actual_ast, test_lines)
    all_results.append(r2)

    # ── Approach 3: Direct binary classifier ──
    logger.info("\n--- Approach 3: Direct Binary Classifier (walk-forward) ---")
    bin_probs = approach_binary_direct(merged, test_mask.values, retrain_days=30)
    r3 = evaluate_probs("Binary XGB", bin_probs, test_over_hit, test_actual_ast, test_lines)
    all_results.append(r3)

    # ── Approach 4: NGBoost ──
    logger.info("\n--- Approach 4: NGBoost (walk-forward) ---")
    ngb_probs = approach_ngboost(merged, test_mask.values, retrain_days=30)
    if ngb_probs is not None:
        r4 = evaluate_probs("NGBoost", ngb_probs, test_over_hit, test_actual_ast, test_lines)
        all_results.append(r4)

    # ── Print comparison ──
    print_comparison(all_results)

    # ── Save ──
    report_path = ROOT / "data" / "reports" / "probability_layer_comparison.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nComparison saved to %s", report_path)


if __name__ == "__main__":
    main()
