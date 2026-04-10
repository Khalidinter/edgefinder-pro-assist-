#!/usr/bin/env python3
"""
Comprehensive evaluation: prediction accuracy, calibration, betting metrics,
bootstrap CIs, per-player analysis, feature importance, heuristic comparison.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, brier_score_loss

from lib.config import logger
from lib.backtest_utils import get_project_root, FEATURE_COLS


# ── Prediction Metrics ──
def prediction_metrics(df: pd.DataFrame) -> dict:
    pred = df["predicted_ast"].values
    actual = df["actual_ast"].values
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    median_ae = np.median(np.abs(pred - actual))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r_squared": round(r2, 4),
        "median_ae": round(median_ae, 4),
        "mean_error": round(np.mean(pred - actual), 4),
        "n_predictions": len(df),
    }


def prediction_metrics_per_season(df: pd.DataFrame) -> dict:
    result = {}
    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season]
        result[season] = prediction_metrics(sdf)
    return result


# ── Calibration ──
def calibration_analysis(df: pd.DataFrame, n_bins: int = 10) -> dict:
    bets = df[df["bet_placed"]].copy()
    if bets.empty:
        return {"brier_score": None, "buckets": {}}

    # Brier score: for over bets, y_true = (actual > line)
    y_true = (bets["actual_ast"] > bets["synthetic_line"]).astype(float).values
    y_pred = bets["over_prob"].values
    brier = brier_score_loss(y_true, y_pred)

    # Bucket calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{int(bin_edges[i]*100)}-{int(bin_edges[i+1]*100)}%" for i in range(len(bin_edges) - 1)]
    bets["prob_bucket"] = pd.cut(bets["over_prob"], bins=bin_edges, labels=bin_labels, include_lowest=True)

    buckets = {}
    for bucket, grp in bets.groupby("prob_bucket", observed=True):
        if len(grp) >= 5:
            actual_rate = (grp["actual_ast"] > grp["synthetic_line"]).mean()
            predicted_rate = grp["over_prob"].mean()
            buckets[str(bucket)] = {
                "n": len(grp),
                "predicted_over_rate": round(predicted_rate * 100, 1),
                "actual_over_rate": round(actual_rate * 100, 1),
            }

    return {"brier_score": round(brier, 4), "buckets": buckets}


def edge_monotonicity(df: pd.DataFrame) -> dict:
    """Check if higher edge → higher realized hit rate (critical test)."""
    bets = df[df["bet_placed"]].copy()
    if len(bets) < 50:
        return {"monotonic": None, "buckets": {}}

    bets["edge_bucket"] = pd.cut(bets["best_edge"].abs(), bins=[0, 3, 5, 8, 12, 100])
    buckets = {}
    for bucket, grp in bets.groupby("edge_bucket", observed=True):
        if len(grp) >= 10:
            hit_rate = grp["bet_won"].mean() * 100
            buckets[str(bucket)] = {"n": len(grp), "hit_rate": round(hit_rate, 1)}

    # Check monotonicity
    rates = [v["hit_rate"] for v in buckets.values()]
    is_mono = all(rates[i] <= rates[i+1] for i in range(len(rates)-1)) if len(rates) > 1 else None

    return {"monotonic": is_mono, "buckets": buckets}


# ── Betting Metrics ──
def betting_metrics(df: pd.DataFrame) -> dict:
    bets = df[df["bet_placed"]].copy()
    if bets.empty:
        return {}

    total_bets = len(bets)
    wins = bets["bet_won"].sum()
    losses = total_bets - wins
    hit_rate = wins / total_bets * 100

    total_pnl = bets["pnl"].sum()
    total_risked = total_bets * 100  # flat $100 bets
    roi = total_pnl / total_risked * 100

    # Daily PnL for Sharpe
    bets["game_date"] = pd.to_datetime(bets["game_date"])
    daily_pnl = bets.groupby("game_date")["pnl"].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0

    # Max drawdown
    cum_pnl = daily_pnl.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    max_drawdown = drawdown.min()

    # Profit factor
    gross_wins = bets[bets["pnl"] > 0]["pnl"].sum()
    gross_losses = abs(bets[bets["pnl"] < 0]["pnl"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    return {
        "total_bets": total_bets,
        "wins": int(wins),
        "losses": int(losses),
        "hit_rate": round(hit_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "roi": round(roi, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": round(profit_factor, 2),
    }


def betting_metrics_per_season(df: pd.DataFrame) -> dict:
    result = {}
    for season in sorted(df["season"].unique()):
        sdf = df[df["season"] == season]
        m = betting_metrics(sdf)
        if m:
            result[season] = m
    return result


# ── Bootstrap Confidence Intervals ──
def bootstrap_ci(
    df: pd.DataFrame, metric_fn, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> dict:
    """Bootstrap 95% CI for a metric computed from bet results."""
    bets = df[df["bet_placed"]].copy()
    if len(bets) < 30:
        return {"lower": None, "upper": None, "n_bets": len(bets)}

    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_bootstrap):
        sample = bets.sample(n=len(bets), replace=True, random_state=rng)
        values.append(metric_fn(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(values, alpha * 100)
    upper = np.percentile(values, (1 - alpha) * 100)

    return {
        "lower": round(lower, 3),
        "upper": round(upper, 3),
        "mean": round(np.mean(values), 3),
        "n_bets": len(bets),
    }


def roi_from_bets(bets_df):
    return bets_df["pnl"].sum() / (len(bets_df) * 100) * 100

def hit_rate_from_bets(bets_df):
    return bets_df["bet_won"].mean() * 100


# ── Per-Player Analysis ──
def per_player_analysis(df: pd.DataFrame, min_predictions: int = 20) -> pd.DataFrame:
    """Per-player MAE, hit rate, ROI for players with enough data."""
    results = []
    for pid, grp in df.groupby("player_id"):
        if len(grp) < min_predictions:
            continue
        name = grp["player_name"].iloc[0]
        mae = mean_absolute_error(grp["actual_ast"], grp["predicted_ast"])
        bets = grp[grp["bet_placed"]]
        n_bets = len(bets)
        if n_bets > 0:
            hr = bets["bet_won"].mean() * 100
            roi = bets["pnl"].sum() / (n_bets * 100) * 100
        else:
            hr, roi = 0, 0

        results.append({
            "player": name,
            "predictions": len(grp),
            "bets": n_bets,
            "mae": round(mae, 2),
            "hit_rate": round(hr, 1),
            "roi": round(roi, 1),
        })

    return pd.DataFrame(results).sort_values("mae")


# ── Full Report ──
def generate_report(df: pd.DataFrame, feature_importance: dict = None) -> dict:
    """Generate comprehensive evaluation report."""

    report = {
        "prediction": prediction_metrics(df),
        "prediction_per_season": prediction_metrics_per_season(df),
        "calibration": calibration_analysis(df),
        "edge_monotonicity": edge_monotonicity(df),
        "betting": betting_metrics(df),
        "betting_per_season": betting_metrics_per_season(df),
        "bootstrap_roi": bootstrap_ci(df, roi_from_bets),
        "bootstrap_hit_rate": bootstrap_ci(df, hit_rate_from_bets),
        "feature_importance": feature_importance or {},
    }

    # Per-player top/bottom
    player_df = per_player_analysis(df)
    if not player_df.empty:
        report["most_predictable"] = player_df.head(10).to_dict("records")
        report["least_predictable"] = player_df.tail(10).to_dict("records")

    return report


def print_report(report: dict) -> None:
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 70)
    print("  EDGEFINDER ASSIST MODEL — V1 EVALUATION REPORT")
    print("=" * 70)

    p = report["prediction"]
    print(f"\n  PREDICTION ACCURACY")
    print(f"  {'MAE:':<20} {p['mae']}")
    print(f"  {'RMSE:':<20} {p['rmse']}")
    print(f"  {'R²:':<20} {p['r_squared']}")
    print(f"  {'Median AE:':<20} {p['median_ae']}")
    print(f"  {'Mean Error:':<20} {p['mean_error']}")
    print(f"  {'Predictions:':<20} {p['n_predictions']}")

    print(f"\n  PER-SEASON ACCURACY")
    for season, m in report.get("prediction_per_season", {}).items():
        print(f"    {season}: MAE={m['mae']}, N={m['n_predictions']}")

    b = report.get("betting", {})
    if b:
        print(f"\n  BETTING METRICS (Synthetic Lines — Provisional)")
        print(f"  {'Total Bets:':<20} {b.get('total_bets', 0)}")
        print(f"  {'Hit Rate:':<20} {b.get('hit_rate', 0)}%")
        print(f"  {'ROI:':<20} {b.get('roi', 0)}%")
        print(f"  {'Sharpe:':<20} {b.get('sharpe', 0)}")
        print(f"  {'Max Drawdown:':<20} ${b.get('max_drawdown', 0)}")
        print(f"  {'Profit Factor:':<20} {b.get('profit_factor', 0)}")

    br = report.get("bootstrap_roi", {})
    if br.get("lower") is not None:
        print(f"\n  BOOTSTRAP 95% CI")
        print(f"  {'ROI CI:':<20} [{br['lower']}%, {br['upper']}%]")
        bhr = report.get("bootstrap_hit_rate", {})
        if bhr.get("lower") is not None:
            print(f"  {'Hit Rate CI:':<20} [{bhr['lower']}%, {bhr['upper']}%]")

    cal = report.get("calibration", {})
    if cal.get("brier_score") is not None:
        print(f"\n  CALIBRATION")
        print(f"  {'Brier Score:':<20} {cal['brier_score']}")
        for bucket, data in cal.get("buckets", {}).items():
            print(f"    {bucket}: predicted={data['predicted_over_rate']}%, actual={data['actual_over_rate']}% (n={data['n']})")

    em = report.get("edge_monotonicity", {})
    if em.get("buckets"):
        print(f"\n  EDGE MONOTONICITY: {'PASS' if em.get('monotonic') else 'FAIL'}")
        for bucket, data in em["buckets"].items():
            print(f"    {bucket}: hit_rate={data['hit_rate']}% (n={data['n']})")

    fi = report.get("feature_importance", {})
    if fi:
        print(f"\n  FEATURE IMPORTANCE")
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_fi[:10]:
            bar = "█" * int(imp * 100)
            print(f"    {name:<25} {imp:.3f} {bar}")

    # V1 gate checks
    print(f"\n  V1 GATE CHECKS")
    print(f"  {'MAE < 1.8:':<30} {'PASS' if p['mae'] < 1.8 else 'FAIL'} ({p['mae']})")
    print(f"  {'ROI > 0:':<30} {'PASS' if b.get('roi', -1) > 0 else 'FAIL'} ({b.get('roi', 0)}%)")
    if br.get("lower") is not None:
        print(f"  {'ROI CI > 0:':<30} {'PASS' if br['lower'] > 0 else 'FAIL'} (lower={br['lower']}%)")
    if em.get("monotonic") is not None:
        print(f"  {'Edge monotonic:':<30} {'PASS' if em['monotonic'] else 'FAIL'}")

    # Single season dominance
    bps = report.get("betting_per_season", {})
    if bps:
        total_pnl = sum(v.get("total_pnl", 0) for v in bps.values())
        for season, m in bps.items():
            pct = m.get("total_pnl", 0) / total_pnl * 100 if total_pnl != 0 else 0
            print(f"    {season}: PnL=${m.get('total_pnl', 0)}, share={pct:.0f}%")
            if abs(pct) > 60:
                print(f"    WARNING: {season} accounts for {abs(pct):.0f}% of profit")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    root = get_project_root()
    results_path = root / "data" / "reports" / "backtest_results.parquet"
    if not results_path.exists():
        logger.error("No backtest results found. Run backtest_engine.py first.")
        sys.exit(1)

    df = pd.read_parquet(results_path)
    report = generate_report(df)
    print_report(report)

    # Save JSON report
    report_path = root / "data" / "reports" / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)
