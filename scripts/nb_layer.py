#!/usr/bin/env python3
"""
Negative Binomial probability layer with empirical Bayes shrinkage for alpha.
Converts XGBoost point predictions into calibrated over/under probabilities.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from typing import Optional


def estimate_population_alpha(
    actuals: pd.Series, predictions: pd.Series, min_samples: int = 100
) -> float:
    """
    Estimate population-level alpha from all player predictions.
    Uses method of moments: alpha = mean(pred)^2 / (var(actual) - mean(pred))
    Only uses players with enough data.
    """
    if len(actuals) < min_samples:
        return 0.35  # fallback

    mean_pred = predictions.mean()
    var_actual = actuals.var(ddof=1)

    if var_actual <= mean_pred or mean_pred <= 0:
        return 0.35

    alpha = (mean_pred ** 2) / (var_actual - mean_pred)
    return max(0.05, min(alpha, 50.0))


def estimate_player_alpha(
    player_actuals: pd.Series,
    player_predictions: pd.Series,
    population_alpha: float,
    min_games_full_weight: int = 40,
    min_games_any_weight: int = 15,
) -> float:
    """
    Per-player alpha with empirical Bayes shrinkage toward population prior.

    alpha_final = w * alpha_individual + (1 - w) * alpha_population
    w = min(games_played / min_games_full_weight, 1.0)
    """
    n = len(player_actuals)

    if n < min_games_any_weight:
        return population_alpha

    mean_pred = player_predictions.mean()
    var_actual = player_actuals.var(ddof=1)

    if var_actual <= mean_pred or mean_pred <= 0:
        alpha_individual = population_alpha
    else:
        alpha_individual = (mean_pred ** 2) / (var_actual - mean_pred)
        alpha_individual = max(0.05, min(alpha_individual, 50.0))

    w = min(n / min_games_full_weight, 1.0)
    return w * alpha_individual + (1 - w) * population_alpha


def nb_over_prob(expected_ast: float, line: float, alpha: float) -> float:
    """
    P(AST > line) using negative binomial distribution.
    line is floored: e.g., line=5.5 → P(AST >= 6).
    """
    if expected_ast <= 0 or alpha <= 0:
        return 0.0

    threshold = int(math.floor(line))  # For 5.5 → 5, so we compute P(X > 5) = P(X >= 6)
    n = alpha
    p = alpha / (alpha + expected_ast)

    # P(X > threshold) = 1 - P(X <= threshold) = 1 - CDF(threshold)
    prob = 1.0 - nbinom.cdf(threshold, n, p)
    return max(0.0, min(1.0, prob))


def nb_under_prob(expected_ast: float, line: float, alpha: float) -> float:
    """P(AST < line) — complement accounting for push."""
    return 1.0 - nb_over_prob(expected_ast, line, alpha)


class NBProbabilityLayer:
    """
    Manages per-player alpha estimation and probability computation.
    Re-initialized at each walk-forward retrain point.
    """

    def __init__(self):
        self.population_alpha: float = 0.35
        self.player_alphas: dict = {}

    def fit(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        player_ids: pd.Series,
    ) -> None:
        """
        Estimate population alpha and per-player alphas from training data.
        Only uses rows where predictions exist (post-warmup).
        """
        mask = predictions.notna() & actuals.notna()
        a, p, ids = actuals[mask], predictions[mask], player_ids[mask]

        self.population_alpha = estimate_population_alpha(a, p)

        self.player_alphas = {}
        for pid in ids.unique():
            pmask = ids == pid
            if pmask.sum() >= 15:
                self.player_alphas[pid] = estimate_player_alpha(
                    a[pmask], p[pmask], self.population_alpha
                )

    def get_alpha(self, player_id: int) -> float:
        return self.player_alphas.get(player_id, self.population_alpha)

    def predict_proba(
        self, expected_ast: float, line: float, player_id: int
    ) -> dict:
        alpha = self.get_alpha(player_id)
        over = nb_over_prob(expected_ast, line, alpha)
        return {
            "over_prob": over,
            "under_prob": 1.0 - over,
            "alpha": alpha,
        }
