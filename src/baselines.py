import numpy as np
import pandas as pd


def days_to_cross(X: pd.DataFrame, lookback=5):
    """tau_t = -g_t / slope_t: days until the MA gap reaches zero if it kept its `lookback`-day trend.

    np.inf when the gap is moving away from zero (no crossover predicted).
    """
    gap = X['Distance_GC'].values
    slope = (X['Distance_GC'] - X['Distance_GC'].shift(lookback)).values / lookback
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = -gap / slope
    return np.where(np.isfinite(tau) & (tau > 0), tau, np.inf)


def extrapolation_rule(X: pd.DataFrame, horizon=30, lookback=5):
    """Flag day t when a linear extrapolation of the MA gap crosses zero within `horizon` days.

    Uses only Distance_GC and its own past. The lookback (5) is fixed by hand; the horizon
    equals the target horizon.
    """
    return (days_to_cross(X, lookback) <= horizon).astype(int)


def extrapolation_score(X: pd.DataFrame, lookback=5):
    """Continuous version of the rule: 1 / (1 + tau_t), 0 when no crossover is predicted.

    A decreasing function of tau_t, so it ranks days like -tau_t. The binary rule is
    score >= 1 / (1 + horizon).
    """
    return 1 / (1 + days_to_cross(X, lookback))


def stratified_expected_f1(p_train, p_test):
    """Expected F1 of a classifier predicting 1 at random with probability p_train."""
    # precision = p_test, recall = p_train
    return 2 * p_train * p_test / (p_train + p_test)
