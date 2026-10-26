import numpy as np
import pandas as pd


def extrapolation_rule(X: pd.DataFrame, horizon=30, lookback=5):
    """Flag day t when a linear extrapolation of the MA gap crosses zero within `horizon` days.

    Uses only Distance_GC and its own past: no learned parameters.
    """
    gap = X['Distance_GC'].values
    slope = (X['Distance_GC'] - X['Distance_GC'].shift(lookback)).values / lookback
    with np.errstate(divide='ignore', invalid='ignore'):
        days_to_cross = -gap / slope
    days_to_cross = np.where(np.isfinite(days_to_cross) & (days_to_cross > 0), days_to_cross, np.inf)
    return (days_to_cross <= horizon).astype(int)


def stratified_expected_f1(p_train, p_test):
    """Expected F1 of a classifier predicting 1 at random with probability p_train."""
    # precision = p_test, recall = p_train
    return 2 * p_train * p_test / (p_train + p_test)
