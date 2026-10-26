import numpy as np
import pandas as pd


def compute_transition(df_raw: pd.DataFrame, horizon=30):
    """y_t = 1 if the MA50/MA200 regime changes within t+1..t+horizon.

    Also returns the row indices of genuine crossovers (MA200 defined on both sides),
    so that events can be counted instead of rows.
    """
    ma50 = df_raw['Close'].rolling(50).mean()
    ma200 = df_raw['Close'].rolling(200).mean()
    gc = (ma50 > ma200).astype(int)

    transition = np.zeros(len(df_raw), dtype=int)
    crossings = []
    label = gc.iloc[0]
    for i in range(1, len(gc)):
        if gc.iloc[i] != label:
            label = gc.iloc[i]
            transition[max(0, i - horizon):i] = 1
            # the first "change" happens when MA200 becomes defined: not a real crossover
            if not np.isnan(ma200.iloc[i - 1]):
                crossings.append(i)

    return pd.Series(transition, name='Transition'), crossings


def regime(close: pd.Series, short=50, long=200):
    return (close.rolling(short).mean() > close.rolling(long).mean()).astype(int)


def return_target(close: pd.Series, period=60, goalreturn=0.05):
    """First target tried: 1 if the return over the next `period` days exceeds `goalreturn`."""
    return ((close.shift(-period) / close - 1) > goalreturn).astype(int)


def noisy_regime(close: pd.Series, sigma, rng, short=50, long=200):
    """Regime recomputed after multiplying prices by (1 + e), e ~ N(0, sigma^2).

    The noise is relative to each day's price, so it uses no information from other dates.
    """
    noisy = close * (1 + rng.normal(0, sigma, len(close)))
    return regime(noisy, short, long)


# Label helpers used by notebooks/01_target_testing.ipynb (string labels, 'Trend' column)

def add_target(df: pd.DataFrame, period=60, goalreturn=0.05):
    """Return-based target as a 'Trend' column ('Bullish' / 'Non-Bullish')."""
    df = df.copy()
    df['Trend'] = return_target(df['Close'], period, goalreturn).map({1: 'Bullish', 0: 'Non-Bullish'})
    return df


def add_target_ma_cross(df: pd.DataFrame, short_window=50, long_window=200):
    """MA regime as a 'Trend' column; NaN while the long average is undefined."""
    df = df.copy()
    label = regime(df['Close'], short_window, long_window).map({1: 'Bullish', 0: 'Non-Bullish'})
    df['Trend'] = label.where(df['Close'].rolling(long_window).mean().notna())
    return df
