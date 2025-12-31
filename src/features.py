import pandas as pd
import numpy as np



def add_MA(df):
    """Add simple moving averages (MA10 and MA50) based on daily closing prices.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'Close' column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with MA10 and MA50 columns added.
    """
    
    df = df.copy()

    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()

    return df


def add_EMA20(df, period=20):
    """Add Exp moving average based on daily closing prices.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'Close' column.
    
    period : the span of the EMA, by default set to 20

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with EMA column added.
    """

    df = df.copy()

    df[f'EMA{period}'] = df['Close'].ewm(span=period).mean()
    return df


def add_returns(df):
    """
    Add daily returns and log-returns based on closing prices.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'Close' column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with 'Return' and 'Log Return' columns added.
    """
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["Log Return"] = np.log(1 + df["Return"])

    return df


def add_volatility(df, window=20):
    """
    Add rolling volatility computed from daily returns.

    Volatility is defined as the rolling standard deviation of
    daily returns over a given time window.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'Close' column.
    window : int, optional
        Rolling window size (in days), by default 20.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with a 'Volatility' column added.
    """
    df = df.copy()

    # Compute daily returns locally
    returns = df["Close"].pct_change()

    # Rolling volatility
    df["Volatility"] = returns.rolling(window=window).std()

    return df
