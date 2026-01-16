import pandas as pd
import numpy as np



def add_MA(df: pd.DataFrame):
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


def add_EMA(df: pd.DataFrame, period=20):
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


def add_returns(df: pd.DataFrame):
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


def add_volatility(df: pd.DataFrame, window=20):
    """
    Add rolling volatility computed from daily returns.

    Volatility is defined as the rolling standard deviation of
    daily returns over a given time window.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'Close' column.
    window : Rolling window size (in days), by default st to 20.

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

def add_distances(df: pd.DataFrame, madist=50, emadist=20):
    """
    Add normalized distance to moving average and exponential moving average.

    Distances are computed as (Close - MA) / MA and (Close - EMA) / EMA.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'Close' column.
    madist : int, optional
        Window size for the moving average, by default 50.
    emadist : int, optional
        Span for the exponential moving average, by default 20.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with distance features added.
    """
    df = df.copy()

    ma = df["Close"].rolling(window=madist).mean()
    ema = df["Close"].ewm(span=emadist).mean()

    df[f"Distance_MA{madist}"] = (df["Close"] - ma) / ma
    df[f"Distance_EMA{emadist}"] = (df["Close"] - ema) / ema

    return df


def add_cumulated_returns(df: pd.DataFrame, period=5):
    """
    Add cumulative returns over a given time period.

    Cumulative return is defined as the compounded return
    over the last `period` days.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'Close' column.
    period : int, optional
        Number of days over which returns are accumulated, by default 5.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with cumulative returns added.
    """

    df = df.copy()

    returns = df["Close"].pct_change()
    df[f"Cumulated_Return_{period}d"] = (1 + returns).rolling(period).apply(lambda x: np.prod(x) - 1, raw=True)
    
    return df


def add_rsi(df: pd.DataFrame, period=14):
    """
    Add Relative Strength Index (RSI) indicator.

    RSI is computed using daily price changes and measures
    momentum over a given time period.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'Close' column.
    period : int, optional
        RSI lookback period, by default 14.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with RSI added.
    """
    df = df.copy()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df[f"RSI{period}"] = 100 - (100 / (1 + rs))

    return df


#Targeting

def add_target(df: pd.DataFrame, period=15, goalreturn=0.008):
    """
    Add a trend classification target based on future cumulative returns.

    The market regime is defined using the cumulative return over a future
    time horizon. Three regimes are considered:
    - Bull: future cumulative return above a positive threshold
    - Bear: future cumulative return below a negative threshold
    - Range: returns between both thresholds

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least a 'Close' column.
    period : int, optional
        Forward-looking horizon (in days) over which returns are accumulated.
    goalreturn : float, optional
        Return threshold defining bullish and bearish regimes (in decimal form).

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with a 'Trend' column added.
    """
    
    df = df.copy()

    returns = df["Close"].pct_change()

    future_cumulated_returns = (
        (1 + returns)
        .rolling(period)
        .apply(lambda x: np.prod(x) - 1, raw=True)
        .shift(-period)
    )



    df["Trend"] = future_cumulated_returns.map(
        lambda r: "Bull" if r > goalreturn else
                "Bear" if r < -goalreturn else
                "Range"
    )
       
    return df


#For practcal this func adds everything to the df

def add__all_features(df: pd.DataFrame):
    """
    Add all engineered feature columns to a price DataFrame.

    This function applies a sequence of feature engineering transformations
    commonly used in financial time series analysis. All features are computed
    using only current and past information, making the resulting DataFrame
    safe for supervised machine learning without look-ahead bias.

    The added features include:
    - Simple moving averages
    - Exponential moving averages
    - Daily returns and log-returns
    - Rolling volatility
    - Normalized distances to moving averages
    - Cumulative past returns
    - Momentum indicator (RSI)

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least a 'Close' price column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame enriched with all engineered feature columns.

    Notes
    -----
    This function does not create any target variable and should be used
    prior to target construction and train/test splitting. All NaN values
    introduced by rolling computations should be handled downstream
    (e.g., by dropping initial rows).
    """

    df = df.copy()
    
    df = add_MA(df)
    df = add_EMA(df)
    df = add_returns(df)
    df = add_volatility(df)
    df = add_distances(df)
    df = add_cumulated_returns(df)
    df = add_rsi(df)

    return df

