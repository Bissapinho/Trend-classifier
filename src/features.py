import pandas as pd
import numpy as np
import yfinance as yf


COLUMN_ORDER = [
    'Return', 'Volatility', 'Cumulated_Return_5d', 'RSI14',
    'Volume_ROC', 'ATR', 'VIX_spike', 'Distance_GC',
    'MA_velocity', 'MA50_slope', 'Distance_normalized', 'MA_cross_momentum'
]


def add_returns(df: pd.DataFrame):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    return df


def add_volatility(df: pd.DataFrame, window=20):
    df = df.copy()
    returns = df["Close"].pct_change()
    df["Volatility"] = returns.rolling(window=window).std()
    return df


def add_cumulated_returns(df: pd.DataFrame, period=5):
    df = df.copy()
    returns = df["Close"].pct_change()
    df[f"Cumulated_Return_{period}d"] = (1 + returns).rolling(period).apply(lambda x: np.prod(x) - 1, raw=True)
    return df


def add_rsi(df: pd.DataFrame, period=14):
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df[f"RSI{period}"] = 100 - (100 / (1 + rs))
    return df


def add_atr(df: pd.DataFrame, period=14):
    df1 = df.copy()
    high_low = df1['High'] - df1['Low']
    high_close = abs(df1['High'] - df1['Close'].shift(1))
    low_close = abs(df1['Low'] - df1['Close'].shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df1['ATR'] = true_range.rolling(window=period).mean()
    return df1


def add_volume_roc(df: pd.DataFrame, period=14):
    df1 = df.copy()
    df1['Volume_ROC'] = df1['Volume'].pct_change(periods=period) * 100
    return df1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Base features
    df = add_returns(df)
    df = add_volatility(df)
    df = add_cumulated_returns(df)
    df = add_rsi(df)
    df = add_volume_roc(df)
    df = add_atr(df)

    # MA-based features
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    df['Distance_GC'] = (df['MA50'] - df['MA200']) / df['MA200']
    df['MA_velocity'] = (df['MA50'] - df['MA50'].shift(5)) - (df['MA200'] - df['MA200'].shift(5))
    df['MA50_slope'] = df['MA50'].diff(10) / df['MA50']
    df['Distance_normalized'] = df['Distance_GC'] / df['Volatility'].rolling(50).mean()
    df['MA50_accel'] = df['MA50'].diff(5) - df['MA50'].diff(10)
    df['MA_cross_momentum'] = df['MA50_accel'] / df['Distance_GC'].abs()

    # VIX spike, we fetch and merge on date
    vix_data = yf.Ticker("^VIX").history(start="2000-01-01", end="2024-12-31")
    vix_close = vix_data['Close'].rename('VIX')
    vix_close.index = vix_close.index.tz_localize(None)

    date_col = pd.to_datetime(df['Date'])
    if date_col.dt.tz is not None:
        date_col = date_col.dt.tz_localize(None)

    df = df.set_index(date_col)
    df = df.drop(columns=['Date'], errors='ignore')
    df = df.join(vix_close, how='left')
    df = df.reset_index(drop=True)

    df['VIX_spike'] = df['VIX'] / df['VIX'].rolling(60).mean()

    # Drop everything not in COLUMN_ORDER
    to_drop = [
        'Close', 'High', 'Low', 'Open', 'MA50', 'MA200', 'VIX', 'MA50_accel',
        'Volume', 'Dividends', 'Stock Splits', 'Capital Gains',
        'Golden_Cross', 'Date'
    ]
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

    return df[COLUMN_ORDER]
