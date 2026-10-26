from pathlib import Path

import pandas as pd
import yfinance as yf

# yfinance's adjusted prices differ by ~1e-4 $ between calls; caching makes runs reproducible
CACHE_DIR = Path(__file__).resolve().parent.parent / 'data'


def load_data(
    ticker='SPY',
    start='2000-01-01',
    end='2024-12-31',
    interval='1d',
    cache=True
    ):
    path = CACHE_DIR / f"{ticker.replace('^', '')}_{start}_{end}_{interval}.csv"
    if cache and path.exists():
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'].str[:10])
        return df

    try:
        tick = yf.Ticker(ticker)
        df = tick.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"{ticker}: no data returned")

        df = df.reset_index()
        if cache:
            CACHE_DIR.mkdir(exist_ok=True)
            df.to_csv(path, index=False, float_format='%.10g')
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'].str[:10])
        return df

    except ValueError as e:
        print("Invalid or delisted ticker :", e)
        return None

    except Exception as e:
        print("Unexpected error :", type(e).__name__, e)
        return None
