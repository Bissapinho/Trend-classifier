import yfinance as yf


def load_data(
    ticker='SPY',
    start='2010-01-01',
    end='2024-12-31',
    interval='1d'
):
    try:
        tick = yf.Ticker(ticker)
        df = tick.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"{ticker}: no data returned")

        df = df.reset_index()
        df.rename(columns={'Date': 'Date'}, inplace=True)
        return df

    except ValueError as e:
        print("Invalid or delisted ticker :", e)
        return None

    except Exception as e:
        print("Unexpected error :", type(e).__name__, e)
        return None

