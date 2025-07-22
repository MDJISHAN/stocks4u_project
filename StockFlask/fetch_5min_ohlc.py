from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from auth import get_kite_client
from select_filter import get_fo_stocks
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

kite = get_kite_client()

# Helper to get NSE instruments

def get_nse_instruments():
    instruments = kite.instruments("NSE")
    return {inst['tradingsymbol']: inst for inst in instruments if inst['segment'] == 'NSE'}

# Fetch 5min OHLC for a single stock
def fetch_5min_ohlc(stock, token, from_date, to_date):
    try:
        data = kite.historical_data(token, from_date, to_date, interval="5minute")
        if data:
            df = pd.DataFrame(data)
            df['stock'] = stock
            return df
        return None
    except Exception as e:
        print(f"[ERROR] {stock} ({token}) 5min fetch failed: {e}")
        return None

# Batch process stocks for 5min OHLC
def batch_fetch_5min_ohlc(stocks, nse_instruments, from_date, to_date, max_workers=4):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for stock in stocks:
            if stock in nse_instruments:
                token = nse_instruments[stock]['instrument_token']
                futures[executor.submit(fetch_5min_ohlc, stock, token, from_date, to_date)] = stock
        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                results.append(df)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

if __name__ == "__main__":
    # Use the current local time for to_date
    to_date = datetime.now()
    from_date = to_date - timedelta(days=5)  # last 5 days
    nse_instruments = get_nse_instruments()
    stocks = [s for s in get_fo_stocks() if s in nse_instruments]
    print(f"[DEBUG] Fetching 5min OHLC for {len(stocks)} stocks from {from_date.date()} to {to_date.date()}")
    df = batch_fetch_5min_ohlc(stocks, nse_instruments, from_date, to_date, max_workers=8)
    print(df.head())
    df.to_csv("5min_ohlc_all_stocks.csv", index=False)
    print("[DEBUG] Saved 5min OHLC data to 5min_ohlc_all_stocks.csv")
