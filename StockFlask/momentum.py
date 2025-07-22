

import pandas as pd
from datetime import datetime, timedelta
import json
from kiteconnect import KiteConnect
from typing import List, Dict, Optional
import time
from auth import get_kite_client
from select_filter import get_fo_stocks  # ✅ only F&O stocks

from concurrent.futures import ThreadPoolExecutor

kite = get_kite_client()

# 🔁 Fetch historical data with sleep and retry
def fetch_historical_data(instrument_token, from_date, to_date, interval, kite):
    for attempt in range(3):  # Retry up to 3 times
        try:
            time.sleep(0.6)  # ⏱️ Sleep more to reduce rate limit hits
            return kite.historical_data(instrument_token, from_date, to_date, interval)
        except Exception as e:
            print(f"❌ Attempt {attempt+1} failed for token {instrument_token}: {e}")
            time.sleep(1.2)
    return []

# 📦 Get instrument token for a symbol
def get_instrument_token(symbol: str, instruments: List[Dict]) -> Optional[int]:
    try:
        for instrument in instruments:
            if instrument['tradingsymbol'] == symbol:
                return instrument['instrument_token']
    except Exception as e:
        print(f"❌ Error fetching token for {symbol}: {e}")
    return None

# 📈 Calculate momentum and % change
def calculate_momentum(data: List[Dict], interval_minutes: int) -> Optional[Dict]:
    if not data or 'date' not in data[0]:
        return None

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.sort_index()

    latest_time = df.index[-1]
    interval_time_target = latest_time - timedelta(minutes=interval_minutes)
    filtered_df = df[df.index <= interval_time_target]

    if not filtered_df.empty:
        start_price = filtered_df.iloc[-1]["close"]
        end_price = df.iloc[-1]["close"]
        momentum = round(end_price - start_price, 2)
        percent_change = round(((end_price - start_price) / start_price) * 100, 2)

        return {
            "momentum": momentum,
            "percent_change": percent_change,
            "ltp": end_price,
            "start_time": filtered_df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            "end_time": latest_time.strftime('%Y-%m-%d %H:%M:%S')
        }
    return None

# 📅 Get last trading day
def get_last_trading_day(current_date):
    while current_date.weekday() > 4:
        current_date -= timedelta(days=1)
    return current_date

# 🔍 Process stock for 5min and 15min
def process_stock_dual(symbol, from_date, to_date, kite, instruments):
    try:
        instrument_token = get_instrument_token(symbol, instruments)
        if not instrument_token:
            return None

        data = fetch_historical_data(instrument_token, from_date, to_date, "minute", kite)
        momentum_5 = calculate_momentum(data, 5)
        momentum_15 = calculate_momentum(data, 15)

        return {
            "symbol": symbol,
            "momentum_5min": momentum_5,
            "momentum_15min": momentum_15
        }
    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")
        return None

# 📊 Scan F&O stocks
def scan_fo_stocks_dual(kite):
    results_5min_positive, results_5min_negative = [], []
    results_15min_positive, results_15min_negative = [], []

    fo_stock_list = fo_stocks
    today = datetime.now()
    last_trading_day = get_last_trading_day(today)

    to_date = last_trading_day.replace(hour=15, minute=30, second=0, microsecond=0)
    from_date = to_date - timedelta(minutes=600)

    print(f"📅 Scanning data from {from_date} to {to_date}")

    try:
        instruments = kite.instruments("NSE")
    except Exception as e:
        print(f"❌ Error fetching instruments list: {e}")
        return [], [], [], []

    with ThreadPoolExecutor(max_workers=3) as executor:  # 🔁 Reduce thread count
        futures = [executor.submit(process_stock_dual, symbol, from_date, to_date, kite, instruments) for symbol in fo_stock_list]

        for future in futures:
            stock_data = future.result()
            if stock_data:
                if stock_data['momentum_5min']:
                    entry = {
                        "symbol": stock_data['symbol'],
                        **stock_data['momentum_5min']
                    }
                    (results_5min_positive if entry['momentum'] > 0 else results_5min_negative).append(entry)

                if stock_data['momentum_15min']:
                    entry = {
                        "symbol": stock_data['symbol'],
                        **stock_data['momentum_15min']
                    }
                    (results_15min_positive if entry['momentum'] > 0 else results_15min_negative).append(entry)

    return results_5min_positive, results_5min_negative, results_15min_positive, results_15min_negative

# 📋 Print Top 10 by % change
def print_top10_momentum(title, stocks):
    import json
    output = {
        "title": title,
        "top10": []
    }
    if not stocks:
        output["top10"] = []
    else:
        sorted_stocks = sorted(stocks, key=lambda x: abs(x['percent_change']), reverse=True)[:10]
        for s in sorted_stocks:
            output["top10"].append(s)
    print(json.dumps(output, indent=2, default=str))

# 🚀 Main execution
if __name__ == "__main__":
    m5_pos, m5_neg, m15_pos, m15_neg = scan_fo_stocks_dual(kite)
    combined_5min = m5_pos + m5_neg
    combined_15min = m15_pos + m15_neg
    output = {
        "top10_5min": sorted(combined_5min, key=lambda x: abs(x['percent_change']), reverse=True)[:10],
        "top10_15min": sorted(combined_15min, key=lambda x: abs(x['percent_change']), reverse=True)[:10]
    }
    print(output)
    # You can now use 'output' as a Python dictionary for further processing

