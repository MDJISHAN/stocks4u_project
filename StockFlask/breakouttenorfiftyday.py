import time
from auth import get_kite_client
from select_filter import get_fo_stocks 
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures

kite = get_kite_client()

# Cache NSE instruments
nse_instruments = {ins['tradingsymbol']: ins for ins in kite.instruments("NSE")}

# Fetch instrument data from cache
def get_instrument_data(tradingsymbol):
    return nse_instruments.get(tradingsymbol)

# Breakout checker with retry and throttling
def check_breakout(tradingsymbol, breakout_type, retry=2):
    try:
        instrument = get_instrument_data(tradingsymbol)
        if not instrument:
            return None

        # Filter out undesirable symbols
        if any(bad in tradingsymbol for bad in ['-SG', '-GB', '-N', '-SM', '-BZ']):
            return None

        instrument_token = instrument['instrument_token']
        to_date = datetime.now()
        from_date = to_date - timedelta(days=70 if breakout_type == "10d" else 100)

        time.sleep(0.4)  # throttle API

        stock_data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval="day"
        )

        if not stock_data:
            return None

        stock_df = pd.DataFrame(stock_data)

        if breakout_type == "50d":
            stock_df['high_val'] = stock_df['high'].rolling(window=50).max().shift(1)
            stock_df['low_val'] = stock_df['low'].rolling(window=50).min().shift(1)
        elif breakout_type == "10d":
            stock_df['high_val'] = stock_df['high'].rolling(window=10).max().shift(1)
            stock_df['low_val'] = stock_df['low'].rolling(window=10).min().shift(1)
        else:
            return None

        for idx in reversed(range(len(stock_df))):
            high_val = stock_df['high_val'].iloc[idx]
            low_val = stock_df['low_val'].iloc[idx]
            if pd.isna(high_val) or pd.isna(low_val):
                continue
            close = stock_df['close'].iloc[idx]
            breakout_date = stock_df['date'].iloc[idx].date()

            if close > high_val:
                return {
                    "symbol": tradingsymbol,
                    "breakout_type": f"{breakout_type} High Breakout",
                    "percentage": ((close - high_val) / high_val) * 100,
                    "close": close,
                    "date": breakout_date
                }
            elif close < low_val:
                return {
                    "symbol": tradingsymbol,
                    "breakout_type": f"{breakout_type} Low Breakout",
                    "percentage": ((low_val - close) / low_val) * 100,
                    "close": close,
                    "date": breakout_date
                }

        return None

    except Exception as e:
        if retry > 0:
            time.sleep(2)
            return check_breakout(tradingsymbol, breakout_type, retry - 1)
        print(f"Error checking {tradingsymbol}: {e}")
        return None

# Run check in parallel with limited threads
def run_parallel_breakouts(stocks, breakout_type):
    breakouts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(check_breakout, stock, breakout_type) for stock in stocks]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                breakouts.append(result)
    return breakouts

# Pretty print breakout results
def print_breakouts(title, results):
    sorted_results = sorted(results, key=lambda x: x['date'], reverse=True)
    
    print(f"\n📈 {title} ({len(sorted_results)} Breakouts)")
    print("=" * 80)
    print(f"{'Symbol':>10} | {'Breakout Type':20} | {'Date':<10} | {'Close':>10} | {'Change':>10}")
    print("-" * 80)
    for r in sorted_results:
        print(f"{r['symbol']:>10} | {r['breakout_type']:20} | {r['date']} | {r['close']:10.2f} | {r['percentage']:>8.2f}%")

# -------------------------------

