from auth import get_kite_client
from select_filter import get_fo_stocks  # function that returns a list
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from analysestockgrowth import change_from_previous_close_percentage

# ✅ Get F&O stock list
symbol_list = get_fo_stocks()  # ← call the function

# ✅ Initialize Kite client
kite = get_kite_client()

# ✅ Instrument lookup map
all_instruments = kite.instruments()
instrument_lookup = {inst['tradingsymbol']: inst for inst in all_instruments}

# ✅ Function to check breakout
def check_breakout(instrument):
    try:
        raw_symbol = instrument['tradingsymbol']
        stock_symbol = re.sub(r'\d{2}[A-Z]{3}(FUT|OPT)$', '', raw_symbol)
        instrument_token = instrument['instrument_token']

        print(f"Fetching data for {stock_symbol}...")

        to_date = datetime.now()
        from_date = to_date - timedelta(days=7)

        try:
            stock_data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
        except Exception as e:
            print(f"Error fetching historical data for {stock_symbol}: {e}")
            return None

        if len(stock_data) < 3:
            print(f"Not enough trading data for {stock_symbol}")
            return None
        print("stock_data",stock_data)

        stock_df = pd.DataFrame(stock_data)
        print("stock_df",stock_df)

        stock_df['2d_high'] = stock_df['high'].rolling(window=2).max().shift(1)
        stock_df['2d_low'] = stock_df['low'].rolling(window=2).min().shift(1)

        latest_close = stock_df['close'].iloc[-1]
        prev_close = stock_df['close'].iloc[-2]
        ltp_change_percentage = ((latest_close - prev_close) / prev_close) * 100

        latest_high = stock_df['2d_high'].iloc[-1]
        latest_low = stock_df['2d_low'].iloc[-1]

        breakout_type = None
        breakout_percentage = None
        breakout_timestamp = None
        breakout_date = None
        breakout_time = None

        if latest_close > latest_high:
            breakout_type = "High Breakout"
            breakout_percentage = ((latest_close - latest_high) / latest_high) * 100
        elif latest_close < latest_low:
            breakout_type = "Low Breakout"
            breakout_percentage = ((latest_low - latest_close) / latest_low) * 100

        if breakout_type:
            breakout_timestamp = stock_df['date'].iloc[-1]
            breakout_date = breakout_timestamp.date()
            breakout_time = breakout_timestamp.time()

            growth_percentage = change_from_previous_close_percentage(stock_symbol, kite)

            return {
                "symbol": stock_symbol,
                "breakout_type": breakout_type,
                "latest_close": latest_close,
                "latest_high": latest_high,
                "latest_low": latest_low,
                "breakout_percentage": breakout_percentage,
                "ltp_change_percentage": ltp_change_percentage,
                "growth_percentage": growth_percentage,
                "breakout_timestamp": breakout_timestamp,
                "breakout_date": breakout_date,
                "breakout_time": breakout_time
            }

        return None

    except Exception as e:
        print(f"Error processing symbol {instrument.get('tradingsymbol', 'UNKNOWN')}: {e}")
        return None

# ✅ Parallel execution using ThreadPoolExecutor
def get_breakouts(symbol_list):
    breakouts = []
    max_workers = 3
    #print("first check",symbol_list, "and")


    def process_symbol(symbol):
        time.sleep(0.4)  # To avoid hitting rate limits
        #print("second check",symbol)

        instrument = instrument_lookup.get(symbol)
        #print("third check",instrument)

        if not instrument:
            print(f"Symbol not found in instrument list: {symbol}")
            return None
        #print("fourth check_breakout", check_breakout(instrument))

        return check_breakout(instrument)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_symbol, symbol): symbol for symbol in symbol_list}

        for future in as_completed(futures):
            try:
                result = future.result()
                if result and result['breakout_type']:
                    breakouts.append(result)
            except Exception as e:
                symbol = futures[future]
                print(f"Error processing {symbol}: {e}")

    return pd.DataFrame(breakouts) if breakouts else None

# ✅ Main execution
if __name__ == "__main__":
    print("Checking F&O stocks...")

    fo_df = get_breakouts(symbol_list)

    print("\n--- Ranked F&O Stock Breakouts ---")
    if fo_df is not None:
        fo_df = fo_df.sort_values(by="growth_percentage", ascending=False)
        result_dict = fo_df.to_dict(orient="records")
        print(result_dict)
        # Optionally save:
        # fo_df.to_csv("fo_breakouts.csv", index=False)
    else:
        result_dict = {"message": "No breakouts found in F&O stocks"}
        print(result_dict)
