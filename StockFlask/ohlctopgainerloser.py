import time
import json
import pandas as pd
import datetime
from kiteconnect import KiteConnect
from select_filter import get_fo_stocks  # ✅ Only F&O stocks needed

# Load credentials
with open("login_credentials.json", "r") as file:
    credentials = json.load(file)

api_key = credentials["api_key"]
#access_token = credentials["access_token"]
access_token= "lzPpBp2quxuT4MCH0fhl3h9fhgEauZJ9"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Fetch NSE instruments
print("Fetching instruments... Please wait.")
instruments = kite.instruments("NSE")
symbol_to_token = {inst['tradingsymbol']: inst['instrument_token'] for inst in instruments}

# Filter symbols to only F&O stocks available in NSE instruments
fo_symbols = [symbol for symbol in get_fo_stocks() if symbol in symbol_to_token]

# Fetch live OHLC in batches
def fetch_live_ohlc(symbols):
    batch_size = 200
    ohlc_data = {}

    for i in range(0, len(symbols), batch_size):
        batch = ["NSE:" + symbol for symbol in symbols[i:i + batch_size]]
        try:
            response = kite.quote(batch)
            for symbol_full, data in response.items():
                symbol = symbol_full.split(":")[1]
                ohlc_data[symbol] = {
                    "last_price": data['last_price'],
                    "prev_close": data['ohlc']['close']
                }
        except Exception as e:
            print(f"Error fetching live data batch {i // batch_size + 1}: {e}")
        time.sleep(0.4)  # to avoid hitting API limits

    return ohlc_data

# Fetch previous day's OHLC for fallback
def fetch_previous_day_ohlc(symbols):
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    if today.weekday() == 0:  # Monday
        yesterday = today - datetime.timedelta(days=3)

    from_date = to_date = yesterday
    interval = "day"
    ohlc_data = {}

    for symbol in symbols:
        try:
            instrument_token = symbol_to_token.get(symbol)
            if instrument_token:
                historical = kite.historical_data(
                    instrument_token,
                    from_date,
                    to_date,
                    interval
                )
                if historical:
                    day_data = historical[0]
                    ohlc_data[symbol] = {
                        "last_price": day_data['close'],
                        "prev_close": day_data['open']  # open used as reference
                    }
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
        time.sleep(0.5)  # historical API is more sensitive

    return ohlc_data

# Main method to combine both sources
def get_final_ohlc_data(symbols):
    live_data = fetch_live_ohlc(symbols)

    # Find missing symbols
    missing_symbols = [symbol for symbol in symbols if symbol not in live_data or live_data[symbol]['prev_close'] == 0]

    if missing_symbols:
        print(f"Fetching fallback historical data for {len(missing_symbols)} symbols...")
        historical_data = fetch_previous_day_ohlc(missing_symbols)
        live_data.update(historical_data)

    return live_data

# Get top gainers and losers
def get_top_gainers_and_losers():
    symbols = fo_symbols  # ✅ only F&O symbols
    ohlc_data = get_final_ohlc_data(symbols)

    stock_data = []
    for symbol, data in ohlc_data.items():
        try:
            last_price = data['last_price']
            prev_close = data['prev_close']

            if prev_close and last_price:
                change_pct = ((last_price - prev_close) / prev_close) * 100
                stock_data.append({"symbol": symbol, "change_pct": change_pct})
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    df = pd.DataFrame(stock_data)
    top_gainers = df.sort_values(by="change_pct", ascending=False).head(60)
    top_losers = df.sort_values(by="change_pct", ascending=True).head(60)

    return top_gainers.to_dict(orient="records"), top_losers.to_dict(orient="records")

# 🚀 Run Gainers/Losers scan for F&O
top_gainers, top_losers = get_top_gainers_and_losers()

# Display results
output = {
    "top_gainers": top_gainers,
    "top_losers": top_losers
}
print(output)
# You can now use 'output' as a Python dictionary for further processing

