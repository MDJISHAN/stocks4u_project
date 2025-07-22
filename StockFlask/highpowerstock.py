from auth import get_kite_client
from select_filter import get_fo_stocks
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import time
import json

kite = get_kite_client()
fo_stocks = get_fo_stocks()  # ✅ correct (this calls the function and returns the list)

def get_last_trading_day() -> str:
    today = datetime.today()
    # If today is Saturday (5) or Sunday (6), go back to Friday
    if today.weekday() == 5:  # Saturday
        last_trading_day = today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        last_trading_day = today - timedelta(days=2)
    else:
        last_trading_day = today
    return last_trading_day.strftime("%Y-%m-%d")

def get_instrument_tokens() -> Dict[str, int]:
    instruments = kite.instruments("NSE")
    token_map = {}
    for stock in instruments:
        if stock['tradingsymbol'] in fo_stocks:
            token_map[stock['tradingsymbol']] = stock['instrument_token']
    return token_map

def fetch_yesterday_volume(symbol: str, instrument_token: int) -> int:
    date = get_last_trading_day()
    try:
        historical = kite.historical_data(
            instrument_token,
            from_date=date,
            to_date=date,
            interval="day"
        )
        if historical:
            return historical[0]['volume']
    except Exception as e:
        print(json.dumps({"error": f"Error fetching historical data for {symbol}: {e}"}))
    return 0

def fetch_live_stock_data(symbols: List[str], token_map: Dict[str, int]) -> Dict[str, Dict]:
    data = {}
    try:
        quote_data = kite.quote([f"NSE:{symbol}" for symbol in symbols])
        for symbol in symbols:
            nse_symbol = f"NSE:{symbol}"
            if nse_symbol in quote_data:
                stock_info = quote_data[nse_symbol]
                ltp = stock_info.get("last_price", 0)
                volume = stock_info.get("volume", 0)

                if volume == 0:
                    # Fetch previous day volume
                    instrument_token = token_map.get(symbol)
                    if instrument_token:
                        volume = fetch_yesterday_volume(symbol, instrument_token)

                turnover = ltp * volume
                data[symbol] = {"ltp": ltp, "volume": volume, "turnover": turnover}
    except Exception as e:
        print(json.dumps({"error": f"Error fetching batch: {e}"}))
    return data

def get_high_turnover_stocks(stock_data: Dict[str, Dict], top_n: int) -> List[Dict]:
    turnover_list = []
    for symbol, info in stock_data.items():
        turnover_list.append({"symbol": symbol, "turnover": info["turnover"]})
    sorted_list = sorted(turnover_list, key=lambda x: x["turnover"], reverse=True)[:top_n]
    return sorted_list

def analyze_high_turnover_stocks_live(top_n: int) -> List[Dict]:
    fo_stocks = get_fo_stocks()  # ✅ correct (this calls the function and returns the list)

    all_stocks = list(fo_stocks)  # Only F&O stocks
    if not all_stocks:
        return []

    stock_data = {}
    print(json.dumps({"info": f"Total F&O stocks to process: {len(all_stocks)}"}))

    batch_size = 100  # How many stocks per API call
    delay = 1.0       # Delay in seconds between batches

    token_map = get_instrument_tokens()

    for i in range(0, len(all_stocks), batch_size):
        batch = all_stocks[i:i + batch_size]
        print(json.dumps({"info": f"Fetching live data batch {i // batch_size + 1}: {len(batch)} stocks"}))
        batch_data = fetch_live_stock_data(batch, token_map)
        stock_data.update(batch_data)
        time.sleep(delay)

    top_stocks = get_high_turnover_stocks(stock_data, top_n)
    return top_stocks

if __name__ == "__main__":
    import json
    top_n = 1000
    result = analyze_high_turnover_stocks_live(top_n)
    print(json.dumps({"top_high_turnover_stocks": result}, indent=2, default=str))
