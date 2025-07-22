from kiteconnect import KiteConnect
import logging
import pandas as pd
import numpy as np
from select_filter import get_fo_stocks
from auth import get_kite_client
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO)

# Load credentials from JSON
with open("login_credentials.json") as f:
    credentials = json.load(f)

API_KEY = credentials.get("api_key")
API_SECRET = credentials.get("api_secret")
ACCESS_TOKEN = credentials.get("access_token")

# ✅ Get Kite client using access_token
def get_kite_client():
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    return kite

# ✅ Historical data fetch
def get_historical_data(kite, trading_symbol, days_back, interval="day", exchange="NSE"):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days_back)

    try:
        instrument_token = kite.ltp(f"{exchange}:{trading_symbol}")[f"{exchange}:{trading_symbol}"]["instrument_token"]

        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=start_date,
            to_date=end_date,
            interval=interval
        )
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "close"]]
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        return None

# ✅ Growth calculation
def calculate_growth(df):
    df["daily_growth"] = df["close"].pct_change() * 100
    return df

# ✅ Calculate percentage change from previous close
def change_from_previous_close_percentage(trading_symbol: str, kite: KiteConnect, exchange="NSE") -> float:
    try:
        historical_data = get_historical_data(kite, trading_symbol, 5, exchange=exchange)  # Get last 2 days of data
        if historical_data is None or len(historical_data) < 2:
            logging.error(f"Not enough data to calculate change for {trading_symbol}")
            return 0.0
        
        latest_close = historical_data["close"].iloc[-1]
        previous_close = historical_data["close"].iloc[-2]
        percentage_change = ((latest_close - previous_close) / previous_close) * 100
        return percentage_change
    except Exception as e:
        logging.error(f"Error calculating change from previous close for {trading_symbol}: {e}")
        return 0.0

# ✅ Final analysis function
def analyze_stock_growth(trading_symbol, recent_days=7, long_term_days=365, threshold=2, exchange="NSE"):
    kite = get_kite_client()

    long_term_df = get_historical_data(kite, trading_symbol, long_term_days, exchange=exchange)
    if long_term_df is None or len(long_term_df) < 2:
        logging.error("Insufficient long-term data")
        return {"error": "Insufficient long-term data"}

    recent_df = long_term_df.tail(recent_days)
    if len(recent_df) < recent_days:
        logging.error(f"Insufficient recent data for {recent_days} days")
        return {"error": f"Insufficient recent data for {recent_days} days"}

    # Defensive: Check if dataframes are valid and have enough rows
    if long_term_df is None or recent_df is None or len(long_term_df) < 2 or len(recent_df) < 2:
        logging.warning(f"{trading_symbol} skipped due to insufficient data (long_term_df: {len(long_term_df) if long_term_df is not None else 'None'}, recent_df: {len(recent_df) if recent_df is not None else 'None'})")
        return None

    try:
        long_term_df = calculate_growth(long_term_df)
        recent_df = calculate_growth(recent_df)

        # Check again after growth calculation
        if 'daily_growth' not in long_term_df or 'daily_growth' not in recent_df:
            logging.warning(f"{trading_symbol} skipped due to missing daily_growth column after calculation.")
            return None
        if len(long_term_df['daily_growth'].dropna()) < 2 or len(recent_df['daily_growth'].dropna()) < 2:
            logging.warning(f"{trading_symbol} skipped due to insufficient daily_growth data.")
            return None

        avg_daily_growth = long_term_df["daily_growth"].mean()
        std_daily_growth = long_term_df["daily_growth"].std()
        recent_total_growth = ((recent_df["close"].iloc[-1] / recent_df["close"].iloc[0]) - 1) * 100
        recent_avg_daily_growth = recent_df["daily_growth"].mean()

        # Get change from previous close percentage
        change_from_previous_close_percentage_val = change_from_previous_close_percentage(trading_symbol, kite, exchange)

        is_abnormal = recent_avg_daily_growth > (avg_daily_growth * threshold)
        z_score = (recent_avg_daily_growth - avg_daily_growth) / std_daily_growth if std_daily_growth > 0 else 0

        return {
            "stock": str(trading_symbol),
            "long_term_avg_daily_growth": round(avg_daily_growth, 2),
            "std_daily_growth": round(std_daily_growth, 2),
            "recent_total_growth (%)": round(recent_total_growth, 2),
            "recent_avg_daily_growth (%)": round(recent_avg_daily_growth, 2),
            "change_from_previous_close (%)": round(change_from_previous_close_percentage_val, 2) if change_from_previous_close_percentage_val is not None else None,
            "z_score": round(z_score, 2),
            "is_abnormal": bool(is_abnormal)
        }
    except Exception as e:
        logging.error(f"{trading_symbol} skipped due to error: {e}")
        return None

# Utility: Add a delay between API calls to avoid rate limits
import time
def throttle(delay_sec=0.6):
    time.sleep(delay_sec)

