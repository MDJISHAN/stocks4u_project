import datetime
import time
import pandas as pd
import logging
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from kiteconnect import KiteConnect
from auth import get_kite_client
from select_filter import get_fo_stocks  # Ensure this is a list of strings like ["RELIANCE", "TCS"]
fo_stocks = get_fo_stocks()
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="stock_analysis.log"
)

# Initialize Kite API
kite = get_kite_client()

def get_previous_trading_day(offset_days: int = 0) -> datetime.date:
    today = datetime.date.today() - datetime.timedelta(days=offset_days)
    while today.weekday() > 4:  # Skip Saturday/Sunday
        today -= datetime.timedelta(days=1)
    return today

def get_historical_data(instrument_token: int, interval: str = "15minute", lookback_days: int = 6) -> pd.DataFrame:
    try:
        time.sleep(0.35)  # Throttle: 3 requests/sec max
        from_date = (datetime.datetime.today() - datetime.timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        to_date = datetime.datetime.today().strftime('%Y-%m-%d')

        data = kite.historical_data(instrument_token, from_date, to_date, interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        logging.error(f"Error fetching data for token {instrument_token}: {e}")
        print(f"[ERROR fetching data] Token: {instrument_token} | {e}")
        return pd.DataFrame()

def get_last_trading_date(df: pd.DataFrame) -> datetime.date:
    if df.empty:
        return None
    return df["date"].dt.date.max()

def calculate_lom(df: pd.DataFrame, momentum_period: int = 10) -> tuple:
    if len(df) < momentum_period + 1:
        return None, None
    momentum = (df["close"] - df["close"].shift(momentum_period)) / df["close"].shift(momentum_period)
    lom = (momentum - momentum.shift(1)) * 100
    last_valid_index = lom.last_valid_index()
    if last_valid_index is not None:
        return lom.iloc[last_valid_index], df["date"].iloc[last_valid_index]
    return None, None

def get_symbol_token_map(symbols: List[str]) -> List[Dict]:
    if not all(isinstance(s, str) for s in symbols):
        logging.error("Symbols must be a list of strings.")
        raise TypeError("Expected list of strings in 'symbols'")

    all_instruments = kite.instruments("NSE")
    symbol_set = set(symbols)
    matched = [s for s in all_instruments if s["tradingsymbol"] in symbol_set]
    found_symbols = {s["tradingsymbol"] for s in matched}
    not_found = symbol_set - found_symbols

    if not_found:
        logging.warning(f"Symbols not found in instruments list: {not_found}")
        print(f"[WARNING] Symbols not found: {not_found}")

    return matched

def process_stock(stock: Dict, interval: str, days: int) -> Dict:
    instrument_token = stock["instrument_token"]
    df = get_historical_data(instrument_token, interval, days)
    print(f"[PROCESSING] {stock['tradingsymbol']} | Rows: {len(df)}")

    if df.empty or len(df) < 20:
        return None

    last_trading_date = get_last_trading_date(df)
    if not last_trading_date:
        return None

    lom_value, lom_time = calculate_lom(df)
    if lom_value is None:
        return None

    ltp = df["close"].iloc[-1]
    previous_close = df["close"].iloc[-2] if len(df) >= 2 else ltp
    ltp_change_percent = round(((ltp - previous_close) / previous_close) * 100, 2)
    lom_time_str = lom_time.strftime("%Y-%m-%d %H:%M:%S") if lom_time else "N/A"

    return {
        "Stock": stock["tradingsymbol"],
        "LOM (%)": round(lom_value, 2),
        "LOM Time": lom_time_str,
        "LTP": ltp,
        "Change (%)": ltp_change_percent
    }

def analyze_stocks(symbols: List[str], interval: str = "15minute", days: int = 6, max_workers: int = 5) -> List[Dict]:
    try:
        if not all(isinstance(s, str) for s in symbols):
            symbols = [s["symbol"] for s in symbols if isinstance(s, dict) and "symbol" in s]

        lom_results = []
        stocks = get_symbol_token_map(symbols)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_stock, stock, interval, days) for stock in stocks]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        lom_results.append(result)
                except Exception as e:
                    logging.error(f"[ERROR in thread] {e}")
                    print(f"[ERROR] {e}")

        return lom_results
    except Exception as e:
        logging.error(f"[analyze_stocks ERROR] {e}")
        print(f"[ERROR] analyze_stocks: {e}")
        return []

def momentum_results_json(title: str, results: List[Dict]):
    if not results:
        return {
            "title": title,
            "positive_momentum": [],
            "negative_momentum": [],
            "summary": "No results to display"
        }
    df = pd.DataFrame(results)

    positive_df = df[df["LOM (%)"] > 0].copy()
    positive_df["abs_LOM"] = positive_df["LOM (%)"].abs()
    positive_df = positive_df.sort_values(by="abs_LOM", ascending=True).drop(columns=["abs_LOM"]).reset_index(drop=True)

    negative_df = df[df["LOM (%)"] < 0].copy()
    negative_df["abs_LOM"] = negative_df["LOM (%)"].abs()
    negative_df = negative_df.sort_values(by="abs_LOM", ascending=True).drop(columns=["abs_LOM"]).reset_index(drop=True)

    positive_df.insert(0, "Rank", positive_df.index + 1)
    negative_df.insert(0, "Rank", negative_df.index + 1)

    return {
        "title": title,
        "positive_momentum": positive_df[["Rank", "Stock", "LOM (%)", "LOM Time", "LTP", "Change (%)"]].head(10).to_dict(orient="records"),
        "negative_momentum": negative_df[["Rank", "Stock", "LOM (%)", "LOM Time", "LTP", "Change (%)"]].head(10).to_dict(orient="records"),
        "summary": {
            "total_positive": len(positive_df),
            "total_negative": len(negative_df),
            "total_analyzed": len(df)
        }
    }

def print_ranked_by_lom_near_zero(df: pd.DataFrame, title: str):
    if df.empty:
        print(f"\n{title}: No data available")
        return

    df["abs_LOM"] = df["LOM (%)"].abs()
    df_sorted = df.sort_values("abs_LOM", ascending=True).drop(columns=["abs_LOM"]).reset_index(drop=True)
    df_sorted.insert(0, "Rank", df_sorted.index + 1)

    print(f"\n🔍 {title} (LOM (%) closest to 0):")
    print(df_sorted[["Rank", "Stock", "LOM (%)", "LOM Time", "LTP", "Change (%)"]].head(20).to_string(index=False))

if __name__ == "__main__":
    start = time.time()

    print("[INFO] Starting stock momentum analysis...")

    if not all(isinstance(s, str) for s in fo_stocks):
        fo_stocks = [s["symbol"] for s in fo_stocks if isinstance(s, dict) and "symbol" in s]

    results_5min = analyze_stocks(fo_stocks, interval="5minute", days=3, max_workers=5)
    results_1day = analyze_stocks(fo_stocks, interval="day", days=30, max_workers=5)

    df_5min = pd.DataFrame(results_5min)
    df_1day = pd.DataFrame(results_1day)

    print("\n📊 Top 5-Minute Momentum Gainers:")
    print(df_5min.sort_values("LOM (%)", ascending=False).head(10).to_string(index=False))

    print("\n📉 Top 5-Minute Momentum Losers:")
    print(df_5min.sort_values("LOM (%)", ascending=True).head(10).to_string(index=False))

    print("\n📊 Top 1-Day Momentum Gainers:")
    print(df_1day.sort_values("LOM (%)", ascending=False).head(10).to_string(index=False))

    print("\n📉 Top 1-Day Momentum Losers:")
    print(df_1day.sort_values("LOM (%)", ascending=True).head(10).to_string(index=False))

    print_ranked_by_lom_near_zero(df_5min, "5-Minute Interval")
    print_ranked_by_lom_near_zero(df_1day, "1-Day Interval")

    end = time.time()
    print(f"\n✅ Analysis completed in {round(end - start, 2)} seconds.")