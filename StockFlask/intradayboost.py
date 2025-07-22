import pandas as pd
import datetime
import json
import time
import logging
from kiteconnect import KiteConnect
from typing import List, Dict, Optional
from dataclasses import dataclass

# 🔹 Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 🔹 Load credentials and initialize Kite client
from auth import get_kite_client
kite = get_kite_client()

@dataclass
class StockAnalysis:
    stock: str
    ltp: Optional[float]
    percent_change: float
    recent_avg_daily_growth: float
    z_score: float
    is_abnormal: bool

# 🔹 Get all F&O equity stocks
"""def get_fo_stocks() -> Dict[str, int]:
    try:
        logger.info("Fetching instruments from Kite API...")
        all_instruments = kite.instruments()

        if not all_instruments:
            logger.error("No instruments data received from Kite API")
            return {}

        logger.info(f"Received {len(all_instruments)} instruments from Kite API")

        fo_symbols = set()
        for inst in all_instruments:
            if inst.get('exchange') == 'NFO' and inst.get('instrument_type') == 'FUT':
                symbol = inst.get('tradingsymbol', '')
                if symbol.endswith('FUT'):
                    symbol = symbol[:-3]
                    for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
                        if month in symbol:
                            symbol = symbol.split(month)[0]
                            if len(symbol) > 2 and symbol[-2:].isdigit():
                                symbol = symbol[:-2]
                            break
                if symbol:
                    fo_symbols.add(symbol)

        logger.info(f"Found {len(fo_symbols)} F&O symbols")

        fo_equity = {}
        for inst in all_instruments:
            if (inst.get('exchange') == 'NSE' and 
                inst.get('instrument_type') == 'EQ' and 
                inst.get('tradingsymbol') in fo_symbols):
                fo_equity[inst['tradingsymbol']] = inst['instrument_token']

        logger.info(f"Found {len(fo_equity)} F&O equity stocks")
        return fo_equity

    except Exception as e:
        logger.error(f"❌ Error fetching F&O stocks: {str(e)}")
        raise"""
from typing import Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)

def get_fo_stocks(filtered_symbols: Optional[Set[str]] = None) -> Dict[str, int]:
    try:
        logger.info("Fetching instruments from Kite API...")
        all_instruments = kite.instruments()

        if not all_instruments:
            logger.error("No instruments data received from Kite API")
            return {}

        logger.info(f"Received {len(all_instruments)} instruments from Kite API")

        fo_symbols = set()
        for inst in all_instruments:
            if inst.get('exchange') == 'NFO' and inst.get('instrument_type') == 'FUT':
                symbol = inst.get('tradingsymbol', '')
                if symbol.endswith('FUT'):
                    symbol = symbol[:-3]
                    for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                                  'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
                        if month in symbol:
                            symbol = symbol.split(month)[0]
                            if len(symbol) > 2 and symbol[-2:].isdigit():
                                symbol = symbol[:-2]
                            break
                if symbol:
                    fo_symbols.add(symbol)

        logger.info(f"Found {len(fo_symbols)} F&O symbols")

        fo_equity = {}
        for inst in all_instruments:
            symbol = inst.get('tradingsymbol')
            if (inst.get('exchange') == 'NSE' and 
                inst.get('instrument_type') == 'EQ' and 
                symbol in fo_symbols):

                # ✅ Apply filtering here if symbols are provided
                if filtered_symbols is None or symbol in filtered_symbols:
                    fo_equity[symbol] = inst['instrument_token']

        logger.info(f"Found {len(fo_equity)} F&O equity stocks (filtered: {filtered_symbols is not None})")
        return fo_equity

    except Exception as e:
        logger.error(f"❌ Error fetching F&O stocks: {str(e)}")
        raise

# 🔹 Fetch historical data
def get_historical_data(
    kite: KiteConnect,
    trading_symbol: str,
    days_back: int,
    interval: str = "day",
    exchange: str = "NSE"
) -> Optional[pd.DataFrame]:
    if days_back <= 0:
        raise ValueError("days_back must be positive")

    max_retries = 3
    base_delay = 1
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=days_back)

    for attempt in range(max_retries):
        try:
            instrument_token = kite.ltp(f"{exchange}:{trading_symbol}")[f"{exchange}:{trading_symbol}"]["instrument_token"]
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=start_date,
                to_date=end_date,
                interval=interval
            )
            if not data:
                logger.warning(f"No historical data available for {trading_symbol}")
                return None

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "close"]]

        except Exception as e:
            if "Too many requests" in str(e) and attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, waiting {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            logger.error(f"❌ Error fetching historical data for {trading_symbol}: {e}")
            return None

    return None

# 🔹 Growth calculation
def calculate_growth(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) < 2:
        raise ValueError("DataFrame must contain at least 2 rows for growth calculation")
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    df = df.copy()
    df["daily_growth"] = df["close"].pct_change() * 100
    return df

# 🔹 Analyze individual stock
def analyze_stock_growth(
    trading_symbol: str,
    recent_days: int = 7,
    long_term_days: int = 365,
    threshold: float = 2.0,
    exchange: str = "NSE"
) -> Optional[Dict]:
    if recent_days <= 0 or long_term_days <= 0:
        raise ValueError("Days parameters must be positive")
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    if recent_days > long_term_days:
        raise ValueError("recent_days cannot be greater than long_term_days")

    try:
        long_term_df = get_historical_data(kite, trading_symbol, long_term_days, exchange=exchange)
        if long_term_df is None or len(long_term_df) < 2:
            logger.warning(f"Insufficient historical data for {trading_symbol}")
            return None

        recent_df = long_term_df.tail(recent_days)
        if len(recent_df) < recent_days:
            logger.warning(f"Insufficient recent data for {trading_symbol}")
            return None

        long_term_df = calculate_growth(long_term_df)
        recent_df = calculate_growth(recent_df)

        avg_daily_growth = long_term_df["daily_growth"].mean()
        std_daily_growth = long_term_df["daily_growth"].std()
        recent_total_growth = ((recent_df["close"].iloc[-1] / recent_df["close"].iloc[0]) - 1) * 100
        recent_avg_daily_growth = recent_df["daily_growth"].mean()

        latest_close = recent_df["close"].iloc[-1]
        previous_close = recent_df["close"].iloc[-2] if len(recent_df) >= 2 else None
        change_from_previous_close_percentage = ((latest_close - previous_close) / previous_close) * 100 if previous_close else None

        try:
            ltp_data = kite.ltp(f"{exchange}:{trading_symbol}")
            ltp = ltp_data[f"{exchange}:{trading_symbol}"]["last_price"]
        except Exception as e:
            logger.warning(f"Could not fetch LTP for {trading_symbol}: {e}")
            ltp = None

        z_score = (recent_avg_daily_growth - avg_daily_growth) / std_daily_growth if std_daily_growth > 0 else 0

        # Always set is_abnormal to True
        is_abnormal = True

        return {
            "stock": trading_symbol,
            "ltp": ltp,
            "percent_change": round(recent_total_growth, 2),
            "recent_avg_daily_growth (%)": round(recent_avg_daily_growth, 2),
            "z_score": round(z_score, 2),
            "is_abnormal": is_abnormal
        }

    except Exception as e:
        logger.error(f"Error analyzing {trading_symbol}: {e}")
        return None

# 🔹 Rank all F&O stocks
def rank_fo_stocks_by_growth() -> pd.DataFrame:
    try:
        fo_stocks = get_fo_stocks()
        fo_stock = dict(list(fo_stocks.items())[:3])
        if not fo_stocks:
            logger.error("No F&O stocks found to analyze")
            return pd.DataFrame()

        results = []
        total_stocks = len(fo_stocks)

        for idx, symbol in enumerate(fo_stock.keys(), 1):
            logger.info(f"🔍 Analyzing {symbol} ({idx}/{total_stocks})...")
            try:
                analysis = analyze_stock_growth(symbol)
                if analysis:
                    results.append(analysis)
                else:
                    logger.warning(f"⚠️ Insufficient data for {symbol}")
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}")

            sleep_time = max(1.2, min(2.0, 1.2 * (idx % 5)))
            time.sleep(sleep_time)

        if not results:
            logger.error("❌ No valid results found")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        required_columns = ["stock", "ltp", "percent_change", "recent_avg_daily_growth (%)", "z_score", "is_abnormal"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"⚠️ Missing columns in final DataFrame: {missing_columns}")
            logger.info("🧪 DataFrame preview:")
            logger.info(df.head())
        else:
            logger.info("\n✅ Top F&O Stocks by Growth Factor:")
            top_stocks = df.sort_values("percent_change", ascending=False)[required_columns].copy()
            top_stocks.reset_index(drop=True, inplace=True)
            top_stocks.insert(0, 'Rank', range(1, len(top_stocks) + 1))

            # Output as Python dictionary
            result_dict = {
                "top_fo_stocks_by_growth": top_stocks.head(60).to_dict(orient="records")
            }
            print(result_dict)
            # You can now use result_dict for further processing

        return df

    except Exception as e:
        logger.error(f"Fatal error in rank_fo_stocks_by_growth: {e}")
        return pd.DataFrame()

# 🔹 Run if this file is executed directly
if __name__ == "__main__":
    logger.info("🚀 Starting F&O Stock Growth Analysis...")
    try:
        result_df = rank_fo_stocks_by_growth()
        if not result_df.empty:
            logger.info(f"✅ Analysis complete. Found {len(result_df)} valid stocks.")
        else:
            logger.error("❌ Analysis failed to produce valid results.")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
