#approved

import pandas as pd
import time
import logging
from kiteconnect import KiteConnect
from typing import Dict
from auth import get_kite_client
from analysestockgrowth import change_from_previous_close_percentage

# 🔹 Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 🔹 Initialize Kite client
kite = get_kite_client()

# 🔹 Get all F&O equity stocks
def get_fo_stocks() -> Dict[str, int]:
    try:
        logger.info("Fetching instruments from Kite API...")
        all_instruments = kite.instruments()

        if not all_instruments:
            logger.error("No instruments data received from Kite API")
            return {}

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
        raise

# 🔹 Rank all F&O stocks based on proximity to day high and low
def rank_fo_stocks() -> pd.DataFrame:
    try:
        fo_stocks = get_fo_stocks()
        if not fo_stocks:
            logger.error("No F&O stocks found to analyze")
            return pd.DataFrame()

        results = []
        total_stocks = len(fo_stocks)

        for idx, symbol in enumerate(fo_stocks.keys(), 1):
            logger.info(f"🔍 Processing {symbol} ({idx}/{total_stocks})...")
            try:
                quote_data = kite.quote(f"NSE:{symbol}")
                quote = quote_data[f"NSE:{symbol}"]
                ltp = quote["last_price"]
                day_high = quote["ohlc"]["high"]
                day_low = quote["ohlc"]["low"]

                percentage_change = change_from_previous_close_percentage(symbol, kite)

                range_diff = day_high - day_low
                if range_diff > 0:
                    proximity_high = ((ltp - day_low) / range_diff) * 100
                    proximity_low = ((day_high - ltp) / range_diff) * 100
                else:
                    proximity_high = 0
                    proximity_low = 0

                results.append({
                    "stock": symbol,
                    "ltp": ltp,
                    "percentage_change": percentage_change,
                    "proximity_to_high": proximity_high,
                    "proximity_to_low": proximity_low
                })
            except Exception as e:
                logger.error(f"❌ Error fetching data for {symbol}: {e}")
            time.sleep(0.5)  # Avoid rate limiting

        if not results:
            logger.error("❌ No valid results found")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        df["rank_near_high"] = df["proximity_to_high"].rank(method="first", ascending=False).astype(int)
        df["rank_near_low"] = df["proximity_to_low"].rank(method="first", ascending=False).astype(int)

        return df

    except Exception as e:
        logger.error(f"Fatal error in rank_fo_stocks: {e}")
        return pd.DataFrame()

# 🔹 Main Runner
if __name__ == "__main__":
    import json
    logger.info("🚀 Starting F&O Stock Proximity Analysis...")
    try:
        result_df = rank_fo_stocks()
        output = {}
        if not result_df.empty:
            logger.info(f"✅ Analysis complete. Found {len(result_df)} valid stocks.")
            top_n = 20
            low_n = 20
            top_performers = result_df.sort_values("rank_near_high")[[
                "rank_near_high", "stock", "ltp", "percentage_change", "proximity_to_high"
            ]].head(top_n).to_dict(orient="records")
            low_performers = result_df.sort_values("rank_near_low")[[
                "rank_near_low", "stock", "ltp", "percentage_change", "proximity_to_low"
            ]].head(low_n).to_dict(orient="records")
            output = {
                "total_valid_stocks": int(len(result_df)),
                "top_performers": top_performers,
                "low_performers": low_performers
            }
        else:
            logger.error("❌ No results to display.")
            output = {
                "total_valid_stocks": 0,
                "top_performers": [],
                "low_performers": [],
                "error": "No results to display."
            }
        print(output)
# You can now use 'output' as a Python dictionary for further processing
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        output = {
            "total_valid_stocks": 0,
            "top_performers": [],
            "low_performers": [],
            "error": str(e)
        }
        print(output)
# You can now use 'output' as a Python dictionary for further processing
