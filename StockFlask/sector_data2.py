# sector_data.py

from kiteconnect import KiteConnect
import datetime
from time import sleep
import json
from random import uniform
import pandas as pd  # Required for DataFrame operations

from auth import get_kite_client
from analysestockgrowth import analyze_stock_growth

kite = get_kite_client()
import logging

# Set up basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


sector_stocks = {
    "Nifty 50": [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "LT", "KOTAKBANK",
        "AXISBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ASIANPAINT", "MARUTI",
        "SUNPHARMA", "WIPRO", "BAJFINANCE", "BAJAJFINSV", "NTPC", "ULTRACEMCO",
        "NESTLEIND", "HCLTECH", "TECHM", "TITAN", "POWERGRID", "ONGC", "TATASTEEL",
        "GRASIM", "HDFCLIFE", "CIPLA", "JSWSTEEL", "DRREDDY", "COALINDIA", "BPCL",
        "ADANIPORTS", "DIVISLAB", "BRITANNIA", "EICHERMOT", "UPL", "SHREECEM",
        "HEROMOTOCO", "M&M", "BAJAJ-AUTO", "INDUSINDBK", "SBILIFE", "IOC", "TATACONSUM",
        "HINDALCO"
    ],
    "Sensex": [
        "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "LT", "KOTAKBANK",
        "AXISBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ASIANPAINT", "MARUTI",
        "SUNPHARMA", "WIPRO", "BAJFINANCE", "BAJAJFINSV", "NTPC", "ULTRACEMCO",
        "NESTLEIND", "HCLTECH", "TECHM", "TITAN", "POWERGRID", "HDFCLIFE",
        "INDUSINDBK", "DRREDDY", "M&M", "TATAMOTORS"
    ],
    "Nifty IT": [
        "INFY", "TCS", "WIPRO", "HCLTECH", "TECHM", "LTIM", "PERSISTENT", "MPHASIS",
        "COFORGE", "ZENSARTECH", "TANLA", "BIRLASOFT", "SONATSOFTW", "NIITTECH", "KPITTECH"
    ],
    "Pharma & Healthcare": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "DIVISLAB", "AUROPHARMA", "ZYDUSLIFE", "BIOCON",
        "APOLLOHOSP", "FORTIS", "METROPOLIS", "NARAYANA", "KIMS", "MAXHEALTH",
        "LAURUSLABS", "GLAND", "ALKEM", "NEULANDLAB", "JBCHEPHARM", "ASTRAZEN", "ERIS", "FDC", "ALEMBICLTD", "NATCOPHARM"
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "COLPAL", "EMAMILTD",
        "GODREJCP", "RADICO", "TATACONSUM", "BAJAJCON"
    ],
    "Auto": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "ASHOKLEY", "TVSMOTOR",
        "BALKRISIND", "SMLISUZU", "ESCORTS", "AMARAJABAT", "EXIDEIND"
    ],
    "Oil & Gas": [
        "RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "PETRONET", "IGL", "MGL",
        "HPCL", "OIL", "GSPL", "GUJGASLTD"
    ],
    "Infra": [
        "LT", "ADANIPORTS", "GMRINFRA", "NBCC", "IRB", "DLF", "GODREJPROP", "OBEROIRLTY",
        "PNCINFRA", "HGINFRA", "NCC", "KNRCON", "ASHOKA", "CAPACITE"
    ],
    "Energy": [
        "NTPC", "POWERGRID", "TATAPOWER", "ADANIGREEN", "JSWENERGY", "TORNTPOWER", "RELIANCE", "COALINDIA",
        "NHPC", "SJVN", "ADANITRANS"
    ],
    "Media": [
        "ZEEL", "SUNTV", "PVRINOX", "DISHTV", "TV18BRDCST", "NETWORK18", "JAGRAN", "DBCORP",
        "HATHWAY", "DEN", "SAREGAMA", "TIPSINDLTD"
    ],
    "Metal": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL", "NMDC", "JINDALSTEL", "NALCO",
        "MOIL", "APLAPOLLO", "RATNAMANI"
    ],
    "Nifty Private Bank": [
        "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "IDFCFIRSTB",
        "RBLBANK", "YESBANK", "BANDHANBNK", "CSBBANK", "DCBBANK", "SOUTHBANK"
    ],
    "Nifty PSU Bank": [
        "SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "BANKINDIA", "INDIANBANK", "UCOBANK",
        "MAHABANK", "IOB", "CENTRALBK"
    ],
    "Realty": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "PRESTIGE", "SOBHA", "SUNTECK", "BRIGADE",
        "LODHA", "MAHLIFE", "ARVINDSMART", "NCC", "NBCC"
    ],
    "Nifty Commodities": [
        "HINDALCO", "VEDL", "TATASTEEL", "RELIANCE", "NTPC", "ULTRACEMCO", "GRASIM", "ONGC",
        "COALINDIA", "JSWSTEEL", "TATAPOWER", "ADANIPOWER",]
}        
def get_all_sector_names():
    return list(sector_stocks.keys())

def get_stock_growth_data(kite, stock_symbol, retries=3, base_delay=1):
    """
    Fetch and analyze stock growth data with improved error handling and rate limiting
    """
    for attempt in range(retries):
        try:
            # Get instrument token
            ltp_response = kite.ltp(f"NSE:{stock_symbol}")
            if not ltp_response:
                logger.error(f"No LTP data available for {stock_symbol}")
                return None
                
            instrument_token = ltp_response[f"NSE:{stock_symbol}"]["instrument_token"]
            
            # Calculate date range
            to_date = datetime.datetime.today()
            from_date = to_date - datetime.timedelta(days=15)
            
            # Fetch historical data
            data = kite.historical_data(instrument_token, from_date, to_date, "day")
            if not data:
                logger.warning(f"No historical data available for {stock_symbol}")
                return None

            # Process data
            df = pd.DataFrame(data)
            if len(df) < 5:  # Need at least 5 days of data
                logger.warning(f"Insufficient historical data for {stock_symbol}")
                return None

            # Calculate metrics
            df = df[["date", "close", "volume"]].copy()
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)

            df["daily_growth"] = df["close"].pct_change() * 100
            
            # Use more robust statistics
            recent_growth = df["daily_growth"].tail(5)
            avg_growth = round(recent_growth.mean(), 2)
            growth_volatility = round(recent_growth.std(), 2)
            volume_traded = round(df["volume"].mean(), 2)
            
            return {
                "symbol": stock_symbol,
                "avg_growth": avg_growth,
                "growth_volatility": growth_volatility,
                "volume_traded": volume_traded,
                "data_points": len(df)
            }

        except Exception as e:
            if "Too many requests" in str(e) and attempt < retries - 1:
                sleep_time = base_delay * (2 ** attempt) + uniform(0.1, 0.5)
                logger.warning(f"Rate limit hit for {stock_symbol}. Waiting {sleep_time:.2f}s...")
                sleep(sleep_time)
            else:
                logger.error(f"Error processing {stock_symbol}: {str(e)}")
                return None

    return None

def get_sector_abnormal_growth(kite, sector, min_stocks= 0 , z_score_threshold=2.0):
    try:
        if sector not in sector_stocks:
            raise ValueError(f"Invalid sector: {sector}")
        stocks = sector_stocks[sector]
        if len(stocks) < min_stocks:
            return {
                "sector_name": sector,
                "abnormal_growth_stocks": [],
                "ranked_stocks": [],
                "error": f"Insufficient stocks in sector (minimum {min_stocks} required)"
            }

        stock_data_list = []
        for stock in stocks:
            data = get_stock_growth_data(kite, stock)
            if data:
                stock_data_list.append(data)
            sleep(uniform(0.7, 1.1))

        if len(stock_data_list) < min_stocks:
            return {
                "sector_name": sector,
                "abnormal_growth_stocks": [],
                "ranked_stocks": [],
                "error": f"Insufficient valid data (got {len(stock_data_list)}, need {min_stocks})"
            }

        df = pd.DataFrame(stock_data_list)

        if df["avg_growth"].std() == 0:
            return {
                "sector_name": sector,
                "abnormal_growth_stocks": [],
                "ranked_stocks": [],
                "error": "No variation in growth rates"
            }

        df["z_score"] = ((df["avg_growth"] - df["avg_growth"].median()) / df["avg_growth"].std()).round(2)
        df = df.sort_values("avg_growth", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        abnormal_stocks = df[abs(df["z_score"]) > z_score_threshold]

        total_sector_percent = round(df["avg_growth"].abs().sum(), 2)
        return {
            "sector_name": sector,
            "sector_stats": {
                "median_growth": round(df["avg_growth"].median(), 2),
                "growth_std": round(df["avg_growth"].std(), 2),
                "total_stocks_analyzed": len(df),
                "total_sector_percent": total_sector_percent
            },
            "abnormal_growth_stocks": abnormal_stocks.to_dict(orient="records"),
            "ranked_stocks": df[["symbol", "avg_growth", "rank"]].to_dict(orient="records")
        }

    except Exception as e:
        logger.error(f"Error analyzing sector {sector}: {str(e)}")
        return {
            "sector_name": sector,
            "abnormal_growth_stocks": [],
            "ranked_stocks": [],
            "error": str(e)
        }

# Run test for all sectors
if __name__ == "__main__":
    try:
        kite = get_kite_client()
        sectors = get_all_sector_names()
        all_results = {}
        for sector in sectors:
            result = get_sector_abnormal_growth(kite, sector)
            all_results[sector] = result
        print(all_results)
        # You can now use 'all_results' as a Python dictionary for further processing

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")