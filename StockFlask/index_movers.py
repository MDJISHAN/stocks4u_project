from kiteconnect import KiteConnect
import json
from typing import Tuple

def load_credentials() -> Tuple[str, str]:
    try:
        with open("login_credentials.json", "r") as f:
            creds = json.load(f)
            return creds["api_key"], creds["access_token"]
    except Exception as e:
        print(json.dumps({"error": f"Error loading credentials: {e}"}))
        return None, None

API_KEY, ACCESS_TOKEN = load_credentials()
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# Define indices with their symbols
indices = {
    "NIFTY 50": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCAPNIFTY": "NSE:NIFTY MIDCAP 50",
    "SENSEX": "BSE:SENSEX"
}
company_to_symbol = {
    "Adani Enterprises": "ADANIENT", "Adani Ports": "ADANIPORTS", "Apollo Hospitals": "APOLLOHOSP", "Asian Paints": "ASIANPAINT", "Axis Bank": "AXISBANK",
    "Bajaj Auto": "BAJAJ-AUTO", "Bajaj Finance": "BAJFINANCE", "Bajaj Finserv": "BAJAJFINSV", "Bharti Airtel": "BHARTIARTL", "BPCL": "BPCL",
    "Britannia": "BRITANNIA", "Cipla": "CIPLA", "Coal India": "COALINDIA", "Divi's Labs": "DIVISLAB", "Dr. Reddy's": "DRREDDY",
    "Eicher Motors": "EICHERMOT", "Grasim": "GRASIM", "HCL Tech": "HCLTECH", "HDFC Bank": "HDFCBANK", "HDFC Life": "HDFCLIFE",
    "Hero MotoCorp": "HEROMOTOCO", "Hindalco": "HINDALCO", "Hindustan Unilever": "HINDUNILVR", "ICICI Bank": "ICICIBANK", "IndusInd Bank": "INDUSINDBK",
    "Infosys": "INFY", "ITC": "ITC", "JSW Steel": "JSWSTEEL", "Kotak Mahindra Bank": "KOTAKBANK", "Larsen & Toubro": "LT",
    "LTIMindtree": "LTIM", "M&M": "M&M", "Maruti Suzuki": "MARUTI", "Nestle India": "NESTLEIND", "NTPC": "NTPC",
    "ONGC": "ONGC", "Power Grid Corp": "POWERGRID", "Reliance Industries": "RELIANCE", "SBI Life": "SBILIFE", "State Bank of India": "SBIN",
    "Sun Pharma": "SUNPHARMA", "TCS": "TCS", "Tata Consumer": "TATACONSUM", "Tata Motors": "TATAMOTORS", "Tata Steel": "TATASTEEL",
    "Tech Mahindra": "TECHM", "Titan Company": "TITAN", "UltraTech Cement": "ULTRACEMCO", "UPL": "UPL", "Wipro": "WIPRO",
    
    "Federal Bank": "FEDERALBNK", "Bandhan Bank": "BANDHANBNK", "IDFC First Bank": "IDFCFIRSTB", "Punjab National Bank": "PNB", "Bank of Baroda": "BANKBARODA", "AU Small Finance Bank": "AUBANK",
    
    "ICICI Prudential Life": "ICICIPRULI", "ICICI Lombard": "ICICIGI", "HDFC AMC": "HDFCAMC", "Aditya Birla Capital": "ABCAPITAL", "Cholamandalam Invest": "CHOLAFIN",
    "LIC Housing Finance": "LICHSGFIN", "Muthoot Finance": "MUTHOOTFIN", "REC": "RECLTD", "PFC": "PFC", "Shriram Finance": "SHRIRAMFIN", "PNB Housing Finance": "PNBHOUSING",
    
    "Adani Green Energy": "ADANIGREEN", "Alkem Laboratories": "ALKEM", "Ashok Leyland": "ASHOKLEY", "Astral Ltd": "ASTRAL", "Bharat Electronics": "BEL",
    "Bharat Forge": "BHARATFORG", "BHEL": "BHEL", "BSE Ltd": "BSE", "Canara Bank": "CANBK", "Cholamandalam Investment": "CHOLAFIN",
    "Coforge Ltd": "COFORGE", "Colgate Palmolive": "COLPAL", "Container Corporation of India": "CONCOR", "Cummins India": "CUMMINSIND", "Dixon Technologies": "DIXON",
    "DLF Ltd": "DLF", "Godrej Consumer Products": "GODREJCP", "Godrej Properties": "GODREJPROP", "GMR Airports": "GMRINFRA", "Hindustan Aeronautics Ltd": "HAL",
    "Hindustan Petroleum Corporation": "HINDPETRO", "Indian Hotels Company": "INDHOTEL", "Indian Oil Corporation": "IOC", "Indus Towers": "INDUSTOWER",
    "IRCTC": "IRCTC", "JSW Energy": "JSWENERGY", "Jubilant FoodWorks": "JUBLFOOD", "Lupin Ltd": "LUPIN", "Mphasis Ltd": "MPHASIS",
    "MRF Ltd": "MRF", "NHPC Ltd": "NHPC", "NMDC Ltd": "NMDC", "Oil India Ltd": "OIL", "Oracle Financial Services Software": "OFSS",
    "Page Industries": "PAGEIND", "PB Fintech Ltd": "POLICYBZR", "Petronet LNG Ltd": "PETRONET", "Phoenix Mills": "PHOENIXLTD", "Pidilite Industries": "PIDILITIND",
    "Polycab India": "POLYCAB", "Power Finance Corporation": "PFC", "Prestige Estates": "PRESTIGE", "REC Ltd": "RECLTD", "TVS Motor Company": "TVSMOTOR",
    
    "Reliance": "RELIANCE", "Kotak Bank": "KOTAKBANK", "HUL": "HINDUNILVR", "Nestle": "NESTLEIND", "Tata Steel": "TATASTEEL",
    "Tata Motors": "TATAMOTORS", "Bajaj Finserv": "BAJAJFINSV", "Titan": "TITAN"

}
company_to_symbol.update({
    "Dr. Reddy's Laboratories": "DRREDDY",
    "Grasim Industries": "GRASIM",
    "HCL Technologies": "HCLTECH",
    "HDFC Life Insurance": "HDFCLIFE",
    "Hindalco Industries": "HINDALCO",
    "Mahindra & Mahindra": "M&M",
    "Maruti Suzuki India": "MARUTI",
    "Nestlé India": "NESTLEIND",
    "Oil & Natural Gas Corporation": "ONGC",
    "Power Grid Corporation of India": "POWERGRID",
    "SBI Life Insurance": "SBILIFE",
    "Sun Pharmaceutical Industries": "SUNPHARMA",
    "Tata Consultancy Services": "TCS",
    "Tata Consumer Products": "TATACONSUM",
    "Trent": "TRENT",
    "Jio Financial Services": "JIOFIN",
    "Reliance Industries Ltd.": "RELIANCE",
    "Power Grid Corporation": "POWERGRID"
})


# Define constituents per index
index_constituents = {
    "NIFTY 50": [
        "Adani Enterprises", "Adani Ports", "Apollo Hospitals", "Asian Paints", "Axis Bank",
    "Bajaj Auto", "Bajaj Finance", "Bajaj Finserv", "Bharti Airtel", "Cipla",
    "Coal India", "Dr. Reddy's Laboratories", "Eicher Motors", "Grasim Industries", "HCL Technologies",
    "HDFC Bank", "HDFC Life Insurance", "Hero MotoCorp", "Hindalco Industries", "Hindustan Unilever",
    "ICICI Bank", "IndusInd Bank", "Infosys", "ITC", "JSW Steel",
    "Kotak Mahindra Bank", "Larsen & Toubro", "LTIMindtree", "Mahindra & Mahindra", "Maruti Suzuki India",
    "Nestlé India", "NTPC", "Oil & Natural Gas Corporation", "Power Grid Corporation of India", "Reliance Industries",
    "SBI Life Insurance", "State Bank of India", "Sun Pharmaceutical Industries", "Tata Consultancy Services", "Tata Consumer Products",
    "Tata Motors", "Tata Steel", "Tech Mahindra", "Titan Company", "UltraTech Cement",
    "Trent", "Wipro", "Shriram Finance", "Bharat Electronics", "Jio Financial Services"
    ],
    "BANKNIFTY": [
        "HDFC Bank", "ICICI Bank", "State Bank of India", "Kotak Mahindra Bank", "Axis Bank",
    "IndusInd Bank", "Federal Bank", "IDFC First Bank", "Punjab National Bank",
    "Bank of Baroda", "AU Small Finance Bank", "Canara Bank"
    ],
    "FINNIFTY": [
    "HDFC Bank", "ICICI Bank", "Kotak Mahindra Bank", "Axis Bank", "State Bank of India",
    "Bajaj Finance", "Bajaj Finserv", "HDFC Life", "SBI Life", "ICICI Prudential Life",
    "ICICI Lombard", "HDFC AMC", "Cholamandalam Invest", "LIC Housing Finance", "Muthoot Finance",
    "REC", "PFC", "Shriram Finance", "Jio Financial Services"
   ],

    "MIDCAPNIFTY50": [
    "Alkem Laboratories", "Ashok Leyland", "AU Small Finance Bank", "Bharat Electronics",
    "Bharat Forge", "BHEL", "Canara Bank", "Cholamandalam Investment", "Coforge Ltd",
    "Colgate Palmolive", "Container Corporation of India", "Cummins India", "Federal Bank",
    "Godrej Properties", "Hindustan Petroleum Corporation", "IDFC First Bank", "IRCTC",
    "Mphasis Ltd", "MRF Ltd", "Muthoot Finance", "NMDC Ltd", "Oil India Ltd",
    "Oracle Financial Services Software", "Page Industries", "PB Fintech Ltd", "Petronet LNG Ltd",
    "Phoenix Mills", "Polycab India", "Prestige Estates", "REC Ltd", "TVS Motor Company"
    ],
    
    "SENSEX": [
       "Reliance Industries Ltd.", "HDFC Bank", "ICICI Bank", "Infosys", "TCS",
    "Larsen & Toubro", "Axis Bank", "State Bank of India", "Bharti Airtel", "Kotak Mahindra Bank",
    "ITC", "Hindustan Unilever", "Bajaj Finance", "HCL Technologies", "Asian Paints",
    "Mahindra & Mahindra", "Maruti Suzuki", "Sun Pharmaceutical Industries", "UltraTech Cement", "NTPC",
    "Power Grid Corporation", "Nestlé India", "Tata Steel", "Tech Mahindra", "IndusInd Bank",
    "Dr. Reddy's Laboratories", "Tata Motors", "Bajaj Finserv", "Wipro", "Titan Company"
    ],
}

def analyze_indices(kite, index_dict, constituents_dict):
    try:
        quote_data = kite.quote(list(index_dict.values()))
        print("\n📈 Index % Change Summary")
        print("----------------------------------------------------------")
        print(f"{'Index':<20} {'% Change':>12}")
        print("----------------------------------------------------------")

        for name, symbol in index_dict.items():
            data = quote_data.get(symbol, {})
            ltp = data.get("last_price")
            prev_close = data.get("ohlc", {}).get("close")

            if ltp and prev_close:
                index_change_percent = ((ltp - prev_close) / prev_close) * 100
                print(f"\n{name} ➤ {index_change_percent:.2f}%")

                print(f"\n{name} Constituents:")
                symbols = constituents_dict.get(name) or constituents_dict.get(name.replace("MIDCAPNIFTY", "MIDCAPNIFTY50"))
                if not symbols:
                    print("⚠️ No constituents listed.")
                    continue

                stock_symbols = []
                for stock in symbols:
                    trading_symbol = company_to_symbol.get(stock)
                    if trading_symbol:
                        stock_symbols.append(f"NSE:{trading_symbol}")
                    else:
                        print(f" - {stock:<35} ❌ No trading symbol mapped")

                stock_quote_data = kite.quote(stock_symbols)

                total_positive = 0
                total_negative = 0

                for stock in symbols:
                    trading_symbol = company_to_symbol.get(stock)
                    if not trading_symbol:
                        continue  # Skip if no trading symbol is found
                    key = f"NSE:{trading_symbol}"
                    sdata = stock_quote_data.get(key, {})
                    last_price = sdata.get("last_price")
                    prev = sdata.get("ohlc", {}).get("close")

                    if last_price and prev:
                        change = ((last_price - prev) / prev) * 100
                        print(f" - {stock:<35} {change:>6.2f}%")
                        if change >= 0:
                            total_positive += change
                        else:
                            total_negative += change
                    else:
                        print(f" - {stock:<35} ❌ No data")

                net_change = total_positive + total_negative
                print(f"\n{name} Summary:")
                print(f"Total Positive Change: {total_positive:.2f}%")
                print(f"Total Negative Change: {total_negative:.2f}%")
                print(f"Net Change (Sum):       {net_change:.2f}%")

            else:
                print(f"{name:<20} ❌ Index data not available")

    except Exception as e:
        print(f"❌ Error fetching data: {e}")

# Run the analysis
analysis_result = analyze_indices(kite, indices, index_constituents)
print(analysis_result)
# You can now use analysis_result as a Python dictionary for further processing