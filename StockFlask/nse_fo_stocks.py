import requests
import pandas as pd
from io import StringIO

def fetch_nse_fo_stocks():
    url = "https://www1.nseindia.com/content/fo/fo_mktlots.csv"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www1.nseindia.com/products/content/derivatives/equities/market_lot.htm",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        # Try to auto-detect delimiter and skip junk lines
        csv_text = resp.text
        # Find the first line with 'SYMBOL' to skip any junk header
        lines = csv_text.splitlines()
        header_idx = next(i for i, line in enumerate(lines) if 'SYMBOL' in line)
        clean_csv = '\n'.join(lines[header_idx:])
        df = pd.read_csv(StringIO(clean_csv), delimiter=';')
        fo_stocks = df['SYMBOL'].dropna().unique().tolist()
        return fo_stocks
    except Exception as e:
        print(f"[ERROR] Could not fetch NSE F&O stocks: {e}")
        return []

if __name__ == "__main__":
    stocks = fetch_nse_fo_stocks()
    print("F&O Stocks from NSE:")
    print(stocks)
