def get_fo_stocks():
    """
    Fetches and returns a list of F&O stocks from NSE.
    Returns:
        list: List of F&O stock symbols
    """
    from auth import get_kite_client
    import pandas as pd

    kite = get_kite_client()
    
    try:
        # Fetch all instruments
        print("Fetching instruments datas...")
        instruments = kite.instruments()
        df = pd.DataFrame(instruments)

        # Filter for NSE only
        df_nse = df[df["exchange"] == "NSE"]

        # F&O instruments (NFO = NSE Futures and Options)
        df_fo = df[df["exchange"] == "NFO"]
        
        if "tradingsymbol" not in df_fo.columns:
            raise KeyError("'tradingsymbol' column not found in the DataFrame")
            
        # Extract base symbols from F&O instruments
        fo_symbols = df_fo["tradingsymbol"].str.extract(r"^([A-Z]+)")[0].unique().tolist()
        
        # Get only equity symbols that are in both NSE and F&O
        fo_stocks = sorted(set(fo_symbols) & set(df_nse["tradingsymbol"]))
        
        print(f"Successfully fetched {len(fo_stocks)} F&O stocks")
        return fo_stocks
        
    except Exception as e:
        print(f"Error fetching F&O stocks: {e}")
        return []

if __name__ == "__main__":
    # When run directly, print the list of F&O stocks
    stocks = get_fo_stocks()
    print(f"\nFound {len(stocks)} F&O stocks:")
    print(", ".join(stocks[:10]) + ("..." if len(stocks) > 10 else ""))
