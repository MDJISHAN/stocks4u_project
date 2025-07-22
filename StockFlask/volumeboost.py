import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
from tqdm import tqdm
from auth import get_kite_client

# Thread-safe print
def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with threading.Lock():
        print(*args, **kwargs)

# Rate limiter for API calls
class RateLimiter:
    def __init__(self, max_calls, period=60):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def wait_for_rate_limit(self):
        with self.lock:
            now = time.time()
            # Remove calls older than the period
            self.calls = [t for t in self.calls if now - t < self.period]
            
            if len(self.calls) >= self.max_calls:
                oldest = self.calls[0]
                sleep_time = self.period - (now - oldest)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.calls.append(time.time())

# Global rate limiter - Be very conservative with Kite's rate limits
rate_limiter = RateLimiter(max_calls=3, period=5)  # 3 calls per 5 seconds
# Cache for storing instrument data
_instrument_cache = None

def get_all_instruments():
    """Get all NSE instruments with caching and robust error handling"""
    global _instrument_cache
    if _instrument_cache is None:
        print("Fetching all NSE instruments...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                _instrument_cache = kite.instruments(exchange="NSE")
                break
            except Exception as e:
                if ("502 Bad Gateway" in str(e)) or ("Unknown Content-Type" in str(e)):
                    wait_time = (attempt + 1) * 3
                    print(f"Error fetching instruments (likely Kite server issue, 502 or unknown content-type): {e}. Waiting {wait_time}s and retrying ({attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"Error fetching instruments: {e}")
                    break
        if _instrument_cache is None:
            print("Failed to fetch instrument list after retries. Exiting or using fallback (empty list).")
            _instrument_cache = []
    return _instrument_cache

def get_non_fo_stocks():
    """Fetch non-F&O stocks from Kite API with caching and filter for regular NSE equities only."""
    all_instruments = get_all_instruments()
    
    # Get F&O stocks for filtering
    fo_stocks = set()
    try:
        from select_filter import get_fo_stocks
        fo_stocks = set(get_fo_stocks())
    except Exception as e:
        print(f"Could not import F&O filter: {e}")
        # fallback: treat as empty
    
    # Only keep regular NSE equities (not bonds, ETFs, SME, etc.)
    equity_instruments = [inst for inst in all_instruments
                         if inst.get('exchange') == 'NSE'
                         and inst.get('segment') == 'NSE'
                         and inst.get('instrument_type') == 'EQ']
    
    # Remove symbols with special suffixes (SME, BE, BZ, N0, etc.)
    def is_regular_equity(symbol):
        for suffix in ['-SM', '-BE', '-BZ', '-N0', '-SG', '-TB', '-X1', '-X2', '-X3', '-N1', '-N2', '-N3', '-N4', '-N5', '-N6', '-N7', '-N8', '-N9', '-NA', '-NB', '-NC', '-ND', '-NE', '-NF', '-NG', '-NH', '-NI', '-NJ', '-NK', '-NL', '-NM', '-NN', '-NO', '-NP', '-NQ', '-NR', '-NS', '-NT', '-NU', '-NV', '-NW', '-NX', '-NY', '-NZ']:
            if symbol.endswith(suffix):
                return False
        return True
    
    non_fo_stocks = [inst['tradingsymbol'] for inst in equity_instruments
                     if inst['tradingsymbol'] not in fo_stocks and is_regular_equity(inst['tradingsymbol'])]
    
    return non_fo_stocks, equity_instruments

import json

kite = get_kite_client()

def get_valid_non_fo_symbols():
    """Get valid non-F&O symbols with instrument data"""
    print("Fetching non-F&O stocks...")
    non_fo_stocks, all_instruments = get_non_fo_stocks()
    
    if not non_fo_stocks:
        print("No non-F&O stocks found. Using a sample list instead.")
        # Fallback to some known non-F&O stocks if API fails
        non_fo_stocks = ["INFY", "TCS", "WIPRO", "HCLTECH", "TECHM"]
    
    # Create a lookup dictionary for instrument data (only regular equities)
    instrument_map = {inst["tradingsymbol"]: inst for inst in all_instruments 
                     if inst["tradingsymbol"] in non_fo_stocks}
    
    print(f"Found {len(non_fo_stocks)} non-F&O stocks")
    return non_fo_stocks, instrument_map

from concurrent.futures import ThreadPoolExecutor, TimeoutError

def process_batch(symbols, progress_bar=None, chunk_size=100, timeout=20, max_retries=3):
    """Process a batch of symbols with rate limiting, larger chunk size, and robust timeout handling via ThreadPoolExecutor timeout. Prints each symbol processed."""
    batch_quotes = {}
    exchange_symbols = ["NSE:" + sym for sym in symbols]

    try:
        # Rate limit before making the API call
        rate_limiter.wait_for_rate_limit()

        # Process symbols in larger chunks (up to 100 per API call)
        for i in range(0, len(exchange_symbols), chunk_size):
            chunk = exchange_symbols[i:i + chunk_size]
            attempt = 0
            while attempt < max_retries:
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(kite.quote, chunk)
                        quote_response = future.result(timeout=timeout)
                    for key, quote in quote_response.items():
                        symbol = key.split(":")[1]
                        batch_quotes[symbol] = quote
                        # Print each processed symbol
                        safe_print(f"Processed: {symbol}")
                    break  # Success, exit retry loop
                except TimeoutError:
                    attempt += 1
                    safe_print(f"Timeout in chunk {i//chunk_size + 1}, retry {attempt}/{max_retries}")
                    if attempt == max_retries:
                        safe_print(f"Failed to fetch chunk after {max_retries} retries due to timeout.")
                    else:
                        time.sleep(3 * attempt)  # Exponential backoff
                except Exception as e:
                    safe_print(f"Error in chunk {i//chunk_size + 1}: {e}")
                    time.sleep(2)  # Longer delay on error
                    break  # For non-timeout errors, don't retry

    except Exception as e:
        safe_print(f"Error fetching batch: {e}")
    finally:
        # No progress bar update needed
        pass

    return batch_quotes

def fetch_quote_data(symbols, max_workers=3, batch_size=100, max_retries=3):
    """Fetch quote data in parallel with batching, rate limiting, and retries"""
    safe_print(f"Fetching quote data for {len(symbols)} symbols...")
    quote_data = {}

    # Split symbols into larger batches
    symbol_batches = [symbols[i:i + batch_size] 
                     for i in range(0, len(symbols), batch_size)]

    def fetch_batch_with_retry(batch, pbar):
        success = False
        batch_quotes = {}
        for attempt in range(max_retries):
            try:
                # Only rate limit right before actual API call
                batch_quotes = process_batch(batch, pbar, chunk_size=100)
                success = True
                break  # Success, exit retry loop
            except Exception as e:
                if "429" in str(e) or "Rate limit" in str(e):
                    wait_time = (attempt + 1) * 3  # Exponential backoff
                    safe_print(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                else:
                    safe_print(f"Error processing batch (attempt {attempt + 1}): {str(e)}")
                    if attempt == max_retries - 1:
                        safe_print(f"Failed to process batch after {max_retries} attempts")
                    else:
                        time.sleep(1)
        if not success:
            safe_print(f"Skipping failed batch of {len(batch)} symbols")
        return batch_quotes

    # Instead of tqdm progress bar, just process and print as we go
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_batch_with_retry, batch, None) for batch in symbol_batches]
        for future in as_completed(futures):
            batch_quotes = future.result()
            quote_data.update(batch_quotes)

    safe_print(f"Fetched quote data for {len(quote_data)} symbols.")
    return quote_data

def get_historical_high(symbol, days=30, max_retries=2):
    """Get the highest high for a stock over the last N days with retries"""
    for attempt in range(max_retries):
        try:
            # Get historical data for the last N days
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Rate limit before making the API call
            rate_limiter.wait_for_rate_limit()
            
            # Get instrument token
            instruments = kite.instruments("NSE")
            instrument = next((i for i in instruments if i["tradingsymbol"] == symbol), None)
            if not instrument:
                safe_print(f"Instrument not found for {symbol}")
                return 0
                
            # Add delay to avoid rate limiting
            time.sleep(2)  # Increased delay between historical data requests
            
            # Rate limit before historical data call
            rate_limiter.wait_for_rate_limit()
            
            # Fetch historical data with timeout
            historical_data = kite.historical_data(
                instrument_token=instrument["instrument_token"],
                from_date=from_date,
                to_date=to_date,
                interval="day",
                continuous=False,
                oi=False
            )
            
            if not historical_data:
                safe_print(f"No historical data found for {symbol}")
                return 0
                
            # Find the highest high in the historical data (excluding today)
            historical_highs = []
            for day in historical_data:
                if 'high' in day and day.get('high'):
                    historical_highs.append(day['high'])
            
            if not historical_highs:
                return 0
                
            historical_high = max(historical_highs)
            safe_print(f"{symbol}: 30D High = {historical_high}")
            return historical_high
            
        except Exception as e:
            if "429" in str(e):  # Rate limit error
                wait_time = (attempt + 1) * 2  # Exponential backoff
                safe_print(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries} for {symbol}")
                time.sleep(wait_time)
            else:
                safe_print(f"Error getting historical data for {symbol} (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:  # Last attempt failed
                    return 0
                        
            # Add extra delay before retry
            time.sleep(1)
    return 0  # If all retries failed

def process_stock(symbol, quote_data, instrument_map, result_queue):
    """
    Process a single stock's data robustly. Only adds valid results to the queue.
    Handles missing/invalid data gracefully and logs each result or error.
    """
    try:
        data = quote_data.get(symbol, {})
        instrument = instrument_map.get(symbol, {})
        if not data or not instrument:
            # safe_print(f"Skipping {symbol}: missing data or instrument info.")
            return
        last_price = data.get("last_price")
        prev_close = data.get("ohlc", {}).get("close")
        volume = data.get("volume")
        avg_vol = instrument.get("average_volume")
        if avg_vol is None or avg_vol == 0:
            # safe_print(f"Skipping {symbol}: missing or zero average_volume.")
            return
        if last_price is None or prev_close is None or volume is None:
            # safe_print(f"Skipping {symbol}: missing price/volume info.")
            return
        price_change = ((last_price - prev_close) / prev_close) * 100 if prev_close else 0
        volume_change = ((volume - avg_vol) / avg_vol) * 100 if avg_vol else 0
        result = {
            "Symbol": symbol,
            "Last_Price": last_price,
            "Prev_Close": prev_close,
            "Price_Change_%": price_change,
            "Volume": volume,
            "Avg_Volume": avg_vol,
            "Volume_Change_%": volume_change
        }
        result_queue.put(result)
        safe_print(f"Processed: {symbol}")
        
    except Exception as e:
        safe_print(f"Error processing stock {symbol}: {e}")

def get_ranked_non_fo_data(print_top_n=10, max_workers=2):
    """Get ranked non-F&O stocks based on volume with parallel processing
    
    Args:
        print_top_n: Number of top stocks to print
        max_workers: Number of parallel workers (keep this low to avoid rate limits)
    """
    valid_symbols, instrument_map = get_valid_non_fo_symbols()
    if not valid_symbols:
        safe_print("No non-F&O stocks found. Exiting...")
        return
        
    # Fetch all quotes in parallel with batching
    quote_data = fetch_quote_data(valid_symbols, max_workers=max_workers)
    
    # Process stocks with limited concurrency
    result_queue = Queue()
    results = []
    
    safe_print("\nProcessing stock data...")
    
    # Process stocks in smaller batches
    batch_size = 20
    total_batches = (len(valid_symbols) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, len(valid_symbols))
        batch_symbols = valid_symbols[batch_start:batch_end]
        
        safe_print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(batch_symbols)} symbols)")
        
        with tqdm(total=len(batch_symbols), desc=f"Batch {batch_num + 1}", unit="stock") as pbar:
            with ThreadPoolExecutor(max_workers=1) as executor:  # Single worker for this batch
                # Submit batch for processing
                futures = {
                    executor.submit(process_stock, symbol, quote_data, instrument_map, result_queue): symbol
                    for symbol in batch_symbols
                }
            
            # Process completed futures
            for future in as_completed(futures):
                pbar.update(1)

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Detect if market is closed (all volumes zero or script run outside market hours)
    def is_market_closed():
        # NSE market hours: 09:15 to 15:30 IST, Mon-Fri
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        is_weekend = now.weekday() >= 5
        if is_weekend or not (market_open <= now <= market_close):
            return True
        # If all volumes are zero, assume market is closed
        if all((r.get('Volume', 0) == 0 for r in results)):
            return True
        return False

    if is_market_closed():
        safe_print("\nMarket appears to be closed. Fetching last available data for all stocks...")
        fallback_results = []
        for symbol in valid_symbols:
            try:
                rate_limiter.wait_for_rate_limit()
                instruments = kite.instruments("NSE")
                instrument = next((i for i in instruments if i["tradingsymbol"] == symbol), None)
                if not instrument:
                    continue
                # Fetch last 2 days to get the most recent non-zero volume day
                to_date = datetime.now().strftime('%Y-%m-%d')
                from_date = (datetime.now() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
                historical_data = kite.historical_data(
                    instrument_token=instrument["instrument_token"],
                    from_date=from_date,
                    to_date=to_date,
                    interval="day",
                    continuous=False,
                    oi=False
                )
                # Find the most recent day with non-zero volume
                for day in reversed(historical_data):
                    if day.get('volume', 0) > 0:
                        fallback_results.append({
                            "Symbol": symbol,
                            "Last_Price": day.get("close"),
                            "Prev_Close": day.get("close"),
                            "Price_Change_%": 0.0,
                            "Volume": day.get("volume"),
                            "Avg_Volume": instrument.get("average_volume", 0),
                            "Volume_Change_%": 0.0,
                            "Source": "Last available"
                        })
                        break
            except Exception as e:
                safe_print(f"Error fetching fallback data for {symbol}: {e}")
        results = fallback_results
        safe_print(f"Fetched last available data for {len(results)} stocks.")

    safe_print(f"Total stocks processed: {len(results)}")
    if results:
        safe_print(f"Sample result: {results[0]}")

    # Always save all results for debugging
    all_results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_filename = f"volume_boosters_ALL_{timestamp}.csv"
    all_results_df.to_csv(all_filename, index=False)
    safe_print(f"All results saved to {all_filename}")

    if not results:
        safe_print("No data available to process.")
        return pd.DataFrame()

    # Create DataFrame and sort by absolute price change
    df = pd.DataFrame(results)
    safe_print(f"DataFrame shape before filtering: {df.shape}")
    if not df.empty:
        safe_print(f"Columns: {df.columns.tolist()}")
        safe_print(f"Head before filtering:\n{df.head()}\n")

    # Filter for significant volume increase (threshold adjustable)
    VOLUME_CHANGE_THRESHOLD = 20  # percent
    df_filtered = df[df['Volume_Change_%'] > VOLUME_CHANGE_THRESHOLD]
    safe_print(f"DataFrame shape after filtering (Volume_Change_% > {VOLUME_CHANGE_THRESHOLD}): {df_filtered.shape}")
    if not df_filtered.empty:
        safe_print(f"Head after filtering:\n{df_filtered.head()}\n")

    if df_filtered.empty:
        safe_print("No stocks with significant volume increase found.")
        return df_filtered

    # Filter out non-stock symbols (bonds, ETFs, REITs, mutual funds, trade-for-trade, etc.)
    import re
    def is_stock_symbol(symbol):
        non_stock_patterns = [
            r'(GB|NCD|SGB|BOND|ETF|IV|MF|INVIT)',   # Bonds, ETFs, Mutual Funds, InvITs
            r'-?(BE|BZ|BL|IV|ETF)$',                # Suffixes
            r'\d{2,}',                             # Maturity years in symbol (e.g., NCD2026)
        ]
        for pat in non_stock_patterns:
            if re.search(pat, str(symbol), re.IGNORECASE):
                return False
        return True

    df_filtered = df_filtered[df_filtered['Symbol'].apply(is_stock_symbol)]

    # Add absolute price change for sorting
    df_filtered['Abs_Price_Change_%'] = df_filtered['Price_Change_%'].abs()

    # Sort by absolute price change (descending)
    df_filtered = df_filtered.sort_values('Abs_Price_Change_%', ascending=False)

    # Save filtered results to CSV
    filename = f"volume_boosters_{timestamp}.csv"
    df_filtered.to_csv(filename, index=False)
    safe_print(f"Filtered results saved to {filename}")

    # Print summary and top N results
    TOP_N = 10
    safe_print(f"Showing top {TOP_N} stocks by absolute price change:")
    safe_print(df_filtered.head(TOP_N).to_string(index=False))
    return df_filtered
    
    # Common display columns
    display_cols = ['Symbol', 'LTP', 'Price_Change_%', 'Volume_Change_%', 'New_High']
    
    # Function to format and display a dataframe
    def display_stock_group(name, df, top_n=print_top_n):
        if df.empty:
            safe_print(f"\nNo {name} found.")
            return
            
        safe_print(f"\n{name} (Top {min(top_n, len(df))}):")
        display_df = df[display_cols].head(top_n).copy()
        
        # Format numeric columns with color coding
        for col in ['LTP', 'Price_Change_%', 'Volume_Change_%']:
            if col in display_df.columns:
                if col == 'Price_Change_%':
                    # Color code price changes
                    def color_price_change(val):
                        if pd.isna(val):
                            return 'N/A'
                        color = '\033[92m' if val >= 0 else '\033[91m'  # Green for positive, red for negative
                        return f"{color}{val:,.2f}%\033[0m"
                    display_df[col] = display_df[col].apply(color_price_change)
                else:
                    # Format other numeric columns
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:,.2f}{'%' if '%' in col else ''}" 
                        if pd.notnull(x) else 'N/A'
                    )
        
        # Print with formatting
        safe_print(display_df.to_string(index=False, justify='left'))
    
    # Display both gainers and losers
    display_stock_group("TOP GAINERS (Price + Volume Boost)", gainers)
    display_stock_group("TOP LOSERS (Price - Volume Boost)", losers)
    
    safe_print(f"\n\nFull results saved to: {filename}")
    safe_print(f"Total stocks analyzed: {len(df)}")
    safe_print(f"Gainers: {len(gainers)} | Losers: {len(losers)}")
    
    return df

if __name__ == "__main__":
    # Get and rank non-F&O stocks based on volume
    get_ranked_non_fo_data(print_top_n=20)
