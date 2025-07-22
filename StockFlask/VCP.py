import pandas as pd
import numpy as np
import sys
import os
import time
import threading
from queue import Queue
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from auth import get_kite_client

# Constants for rate limiting
MAX_REQUESTS_PER_MINUTE = 30  # Kite Connect's rate limit is 30 requests per minute
REQUEST_INTERVAL = 60.0 / MAX_REQUESTS_PER_MINUTE  # Time to wait between requests in seconds

# Thread-safe rate limiter
class RateLimiter:
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.lock = threading.Lock()

    def wait_for_rate_limit(self):
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.interval:
                time.sleep(self.interval - time_since_last)
            
            self.last_request_time = time.time()

# Global rate limiter
rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)

# Thread-safe print
print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

# Add the current directory to path to import select_filter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from select_filter import get_fo_stocks

# Initialize Kite client
kite = get_kite_client()

def calculate_contractions(data, num_contractions=3, lookback_days=120, timeframe='1d'):
    """
    Optimized VCP contraction calculation using vectorized operations
    """
    try:
        # Use numpy arrays for faster computation
        highs = data['High'].values
        lows = data['Low'].values
        
        # Simple swing high/low detection (faster than rolling)
        swing_highs = (highs[1:-1] > highs[0:-2]) & (highs[1:-1] > highs[2:])
        swing_lows = (lows[1:-1] < lows[0:-2]) & (lows[1:-1] < lows[2:])
        
        # Get the indices of swing points
        high_indices = np.where(swing_highs)[0] + 1  # +1 because we sliced the array
        low_indices = np.where(swing_lows)[0] + 1
        
        # We need at least 2 highs and 1 low for a pattern
        if len(high_indices) < 2 or len(low_indices) < 1:
            return None
            
        # Get the last few swing points (most recent)
        last_highs = high_indices[-3:]  # Look at last 3 highs max
        last_lows = low_indices[-3:]     # Look at last 3 lows max
        
        contractions = []
        
        # Check for VCP pattern in recent swings
        for i in range(1, min(3, len(last_highs))):
            if i >= len(last_highs) or (i-1) >= len(last_lows):
                break
                
            high1_idx = last_highs[-i-1]
            low_idx = last_lows[-i]
            high2_idx = last_highs[-i]
            
            # Ensure proper high-low-high sequence
            if not (high1_idx < low_idx < high2_idx):
                continue
                
            high1 = highs[high1_idx]
            low = lows[low_idx]
            high2 = highs[high2_idx]
            
            # Basic validation
            if low >= high1 or low >= high2:
                continue
                
            contraction = (high1 - low) / high1 * 100
            contractions.append(contraction)
        
        # Need at least 2 contractions in descending order
        if len(contractions) >= 2 and all(contractions[i] > contractions[i+1] for i in range(len(contractions)-1)):
            return contractions
            
    except Exception as e:
        safe_print(f"Error in calculate_contractions: {e}")
        
    return None

    # Old implementation (kept for reference)
    # Convert to numpy arrays for faster computation
    highs_values = highs.values
    lows_values = lows.values
    
    contractions = []
    i = 0
    data_length = len(highs_values)
    
    while i < data_length - 1 and len(contractions) < num_contractions:
        # Find swing high
        if i >= len(highs_values[i:]):
            break
            
        high_idx = i + np.argmax(highs_values[i:])
        if high_idx >= data_length:
            break
            
        high_price = data.iloc[high_idx]['High']
        
        # Find next swing low
        if high_idx + 1 >= data_length:
            break
            
        low_window = lows_values[high_idx + 1:min(high_idx + 1 + 20, data_length)]  # Limit lookahead
        if len(low_window) == 0:
            break
            
        low_idx = high_idx + 1 + np.argmin(low_window)
        if low_idx >= data_length:
            break
            
        low_price = data.iloc[low_idx]['Low']
        
        # Calculate contraction percentage
        contraction = (high_price - low_price) / high_price * 100
        contractions.append(contraction)
        
        # Move to after the current low
        i = low_idx + 1
    
    # Check if we have enough contractions and they're in descending order
    if len(contractions) < num_contractions:
        return None
        
    # Only return the requested number of contractions
    contractions = contractions[:num_contractions]
    
    # Check if each contraction is smaller than the previous
    if all(contractions[i] > contractions[i+1] for i in range(len(contractions)-1)):
        return contractions
    return None

# Cache for instrument tokens to avoid repeated API calls
instrument_cache = {}
instrument_cache_lock = threading.Lock()

def get_instrument_token(symbol):
    """Get instrument token for a symbol with caching"""
    with instrument_cache_lock:
        if symbol in instrument_cache:
            return instrument_cache[symbol]
            
        try:
            rate_limiter.wait_for_rate_limit()
            instruments = kite.instruments()
            df_instruments = pd.DataFrame(instruments)
            
            instrument = df_instruments[
                (df_instruments['tradingsymbol'] == symbol) & 
                (df_instruments['exchange'] == 'NSE')
            ].iloc[0]
            
            instrument_cache[symbol] = instrument['instrument_token']
            return instrument['instrument_token']
            
        except Exception as e:
            safe_print(f"Error getting instrument token for {symbol}: {e}")
            return None

def get_historical_data(symbol, from_date, to_date, interval='day'):
    """
    Fetch historical data for a symbol using Kite Connect with rate limiting
    
    Args:
        symbol: Stock symbol
        from_date: Start date in 'YYYY-MM-DD' or datetime object
        to_date: End date in 'YYYY-MM-DD' or datetime object
        interval: 'minute', 'day', etc. Default is 'day' for daily data
    """
    """
    Fetch historical data for a symbol using Kite Connect with rate limiting
    """
    try:
        # Get instrument token with rate limiting
        instrument_token = get_instrument_token(symbol)
        if not instrument_token:
            return None
        
        # Convert dates to datetime objects if they're strings
        if isinstance(from_date, str):
            from_date = datetime.strptime(from_date, '%Y-%m-%d')
        if isinstance(to_date, str):
            to_date = datetime.strptime(to_date, '%Y-%m-%d')
        
        # Kite Connect expects dates in 'yyyy-mm-dd' format
        from_date_str = from_date.strftime('%Y-%m-%d')
        to_date_str = to_date.strftime('%Y-%m-%d')
        
        # Apply rate limiting
        rate_limiter.wait_for_rate_limit()
        
        # For daily data, we need to ensure we get complete days
        if interval == 'day':
            interval = 'day'
            # For daily data, we can fetch more historical data if needed
            from_date_obj = datetime.strptime(from_date_str, '%Y-%m-%d') if isinstance(from_date_str, str) else from_date_str
            to_date_obj = datetime.strptime(to_date_str, '%Y-%m-%d') if isinstance(to_date_str, str) else to_date_str
            
            # Ensure we have enough data points for the analysis
            min_days_needed = 20  # We only need 20 days of data
            if (to_date_obj - from_date_obj).days < min_days_needed:
                from_date_obj = to_date_obj - timedelta(days=min_days_needed)
                from_date_str = from_date_obj.strftime('%Y-%m-%d')
        
        # Fetch historical data
        try:
            records = kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date_str,
                to_date=to_date_str,
                interval=interval,
                continuous=False,
                oi=False
            )
        except Exception as e:
            safe_print(f"Error fetching data for {symbol}: {e}")
            return None
        
        if not records:
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Set date as index and convert to datetime
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Rename columns to match yfinance format
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        return df
        
    except Exception as e:
        safe_print(f"Error fetching data for {symbol}: {e}")
        return None

def process_stock(symbol, start_date, result_queue):
    """Optimized stock processing with early exits"""
    try:
        # Use yesterday's date as end date
        end_date = datetime.now() - timedelta(days=1)
        effective_end_date = end_date.strftime('%Y-%m-%d')
        
        # Fetch only necessary columns and limit data points
        data = get_historical_data(symbol, start_date, effective_end_date, interval='day')
        
        # Early exit conditions
        if data is None or len(data) < 8:  # Reduced from 10 to 8 for more leniency
            return
            
        # Quick price movement check (faster than calculating max/min)
        price_range = data['High'].iloc[-1] - data['Low'].iloc[0]
        if price_range / data['Low'].iloc[0] < 0.04:  # Slightly more lenient 4% threshold
            return
        
        # Only keep recent data (last 30 days max) to speed up processing
        if len(data) > 30:
            data = data.iloc[-30:]

        contractions = calculate_contractions(data)
        if contractions:
            result_queue.put((symbol, contractions))
            safe_print(f"✅ Found VCP for {symbol}")
        else:
            safe_print(f"❌ No VCP for {symbol}")
            
    except Exception as e:
        safe_print(f"⚠️  Error processing {symbol}: {e}")

def detect_vcp_stocks(stock_list, start_date='2024-01-01', end_date=None, max_workers=8):  
    """
    Optimized parallel processing with batch processing and better resource management
    """
    result_queue = Queue()
    vcp_stocks = []
    
    # Process in smaller batches to prevent memory issues
    batch_size = 20
    
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_stock, symbol, start_date, result_queue): symbol 
                for symbol in batch
            }
            
            # Process batch results
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    future.result()  # Check for exceptions
                except Exception as e:
                    safe_print(f"⚠️  Error processing {symbol}: {e}")
        
        # Collect results after each batch
        batch_results = []
        while not result_queue.empty():
            result = result_queue.get()
            if result is not None:
                batch_results.append(result)
        
        vcp_stocks.extend(batch_results)
        
        # Small delay between batches
        if i + batch_size < len(stock_list):
            time.sleep(1)
    
    return vcp_stocks

def main():
    try:
        # Get F&O stocks
        safe_print("Fetching F&O stocks...")
        fo_stocks = get_fo_stocks()  # Get base symbols without .NS
        
        if not fo_stocks:
            safe_print("No F&O stocks found. Please check your connection and try again.")
            return
            
        safe_print(f"Found {len(fo_stocks)} F&O stocks")
        
        # For 1-day timeframe, get last 20 days of historical data
        end_date = datetime.now() - timedelta(days=1)  # Yesterday's date
        start_date = (end_date - timedelta(days=20)).strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        safe_print(f"\n🔍 Scanning for VCP patterns in 1-day timeframe...")
        safe_print(f"Analyzing last 20 days of data: {start_date} to {end_date_str}")
        safe_print(f"Using {min(5, len(fo_stocks))} worker threads")
        
        # Process stocks in batches of 50 (smaller batch size for 1-day data)
        batch_size = 50
        total_batches = (len(fo_stocks) - 1) // batch_size + 1
        all_vcp_stocks = []
        
        start_time = time.time()
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, len(fo_stocks))
            current_batch = fo_stocks[batch_start:batch_end]
            
            safe_print(f"\n{'='*70}")
            safe_print(f"📊 Processing batch {batch_num + 1}/{total_batches} (stocks {batch_start + 1}-{batch_end})")
            safe_print(f"{'='*70}")
            
            batch_start_time = time.time()
            
            # Process current batch
            batch_vcp_stocks = detect_vcp_stocks(
                current_batch,
                start_date=start_date,
                end_date=end_date_str,
                max_workers=5
            )
            all_vcp_stocks.extend(batch_vcp_stocks)
            
            batch_elapsed = time.time() - batch_start_time
            safe_print(f"\n✅ Batch {batch_num + 1} completed in {batch_elapsed:.2f} seconds")
            safe_print(f"✅ Found {len(batch_vcp_stocks)} VCP patterns in this batch")
            
            # Add a small delay between batches to avoid rate limiting
            if batch_num < total_batches - 1:
                safe_print("\n⏳ Waiting before next batch...")
                time.sleep(3)  # 3 second delay between batches
        
        # Calculate and display overall performance metrics
        elapsed_time = time.time() - start_time
        safe_print(f"\n{'='*70}")
        safe_print(f"✅ Processed {len(fo_stocks)} stocks in {elapsed_time/60:.2f} minutes")
        safe_print(f"⏱️  Average time per stock: {elapsed_time/len(fo_stocks):.2f} seconds")
        
        # Print results
        safe_print("\n" + "="*70)
        safe_print("📈 Stocks forming VCP with 3 contractions:")
        safe_print("="*70)
        
        if all_vcp_stocks:
            # Sort by the first contraction percentage (descending)
            all_vcp_stocks.sort(key=lambda x: x[1][0] if x[1] else 0, reverse=True)
            
            for stock, contractions in all_vcp_stocks:
                safe_print(f"{stock}: contractions = {[f'{c:.2f}%' for c in contractions]}")
            
            # Save results to CSV
            result_df = pd.DataFrame([{
                'Stock': stock,
                'Contraction1%': contractions[0] if len(contractions) > 0 else None,
                'Contraction2%': contractions[1] if len(contractions) > 1 else None,
                'Contraction3%': contractions[2] if len(contractions) > 2 else None,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            } for stock, contractions in all_vcp_stocks])
            
            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f'results/vcp_stocks_{timestamp}.csv'
            result_df.to_csv(csv_filename, index=False)
            safe_print(f"\n✅ Results saved to {os.path.abspath(csv_filename)}")
        else:
            safe_print("❌ No stocks forming VCP pattern found.")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
