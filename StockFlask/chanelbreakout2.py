from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
import json
import time
from auth import get_kite_client
from select_filter import get_fo_stocks
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from typing import List, Dict, Any
import threading
from datetime import datetime, timedelta

import time


kite = get_kite_client()
# rate limiter class
from threading import Lock

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            self.calls = [call for call in self.calls if now - call < self.period]

            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    print(f"Rate limiter triggered: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)

            self.calls.append(time.time())



# Define valid timeframes & parameters
valid_timeframes = {
    "15minute": {"ma_length": 20, "atr_period": 14, "lookback_days": 10},
    "hour": {"ma_length": 20, "atr_period": 14, "lookback_days": 20},
    "day": {"ma_length": 20, "atr_period": 14, "lookback_days": 60},
    "week": {"ma_length": 15, "atr_period": 10, "lookback_days": 180},
    "month": {"ma_length": 15, "atr_period": 10, "lookback_days": 365}
}

# Cache for storing already fetched data to avoid redundant API calls
_data_cache = {}

# Fetch historical data with caching
def fetch_data(token, timeframe, days):
    cache_key = f"{token}_{timeframe}_{days}"
    
    # Return cached data if available
    if cache_key in _data_cache:
        return _data_cache[cache_key].copy()
    
    # Calculate date range
    to_date = datetime.today()
    from_date = to_date - timedelta(days=days)
    
    # Fetch data with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = kite.historical_data(token, from_date, to_date, timeframe)
            df = pd.DataFrame(data)
            
            # Process only if we have data
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                # Cache the result
                _data_cache[cache_key] = df.copy()
                return df
            return df
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error fetching data for {token} ({timeframe}): {str(e)}")
                return pd.DataFrame()
            time.sleep(1)  # Wait before retry

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

def detect_breakout(df, ma_length, atr_period, timeframe):
    try:
        if df.empty or len(df) < max(ma_length, atr_period) * 2:
            print(f"Not enough data for detection. Data points: {len(df)}, Required: {max(ma_length, atr_period) * 2}")
            return None
            
        print(f"\nProcessing {len(df)} candles for {timeframe} timeframe")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Latest close: {df['close'].iloc[-1]}, Volume: {df['volume'].iloc[-1]}")
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calculate 20-period EMA
        ema = pd.Series(close).ewm(span=20, adjust=False).mean().values
        
        # Calculate ATR more efficiently
        tr0 = np.abs(high - low)
        tr1 = np.abs(high - np.roll(close, 1))
        tr2 = np.abs(low - np.roll(close, 1))
        tr = np.maximum.reduce([tr0, tr1, tr2])
        atr = pd.Series(tr).rolling(window=atr_period).mean().values
        
        # Calculate channel using EMA +- ATR
        upper_channel = ema + (atr * 2)
        lower_channel = ema - (atr * 2)
        
        # Volume analysis
        volume_ma = pd.Series(volume).rolling(window=20).mean().values
        
        # Ensure we have enough data points for valid calculations
        min_required_bars = max(20, atr_period)  # At least 20 bars for EMA and ATR
        
        # Initialize with False arrays
        valid_breakout = np.full_like(close, False, dtype=bool)
        valid_confirmation = np.full_like(close, False, dtype=bool)
        
        # Only calculate breakouts where we have enough data
        for i in range(min_required_bars, len(close)):
            # Check if we have valid channel values
            if not (np.isnan(upper_channel[i-1]) or np.isnan(lower_channel[i-1])):
                # Breakout: Price closes above upper channel with volume > 1.5x average
                breakout_condition = (close[i] > upper_channel[i-1]) and (volume[i] > volume_ma[i-1] * 1.5)
                valid_breakout[i] = breakout_condition
                
                # Confirmation: Next candle closes above breakout candle's high
                if i < len(close) - 1:  # Ensure we have a next candle
                    confirmation_condition = breakout_condition and (close[i+1] > high[i])
                    valid_confirmation[i] = confirmation_condition
        
        # Final breakout signal (either initial breakout or confirmation)
        breakout_up = valid_breakout | valid_confirmation
        
        # Get all breakout rows where we have valid data
        valid_indices = np.where(~np.isnan(upper_channel) & ~np.isnan(lower_channel))[0]
        breakout_indices = np.intersect1d(
            np.where(breakout_up)[0],
            valid_indices,
            assume_unique=True
        )
        
        # Debug output
        if len(breakout_indices) > 0:
            print(f"Found {len(breakout_indices)} valid breakouts with proper channel data")
        
        print(f"Found {len(breakout_indices)} potential breakouts")
        if len(breakout_indices) > 0:
            print(f"Latest breakout at index: {breakout_indices[-1]} (time: {df.index[breakout_indices[-1]]})")
            # Get the most recent breakout
            latest_idx = breakout_indices[-1]
            
            # Get the actual timestamp from the index
            breakout_time = df.index[latest_idx]
            
            # Format the time - ensure we're working with a timezone-naive datetime
            if hasattr(breakout_time, 'tz_localize'):
                breakout_time = breakout_time.tz_localize(None)
            # Convert to string and extract the time part
            formatted_time = breakout_time.strftime('%Y-%m-%d %H:%M:%S')
            # For 5-minute candles, round down to the nearest 5 minutes
            if timeframe == '5minute':
                dt = pd.to_datetime(breakout_time)
                minutes = (dt.minute // 5) * 5
                formatted_time = dt.replace(minute=minutes, second=0).strftime('%Y-%m-%d %H:%M:%S')
            
            # Validate channel values at breakout point
            if np.isnan(upper_channel[latest_idx]) or np.isnan(lower_channel[latest_idx]):
                print(f"Skipping breakout at {breakout_time} - invalid channel values")
                return None
                
            # Calculate metrics
            breakout_price = close[latest_idx]
            current_price = close[-1]
            current_atr = atr[latest_idx]
            # Additional validation: Ensure price is actually above the EMA
            if close[latest_idx] <= ema[latest_idx]:
                print(f"Skipping breakout at {breakout_time} - price not above EMA")
                return None
            
            # Check if breakout is confirmed (next candle closed above breakout high)
            is_confirmed = False
            if latest_idx < len(close) - 1:  # If we have a next candle
                is_confirmed = close[latest_idx + 1] > high[latest_idx]
            
            # Calculate targets and stop loss
            target_1 = close[latest_idx] + atr[latest_idx]
            target_2 = close[latest_idx] + (2 * atr[latest_idx])
            stop_loss = close[latest_idx] - atr[latest_idx]
            
            # Calculate price change percentage
            price_change_pct = ((close[-1] - close[latest_idx]) / close[latest_idx]) * 100
            
            return {
                'breakout_time': formatted_time,
                'current_price': round(close[-1], 2),
                'breakout_price': round(close[latest_idx], 2),
                'price_change_pct': round(price_change_pct, 2),
                'target_1': round(target_1, 2),
                'target_2': round(target_2, 2),
                'stop_loss': round(stop_loss, 2),
                'volume': int(volume[latest_idx]),
                'volume_ma': round(volume_ma[latest_idx], 2),
                'breakout_confirmed': is_confirmed,
                'timeframe': timeframe
            }
            
    except Exception as e:
        print(f"Error in detect_breakout: {str(e)}")
    
    return None

'''class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            self.lock.acquire()
            try:
                now = time.time()
                # Remove calls older than the period
                self.calls = [t for t in self.calls if now - t < self.period]
                
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (now - self.calls[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                result = func(*args, **kwargs)
                self.calls.append(time.time())
                return result
            finally:
                self.lock.release()
        return wrapped'''

# Create a rate limiter: 5 calls per second
rate_limiter = RateLimiter(max_calls=1, period=1.2  )

def process_stock_batch(instruments, timeframes, rate_limiter):
    all_breakouts = []
    
    for instrument in instruments:
        with rate_limiter:
            breakouts = []
            for tf in timeframes:
                params = valid_timeframes[tf]
                try:
                    # Use the same data for all timeframes to reduce API calls
                    df = fetch_data(instrument['instrument_token'], tf, params['lookback_days'])
                    if df is not None and not df.empty:
                        result = detect_breakout(df, params['ma_length'], params['atr_period'], tf)
                        if result:
                            result['stock'] = instrument['tradingsymbol']
                            breakouts.append(result)
                            
                except Exception as e:
                    print(f"Error processing {instrument['tradingsymbol']} ({tf}): {str(e)}")
                    continue
            
            all_breakouts.extend(breakouts)
    
    return all_breakouts

def process_batch(batch, timeframes_config):
    """Process a batch of stocks with the given timeframes."""
    batch_breakouts = []
    print(f"\nProcessing batch of {len(batch)} items")


    for item in batch:
        # Handle string symbol or instrument dict
        if isinstance(item, str):
            print(f"\nProcessing symbol: {item}")
            instrument = {'tradingsymbol': item}
            instrument['instrument_token'] = get_instrument_token_for_symbol(item)
            if not instrument['instrument_token']:
                print(f"Could not find instrument_token for {item}")
                continue
        elif isinstance(item, dict):
            instrument = item
            print(f"\nProcessing instrument: {instrument.get('tradingsymbol', 'N/A')}")
            if 'instrument_token' not in instrument:
                print("Skipping instrument (missing instrument_token)")
                continue
        else:
            print(f"Skipping invalid item type: {type(item)}")
            continue

        for timeframe, params in timeframes_config.items():
            retries = 3
            while retries > 0:
                try:
                  rate_limiter.wait()  # ✅ throttle here before API call
                  df = fetch_data(
                      instrument['instrument_token'],
                      timeframe,
                      params['lookback_days']
                        )
                  break  # Success, exit retry loop
                except Exception as e:
                    if "Too many requests" in str(e) or "429" in str(e):
                        print(f"⚠️ Rate limit hit for {instrument['tradingsymbol']} ({timeframe}). Retrying in 2s...")
                        time.sleep(2)
                        retries -= 1
                    else:
                        raise e
            else:
                print(f"❌ Skipping {instrument['tradingsymbol']} due to repeated rate limits.")
                continue

            try:
                if df is None or df.empty:
                    continue

                result = detect_breakout(
                    df,
                    params['ma_length'],
                    params['atr_period'],
                    timeframe
                )

                if result:
                    result['stock'] = instrument.get('tradingsymbol', 'N/A')
                    batch_breakouts.append(result)

            except Exception as e:
                print(f"Error processing {instrument.get('tradingsymbol', 'unknown')} ({timeframe}): {str(e)}")
                continue

    return batch_breakouts


def find_breakouts(timeframes: Dict) -> List[Dict]:
    """Find breakouts across all stocks and timeframes with batching and retries."""
    all_breakouts = []
    batch_size = 10
    max_retries = 2
    failed_batches = []

    # Ensure we get a proper list of FO stocks
    raw_fo_stocks = get_fo_stocks()
    stocks_list = list(raw_fo_stocks) if not isinstance(raw_fo_stocks, list) else raw_fo_stocks
    total_batches = (len(stocks_list) - 1) // batch_size + 1

    for i in range(0, len(stocks_list), batch_size):
        batch = stocks_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"\nProcessing batch {batch_num}/{total_batches} (stocks {i+1}-{min(i+batch_size, len(stocks_list))})")

        success = False
        for attempt in range(max_retries):
            try:
                batch_results = process_batch(batch, timeframes)
                all_breakouts.extend(batch_results)
                success = True
                break  # Exit retry loop
            except Exception as e:
                print(f"⚠️ Error processing batch {batch_num} (attempt {attempt+1}/{max_retries}): {str(e)}")
                print("Sleeping 60 seconds before retry...")
                time.sleep(60)

        if not success:
            failed_batches.append(batch)
            print(f"❌ Batch {batch_num} permanently failed after {max_retries} attempts.")

        # Cool down after each batch regardless of success
        if i + batch_size < len(stocks_list):
            print("Cooling down for 5 seconds before next batch...")
            time.sleep(5)

    if failed_batches:
        print(f"\n❌ {len(failed_batches)} batches failed. You may retry them later.")

    return all_breakouts

'''@rate_limited_fetch
def fetch_data_with_retry(token, timeframe, days, max_retries=3):
    """Fetch data with retry logic for rate limiting."""
    for attempt in range(max_retries):
        try:
            return fetch_data(token, timeframe, days)
        except Exception as e:
            if "429" in str(e) or "Too many" in str(e):
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise
            else:
                raise
    return None'''

def process_stock(stock: str, token: int, timeframe: str, params: Dict) -> Dict:
    """Process a single stock and return breakout details if found."""
    try:
        # Add small jitter to prevent thundering herd
        time.sleep(random.uniform(0.05, 0.2))
        
        # Fetch data with retry logic
        df = fetch_data_with_retry(token, timeframe, params['lookback_days'])
        if df is None or df.empty:
            print(f"No data for {stock} ({timeframe})")
            return None
            
        breakout_details = detect_breakout(df, params, timeframe)
        if not breakout_details:
            return None
            
        return {
            'stock': stock,
            'timeframe': timeframe,
            'breakout_time': breakout_details['breakout_time'],
            'breakout_price': breakout_details['breakout_price'],
            'current_price': breakout_details['current_price'],
            'price_change_pct': breakout_details['price_change_pct'],
            'upper_channel': breakout_details['upper_channel'],
            'lower_channel': breakout_details['lower_channel'],
            'target_1': breakout_details['target_1'],
            'target_2': breakout_details['target_2'],
            'stop_loss': breakout_details['stop_loss'],
            'atr': breakout_details['atr'],
            'volume_ratio': breakout_details['volume_ratio'],
            'breakout_confirmed': breakout_details['breakout_confirmed']
        }
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Too many" in error_msg:
            print(f"Rate limited on {stock} ({timeframe})")
        else:
            print(f"Error processing {stock} ({timeframe}): {error_msg}")
        return None

# Fetch instrument token for stock symbol
def get_instrument_token_for_symbol(stock_symbol):
    instruments = kite.instruments()
    for instrument in instruments:
        if instrument["tradingsymbol"] == stock_symbol:
            return instrument["instrument_token"]
    return None

def get_user_timeframes():
    print("\nAvailable timeframes: 15minute, hour, day, week, month")
    print("Each timeframe has optimized parameters for better results")
    selected = input("Enter desired timeframes (comma-separated, or 'all'): ").strip()
    
    if selected.lower() == 'all':
        return list(valid_timeframes.keys())
    return [tf.strip() for tf in selected.split(",") if tf.strip() in valid_timeframes]

def validate_timeframes(user_timeframes):
    return {tf: valid_timeframes[tf] for tf in user_timeframes if tf in valid_timeframes}

def format_output(breakouts):
    if not breakouts:
        return {"status": "success", "message": "No breakouts found", "data": []}
    
    # Sort by timeframe and then by price change percentage
    sorted_breakouts = sorted(
        breakouts,
        key=lambda x: (x['timeframe'], x['price_change_pct']),
        reverse=True
    )
    
    return {
        "status": "success",
        "count": len(sorted_breakouts),
        "data": sorted_breakouts,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

if __name__ == "__main__":
    print("=== Enhanced Channel Breakout Scanner ===")
    print("Scanning for breakouts with volume confirmation...\n")
    
    try:
        # Get user input for timeframes
        selected_timeframes = get_user_timeframes()
        
        if not selected_timeframes:
            print("No valid timeframes selected. Exiting...")
            exit(1)
            
        # Validate and get parameters for selected timeframes
        timeframes_config = validate_timeframes(selected_timeframes)
        
        if not timeframes_config:
            print("No valid timeframes selected. Exiting...")
            exit(1)
            
        print(f"\nScanning timeframes: {', '.join(timeframes_config.keys())}")
        print("This may take a moment...\n")
        
        # Find and process breakouts
        breakouts = find_breakouts(timeframes_config)
        
        # Print results in a clean format
        if breakouts:
            print("\nBreakout Signals:")
            print("-" * 80)
            print(f"{'Stock':<10} {'TF':<8} {'Date/Time':<19} {'Price':<10} {'Chg%':<6} {'Timeframe'}")
            print("-" * 80)
            
            # Sort by time (most recent first)
            sorted_breakouts = sorted(
                breakouts, 
                key=lambda x: x['breakout_time'], 
                reverse=True
            )
            
            for breakout in sorted_breakouts:
                # Parse the datetime string to extract date and time
                dt = pd.to_datetime(breakout['breakout_time'])
                
                # Format based on timeframe
                if breakout['timeframe'] == '5minute':
                    # For 5min timeframe, show full datetime
                    time_str = dt.strftime('%Y-%m-%d %H:%M')
                else:
                    # For other timeframes, just show the time
                    time_str = dt.strftime('%Y-%m-%d %H:%M')
                
                print(f"{breakout['stock']:<10} "
                      f"{breakout['timeframe']:<8} "
                      f"{time_str:<19} "
                      f"{breakout['current_price']:<10.1f} ")
        else:
            print("\nNo breakout signals found.")
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nScan complete.")
