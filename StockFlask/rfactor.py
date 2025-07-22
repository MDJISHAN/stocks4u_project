import pandas as pd
from datetime import datetime
from kiteconnect import KiteConnect
from auth import get_kite_client
from select_filter import get_fo_stocks
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

kite = get_kite_client()

def get_nse_instruments():
    instruments = kite.instruments("NSE")
    return {inst['tradingsymbol']: inst for inst in instruments if inst['segment'] == 'NSE'}

def get_option_instruments():
    return kite.instruments("NFO")

def find_atm_strike_price(ltp, step=50):
    return int(round(ltp / step) * step)

def get_atm_option_tokens(stock, option_instruments, ltp):
    atm_strike = find_atm_strike_price(ltp)
    print(f"[DEBUG] Finding ATM tokens for {stock} at strike {atm_strike}")

    stock_options = [
        inst for inst in option_instruments
        if inst['name'] == stock and inst['strike'] == atm_strike
    ]

    expiries = sorted(list(set(inst['expiry'] for inst in stock_options)))
    for expiry in expiries:
        ce_token, pe_token = None, None
        for inst in stock_options:
            if inst['expiry'] != expiry:
                continue
            if inst['instrument_type'] == 'CE':
                ce_token = inst['instrument_token']
            elif inst['instrument_type'] == 'PE':
                pe_token = inst['instrument_token']
        print(f"[DEBUG] Trying expiry {expiry} for {stock}: CE token={ce_token}, PE token={pe_token}")
        if ce_token and pe_token:
            return ce_token, pe_token, atm_strike, expiry
    print(f"[DEBUG] No valid CE/PE for {stock} at {atm_strike}")
    return None, None, atm_strike, None

def batcher(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch

import json
import os

PREV_PRICE_FILE = 'previous_prices.json'
# Load previous prices from file if it exists
if os.path.exists(PREV_PRICE_FILE):
    with open(PREV_PRICE_FILE, 'r') as f:
        previous_prices = json.load(f)
else:
    previous_prices = {}

def save_previous_prices():
    with open(PREV_PRICE_FILE, 'w') as f:
        json.dump(previous_prices, f)

from datetime import datetime, timedelta

def fetch_previous_price(kite, instrument_token):
    # Fetch previous day's close price
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    from_date = yesterday.replace(hour=15, minute=29, second=0, microsecond=0)
    to_date = yesterday.replace(hour=15, minute=30, second=0, microsecond=0)
    try:
        data = kite.historical_data(
            instrument_token,
            from_date.strftime('%Y-%m-%d %H:%M:%S'),
            to_date.strftime('%Y-%m-%d %H:%M:%S'),
            interval='minute'
        )
        if data:
            return data[-1]['close']  # Last close price of yesterday
    except Exception as e:
        print(f"[ERROR] Fetching historical price for {instrument_token}: {e}")
    return None

def process_stock_batch(stock_batch, nse_instruments, option_instruments):
    batch_data = []
    stock_tokens = [nse_instruments[stock]['instrument_token'] for stock in stock_batch]

    try:
        ltp_dict = kite.ltp(stock_tokens)
    except Exception as e:
        print(f"[ERROR] LTP fetch failed: {e}")
        return batch_data

    batch_option_tokens = []
    stock_ce_pe_map = {}
    stock_ltp_map = {}

    for stock in stock_batch:
        try:
            ltp = ltp_dict[str(nse_instruments[stock]['instrument_token'])]['last_price']
            stock_ltp_map[stock] = ltp
            ce_token, pe_token, atm_strike, expiry = get_atm_option_tokens(stock, option_instruments, ltp)
            if ce_token and pe_token:
                stock_ce_pe_map[stock] = (ce_token, pe_token, atm_strike, expiry)
                batch_option_tokens.extend([ce_token, pe_token])
            else:
                print(f"[DEBUG] No CE/PE token for {stock} (ATM strike: {atm_strike})")
        except Exception as e:
            print(f"[ERROR] Error with {stock}: {e}")

    if not batch_option_tokens:
        print("[DEBUG] One batch returned no data.")
        return batch_data

    try:
        quotes = kite.ltp(batch_option_tokens)
    except Exception as e:
        print(f"[ERROR] Option LTP fetch failed: {e}")
        return batch_data

    for stock, (ce_token, pe_token, atm_strike, expiry) in stock_ce_pe_map.items():
        try:
            ce_quote = quotes.get(str(ce_token), {})
            pe_quote = quotes.get(str(pe_token), {})

            print(f"[DATA] {stock} CE LTP: {ce_quote.get('last_price')} CE Close: {ce_quote.get('ohlc', {}).get('close')}")
            print(f"[DATA] {stock} PE LTP: {pe_quote.get('last_price')} PE Close: {pe_quote.get('ohlc', {}).get('close')}")
            # Debug prints for diagnosing percentage change calculation
            print(f"[DEBUG] {stock} ce_quote: {ce_quote}")
            print(f"[DEBUG] {stock} pe_quote: {pe_quote}")
            # --- Fetch previous price from API ---
            curr_ce_price = ce_quote.get('last_price')
            curr_pe_price = pe_quote.get('last_price')
            prev_ce_price = fetch_previous_price(kite, ce_token)
            prev_pe_price = fetch_previous_price(kite, pe_token)
            print(f"[API PREV] {stock} CE: prev={prev_ce_price}, curr={curr_ce_price}")
            print(f"[API PREV] {stock} PE: prev={prev_pe_price}, curr={curr_pe_price}")
            ce_change = None
            pe_change = None
            if prev_ce_price is not None and curr_ce_price is not None and prev_ce_price != 0:
                ce_change = ((curr_ce_price - prev_ce_price) / prev_ce_price) * 100
                print(f"[API CHANGE] {stock} CE: pct_change={ce_change}")
            if prev_pe_price is not None and curr_pe_price is not None and prev_pe_price != 0:
                pe_change = ((curr_pe_price - prev_pe_price) / prev_pe_price) * 100
                print(f"[API CHANGE] {stock} PE: pct_change={pe_change}")

            # Calculate % change for CE and PE using API-fetched previous prices
            ce_change = None
            pe_change = None
            direction = None
            if prev_ce_price is not None and curr_ce_price is not None and prev_ce_price != 0:
                ce_change = ((curr_ce_price - prev_ce_price) / prev_ce_price) * 100
            if prev_pe_price is not None and curr_pe_price is not None and prev_pe_price != 0:
                pe_change = ((curr_pe_price - prev_pe_price) / prev_pe_price) * 100

            # Always use both (even if 0%) and pick the greater absolute change
            if ce_change is not None and pe_change is not None:
                if abs(ce_change) >= abs(pe_change):
                    max_change = ce_change
                    direction = 'CE'
                else:
                    max_change = pe_change
                    direction = 'PE'
            elif ce_change is not None:
                max_change = ce_change
                direction = 'CE'
            elif pe_change is not None:
                max_change = pe_change
                direction = 'PE'
            else:
                max_change = 0.0
                direction = 'N/A'
            batch_data.append({
                'Stock': stock,
                'Direction': direction,
                'Change %': round(max_change, 2) if max_change is not None else 0.0,
                'ATM Strike': atm_strike,
                'Expiry': expiry,
                'CE LTP': ce_quote.get('last_price', ''),
                'PE LTP': pe_quote.get('last_price', ''),
                'CE Prev': prev_ce_price,
                'PE Prev': prev_pe_price,
            })

        except Exception as e:
            print(f"[ERROR] Failed processing {stock} quotes: {e}")

    return batch_data

def fetch_and_rank_changes(batch_size=10, max_workers=4):
    nse_instruments = get_nse_instruments()
    option_instruments = get_option_instruments()
    data = []

    stocks = [s for s in get_fo_stocks() if s in nse_instruments]
    batches = list(batcher(stocks, batch_size))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_stock_batch, batch, nse_instruments, option_instruments)
            for batch in batches
        ]
        for future in as_completed(futures):
            data.extend(future.result())

    ranked_df = pd.DataFrame(data)
    if ranked_df.empty:
        print("[DEBUG] Final DataFrame is empty. Printing raw batch data and ranking...")
        if not data:
            print("[DEBUG] No data collected from any stock batch.")
            return ranked_df
        # Rank the raw data by Change %
        sorted_data = sorted(data, key=lambda x: x['Change %'], reverse=True)
        print("Stock | Direction | Change % | ATM Strike | Expiry")
        for entry in sorted_data:
            print(f"{entry['Stock']} | {entry['Direction']} | {entry['Change %']} | {entry.get('ATM Strike','')} | {entry.get('Expiry','')}")
        # Return as DataFrame for compatibility
        return pd.DataFrame(sorted_data)

    result = ranked_df.sort_values(by='Change %', ascending=False).reset_index(drop=True)
    save_previous_prices()
    result.to_csv('option_rankings.csv', index=False)
    print("[INFO] Saved option_rankings.csv")
    return result


if __name__ == "__main__":
    df = fetch_and_rank_changes()
    if df.empty:
        print("[DEBUG] No ranked DataFrame to display.")
    else:
        # Show all columns including ATM Strike and Expiry
        print(df[['Stock','Direction','Change %','ATM Strike','Expiry']].to_string(index=False))
