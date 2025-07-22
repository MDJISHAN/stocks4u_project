import datetime
import pandas as pd
import os
import time
import pytz
from auth import get_kite_client

# Initialize Kite client
kite = get_kite_client()

# Indexes to track
indices = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"]
OI_FILE = "oi_data.csv"

# 🕒 Ask user for desired output timestamp (IST)
user_input = input("Enter output time in IST as 'YYYY-MM-DD HH:MM', or press Enter to use current time: ")
if user_input:
    try:
        # Parse user-specified IST timestamp
        ist = pytz.timezone("Asia/Kolkata")
        output_dt = datetime.datetime.strptime(user_input, "%Y-%m-%d %H:%M")
        output_dt = ist.localize(output_dt)
    except Exception:
        print("⚠️ Invalid format. Falling back to current IST time.")
        output_dt = None
else:
    output_dt = None

# 🕒 Helper: get IST timestamp string
def get_current_ist_time():
    if output_dt:
        return output_dt.strftime("%Y-%m-%d %H:%M:%S")
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")

# Check if market is open (Mon-Fri, 9:15–user-specified time or 15:30 IST)
def is_market_open():
    now = datetime.datetime.now(pytz.timezone("Asia/Kolkata"))
    if now.weekday() >= 5:
        return False
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    # Determine market end: use user-specified time if today and provided
    if output_dt and output_dt.date() == now.date():
        market_end = output_dt
    else:
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_start <= now <= market_end

# 🔁 Get nearest expiry for index
def get_nearest_expiry_date(index_name):
    try:
        instruments = kite.instruments("NFO")
        expiry_dates = sorted({
            inst["expiry"]
            for inst in instruments
            if inst.get("name") == index_name and "CE" in inst["tradingsymbol"]
        })
        return expiry_dates[0] if expiry_dates else None
    except Exception as e:
        print(f"❌ Error fetching expiry for {index_name}: {e}")
        return None

# ⏳ Get expiry for all indices
expiry_dates = {index: get_nearest_expiry_date(index) for index in indices}

# Fetch option tokens for each index
def get_option_tokens(index):
    expiry_date = expiry_dates.get(index)
    if not expiry_date:
        print(f"⚠️ No expiry date found for {index}")
        return {}
    try:
        instruments = kite.instruments("NFO")
        return {
            str(inst["instrument_token"]): {
                "tradingsymbol": inst["tradingsymbol"],
                "strike_price": inst["strike"],
                "option_type": "CE" if "CE" in inst["tradingsymbol"] else "PE"
            }
            for inst in instruments
            if inst.get("name") == index and inst.get("expiry") == expiry_date and ("CE" in inst["tradingsymbol"] or "PE" in inst["tradingsymbol"])
        }
    except Exception as e:
        print(f"❌ Error fetching tokens for {index}: {e}")
        return {}

# Load previous historical OI data
def load_historical_oi():
    if os.path.exists(OI_FILE) and os.path.getsize(OI_FILE) > 0:
        try:
            return pd.read_csv(OI_FILE).to_dict(orient="records")
        except Exception:
            return []
    return []

# Save the snapshot data for next run (overwrite CSV)
def save_snapshot(snapshot_data):
    pd.DataFrame(snapshot_data).to_csv(OI_FILE, index=False)

# Fetch OI data and return full historical + current data
def fetch_all_oi_data():
    historical = load_historical_oi()
    current_snapshot = []
    combined_data = []

    if not is_market_open():
        print("⚠️ Current time is outside market window. Data may be stale or incomplete.")

    for index in indices:
        tokens = get_option_tokens(index)
        if not tokens:
            continue
        try:
            quotes = kite.quote(list(tokens.keys()))
            time.sleep(0.5)
            for tk, data in quotes.items():
                info = tokens[tk]
                ts = get_current_ist_time()
                oi_val = int(data.get("oi", 0))
                record = {
                    "Index": index,
                    "Strike Price": info["strike_price"],
                    "Option Type": info["option_type"],
                    "Tradingsymbol": info["tradingsymbol"],
                    "OI": oi_val,
                    "Timestamp": ts
                }
                combined_data.append(record)
                current_snapshot.append({
                    "Index": index,
                    "Strike Price": info["strike_price"],
                    "Option Type": info["option_type"],
                    "OI": oi_val
                })
        except Exception as e:
            print(f"❌ Error fetching data for {index}: {e}")

    # Save latest snapshot
    save_snapshot(current_snapshot)

    # Combine historical and current
    return historical + combined_data

# --- API-friendly wrapper ---
def fetch_oi_data(return_debug=False):
    """
    Returns OI data in dictionary format for API use.
    Output: (oi_data, summary, diagnostics) if return_debug else (oi_data, summary, None)
    """
    diagnostics = {}
    try:
        oi_data = fetch_all_oi_data()
        diagnostics['oi_data_len'] = len(oi_data)
        # Simple summary: total OI per index
        summary = {}
        for row in oi_data:
            idx = row.get('Index', 'UNKNOWN')
            summary.setdefault(idx, 0)
            summary[idx] += row.get('OI', 0)
        diagnostics['summary_keys'] = list(summary.keys())
        if return_debug:
            return oi_data, summary, diagnostics
        else:
            return oi_data, summary, None
    except Exception as e:
        diagnostics['error'] = str(e)
        if return_debug:
            return [], {}, diagnostics
        else:
            return [], {}, None

# 🔄 Run and display all data
def main():
    all_data = fetch_all_oi_data()
    df = pd.DataFrame(all_data)
    pd.set_option('display.max_rows', None)
    print(df)

if __name__ == "__main__":
    main()
