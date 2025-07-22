# pcr_calculation.py
from kiteconnect import KiteConnect
import pandas as pd
import datetime
import json
import os

from auth import get_kite_client

kite = get_kite_client()

# Fetch instrument list for NFO
instruments = pd.DataFrame(kite.instruments("NFO"))
instruments["expiry"] = pd.to_datetime(instruments["expiry"])  # Fix: ensure expiry is datetime

# Get nearest expiry date for index
def get_nearest_expiry(index):
    today = datetime.date.today()
    expiry_dates = sorted(set(instruments[instruments["name"] == index]["expiry"].dt.date))
    for expiry in expiry_dates:
        if expiry >= today:
            return expiry
    return None

# PCR calculation function
def calculate_pcr(index):
    expiry_date = get_nearest_expiry(index)
    if not expiry_date:
        return {"error": f"No expiry found for {index}"}

    options_chain = instruments[
        (instruments["name"] == index) &
        (instruments["expiry"].dt.date == expiry_date)
    ]

    # Separate Calls and Puts
    call_options = options_chain[options_chain["instrument_type"] == "CE"]
    put_options = options_chain[options_chain["instrument_type"] == "PE"]

    call_symbols = ["NFO:" + s for s in call_options["tradingsymbol"]]
    put_symbols = ["NFO:" + s for s in put_options["tradingsymbol"]]
    all_symbols = call_symbols + put_symbols

    try:
        quote_data = kite.quote(all_symbols)

        call_oi = sum(
            quote_data[s]["oi"] for s in call_symbols if s in quote_data and "oi" in quote_data[s]
        )
        put_oi = sum(
            quote_data[s]["oi"] for s in put_symbols if s in quote_data and "oi" in quote_data[s]
        )

        pcr = put_oi / call_oi if call_oi != 0 else 0
        return {
            "index": index,
            "expiry_date": str(expiry_date),
            "total_call_oi": call_oi,
            "total_put_oi": put_oi,
            "pcr": round(pcr, 2)
        }

    except Exception as e:
        return {"error": f"Error fetching market data for {index}: {str(e)}"}

# Main block to execute the function
if __name__ == "__main__":
    result = calculate_pcr("NIFTY")
    print(result)
    # You can now use 'result' as a Python dictionary for further processing

