from kiteconnect import KiteConnect
import json
from datetime import datetime

# Step 1: Put your details here
api_key = "1d266p9bi0sepv5h"
api_secret = "9d6a1z1973svy5tay5mrpxv49kyx5qzf"
request_token = "BqoG34Lkn66v3FDzjUFSzZJ1ZqdIdrKY"  # ← from Zerodha URL

# Step 2: Create KiteConnect object
kite = KiteConnect(api_key=api_key)

# Step 3: Try to generate session
try:
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    print("✅ Access Token:", access_token)

    # Fix: convert datetime to string before saving
    def convert(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError("Object of type %s is not JSON serializable" % type(obj))

    with open("login_credentials.json", "w") as f:
        json.dump(data, f, indent=2, default=convert)
        print("💾 Access token saved to login_credentials.json")

except Exception as e:
    print("❌ Failed to get access_token:", e)
