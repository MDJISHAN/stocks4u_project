from flask import Flask, redirect, request
from kiteconnect import KiteConnect
import json
import os

app = Flask(__name__)

# Put your actual values here
API_KEY = "1d266p9bi0sepv5h"
API_SECRET = "9d6a1z1973svy5tay5mrpxv49kyx5qzf"
REDIRECT_URL = "https://stocks4u.in/callback"  # Must be whitelisted in Kite dev console

kite = KiteConnect(api_key=API_KEY)

@app.route('/')
def login():
    # Redirects user to Zerodha login page
    login_url = kite.login_url()
    return redirect(login_url)

@app.route('/login/callback')
def callback():
    request_token = request.args.get("request_token")
    if not request_token:
        return "Missing request token", 400

    try:
        # Generate access_token
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]

        # Save the token to a JSON file
        credentials_path = os.path.join('access_token', 'login_credentials.json')
        os.makedirs(os.path.dirname(credentials_path), exist_ok=True)
        with open(credentials_path, 'w') as f:
            json.dump(data, f)

        # Redirect user to your app/dashboard
        return redirect("https://your-app-url.com/dashboard")  # Change this to your actual app URL

    except Exception as e:
        print("Error:", e)
        return "Login failed", 500

if __name__ == '__main__':
    app.run(debug=True)
