import json
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ----------------------------
# CONFIGURATION
# ----------------------------
KITE_EMAIL = "uppalroy127020027@gmail.com"
KITE_PASSWORD = "Uppal@12/07/Roy"
KITE_API_KEY = "LDtrGhB9RuorphzxOEK1xcd8Hx1Uwdwy"
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "kite_token.json")
LOGIN_URL = f"https://kite.zerodha.com/connect/login?api_key={KITE_API_KEY}"

# ----------------------------
# FUNCTIONS
# ----------------------------
def save_token(token):
    data = {"access_token": token, "timestamp": time.time()}
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f)

def load_token():
    if not os.path.exists(TOKEN_FILE):
        return None
    with open(TOKEN_FILE, "r") as f:
        return json.load(f)

def is_token_valid():
    token_data = load_token()
    if not token_data:
        return False
    # Consider token invalid if older than 23 hours (to renew before expiry)
    return time.time() - token_data["timestamp"] < 23 * 3600

def get_kite_token():
    if is_token_valid():
        return load_token()["access_token"]

    # ----------------------------
    # SETUP HEADLESS CHROME
    # ----------------------------
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(LOGIN_URL)

        # Wait for login fields
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "userid"))
        )

        # Enter email and password
        driver.find_element(By.ID, "TPR956").send_keys(KITE_EMAIL)
        driver.find_element(By.ID, "Uppal@12/07/Roy").send_keys(KITE_PASSWORD)
        driver.find_element(By.XPATH, "//button[@type='submit']").click()

        # Wait for 2FA / PIN page (you may need to handle 2FA here)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "pin"))
        )

        # Enter PIN (if needed, or skip if OTP is required via app)
        # driver.find_element(By.ID, "pin").send_keys(KITE_PIN)
        # driver.find_element(By.XPATH, "//button[@type='submit']").click()

        # Wait for redirect URL containing request_token
        WebDriverWait(driver, 30).until(
            lambda d: "request_token" in d.current_url
        )

        # Extract request_token from URL
        url = driver.current_url
        request_token = url.split("request_token=")[1].split("&")[0]

        # Exchange request_token for access_token using Zerodha API
        import requests
        res = requests.post(
            "https://api.kite.trade/session/token",
            data={"api_key": KITE_API_KEY, "request_token": request_token, "checksum": ""}  # Add checksum logic if needed
        )
        data = res.json()
        access_token = data["data"]["access_token"]

        save_token(access_token)
        return access_token

    finally:
        driver.quit()

# ----------------------------
# USAGE
# ----------------------------
if __name__ == "__main__":
    token = get_kite_token()
    print("Access Token:", token)
