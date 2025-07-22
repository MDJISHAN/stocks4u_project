# auth.py
import json
import logging
from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

def get_kite_client():
    try:
        # Load credentials
        with open("login_credentials.json", "r") as f:
            login_data = json.load(f)

        with open("access_token.json", "r") as f:
            token_data = json.load(f)

        # Check for required fields
        if 'api_key' not in login_data:
            raise ValueError("api_key not found in login_credentials.json")
                        
        if 'access_token' not in token_data:
            raise ValueError("access_token not found in access_token.json")

        api_key = login_data["api_key"]
        print("api_key",api_key)
        access_token = token_data["access_token"]
        print("access_token",access_token)

        # Initialize Kite client
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        kite.timeout = 60  # Increase timeout for all API requests

        # Verify connection
        try:
            profile = kite.profile()
            logger.info(f"✅ Connected to Kite API as {profile.get('user_name', 'Unknown')}")
        except Exception as e:
            logger.error(f"❌ Failed to verify Kite connection: {e}")
            raise

        return kite

    except FileNotFoundError as e:
        logger.error(f"Required credential file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in credential files: {e}")
        raise
    except Exception as e:
        logger.error(f"Error initializing Kite client: {e}")
        raise


"""def get_instrument_token(symbol, exchange="NSE"):
    
    Reads from pre-downloaded instruments.json and returns instrument_token for the given symbol.
    
    try:
        with open("instruments_tokens.json", "r") as f:
            instruments = json.load(f)

        for inst in instruments:
            if inst.get("tradingsymbol") == symbol and inst.get("exchange") == exchange:
                return inst.get("instrument_token")

        logger.warning(f"⚠️ Instrument token for {symbol} not found in instruments.json.")
        return None

    except FileNotFoundError:
        logger.error("❌ instruments_tokens.json not found.")
        return None
    except Exception as e:
        logger.error(f"Error while reading instrument token for {symbol}: {e}")
        return None"""
