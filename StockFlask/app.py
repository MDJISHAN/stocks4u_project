from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import time
import logging
import json
import pandas as pd
from allbreakout import get_breakouts
from auth import get_kite_client
from select_filter import get_fo_stocks
from breakouttenorfiftyday import check_breakout
from breakoutthreepercentage import get_fo_stocks
from chanelbreakout2 import find_breakouts
from highpowerstock import analyze_high_turnover_stocks_live, get_fo_stocks
from index_movers import analyze_indices, kite, indices, index_constituents
import io
from topandlowlevel import rank_fo_stocks
import sys
from intradayboost import rank_fo_stocks_by_growth
from momentum import scan_fo_stocks_dual
from ohlctopgainerloser import get_top_gainers_and_losers
from oi_calculation import fetch_oi_data 
from pcr_calculation import calculate_pcr
from sector_data2 import get_sector_abnormal_growth, get_all_sector_names
from stockanalysis import analyze_stocks
from select_filter import get_fo_stocks
import logging
from analysestockgrowth import analyze_stock_growth, change_from_previous_close_percentage
from sip_calculator import sip_calculator
from swp_calculator import swp_calculator
from waitress import serve
from flask import Flask, jsonify, request
import logging
import time
from select_filter import get_fo_stocks
from turnover_analysis import analyze_high_turnover_stocks_live  # adjust import if needed
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
kite = get_kite_client()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler('app_error.log'), logging.StreamHandler()]
)
#DUMY ROUTES TO CHECK
@app.route('/top-and-tip-growth', methods=['GET'])
def top_and_low_growth():
    return jsonify({"status": "success", "message": "API working"})
    
@app.route('/')
def home():
    return "Backend is Live"

@app.route('/api/pcr', methods=['GET'])
def get_pcr():
    index = request.args.get('index')
    if not index:
        return jsonify({
            "success": False,
            "message": "Missing required query parameter: index"
        }), 400

    result = calculate_pcr(index.upper())
    if "error" in result:
        return jsonify({
            "success": False,
            "message": result["error"]
        }), 500

    return jsonify({
        "success": True,
        "message": f"PCR calculated for {index.upper()}",
        "data": result
    })

@app.route('/api/oi', methods=['GET'])
def get_oi_data():
    try:
        result = fetch_oi_data()
        return jsonify({
            "success": True,
            "message": "OI data fetched successfully",
            "data": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error fetching OI data: {str(e)}"
        }), 500

@app.route('/analyze', methods=['GET'])
def analyze_route():
    # Capture stdout
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer

    try:
        analyze_indices(kite, indices, index_constituents)
        output = buffer.getvalue()
    except Exception as e:
        output = f" Error running analysis: {e}"
    finally:
        sys.stdout = sys_stdout
        buffer.close()

    # Always return as JSON with a clear key
    return jsonify({"output": output})
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_call(func, *args, retries=3, delay=2, **kwargs):
    """
    Retry wrapper for functions that call external APIs (like Kite).
    """
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"❌ Error in {func.__name__} (attempt {attempt+1}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError(f"{func.__name__} failed after {retries} retries")
@app.route('/high-turnover-stocks', methods=['GET'])
def high_turnover_stocks():
    try:
        # Get the optional top_n parameter from query string, default to 100
        top_n = int(request.args.get('top', 100))

        # Safe calls with retries
        fo_stock_symbols = safe_call(get_fo_stocks)  
        result = safe_call(analyze_high_turnover_stocks_live, top_n)

        # Filter stocks into FO and non-FO based on symbol membership
        fo_stocks_result = [stock for stock in result if stock['symbol'] in fo_stock_symbols]
        non_fo_stocks_result = [stock for stock in result if stock['symbol'] not in fo_stock_symbols]

        return jsonify({
            "fo_stocks": fo_stocks_result,
            "non_fo_stocks": non_fo_stocks_result
        }), 200

    except Exception as e:
        logger.exception("❌ Fatal error in /high-turnover-stocks route")
        return jsonify({
            "error": "Failed to fetch high turnover stocks",
            "details": str(e)
        }), 502
from flask import jsonify

@app.route('/channel-breakout', methods=['GET'])
def channel_breakout():
    try:
        default_timeframes = {
            '5minute': {'lookback_days': 5, 'ma_length': 20, 'atr_period': 14},
            '15minute': {'lookback_days': 7, 'ma_length': 20, 'atr_period': 14},
            '30minute': {'lookback_days': 10, 'ma_length': 20, 'atr_period': 14}
        }

        result = find_breakouts(default_timeframes)

        # Filter out invalid entries (non-dict)
        filtered_result = [r for r in result if isinstance(r, dict)]
        sanitized_result = json.loads(json.dumps(result, default=str))
        return jsonify(sanitized_result), 200
    except Exception as e:
        print("❌ ERROR RETURNING BREAKOUT:", str(e))
        return jsonify({"error": str(e)}), 500



@app.route('/breakouts', methods=['POST'])
def breakout_api():
    try:
        # Accept empty or no JSON body
        data = request.get_json(silent=True)
        if data and 'stock_list' in data:
            if isinstance(data['stock_list'], str):
                stock_list = [s.strip().upper() for s in data['stock_list'].split(',')]
            else:
                stock_list = [s.upper() for s in data['stock_list']]
        else:
            stock_list = fo_stocks + non_fo_stocks

        """if data and 'stock_list' in data:
            stock_list = data['stock_list']
            print("check1",stock_list)

        else:
            stock_list = fo_stocks + non_fo_stocks
            print("check2",stock_list)"""
        breakout_df = get_breakouts(stock_list)
        #print("check",breakout_df)

        if breakout_df is not None and not breakout_df.empty:
            # Convert all `datetime.time` objects to string
            for col in breakout_df.columns:
                if breakout_df[col].apply(lambda x: isinstance(x, time)).any():
                    breakout_df[col] = breakout_df[col].astype(str)

            return jsonify(breakout_df.to_dict(orient='records')), 200
        else:
            return jsonify({"message": "No breakouts found"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/breakout-tenor-fifty', methods=['POST'])
def breakout_route():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = {}

        breakout_type = data.get("breakout_type")  # "10d" or "50d"
        limit = data.get("limit")
        if limit is None:
            limit = 60  # Default value, adjust as needed
        else:
            limit = int(limit)
        throttle = data.get("throttle", True)

        results = check_breakout(
            breakout_type=breakout_type,
            limit=limit,
            throttle=throttle
        )

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/breakout-three-percent', methods=['POST'])
def breakout_three_percent():
    try:
        # 🔸 Extract category from request JSON
        if request.is_json:
            data = request.get_json()
        else:
            data = {}
        category = data.get("category", "fo").lower()

        if category in ["fo", "fno", "all"]:
            # 🧠 Run optimized logic from rankfostocks.py
            result_df = rank_fo_stocks()

            if result_df.empty:
                return jsonify({
                    "total_valid_stocks": 0,
                    "top_performers": [],
                    "low_performers": [],
                    "error": "No valid results."
                }), 200

            top_n = 20
            low_n = 20

            top_performers = result_df.sort_values("rank_near_high")[[ 
                "rank_near_high", "stock", "ltp", "percentage_change", "proximity_to_high"
            ]].head(top_n).to_dict(orient="records")

            low_performers = result_df.sort_values("rank_near_low")[[ 
                "rank_near_low", "stock", "ltp", "percentage_change", "proximity_to_low"
            ]].head(low_n).to_dict(orient="records")

            output = {
                "total_valid_stocks": int(len(result_df)),
                "top_performers": top_performers,
                "low_performers": low_performers
            }

            return jsonify(output), 200

        else:
            return jsonify({
                "error": f"Invalid category '{category}'. Use 'fo', 'fno', or 'all'."
            }), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rank-fo-stocks', methods=['GET'])
def get_ranked_fo_stocks():
    try:
        df = rank_fo_stocks_by_growth()
        if df.empty:
            return jsonify({"success": False, "message": "No valid data found"}), 404

        # Convert to list of dictionaries for JSON response
        result = df.sort_values("z_score", ascending=False).head(10).to_dict(orient='records')
        return jsonify({"success": True, "data": result}), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/scan-momentum', methods=['POST'])
def scan_momentum():
    try:
        data = request.get_json()
        # Note: scan_fo_stocks_dual only takes kite as argument.
        fno_results, fno_neg, non_fno_results, non_fno_neg = scan_fo_stocks_dual(kite)
        return jsonify({
            "status": "success",
            "fno_momentum_gainers": fno_results,
            "fno_momentum_losers": fno_neg,
            "non_fno_momentum_gainers": non_fno_results,
            "non_fno_momentum_losers": non_fno_neg
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/top-gainers-losers', methods=['GET'])
def top_gainers_losers():
    try:
        result = get_top_gainers_and_losers()
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/sector-abnormal-growth', methods=['GET'])
def sector_abnormal_growth():
    sector_name = request.args.get('sector')
    print("sector_name",)

    if not sector_name:
        return jsonify({"error": "Please provide a sector name in the query params, e.g. ?sector=Auto"}), 400

    if sector_name not in get_all_sector_names():
        return jsonify({"error": f"Invalid sector name: '{sector_name}'. Use one of: {get_all_sector_names()}"}), 400

    result = get_sector_abnormal_growth(kite, sector_name)
    return jsonify(result), 200

# 1. Analyze custom stocks (via POST request)
@app.route('/stockanalyze', methods=['POST'])
def stock_analyze_route():
    try:
        data = request.get_json()
        if not data or "stocks" not in data:
            return jsonify({"error": "Missing 'stocks' in request body"}), 400

        interval = data.get("interval", "5minute")
        days = data.get("days", 2)
        max_workers = data.get("max_workers", 10)

        stocks = data["stocks"]
        if not isinstance(stocks, list):
            return jsonify({"error": "'stocks' must be a list of stock objects"}), 400

        results = analyze_stocks(stocks, interval=interval, days=days, max_workers=max_workers)
        return jsonify(results)

    except Exception as e:
        logging.error(f"Error in /analyze route: {e}")
        return jsonify({"error": str(e)}), 500

# 2. Analyze F&O stocks
@app.route('/stockanalyze/fno', methods=['GET'])
def analyze_fno():
    try:
        fno = fo_stocks()
        results = analyze_stocks(fno)
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error in /analyze/fno: {e}")
        return jsonify({"error": str(e)}), 500

# 3. Analyze Non-F&O stocks
@app.route('/stockanalyze/nonfno', methods=['GET'])
def analyze_nonfno():
    try:
        nonfno = non_fo_stocks()
        results = analyze_stocks(nonfno)
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error in /analyze/nonfno: {e}")
        return jsonify({"error": str(e)}), 500

# 4. Analyze All NSE stocks
@app.route('/stockanalyze/all', methods=['GET'])
def analyze_all():
    try:
        all_nse = fo_stocks()
        results = analyze_stocks(all_nse)
        return jsonify(results)
    except Exception as e:
        logging.error(f"Error in /analyze/all: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/growth/analyze', methods=['POST'])
def growth_analyze():
    try:
        data = request.get_json()
        symbol = data.get("symbol")
        if not symbol:
            return jsonify({"error": "Missing 'symbol' in request body"}), 400

        recent_days = data.get("recent_days", 7)
        long_term_days = data.get("long_term_days", 365)
        threshold = data.get("threshold", 2)
        exchange = data.get("exchange", "NSE")

        result = analyze_stock_growth(
            trading_symbol=symbol,
            recent_days=recent_days,
            long_term_days=long_term_days,
            threshold=threshold,
            exchange=exchange
        )
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in /growth/analyze: {e}")
        return jsonify({"error": str(e)}), 500

# Rank multiple stocks by growth abnormality (z-score)
@app.route('/growth/rank', methods=['POST'])
def growth_rank():
    try:
        data = request.get_json()
        symbols = data.get("symbols")
        if not symbols or not isinstance(symbols, list):
            return jsonify({"error": "Missing or invalid 'symbols' (should be a list)"}), 400

        recent_days = data.get("recent_days", 7)
        long_term_days = data.get("long_term_days", 365)
        threshold = data.get("threshold", 2)
        exchange = data.get("exchange", "NSE")

        df = change_from_previous_close_percentage(
            symbols=symbols,
            recent_days=recent_days,
            long_term_days=long_term_days,
            threshold=threshold,
            exchange=exchange
        )

        return df.to_json(orient="records")
    except Exception as e:
        logging.error(f"Error in /growth/rank: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sip', methods=['POST'])
def calculate_sip():
    try:
        data = request.get_json()
        amount = data.get("amount")
        yearly_rate = data.get("yearly_rate")
        years = data.get("years")

        # Validate inputs
        if amount is None or yearly_rate is None or years is None:
            return jsonify({"error": "Missing 'amount', 'yearly_rate', or 'years' in request body"}), 400

        future_value = sip_calculator(amount, yearly_rate, years)

        return jsonify({
            "monthly_investment": amount,
            "yearly_rate (%)": yearly_rate,
            "investment_duration (years)": years,
            "future_value": future_value
        })
    except Exception as e:
        logging.error(f"Error in /sip: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/swp', methods=['POST'])
def calculate_swp():
    try:
        data = request.get_json()
        initial_investment = data.get("initial_investment")
        withdrawal_amount = data.get("withdrawal_amount")
        annual_return = data.get("annual_return")
        months = data.get("months")

        # Validate required inputs
        if None in [initial_investment, withdrawal_amount, annual_return, months]:
            return jsonify({"error": "Missing one or more required fields: 'initial_investment', 'withdrawal_amount', 'annual_return', 'months'"}), 400

        result = swp_calculator(
            initial_investment=initial_investment,
            withdrawal_amount=withdrawal_amount,
            annual_return=annual_return,
            months=months
        )

        return jsonify({"swp_schedule": result})

    except Exception as e:
        logging.error(f"Error in /swp: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/top-and-low-growth', methods=['GET'])
def get_top_and_low_growth():
    try:
        logger.info(" Running F&O stock growth ranking...")
        df: pd.DataFrame = rank_fo_stocks_by_growth()

        if df.empty:
            return jsonify({"message": "No data available"}), 404

        # Convert DataFrame to JSON
        results = df.sort_values("z_score", ascending=False).head(10).to_dict(orient="records")
        return jsonify(results), 200

    except Exception as e:
        logger.error(f" Error in /top-and-low-growth route: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Internal Server Error: {error}")
    return jsonify({"error": "Internal server error occurred."}), 500

if __name__ == "__main__":
    # For production, use a WSGI server like 'waitress' or 'gunicorn'.
    app.run(debug=False, port=5000)


