import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from datetime import datetime, timedelta
import time

# Import Kite and F&O stocks from select_filter
try:
    from select_filter import kite, fo_stocks
except ImportError:
    print("Error: Could not import kite or fo_stocks from select_filter\n"
          "Please ensure select_filter.py is in the same directory and properly set up.")
    raise

# Global variable for the current dataframe
df = None

# Core analysis functions
def isPivot(candle, window):
    global df
    if isinstance(candle, (str, pd.Timestamp)) and candle in df.index:
        candle = df.index.get_loc(candle)
    if not isinstance(candle, int):
        return 0
    if candle - window < 0 or candle + window >= len(df):
        return 0
    pivotHigh = 1
    pivotLow = 2
    current_candle = df.iloc[candle]
    for i in range(candle - window, candle + window + 1):
        if i == candle:
            continue
        window_candle = df.iloc[i]
        if current_candle['Low'] > window_candle['Low']:
            pivotLow = 0
        if current_candle['High'] < window_candle['High']:
            pivotHigh = 0
        if not (pivotHigh or pivotLow):
            return 0
    if pivotHigh and pivotLow:
        return 3
    elif pivotHigh:
        return 1
    elif pivotLow:
        return pivotLow
    else:
        return 0

def pointpos(x):
    if x['isPivot'] == 2:
        return x['Low'] - 1e-3
    elif x['isPivot'] == 1:
        return x['High'] + 1e-3
    else:
        return np.nan

def collect_channel(candle, backcandles, window):
    global df
    localdf = df[candle - backcandles - window:candle - window].copy()
    localdf['isPivot'] = localdf.apply(lambda x: isPivot(x.name, window), axis=1)
    highs = localdf[localdf['isPivot'] == 1].High.values
    idxhighs = localdf[localdf['isPivot'] == 1].index.map(lambda x: df.index.get_loc(x))
    lows = localdf[localdf['isPivot'] == 2].Low.values
    idxlows = localdf[localdf['isPivot'] == 2].index.map(lambda x: df.index.get_loc(x))

    if len(lows) >= 2 and len(highs) >= 2:
        sl_lows, interc_lows, r_value_l, _, _ = stats.linregress(idxlows, lows)
        sl_highs, interc_highs, r_value_h, _, _ = stats.linregress(idxhighs, highs)
        return (sl_lows, interc_lows, sl_highs, interc_highs, r_value_l**2, r_value_h**2)
    else:
        return (0, 0, 0, 0, 0, 0)

def isBreakOut(candle, backcandles, window):
    global df
    if (candle - backcandles - window) < 0:
        return 0
    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(candle, backcandles, window)
    prev_idx = candle - 1
    prev_high = df.iloc[prev_idx].High
    prev_low = df.iloc[prev_idx].Low
    prev_close = df.iloc[prev_idx].Close
    curr_idx = candle
    curr_high = df.iloc[curr_idx].High
    curr_low = df.iloc[curr_idx].Low
    curr_close = df.iloc[curr_idx].Close
    curr_open = df.iloc[curr_idx].Open

    if (prev_high > (sl_lows * prev_idx + interc_lows) and
        prev_close < (sl_lows * prev_idx + interc_lows) and
        curr_open < (sl_lows * curr_idx + interc_lows) and
        curr_close < (sl_lows * prev_idx + interc_lows)):
        return 1
    elif (prev_low < (sl_highs * prev_idx + interc_highs) and
          prev_close > (sl_highs * prev_idx + interc_highs) and
          curr_open > (sl_highs * curr_idx + interc_highs) and
          curr_close > (sl_highs * prev_idx + interc_highs)):
        return 2
    else:
        return 0

def breakpointpos(x):
    if x['isBreakOut'] == 2:
        return x['Low'] - 3e-3
    elif x['isBreakOut'] == 1:
        return x['High'] + 3e-3
    else:
        return np.nan

# Load NSE instruments and create a mapping from symbol to instrument data
print("Fetching instrument list from NSE...")
try:
    nse_instruments = kite.instruments("NSE")
    instrument_map = {i['tradingsymbol']: i for i in nse_instruments}
except Exception as e:
    print(f"Error fetching instruments: {e}")
    exit(1)

if not fo_stocks:
    print("No F&O stocks found in select_filter.py")
    exit(1)

def get_historical_data(instrument_token, days=365, interval="day"):
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval,
            continuous=False,
            oi=False
        )
        if not data:
            print(f"No data returned for instrument {instrument_token}")
            return None
        df = pd.DataFrame(data)
        if df.empty:
            print(f"Empty dataframe for instrument {instrument_token}")
            return None
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for instrument {instrument_token}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_stock(symbol, plot_results=True):
    """Analyze a single stock for channel breakouts"""
    global df
    print(f"\nAnalyzing {symbol}...")
    
    # Get instrument data
    instrument_data = instrument_map.get(symbol)
    if not instrument_data:
        print(f"  Instrument data not found for {symbol}")
        return None
    
    # Fetch historical data
    df = get_historical_data(instrument_data['instrument_token'])
    if df is None or df.empty:
        print(f"  Failed to fetch data for {symbol}")
        return None
    
    print(f"  Successfully fetched {len(df)} data points")
    
    # Add pivot points and point positions
    window = 3
    backcandles = 40
    
    # Calculate isPivot and pointpos for the entire dataframe first
    df['isPivot'] = df.apply(lambda x: isPivot(x.name, window), axis=1)
    df['pointpos'] = df.apply(pointpos, axis=1)
    
    # Find breakouts in the last 20 candles
    lookback = min(20, len(df) - backcandles - window - 1)
    
    if lookback <= 0:
        print(f"  Not enough data points for analysis")
        return None
    
    # Create a copy of the relevant data for plotting
    dfpl = df.iloc[-lookback:].copy()
    
    # Calculate breakouts for each candle in the lookback period
    breakout_signals = []
    for i in range(len(dfpl)):
        candle_idx = df.index.get_loc(dfpl.index[i])
        breakout_signals.append(isBreakOut(candle_idx, backcandles, window))
    
    # Add the breakout signals to the dataframe
    dfpl['isBreakOut'] = breakout_signals
    
    # Add point positions for plotting
    dfpl['pointpos'] = dfpl.apply(pointpos, axis=1)
    dfpl['breakpointpos'] = dfpl.apply(breakpointpos, axis=1)
    
    # Ensure the main dataframe has the same columns for plotting
    if 'isBreakOut' not in df.columns:
        df['isBreakOut'] = 0
        df.loc[dfpl.index, 'isBreakOut'] = dfpl['isBreakOut']
    
    # Get the most recent breakout if any
    recent_breakouts = dfpl[dfpl['isBreakOut'] > 0]
    
    if not recent_breakouts.empty:
        last_breakout = recent_breakouts.iloc[-1]
        signal_type = "BUY" if last_breakout['isBreakOut'] == 1 else "SELL"
        # Get the date as a string in the desired format
        date_str = str(last_breakout.name.date()) if hasattr(last_breakout.name, 'date') else str(last_breakout.name)
        print(f"  {signal_type} signal detected on {date_str}")
        
        if plot_results:
            plot_breakout(df, dfpl, backcandles, window, symbol)
            
        return {
            'symbol': symbol,
            'date': date_str,
            'signal': signal_type,
            'price': last_breakout['Close']
        }
    else:
        print(f"  No breakout signals found")
        return None

def plot_initial_chart(df):
    """Plot initial chart with pivot points"""
    window = 3
    df['isPivot'] = df.apply(lambda x: isPivot(x.name, window), axis=1)
    df['pointpos'] = df.apply(pointpos, axis=1)
    
    dfpl = df[0:100].copy()
    fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                    open=dfpl['Open'],
                    high=dfpl['High'],
                    low=dfpl['Low'],
                    close=dfpl['Close'])])

    fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                    marker=dict(size=5, color="MediumPurple"),
                    name="pivot")
    fig.show()
    return df

def plot_breakout(df, dfpl, backcandles, window, symbol):
    """Plot the breakout chart for a stock"""
    # Get the last candle with a breakout
    last_breakout_idx = dfpl[dfpl['isBreakOut'] > 0].index[-1]
    candle = df.index.get_loc(last_breakout_idx)
    
    # Prepare data for plotting
    plot_start = max(0, candle - backcandles - window - 20)
    plot_end = min(len(df), candle + 20)
    plot_df = df.iloc[plot_start:plot_end].copy()
    
    # Create figure
    fig = go.Figure(data=[
        go.Candlestick(
            x=plot_df.index,
            open=plot_df['Open'],
            high=plot_df['High'],
            low=plot_df['Low'],
            close=plot_df['Close'],
            name='OHLC'
        )
    ])
    
    # Add pivot points
    pivot_points = plot_df[plot_df['pointpos'].notna()]
    if not pivot_points.empty:
        fig.add_scatter(
            x=pivot_points.index,
            y=pivot_points['pointpos'],
            mode='markers',
            marker=dict(size=5, color="MediumPurple"),
            name="Pivot"
        )
    
    # Add breakout markers
    breakouts = plot_df[plot_df['isBreakOut'] > 0]
    if not breakouts.empty:
        fig.add_scatter(
            x=breakouts.index,
            y=breakouts.apply(
                lambda x: x['Low'] - 3e-3 if x['isBreakOut'] == 2 else x['High'] + 3e-3,
                axis=1
            ),
            mode='markers',
            marker=dict(size=8, color="Black", symbol="hexagram"),
            name="Breakout"
        )
    
    # Add channel lines
    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(
        candle, backcandles, window
    )
    
    if r_sq_l > 0.3 and r_sq_h > 0.3:  # Only plot if channels are valid
        x = np.array(range(candle - backcandles - window, candle + 1))
        fig.add_trace(go.Scatter(
            x=df.index[x],
            y=sl_lows * x + interc_lows,
            mode='lines',
            name='Support',
            line=dict(color='green', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index[x],
            y=sl_highs * x + interc_highs,
            mode='lines',
            name='Resistance',
            line=dict(color='red', width=1)
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Channel Breakout",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    fig.show()

def isPivot(candle, window):
    global df
    if isinstance(candle, (str, pd.Timestamp)) and candle in df.index:
        candle = df.index.get_loc(candle)
    if not isinstance(candle, int):
        return 0
    if candle - window < 0 or candle + window >= len(df):
        return 0
    pivotHigh = 1
    pivotLow = 2
    current_candle = df.iloc[candle]
    for i in range(candle - window, candle + window + 1):
        if i == candle:
            continue
        window_candle = df.iloc[i]
        if current_candle['Low'] > window_candle['Low']:
            pivotLow = 0
        if current_candle['High'] < window_candle['High']:
            pivotHigh = 0
        if not (pivotHigh or pivotLow):
            return 0
    if pivotHigh and pivotLow:
        return 3
    elif pivotHigh:
        return 1
    elif pivotLow:
        return pivotLow
    else:
        return 0

def collect_channel(candle, backcandles, window):
    localdf = df[candle - backcandles - window:candle - window].copy()
    localdf['isPivot'] = localdf.apply(lambda x: isPivot(x.name, window), axis=1)
    highs = localdf[localdf['isPivot'] == 1].High.values
    idxhighs = localdf[localdf['isPivot'] == 1].index.map(lambda x: df.index.get_loc(x))
    lows = localdf[localdf['isPivot'] == 2].Low.values
    idxlows = localdf[localdf['isPivot'] == 2].index.map(lambda x: df.index.get_loc(x))

    if len(lows) >= 2 and len(highs) >= 2:
        sl_lows, interc_lows, r_value_l, _, _ = stats.linregress(idxlows, lows)
        sl_highs, interc_highs, r_value_h, _, _ = stats.linregress(idxhighs, highs)
        return (sl_lows, interc_lows, sl_highs, interc_highs, r_value_l**2, r_value_h**2)
    else:
        return (0, 0, 0, 0, 0, 0)

def isBreakOut(candle, backcandles, window):
    if (candle - backcandles - window) < 0:
        return 0
    sl_lows, interc_lows, sl_highs, interc_highs, r_sq_l, r_sq_h = collect_channel(candle, backcandles, window)
    prev_idx = candle - 1
    prev_high = df.iloc[prev_idx].High
    prev_low = df.iloc[prev_idx].Low
    prev_close = df.iloc[prev_idx].Close
    curr_idx = candle
    curr_high = df.iloc[curr_idx].High
    curr_low = df.iloc[curr_idx].Low
    curr_close = df.iloc[curr_idx].Close
    curr_open = df.iloc[curr_idx].Open

    if (prev_high > (sl_lows * prev_idx + interc_lows) and
        prev_close < (sl_lows * prev_idx + interc_lows) and
        curr_open < (sl_lows * curr_idx + interc_lows) and
        curr_close < (sl_lows * prev_idx + interc_lows)):
        return 1
    elif (prev_low < (sl_highs * prev_idx + interc_highs) and
          prev_close > (sl_highs * prev_idx + interc_highs) and
          curr_open > (sl_highs * curr_idx + interc_highs) and
          curr_close > (sl_highs * prev_idx + interc_highs)):
        return 2
    else:
        return 0

def breakpointpos(x):
    if x['isBreakOut'] == 2:
        return x['Low'] - 3e-3
    elif x['isBreakOut'] == 1:
        return x['High'] + 3e-3
    else:
        return np.nan

# Main execution
if __name__ == "__main__":
    print(f"\nAnalyzing {len(fo_stocks)} F&O stocks...")
    results = []
    
    for symbol in fo_stocks:  # Process all F&O stocks
        try:
            result = analyze_stock(symbol, plot_results=True)
            if result:
                results.append(result)
                print(f"  Found breakout in {symbol}")
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
    
    if not results:
        print("\nNo breakout signals found in any stocks.")
    else:
        print(f"\nFound breakout signals in {len(results)} stocks.")
        for r in results:
            print(f"  {r['symbol']}: {r['signal']} on {r['date']}")

    # Keep the plot windows open
    input("\nPress Enter to exit...")
