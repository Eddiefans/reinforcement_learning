import os
import yfinance as yf
import pandas as pd
import ta

def download_raw_data(ticker, start_date, end_date, output_path=None, verbose=True):
    """
    Download raw historical price data from Yahoo Finance.
    
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_path: Path to save the raw data CSV file
            verbose: Whether to print progress messages 
            
        Returns:
            pandas.DataFrame: Raw historical price data 
    """
    if verbose:
        print("Downloading raw data for {} from {} to {}...".format(ticker, start_date, end_date))
    
    # Download data from Yahoo Finance instance
    raw = yf.download(ticker, start=start_date, end=end_date, threads=False)
    
    # Downlaod it there is an output path
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        raw.to_csv(output_path)
        print(f"Raw data saved to {output_path}")
    
    return raw

def create_features_dataset(
    ticker,
    lags=5,
    start_date="2000-01-01",
    end_date="2023-12-31",
    output_path=None,
    raw_data_path=None,
    sma_window=20,
    ema_window=20,
    rsi_window=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bb_window=20,
    bb_std=2,
    atr_window=14,
    verbose=True,
):
    """
    Create a dataset with technical indicators for trading.
    
    Args:
        ticker: Stock ticker symbol
        lags: Number of lagged returns to include
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Path to save the processed dataset
        raw_data_path: Path to load raw data from (if None, download fresh)
        sma_window: Simple Moving Average window
        ema_window: Exponential Moving Average window
        rsi_window: Relative Strength Index window
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal period
        bb_window: Bollinger Bands window
        bb_std: Bollinger Bands standard deviation
        atr_window: Average True Range window
        verbose: Whether to print progress messages
    
    Returns:
        pandas.DataFrame: Processed dataset with technical indicators
    """
    
    # Either load raw data from file or download it
    if raw_data_path and os.path.exists(raw_data_path):
        if verbose:
            print("Loading raw data from {}".format(raw_data_path))
        raw_price_data = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
        
        # Ensure the OHLCV columns are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in raw_price_data.columns:
                raw_price_data[col] = pd.to_numeric(raw_price_data[col], errors='coerce')
        
        # Download VIX separately
        vix_data = yf.download("^VIX", start=start_date, end=end_date, threads=False)
        vix = vix_data["Close"].copy()
        vix.name = "VIX"
    else:
        # Download both ticker and VIX in one call
        if verbose:
            print("Downloading data for {} and VIX...".format(ticker))
        raw = yf.download([ticker, "^VIX"], start=start_date, end=end_date, threads=False)
        
        # Take the OHLCV
        raw_price_data = raw.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]]
        
        # Take only the close price for VIX
        vix = vix_data["Close"].copy()
        vix.name = "VIX"

    # Create working frame with only the Close
    df = pd.DataFrame(index=raw_price_data.index)
    df.index.name = "Date"
    df["Close"] = raw_price_data["Close"]
    df = df.dropna()
    
    # Ensure Close is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')

    # Calculate returns and add lagged returns
    rends = df["Close"].pct_change().dropna()
    for lag in range(1, lags + 1):
        df["rend_lag_{}".format(lag)] = rends.shift(lag)

    # Compute technical indicators
    if verbose:
        print("Computing technical indicators...")
    df["SMA"] = ta.trend.sma_indicator(raw_price_data["Close"], window=sma_window)
    df["EMA"] = ta.trend.ema_indicator(raw_price_data["Close"], window=ema_window)
    df["RSI"] = ta.momentum.rsi(raw_price_data["Close"], window=rsi_window)
    
    macd = ta.trend.MACD(
        raw_price_data["Close"],
        window_slow=macd_slow,
        window_fast=macd_fast,
        window_sign=macd_signal,
    )
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    
    bb = ta.volatility.BollingerBands(
        raw_price_data["Close"], window=bb_window, window_dev=bb_std
    )
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    atr = ta.volatility.AverageTrueRange(
        high=raw_price_data["High"],
        low=raw_price_data["Low"],
        close=raw_price_data["Close"],
        window=atr_window,
    )
    df["ATR"] = atr.average_true_range()

    df["OBV"] = ta.volume.on_balance_volume(
        raw_price_data["Close"], raw_price_data["Volume"]
    )
    df["VWAP"] = ta.volume.volume_weighted_average_price(
        raw_price_data["High"],
        raw_price_data["Low"],
        raw_price_data["Close"],
        raw_price_data["Volume"],
    )
    
    # Add VIX
    df["VIX"] = vix
    
    # Normalized ratios
    ratio_cols = [
        "SMA", "EMA", "RSI", "MACD", "MACD_hist",
        "BB_upper", "BB_lower", "ATR", "OBV", "VWAP",
    ]
    for col in ratio_cols:
        df["close_{}_ratio".format(col.lower())] = df["Close"] / df[col]
    
    # Drop any NaNs
    df.dropna(inplace=True)
    
    # Reset index to make Date a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Save processed data if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print("Processed features saved to {}".format(output_path))
    
    return df

if __name__ == "__main__":
    # Usage for amazon (AMZN) stock data
    ticker = 'AMZN'
    start_date = '2000-01-01'
    end_date = '2023-12-31'
    
    # Create directory structure if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Define paths
    raw_path = "data/raw/{}_raw_{}_{}.csv".format(ticker, start_date, end_date)
    processed_path = "data/processed/{}_features.csv".format(ticker)
    
    # Download raw data
    raw_data = download_raw_data(ticker, start_date, end_date, raw_path)
    
    # Process data and create features
    features_df = create_features_dataset(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        output_path=processed_path,
        raw_data_path=raw_path
    )
    
    print("Created dataset with {} rows and {} features".format(len(features_df), len(features_df.columns)))