import requests
import pandas as pd
from io import StringIO
import time
import os
from datetime import datetime, timedelta
from config import API_KEY, BASE_URL, CACHE_DIR, CACHE_EXPIRY_DAYS

def get_cache_path(symbol, interval, month=None):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    cache_file = f"{symbol}_{interval}.parquet" if month is None else f"{symbol}_{interval}_{month}.parquet"
    return os.path.join(CACHE_DIR, cache_file)

def is_cache_valid(cache_path):
    if not os.path.exists(cache_path):
        return False
    
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    expiry_time = datetime.now() - timedelta(days=CACHE_EXPIRY_DAYS)
    return cache_time > expiry_time

def download_stock_data(symbol, interval="1min", month=None):
    cache_path = get_cache_path(symbol, interval, month)
    
    # Check cache first
    if is_cache_valid(cache_path):
        print(f"Loading cached data for {symbol} {'(current)' if month is None else f'({month})'}")
        return pd.read_parquet(cache_path)

    # If not in cache, download
    print(f"Downloading fresh data for {symbol} {'(current)' if month is None else f'({month})'}")
    
    # Build URL based on whether we're getting current or future data
    base_url = f"{BASE_URL}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={API_KEY}&datatype=csv&extended_hours=false&outputsize=full"
    url = f"{base_url}&month={month}" if month else base_url
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if df.empty:
                raise ValueError(f"Empty data received for {symbol}")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.to_parquet(cache_path, index=False)
            return df
        elif response.status_code == 429:
            print(f"Rate limit exceeded for {symbol}. Waiting 60 seconds...")
            time.sleep(60)
            return download_stock_data(symbol, interval, month)
        else:
            raise ValueError(f"Error fetching data for {symbol}: {response.status_code}")
    except Exception as e:
        print(f"Error downloading data for {symbol}: {str(e)}")
        return None

def get_stock_data(symbols):
    stock_data = {}
    all_timestamps = set()
    
    for symbol in symbols:
        # Download current data
        current_df = download_stock_data(symbol)
        
        # Download data for the months April 2024 to June 2024
        future_dfs = []
        for month in ["2022-11","2022-12","2023-09","2023-10", "2023-11", "2023-12", "2024-06", "2024-07"]:
            future_df = download_stock_data(symbol, month=month)
            if future_df is not None:
                future_dfs.append(future_df)
        
        # Combine the dataframes if both current and future data are available
        if current_df is not None and future_dfs:
            df = pd.concat([current_df] + future_dfs, ignore_index=True)
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            stock_data[symbol] = df
            all_timestamps.update(df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Rest of the function remains the same
    all_timestamps = sorted(list(all_timestamps))
    
    for symbol in stock_data:
        full_idx = pd.DatetimeIndex(all_timestamps)
        stock_data[symbol] = (stock_data[symbol]
            .set_index('timestamp')
            .reindex(full_idx)
            .ffill()
            .bfill()
            .reset_index()
            .rename(columns={'index': 'timestamp'})
        )
    
    return stock_data

def split_data(stock_data):
    train, val, test = {}, {}, {}
    for symbol, df in stock_data.items():
        df = df.sort_values('timestamp')
        df['date'] = df['timestamp'].dt.date
        grouped = df.groupby('date')
        days = list(grouped.groups.keys())

        train_end = int(len(days) * 0.6)
        val_end = int(len(days) * 0.8)

        train[symbol] = df[df['date'].isin(days[:train_end])]
        val[symbol] = df[df['date'].isin(days[train_end:val_end])]
        test[symbol] = df[df['date'].isin(days[val_end:])]

    return train, val, test