import requests
import pandas as pd
from io import StringIO
import time
import os
from datetime import datetime, timedelta
from config import API_KEY, BASE_URL, CACHE_DIR, CACHE_EXPIRY_DAYS

def get_cache_path(symbol, interval):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"{symbol}_{interval}.parquet")

def is_cache_valid(cache_path):
    if not os.path.exists(cache_path):
        return False
    
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    expiry_time = datetime.now() - timedelta(days=CACHE_EXPIRY_DAYS)
    return cache_time > expiry_time

def download_stock_data(symbol, interval="1min"):
    cache_path = get_cache_path(symbol, interval)
    
    # Check cache first
    if is_cache_valid(cache_path):
        print(f"Loading cached data for {symbol}")
        df = pd.read_parquet(cache_path)
        return df

    # If not in cache, download
    print(f"Downloading fresh data for {symbol}")
    url = f"{BASE_URL}?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={API_KEY}&datatype=csv&extended_hours=false&outputsize=full"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if df.empty:
                raise ValueError(f"Empty data received for {symbol}")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Save to cache
            df.to_parquet(cache_path, index=False)
            return df
        elif response.status_code == 429:
            print(f"Rate limit exceeded for {symbol}. Waiting 60 seconds...")
            time.sleep(60)
            return download_stock_data(symbol, interval)
        else:
            raise ValueError(f"Error fetching data for {symbol}: {response.status_code}")
    except Exception as e:
        print(f"Error downloading data for {symbol}: {str(e)}")
        return None

def get_stock_data(symbols):
    stock_data = {}
    all_timestamps = set()
    
    for symbol in symbols:
        print(f"Downloading data for {symbol}...")
        df = download_stock_data(symbol)
        if df is not None:
            stock_data[symbol] = df
            all_timestamps.update(df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
    
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