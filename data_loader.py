import requests
import pandas as pd
import polars as pl
import numpy as np
from io import StringIO
import time
import os
from datetime import datetime, timedelta
from config import API_KEY, BASE_URL, MODEL_PARAMS, DATA_PARAMS
from tqdm import trange

def get_cache_path(symbol, interval, month=None):
    if not os.path.exists(DATA_PARAMS['CACHE_DIR']):
        os.makedirs(DATA_PARAMS['CACHE_DIR'])
    cache_file = f"{symbol}_{interval}.parquet" if month is None else f"{symbol}_{interval}_{month}.parquet"
    return os.path.join(DATA_PARAMS['CACHE_DIR'], cache_file)

def get_combined_cache_path(symbol):
    return get_cache_path(symbol, "combined")

def is_cache_valid(cache_path):
    if not os.path.exists(cache_path):
        return False
    
    cache_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
    expiry_time = datetime.now() - timedelta(days=DATA_PARAMS['CACHE_EXPIRY_DAYS'])
    return cache_time > expiry_time

def filter_and_fill(df):
    # Extract date from timestamp
    df['date'] = df['timestamp'].dt.date  

    # Group by date and validate times
    valid_dates = []
    for date, group in df.groupby('date'):
        start_time = group['timestamp'].min().time()
        end_time = group['timestamp'].max().time()
    
        # Check if the day starts at 09:30 and ends at 15:59
        if start_time == pd.Timestamp("09:30").time() and end_time == pd.Timestamp("15:59").time():
            valid_dates.append(date)

        # Filter for valid dates
    df = df[df['date'].isin(valid_dates)]

    # Generate a complete list of timestamps for each date
    complete_df = []
    for date in df['date'].unique():
        day_data = df[df['date'] == date]
        start = pd.Timestamp(f"{date} 09:30")
        end = pd.Timestamp(f"{date} 15:59")
        complete_timestamps = pd.date_range(start=start, end=end, freq='1min')
        
        # Reindex day_data to include all minutes between 09:30 and 15:59
        day_data = day_data.set_index('timestamp').reindex(complete_timestamps)
        day_data.index.name = 'timestamp'
        
        # Perform linear interpolation for missing values
        for col in day_data:
            day_data[col] = pd.to_numeric(day_data[col], errors='coerce')
        day_data = day_data.interpolate(method='linear').reset_index()
        day_data['date'] = date  # Restore date column

        complete_df.append(day_data)
    
    if complete_df:
        return pd.concat(complete_df, ignore_index=True)
    else:
        return None

def download_stock_data(symbol, interval="1min", month=None):
    cache_path = get_cache_path(symbol, interval, month)
    
    # Check cache first
    if is_cache_valid(cache_path):
        print(f"Loading cached data for {symbol} {'(current)' if month is None else f'({month})'}")
        df = pd.read_parquet(cache_path)
        # Verify if any day has fewer than 100 rows
        return filter_and_fill(df)

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
        
            return filter_and_fill(df)
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
    
    for i in trange(len(symbols)): 
        symbol = symbols[i]
        combined_cache_path = get_combined_cache_path(symbol)
        if is_cache_valid(combined_cache_path):
            print(f"Loading combined cached data for {symbol}")
            df = pd.read_parquet(combined_cache_path)
            stock_data[symbol] = df
            all_timestamps.update(df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S'))
            continue

        # Download current data
        current_df = download_stock_data(symbol)
        
        # Download data for the months April 2024 to June 2024
        future_dfs = []
        months = [f"{yy}-{mm:02d}" for yy in range(int(DATA_PARAMS['firtst_year']), int(DATA_PARAMS['last_year']) + 1)
                  for mm in range(1, 13)]
        for month in months:
            future_df = download_stock_data(symbol, month=month)
            if future_df is not None:
                future_dfs.append(future_df)
        
        # Combine the dataframes if both current and future data are available
        if current_df is not None and future_dfs:
            df = pd.concat([current_df] + future_dfs, ignore_index=True)
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
            df.to_parquet(combined_cache_path, index=False)
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
    
    # Convert to Polars DataFrames
    stock_data = {symbol: pl.from_pandas(df) for symbol, df in stock_data.items()}
    
    return stock_data

def split_data(stock_data):
    train, val, test = {}, {}, {}
    train_symbols, val_symbols, test_symbols = [], [], []
    
    for symbol, df in stock_data.items():
        df = df.sort('timestamp')
        
        days = df.select(pl.col('date')).unique().sort('date')['date'].to_list()

        train_end = int(len(days) * DATA_PARAMS['train_end'])
        val_end = int(len(days))

        # Split the data
        train_df = df.filter(pl.col('date').is_in(days[:train_end]))
        val_df = df.filter(pl.col('date').is_in(days[train_end:val_end]))
        test_df = df.filter(pl.col('date').is_in(days[val_end:]))

        # Process each split
        if not train_df.is_empty():
            train[symbol] = train_df
            train_dates = train_df['timestamp'].dt.date().unique().to_list()
            train_symbols.extend([(date, symbol) for date in train_dates])
            
        if not val_df.is_empty():
            val[symbol] = val_df
            val_dates = val_df['timestamp'].dt.date().unique().to_list()
            val_symbols.extend([(date, symbol) for date in val_dates])
            
        if not test_df.is_empty():
            test[symbol] = test_df
            test_dates = test_df['timestamp'].dt.date().unique().to_list()
            test_symbols.extend([(date, symbol) for date in test_dates])

    # Process data for each split
    train_processed = prepare_dataset_data(train, train_symbols, 'train')
    val_processed = prepare_dataset_data(val, val_symbols, 'train')
    test_processed = prepare_dataset_data(test, test_symbols, 'viz')
    
    print(f"Train: {len(train_processed[0])} days, {sum(len(v) for v in train_processed[0].values())} data points")
    print(f"Val: {len(val_processed[0])} days, {sum(len(v) for v in val_processed[0].values())} data points")
    print(f"Test: {len(test_processed[0])} days, {sum(len(v) for v in test_processed[0].values())} data points")
    return train_processed, val_processed, test_processed

def process_day_data(day_data, mode='train', input_split=None):
    """Process a single day's data for model input"""
    input_values = day_data['close'][:input_split].to_numpy()
    
    if mode == 'train':
        # Calculate normalized price differences
        diffs = np.diff(input_values, prepend=input_values[0])
        max_diff = np.percentile(diffs, 99)
        min_diff = np.percentile(diffs, 1)
        
        if (max_diff - min_diff) == 0:
            normalized_diffs = np.zeros_like(diffs)
        else:
            normalized_diffs = (21*(diffs - min_diff) / (max_diff - min_diff)).astype(np.int8)
            normalized_diffs = np.clip(normalized_diffs, -31, 31)
        return pl.Series(normalized_diffs)
    else:
        return pl.Series(input_values)

def prepare_stock_data(stock_data):
    """Prepare stock data dictionary with processed data"""
    processed_data = {}
    date_symbols = []
    
    for symbol, df in stock_data.items():
        if df is not None and not df.is_empty():
            required_cols = {'timestamp', 'close'}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns for {symbol}: {missing_cols}")
            
            try:
                processed_data[symbol] = df
                dates = df['timestamp'].dt.date().unique().to_list()
                date_symbols.extend([(date, symbol) for date in dates])
            except Exception as e:
                raise ValueError(f"Error processing timestamps for {symbol}: {str(e)}")
    
    return processed_data, date_symbols

def prepare_dataset_data(data_dict, date_symbols, mode='train'):
    """Convert raw data into format ready for StockDataset"""
    processed_data = {}
    
    for symbol, df in data_dict.items():
        processed_data[symbol] = {}
        for date in df['timestamp'].dt.date().unique():
            day_data = df.filter(pl.col('timestamp').dt.date() == date)
            if len(day_data) == 0:
                continue
                
            input_split = int(len(day_data) * MODEL_PARAMS['input_ratio'])
            target_split = int(len(day_data) * MODEL_PARAMS['target_ratio'])
            
            # Process input values
            input_values = process_day_data(day_data, mode, input_split)
            
            # Calculate target
            target_value = (day_data['close'][target_split:].max() - day_data['close'][input_split-1])
            
            processed_data[symbol][date] = (input_values, target_value)
    
    return processed_data, date_symbols