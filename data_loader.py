import requests
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
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

def filter_and_fill(table):
    # Ensure volume is int64
    table = table.set_column(
        table.schema.get_field_index('volume'),
        'volume',
        pa.array(table.column('volume').to_pandas().astype('int64'))
    )
    # Extract date from timestamp
    table = table.append_column('date', pc.cast(table['timestamp'], pa.date32()))

    # Group by date and validate times
    valid_dates = []
    for date in pc.unique(table['date']).to_pylist():
        day_data = table.filter(pc.equal(table['date'], date))
        start_time = pc.min(day_data['timestamp']).as_py().time()
        end_time = pc.max(day_data['timestamp']).as_py().time()
    
        # Check if the day starts at 09:30 and ends at 15:59
        if start_time == pd.Timestamp("09:30").time() and end_time == pd.Timestamp("15:59").time():
            valid_dates.append(date)

    # Filter for valid dates
    table = table.filter(pc.is_in(table['date'], pa.array(valid_dates)))

    # Filter out days with fewer than 100 rows
    valid_dates = []
    for date in pc.unique(table['date']).to_pylist():
        day_data = table.filter(pc.equal(table['date'], date))
        if day_data.num_rows >= 100:
            valid_dates.append(date)
    
    # Filter for valid dates
    table = table.filter(pc.is_in(table['date'], pa.array(valid_dates)))

    # Generate a complete list of timestamps for each date
    complete_tables = []
    for date in pc.unique(table['date']).to_pylist():
        day_data = table.filter(pc.equal(table['date'], date))
        start = pd.Timestamp(f"{date} 09:30")
        end = pd.Timestamp(f"{date} 15:59")
        complete_timestamps = pd.date_range(start=start, end=end, freq='1min')
        
        # Reindex day_data to include all minutes between 09:30 and 15:59
        complete_day_data = pa.Table.from_pandas(
            day_data.to_pandas().set_index('timestamp').reindex(complete_timestamps).reset_index()
        )
        
        # Perform linear interpolation for missing values
        for col in complete_day_data.column_names:
            if pa.types.is_integer(complete_day_data[col].type) or pa.types.is_floating(complete_day_data[col].type):
                complete_day_data = complete_day_data.set_column(
                    complete_day_data.schema.get_field_index(col),
                    col,
                    pc.fill_null(complete_day_data[col], pc.mean(complete_day_data[col]))
                )
        
        # Ensure volume is int64
        complete_day_data = complete_day_data.set_column(
            complete_day_data.schema.get_field_index('volume'),
            'volume',
            pa.array(complete_day_data.column('volume').to_pandas().astype('int64'))
        )
        
        complete_tables.append(complete_day_data)
    
    if complete_tables:
        final_table = pa.concat_tables(complete_tables)
        # Ensure 'timestamp' column is included
        if 'timestamp' not in final_table.column_names:
            min_timestamp = pc.min(table['timestamp']).as_py()
            max_timestamp = pc.max(table['timestamp']).as_py()
            all_timestamps = pd.date_range(start=min_timestamp, end=max_timestamp, freq='1min')
            final_table = final_table.append_column('timestamp', pa.array(all_timestamps[:len(final_table)]))
        return final_table
    else:
        return None

def download_stock_data(symbol, interval="1min", month=None):
    cache_path = get_cache_path(symbol, interval, month)
    
    # Check cache first
    if is_cache_valid(cache_path):
        print(f"Loading cached data for {symbol} {'(current)' if month is None else f'({month})'}")
        table = pq.read_table(cache_path)
        # Verify if any day has fewer than 100 rows
        return filter_and_fill(table)

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
            df['volume'] = df['volume'].astype('int64')  # Ensure volume is int64
            table = pa.Table.from_pandas(df)
            pq.write_table(table, cache_path)
        
            return filter_and_fill(table)
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
        current_table = download_stock_data(symbol)
        
        # Download data for the months April 2024 to June 2024
        future_tables = []
        months = [f"{yy}-{mm:02d}" for yy in range(2021, 2025) for mm in range(1, 13)]
        for month in months:
            future_table = download_stock_data(symbol, month=month)
            if future_table is not None:
                future_tables.append(future_table)
        
        # Combine the tables if both current and future data are available
        if current_table is not None and future_tables:
            future_tables = [t.set_column(
                t.schema.get_field_index('volume'),
                'volume',
                pa.array(t.column('volume').to_pandas().astype('int64'))
            ) for t in future_tables]  # Ensure volume is int64
            current_table = current_table.set_column(
                current_table.schema.get_field_index('volume'),
                'volume',
                pa.array(current_table.column('volume').to_pandas().astype('int64'))
            )  # Ensure volume is int64 for current table
            table = pa.concat_tables([current_table] + future_tables)
            
            # Filter out days with fewer than 100 rows
            valid_dates = []
            for date in pc.unique(table['date']).to_pylist():
                day_data = table.filter(pc.equal(table['date'], date))
                if day_data.num_rows >= 100:
                    valid_dates.append(date)
            table = table.filter(pc.is_in(table['date'], pa.array(valid_dates)))
            
            stock_data[symbol] = table
            if 'timestamp' in table.schema.names:
                all_timestamps.update(table['timestamp'].to_pandas().dt.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Rest of the function remains the same
    all_timestamps = sorted(list(all_timestamps))
    
    return stock_data

def split_data(stock_data):
    train, val, test = {}, {}, {}
    for symbol, table in stock_data.items():
        table = table.sort_by('timestamp')
        grouped = table.group_by('date')
        days = [group[0] for group in grouped.aggregate([]).to_pandas().itertuples(index=False)]

        train_end = int(len(days) * 0.6)
        val_end = int(len(days) * 0.8)

        train[symbol] = table.filter(pc.is_in(table['date'], pa.array(days[:train_end])))
        val[symbol] = table.filter(pc.is_in(table['date'], pa.array(days[train_end:val_end])))
        test[symbol] = table.filter(pc.is_in(table['date'], pa.array(days[val_end:])))

    return train, val, test