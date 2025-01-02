import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from config import MODEL_PARAMS

class StockDataset(Dataset):
    def __init__(self, data):
        self.data = {}
        self.date_symbols = []
        
        for symbol, df in data.items():
            if df is not None and not df.empty:
                # Ensure DataFrame has required columns
                required_cols = {'timestamp', 'close'}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    raise ValueError(f"Missing required columns for {symbol}: {missing_cols}")
                
                # Handle timestamp column
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    self.data[symbol] = df
                    dates = df['timestamp'].dt.date.unique()
                    self.date_symbols.extend([(date, symbol) for date in dates])
                except Exception as e:
                    raise ValueError(f"Error processing timestamps for {symbol}: {str(e)}")

    def __len__(self):
        return len(self.date_symbols)

    def __getitem__(self, idx):
        date, symbol = self.date_symbols[idx]
        day_data = self.data[symbol][self.data[symbol]['timestamp'].dt.date == date]
        
        if len(day_data) == 0:
            raise IndexError(f"No data found for {symbol} on {date}")
            
        # Calculate split points
        input_split = int(len(day_data) * MODEL_PARAMS['input_ratio'])
        target_split = int(len(day_data) * MODEL_PARAMS['target_ratio'])
        
        # Get input values and target
        input_values = day_data['close'].iloc[:input_split].values
        target_value = (day_data['close'].iloc[target_split:] - day_data['close'].iloc[input_split-1]).max()
        
        # Convert to tensors and ensure correct shape
        input_tensor = torch.FloatTensor(input_values)
        target_tensor = torch.FloatTensor([target_value])
        
        return input_tensor, target_tensor