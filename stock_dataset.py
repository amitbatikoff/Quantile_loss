import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from config import MODEL_PARAMS

class StockDataset(Dataset):
    def __init__(self, data, mode='train'):
        """
        Args:
            data: Dictionary of stock data
            mode: 'train' for training (returns diffs) or 'viz' for visualization (returns raw prices)
        """
        self.mode = mode
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
        
        # Convert to tensors
        if self.mode == 'train':
            # For training, use price differences
            diffs = np.diff(input_values, prepend=input_values[0])
            max_diff = diffs.max()
            min_diff = diffs.min()
            if (max_diff - min_diff) == 0:
                normalized_diffs = np.zeros_like(diffs)
            else:
                normalized_diffs = (5*(diffs - min_diff) / (max_diff - min_diff)).astype(np.uint8)
            input_tensor = torch.from_numpy(normalized_diffs).float()
        else:
            # For visualization, use raw prices
            input_tensor = torch.from_numpy(input_values).float()

        target_value = (day_data['close'].iloc[target_split:] - day_data['close'].iloc[input_split-1]).max()            
        target_tensor = torch.tensor([target_value], dtype=torch.float)

        return input_tensor.clone().detach(), target_tensor.clone().detach()

    def find_date_index(self, target_date):
        """Find the index for a specific date in the dataset"""
        for idx, (date, _) in enumerate(self.date_symbols):
            if date == target_date:
                return idx
        return None