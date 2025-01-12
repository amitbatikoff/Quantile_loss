import torch
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.compute as pc
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
        
        for symbol, table in data.items():
            # Ensure the data is in the form of Apache Arrow Table
            if isinstance(table, pd.DataFrame):
                table = pa.Table.from_pandas(table)
            
            if table is not None and table.num_rows > 0:
                # Ensure Table has required columns
                required_cols = {'timestamp', 'close'}
                missing_cols = required_cols - set(table.column_names)
                if missing_cols:
                    raise ValueError(f"Missing required columns for {symbol}: {missing_cols}")
                
                # Handle timestamp column
                try:
                    if not pa.types.is_timestamp(table.schema.field('timestamp').type):
                        table = table.set_column(
                            table.schema.get_field_index('timestamp'),
                            'timestamp',
                            pc.cast(table['timestamp'], pa.timestamp('s'))
                        )
                    
                    self.data[symbol] = table
                    dates = pc.unique(pc.cast(table['date'], pa.date32())).to_pylist()
                    self.date_symbols.extend([(date, symbol) for date in dates])
                except Exception as e:
                    raise ValueError(f"Error processing timestamps for {symbol}: {str(e)}")

    def __len__(self):
        print(len(self.date_symbols))
        return len(self.date_symbols)

    def __getitem__(self, idx):
        date, symbol = self.date_symbols[idx]
        day_data = self.data[symbol].filter(pc.equal(pc.cast(self.data[symbol]['date'], pa.date32()), date))
        
        if day_data.num_rows == 0:
            raise IndexError(f"No data found for {symbol} on {date}")
            
        # Calculate split points
        input_split = int(day_data.num_rows * MODEL_PARAMS['input_ratio'])
        target_split = int(day_data.num_rows * MODEL_PARAMS['target_ratio'])
        
        # Get input values and target
        input_values = day_data['close'][:input_split].to_numpy()
        
        if len(input_values) == 0:
            raise IndexError(f"No input values found for {symbol} on {date}")
        
        # Convert to tensors
        if self.mode == 'train':
            # For training, use price differences
            diffs = np.diff(input_values, prepend=input_values[0])
            max_diff = np.percentile(diffs, 99)
            min_diff = np.percentile(diffs, 1)
            if (max_diff - min_diff) == 0:
                normalized_diffs = np.zeros_like(diffs)
            else:
                normalized_diffs = (21*(diffs - min_diff) / (max_diff - min_diff)).astype(np.int8)
                normalized_diffs = np.clip(normalized_diffs, -31, 31)
            input_tensor = torch.from_numpy(normalized_diffs).float()
        else:
            # For visualization, use raw prices
            input_tensor = torch.from_numpy(input_values).float()

        target_value = (day_data['close'][target_split:].to_numpy() - day_data['close'][input_split-1].as_py()).max()            
        target_tensor = torch.tensor([target_value], dtype=torch.float)

        return input_tensor.clone().detach(), target_tensor.clone().detach()

    def find_date_index(self, target_date):
        """Find the index for a specific date in the dataset"""
        for idx, (date, _) in enumerate(self.date_symbols):
            if date == target_date:
                return idx
        return None