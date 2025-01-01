import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class StockDataset(Dataset):
    def __init__(self, data_dict, input_ratio=0.7, target_ratio=0.9):
        self.data = pd.concat([df.assign(symbol=symbol) for symbol, df in data_dict.items()])
        self.input_ratio = input_ratio
        self.target_ratio = target_ratio
        numerical_cols = self.data.select_dtypes(include=np.number).columns
        self.data[numerical_cols] = self.data[numerical_cols].astype('float64')
        self.numerical_cols = numerical_cols
        self.grouped = self.data.groupby([self.data['timestamp'].dt.date, 'symbol'])
        self.date_symbols = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.date_symbols)

    def __getitem__(self, idx):
        day, symbol = self.date_symbols[idx]
        day_data = self.grouped.get_group((day, symbol)).sort_values('timestamp')

        total_len = len(day_data)
        input_len = int(total_len * self.input_ratio)
        target_idx = int(total_len * self.target_ratio)

        input_data = day_data.iloc[:input_len][['close']].values.astype('float32')
        target_data = day_data.iloc[target_idx]['close'].astype('float32')-input_data[-1]

        return torch.tensor(input_data, dtype=torch.float32), torch.tensor([target_data], dtype=torch.float32)