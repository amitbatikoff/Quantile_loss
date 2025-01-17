import torch
from torch.utils.data import Dataset
import numpy as np
from config import MODEL_PARAMS

class StockDataset(Dataset):
    def __init__(self, processed_data):
        """
        Args:
            processed_data: Tuple of (data_dict, date_symbols) where data is preprocessed but not tensorized
        """
        self.data, self.date_symbols = processed_data

    def __len__(self):
        return len(self.date_symbols)

    def __getitem__(self, idx):
        date, symbol = self.date_symbols[idx]
        day_data = self.data[symbol][date]
        
        if day_data is None:
            raise IndexError(f"No data found for {symbol} on {date}")
        
        input_values, target_value = day_data
        
        # Convert to tensors
        input_values_copy = np.array(input_values.to_numpy(), copy=True)
        input_tensor = torch.from_numpy(input_values_copy).float()
        target_tensor = torch.tensor([target_value], dtype=torch.float)
        
        return input_tensor.clone().detach(), target_tensor.clone().detach()

    def find_date_index(self, target_date):
        for idx, (date, _) in enumerate(self.date_symbols):
            if date == target_date:
                return idx
        return None