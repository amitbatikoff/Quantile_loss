import pickle
import pandas as pd
import numpy as np
from stock_dataset import StockDataset
from data_analysis import analyze_data

def compute_features(arr):
    # Compute summary features from the input array
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
    }

def main():
    # Load processed train data (processed_data, date_symbols)
    with open("train.pkl", "rb") as f:
        processed_data = pickle.load(f)
        
    dataset = StockDataset(processed_data)
    features_list = []
    targets = []
    
    for input_tensor, target_tensor in dataset:
        # Get numpy array from tensor
        arr = input_tensor.numpy()
        feats = compute_features(arr)
        features_list.append(feats)
        targets.append(target_tensor.item())
    
    X = pd.DataFrame(features_list)
    y = pd.Series(targets)
    analyze_data(X, y, "stock_train_")

if __name__ == "__main__":
    main()
