import json
import os

API_KEY = "FGU3D4XLZIJ9Q8YJ"
BASE_URL = "https://www.alphavantage.co/query"

# Load symbols from JSON file
with open(os.path.join(os.path.dirname(__file__), 'stock_list.json'), 'r') as f:
    SYMBOLS = json.load(f)['symbols']

MODEL_PARAMS = {
    "learning_rate": 1e-4,
    "input_ratio": 0.5,  # Added for consistency
    "target_ratio": 0.6,  # Added for consistency
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "architecture": {
        "hidden_sizes": [512, 512, 256, 256, 128, 128, 64, 64, 32, 32],  # List of hidden layer sizes
        "dropout_rates": [0.2, 0.2, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],  # Per-layer dropout rates
        "use_batch_norm": [True, True, True, True, False, False, False, False, False, False],  # Per-layer batch norm flags
        "attention": {
            "num_heads": 8,
            "head_dim": 64,
            "attention_dropout": 0.1
        }
    }
}

DATALOADER_PARAMS = {
    "batch_size": 4096,
    "pin_memory": True
}

CACHE_DIR = "cache"
CACHE_EXPIRY_DAYS = 10  # How many days before cache expires