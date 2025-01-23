import json
import os

API_KEY = "FGU3D4XLZIJ9Q8YJ"
BASE_URL = "https://www.alphavantage.co/query"

# Load symbols from JSON file
with open(os.path.join(os.path.dirname(__file__), 'stock_list.json'), 'r') as f:
    SYMBOLS = json.load(f)['symbols']

MODEL_PARAMS = {
    "input_ratio": 0.5,  # Added for consistency
    "target_ratio": 0.6,  # Added for consistency
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "architecture": {
        "hidden_sizes":   [16], 
        "dropout_rates":  [0  ], 
        "use_batch_norm": [False], 
        "attention": {
            "num_heads": 8,
            "head_dim": 64,
            "attention_dropout": 0.7
        },
        "num_attention_blocks": 5,
        "feedforward_dim": 64
    }
}

DATALOADER_PARAMS = {
    "batch_size": 32768,
}

DATA_PARAMS = {
    "train_end": 0.8,
    "firtst_year": "2016",
    "last_year": "2024",
    "CACHE_DIR": "cache",
    "CACHE_EXPIRY_DAYS": 10000
}

OPTIMIZER_PARAMS = {
    "learning_rate": 2e-5,
    "base_lr_factor": 0.15,
    "step_size_up": 400,
    "gamma": 0.999994,
    "cycle_momentum": True
}