API_KEY = "FGU3D4XLZIJ9Q8YJ"
BASE_URL = "https://www.alphavantage.co/query"

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "NFLX"
]

MODEL_PARAMS = {
    "hidden_dim": 512,
    "learning_rate": 1e-5,
    "input_ratio": 0.7,  # Added for consistency
    "target_ratio": 0.9  # Added for consistency
}

DATALOADER_PARAMS = {
    "batch_size": 128,
    "pin_memory": True
}

CACHE_DIR = "cache"
CACHE_EXPIRY_DAYS = 10  # How many days before cache expires