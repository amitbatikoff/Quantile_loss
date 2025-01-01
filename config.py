API_KEY = "FGU3D4XLZIJ9Q8YJ"
BASE_URL = "https://www.alphavantage.co/query"

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "BRK-B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "NFLX",
    "PYPL", "INTC", "CSCO", "PEP", "KO", "MRK", "ABT", "ABBV", "T", "PFE",
    "XOM", "CVX", "NKE", "WMT", "MCD", "COST", "LLY", "MDT", "NEE", "DHR",
    "TXN", "AVGO", "QCOM", "HON", "UNP", "LIN", "PM", "ACN", "IBM", "ORCL",
    "TMO", "BMY", "SBUX", "GS", "RTX", "AMGN", "CAT", "BA", "SPGI", "BLK",
    "AXP", "CVS", "LOW", "MS", "INTU", "PLD", "CHTR", "GE", "ISRG", "NOW",
    "MDLZ", "LMT", "BKNG", "DE", "ADP", "SYK", "CI", "ZTS", "CB", "ANTM",
    "MMC", "GILD", "ADI", "DUK", "SO", "TGT", "MO", "BDX", "APD", "EW",
    "ITW", "ICE", "SCHW", "HUM", "PNC", "NSC", "CL", "SHW", "CME", "FIS"
]

MODEL_PARAMS = {
    "hidden_dim": 512,
    "learning_rate": 1e-6,
    "input_ratio": 0.7,  # Added for consistency
    "target_ratio": 0.9  # Added for consistency
}

DATALOADER_PARAMS = {
    "batch_size": 1024,
    "pin_memory": True
}

CACHE_DIR = "cache"
CACHE_EXPIRY_DAYS = 10  # How many days before cache expires