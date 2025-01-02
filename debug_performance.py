import torch
from predictor import StockPredictor
from data_loader import get_stock_data, split_data
from config import SYMBOLS, MODEL_PARAMS
from stock_dataset import StockDataset
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def load_debug_model():
    try:
        state_dict = torch.load('unified_stock_model.pt', map_location=torch.device('cpu'))
        input_size = state_dict['model.1.weight'].shape[1]
        model = StockPredictor(input_size=input_size)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

def debug_performance(model, test_data, verbose=False):
    performances = []
    
    for symbol in SYMBOLS:
        if symbol not in test_data:
            logging.warning(f"Symbol {symbol} not found in test data")
            continue
            
        dataset = StockDataset({symbol: test_data[symbol]})
        if len(dataset) == 0:
            logging.warning(f"Empty dataset for symbol {symbol}")
            continue

        for idx, (date, sym) in enumerate(dataset.date_symbols):
            if sym != symbol:
                continue

            try:
                # Get input data
                input_values, actual_price = dataset[idx]
                last_input_price = input_values[-1].item()
                actual_price = actual_price.item() + last_input_price

                # Get predictions
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(input_values).unsqueeze(0)
                    predictions = model(input_tensor)
                    predictions = predictions.squeeze().numpy() + last_input_price

                # Calculate performance using P20 and P80
                p20_idx = MODEL_PARAMS['quantiles'].index(0.2)
                p80_idx = MODEL_PARAMS['quantiles'].index(0.8)
                performance = (np.min(predictions) - last_input_price) / max(0.1, last_input_price)

                if verbose:
                    logging.info(f"\nSymbol: {symbol}, Date: {date}")
                    logging.info(f"Last Input Price: {last_input_price:.2f}")
                    logging.info(f"Actual Price: {actual_price:.2f}")
                    logging.info(f"Predictions: {[f'{p:.2f}' for p in predictions]}")
                    logging.info(f"Performance Score: {performance:.4f}")

                performances.append((symbol, date, float(performance), float(last_input_price)))

            except Exception as e:
                logging.error(f"Error processing {symbol} for {date}: {str(e)}")
                continue

    return sorted(performances, key=lambda x: x[2], reverse=True)

def main():
    logging.info("Loading model...")
    model = load_debug_model()
    if model is None:
        return

    logging.info("Loading stock data...")
    train_data, val_data, test_data = split_data(get_stock_data(SYMBOLS))
    
    logging.info("Calculating performances...")
    performances = debug_performance(model, test_data)
    
    logging.info("\nTop 10 Predictions:")
    for symbol, date, perf, last_price in performances[:10]:
        logging.info(f"{symbol} on {date.strftime('%Y-%m-%d')}: {perf*100:.2f}% (Last: ${last_price:.2f})")

if __name__ == "__main__":
    main()
