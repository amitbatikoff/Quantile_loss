import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from stock_dataset import StockDataset
from predictor import StockPredictor
from data_loader import get_stock_data, split_data
from config import SYMBOLS, MODEL_PARAMS

def main():
    # Get and process data
    stock_data = get_stock_data(SYMBOLS)
    train, val, test = split_data(stock_data)

    # Create dataloaders
    train_loader = DataLoader(StockDataset(train), batch_size=128, shuffle=True)
    val_loader = DataLoader(StockDataset(val), batch_size=128, shuffle=False)
    test_loader = DataLoader(StockDataset(test), batch_size=128, shuffle=False)

    # Get input size from a sample batch
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[1]

    # Initialize model
    model = StockPredictor(input_size=input_size, hidden_dim=MODEL_PARAMS["hidden_dim"])

    # Setup callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='stock-predictor-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=1000,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=1,
        deterministic=True
    )

    # Train model
    print("\nTraining unified model for all stocks")
    trainer.fit(model, train_loader, val_loader)

    # Save model
    torch.save(model.state_dict(), 'unified_stock_model.pt')

if __name__ == "__main__":
    main()