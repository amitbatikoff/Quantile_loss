import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import multiprocessing
from stock_dataset import StockDataset
from predictor import StockPredictor
from data_loader import get_stock_data, split_data
from config import SYMBOLS, DATALOADER_PARAMS
import time

def main():
    # Start time measurement
    start_time = time.time()
    stock_data = get_stock_data(SYMBOLS)
    train, val, _ = split_data(stock_data)
    end_time = time.time()
    print(f"Data loading took {end_time - start_time:.2f} seconds")

    # Calculate optimal number of workers
    num_workers = min(multiprocessing.cpu_count() - 1, 11)

    # Use a smaller batch size or ensure dataset is large enough
    batch_size = DATALOADER_PARAMS['batch_size']
    
    start_time = time.time()
    train_loader = DataLoader(StockDataset(train), batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(StockDataset(val), batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, persistent_workers=True)
    end_time = time.time()
    print(f"Dataset creation took {end_time - start_time:.2f} seconds")

    start_time = time.time()
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[1]
    end_time = time.time()
    print(f"sample_batch {end_time - start_time:.2f} seconds, input_size: {input_size}")

    # Calculate total steps for the scheduler
    total_steps = len(train_loader) * 1000  # 1000 is max_epochs

    start_time = time.time()
    model = StockPredictor(input_size=input_size, total_steps=total_steps)
    end_time = time.time()
    print(f"model setting took {end_time - start_time:.2f} seconds")

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
        patience=75,
        mode='min'
    )

    # Initialize lr scheduler callback
    lr_scheduler = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # Initialize trainer with scheduler
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=[checkpoint_callback, early_stop_callback, lr_scheduler],
        log_every_n_steps=1,
        deterministic=True,
        enable_model_summary=True,
        gradient_clip_val=0.5,  # Added gradient clipping for stability
        enable_progress_bar=True,
    )

    # Train model
    print("\nTraining unified model for all stocks")
    trainer.fit(model, train_loader, val_loader)

    # Save model
    torch.save(model.state_dict(), 'unified_stock_model.pt')

if __name__ == "__main__":
    main()