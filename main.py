import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import multiprocessing
from stock_dataset import StockDataset
from predictor import StockPredictor
from data_loader import get_stock_data, split_data
from config import SYMBOLS, DATALOADER_PARAMS, MODEL_PARAMS
import time
import os
import hashlib
import pickle

def calculate_folder_hash(folder_path):
    hash_obj = hashlib.md5()
    for root, dirs, files in os.walk(folder_path):
        for names in files:
            file_path = os.path.join(root, names)
            # Update hash with file content and metadata
            try:
                with open(file_path, 'rb') as f:
                    while chunk := f.read(8192):
                        hash_obj.update(chunk)
                hash_obj.update(names.encode('utf-8'))
            except Exception as e:
                print(f"Error hashing file {file_path}: {e}")
    return hash_obj.hexdigest()



def main():
    print('start')
    # Start time measurement and ClearML Task initialization
    # task = Task.init(project_name="1min pred", task_name="test1")
    # task.connect(DATALOADER_PARAMS)
    # task.connect(MODEL_PARAMS)
    
    start_time = time.time()
    # Data Loading and Logging

    # saved_hash = None
    # if os.path.exists('updated_hash.pkl'):
    #     print(f"File {'updated_hash.pkl'} exists. Opening it...")
    #     with open('updated_hash.pkl', 'rb') as file:
    #         saved_hash = pickle.load(file)
    # updated_hash = calculate_folder_hash('cache\\')

    # if saved_hash != updated_hash:
    #     stock_data = get_stock_data(SYMBOLS)
    #     train, val, _ = split_data(stock_data)
    #     with open("train.pkl", "wb") as f:
    #         pickle.dump(train, f)
    #     with open("val.pkl", "wb") as f:
    #         pickle.dump(val, f)
    #     with open("updated_hash.pkl", "wb") as f:
    #         pickle.dump(updated_hash, f)
    # else:
    with open("train.pkl", "rb") as f:
        train = pickle.load(f)
    with open("val.pkl", "rb") as f:
        val = pickle.load(f)

    print("Data Loading", "Time (seconds)", time.time() - start_time)
    # task.get_logger().report_scalar("Data Loading", "Time (seconds)", time.time() - start_time,0) 

    # Worker and Batch Size Calculation
    num_workers = min(multiprocessing.cpu_count() - 1, 4)
    batch_size = DATALOADER_PARAMS['batch_size']

    # Data Loader Creation and Logging
    start_time = time.time()
    train_loader = DataLoader(StockDataset(train), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(StockDataset(val), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, persistent_workers=True)
    print("Data Loader Creation", "Time (seconds)", time.time() - start_time)
    # task.get_logger().report_scalar("Data Loader Creation", "Time (seconds)", time.time() - start_time,0)

    # Sample Batch and Input Size (commented out for efficiency)
    try:
        sample_batch = next(iter(train_loader)) 
    except Exception as e:
        print('error')
    input_size = 195#sample_batch[0].shape[1] 

    # Total Steps for Scheduler
    total_steps = len(train_loader) * 1000 

    # Model Creation and Logging
    start_time = time.time()
    model = StockPredictor(input_size=input_size, total_steps=total_steps)
    # task.get_logger().report_scalar("Model Creation", "Time (seconds)", time.time() - start_time,0)
    # Log model parameters (optional)
    # task.connect(model) 

    # Callback Setup
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss', 
        dirpath='checkpoints', 
        filename='stock-predictor-{epoch:02d}-{val_loss:.2f}', 
        save_top_k=3, 
        mode='min'
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=200, 
        mode='min'
    )
    lr_scheduler = pl.callbacks.LearningRateMonitor(logging_interval='step')
    # clearml_logger = ClearMLLogger(task=task)

    # Trainer Initialization
    trainer = pl.Trainer(
        max_epochs=2000, 
        callbacks=[checkpoint_callback, early_stop_callback, lr_scheduler], 
        log_every_n_steps=1, 
        deterministic=True, 
        enable_model_summary=True, 
        gradient_clip_val=0.5, 
        enable_progress_bar=True,
        # logger=clearml_logger  # Use ClearML logger
    )

    # Training with ClearML Logging
    trainer.fit(model, train_loader, val_loader)

    # Model Saving
    torch.save(model.state_dict(), 'unified_stock_model.pt')

if __name__ == "__main__":
    main()
