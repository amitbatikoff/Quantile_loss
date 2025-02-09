import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import multiprocessing
from stock_dataset import StockDataset
from predictor import StockPredictor
from data_loader import get_stock_data, split_data
from config import SYMBOLS, DATALOADER_PARAMS, MODEL_PARAMS, DATA_PARAMS, OPTIMIZER_PARAMS
import time
import pickle
from clearml import Task
from pytorch_lightning.loggers import TensorBoardLogger
from clearml_logger import ClearMLLogger
import logging
logging.getLogger('clearml.frameworks').setLevel(logging.WARNING)

def main():
    # Initialize ClearML task
    task = Task.init(project_name="1min pred", task_name="hihgher decay")
    task.connect(MODEL_PARAMS)
    task.connect(DATALOADER_PARAMS)
    task.connect(DATA_PARAMS)
    task.connect(OPTIMIZER_PARAMS)
    
    # Start time measurement
    print('start')
    start_time = time.time()
    # Data Loading and Logging

    # saved_hash = None
    # if os.path.exists('updated_hash.pkl'):
    #     print(f"File {'updated_hash.pkl'} exists. Opening it...")
    #     with open('updated_hash.pkl', 'rb') as file:
    #         saved_hash = pickle.load(file)
    # updated_hash = calculate_folder_hash('cache\\')
    # saved_hash = None
    # if os.path.exists('updated_hash.pkl'):
    #     print(f"File {'updated_hash.pkl'} exists. Opening it...")
    #     with open('updated_hash.pkl', 'rb') as file:
    #         saved_hash = pickle.load(file)
    # updated_hash = calculate_folder_hash('cache\\')

    # # if saved_hash != updated_hash:
    # stock_data = get_stock_data(SYMBOLS)
    # print("Data Loading", "Time (seconds)", time.time() - start_time)
    # start_time = time.time()
    # train, val, _ = split_data(stock_data,parallel=True)
    # print("Data Split", "Time (seconds)", time.time() - start_time)

    # with open("train.pkl", "wb") as f:
    #     pickle.dump(train, f)
    # with open("val.pkl", "wb") as f:
    #     pickle.dump(val, f)

        # with open("updated_hash.pkl", "wb") as f:
        #     pickle.dump(updated_hash, f)
    # else:
    with open("train.pkl", "rb") as f:
        train = pickle.load(f)
    with open("val.pkl", "rb") as f:
        val = pickle.load(f)


    task.get_logger().report_scalar("Data Loading", "Time (seconds)", time.time() - start_time,0) 

    # Worker and Batch Size Calculation
    num_workers = min(multiprocessing.cpu_count() - 1, 5)
    batch_size = DATALOADER_PARAMS['batch_size']
    
    start_time = time.time()
    train_loader = DataLoader(StockDataset(train), batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True,persistent_workers=True)
    val_loader = DataLoader(StockDataset(val), batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True,persistent_workers=True)
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
        monitor='loss/val',
        dirpath='checkpoints',
        filename='stock-predictor-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min'
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='loss/val',
        patience=150,
        mode='min'
    )

    # Initialize lr scheduler callback
    lr_scheduler = pl.callbacks.LearningRateMonitor(logging_interval='step')

    # Create loggers
    tb_logger = TensorBoardLogger(save_dir='lightning_logs', name='quantile_loss')
    clearml_logger = ClearMLLogger(task, upload_interval=500)

    # Initialize trainer without the ModelUploadCallback
    trainer = pl.Trainer(
        max_epochs=1000,
        callbacks=[checkpoint_callback, early_stop_callback, lr_scheduler],
        log_every_n_steps=1,
        deterministic=True,
        enable_model_summary=True,
        gradient_clip_val=0.5,
        enable_progress_bar=True,
        logger=[tb_logger, clearml_logger]  # Use both loggers
    )

    # Train model
    print("\nTraining unified model for all stocks")
    trainer.fit(model, train_loader, val_loader)

    # Save final model with metadata
    final_val_loss = trainer.callback_metrics.get('loss/val', 0)
    final_train_loss = trainer.callback_metrics.get('loss/train', 0)
    
    torch.save(model.state_dict(), 'unified_stock_model.pt')
    task.upload_artifact(
        'final_model',
        'unified_stock_model.pt',
        metadata={
            'final_epoch': trainer.current_epoch,
            'val_loss': float(final_val_loss),
            'train_loss': float(final_train_loss),
        }
    )

if __name__ == "__main__":
    main()