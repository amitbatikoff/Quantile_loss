import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from config import MODEL_PARAMS
import matplotlib.pyplot as plt

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self, preds, target):
        # preds shape: [batch_size, num_quantiles]
        # target shape: [batch_size, 1]
        target = target.expand_as(preds)
        errors = target - preds
        loss = torch.max(
            (self.quantiles - 1) * errors,
            self.quantiles * errors
        )
        return loss.mean()

class StockPredictor(pl.LightningModule):
    def __init__(self, input_size, total_steps=1000):
        super(StockPredictor, self).__init__()
        self.save_hyperparameters()
        self.total_steps = total_steps
        self.quantiles = MODEL_PARAMS['quantiles']
        
        # Build model dynamically based on config
        layers = []
        
        # Input layer
        prev_size = input_size
        layers.extend([
            nn.Flatten(),
        ])
        
        # Hidden layers
        for i, hidden_size in enumerate(MODEL_PARAMS['architecture']['hidden_sizes']):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
            ])
            # Add dropout after first two layers
            if i < 2:
                layers.append(nn.Dropout(p=0.1))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, len(self.quantiles)))
        
        self.model = nn.Sequential(*layers)
        self.loss_fn = QuantileLoss(self.quantiles)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)  # Shape: [batch_size, num_quantiles]
        targets = targets.view(-1, 1)  # Shape: [batch_size, 1]
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        targets = targets.view(-1, 1)
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = MODEL_PARAMS['learning_rate']
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.05,
                patience=10,
                threshold=0.3,
                min_lr=1e-6,
                threshold_mode = 'abs'
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}