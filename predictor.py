import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

class QuantileLoss(nn.Module):
    def __init__(self, quantile):
        super(QuantileLoss, self).__init__()
        self.quantile = torch.tensor(quantile, dtype=torch.float32)

    def forward(self, preds, targets):
        errors = targets - preds
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(loss)

class StockPredictor(pl.LightningModule):
    def __init__(self, input_size, hidden_dim):
        super(StockPredictor, self).__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.loss_fn = QuantileLoss(quantile=0.5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        targets = targets.view(-1, 1)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=0.001,
                epochs=1000,
                steps_per_epoch=len(self.trainer.train_dataloader),
                pct_start=0.3,  # Use 30% of training for warmup
                div_factor=25.0,  # initial_lr = max_lr/25
                final_div_factor=10000.0,  # min_lr = initial_lr/10000
            ),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}