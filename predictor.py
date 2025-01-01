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
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }