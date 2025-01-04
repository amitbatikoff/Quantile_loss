import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from config import MODEL_PARAMS
import matplotlib.pyplot as plt
import math

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

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

        self.q_proj = nn.Linear(input_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(input_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(input_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        
        # Project inputs to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        pe = torch.zeros(max_len, d_model)
        
        # Handle dimensions properly regardless of odd/even
        d_pairs = (d_model + 1) // 2  # Ceiling division to handle odd dimensions
        div_term = torch.exp(torch.arange(0, d_pairs) * (-math.log(10000.0) / d_model))
        position_enc = position * div_term
        
        # First fill all even indices
        pe[:, 0::2] = torch.sin(position_enc[:, :d_model//2 + d_model%2])
        # Then fill all odd indices (up to available dimensions)
        pe[:, 1::2] = torch.cos(position_enc[:, :d_model//2])
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class StockPredictor(pl.LightningModule):
    def __init__(self, input_size, total_steps=1000):
        super(StockPredictor, self).__init__()
        self._input_size = input_size  # Store input size as protected attribute
        print(input_size)
        self.save_hyperparameters()
        self.total_steps = total_steps
        self.quantiles = MODEL_PARAMS['quantiles']
        
        # Build model dynamically based on config
        layers = []
        
        # Input layer
        prev_size = input_size
        layers.extend([
            nn.Flatten(),
            nn.Linear(prev_size, prev_size),
            nn.Unflatten(1, (1, prev_size))  # Reshape for attention: [batch, 1, features]
        ])
        
        # Add positional encoding
        layers.append(PositionalEncoding(d_model=prev_size))
        
        # Add attention layer
        attn_config = MODEL_PARAMS['architecture']['attention']
        layers.append(
            MultiHeadAttention(
                input_dim=input_size,
                num_heads=attn_config['num_heads'],
                head_dim=attn_config['head_dim'],
                dropout=attn_config['attention_dropout']
            )
        )
        
        layers.append(nn.Flatten(start_dim=1))  # Flatten after attention, starting from dimension 1
        
        # Hidden layers
        prev_size = input_size  # Keep track of size after flattening
        arch_config = MODEL_PARAMS['architecture']
        for i, hidden_size in enumerate(arch_config['hidden_sizes']):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if arch_config['use_batch_norm'][i]:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(nn.ReLU())
            
            if arch_config['dropout_rates'][i] > 0:
                layers.append(nn.Dropout(p=arch_config['dropout_rates'][i]))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, len(self.quantiles)))
        
        self.model = nn.Sequential(*layers)
        self.loss_fn = QuantileLoss(self.quantiles)

    @property
    def input_size(self):
        """Get the model's input size"""
        return self._input_size

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
            "scheduler": torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr/1000,  # lower learning rate
                max_lr=lr,      # upper learning rate
                step_size_up=200,  # steps per half cycle
                mode='exp_range',  # exponential scaling
                gamma=0.99994,    # decay factor
                cycle_momentum=False  # don't cycle momentum
            ),
            "interval": "step",  # update lr every step
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}