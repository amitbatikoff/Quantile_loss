import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from config import MODEL_PARAMS, OPTIMIZER_PARAMS
import matplotlib.pyplot as plt
import math
import numpy as np  # Added import for numpy

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super(QuantileLoss, self).__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, preds, target, reduction='mean'):
        quantiles = self.quantiles.to(preds.device)
        target = target.expand_as(preds)
        errors = target - preds
        loss = torch.max((quantiles - 1) * errors, quantiles * errors)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            # Average over quantiles only, resulting in per-sample loss (shape: [batch_size])
            return loss.mean(dim=1)
        else:
            return loss

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

class T5Block(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim, feedforward_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.self_attn = MultiHeadAttention(input_dim, num_heads, head_dim, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(input_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.self_attn(self.ln1(x)))
        x = x + self.dropout(self.feedforward(self.ln2(x)))
        return x

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
        if MODEL_PARAMS['architecture']['attention']['use_pos_enc']:
            layers.append(PositionalEncoding(d_model=prev_size))
        
        # Add attention layer
        attn_config = MODEL_PARAMS['architecture']['attention']
        num_blocks = MODEL_PARAMS['architecture']['num_attention_blocks']
        feedforward_dim = MODEL_PARAMS['architecture']['feedforward_dim']
        for _ in range(num_blocks):
            layers.append(
                T5Block(
                    prev_size,
                    attn_config['num_heads'],
                    attn_config['head_dim'],
                    feedforward_dim,
                    attn_config['attention_dropout']
                )
            )
        
        layers.append(nn.Flatten())  # Flatten after attention
        
        # Hidden layers
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
        self.validation_step_outputs = []  # Initialize container for validation outputs

    @property
    def input_size(self):
        """Get the model's input size"""
        return self._input_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        targets = targets.view(-1, 1)
        # Compute per-sample loss
        per_sample_loss = self.loss_fn(outputs, targets, reduction='none')  # shape: [batch]
        
        # # Compute input norms to detect out-of-distribution examples
        # norms = torch.norm(inputs.view(inputs.shape[0], -1), p=2, dim=1)
        # # Get OOD threshold and amplification factor from config if available, otherwise use defaults
        # ood_threshold = MODEL_PARAMS.get('ood_threshold', 10.0)
        # amplification_factor = MODEL_PARAMS.get('ood_amplification', 2.0)
        # # Identify OOD samples and amplify their losses
        # ood_mask = norms > ood_threshold
        
        # # Log number of OOD samples to clearML
        # ood_count = int(ood_mask.sum().item())
        # if hasattr(self, "trainer"):
        #     for logger in self.trainer.loggers:
        #         if hasattr(logger, "task"):
        #             logger.task.get_logger().report_scalar(
        #                 title="Out-of-Distribution Samples",
        #                 series="train",
        #                 iteration=self.global_step,
        #                 value=ood_count
        #             )
        #             break
        
        # per_sample_loss = per_sample_loss * (amplification_factor * ood_mask.float() + (~ood_mask).float())
        # per_sample_loss = per_sample_loss **1.5
        loss = per_sample_loss.mean()
        
        self.log('loss/train', loss, prog_bar=True)
        # Log the learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        targets = targets.view(-1, 1)
        loss = self.loss_fn(outputs, targets)
        per_sample_loss = self.loss_fn(outputs, targets, reduction='none')  # [batch]
        max_loss, max_idx = per_sample_loss.max(dim=0)
        worst_input = inputs[max_idx]
        worst_target = targets[max_idx]
        worst_output = outputs[max_idx]
        self.log('loss/val', loss, prog_bar=True)
        batch_output = {
            "loss": loss,
            "max_loss": max_loss.item(),
            "worst_input": worst_input.detach().cpu(),
            "worst_target": worst_target.detach().cpu(),
            "worst_output": worst_output.detach().cpu()
        }
        self.validation_step_outputs.append(batch_output)
        return batch_output

    def on_validation_epoch_end(self):
        worst_sample = None
        worst_loss = -float('inf')
        for out in self.validation_step_outputs:
            if out["max_loss"] > worst_loss:
                worst_loss = out["max_loss"]
                worst_sample = out
        if worst_sample is not None and hasattr(self, "trainer"):
            for logger in self.trainer.loggers:
                if hasattr(logger, "task"):
                    # Prepare scatter2d data using worst_input values
                    worst_input_np = worst_sample["worst_input"].cpu().numpy().flatten()
                    scatter2d = np.hstack((
                        np.atleast_2d(np.arange(len(worst_input_np))).T,
                        worst_input_np.reshape(-1, 1)
                    ))
                    # Report 2d scatter plot with lines
                    logger.task.get_logger().report_scatter2d(
                        title="Worst Sample Input",
                        series="lines",
                        iteration=self.current_epoch,
                        scatter=scatter2d,
                        xaxis="Index",
                        yaxis="Value"
                    )
                    # Report 2d scatter plot with markers
                    logger.task.get_logger().report_scatter2d(
                        title="Worst Sample Input",
                        series="markers",
                        iteration=self.current_epoch,
                        scatter=scatter2d,
                        xaxis="Index",
                        yaxis="Value",
                        mode='markers'
                    )
                    # Report 2d scatter plot with lines and markers
                    logger.task.get_logger().report_scatter2d(
                        title="Worst Sample Input",
                        series="lines+markers",
                        iteration=self.current_epoch,
                        scatter=scatter2d,
                        xaxis="Index",
                        yaxis="Value",
                        mode='lines+markers'
                    )
                    break
        avg_loss = torch.stack([out["loss"] for out in self.validation_step_outputs]).mean()
        self.log('loss/val_epoch', avg_loss)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        lr = OPTIMIZER_PARAMS['learning_rate']
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=lr * OPTIMIZER_PARAMS['base_lr_factor'],  # lower learning rate
            max_lr=lr,      # upper learning rate
            step_size_up=OPTIMIZER_PARAMS['step_size_up'],  # steps per half cycle
            mode='exp_range',  # exponential scaling
            gamma=OPTIMIZER_PARAMS['gamma'],    # decay factor
            cycle_momentum=OPTIMIZER_PARAMS['cycle_momentum']  # don't cycle momentum
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # update lr every step
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}