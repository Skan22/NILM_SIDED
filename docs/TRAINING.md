# Training Procedure

## Overview

This document describes the complete training procedure for single-appliance NILM models, including hyperparameters, optimization, and training loop.

---

## Training Configuration

### Hyperparameters (from paper)

```python
CONFIG = {
    # Data
    'seq_length': 288,        # 24h window at 5-min intervals
    'train_stride': 5,        # 25 min stride for training
    'eval_stride': 1,         # Dense evaluation
    
    # Training
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 20,
    'warmup_epochs': 3,
    'min_lr': 1e-6,
    'early_stopping_patience': 5,
    
    # Model
    'dropout': 0.33,
    'tcn_layers': [128] * 8,  # 8 layers of 128 channels
    'lstm_hidden': 128,
    'lstm_layers': 3,
    'transformer_dmodel': 128,
    'transformer_nhead': 4,
    'transformer_layers': 3,
    
    # System
    'num_workers': 0,         # Windows compatibility
    'pin_memory': True
}
```

### Fair Comparison Configuration (`reproduce_paper_fair.py`)

To ensure a scientifically valid comparison, we provide a configuration where all models are matched to **~1.05M parameters**:

```python
CONFIG_FAIR = {
    # TCN (Unchanged Baseline)
    'tcn_layers': [128] * 8,  # ~1.05M params
    
    # ATCN (Reduced)
    'atcn_layers': [100] * 8, # ~1.05M params (reduced from 128)
    
    # LSTM (Increased)
    'lstm_hidden': 356,       # ~1.05M params (increased from 128)
    'lstm_layers': 5          # Increased from 3
}
```

---

## Model Architectures

### 1. TCN (Temporal Convolutional Network)

```python
TCNModel(
    input_size=1,
    num_channels=[128]*8,     # 8 layers
    dropout=0.33,
    output_size=1,
    causal=False              # Non-Causal
)
```

**Architecture**:
- 8 temporal convolutional blocks with **TemporalLayerNorm**
- Exponentially increasing dilation
- **Non-Causal**: Uses standard padding to see future context
- Global pooling/Midpoint selection

### 2. ATCN (Attention-TCN)

```python
ATCNModel(
    input_size=1,
    num_channels=[128]*8,
    dropout=0.33,
    output_size=1,
    causal=False
)
```

**Architecture**:
- Base: Non-Causal TCN
- **Attention**: Multi-Head Self-Attention (4 heads)
- Captures complex dependencies and transients

### 3. BiLSTM

```python
LSTMModel(
    input_size=1,
    hidden_size=128,
    num_layers=3,
    output_size=1,
    bidirectional=True
)
```

**Architecture**:
- 3 Bidirectional LSTM layers
- Input moves in both directions (past->future, future->past)
- Concatenates hidden states (256 dims total)

### 4. Transformer

```python
TransformerModel(
    input_size=1,
    d_model=128,
    nhead=4,
    num_layers=3
)
```

**Architecture**:
- Positional Encoding
- 3 Transformer Encoder Layers
- Self-Attention mechanism
- State-of-the-art sequence modeling

---

## Optimization

### Optimizer

**Adam** with learning rate 0.001:

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Why Adam?**
- Adaptive learning rates
- Momentum-based updates
- Works well for time series

### Learning Rate Schedule

**Warmup + Cosine Annealing**:

```python
# Warmup: Linear increase for first 3 epochs
warmup_lambda = lambda epoch: (epoch + 1) / 3 if epoch < 3 else 1.0
warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

# Cosine Annealing: Smooth decay after warmup
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=17,           # 20 - 3 warmup epochs
    eta_min=1e-6
)
```

**Schedule**:
```
Epoch 1: LR = 0.000333  (warmup)
Epoch 2: LR = 0.000667  (warmup)
Epoch 3: LR = 0.001000  (warmup)
Epoch 4: LR = 0.000999  (cosine)
Epoch 5: LR = 0.000995  (cosine)
...
Epoch 20: LR = 0.000001 (min_lr)
```

**Benefits**:
- Warmup prevents early instability
- Cosine decay improves convergence
- Smooth learning rate changes

---

## Training Loop

### Per-Appliance Training

```python
for appliance_name in ['EVSE', 'PV', 'CS', 'CHP', 'BA']:
    print(f"Training models for {appliance_name}")
    
    # 1. Preprocess data for THIS appliance
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = \
        preprocess_single_appliance(train_df, test_df, appliance_name)
    
    # 2. Create sequences
    X_train_seq, y_train_seq = create_sequences(
        X_train, y_train,
        seq_length=288,
        stride=5
    )
    
    # 3. Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_seq, y_train_seq, X_test_seq, y_test_seq
    )
    
    # 4. Train each architecture
    for model_name in ['TCN', 'ATCN', 'LSTM']:
        model = create_model(model_name)
        train_single_model(model, train_loader, val_loader, appliance_name)
```

### Single Model Training

```python
def train_single_appliance_model(model, train_loader, val_loader, criterion,
                                  optimizer, scheduler, num_epochs, device,
                                  early_stopping_patience, model_name, appliance_name):
    
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model
```

---

## Loss Function

### MSE Loss

```python
criterion = nn.MSELoss()
```

**Why MSE?**
- Standard for regression tasks
- Penalizes large errors more
- Differentiable for backpropagation

**Formula**:
```
MSE = (1/N) * Σ(y_pred - y_true)²
```

---

## Regularization

### 1. Dropout

**TCN/ATCN**: 0.33 dropout rate
```python
nn.Dropout(0.33)
```

**LSTM**: 0.2 dropout between layers
```python
nn.LSTM(..., dropout=0.2)
```

### 2. Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why?**
- Prevents exploding gradients
- Stabilizes training
- Especially important for RNNs

### 3. Early Stopping

```python
patience = 5  # Stop if no improvement for 5 epochs
```

**Why?**
- Prevents overfitting
- Saves training time
- Automatically finds optimal epoch

---

## Validation Split

### 10% Validation Set

```python
val_size = int(len(train_dataset) * 0.1)
train_size = len(train_dataset) - val_size

train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)
```

**Purpose**:
- Monitor overfitting
- Early stopping criterion
- Hyperparameter tuning

---

## Training Time

### Expected Duration

| Configuration | Time |
|---------------|------|
| **Single model** | ~20-40 minutes |
| **Single appliance (3 models)** | ~1-2 hours |
| **All 15 models** | ~5-10 hours |

**Factors**:
- GPU vs CPU
- Dataset size
- Early stopping

---

## Console Output

### Training Progress

```
################################################################################
# APPLIANCE: EVSE
################################################################################

Preprocessing data for EVSE...
Creating sequences for EVSE...

--------------------------------------------------------------------------------
Training TCN for EVSE
--------------------------------------------------------------------------------

Epoch 1/20 | Train Loss: 0.123456 | Val Loss: 0.098765 | LR: 0.000333
Epoch 2/20 | Train Loss: 0.098765 | Val Loss: 0.087654 | LR: 0.000667
Epoch 3/20 | Train Loss: 0.087654 | Val Loss: 0.076543 | LR: 0.001000
Epoch 4/20 | Train Loss: 0.076543 | Val Loss: 0.065432 | LR: 0.000999
Epoch 5/20 | Train Loss: 0.065432 | Val Loss: 0.054321 | LR: 0.000995
...
Epoch 15/20 | Train Loss: 0.012345 | Val Loss: 0.013456 | LR: 0.000234
Epoch 16/20 | Train Loss: 0.011234 | Val Loss: 0.013567 | LR: 0.000156
Early stopping at epoch 16

Evaluating TCN on EVSE...

EVSE Metrics:
  MAE: 1234.56 W (0.001235 MW)
  MSE: 5678.90 W² (0.000006 MW²)
  R²:  0.8500
  NDE: 0.1200

💾 Saved model to TCN_EVSE_best.pth
💾 Saved predictions to TCN_EVSE_predictions.npz
```

---

## Best Practices

### 1. GPU Usage

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

**Benefits**:
- 10-50× faster training
- Larger batch sizes possible

### 2. Mixed Precision (Optional)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- Faster training
- Lower memory usage
- Minimal accuracy loss

### 3. Reproducibility

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

---

## Troubleshooting

### Issue: NaN Loss

**Causes**:
- Learning rate too high
- Gradient explosion
- Bad initialization

**Solutions**:
- Reduce learning rate
- Add gradient clipping
- Check data for NaN/Inf

### Issue: No Convergence

**Causes**:
- Learning rate too low
- Model too small
- Insufficient data

**Solutions**:
- Increase learning rate
- Increase model capacity
- Check data quality

### Issue: Overfitting

**Symptoms**:
- Train loss << Val loss
- High variance in predictions

**Solutions**:
- Increase dropout
- Add more data augmentation
- Reduce model size
- Early stopping (already implemented)

---

## Summary

✅ **Adam optimizer** with LR=0.001  
✅ **Warmup + Cosine Annealing** schedule  
✅ **MSE loss** for regression  
✅ **Gradient clipping** for stability  
✅ **Early stopping** to prevent overfitting  
✅ **10% validation split** for monitoring  
✅ **~5-10 hours** total training time  

The training procedure follows the paper's specifications and includes best practices for stable, efficient training.
