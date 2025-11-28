# Data Pipeline and Preprocessing

## Overview

This document describes the complete data pipeline from raw CSV files to model-ready tensors, including comprehensive missing value handling.

---

## Pipeline Stages

```
Raw CSV Files
      ↓
Data Loading & Quality Checks
      ↓
Resampling (1min → 5min)
      ↓
Missing Value Handling
      ↓
Single-Appliance Extraction
      ↓
RobustScaler Normalization
      ↓
Sequence Creation (S2P)
      ↓
PyTorch Tensors
      ↓
DataLoaders
```

---

## Stage 1: Data Loading

### Function: `load_data_by_location()`

**Purpose**: Load and split data by geographic location for domain adaptation.

**Configuration**:
```python
load_data_by_location(
    base_path='./AMDA_SIDED',
    target_locations=['Tokyo'],      # Test set (target domain)
    source_locations=['LA', 'Offenbach'],  # Train set (source domain)
    resample_rule='5min'
)
```

**Process**:
1. Load CSV files from 3 facilities × 3 locations
2. Separate into source (train) and target (test) domains
3. Resample from 1-minute to 5-minute intervals
4. Handle missing values (see below)
5. Concatenate into train/test dataframes

**Output**:
- `train_df`: Source domain data (LA + Offenbach)
- `test_df`: Target domain data (Tokyo)

---

## Stage 2: Missing Value Handling

### Three-Layer Defense System

#### Layer 1: Data Loading

**Location**: `load_data_by_location()` in `dataset.py`

**Strategy**:
```python
# 1. Check BEFORE resampling
missing_before = df[appliance_columns].isnull().sum().sum()

if missing_before > 0:
    print(f"⚠️ Found {missing_before} missing values")
    
    # Forward fill: Propagate last valid observation
    df = df.fillna(method='ffill')
    
    # Backward fill: Fill leading NaN
    df = df.fillna(method='bfill')
    
    # Zero fill: If entire column is NaN
    df = df.fillna(0)

# 2. Check AFTER resampling
missing_after = df[appliance_columns].isnull().sum().sum()
if missing_after > 0:
    df[appliance_columns] = df[appliance_columns].fillna(0)

# 3. Check for inf values
inf_count = np.isinf(df[appliance_columns]).sum().sum()
if inf_count > 0:
    df[appliance_columns] = df[appliance_columns].replace([np.inf, -np.inf], np.nan)
    df[appliance_columns] = df[appliance_columns].fillna(0)
```

**Why This Order?**
1. **Forward fill**: Handles most gaps (assumes temporal continuity)
2. **Backward fill**: Handles leading NaN values
3. **Zero fill**: Handles edge case where entire column is missing

**Data Quality Report**:
```
============================================================
Data Quality Check:
============================================================
Training Data:
  Total Samples: 45000
  NaN values: 0
  Inf values: 0
Testing Data:
  Total Samples: 15000
  NaN values: 0
  Inf values: 0
============================================================
```

#### Layer 2: Preprocessing

**Location**: `preprocess_single_appliance()` in `reproduce_paper.py`

**Strategy**:
```python
# Check each array individually
for name, arr in [("X_train", X_train_raw), ("y_train", y_train_raw), 
                   ("X_test", X_test_raw), ("y_test", y_test_raw)]:
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    
    if nan_count > 0 or inf_count > 0:
        print(f"⚠️ {appliance_name} - {name}: {nan_count} NaN, {inf_count} Inf")
        
        # Replace inf with nan
        arr = np.where(np.isinf(arr), np.nan, arr)
        
        # Replace all NaN with 0
        arr = np.nan_to_num(arr, nan=0.0)
```

**Why This Matters**:
- Even if data loading is clean, individual appliances might have issues
- Catches problems **before** RobustScaler (which can't handle NaN/inf)
- Provides appliance-specific diagnostics

#### Layer 3: Evaluation

**Location**: `calculate_single_appliance_metrics()` in `reproduce_paper.py`

**Strategy**:
```python
# Sanitize predictions before inverse transform
predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
predictions = np.clip(predictions, -8.0, 8.0)  # Clip in standardized space

targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
targets = np.clip(targets, -8.0, 8.0)

# Inverse transform to original scale (safely)
targets_real = scaler_y.inverse_transform(targets.astype(np.float64))
predictions_real = scaler_y.inverse_transform(predictions.astype(np.float64))
```

**Why This Matters**:
- Model outputs can occasionally produce extreme values
- Clipping in standardized space prevents overflow during inverse transform
- Final sanitization ensures metrics calculation doesn't fail

---

## Stage 3: Resampling

### From 1-Minute to 5-Minute Intervals

**Paper Requirement** (Section V-B):
> *"For the current application, the one-minute resolution of the original data in the SIDED dataset is not necessary and we re-sample all time series to a sampling period of T_s=5 min."*

**Implementation**:
```python
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').resample('5min').mean().dropna().reset_index()
```

**Effect**:
- Reduces data size by 5×
- Reduces computational cost
- Maintains temporal patterns

**Example**:
```
Before: 1440 samples/day (1-min intervals)
After:  288 samples/day (5-min intervals)
```

---

## Stage 4: Single-Appliance Extraction

### Function: `preprocess_single_appliance()`

**Purpose**: Extract and normalize data for ONE appliance at a time.

**Process**:
```python
def preprocess_single_appliance(train_df, test_df, appliance_name):
    # Extract aggregate power (input)
    X_train_raw = train_df['Aggregate'].values.reshape(-1, 1)
    X_test_raw = test_df['Aggregate'].values.reshape(-1, 1)
    
    # Extract single appliance power (output)
    y_train_raw = train_df[appliance_name].values.reshape(-1, 1)  # Single column!
    y_test_raw = test_df[appliance_name].values.reshape(-1, 1)
    
    # Handle missing/inf values (Layer 2)
    # ... (see above)
    
    # Normalize using RobustScaler
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()  # Separate scaler per appliance!
    
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)
    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y
```

**Key Points**:
- Each appliance gets its own scaler
- Fit scaler ONLY on training data (prevents data leakage)
- Output is 1D (single appliance), not 5D

---

## Stage 5: Normalization

### RobustScaler

**Paper Requirement** (Section V-B):
> *"Before feeding the data to any model, we normalize it for each building configuration with a RobustScaler."*

**Why RobustScaler?**
- Robust to outliers (uses median and IQR)
- Better for industrial data with extreme values
- Prevents influence of anomalies

**Formula**:
```
X_scaled = (X - median(X)) / IQR(X)
```

where IQR = Q3 - Q1 (interquartile range)

**Separate Scalers Per Appliance**:
```python
# EVSE scaler (fitted on EVSE data)
scaler_evse = RobustScaler()
y_evse_scaled = scaler_evse.fit_transform(y_evse_raw)

# PV scaler (fitted on PV data)
scaler_pv = RobustScaler()
y_pv_scaled = scaler_pv.fit_transform(y_pv_raw)

# Different scalers because different power ranges!
```

**Why Separate?**
- EVSE: 0-100 kW (load, positive)
- PV: -50-0 kW (generation, negative)
- Different distributions require different scaling

---

## Stage 6: Sequence Creation

### Function: `create_sequences()`

**Purpose**: Create sliding window sequences for Sequence-to-Point (S2P) models.

**Parameters**:
```python
create_sequences(
    X, y,
    seq_length=288,      # 24 hours at 5-min intervals
    stride=5,            # 25 minutes (training)
    target_pos='mid'     # Target at window midpoint
)
```

**Process**:
```python
def create_sequences(X, y, seq_length, stride=1, target_pos='mid'):
    X_seq, y_seq = [], []
    N = len(X)
    
    for i in range(0, N - seq_length, stride):
        start = i
        end = i + seq_length
        
        # Input: entire window
        X_seq.append(X[start:end])
        
        # Target: midpoint of window
        if target_pos == 'mid':
            t_idx = start + seq_length // 2
        
        y_seq.append(y[t_idx])
    
    return np.array(X_seq), np.array(y_seq)
```

**Example**:
```
Input sequence:  [x_0, x_1, ..., x_287]  (288 timesteps)
Target:          y_144                   (midpoint)

Next sequence:   [x_5, x_6, ..., x_292]  (stride=5)
Target:          y_149
```

**Training vs Evaluation**:
- **Training**: `stride=5` (25 min) - reduces overlap, faster training
- **Evaluation**: `stride=1` (5 min) - dense predictions, better metrics

---

## Stage 7: PyTorch Tensors

### Conversion

```python
# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_seq)
y_train_tensor = torch.FloatTensor(y_train_seq)

X_test_tensor = torch.FloatTensor(X_test_seq)
y_test_tensor = torch.FloatTensor(y_test_seq)
```

**Shapes**:
```
X_train_tensor: (num_sequences, seq_length, input_size)
                (M, 288, 1)

y_train_tensor: (num_sequences, output_size)
                (M, 1)
```

---

## Stage 8: DataLoaders

### Creation

```python
from torch.utils.data import DataLoader

# Create dataset
train_dataset = NILMDataset(X_train_tensor, y_train_tensor)
test_dataset = NILMDataset(X_test_tensor, y_test_tensor)

# Split validation set
val_size = int(len(train_dataset) * 0.1)
train_size = len(train_dataset) - val_size
train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

# Create loaders
train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True,
    num_workers=0,      # Windows compatibility
    pin_memory=True     # Faster GPU transfer
)

val_loader = DataLoader(
    val_subset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
```

---

## Data Quality Checks

### What Gets Checked

| Stage | Check | Action |
|-------|-------|--------|
| **CSV Loading** | NaN in columns | Forward/backward fill → Zero fill |
| **After Resampling** | NaN from aggregation | Zero fill |
| **Inf Detection** | ±Inf values | Replace with NaN → Zero fill |
| **Preprocessing** | Per-appliance NaN/Inf | `np.nan_to_num` |
| **Evaluation** | Extreme values | Clip to [-8, 8] |

### Console Output

```
Loading Data from: ./AMDA_SIDED
⚠️ Resampling data to 5min intervals (Paper Requirement)
  [TRAIN/Source] Loaded Dealer_LA: 8640 samples
  ⚠️ Found 150 missing values in Office_LA
  ✅ Missing values handled via forward/backward fill
  [TRAIN/Source] Loaded Office_LA: 8640 samples

============================================================
Data Quality Check:
============================================================
Training Data:
  Total Samples: 45000
  NaN values: 0
  Inf values: 0
Testing Data:
  Total Samples: 15000
  NaN values: 0
  Inf values: 0
============================================================
```

---

## Error Prevention

### Before (No Handling)

```python
# RobustScaler fails with NaN
scaler.fit_transform(data_with_nan)
# ValueError: Input contains NaN

# Metrics fail with inf
mean_absolute_error(targets, predictions_with_inf)
# RuntimeWarning: overflow encountered
```

### After (With Handling)

```python
# Data is cleaned before scaling
data_clean = handle_missing_values(data)
scaler.fit_transform(data_clean)  # ✅ Works

# Predictions are sanitized before metrics
predictions_clean = sanitize(predictions)
mean_absolute_error(targets, predictions_clean)  # ✅ Works
```

---

## Summary

✅ **Three-layer missing value handling** prevents training failures  
✅ **Resampling to 5-min intervals** as per paper  
✅ **Separate RobustScaler per appliance** for proper normalization  
✅ **Sequence-to-Point (S2P)** with 24-hour windows  
✅ **Data quality reporting** for transparency  
✅ **No data leakage** (fit scalers on train only)  

The pipeline ensures clean, properly formatted data ready for model training while maintaining paper compliance.
