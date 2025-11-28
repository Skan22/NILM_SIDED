# Single-Appliance NILM Methodology

## Overview

This document explains the **single-appliance NILM approach** implemented in this repository, as described in the paper.

---

## What is Single-Appliance NILM?

### Paper Quote (Section V-A)

> *"We employ the **single-appliance NILM setting** where the input is the aggregated power signal and the model extracts the power of a **single appliance** e.g., the CHP."*

### Definition

**Single-Appliance NILM**: Train separate models where each model learns to disaggregate **one specific appliance** from the aggregate power signal.

---

## Architecture Comparison

### ❌ Multi-Output NILM (NOT Used)

```
Input: Aggregate Power [1D]
         │
    ┌────▼────┐
    │  Model  │
    └────┬────┘
         │
Output: [EVSE, PV, CS, CHP, BA]  ← All 5 appliances

Total Models: 3 (one per architecture)
```

**Problems**:
- Model must learn shared features for all appliances
- Cannot specialize for individual appliance patterns
- Difficult to add/remove appliances
- Lower accuracy

### ✅ Single-Appliance NILM (Used in This Implementation)

```
FOR EACH APPLIANCE:
    Input: Aggregate Power [1D]
             │
        ┌────▼────┐
        │  Model  │
        └────┬────┘
             │
    Output: [EVSE]  ← Single appliance only

Total Models: 15 (3 architectures × 5 appliances)
```

**Benefits**:
- Each model specializes in one appliance
- Learns appliance-specific temporal patterns
- Easy to add/remove appliances
- Higher accuracy through specialization

---

## Implementation Details

### Training Loop Structure

```python
for appliance in ['EVSE', 'PV', 'CS', 'CHP', 'BA']:  # 5 appliances
    # Preprocess data for THIS appliance only
    X_train, y_train = preprocess_single_appliance(train_df, test_df, appliance)
    
    for model_name in ['TCN', 'ATCN', 'LSTM']:  # 3 architectures
        # Create model with 1 output neuron
        model = create_model(input_size=1, output_size=1)
        
        # Train on single-appliance data
        train(model, X_train, y_train)
        
        # Save appliance-specific model
        save(f"{model_name}_{appliance}_best.pth")

# Result: 15 specialized models
```

### Key Differences from Multi-Output

| Aspect | Multi-Output | Single-Appliance |
|--------|--------------|------------------|
| **Output Neurons** | 5 | 1 |
| **Target Data** | All 5 appliances | One appliance |
| **Scaler** | 1 shared | 5 separate |
| **Models Trained** | 3 | 15 |
| **Specialization** | Low | High |

---

## Data Preprocessing

### Separate Scalers Per Appliance

Each appliance gets its own `RobustScaler`:

```python
def preprocess_single_appliance(train_df, test_df, appliance_name):
    # Extract data for ONE appliance
    X_train = train_df['Aggregate'].values.reshape(-1, 1)
    y_train = train_df[appliance_name].values.reshape(-1, 1)  # Single column!
    
    # Separate scaler for this appliance
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()  # Fitted on THIS appliance's data only
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    return X_train_scaled, y_train_scaled, scaler_X, scaler_y
```

**Why Separate Scalers?**
- Each appliance has different power ranges
- EVSE: 0-100 kW (load)
- PV: -50-0 kW (generation)
- Separate scaling preserves appliance-specific distributions

---

## Model Architecture

### Output Layer Configuration

```python
# TCN Model
TCNModel(
    input_size=1,      # Aggregate power
    output_size=1,     # Single appliance (NOT 5!)
    num_channels=[128]*8,
    dropout=0.33
)

# LSTM Model
LSTMModel(
    input_size=1,
    output_size=1,     # Single appliance (NOT 5!)
    hidden_size=128,
    num_layers=3
)
```

### Sequence-to-Point (S2P)

Input: Window of aggregate power readings  
Output: Single appliance power at window midpoint

```
Window: [x_{t-144}, ..., x_t, ..., x_{t+144}]  (288 timesteps = 24 hours)
                            ↓
                         Model
                            ↓
Output: y_t  (single appliance power at time t)
```

---

## Evaluation

### Per-Appliance Metrics

Each model is evaluated independently:

```python
# Load appliance-specific model
model = load_model("TCN_EVSE_best.pth")

# Predict on test data
predictions = model(X_test)  # Shape: (N, 1)

# Calculate metrics for EVSE only
mae = mean_absolute_error(y_test_evse, predictions)
r2 = r2_score(y_test_evse, predictions)
```

### Physical Constraints

Applied during evaluation based on appliance type:

```python
# Loads (consume power) - must be non-negative
if appliance in ['EVSE', 'CS', 'BA']:
    predictions = np.maximum(predictions, 0)

# Generation (produce power) - must be non-positive
elif appliance in ['PV', 'CHP']:
    predictions = np.minimum(predictions, 0)
```

---

## Advantages

### 1. Specialization
Each model learns unique patterns:
- **EVSE**: Charging patterns, time-of-day usage
- **PV**: Solar generation curves, weather dependence
- **CS**: Cooling cycles, temperature correlation

### 2. Better Performance
- Focused learning on one appliance
- No interference from other appliances
- Appliance-specific feature extraction

### 3. Flexibility
- Add new appliance: Train 3 new models (one per architecture)
- Remove appliance: Delete 3 models
- No need to retrain all models

### 4. Interpretability
- Clear which model predicts which appliance
- Easy to debug appliance-specific issues
- Transparent prediction pipeline

### 5. Paper Compliance
- Exactly matches paper methodology
- Reproducible results
- Valid comparison with paper benchmarks

---

## Disadvantages

### 1. More Models to Train
- 15 models instead of 3
- Longer total training time (~5-10 hours vs ~1-2 hours)

### 2. More Storage
- 15 model files (~250 MB) vs 3 files (~50 MB)

### 3. Slower Inference
- Need 5 forward passes (one per appliance)
- vs 1 forward pass for multi-output

**Trade-off**: Higher accuracy and paper compliance vs computational cost

---

## Results Structure

### Model Files

```
TCN_EVSE_best.pth     ← TCN specialized for EVSE
TCN_PV_best.pth       ← TCN specialized for PV
TCN_CS_best.pth       ← TCN specialized for CS
TCN_CHP_best.pth      ← TCN specialized for CHP
TCN_BA_best.pth       ← TCN specialized for BA

ATCN_EVSE_best.pth    ← ATCN specialized for EVSE
... (and so on)

Total: 15 model files
```

### Predictions

```
TCN_EVSE_predictions.npz   ← Predictions for EVSE using TCN
TCN_PV_predictions.npz     ← Predictions for PV using TCN
... (and so on)

Total: 15 prediction files
```

### Metrics

```json
{
  "EVSE": {
    "TCN": {"MAE_W": 1234.56, "R2": 0.85, "NDE": 0.12},
    "ATCN": {"MAE_W": 1100.23, "R2": 0.88, "NDE": 0.10},
    "LSTM": {"MAE_W": 1300.45, "R2": 0.82, "NDE": 0.15}
  },
  "PV": { ... },
  ...
}
```

---

## Inference Example

### Multi-Output Approach (NOT Used)

```python
# One forward pass gets all appliances
model = load_model("TCN_best.pth")
predictions = model(aggregate_signal)
# predictions = [EVSE: 20, PV: -15, CS: 30, CHP: -10, BA: 75]
```

### Single-Appliance Approach (Used)

```python
# Need 5 forward passes (one per appliance)
model_evse = load_model("TCN_EVSE_best.pth")
model_pv = load_model("TCN_PV_best.pth")
model_cs = load_model("TCN_CS_best.pth")
model_chp = load_model("TCN_CHP_best.pth")
model_ba = load_model("TCN_BA_best.pth")

evse_pred = model_evse(aggregate_signal)  # [20]
pv_pred = model_pv(aggregate_signal)      # [-15]
cs_pred = model_cs(aggregate_signal)      # [30]
chp_pred = model_chp(aggregate_signal)    # [-10]
ba_pred = model_ba(aggregate_signal)      # [75]

# Each model is a specialist
```

---

## Summary

✅ **Single-appliance NILM** trains separate models for each appliance  
✅ **15 models total** (3 architectures × 5 appliances)  
✅ **1 output neuron** per model (not 5)  
✅ **Separate RobustScaler** per appliance  
✅ **Higher accuracy** through specialization  
✅ **Paper-compliant** methodology  

This approach exactly matches the paper's description and provides better performance through model specialization.
