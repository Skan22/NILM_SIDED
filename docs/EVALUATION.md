# Evaluation Protocol

## Overview

This document describes the evaluation methodology, metrics, and result interpretation for single-appliance NILM models.

---

## Evaluation Metrics

### 1. MAE (Mean Absolute Error)

**Formula**:
```
MAE = (1/N) * Σ|y_pred - y_true|
```

**Units**: Watts (W) or Megawatts (MW)

**Interpretation**:
- Average absolute prediction error
- Lower is better
- Same units as target variable
- Easy to interpret

**Example**:
```
MAE = 1234.56 W
→ On average, predictions are off by ~1.2 kW
```

### 2. MSE (Mean Squared Error)

**Formula**:
```
MSE = (1/N) * Σ(y_pred - y_true)²
```

**Units**: W² or MW²

**Interpretation**:
- Penalizes large errors more than MAE
- Lower is better
- Sensitive to outliers

### 3. R² (Coefficient of Determination)

**Formula**:
```
R² = 1 - (SS_res / SS_tot)

where:
SS_res = Σ(y_true - y_pred)²
SS_tot = Σ(y_true - mean(y_true))²
```

**Range**: (-∞, 1]

**Interpretation**:
- Proportion of variance explained by model
- 1.0 = perfect predictions
- 0.0 = model as good as mean baseline
- < 0 = model worse than mean baseline

**Example**:
```
R² = 0.85
→ Model explains 85% of variance in target
```

### 4. NDE (Normalized Disaggregation Error)

**Formula**:
```
NDE = Σ|y_true - y_pred| / Σ|y_true|
```

**Range**: [0, ∞)

**Interpretation**:
- Normalized error metric
- 0.0 = perfect predictions
- Lower is better
- Allows comparison across appliances

**Example**:
```
NDE = 0.12
→ Total error is 12% of total actual consumption
```

---

## Evaluation Process

### 1. Model Loading

```python
# Load appliance-specific model
model = TCNModel(input_size=1, output_size=1, ...)
model.load_state_dict(torch.load("TCN_EVSE_best.pth"))
model.to(device)
model.eval()
```

### 2. Prediction

```python
all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        outputs = model(batch_X)
        
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

predictions = np.vstack(all_predictions)
targets = np.vstack(all_targets)
```

### 3. Sanitization

```python
# Sanitize predictions (prevent NaN/Inf)
predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
predictions = np.clip(predictions, -8.0, 8.0)  # Clip in standardized space

targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
targets = np.clip(targets, -8.0, 8.0)
```

**Why?**
- Model outputs can occasionally be extreme
- Clipping in standardized space prevents overflow
- Ensures metrics calculation succeeds

### 4. Inverse Transform

```python
# Transform back to original scale
targets_real = scaler_y.inverse_transform(targets.astype(np.float64))
predictions_real = scaler_y.inverse_transform(predictions.astype(np.float64))
```

### 5. Physical Constraints

```python
# Apply appliance-specific constraints
if appliance_name in ['EVSE', 'CS', 'BA']:  # Loads
    predictions_real = np.maximum(predictions_real, 0)  # Non-negative

elif appliance_name in ['PV', 'CHP']:  # Generation
    predictions_real = np.minimum(predictions_real, 0)  # Non-positive
```

**Why?**
- Loads consume power (≥ 0)
- Generation produces power (≤ 0)
- Enforces physical reality

### 6. Metric Calculation

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae_w = mean_absolute_error(targets_real, predictions_real)
mse_w = mean_squared_error(targets_real, predictions_real)
r2 = r2_score(targets_real, predictions_real)

# NDE
numerator = np.sum(np.abs(targets_real - predictions_real))
denominator = np.sum(np.abs(targets_real))
nde = numerator / denominator if denominator > 0 else float('inf')

# Convert to MW
mae_mw = mae_w / 1e6
mse_mw = mse_w / (1e6 ** 2)
```

---

## Results Format

### Console Output

```
EVSE Metrics:
  MAE: 1234.56 W (0.001235 MW)
  MSE: 5678.90 W² (0.000006 MW²)
  R²:  0.8500
  NDE: 0.1200
```

### JSON Output

```json
{
  "EVSE": {
    "TCN": {
      "MAE_W": 1234.56,
      "MAE_MW": 0.001235,
      "MSE_W": 5678.90,
      "MSE_MW2": 0.000006,
      "R2": 0.8500,
      "NDE": 0.1200
    },
    "ATCN": {
      "MAE_W": 1100.23,
      "MAE_MW": 0.001100,
      "MSE_W": 4500.12,
      "MSE_MW2": 0.000005,
      "R2": 0.8800,
      "NDE": 0.1000
    },
    "LSTM": {
      "MAE_W": 1300.45,
      "MAE_MW": 0.001300,
      "MSE_W": 6000.34,
      "MSE_MW2": 0.000006,
      "R2": 0.8200,
      "NDE": 0.1500
    }
  },
  "PV": { ... },
  "CS": { ... },
  "CHP": { ... },
  "BA": { ... }
}
```

### Summary Table

```
==================================================
SUMMARY: Single-Appliance NILM Results
==================================================
Appliance  | Model | MAE (W)  | MSE (W²) | R²    | NDE
--------------------------------------------------
EVSE       | TCN   | 1234.56  | 5678.90  | 0.85  | 0.12
EVSE       | ATCN  | 1100.23  | 4500.12  | 0.88  | 0.10
EVSE       | LSTM  | 1300.45  | 6000.34  | 0.82  | 0.15
PV         | TCN   | 2345.67  | 7890.12  | 0.82  | 0.14
PV         | ATCN  | 2100.34  | 6500.45  | 0.85  | 0.12
PV         | LSTM  | 2400.56  | 8100.78  | 0.80  | 0.16
CS         | TCN   | 3456.78  | 9012.34  | 0.80  | 0.16
...
```

---

## Interpretation Guidelines

### MAE Interpretation

| MAE (W) | Quality | Interpretation |
|---------|---------|----------------|
| < 500 | Excellent | Very accurate predictions |
| 500-1000 | Good | Acceptable for most applications |
| 1000-2000 | Fair | Usable but room for improvement |
| > 2000 | Poor | Needs model/data improvement |

### R² Interpretation

| R² | Quality | Interpretation |
|----|---------|----------------|
| > 0.9 | Excellent | Model captures variance very well |
| 0.8-0.9 | Good | Strong predictive power |
| 0.6-0.8 | Fair | Moderate predictive power |
| < 0.6 | Poor | Weak predictive power |

### NDE Interpretation

| NDE | Quality | Interpretation |
|-----|---------|----------------|
| < 0.1 | Excellent | Total error < 10% of consumption |
| 0.1-0.2 | Good | Acceptable disaggregation quality |
| 0.2-0.3 | Fair | Moderate disaggregation quality |
| > 0.3 | Poor | Poor disaggregation quality |

---

## Comparison Across Models

### Best Model Selection

For each appliance, select the model with:
1. **Lowest MAE** (primary criterion)
2. **Highest R²** (secondary criterion)
3. **Lowest NDE** (tertiary criterion)

**Example**:
```
EVSE Results:
- TCN:  MAE=1234.56, R²=0.85, NDE=0.12
- ATCN: MAE=1100.23, R²=0.88, NDE=0.10  ← Best (lowest MAE)
- LSTM: MAE=1300.45, R²=0.82, NDE=0.15
```

### Comparison Across Appliances

**Expected Performance Order** (easiest to hardest):
1. **BA** (Building Automation) - Predictable patterns
2. **CS** (Cooling System) - Temperature-dependent
3. **EVSE** (EV Charging) - Scheduled charging
4. **CHP** (Combined Heat & Power) - Complex operation
5. **PV** (Photovoltaic) - Weather-dependent

---

## Visualization

### Prediction Plots

```python
import matplotlib.pyplot as plt

def plot_predictions(targets, predictions, appliance_name, save_path=None):
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Plot subset (first 1000 points)
    subset = 1000
    ax.plot(targets[:subset], label='Actual', alpha=0.7)
    ax.plot(predictions[:subset], label='Predicted', alpha=0.7, linestyle='--')
    
    ax.set_title(f'{appliance_name} - Actual vs Predicted')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Power (W)')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
```

### Error Distribution

```python
def plot_error_distribution(targets, predictions, appliance_name):
    errors = predictions - targets
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax1.hist(errors, bins=50, edgecolor='black')
    ax1.set_title(f'{appliance_name} - Error Distribution')
    ax1.set_xlabel('Error (W)')
    ax1.set_ylabel('Frequency')
    
    # Scatter plot
    ax2.scatter(targets, predictions, alpha=0.3)
    ax2.plot([targets.min(), targets.max()], 
             [targets.min(), targets.max()], 
             'r--', label='Perfect Prediction')
    ax2.set_title(f'{appliance_name} - Predictions vs Actual')
    ax2.set_xlabel('Actual (W)')
    ax2.set_ylabel('Predicted (W)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

---

## Saved Outputs

### 1. Model Checkpoints

```
TCN_EVSE_best.pth
TCN_PV_best.pth
...
(15 files total)
```

### 2. Predictions

```
TCN_EVSE_predictions.npz
  - predictions: (N, 1) array
  - targets: (N, 1) array
```

### 3. Metrics

```
all_results_single_appliance.json
  - Comprehensive metrics for all models and appliances
```

---

## Validation Checklist

Before accepting results, verify:

- [ ] No NaN values in predictions
- [ ] No Inf values in predictions
- [ ] Physical constraints satisfied (loads ≥ 0, generation ≤ 0)
- [ ] R² > 0 (model better than mean baseline)
- [ ] MAE reasonable for appliance power range
- [ ] Results saved correctly
- [ ] Predictions match expected patterns

---

## Troubleshooting

### Issue: R² < 0

**Cause**: Model worse than mean baseline

**Solutions**:
- Check data quality
- Verify model architecture
- Increase training epochs
- Check for data leakage

### Issue: Very High MAE

**Cause**: Poor predictions or scaling issues

**Solutions**:
- Verify scaler is fitted correctly
- Check inverse transform
- Inspect prediction distribution
- Validate data preprocessing

### Issue: NaN in Metrics

**Cause**: NaN/Inf in predictions or targets

**Solutions**:
- Check sanitization step
- Verify clipping ranges
- Inspect model outputs
- Review data quality

---

## Summary

✅ **Four metrics**: MAE, MSE, R², NDE  
✅ **Sanitization** prevents NaN/Inf errors  
✅ **Physical constraints** enforce reality  
✅ **Comprehensive results** in JSON format  
✅ **Visualization** for interpretation  
✅ **Validation checklist** ensures quality  

The evaluation protocol ensures robust, interpretable results that can be compared across models and appliances.
