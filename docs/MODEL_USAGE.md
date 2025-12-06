# Model Usage Guide

This document details exactly how to use the trained NILM models, specifying the precise input and output formats.

## 1. Model Overview

The models (TCN, ATCN, LSTM) are designed for **Sequence-to-Point (S2P)** disaggregation.
*   **Input**: A window of aggregate power readings (e.g., 24 hours).
*   **Input**: A window of aggregate power readings (e.g., 24 hours).
*   **Output**: The estimated power consumption of a specific appliance at the **midpoint** of that window (Midpoint Prediction).

## 2. Input Specification

### Raw Input
*   **Data**: Aggregate power consumption (Main meter).
*   **Unit**: Watts (W).
*   **Sampling Rate**: 5 minutes (resampled from original 1-min data).

### Preprocessing Steps
Before passing data to the model, it must be processed as follows:

1.  **Scaling**:
    *   The raw aggregate power ($X$) is normalized using a **RobustScaler**.
    *   $$ X_{scaled} = \frac{X - \text{median}(X)}{\text{IQR}(X)} $$
    *   *Note: The scaler must be fit ONLY on the training data.*

2.  **Windowing (Sliding Window)**:
    *   **Sequence Length**: 288 time steps (equivalent to 24 hours at 5-min intervals).
    *   **Stride**:
        *   Training: 5 steps (25 mins) to reduce redundancy.
        *   Inference: 1 step (5 mins) for dense predictions.

### Final Model Input Tensor
*   **Shape**: `(Batch_Size, Sequence_Length, Channels)`
*   **Dimensions**: `(N, 288, 1)`
*   **Data Type**: `torch.float32`

---

## 3. Output Specification

### Raw Model Output
*   **Shape**: `(Batch_Size, 1)`
*   **Data Type**: `torch.float32`
*   **Value**: This is a **normalized** scalar value representing the appliance power at the window's midpoint.

### Postprocessing (Getting Watts)
To convert the model's output back to interpretable power readings (Watts):

1.  **Inverse Scaling**:
    *   Apply the inverse of the target scaler (StandardScaler) used during training.
    *   $$ \hat{y}_{watts} = (\hat{y}_{scaled} \times \sigma_{appliance}) + \mu_{appliance} $$
    *   *Where $\mu$ and $\sigma$ are the mean and std of the appliance power in the training set.*

2.  **Physical Constraints (Sanitization)**:
    *   **Loads (EVSE, CS, BA)**: Apply ReLU or `max(0, x)`. Negative power is impossible for loads.
    *   **Generation (PV, CHP)**: Apply `min(0, x)`. Generation is typically represented as negative or handled separately (check specific appliance logic).
    *   **Clamping**: Remove extreme outliers (e.g., > 100kW) if necessary.

### Final Output
*   **Value**: Estimated power consumption of the target appliance.
*   **Unit**: Watts (W).
*   **Timepoint**: Corresponds to the timestamp at $t$ (center of the window).

## 4. Example Usage (Python)

```python
import torch
import numpy as np

# 1. Prepare Input (Batch of 1 window)
# window_data: numpy array of shape (288,) containing aggregate power in Watts
window_scaled = scaler_X.transform(window_data.reshape(-1, 1))
input_tensor = torch.FloatTensor(window_scaled).unsqueeze(0) # Shape: (1, 288, 1)

# 2. Inference
model.eval()
with torch.inference_mode():
    prediction_scaled = model(input_tensor) # Shape: (1, 1)

# 3. Postprocessing
prediction_watts = scaler_y.inverse_transform(prediction_scaled.numpy())
prediction_watts = np.maximum(prediction_watts, 0) # Enforce non-negative constraint

print(f"Predicted Appliance Power: {prediction_watts.item():.2f} W")
```
