# NILM Model API Contract

This document defines the JSON format for the input and output of the NILM (Non-Intrusive Load Monitoring) AI model for backend integration.

## 1. Overview
The AI model acts as a function that takes a time-series sequence of **aggregate power** as input and returns the estimated power consumption for **5 specific appliances** at the final time step.

*   **Input Resolution**: 5 minutes
*   **Input Length**: 288 samples (representing the past 24 hours)
*   **Units**: The model expects and outputs power values (e.g., kW or MW). *Ensure the input units match the training data.*

---

## 2. API Request (Input)

The backend should send a JSON object containing the sequence of aggregate power readings.

### JSON Schema
```json
{
  "request_id": "string (optional, for tracking)",
  "timestamp": "ISO8601 string (optional, reference time for the last point)",
  "aggregate_sequence": [
    number,
    number,
    ...
  ]
}
```

### Constraints
*   `aggregate_sequence`: Must be an array of **exactly 288** floating-point numbers.
    *   These values represent the aggregate power readings for the last 24 hours at 5-minute intervals.
    *   `index 0`: Oldest reading (T - 24h)
    *   `index 287`: Most recent reading (Current Time T)
*   **Data Preprocessing Note**: The backend wrapper script must handle normalization (Z-score scaling) before passing data to the model inference engine, as the raw model expects standardized inputs.

### Example Payload
```json
{
  "request_id": "req_12345",
  "timestamp": "2023-11-23T14:30:00Z",
  "aggregate_sequence": [
    150.5, 152.1, 148.9, 155.0, 160.2, 
    ... (283 more values) ...
  ]
}
```

---

## 3. API Response (Output)

The model returns the disaggregated power values for the **last time step** (corresponding to the last value in the input sequence).

### JSON Schema
```json
{
  "request_id": "string (matches input)",
  "predictions": {
    "EVSE": number,
    "PV": number,
    "CS": number,
    "CHP": number,
    "BA": number
  },
  "status": "success"
}
```

### Fields Description
*   `EVSE`: Electric Vehicle Supply Equipment (EV Charger) power.
*   `PV`: Photovoltaic (Solar) power generation (typically negative or zero).
*   `CS`: Cooling System power.
*   `CHP`: Combined Heat and Power generation.
*   `BA`: Battery / Building Automation power.

### Example Response
```json
{
  "request_id": "req_12345",
  "predictions": {
    "EVSE": 45.2,
    "PV": -12.5,
    "CS": 22.1,
    "CHP": 0.0,
    "BA": 5.3
  },
  "status": "success"
}
```

---

## 4. Integration Notes for Backend Developer

1.  **Preprocessing Wrapper**: The raw PyTorch model expects a tensor of shape `(1, 288, 1)` with **normalized** values (mean=0, std=1).
    *   **Do not** send raw sensor values directly to the `.pth` model.
    *   The backend service must:
        1.  Receive the JSON `aggregate_sequence`.
        2.  Apply `StandardScaler` transform (using the mean/std from training).
        3.  Convert to PyTorch Tensor.
        4.  Run Inference.
        5.  Inverse transform the output using the scaler.
        6.  Return the JSON response.

2.  **Handling Generation**:
    *   `PV` and `CHP` are generation sources. In the dataset, these might be represented as negative values. Ensure the frontend/database handles the sign convention correctly (e.g., Negative = Generation, Positive = Load).
