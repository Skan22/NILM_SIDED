# Training Output Files

After running `reproduce_paper.py`, the following files will be automatically saved to your project directory:

## Model Weights
- `TCN_best.pth` - Best TCN model weights
- `ATCN_best.pth` - Best ATCN model weights  
- `LSTM_best.pth` - Best LSTM model weights

## Evaluation Metrics (JSON)
- `TCN_metrics.json` - Per-appliance metrics (MAE, MSE, R²) for TCN
- `ATCN_metrics.json` - Per-appliance metrics for ATCN
- `LSTM_metrics.json` - Per-appliance metrics for LSTM

Each JSON file contains metrics for all 5 appliances:
```json
{
  "EVSE": {
    "MAE_W": 12345.67,
    "MAE_MW": 0.012346,
    "MSE_W": 123456789.0,
    "MSE_MW2": 0.123457,
    "R2": 0.85
  },
  "PV": { ... },
  ...
}
```

## Visualization Plots (PNG)
- `TCN_plot.png` - Actual vs Predicted plots for all appliances (TCN)
- `ATCN_plot.png` - Actual vs Predicted plots for all appliances (ATCN)
- `LSTM_plot.png` - Actual vs Predicted plots for all appliances (LSTM)

Each plot shows the first 1000 time steps of predictions vs actual values for all 5 appliances.

## Location on Colab
If running on Google Colab, these files will be saved in:
```
/content/drive/MyDrive/NILM_Project/
```

You can download them directly from the Colab file browser or access them from your Google Drive.
