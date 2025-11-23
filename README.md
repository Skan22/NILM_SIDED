# NILM_SIDED: Industrial Energy Disaggregation with AMDA, TCN / ATCN / BiLSTM

Non-Intrusive Load Monitoring (NILM) for industrial facilities using the SIDED dataset and Appliance‑Modulated Data Augmentation (AMDA). This project trains deep sequence models (TCN, Attention TCN, BiLSTM) to disaggregate aggregate power signals into appliance‑level consumption / generation.

---
## 1. Project Overview
- **Goal**: Predict individual appliance (EVSE, PV, CS, CHP, BA) power from a single aggregate measurement.
- **Setting**: Cross-domain generalization. Train on Los Angeles + Offenbach; test on Tokyo.
- **Models**: Temporal Convolutional Network (TCN), Attention TCN (ATCN), BiLSTM (plus auxiliary architectures: vanilla LSTM, GRU, CNN-LSTM).
- **Sequence Length**: 288 (24 hours at 5‑minute resolution).
- **Metrics**: MAE (W & MW), MSE, R² per appliance.

---
## 2. Dataset & AMDA Augmentation
### Base Dataset
Original facility CSVs in `SIDED/` (Dealer, Logistic, Office) contain aggregate + appliance traces.

### AMDA (Appliance-Modulated Data Augmentation)
AMDA scales appliance traces using the formula:
```
S_i = s * (1 - p_i)   where   p_i = (|P_i|) / (Σ_j |P_j|)
```
- Uses **absolute power** ensuring generation appliances (PV, CHP) contribute correctly.
- Produces augmented CSVs in `AMDA_SIDED/` with more diverse appliance magnitude distributions.

### Resampling
Data is downsampled to **5-minute intervals** (averaging every 5 original rows) to match paper configuration: 288 timesteps per day.

---
## 3. Problem Formulation
Given time series:
- Input: `Aggregate[t - L : t]` (window of length L=288)
- Output: `Appliances[t]` (EVSE, PV, CS, CHP, BA)
Supervised multi-output regression on standardized data (separate `StandardScaler` for X and y).

---
## 4. Model Architectures
### TCN
- Stacked dilated causal 1D convolutions.
- Residual temporal blocks with dropout.
- Global average pooling + linear head.

### ATCN
- Same TCN feature extractor + temporal attention layer over sequence dimension.

### BiLSTM
- Bidirectional LSTM layers, final time step concatenated hidden states → linear head.

### Additional (Experimental)
- GRU, CNN-LSTM, vanilla LSTM included for comparison.

Model hyperparameters centralised in `CONFIG` (inside `workspace.ipynb`):
```
'num_layers': 8
'num_channels': [64,64,64,64,128,128,128,128]
'hidden_size': 128
'dropout': 0.33
```

---
## 5. Training Pipeline (Optimized)
Key features (implemented in notebook):
- **Automatic Mixed Precision (AMP)** for speed.
- **Warmup + Cosine LR Scheduler**: linear warmup (3 epochs) → cosine decay to `min_lr`.
- **Early Stopping** with patience=5.
- **Gradient Clipping** (norm ≤ 1.0).
- **Best Checkpoint Restore**.
- **cuDNN / TF32 optimizations** when CUDA available.
- **Non-blocking, pinned memory DataLoaders** (Windows requires `num_workers=0`).

Pseudo-code outline:
```
for epoch in epochs:
    train one pass (AMP, clip grads)
    validate
    scheduler.step()
    track best model / early stop
```

---
## 6. Evaluation & Metrics
After training, saved model weights are stored in `saved_models/` as `*_best.pth`.

Standalone evaluation cell performs:
1. Load weights.
2. Forward inference over `test_loader`.
3. Clamp standardized outputs to [-8, 8] (stability).
4. Inverse transform with `scaler_y` (float64 precision).
5. Enforce sign conventions:
   - Loads (EVSE, CS, BA) ≥ 0
   - Generation (PV, CHP) ≤ 0
6. Compute per-appliance metrics.

Example (current project run):
```
TCN_BEST:
EVSE  MAE=0.005008 MW  R²=0.8045
PV    MAE=0.042131 MW  R²=0.9484
CS    MAE=0.025614 MW  R²=0.7420
CHP   MAE=0.039977 MW  R²=0.9722
BA    MAE=0.044810 MW  R²=0.9365
```

Visualizations include:
- Scatter: Predicted vs Actual (MW)
- Error histograms (MW)
- Time series overlays (first 500 samples)
- MAE & R² bar comparisons
- Heatmap (MAE per model/appliance)

---
## 7. Repository Layout
```
AMDA_SIDED/         # Augmented facility data
SIDED/              # Original facility data
saved_models/       # Trained model weights (*.pth)
workspace.ipynb     # Main notebook (training + evaluation)
data_augmentation.py# AMDA augmentation script
README.md           # Project documentation
```

---
## 8. Installation
### Python Environment
Recommended Python ≥ 3.9. Create virtual environment:
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
```

### Dependencies (minimal)
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```
(Adjust CUDA wheel as needed; use CPU-only wheel if no GPU.)

---
## 9. Usage
### 1. Prepare Augmented Data (if regenerating)
Edit `data_augmentation.py` as needed then run:
```cmd
python data_augmentation.py
```
Ensure output resides in `AMDA_SIDED/` matching expected folder structure.

### 2. Open Notebook
Launch Jupyter (or VS Code notebook):
```cmd
python -m jupyter notebook
```
Execute cells sequentially:
1. Configuration
2. Data loading & resampling
3. Preprocessing & scaling
4. Sequence construction
5. Model definitions
6. Training loop

### 3. Evaluate Saved Models
Run the "Standalone Model Evaluation" cell (no retraining required).
Run visualization cell for plots.

### 4. Verify Metrics
Use the verification cell to confirm graph correctness.

---
## 10. Reproducing Paper-like Setup
To move closer to target metrics (e.g., R² ≈ 0.95 for all loads):
- Increase model depth or receptive field (larger dilation range).
- Introduce attention for spike-heavy appliances (EVSE).
- Add event-based augmentation for transient loads.
- Perform hyperparameter sweeps: learning rate, dropout, channel widths.
- Consider ensemble (BiLSTM for EVSE + TCN for others).

---
## 11. Extending
Ideas:
- Add Transformer-based model (Temporal Fusion Transformer).
- Implement multi-resolution inputs (1min + 5min fusion).
- Add domain adaptation loss (e.g., CORAL / MMD between source & target features).
- Integrate anomaly / spike detector for EVSE improvement.

---
## 12. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| Overflow in inverse transform | Extreme z-scores | Clamp standardized outputs to [-8, 8] |
| Poor EVSE spike capture | Model smoothing | Add attention / spike augmentation |
| CS low R² | Underfitting or noisy data | Increase capacity, inspect raw CS signal |
| CUDA memory errors | Large batch or model | Reduce `batch_size`, clear cache between models |

---
## 13. Notes on Data Leakage
- Only fit scalers on **source (training)** data.
- Resampling applied consistently across source & target.
- Domain split maintains geographic separation.

---
## 14. License & Attribution
Write your chosen license (MIT, Apache 2.0, etc.) here if you intend to distribute.
Dataset and AMDA concepts based on publicly available research; implementation here is original.

---
## 15. Quick Start Summary
```cmd
git clone <repo>
cd NILM_SIDED
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt  # (Create this file optionally)
python data_augmentation.py      # (If regenerating)
python -m jupyter notebook       # Run notebook cells
# After training:
# Run evaluation + visualization cells
```

---
## 16. Acknowledgements
- SIDED dataset creators
- Research on AMDA for industrial NILM
- PyTorch community

---
## 17. Disclaimer
This codebase is for research & educational purposes. Validate models before production deployment.

---
**Happy Disaggregating!** 🔌⚡
