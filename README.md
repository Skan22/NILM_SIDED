# SIDED: Industrial Energy Disaggregation

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/html/2506.20525v2)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official implementation of **"Industrial Energy Disaggregation with Digital Twin-generated Dataset and Efficient Data Augmentation"**

> **Paper**: [Industrial Energy Disaggregation with Digital Twin-generated Dataset and Efficient Data Augmentation](https://arxiv.org/html/2506.20525v2)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Documentation](#documentation)
- [Citation](#citation)

---

## 🎯 Overview

This repository implements **single-appliance Non-Intrusive Load Monitoring (NILM)** for industrial facilities using deep learning models trained on the SIDED dataset. The implementation follows the exact methodology described in the paper.

### What is NILM?

Non-Intrusive Load Monitoring (NILM) is the process of disaggregating total power consumption into individual appliance-level consumption without installing sensors on each appliance.

### Single-Appliance Approach

Following the paper's methodology, we train **separate models for each appliance**:
- Input: Aggregate power signal
- Output: Single appliance power consumption
- Total: 15 models (3 architectures × 5 appliances)

---

## ✨ Key Features

- ✅ **Paper-Compliant Implementation**: Exact replication of the paper's methodology
- ✅ **Single-Appliance NILM**: Specialized models for each appliance
- ✅ **AMDA Data Augmentation**: Appliance-Modulated Data Augmentation
- ✅ **Multiple Architectures**: TCN, ATCN (Attention-TCN), and LSTM
- ✅ **Robust Data Handling**: Three-layer missing value handling system
- ✅ **Comprehensive Metrics**: MAE, MSE, R², and NDE evaluation

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SIDED-NILM.git
cd SIDED-NILM

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---

## 🚀 Quick Start

### 1. Prepare the Dataset

Place the SIDED dataset in the `./SIDED` directory with the following structure:

```
SIDED/
├── Dealer/
│   ├── Dealer_LA.csv
│   ├── Dealer_Offenbach.csv
│   └── Dealer_Tokyo.csv
├── Logistic/
│   ├── Logistic_LA.csv
│   ├── Logistic_Offenbach.csv
│   └── Logistic_Tokyo.csv
└── Office/
    ├── Office_LA.csv
    ├── Office_Offenbach.csv
    └── Office_Tokyo.csv
```

### 2. Generate Augmented Data

```bash
python data_augmentation.py
```

This applies AMDA (Appliance-Modulated Data Augmentation) with scaling factors s=1.5, 2.5, 4.0.

**Output**: `./AMDA_SIDED/` directory with augmented datasets

### 3. Train Models

```bash
python reproduce_paper.py
```

**Training Details**:
- **Duration**: ~5-10 hours (15 models)
- **Models**: TCN, ATCN, LSTM for each of 5 appliances
- **Output**: 15 model files + predictions + metrics

### 4. View Results

Results are saved in:
- `all_results_single_appliance.json` - All metrics
- `{MODEL}_{APPLIANCE}_best.pth` - Trained models (15 files)
- `{MODEL}_{APPLIANCE}_predictions.npz` - Predictions (15 files)

---

## 📊 Dataset

### SIDED Dataset

The **Synthetic Industrial Dataset for Energy Disaggregation** contains:

- **Facilities**: 3 types (Dealer, Logistic, Office)
- **Locations**: 3 locations (LA, Offenbach, Tokyo)
- **Appliances**: 5 appliances
  - **EVSE**: Electric Vehicle Supply Equipment (Load)
  - **PV**: Photovoltaic System (Generation)
  - **CS**: Cooling System (Load)
  - **CHP**: Combined Heat and Power (Generation)
  - **BA**: Building Automation (Load)

### Data Augmentation (AMDA)

Appliance-Modulated Data Augmentation scales appliances inversely to their contribution:

```
S_i = s × (1 - p_i)
```

where:
- `s`: scaling factor (1.5, 2.5, 4.0)
- `p_i`: relative contribution of appliance i
- `S_i`: scaling factor for appliance i

**Effect**: Reduces dominance of major appliances, increases diversity.

---

## 🧠 Models

### Architectures

#### 1. TCN (Temporal Convolutional Network)
- **Layers**: 8 temporal convolutional blocks
- **Channels**: 128 per layer
- **Kernel Size**: 3
- **Dropout**: 0.33
- **Receptive Field**: Covers entire input sequence
- **Parameters**: ~1.05M

#### 2. ATCN (Attention-TCN)
- **Base**: TCN architecture
- **Addition**: Attention mechanism
- **Purpose**: Highlights important temporal features
- **Parameters**: ~1.4-1.6M

#### 3. LSTM (Long Short-Term Memory)
- **Layers**: 3 LSTM layers
- **Hidden Size**: 128
- **Dropout**: 0.2
- **Bidirectional**: No (standard LSTM)
- **Parameters**: ~265k

> **⚠️ IMPORTANT**: The current model configurations are **NOT parameter-matched**, which makes direct comparison unfair:
> - **ATCN** has 40-50% MORE parameters than TCN
> - **LSTM** has only 25% of TCN's parameters
> 
> For a scientifically fair comparison, see [docs/PARAMETER_ANALYSIS.md](docs/PARAMETER_ANALYSIS.md) for parameter-matched configurations.

### Training Configuration

```python
CONFIG = {
    'seq_length': 288,        # 24h window at 5-min intervals
    'train_stride': 5,        # 25 min stride
    'eval_stride': 1,         # Dense evaluation
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 20,
    'warmup_epochs': 3,
    'early_stopping_patience': 5,
    'dropout': 0.33
}
```

### Sequence-to-Point (S2P)

Input: Time series window of aggregate power  
Output: Single point prediction at window midpoint

```
X_t = {x_{t-144}, ..., x_t, ..., x_{t+144}}  → y_t
```

---

## 📈 Results

### Expected Performance

Results vary by appliance and model. Example metrics:

| Appliance | Model | MAE (W) | R² | NDE |
|-----------|-------|---------|-----|-----|
| EVSE | TCN | ~1200 | ~0.85 | ~0.12 |
| EVSE | ATCN | ~1100 | ~0.88 | ~0.10 |
| EVSE | LSTM | ~1300 | ~0.82 | ~0.15 |

**Note**: Actual results depend on dataset and training conditions.

### Metrics

- **MAE**: Mean Absolute Error (Watts)
- **MSE**: Mean Squared Error (Watts²)
- **R²**: Coefficient of Determination
- **NDE**: Normalized Disaggregation Error

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[METHODOLOGY.md](docs/METHODOLOGY.md)** - Single-appliance NILM approach
- **[DATA_PIPELINE.md](docs/DATA_PIPELINE.md)** - Data loading and preprocessing
- **[TRAINING.md](docs/TRAINING.md)** - Training procedure and hyperparameters
- **[EVALUATION.md](docs/EVALUATION.md)** - Metrics and evaluation protocol
- **[PARAMETER_ANALYSIS.md](docs/PARAMETER_ANALYSIS.md)** - Model parameter comparison and fair configs ⚠️

---

## 🗂️ Project Structure

```
SIDED-NILM/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── data_augmentation.py              # AMDA implementation
├── reproduce_paper.py                # Main training script
│
├── src/                              # Source code
│   ├── __init__.py                   # Package initialization
│   ├── dataset.py                    # Data loading and preprocessing
│   ├── models.py                     # Model architectures
│   ├── train.py                      # Training utilities
│   └── visualization.py              # Plotting and visualization
│
├── docs/                             # Documentation
│   ├── METHODOLOGY.md                # Single-appliance approach
│   ├── DATA_PIPELINE.md              # Data handling
│   ├── TRAINING.md                   # Training details
│   └── EVALUATION.md                 # Evaluation protocol
│
├── SIDED/                            # Original dataset (not included)
│   ├── Dealer/
│   ├── Logistic/
│   └── Office/
│
└── AMDA_SIDED/                       # Augmented dataset (generated)
    ├── Dealer/
    ├── Logistic/
    └── Office/
```

---

## 🔬 Methodology

### Single-Appliance NILM

Following the paper (Section V-A):

> *"We employ the **single-appliance NILM setting** where the input is the aggregated power signal and the model extracts the power of a **single appliance** e.g., the CHP."*

**Implementation**:
- Train **one model per appliance** (not multi-output)
- Each model has **1 output neuron** (not 5)
- Separate **RobustScaler** per appliance
- Total: **15 models** (3 architectures × 5 appliances)

**Benefits**:
- Model specialization for each appliance
- Better performance through focused learning
- Easy to add/remove appliances
- Clear interpretability

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for details.

---

## 🛡️ Data Quality

### Three-Layer Defense System

1. **Layer 1: Data Loading**
   - Forward/backward fill for missing values
   - Inf value detection and replacement
   - Data quality reporting

2. **Layer 2: Preprocessing**
   - Per-appliance NaN/Inf checks
   - Sanitization before scaling
   - Prevents RobustScaler failures

3. **Layer 3: Evaluation**
   - Prediction clipping in standardized space
   - Inverse transform safety
   - Physical constraint enforcement

See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) for details.

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{sided2025,
  title={Industrial Energy Disaggregation with Digital Twin-generated Dataset and Efficient Data Augmentation},
  author={[Authors]},
  journal={arXiv preprint arXiv:2506.20525},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

## 🙏 Acknowledgments

- Original paper authors for the SIDED dataset and methodology
- PyTorch team for the deep learning framework
- Open-source community for various tools and libraries

---

**⭐ If you find this repository useful, please consider giving it a star!**
