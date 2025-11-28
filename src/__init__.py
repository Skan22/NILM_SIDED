"""
SIDED - NILM (Non-Intrusive Load Monitoring) Package
Implements TCN, ATCN, LSTM, and BiLSTM models for energy disaggregation.
"""

# Dataset utilities
from src.dataset import (
    NILMDataset,
    create_sequences,
    load_data_by_location,
    preprocess_data
)

# Model architectures
from src.models import (
    TCNModel,
    ATCNModel,
    LSTMModel,
    BiLSTMModel,
    GRUModel,
    CNN_LSTM
)

# Training and evaluation
from src.train import (
    train_model,
    evaluate_model,
    calculate_metrics
)

# Visualization
from src.visualization import (
    plot_results,
    evaluate_saved_model_robust
)

__all__ = [
    # Dataset
    'NILMDataset',
    'create_sequences',
    'load_data_by_location',
    'preprocess_data',
    # Models
    'TCNModel',
    'ATCNModel',
    'LSTMModel',
    'BiLSTMModel',
    'GRUModel',
    'CNN_LSTM',
    # Training
    'train_model',
    'evaluate_model',
    'calculate_metrics',
    # Visualization
    'plot_results',
    'evaluate_saved_model_robust',
]

__version__ = '1.0.0'
