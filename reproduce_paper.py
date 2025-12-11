import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.dataset import load_data_by_location, create_sequences, NILMDataset, preprocess_multi_appliance
from src.models import TCNModel, ATCNModel, LSTMModel, ImprovedATCNModel
from src.train import train_multi_appliance_model, evaluate_multi_appliance_model
from torch.utils.data import DataLoader
import numpy as np
import json

# Configuration matching the paper
CONFIG = {
    'data_path': './SIDED', # change this 
    'input_size': 1,
    'output_size': 5,        # Multi-output: 5 appliances
    'seq_length': 288,        # 24h window at 5-min intervals
    'train_stride': 5,        # 25 min stride for training
    'eval_stride': 1,         # Dense evaluation
    'batch_size': 128,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'warmup_epochs': 3,
    'min_lr': 1e-6,
    'early_stopping_patience': 5,
    'num_workers': 0,         # Windows compatibility
    'pin_memory': True,
    'dropout': 0.33,
    'tcn_layers': [128]*6, 
    'lstm_hidden': 128,       # LSTM hidden size
    'lstm_layers': 6
    }

# Appliances to disaggregate (Multi-Output)
APPLIANCES = ['EVSE', 'PV', 'CS', 'CHP', 'BA']

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Only use pin_memory on CUDA for efficiency
    CONFIG['pin_memory'] = (device.type == 'cuda')
    print(f"Using device: {device}")
    print(f"Pin memory: {CONFIG['pin_memory']}")
    print(f"\n{'='*80}")
    print("MULTI-APPLIANCE NILM TRAINING (5-Output Regression)")
    print(f"{'='*80}\n")
    
    # 1. Load Data
    print("Loading data...")
    try:
        train_df, test_df = load_data_by_location(
            base_path=CONFIG['data_path'],
            target_locations=['Tokyo'],
            source_locations=['LA', 'Offenbach'],
            resample_rule='5min'
        )
    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Please ensure you have run data_augmentation.py first.")
        return
    
    # Preprocess data for ALL appliances at once
    print(f"Preprocessing data for all 5 appliances: {APPLIANCES}...")
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y = \
        preprocess_multi_appliance(train_df, test_df, APPLIANCES)
    
    # Create sequences
    print(f"Creating sequences...")
    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled, y_train_scaled, 
        CONFIG['seq_length'], 
        stride=CONFIG['train_stride'],
        target_pos='mid'
    )
    X_test_seq, y_test_seq = create_sequences(
        X_test_scaled, y_test_scaled, 
        CONFIG['seq_length'], 
        stride=CONFIG['eval_stride'],
        target_pos='mid'
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_seq) # (N, L, 1)
    y_train_tensor = torch.FloatTensor(y_train_seq) # (N, 5)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq)
    
    # Create datasets
    train_dataset = NILMDataset(X_train_tensor, y_train_tensor)
    test_dataset = NILMDataset(X_test_tensor, y_test_tensor)
    
    # Split validation set
    val_size = int(len(train_dataset) * 0.1)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG['pin_memory']
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG['pin_memory']
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG['pin_memory']
    )
    
    # Model configs (Now output_size is 5)
    model_configs = {
        'LSTM': lambda: LSTMModel(
            input_size=CONFIG['input_size'], 
            hidden_size=CONFIG['lstm_hidden'], 
            num_layers=CONFIG['lstm_layers'],
            output_size=CONFIG['output_size'],
            bidirectional=False
        ),
                'BiLSTM': lambda: LSTMModel(
            input_size=CONFIG['input_size'], 
            hidden_size=CONFIG['lstm_hidden'], 
            num_layers=CONFIG['lstm_layers'],
            output_size=CONFIG['output_size'],
            bidirectional=True
        ),
        'TCN': lambda: TCNModel(
            input_size=CONFIG['input_size'], 
            num_channels=CONFIG['tcn_layers'], 
            dropout=CONFIG['dropout'],
            output_size=CONFIG['output_size'],
            causal=False
        ),
        'ATCN': lambda: ImprovedATCNModel(
            input_size=CONFIG['input_size'], 
            num_channels=CONFIG['tcn_layers'],
            dropout=0.25, 
            output_size=CONFIG['output_size'],
            causal=False,
            num_heads=4
        ),
    }
    
    all_results = {}
    
    # 2. Train Multi-Output Models
    for model_name, model_factory in model_configs.items():
        print(f"\n{'-'*80}")
        print(f"Training {model_name} (Multi-Output)")
        print(f"{'-'*80}")
        
        # Create fresh model
        model = model_factory()
        criterion = nn.MSELoss() # Average MSE over all outputs
        
        # Learning rates
        model_lr_multipliers = {
            'LSTM': 1.0,
            'TCN': 1.0, 
            'ATCN': 1.0,
        }
        
        lr_multiplier = model_lr_multipliers.get(model_name, 1.0)
        effective_lr = CONFIG['learning_rate'] * lr_multiplier
        
        print(f"Using learning rate: {effective_lr}")
        
        optimizer = optim.Adam(model.parameters(), lr=effective_lr)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=CONFIG['min_lr']
        )
        
        # Train Multi-Appliance Model
        model = train_multi_appliance_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=CONFIG['num_epochs'],
            device=device,
            early_stopping_patience=CONFIG['early_stopping_patience'],
            model_name=model_name,
            appliance_names=APPLIANCES
        )
        
        # Evaluate
        print(f"\nEvaluating {model_name}...")
        metrics, preds_real, targets_real = evaluate_multi_appliance_model(
            model, test_loader, device, scaler_y, APPLIANCES
        )
        
        all_results[model_name] = metrics
        
        # Save model
        model_save_path = Path(f"{model_name}_multi_output_best.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 Saved model to {model_save_path}")
        
        # Save predictions
        pred_save_path = Path(f"{model_name}_multi_output_predictions.npz")
        np.savez(pred_save_path, predictions=preds_real, targets=targets_real)
        print(f"💾 Saved predictions to {pred_save_path}")

    # 3. Save Results
    results_path = Path("all_results_multi_appliance.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n{'='*80}")
    print(f"✅ All results saved to {results_path}")
    print(f"{'='*80}\n")
    
    # 4. Summary Table
    print("\n" + "="*120)
    print("SUMMARY: Multi-Appliance NILM Results (Average MAE)")
    print("="*120)
    
    # Header
    print(f"{'Model':<10} | ", end="")
    for app in APPLIANCES:
        print(f"{app:<10} | ", end="")
    print(f"{'AVG':<10} |")
    print("-" * 120)
    
    # Rows
    for model_name, metrics in all_results.items():
        print(f"{model_name:<10} | ", end="")
        avg_mae = 0
        for app in APPLIANCES:
            mae = metrics[app]['MAE_W']
            avg_mae += mae
            print(f"{mae:<10.2f} | ", end="")
        print(f"{avg_mae/len(APPLIANCES):<10.2f} |")
    print("="*120)

if __name__ == "__main__":
    main()
