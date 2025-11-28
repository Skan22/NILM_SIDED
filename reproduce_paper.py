import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.dataset import load_data_by_location, create_sequences, NILMDataset, preprocess_single_appliance
from src.models import TCNModel, ATCNModel, LSTMModel
from src.train import train_single_appliance_model, evaluate_single_appliance_model, calculate_single_appliance_metrics
from torch.utils.data import DataLoader
import numpy as np
import json

# Configuration matching the paper
CONFIG = {
    'data_path': './AMDA_SIDED',
    'input_size': 1,
    'output_size': 1,        # Single appliance output (changed from 5)
    'seq_length': 288,        # 24h window at 5-min intervals
    'train_stride': 5,        # 25 min stride for training
    'eval_stride': 1,         # Dense evaluation
    'batch_size': 64,
    'learning_rate': 0.0003,
    'num_epochs': 20,
    'warmup_epochs': 3,
    'min_lr': 1e-6,
    'early_stopping_patience': 5,
    'num_workers': 0,         # Windows compatibility
    'pin_memory': True,
    'dropout': 0.33,
    'tcn_layers': [128] * 8,  # 8 layers of 128 channels
    'lstm_hidden': 128,       # LSTM hidden size
    'lstm_layers': 3          # LSTM layers
}

# Appliances to disaggregate (single-appliance NILM)
APPLIANCES = ['EVSE', 'PV', 'CS', 'CHP', 'BA']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\n{'='*80}")
    print("SINGLE-APPLIANCE NILM TRAINING (As per Paper)")
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
    
    # Model architectures to train
    model_configs = {
        'LSTM': lambda: LSTMModel(
            input_size=CONFIG['input_size'], 
            hidden_size=CONFIG['lstm_hidden'], 
            num_layers=CONFIG['lstm_layers'],
            output_size=CONFIG['output_size']
        ),

        'TCN': lambda: TCNModel(
            input_size=CONFIG['input_size'], 
            num_channels=CONFIG['tcn_layers'], 
            dropout=CONFIG['dropout'],
            output_size=CONFIG['output_size']
        ),
        'ATCN': lambda: ATCNModel(
            input_size=CONFIG['input_size'], 
            num_channels=CONFIG['tcn_layers'], 
            dropout=CONFIG['dropout'],
            output_size=CONFIG['output_size']
        )
    }
    
    # Store all results
    all_results = {}
    
    # 2. Train separate model for each appliance
    for appliance_name in APPLIANCES:
        print(f"\n{'#'*80}")
        print(f"# APPLIANCE: {appliance_name}")
        print(f"{'#'*80}\n")
        
        # Preprocess data for this specific appliance
        print(f"Preprocessing data for {appliance_name}...")
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y = \
            preprocess_single_appliance(train_df, test_df, appliance_name)
        
        # Create sequences
        print(f"Creating sequences for {appliance_name}...")
        X_train_seq, y_train_seq = create_sequences(
            X_train_scaled, y_train_scaled, 
            CONFIG['seq_length'], 
            stride=CONFIG['train_stride']
        )
        X_test_seq, y_test_seq = create_sequences(
            X_test_scaled, y_test_scaled, 
            CONFIG['seq_length'], 
            stride=CONFIG['eval_stride']
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(y_train_seq)
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
        
        # Initialize results for this appliance
        all_results[appliance_name] = {}
        
        # 3. Train each model architecture for this appliance
        for model_name, model_factory in model_configs.items():
            print(f"\n{'-'*80}")
            print(f"Training {model_name} for {appliance_name}")
            print(f"{'-'*80}")
            
            # Create fresh model
            model = model_factory()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
            
            # Setup learning rate scheduler (Warmup + Cosine Annealing)
            warmup_lambda = lambda epoch: (epoch + 1) / CONFIG['warmup_epochs'] if epoch < CONFIG['warmup_epochs'] else 1.0
            warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=CONFIG['num_epochs'] - CONFIG['warmup_epochs'], 
                eta_min=CONFIG['min_lr']
            )
            
            class CombinedScheduler:
                def __init__(self):
                    self.current_epoch = 0
                
                def step(self):
                    if self.current_epoch < CONFIG['warmup_epochs']:
                        warmup_scheduler.step()
                    else:
                        cosine_scheduler.step()
                    self.current_epoch += 1
            
            scheduler = CombinedScheduler()
            
            # Train the model
            model = train_single_appliance_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=CONFIG['num_epochs'],
                device=device,
                early_stopping_patience=CONFIG['early_stopping_patience'],
                model_name=model_name,
                appliance_name=appliance_name
            )
            
            # Evaluate the model
            print(f"\nEvaluating {model_name} on {appliance_name}...")
            metrics, preds_real, targets_real = evaluate_single_appliance_model(
                model, test_loader, device, scaler_y, appliance_name
            )
            
            # Store results
            all_results[appliance_name][model_name] = metrics
            
            # Save model
            model_save_path = Path(f"{model_name}_{appliance_name}_best.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 Saved model to {model_save_path}")
            
            # Save predictions for visualization
            pred_save_path = Path(f"{model_name}_{appliance_name}_predictions.npz")
            np.savez(pred_save_path, predictions=preds_real, targets=targets_real)
            print(f"💾 Saved predictions to {pred_save_path}")
    
    # 4. Save all results to JSON
    results_path = Path("all_results_single_appliance.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n{'='*80}")
    print(f"✅ All results saved to {results_path}")
    print(f"{'='*80}\n")
    
    # Print summary table
    print("\n" + "="*120)
    print("SUMMARY: Single-Appliance NILM Results")
    print("="*120)
    print(f"{'Appliance':<10} | {'Model':<8} | {'MAE (W)':<12} | {'MSE (W²)':<15} | {'R²':<10} | {'NDE':<10}")
    print("-"*120)
    
    for appliance_name in APPLIANCES:
        for model_name in model_configs.keys():
            metrics = all_results[appliance_name][model_name]
            print(f"{appliance_name:<10} | {model_name:<8} | {metrics['MAE_W']:<12.2f} | "
                  f"{metrics['MSE_W']:<15.2f} | {metrics['R2']:<10.4f} | {metrics['NDE']:<10.4f}")
    
    print("="*120)


if __name__ == "__main__":
    main()
