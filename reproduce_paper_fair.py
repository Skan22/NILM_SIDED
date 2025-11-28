import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.dataset import load_data_by_location, create_sequences, NILMDataset, preprocess_single_appliance
from src.models import TCNModel, ATCNModel, LSTMModel, count_parameters
from src.train import train_single_appliance_model, evaluate_single_appliance_model, calculate_single_appliance_metrics
from torch.utils.data import DataLoader
import numpy as np
import json

# Configuration for FAIR comparison (~1.0M parameters each)
CONFIG = {
    'data_path': './AMDA_SIDED',
    'input_size': 1,
    'output_size': 1,        
    'seq_length': 288,        
    'train_stride': 5,        
    'eval_stride': 1,         
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 20,
    'warmup_epochs': 3,
    'min_lr': 1e-6,
    'early_stopping_patience': 5,
    'num_workers': 0,         
    'pin_memory': True,
    'dropout': 0.33,
    
    # --- FAIR CONFIGURATIONS (~1.05M Params) ---
    # TCN: Unchanged (Baseline)
    'tcn_layers': [128] * 8,  
    
    # ATCN: Reduced channels to match TCN params
    'atcn_layers': [100] * 8, 
    
    # LSTM: Increased size to match TCN params
    'lstm_hidden': 356,       
    'lstm_layers': 5          
}

# Appliances to disaggregate
APPLIANCES = ['EVSE', 'PV', 'CS', 'CHP', 'BA']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\n{'='*80}")
    print("FAIR COMPARISON: SINGLE-APPLIANCE NILM TRAINING")
    print("All models matched to ~1.05M parameters")
    print(f"{'='*80}\n")
    
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
        return
    
    # FAIR CONFIGURATIONS
    model_configs = {
        'TCN': lambda: TCNModel(
            input_size=CONFIG['input_size'], 
            num_channels=CONFIG['tcn_layers'],  # 128 channels (Baseline)
            dropout=CONFIG['dropout'],
            output_size=CONFIG['output_size']
        ),
        'ATCN': lambda: ATCNModel(
            input_size=CONFIG['input_size'], 
            num_channels=CONFIG['atcn_layers'], # 100 channels (Reduced)
            dropout=CONFIG['dropout'],
            output_size=CONFIG['output_size']
        ),
        'LSTM': lambda: LSTMModel(
            input_size=CONFIG['input_size'], 
            hidden_size=CONFIG['lstm_hidden'],  # 356 hidden (Increased)
            num_layers=CONFIG['lstm_layers'],   # 5 layers (Increased)
            output_size=CONFIG['output_size']
        )
    }
    
    all_results = {}
    
    for appliance_name in APPLIANCES:
        print(f"\n{'#'*80}")
        print(f"# APPLIANCE: {appliance_name}")
        print(f"{'#'*80}\n")
        
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y = \
            preprocess_single_appliance(train_df, test_df, appliance_name)
        
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
        
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(y_train_seq)
        X_test_tensor = torch.FloatTensor(X_test_seq)
        y_test_tensor = torch.FloatTensor(y_test_seq)
        
        train_dataset = NILMDataset(X_train_tensor, y_train_tensor)
        test_dataset = NILMDataset(X_test_tensor, y_test_tensor)
        
        val_size = int(len(train_dataset) * 0.1)
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_subset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
        val_loader = DataLoader(val_subset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
        
        all_results[appliance_name] = {}
        
        for model_name, model_factory in model_configs.items():
            print(f"\n{'-'*80}")
            print(f"Training {model_name} for {appliance_name}")
            print(f"{'-'*80}")
            
            model = model_factory()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
            
            warmup_lambda = lambda epoch: (epoch + 1) / CONFIG['warmup_epochs'] if epoch < CONFIG['warmup_epochs'] else 1.0
            warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=CONFIG['num_epochs'] - CONFIG['warmup_epochs'], 
                eta_min=CONFIG['min_lr']
            )
            
            class CombinedScheduler:
                def __init__(self): self.current_epoch = 0
                def step(self):
                    if self.current_epoch < CONFIG['warmup_epochs']: warmup_scheduler.step()
                    else: cosine_scheduler.step()
                    self.current_epoch += 1
            
            scheduler = CombinedScheduler()
            
            model = train_single_appliance_model(
                model, train_loader, val_loader, criterion, optimizer, 
                scheduler, CONFIG['num_epochs'], device, 
                CONFIG['early_stopping_patience'], model_name, appliance_name
            )
            
            print(f"Evaluating {model_name} on {appliance_name}...")
            metrics, predictions, targets = evaluate_single_appliance_model(
                model, test_loader, device, scaler_y, appliance_name
            )
            
            all_results[appliance_name][model_name] = metrics
            
            torch.save(model.state_dict(), f"FAIR_{model_name}_{appliance_name}_best.pth")
            np.savez(f"FAIR_{model_name}_{appliance_name}_predictions.npz", predictions=predictions, targets=targets)
            print(f"💾 Saved fair model and predictions for {model_name}_{appliance_name}")
            
    with open('all_results_fair.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\n✅ Fair comparison training complete. Results saved to all_results_fair.json")

if __name__ == "__main__":
    main()
