import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.dataset import load_data_by_location, create_sequences, NILMDataset
from src.models import TCNModel, ATCNModel, LSTMModel
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import RobustScaler
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


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def preprocess_single_appliance(train_df, test_df, appliance_name):
    """Preprocess data for single-appliance NILM."""
    X_train_raw = train_df['Aggregate'].values.reshape(-1, 1)
    y_train_raw = train_df[appliance_name].values.reshape(-1, 1)
    
    X_test_raw = test_df['Aggregate'].values.reshape(-1, 1)
    y_test_raw = test_df[appliance_name].values.reshape(-1, 1)
    
    for name, arr in [("X_train", X_train_raw), ("y_train", y_train_raw), 
                       ("X_test", X_test_raw), ("y_test", y_test_raw)]:
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️ {appliance_name} - {name}: {nan_count} NaN, {inf_count} Inf values")
            arr = np.where(np.isinf(arr), np.nan, arr)
            arr = np.nan_to_num(arr, nan=0.0)
            
            if name == "X_train": X_train_raw = arr
            elif name == "y_train": y_train_raw = arr
            elif name == "X_test": X_test_raw = arr
            elif name == "y_test": y_test_raw = arr
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)
    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y


def train_single_appliance_model(model, train_loader, val_loader, criterion, optimizer, 
                                  scheduler, num_epochs, device, early_stopping_patience, 
                                  model_name, appliance_name):
    
    model.to(device)
    
    # Log parameter count
    param_count = count_parameters(model)
    print(f"Model: {model_name} | Parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


def evaluate_single_appliance_model(model, test_loader, device, scaler_y, appliance_name):
    model.eval()
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
    
    metrics = calculate_single_appliance_metrics(targets, predictions, appliance_name, scaler_y)
    metrics['parameters'] = count_parameters(model)
    
    return metrics, predictions, targets


def calculate_single_appliance_metrics(targets, predictions, appliance_name, scaler_y):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    predictions = np.clip(predictions, -8.0, 8.0)
    
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.clip(targets, -8.0, 8.0)
    
    targets_real = scaler_y.inverse_transform(targets.astype(np.float64)).astype(np.float32)
    predictions_real = scaler_y.inverse_transform(predictions.astype(np.float64)).astype(np.float32)
    
    load_appliances = ['EVSE', 'CS', 'BA']
    generation_appliances = ['PV', 'CHP']
    
    if appliance_name in load_appliances:
        predictions_real = np.maximum(predictions_real, 0)
    elif appliance_name in generation_appliances:
        predictions_real = np.minimum(predictions_real, 0)
    
    mae_w = mean_absolute_error(targets_real, predictions_real)
    mse_w = mean_squared_error(targets_real, predictions_real)
    r2 = r2_score(targets_real, predictions_real)
    
    mae_mw = mae_w / 1e6
    mse_mw = mse_w / (1e6 ** 2)
    
    numerator = np.sum(np.abs(targets_real - predictions_real))
    denominator = np.sum(np.abs(targets_real))
    nde = numerator / denominator if denominator > 0 else float('inf')
    
    metrics = {
        'MAE_W': float(mae_w),
        'MAE_MW': float(mae_mw),
        'MSE_W': float(mse_w),
        'MSE_MW2': float(mse_mw),
        'R2': float(r2),
        'NDE': float(nde)
    }
    
    print(f"\n{appliance_name} Metrics:")
    print(f"  MAE: {mae_w:.2f} W ({mae_mw:.6f} MW)")
    print(f"  MSE: {mse_w:.2f} W² ({mse_mw:.6f} MW²)")
    print(f"  R²:  {r2:.4f}")
    print(f"  NDE: {nde:.4f}")
    
    return metrics


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
