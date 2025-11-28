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

# Configuration matching the paper
CONFIG = {
    'data_path': './AMDA_SIDED',
    'input_size': 1,
    'output_size': 1,        # Single appliance output (changed from 5)
    'seq_length': 288,        # 24h window at 5-min intervals
    'train_stride': 5,        # 25 min stride for training
    'eval_stride': 1,         # Dense evaluation
    'batch_size': 64,
    'learning_rate': 0.001,
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


def preprocess_single_appliance(train_df, test_df, appliance_name):
    """
    Preprocess data for single-appliance NILM using RobustScaler with missing value handling.
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        appliance_name: Name of the target appliance
    
    Returns:
        Scaled data and scalers for input (aggregate) and output (single appliance)
    """
    # Extract aggregate power (input) and single appliance power (output)
    X_train_raw = train_df['Aggregate'].values.reshape(-1, 1)
    y_train_raw = train_df[appliance_name].values.reshape(-1, 1)
    
    X_test_raw = test_df['Aggregate'].values.reshape(-1, 1)
    y_test_raw = test_df[appliance_name].values.reshape(-1, 1)
    
    # Check for and handle missing/inf values
    for name, arr in [("X_train", X_train_raw), ("y_train", y_train_raw), 
                       ("X_test", X_test_raw), ("y_test", y_test_raw)]:
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️ {appliance_name} - {name}: {nan_count} NaN, {inf_count} Inf values")
            # Replace inf with nan, then fill with 0
            arr = np.where(np.isinf(arr), np.nan, arr)
            arr = np.nan_to_num(arr, nan=0.0)
            
            # Update the original arrays
            if name == "X_train":
                X_train_raw = arr
            elif name == "y_train":
                y_train_raw = arr
            elif name == "X_test":
                X_test_raw = arr
            elif name == "y_test":
                y_test_raw = arr
    
    # Normalize using RobustScaler (fit on training data only)
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
    """Train a model for a single appliance with early stopping."""
    from tqdm import tqdm
    
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print(f"Training {model_name} for {appliance_name}")
    print(f"{'='*80}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model state
    model.load_state_dict(best_model_state)
    return model


def evaluate_single_appliance_model(model, test_loader, criterion, device):
    """Evaluate a single-appliance model."""
    model.eval()
    all_predictions = []
    all_targets = []
    test_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    avg_test_loss = test_loss / len(test_loader)
    
    return predictions, targets, avg_test_loss


def calculate_single_appliance_metrics(targets, predictions, appliance_name, scaler_y):
    """Calculate metrics for a single appliance."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Sanitize predictions
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    predictions = np.clip(predictions, -8.0, 8.0)
    
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.clip(targets, -8.0, 8.0)
    
    # Inverse transform to original scale
    targets_real = scaler_y.inverse_transform(targets.astype(np.float64)).astype(np.float32)
    predictions_real = scaler_y.inverse_transform(predictions.astype(np.float64)).astype(np.float32)
    
    # Apply physical constraints
    load_appliances = ['EVSE', 'CS', 'BA']
    generation_appliances = ['PV', 'CHP']
    
    if appliance_name in load_appliances:
        predictions_real = np.maximum(predictions_real, 0)  # Loads are non-negative
    elif appliance_name in generation_appliances:
        predictions_real = np.minimum(predictions_real, 0)  # Generation is non-positive
    
    # Calculate metrics
    mae_w = mean_absolute_error(targets_real, predictions_real)
    mse_w = mean_squared_error(targets_real, predictions_real)
    r2 = r2_score(targets_real, predictions_real)
    
    # Convert to MW for readability
    mae_mw = mae_w / 1e6
    mse_mw = mse_w / (1e6 ** 2)
    
    # Calculate NDE (Normalized Disaggregation Error)
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
    
    return metrics, predictions_real, targets_real


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
        ),
        'LSTM': lambda: LSTMModel(
            input_size=CONFIG['input_size'], 
            hidden_size=CONFIG['lstm_hidden'], 
            num_layers=CONFIG['lstm_layers'],
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
            preds, targets, test_loss = evaluate_single_appliance_model(
                model, test_loader, criterion, device
            )
            
            # Calculate metrics
            metrics, preds_real, targets_real = calculate_single_appliance_metrics(
                targets, preds, appliance_name, scaler_y
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
