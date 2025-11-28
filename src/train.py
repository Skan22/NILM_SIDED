import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.amp.autocast_mode import autocast 
from torch.amp.grad_scaler import GradScaler


def train_single_appliance_model(model, train_loader, val_loader, criterion, optimizer, 
                                  scheduler, num_epochs, device, early_stopping_patience, 
                                  model_name, appliance_name):
    """
    Optimized training loop for single-appliance models.
    Includes: AMP, Gradient Clipping, set_to_none=True, Parameter Logging, and Robustness Checks.
    """
    model.to(device)
    use_amp = (device.type == 'cuda')
    scaler = GradScaler(enabled=use_amp,device="cuda")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in tqdm(range(num_epochs), desc=f'Training {model_name} | {appliance_name}'):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        for batch_X, batch_y in progress_bar:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            
            
            with autocast(enabled=use_amp,device_type="cuda"):
                outputs = model(batch_X)
                # Detect exploding outputs early
                if not torch.isfinite(outputs).all():
                    print("⚠️ Detected non-finite model outputs. Clamping and continuing.")
                    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = criterion(outputs, batch_y)
            optimizer.zero_grad(set_to_none=True)
            if (not math.isfinite(loss.item())) or math.isnan(loss.item()):
                print(f"❌ Invalid loss (NaN/Inf) at epoch {epoch+1}. Stopping training for {model_name}.")
                break
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / max(1, len(train_loader))
        
        model.eval()
        val_loss = 0.0
        
        with torch.inference_mode():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                with autocast(enabled=use_amp,device_type="cuda"):
                    outputs = model(batch_X)
                    if not torch.isfinite(outputs).all():
                        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
                    loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / max(1, len(val_loader))
        epoch_time = time.time() - epoch_start_time
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"⚠️ Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"✅ Restored best model with val_loss: {best_val_loss:.6f}")
    
    return model


def evaluate_single_appliance_model(model, test_loader, device, scaler_y, appliance_name):
    """
    Optimized evaluation for single-appliance models using inference_mode and tqdm.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.inference_mode():
        for batch_X, batch_y in tqdm(test_loader, desc=f'Evaluating {appliance_name}', leave=False):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            outputs = model(batch_X)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Sanitize predictions
    if not np.isfinite(predictions).all():
        print("⚠️ Non-finite predictions detected. Replacing with zeros.")
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
        
    metrics = calculate_single_appliance_metrics(targets, predictions, appliance_name, scaler_y)
    metrics['parameters'] = count_parameters(model)
    
    # Inverse transform for returning real values
    predictions_real = scaler_y.inverse_transform(predictions.astype(np.float64)).astype(np.float32)
    targets_real = scaler_y.inverse_transform(targets.astype(np.float64)).astype(np.float32)
    
    return metrics, predictions_real, targets_real


def calculate_single_appliance_metrics(targets, predictions, appliance_name, scaler_y):
    """Calculate metrics for a single appliance with robust sanitization."""
    
    # Sanitize inputs
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    predictions = np.clip(predictions, -10.0, 10.0) # Clip scaled values to reasonable range
    
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.clip(targets, -10.0, 10.0)
    
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
    
    return metrics
