import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, 
                 early_stopping_patience=5, model_name='Model', gradient_clip=1.0):
    """Optimized training with validation, early stopping, AMP, and monitoring"""
    model.to(device)
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    history = {'train_loss': [], 'val_loss': [], 'epoch_times': [], 'learning_rates': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\\n{'='*60}\\nTraining {model_name}\\n{'='*60}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        for batch_X, batch_y in progress_bar:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch_X)
                # Detect exploding outputs early
                if not torch.isfinite(outputs).all():
                    print("⚠️ Detected non-finite model outputs. Clamping and continuing.")
                    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = criterion(outputs, batch_y)
            
            if (not math.isfinite(loss.item())) or math.isnan(loss.item()):
                print(f"❌ Invalid loss (NaN/Inf) at epoch {epoch+1}. Stopping training for {model_name}.")
                return history
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / max(1, len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(batch_X)
                    if not torch.isfinite(outputs).all():
                        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
                    loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / max(1, len(val_loader))
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['epoch_times'].append(epoch_time)
        history['learning_rates'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")
        scheduler.step()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"⚠️ Early stopping at epoch {epoch+1}. Best val loss: {best_val_loss:.6f}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Restored best model with val_loss: {best_val_loss:.6f}")
    return history

def evaluate_model(model, test_loader, criterion, device):
    """Optimized evaluation with memory efficiency + AMP and stability checks"""
    model.to(device)
    model.eval()
    use_amp = (device.type == 'cuda')
    all_predictions = []
    all_targets = []
    test_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc='Evaluating', leave=False):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch_X)
                if not torch.isfinite(outputs).all():
                    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Sanitize any non-finite values before metric computations
    if not np.isfinite(predictions).all():
        print("⚠️ Non-finite predictions detected. Replacing with zeros.")
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
    if not np.isfinite(targets).all():
        print("⚠️ Non-finite targets detected. Replacing with zeros.")
        targets = np.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
    
    avg_test_loss = test_loss / max(1, len(test_loader))
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return predictions, targets, avg_test_loss

def calculate_metrics(targets, predictions, appliance_names, scaler_y):
    """Calculate and print metrics per appliance with robust sanitization"""
    results = {}
    
    # Inverse transform to get real power values (Watts)
    targets_real = scaler_y.inverse_transform(targets)
    predictions_real = scaler_y.inverse_transform(predictions)
    
    # Sanitize non-finite values post inverse transform
    def sanitize(arr, name):
        if not np.isfinite(arr).all():
            finite_mask = np.isfinite(arr)
            if finite_mask.any():
                median_vals = np.median(arr[finite_mask], axis=0)
                arr[~finite_mask] = median_vals
            else:
                arr[:] = 0.0
            print(f"⚠️ Sanitized non-finite values in {name}.")
        # Clamp extreme outliers to 10x max(abs(targets)) to avoid metric distortion
        max_ref = np.max(np.abs(targets_real)) if np.isfinite(targets_real).any() else 1.0
        arr[:] = np.clip(arr, -10*max_ref, 10*max_ref)
        return arr
    
    predictions_real = sanitize(predictions_real, 'predictions_real')
    targets_real = sanitize(targets_real, 'targets_real')
    
    # Correctly handle negative values (Generation vs Load)
    load_appliances = ['EVSE', 'CS', 'BA']
    generation_appliances = ['PV', 'CHP']
    
    print("\\n" + "="*80)
    print("       PER-APPLIANCE PERFORMANCE METRICS")
    print("="*80)
    print(f"{'Appliance':<10} | {'MAE (W)':<10} | {'MAE (MW)':<10} | {'MSE (MW²)':<12} | {'R2 Score':<10}")
    print("-" * 80)
    
    for i, app_name in enumerate(appliance_names):
        if app_name in load_appliances:
            predictions_real[:, i] = np.maximum(predictions_real[:, i], 0)
        elif app_name in generation_appliances:
            predictions_real[:, i] = np.minimum(predictions_real[:, i], 0)
            
        mae_w = mean_absolute_error(targets_real[:, i], predictions_real[:, i])
        mse_w = mean_squared_error(targets_real[:, i], predictions_real[:, i])
        r2 = r2_score(targets_real[:, i], predictions_real[:, i])
        mae_mw = mae_w / 1e6
        mse_mw = mse_w / (1e6 ** 2)
        
        results[app_name] = {'MAE_W': mae_w, 'MAE_MW': mae_mw, 'MSE_W': mse_w, 'MSE_MW2': mse_mw, 'R2': r2}
        print(f"{app_name:<10} | {mae_w:<10.2f} | {mae_mw:<10.6f} | {mse_mw:<12.6f} | {r2:<10.4f}")
    
    print("="*80)
    return results, predictions_real, targets_real
