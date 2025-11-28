import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_saved_model_robust(model, model_path, test_loader, device, appliance_names, scaler_y):
    """
    Load a saved model and evaluate with robust metrics.
    Clamps predictions in standardized space before inverse_transform to prevent overflow.
    """
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print(f"Evaluating {model_path.name}...")
    
    with torch.inference_mode():
        for batch_X, batch_y in tqdm(test_loader, desc='Evaluating', leave=False):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            # No AMP in evaluation for simplicity
            outputs = model(batch_X)
            
            # Clamp outputs in standardized space (prevent extreme values)
            outputs = torch.clamp(outputs, min=-8.0, max=8.0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Additional safety: sanitize before inverse_transform
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    predictions = np.clip(predictions, -8.0, 8.0)
    
    targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.clip(targets, -8.0, 8.0)
    
    # Inverse transform with float64 to reduce overflow risk
    targets_real = scaler_y.inverse_transform(targets.astype(np.float64)).astype(np.float32)
    predictions_real = scaler_y.inverse_transform(predictions.astype(np.float64)).astype(np.float32)
    
    # Final sanitization
    def sanitize_final(arr, name):
        if not np.isfinite(arr).all():
            print(f"⚠️ Sanitizing non-finite values in {name}")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    
    predictions_real = sanitize_final(predictions_real, 'predictions')
    targets_real = sanitize_final(targets_real, 'targets')
    
    # Apply appliance-specific clipping
    load_appliances = ['EVSE', 'CS', 'BA']
    generation_appliances = ['PV', 'CHP']
    
    for i, app_name in enumerate(appliance_names):
        if app_name in load_appliances:
            predictions_real[:, i] = np.maximum(predictions_real[:, i], 0)
        elif app_name in generation_appliances:
            predictions_real[:, i] = np.minimum(predictions_real[:, i], 0)
    
    # Calculate metrics
    print("\n" + "="*80)
    print(f"       {model_path.stem.upper()} - PERFORMANCE METRICS")
    print("="*80)
    print(f"{'Appliance':<10} | {'MAE (W)':<10} | {'MAE (MW)':<10} | {'MSE (MW²)':<12} | {'R2 Score':<10}")
    print("-" * 80)
    
    results = {}
    for i, app_name in enumerate(appliance_names):
        mae_w = mean_absolute_error(targets_real[:, i], predictions_real[:, i])
        mse_w = mean_squared_error(targets_real[:, i], predictions_real[:, i])
        r2 = r2_score(targets_real[:, i], predictions_real[:, i])
        mae_mw = mae_w / 1e6
        mse_mw = mse_w / (1e6 ** 2)
        
        results[app_name] = {'MAE_W': mae_w, 'MAE_MW': mae_mw, 'MSE_W': mse_w, 'MSE_MW2': mse_mw, 'R2': r2}
        print(f"{app_name:<10} | {mae_w:<10.2f} | {mae_mw:<10.6f} | {mse_mw:<12.6f} | {r2:<10.4f}")
    
    print("="*80 + "\n")
    return results, predictions_real, targets_real

def plot_results(targets, predictions, appliance_names, title="Model Evaluation", save_path=None):
    """
    Plot predicted vs actual values for each appliance.
    """
    num_appliances = len(appliance_names)
    fig, axes = plt.subplots(num_appliances, 1, figsize=(15, 4 * num_appliances), sharex=True)
    
    if num_appliances == 1:
        axes = [axes]
    
    for i, app_name in enumerate(appliance_names):
        ax = axes[i]
        # Plot a subset of data for clarity (e.g., first 1000 points)
        subset = 1000
        ax.plot(targets[:subset, i], label='Actual', alpha=0.7)
        ax.plot(predictions[:subset, i], label='Predicted', alpha=0.7, linestyle='--')
        ax.set_title(f'{app_name} - Actual vs Predicted')
        ax.set_ylabel('Power (W)')
        ax.legend()
        ax.grid(True)
    
    plt.xlabel('Time Step')
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
