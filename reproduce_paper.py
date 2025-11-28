import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.dataset import load_data_by_location, preprocess_data, create_sequences, NILMDataset
from src.models import TCNModel, ATCNModel, BiLSTMModel, LSTMModel
from src.train import train_model, evaluate_model, calculate_metrics
from src.visualization import plot_results
from torch.utils.data import DataLoader
import numpy as np

# Configuration matching the paper
CONFIG = {
    'data_path': './AMDA_SIDED',
    'input_size': 1,
    'output_size': 5,
    'seq_length': 288,        # 24h window
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
    'tcn_layers': [128] * 8   # 8 layers of 128 channels
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # 2. Preprocess (RobustScaler)
    print("Preprocessing data (RobustScaler)...")
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y = preprocess_data(train_df, test_df)

    # 3. Create Sequences
    print("Creating sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, CONFIG['seq_length'], stride=CONFIG['train_stride'])
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, CONFIG['seq_length'], stride=CONFIG['eval_stride'])

    # Convert to Tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_train_tensor = torch.FloatTensor(y_train_seq)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_test_tensor = torch.FloatTensor(y_test_seq)

    # 4. DataLoaders
    train_dataset = NILMDataset(X_train_tensor, y_train_tensor)
    test_dataset = NILMDataset(X_test_tensor, y_test_tensor)
    
    # Split validation
    val_size = int(len(train_dataset) * 0.1)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
    val_loader = DataLoader(val_subset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])

    # 5. Initialize Models
    models = {
        'TCN': TCNModel(input_size=CONFIG['input_size'], num_channels=CONFIG['tcn_layers'], dropout=CONFIG['dropout']),
        'ATCN': ATCNModel(input_size=CONFIG['input_size'], num_channels=CONFIG['tcn_layers'], dropout=CONFIG['dropout']),
        # LSTM matches paper (3 layers, 128 hidden)
        'LSTM': LSTMModel(input_size=CONFIG['input_size'], hidden_size=128, num_layers=3)
    }

    criterion = nn.MSELoss()

    # 6. Train Loop
    for name, model in models.items():
        print(f"\nTraining {name}...")
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        
        # Scheduler setup (Warmup + Cosine)
        warmup_lambda = lambda epoch: (epoch + 1) / CONFIG['warmup_epochs'] if epoch < CONFIG['warmup_epochs'] else 1.0
        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'] - CONFIG['warmup_epochs'], eta_min=CONFIG['min_lr'])
        
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

        train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=CONFIG['num_epochs'],
            device=device,
            early_stopping_patience=CONFIG['early_stopping_patience'],
            model_name=name
        )

        # Evaluate
        print(f"Evaluating {name}...")
        preds, targets, test_loss = evaluate_model(model, test_loader, criterion, device)
        results, preds_real, targets_real = calculate_metrics(targets, preds, ['EVSE', 'PV', 'CS', 'CHP', 'BA'], scaler_y)
        
        # Save the best model
        save_path = Path(f"{name}_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved best {name} model to {save_path.absolute()}")

        # Save Metrics to JSON
        import json
        metrics_path = Path(f"{name}_metrics.json")
        
        # Convert numpy types to python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_results = {k: {m: convert_to_serializable(v) for m, v in metrics.items()} for k, metrics in results.items()}
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"📊 Saved metrics to {metrics_path.absolute()}")

        # Generate and Save Plots
        plot_path = Path(f"{name}_plot.png")
        plot_results(targets_real, preds_real, ['EVSE', 'PV', 'CS', 'CHP', 'BA'], title=f"{name} Evaluation", save_path=str(plot_path))

if __name__ == "__main__":
    main()
