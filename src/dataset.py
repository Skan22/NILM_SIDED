import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import RobustScaler

class NILMDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(X, y, seq_length, stride=1, target_pos='mid'):
    """Create sequences for S2P models.
    - X: (N, input_dim)
    - y: (N, output_dim)
    - seq_length: window length (e.g., 288)
    - stride: step between windows (e.g., 5 for 25 minutes)
    - target_pos: 'mid' uses midpoint of the window; 'end' uses next step after window
    Returns: X_seq (M, seq_len, input_dim), y_seq (M, output_dim)
    """
    X_seq, y_seq = [], []
    N = len(X)
    for i in range(0, N - seq_length, stride):
        start = i
        end = i + seq_length
        if target_pos == 'mid':
            t_idx = start + seq_length // 2
        elif target_pos == 'end':
            t_idx = end if end < N else (N - 1)
        else:
            t_idx = start + seq_length // 2
        X_seq.append(X[start:end])
        y_seq.append(y[t_idx])
    return np.array(X_seq), np.array(y_seq)

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
            if name == "X_train": X_train_raw = arr
            elif name == "y_train": y_train_raw = arr
            elif name == "X_test": X_test_raw = arr
            elif name == "y_test": y_test_raw = arr
    
    # Normalize using RobustScaler for X (Aggregate usually has variance)
    # Use StandardScaler for y (Appliance) because sparse data often has IQR=0, causing RobustScaler to fail
    from sklearn.preprocessing import StandardScaler
    scaler_X = RobustScaler()
    scaler_y = StandardScaler()
    
    print(f"  Raw Data Stats for {appliance_name}:")
    print(f"    X_train_raw: min={X_train_raw.min():.2f}, max={X_train_raw.max():.2f}, mean={X_train_raw.mean():.2f}, std={X_train_raw.std():.2f}")
    print(f"    y_train_raw: min={y_train_raw.min():.2f}, max={y_train_raw.max():.2f}, mean={y_train_raw.mean():.2f}, std={y_train_raw.std():.2f}")
    
    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)
    
    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)
    
    print(f"  Data Stats for {appliance_name}:")
    print(f"    X_train: min={X_train_scaled.min():.2f}, max={X_train_scaled.max():.2f}, mean={X_train_scaled.mean():.2f}, std={X_train_scaled.std():.2f}")
    print(f"    y_train: min={y_train_scaled.min():.2f}, max={y_train_scaled.max():.2f}, mean={y_train_scaled.mean():.2f}, std={y_train_scaled.std():.2f}")
    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y

def load_data_by_location(base_path='./AMDA_SIDED', target_locations=['Tokyo'], source_locations=['LA', 'Offenbach'], resample_rule='5min'):
    """
    Load data split by location for Domain Adaptation tasks with robust missing value handling.
    Args:
        base_path: Path to data
        target_locations: List of locations to use for testing (Target Domain)
        source_locations: List of locations to use for training (Source Domain)
        resample_rule: Pandas resampling rule (e.g., '5min' for 5 minutes). None to disable.
    """
    train_dfs = []
    test_dfs = []
    
    facilities = ['Dealer', 'Logistic', 'Office']
    appliance_columns = ['EVSE', 'PV', 'CS', 'CHP', 'BA', 'Aggregate']
    
    print(f"Loading Data from: {base_path}")
    if resample_rule:
        print(f"⚠️ Resampling data to {resample_rule} intervals (Paper Requirement)")
    
    for facility in facilities:
        # Load Source Domain (Training Data)
        for loc in source_locations:
            file_path = Path(base_path) / facility / f'augmented_{facility}_{loc}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Check for missing values BEFORE processing
                missing_before = df[appliance_columns].isnull().sum().sum()
                if missing_before > 0:
                    print(f"  ⚠️ Found {missing_before} missing values in {facility}_{loc}")
                    # Forward fill then backward fill to handle missing values
                    df[appliance_columns] = df[appliance_columns].ffill().bfill()
                    # If still NaN (entire column is NaN), fill with 0
                    df[appliance_columns] = df[appliance_columns].fillna(0)
                    print(f"  ✅ Missing values handled via forward/backward fill")
                
                # Resample if requested (Paper uses 5-min intervals)
                if resample_rule:
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp').resample(resample_rule).mean().dropna().reset_index()
                    else:
                        # Fallback: Group by index // 5 for 1-min to 5-min conversion
                        # Assuming 1-min data if no timestamp
                        df = df.groupby(df.index // 5).mean()
                
                # Final check after resampling
                missing_after = df[appliance_columns].isnull().sum().sum()
                if missing_after > 0:
                    print(f"  ⚠️ {missing_after} NaN values after resampling, filling with 0")
                    df[appliance_columns] = df[appliance_columns].fillna(0)

                df['facility'] = facility
                df['location'] = loc
                df['domain'] = 'source'
                train_dfs.append(df)
                print(f"  [TRAIN/Source] Loaded {facility}_{loc}: {len(df)} samples")
            else:
                print(f"  [WARN] File not found: {file_path}")
                
        # Load Target Domain (Testing Data)
        for loc in target_locations:
            file_path = Path(base_path) / facility / f'augmented_{facility}_{loc}.csv'
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Check for missing values BEFORE processing
                missing_before = df[appliance_columns].isnull().sum().sum()
                if missing_before > 0:
                    print(f"  ⚠️ Found {missing_before} missing values in {facility}_{loc}")
                    df[appliance_columns] = df[appliance_columns].ffill().bfill()
                    df[appliance_columns] = df[appliance_columns].fillna(0)
                    print(f"  ✅ Missing values handled via forward/backward fill")
                
                # Resample
                if resample_rule:
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.set_index('timestamp').resample(resample_rule).mean().dropna().reset_index()
                    else:
                        df = df.groupby(df.index // 5).mean()
                
                # Final check after resampling
                missing_after = df[appliance_columns].isnull().sum().sum()
                if missing_after > 0:
                    print(f"  ⚠️ {missing_after} NaN values after resampling, filling with 0")
                    df[appliance_columns] = df[appliance_columns].fillna(0)

                df['facility'] = facility
                df['location'] = loc
                df['domain'] = 'target'
                test_dfs.append(df)
                print(f"  [TEST/Target]  Loaded {facility}_{loc}: {len(df)} samples")
            else:
                print(f"  [WARN] File not found: {file_path}")
    
    if not train_dfs or not test_dfs:
        raise ValueError("Could not load data. Check paths and locations.")

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Final validation: Check for any remaining NaN or inf values
    print(f"\n{'='*60}")
    print("Data Quality Check:")
    print(f"{'='*60}")
    
    for df_name, df in [("Training", train_df), ("Testing", test_df)]:
        nan_count = df[appliance_columns].isnull().sum().sum()
        inf_count = np.isinf(df[appliance_columns].select_dtypes(include=[np.number])).sum().sum()
        
        print(f"{df_name} Data:")
        print(f"  Total Samples: {len(df)}")
        print(f"  NaN values: {nan_count}")
        print(f"  Inf values: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print(f"  ⚠️ WARNING: Data quality issues detected!")
            # Clean up any remaining issues
            df[appliance_columns] = df[appliance_columns].replace([np.inf, -np.inf], np.nan)
            df[appliance_columns] = df[appliance_columns].fillna(0)
            print(f"  ✅ Cleaned: Replaced inf/nan with 0")
    
    print(f"{'='*60}\n")
    print(f"Total Training Samples (Source): {len(train_df)}")
    print(f"Total Testing Samples (Target): {len(test_df)}")
    
    return train_df, test_df

