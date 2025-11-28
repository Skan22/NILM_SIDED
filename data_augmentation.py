import pandas as pd
import numpy as np 
from pathlib import Path
import argparse
import sys

# Augmentation Function
def amda_augmentation(original_df: pd.DataFrame, s=2.5, appliance_columns=["EVSE","PV","CS","CHP","BA"]):
    # Calculating Total Power 
    augmented_df = original_df.copy()
    abs_power = np.abs(augmented_df[appliance_columns])
    P_Total = abs_power.sum().sum()
    
    # Scaling Appliances  
    for column in appliance_columns:
        P_total_i = abs_power[column].sum()
        if P_Total == 0:
            p_i = 0
        else:
            p_i = P_total_i / P_Total
        
        S_i = s * (1 - p_i)         
        augmented_df[column] *= S_i 
        
    # Re-Calculating Aggregate power 
    augmented_df["Aggregate"] = augmented_df[appliance_columns].sum(axis=1)
    
    return augmented_df

# Augmented DataSet creation function
def create_augmented_dataset(original_data_dir: Path,
                             augmented_data_dir: Path,
                             s_values=(1.5, 4.0),
                             include_original: bool = True,
                             aug_fn=amda_augmentation):
    """Augmented dataset creation.
    For each original CSV, generate multiple AMDA-augmented variants with the given s_values (alpha)
    and concatenate them (optionally including the original) into a single 'augmented_{file.name}'.
    """
    if not original_data_dir.exists():
        print(f"Error: Source directory '{original_data_dir}' does not exist.")
        sys.exit(1)

    if not augmented_data_dir.exists():
        augmented_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {augmented_data_dir}")

    for facility_dir in original_data_dir.iterdir():
        if not facility_dir.is_dir():
            continue
            
        aug_dir = augmented_data_dir / facility_dir.name
        if not aug_dir.exists():
            aug_dir.mkdir(parents=True, exist_ok=True)
            
        for file in facility_dir.iterdir():
            if not file.name.lower().endswith('.csv'):
                continue
                
            print(f"Processing {file.name}...")
            original_df = pd.read_csv(file)
            frames = []
            
            if include_original:
                frames.append(original_df)
                
            for s in s_values:
                # print(f"  - Augmenting with alpha={s}")
                frames.append(aug_fn(original_df, s=s))
                
            combined = pd.concat(frames, ignore_index=True)
            out_path = aug_dir / f"augmented_{file.name}"
            combined.to_csv(out_path, columns=original_df.columns.to_list(), index=False)
            print(f"  -> Wrote {len(combined)} rows to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate AMDA augmented dataset for NILM.")
    parser.add_argument("--input", "-i", type=str, default="./SIDED", help="Path to original data directory")
    parser.add_argument("--output", "-o", type=str, default="./AMDA_SIDED", help="Path to output augmented data directory")
    parser.add_argument("--alphas", "-a", type=float, nargs="+", default=[2.5], help="List of alpha (s) values for augmentation")
    parser.add_argument("--no-original", action="store_true", help="Do not include original data in the output files")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    print(f"Source: {input_path}")
    print(f"Destination: {output_path}")
    print(f"Alphas: {args.alphas}")
    print(f"Include Original: {not args.no_original}")
    
    create_augmented_dataset(
        original_data_dir=input_path,
        augmented_data_dir=output_path,
        s_values=tuple(args.alphas),
        include_original=not args.no_original
    )
    print("Augmentation complete.")

if __name__ == "__main__":
    main()

