import pandas  as pd
import numpy as np 
from pathlib import Path
import pathlib

#Augmentation Function
def amda_augmentation(original_df:pd.DataFrame, s=2.5, appliance_columns=["EVSE","PV","CS","CHP","BA"]):
    #Calculating Total Power 
    augmented_df = original_df.copy()
    abs_power= np.abs(augmented_df[appliance_columns])
    P_Total = abs_power.sum().sum()
    #Scaling Appliances  
    for column in appliance_columns :
        P_total_i = abs_power[column].sum()
        p_i = P_total_i/P_Total
        S_i = s *(1-p_i)         
        augmented_df[column]*=S_i 
    #Re-Calculating Aggregate power 
    augmented_df["Aggregate"] = augmented_df[appliance_columns].sum(axis=1)
    
    return augmented_df

# Augmented DataSet creation funcion (Assumes the original dir follows the structure of SIDED->Facilities->CSV file of different locations)
def create_augmented_dataset(original_data_dir :Path,augmented_data_dir:Path,aug_fn=amda_augmentation):
    """ Augmented DataSet creation funcion (Assumes the original dir follows
      the structure of SIDED->Facilities->CSV file of different locations)"""
    if augmented_data_dir.exists()==False:
        augmented_data_dir.mkdir()

    for dir in original_data_dir.iterdir():
        aug_dir = Path(augmented_data_dir/Path(dir.name))
        if aug_dir.exists()==False:
            aug_dir.mkdir()
        for file in dir.iterdir():
            original_df = pd.read_csv(file)
            print(f"Augmenting {file.name}")
            augmented_df = amda_augmentation(original_df)
            augmented_df.to_csv(f"./{aug_dir}/augmented_{file.name}",columns=original_df.columns.to_list(),index=False)




if __name__ == "__main__":
    create_augmented_dataset(original_data_dir=Path(r"./SIDED"),augmented_data_dir=Path(r"./AMDA_SIDED"))

