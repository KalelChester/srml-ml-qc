import pandas as pd
import numpy as np
import os
import glob
from solar_features import add_features
from solar_model import SolarHybridModel

# --- CONFIGURATION ---
DATA_FOLDER = 'data'
TS_COL = 'YYYY-MM-DD--HH:MM:SS' 
HEADER_ROWS = 43  # This skips the top info block + daily summary

def load_and_prep_data(file_paths):
    df_list = []
    for f in file_paths:
        # Load raw data
        temp = pd.read_csv(f, skiprows=HEADER_ROWS)
        
        # Clean column names (strip whitespace)
        temp.columns = [c.strip() for c in temp.columns]
        
        # CRITICAL: Create a unique string key for every row
        # This preserves the exact format "2025-01-01--00:01:00"
        temp['_raw_ts'] = temp[TS_COL].astype(str).str.strip()
        
        # Create proper datetime for filtering
        temp['Timestamp_dt'] = pd.to_datetime(temp['_raw_ts'], format='%Y-%m-%d--%H:%M:%S')
        temp['_source_file'] = f 
        
        df_list.append(temp)
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Feature Engineering
    full_df = add_features(full_df)
    
    # Index by time for easy slicing
    full_df.index = full_df['Timestamp_dt']
    return full_df

def write_flags_to_csv(df_subset, target_col):
    """
    Writes predictions back to CSVs using the _raw_ts key for exact alignment.
    Counts how many flags actually changed.
    """
    total_changed = 0
    
    for file_path, group in df_subset.groupby('_source_file'):
        print(f"Processing {file_path}...")
        
        # 1. Load original file structure
        original_data = pd.read_csv(file_path, skiprows=HEADER_ROWS)
        original_data.columns = [c.strip() for c in original_data.columns]
        
        # 2. Build the Update Map: { "2025-01-01--00:01:00" : 99 }
        # Only contains rows that were in our prediction set
        new_flags_map = group.set_index('_raw_ts')[target_col].to_dict()
        
        # 3. Apply Updates
        # We iterate the ORIGINAL file. If a timestamp exists in our map, we use the new flag.
        # Otherwise, we keep the old one.
        
        # Helper to track changes
        def update_val(row):
            ts = str(row[TS_COL]).strip()
            old_val = row[target_col]
            
            if ts in new_flags_map:
                new_val = new_flags_map[ts]
                return new_val, (1 if new_val != old_val else 0)
            return old_val, 0

        # Run update
        # (Using a list comprehension is faster/cleaner than .apply for this double return)
        results = [update_val(row) for _, row in original_data.iterrows()]
        
        # Unzip results
        new_column_values = [r[0] for r in results]
        changes_count = sum(r[1] for r in results)
        total_changed += changes_count
        
        # Assign back
        original_data[target_col] = new_column_values
        original_data[target_col] = original_data[target_col].astype(int)

        # 4. Save with Header Preservation
        with open(file_path, 'r') as f:
            header_lines = [next(f) for _ in range(HEADER_ROWS)]
            
        with open(file_path, 'w', newline='') as f:
            f.writelines(header_lines)
            original_data.to_csv(f, index=False)
            
        print(f"  -> File updated. {changes_count} flags changed value.")

    print(f"Total flags changed across all files: {total_changed}")

def run_cycle(train_start, train_end, pred_start, pred_end, target_flag):
    print(f"--- Starting Active Learning Cycle for {target_flag} ---")
    all_files = sorted(glob.glob(os.path.join(DATA_FOLDER, '**', '*.csv'), recursive=True))
    
    print("Loading data...")
    full_dataset = load_and_prep_data(all_files)
    
    # --- TRAINING ---
    print(f"Training Period: {train_start} to {train_end}")
    train_mask = (full_dataset.index >= pd.to_datetime(train_start)) & \
                 (full_dataset.index <= pd.to_datetime(train_end))
    train_df = full_dataset[train_mask]
    
    if len(train_df) == 0:
        print("Error: No training data found.")
        return

    model = SolarHybridModel()
    model.fit(train_df, target_col=target_flag)
    
    # --- PREDICTION ---
    print(f"Prediction Period: {pred_start} to {pred_end}")
    pred_mask = (full_dataset.index >= pd.to_datetime(pred_start)) & \
                (full_dataset.index <= pd.to_datetime(pred_end))
    pred_df = full_dataset[pred_mask].copy()
    
    if len(pred_df) == 0:
        print("Error: No prediction data found.")
        return

    print(f"Predicting flags for {len(pred_df)} rows...")
    new_flags = model.predict(pred_df)
    pred_df[target_flag] = new_flags
    
    # --- WRITE BACK ---
    write_flags_to_csv(pred_df, target_flag)
    print("\nDone.")

if __name__ == "__main__":
    # --- USER SETTINGS ---
    # TARGET: Change this to 'Flag_DNI' or 'Flag_DHI' when ready
    TARGET = 'Flag_GHI' 
    
    # TIMEFRAME
    TRAIN_S, TRAIN_E = "2025-01-01 00:00:00", "2025-01-31 23:59:00"
    PRED_S,  PRED_E  = "2025-03-01 00:00:00", "2025-03-31 23:59:00"
    
    run_cycle(TRAIN_S, TRAIN_E, PRED_S, PRED_E, TARGET)