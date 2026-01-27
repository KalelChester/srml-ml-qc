import pandas as pd
import numpy as np
import os
import glob
from solar_features import add_features
from solar_model import SolarHybridModel

# --- CONFIGURATION ---
DATA_FOLDER = 'data'
TS_COL = 'YYYY-MM-DD--HH:MM:SS' 
HEADER_ROWS = 43 

def load_and_prep_data(file_paths):
    df_list = []
    print(f"Loading {len(file_paths)} files...")
    for f in file_paths:
        try:
            temp = pd.read_csv(f, skiprows=HEADER_ROWS)
            temp.columns = [c.strip() for c in temp.columns]
            
            temp['_raw_ts'] = temp[TS_COL].astype(str).str.strip()
            temp['Timestamp_dt'] = pd.to_datetime(temp['_raw_ts'], format='%Y-%m-%d--%H:%M:%S')
            temp['_source_file'] = f 
            df_list.append(temp)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    
    full_df = pd.concat(df_list, ignore_index=True)
    full_df = add_features(full_df)
    full_df.index = full_df['Timestamp_dt']
    return full_df

def write_flags_to_csv(df_subset, target_col):
    """
    Writes predictions back to CSVs.
    """
    total_changed = 0
    bad_count = (df_subset[target_col] == 99).sum()
    print(f"    -> Writing updates... (Includes {bad_count} 'Bad' flags)")

    for file_path, group in df_subset.groupby('_source_file'):
        original_data = pd.read_csv(file_path, skiprows=HEADER_ROWS)
        original_data.columns = [c.strip() for c in original_data.columns]
        
        new_flags_map = group.set_index('_raw_ts')[target_col].to_dict()
        
        def update_val(row):
            ts = str(row[TS_COL]).strip()
            old_val = row[target_col] if not pd.isna(row[target_col]) else 0
            if ts in new_flags_map:
                new_val = new_flags_map[ts]
                return new_val, (1 if new_val != old_val else 0)
            return old_val, 0

        results = [update_val(row) for _, row in original_data.iterrows()]
        
        new_column_values = [r[0] for r in results]
        total_changed += sum(r[1] for r in results)
        
        original_data[target_col] = new_column_values
        original_data[target_col] = original_data[target_col].fillna(0).astype(int)

        with open(file_path, 'r') as f:
            header_lines = [next(f) for _ in range(HEADER_ROWS)]
            
        with open(file_path, 'w', newline='') as f:
            f.writelines(header_lines)
            original_data.to_csv(f, index=False)
            
    print(f"    -> Total flags changed for {target_col}: {total_changed}")

def run_cycle(train_start, train_end, pred_start, pred_end, target_flags):
    print(f"--- Starting Active Learning Cycle ---")
    all_files = sorted(glob.glob(os.path.join(DATA_FOLDER, '**', '*.csv'), recursive=True))
    
    full_dataset = load_and_prep_data(all_files)
    
    # --- TRAINING DATA ---
    print(f"Training Period: {train_start} to {train_end}")
    train_mask = (full_dataset.index >= pd.to_datetime(train_start)) & \
                 (full_dataset.index <= pd.to_datetime(train_end))
    train_df = full_dataset[train_mask]
    
    # --- PREDICTION DATA (Loaded ONCE, updated in memory) ---
    print(f"Prediction Period: {pred_start} to {pred_end}")
    pred_mask = (full_dataset.index >= pd.to_datetime(pred_start)) & \
                (full_dataset.index <= pd.to_datetime(pred_end))
    pred_df = full_dataset[pred_mask].copy()
    
    if len(train_df) == 0 or len(pred_df) == 0:
        print("Error: Missing data for selected date ranges.")
        return

    for target in target_flags:
        print(f"\n=== Processing {target} ===")
        
        if target not in train_df.columns:
            print(f"Skipping {target}: Column missing.")
            continue
            
        # Initialize fresh model for this target
        model = SolarHybridModel()
        model.fit(train_df, target_col=target)
        
        print(f"Predicting {target}...")
        
        # Note: Pass the 'pred_df' which might have updates from previous loops
        # (Though currently we aren't using Flag_X as a feature for Flag_Y, 
        # this structure allows it if we add it to features_map later)
        new_flags = model.predict(pred_df, target_col=target)
        pred_df[target] = new_flags
        
        # Write to disk immediately
        write_flags_to_csv(pred_df, target)

if __name__ == "__main__":
    TARGETS = ['Flag_DNI', 'Flag_DHI', 'Flag_GHI'] 
    
    TRAIN_S, TRAIN_E = "2025-01-01 00:00:00", "2025-03-31 23:59:00"
    PRED_S,  PRED_E  = "2025-04-01 00:00:00", "2025-06-30 23:59:00"
    
    run_cycle(TRAIN_S, TRAIN_E, PRED_S, PRED_E, TARGETS)