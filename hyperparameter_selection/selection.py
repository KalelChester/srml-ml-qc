"""
selection.py
=================

Hyperparameter selection driver script for solar QC models.

OPTIMIZED ARCHITECTURE:
1. Loads Raw Data.
2. Generates Features ONCE (using static ../solar_features.py).
3. Iterates through Model Hyperparameters (Grid Search).
4. Performs K-Fold CV on the pre-processed data.

This setup is significantly faster for tuning Neural Network and Random Forest 
parameters because it avoids recalculating rolling correlations and solar 
geometry for every iteration.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ---------------- IMPORT SETUP ----------------
# Add parent directory to path to find the static 'solar_features.py'
sys.path.append('../') 

try:
    from solar_features import add_features
except ImportError:
    print("CRITICAL ERROR: Could not import 'solar_features'.")
    print("Ensure 'solar_features.py' exists in the parent directory '../'")
    sys.exit(1)

from solar_model_modular import SolarHybridModel


# ---------------- CONFIG ----------------
DATA_FOLDER = 'data'
HEADER_ROWS = 43
TS_COL = 'YYYY-MM-DD--HH:MM:SS'

# Persistence files
HYPER_FILE = 'hyperparameters.csv'
LOG_FILE = 'feature_performance.log'

# ----------------- TEST CONFIGURATION (EDIT HERE) -----------------
# Examples: 'nn_learning_rate', 'rf_n_estimators', 'nn_layers', 'nn_batch_size'
PARAM_TO_TEST = 'nn_learning_rate'

# Select the range of values to test for this parameter
TEST_VALUES = [0.01, 0.005, 0.001, 0.0001]

SITE_CONFIG = {
    'latitude': 47.654,     # <-- SET ME
    'longitude': -122.309,  # <-- SET ME
    'altitude': 70,   # meters (optional)
    'timezone': 'Etc/GMT+8'
}

# Target column to train on
TARGET = 'Flag_GHI'


# ---------------- I/O helpers ----------------
def load_hyperparameters(filepath):
    """
    Reads a CSV with columns [parameter, value].
    Returns a dictionary of params.
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Returning empty dict.")
        return {}
        
    df = pd.read_csv(filepath)
    params = {}
    for _, row in df.iterrows():
        name = row['parameter']
        val_str = str(row['value'])
        try:
            # Safely evaluate literals (numbers, tuples, lists)
            val = ast.literal_eval(val_str)
        except:
            # Fallback to string if eval fails
            val = val_str
        params[name] = val
    return params

def load_raw_data():
    """
    Loads all CSVs from DATA_FOLDER.
    """
    files = sorted(glob.glob(os.path.join('..', DATA_FOLDER, '**', '*.csv'), recursive=True))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, skiprows=HEADER_ROWS)
            df.columns = [c.strip() for c in df.columns]
            if TS_COL in df.columns:
                df['_raw_ts'] = df[TS_COL].astype(str).str.strip()
            else:
                df['_raw_ts'] = df.iloc[:, 0].astype(str).str.strip()
            frames.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------- MAIN ----------------
def run_selection_cycle():
    # 1. Load Baseline Parameters
    base_params = load_hyperparameters(HYPER_FILE)
    print(f"Base Parameters loaded: {base_params}")
    
    # 2. Load Data
    print("Loading raw data...")
    raw_df = load_raw_data()
    if raw_df.empty:
        print("No data found in 'data' folder.")
        return

    # 3. Generate Features ONCE (The Speedup)
    print("Generating features (Static)...")
    # We do NOT pass 'params' here anymore, using standard defaults/logic in solar_features
    df_full = add_features(raw_df, SITE_CONFIG)
    
    # Pre-filter for valid targets to save memory/time in loop if possible, 
    # but strictly we need to respect K-Fold indices, so we keep full structure.
    print(f"Feature generation complete. Shape: {df_full.shape}")
    
    print(f"\nStarting Grid Search for: {PARAM_TO_TEST}")
    print(f"Values to test: {TEST_VALUES}")

    # 4. Iteration Loop (Model Params Only)
    for val in TEST_VALUES:
        print(f"\n--- Testing {PARAM_TO_TEST} = {val} ---")
        
        # Merge baseline with current test value
        current_params = base_params.copy()
        current_params[PARAM_TO_TEST] = val
        
        # 5. K-Fold Cross Validation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'mod_accuracy': []
        }

        fold_count = 0
        for train_idx, val_idx in kf.split(df_full):
            fold_count += 1
            train_sub = df_full.iloc[train_idx].copy()
            val_sub = df_full.iloc[val_idx].copy()
            
            # Filter for labeled data only
            train_labeled = train_sub[train_sub[TARGET].notna()]
            val_labeled = val_sub[val_sub[TARGET].notna()]
            
            if train_labeled.empty or val_labeled.empty:
                continue

            # Instantiate Model with current params
            # The model handles the already-generated features automatically
            model = SolarHybridModel(params=current_params)
            
            # Fit
            model.fit(train_labeled, target_col=TARGET)
            
            # Predict
            flags = model.predict(val_labeled, TARGET)
            
            # Evaluation
            # Map 99(BAD)->0, 1(GOOD)->1 for sklearn metrics
            y_true = np.where(val_labeled[TARGET] == 1, 1, 0)
            y_pred = np.where(flags == 1, 1, 0)
            
            # Standard Metrics
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            
            # Modified Accuracy Calculation
            prop_0 = np.mean(y_true == 0)
            prop_1 = np.mean(y_true == 1)
            balance = max(prop_0, prop_1)
            
            if balance >= 1.0 - 1e-9:
                mod_acc = 0.0 
            else:
                mod_acc = (acc - balance) / (1.0 - balance)

            metrics['precision'].append(p)
            metrics['recall'].append(r)
            metrics['f1'].append(f1)
            metrics['mod_accuracy'].append(mod_acc)

        # 6. Aggregate Results
        avg_p = np.mean(metrics['precision'])
        avg_r = np.mean(metrics['recall'])
        avg_f1 = np.mean(metrics['f1'])
        avg_mod_acc = np.mean(metrics['mod_accuracy'])
        
        # 7. Logging
        log_entry = (
            f"PARAM: {PARAM_TO_TEST}={val} | "
            f"F1: {avg_f1:.4f} | "
            f"ModAcc: {avg_mod_acc:.4f} | "
            f"Prec: {avg_p:.4f} | "
            f"Rec: {avg_r:.4f} | "
            f"FullParams: {current_params}"
        )
        
        print(f"   >>> Result: F1={avg_f1:.4f}, ModAcc={avg_mod_acc:.4f}")
        
        with open(LOG_FILE, 'a') as f:
            f.write(log_entry + "\n")

    print(f"\nCompleted testing for {PARAM_TO_TEST}. Check {LOG_FILE} for details.")

if __name__ == '__main__':
    run_selection_cycle()