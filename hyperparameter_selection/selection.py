"""
selection.py
=================

Hyperparameter selection driver script for solar QC models with RNN.

ARCHITECTURE:
1. Loads Raw Data from manual_confirmed_data_backup (one month per file).
2. Generates Features ONCE (using static ../solar_features.py).
3. Performs Block-Based Train/Test Split (by month, not random sampling).
4. Trains RNN on temporal sequences without breaking temporal structure.
5. Evaluates on held-out months/periods.
6. Iterates through Model Hyperparameters (Grid Search).

Block-based splitting is critical for RNN: sequences must be contiguous by month/day,
not randomly shuffled. This preserves temporal patterns learned by the model.
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Add parent directory to path to find solar_features.py
sys.path.append('../') 

try:
    from solar_features import add_features
    from config import SITE_CONFIG
except ImportError:
    print("CRITICAL ERROR: Could not import 'solar_features'.")
    print("Ensure 'solar_features.py' exists in the parent directory '../'")
    sys.exit(1)

from solar_model_modular import SolarHybridModel


# ============ CONFIG ============
DATA_FOLDER = '../manual_confirmed_data_backup'
HEADER_ROWS = 43
TS_COL = 'YYYY-MM-DD--HH:MM:SS'

# Persistence files
HYPER_FILE = 'hyperparameters.csv'
LOG_FILE = 'feature_performance.log'

# ============ TEST CONFIGURATION (EDIT HERE) ============
# RNN hyperparameters to test: seq_length, hidden_dim, num_layers, dropout_rate, nn_epochs, etc.
PARAM_TO_TEST = 'hidden_dim'
TEST_VALUES = [32, 64, 128]

# Target columns to train on - tests all three irradiance components
TARGETS = ['Flag_GHI', 'Flag_DNI', 'Flag_DHI']

# ============ BLOCK-BASED SPLITTING CONFIG ============
# For RNN training, we use contiguous blocks (months) instead of random splitting
# Each CSV file = 1 month of data at 1-minute resolution
# Block-based split: use first N months for training, remaining for testing
TRAIN_MONTHS = 2  # Use first 2 months for training
TEST_MONTHS = 1   # Use next 1 month for testing


# ============ I/O HELPERS ============
def load_hyperparameters(filepath):
    """
    Reads a CSV with columns [parameter, value].
    Returns a dictionary of params with proper types.
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Returning empty dict.")
        return {}
        
    df = pd.read_csv(filepath)
    params = {}
    
    # Parameters that must be integers
    INT_PARAMS = {'rf_n_estimators', 'rf_max_depth', 'seq_length', 'hidden_dim', 'num_layers', 'nn_batch_size', 'nn_epochs'}
    
    for _, row in df.iterrows():
        name = row['parameter']
        val_str = str(row['value']).strip()
        
        try:
            # Try to parse as Python literal
            val = ast.literal_eval(val_str)
            
            # Convert to int if required
            if name in INT_PARAMS:
                val = int(val)
            
        except:
            # Fallback to string
            val = val_str
        
        params[name] = val
    
    return params


def load_raw_data_by_months():
    """
    Loads CSVs from DATA_FOLDER as a list of dataframes, each representing one month.
    Returns list of (filename, dataframe) tuples in sorted order.
    
    Important: Each CSV = 1 month of 1-minute data. We maintain this structure
    for proper block-based splitting.
    """
    data_path = os.path.join('..', 'manual_confirmed_data_backup', '*.csv')
    files = sorted(glob.glob(data_path))
    
    month_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, skiprows=HEADER_ROWS)
            df.columns = [c.strip() for c in df.columns]
            if TS_COL in df.columns:
                df['_raw_ts'] = df[TS_COL].astype(str).str.strip()
            else:
                df['_raw_ts'] = df.iloc[:, 0].astype(str).str.strip()
            month_dfs.append((os.path.basename(f), df))
        except Exception as e:
            print(f"Skipping {f}: {e}")
    
    return month_dfs


def block_train_test_split(month_dfs, train_months, test_months):
    """
    Perform block-based train/test split on monthly data.
    
    Parameters
    ----------
    month_dfs : list of (filename, dataframe) tuples
        Monthly dataframes in chronological order
    train_months : int
        Number of months to use for training
    test_months : int
        Number of months to use for testing
    
    Yields
    ------
    train_df, test_df : (pd.DataFrame, pd.DataFrame)
        Training and testing dataframes for each fold
    fold_info : str
        Description of the fold (months used)
    """
    n_months = len(month_dfs)
    
    if n_months < (train_months + test_months):
        print(f"WARNING: Only {n_months} months available, need {train_months + test_months}")
        print("Adjusting: will use available months for training/testing")
    
    # Generate folds: slide window across available months
    fold_idx = 0
    for start_idx in range(n_months - train_months - test_months + 1):
        fold_idx += 1
        
        # Get training months
        train_files = month_dfs[start_idx : start_idx + train_months]
        train_frames = [df for _, df in train_files]
        train_df = pd.concat(train_frames, ignore_index=True)
        
        # Get testing months
        test_files = month_dfs[start_idx + train_months : start_idx + train_months + test_months]
        test_frames = [df for _, df in test_files]
        test_df = pd.concat(test_frames, ignore_index=True)
        
        # Generate fold info
        train_months_str = ', '.join([f[0] for f in train_files])
        test_months_str = ', '.join([f[0] for f in test_files])
        fold_info = f"Fold {fold_idx}: Train=[{train_months_str}] Test=[{test_months_str}]"
        
        yield train_df, test_df, fold_info


# ============ MAIN ============
def run_selection_cycle():
    # 1. Load Baseline Parameters
    base_params = load_hyperparameters(HYPER_FILE)
    print(f"Base Parameters loaded: {base_params}")
    print(f"RNN Focal Model Configuration:")
    print(f"  - Sequence Length: {base_params.get('seq_length', 'Not set')} minutes")
    print(f"  - Hidden Dim: {base_params.get('hidden_dim', 'Not set')}")
    print(f"  - Num Layers: {base_params.get('num_layers', 'Not set')}")
    print(f"  - Dropout Rate: {base_params.get('dropout_rate', 'Not set')}")
    print(f"  - Target Columns: {TARGETS}")
    
    # 2. Load Data by months (maintains temporal structure)
    print("\nLoading monthly data from manual_confirmed_data_backup...")
    month_dfs = load_raw_data_by_months()
    if not month_dfs:
        print("No data found in 'manual_confirmed_data_backup' folder.")
        return
    
    print(f"Loaded {len(month_dfs)} months of data")
    for fname, df in month_dfs:
        print(f"  {fname}: {len(df)} rows")
    
    # 3. Generate Features ONCE
    print("\nGenerating features for all data (Static)...")
    all_frames = [df for _, df in month_dfs]
    raw_df = pd.concat(all_frames, ignore_index=True)
    df_full = add_features(raw_df, SITE_CONFIG)
    print(f"Feature generation complete. Total shape: {df_full.shape}")
    
    # Map back to monthly structure (for block-based split)
    # This is approximate but maintains temporal ordering
    month_dfs_featured = []
    idx_start = 0
    for fname, orig_df in month_dfs:
        n_rows = len(orig_df)
        month_dfs_featured.append((fname, df_full.iloc[idx_start : idx_start + n_rows].copy()))
        idx_start += n_rows
    
    print(f"\nStarting Grid Search for: {PARAM_TO_TEST}")
    print(f"Values to test: {TEST_VALUES}")
    print(f"Block-based splitting: {TRAIN_MONTHS} training months, {TEST_MONTHS} testing months")
    print(f"Testing all targets: {TARGETS}\n")

    # 4. Iteration Loop (Model Params Only)
    for val in TEST_VALUES:
        print(f"\n{'='*80}")
        print(f"Testing {PARAM_TO_TEST} = {val}")
        print(f"{'='*80}")
        
        # Merge baseline with current test value
        current_params = base_params.copy()
        current_params[PARAM_TO_TEST] = val
        
        # Store results for all targets
        all_target_metrics = {}
        
        # 5. Loop through each target column
        for target in TARGETS:
            print(f"\n  Target: {target}")
            print(f"  {'-'*70}")
            
            # Metrics storage for this target
            metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'mod_accuracy': []
            }
            
            fold_count = 0
            for train_df, test_df, fold_info in block_train_test_split(
                month_dfs_featured, TRAIN_MONTHS, TEST_MONTHS
            ):
                fold_count += 1
                
                # Filter for labeled data only
                train_labeled = train_df[train_df[target].notna()].copy()
                test_labeled = test_df[test_df[target].notna()].copy()
                
                if train_labeled.empty or test_labeled.empty:
                    continue
                
                # Instantiate Model with RNN architecture
                model = SolarHybridModel(params=current_params, use_rnn=True)
                
                # Fit on training block
                try:
                    model.fit(train_labeled, target_col=target)
                except Exception as e:
                    print(f"    ERROR during fit: {e}")
                    continue
                
                # Predict on test block
                try:
                    flags = model.predict(test_labeled, target)
                except Exception as e:
                    print(f"    ERROR during predict: {e}")
                    continue
                
                # Convert to binary predictions
                y_true = np.where(test_labeled[target] == 99, 0, 1)
                y_pred = np.where(flags == 1, 1, 0)
                
                # Calculate Metrics
                acc = accuracy_score(y_true, y_pred)
                p = precision_score(y_true, y_pred, zero_division=0)
                r = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Modified Accuracy (accounts for class imbalance)
                prop_0 = np.mean(y_true == 0)
                prop_1 = np.mean(y_true == 1)
                balance = max(prop_0, prop_1)
                
                if balance >= 1.0 - 1e-9:
                    mod_acc = 0.0 
                else:
                    mod_acc = (acc - balance) / (1.0 - balance)
                
                metrics['accuracy'].append(acc)
                metrics['precision'].append(p)
                metrics['recall'].append(r)
                metrics['f1'].append(f1)
                metrics['mod_accuracy'].append(mod_acc)
            
            # Aggregate results for this target
            if metrics['f1']:
                avg_acc = np.mean(metrics['accuracy'])
                avg_p = np.mean(metrics['precision'])
                avg_r = np.mean(metrics['recall'])
                avg_f1 = np.mean(metrics['f1'])
                avg_mod_acc = np.mean(metrics['mod_accuracy'])
                
                all_target_metrics[target] = {
                    'accuracy': avg_acc,
                    'precision': avg_p,
                    'recall': avg_r,
                    'f1': avg_f1,
                    'mod_accuracy': avg_mod_acc,
                    'folds': fold_count
                }
                
                print(f"    Acc: {avg_acc:.6f} | F1: {avg_f1:.6f} | ModAcc: {avg_mod_acc:.6f} | Prec: {avg_p:.6f} | Rec: {avg_r:.6f}")
            else:
                print(f"    No successful folds for {target}")
        
        # 6. Generate Combined Report
        if all_target_metrics:
            # Calculate averages across all targets
            combined_metrics = {
                'accuracy': np.mean([m['accuracy'] for m in all_target_metrics.values()]),
                'precision': np.mean([m['precision'] for m in all_target_metrics.values()]),
                'recall': np.mean([m['recall'] for m in all_target_metrics.values()]),
                'f1': np.mean([m['f1'] for m in all_target_metrics.values()]),
                'mod_accuracy': np.mean([m['mod_accuracy'] for m in all_target_metrics.values()]),
            }
            
            # Print combined summary
            print(f"\n  {'='*70}")
            print(f"  COMBINED RESULTS (Average across all targets)")
            print(f"  {'='*70}")
            print(f"    Accuracy:      {combined_metrics['accuracy']:.6f}")
            print(f"    F1 Score:      {combined_metrics['f1']:.6f}")
            print(f"    Mod Accuracy:  {combined_metrics['mod_accuracy']:.6f}")
            print(f"    Precision:     {combined_metrics['precision']:.6f}")
            print(f"    Recall:        {combined_metrics['recall']:.6f}")
            
            # Print per-target summary
            print(f"\n  Per-Target Results:")
            for target in TARGETS:
                if target in all_target_metrics:
                    m = all_target_metrics[target]
                    print(f"    {target}: Acc={m['accuracy']:.6f} F1={m['f1']:.6f} ModAcc={m['mod_accuracy']:.6f}")
            
            # 7. Logging - Combined report
            log_entry = (
                f"PARAM: {PARAM_TO_TEST}={val} | "
                f"Targets: {','.join(TARGETS)} | "
                f"Accuracy: {combined_metrics['accuracy']:.6f} | "
                f"F1: {combined_metrics['f1']:.6f} | "
                f"ModAcc: {combined_metrics['mod_accuracy']:.6f} | "
                f"Precision: {combined_metrics['precision']:.6f} | "
                f"Recall: {combined_metrics['recall']:.6f} | "
                f"FullParams: {current_params}"
            )
            
            with open(LOG_FILE, 'a') as f:
                f.write(log_entry + "\n")
            
            # Log per-target details
            for target in TARGETS:
                if target in all_target_metrics:
                    m = all_target_metrics[target]
                    detail_entry = (
                        f"  └─ {target}: Accuracy={m['accuracy']:.6f} F1={m['f1']:.6f} "
                        f"ModAcc={m['mod_accuracy']:.6f} Prec={m['precision']:.6f} Rec={m['recall']:.6f}"
                    )
                    with open(LOG_FILE, 'a') as f:
                        f.write(detail_entry + "\n")
        else:
            print(f"\nNo successful results for parameter value {val}")

    with open(LOG_FILE, 'a') as f:
        f.write("\n")
            
    print(f"\n{'='*80}")
    print(f"Completed testing for {PARAM_TO_TEST}.")
    print(f"Combined results logged to: {LOG_FILE}")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_selection_cycle()