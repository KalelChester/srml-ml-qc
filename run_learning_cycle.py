"""
run_learning_cycle.py
=====================

Training and Prediction Orchestration with Synthetic Data Augmentation

This script orchestrates the complete machine learning workflow for solar irradiance
quality control, from data loading through model training to prediction, with optional
synthetic data augmentation using error injection.

Workflow Overview
-----------------
1. Load raw CSV files from data directory
2. Engineer features (temporal, solar geometry, anomalies)
3. Extract training data from specified date range
4. [OPTIONAL] Augment training data with synthetic error-injected samples:
   - Randomly sample whole months from training period (not necessarily continuous)
   - Inject realistic errors using error_injection module
   - Combine synthetic data with original training data
5. Train hybrid RNN models for each target (GHI, DNI, DHI) on augmented data
6. Save trained models to models/ folder
7. Run predictions on prediction period using predict_with_saved_model

Key Features
------------
- **Synthetic Data Augmentation**: Configurable ratio (default 2:1 real:synthetic)
  - Samples whole months from training data (e.g., Feb, Apr, Jun from Jan-Jun)
  - Injects realistic errors to create more bad data examples
  - Increases model robustness to various failure modes
- **RNN Time-Series Models**: Uses fixed-length sequences for temporal awareness
- **Model Persistence**: Saves trained models for reuse without retraining
- **Automatic Prediction**: Calls predict_with_saved_model after training
- **Flag Preservation**: By default, preserves existing bad flags (99) during predictions
- **Header Preservation**: Maintains original CSV header rows (43 metadata + 1 column names)

Configuration
-------------
Key parameters to adjust in this file:

SITE_CONFIG:
    - latitude: Site latitude in decimal degrees
    - longitude: Site longitude in decimal degrees  
    - altitude: Site altitude in meters (default: 0)
    - timezone: Local timezone string (e.g., 'Etc/GMT+8', 'America/Los_Angeles')
                Used ONLY for solar position calculations, does NOT convert times

SYNTHETIC_DATA_RATIO:
    - Ratio of real to synthetic training data (default: 2.0 = 2:1)
    - E.g., 6 months real data + 3 months synthetic = 9 months total
    - Set to 0 to disable augmentation
    - Higher ratio = less synthetic data (3.0 = 3:1 means only 1/3 as much synthetic)

TRAIN/PRED Dates:
    - Define training period and prediction period
    - Recommendation: 3-4 months training, 1-2 months prediction
    - Augmentation samples from training period only

Usage
-----
1. Update SITE_CONFIG in config.py with your location details
2. Adjust SYNTHETIC_DATA_RATIO for desired augmentation level
3. Adjust date ranges for training/prediction periods in main block
4. Verify DATA_FOLDER contains CSV files with proper structure
5. Run: python run_learning_cycle.py
6. Check models/ folder for saved models
7. Check updated CSVs for predictions with Flag_* and Flag_*_prob columns

Output Files
------------
- models/model_Flag_GHI.pkl: Trained GHI quality model
- models/model_Flag_DNI.pkl: Trained DNI quality model
- models/model_Flag_DHI.pkl: Trained DHI quality model
- Updated source CSVs: With Flag_* and Flag_*_prob columns

Synthetic Data Augmentation Details
------------------------------------
- Samples whole months randomly (not necessarily continuous)
- E.g., from Jan-Jun training, might sample Feb, Apr, Jun
- Uses error_injection module to inject realistic errors
- Automatically flags injected errors as bad (99)
- Re-engineers features after error injection
- Preserves temporal structure within each month
- Increases diversity of failure modes in training set

File Structure Expected
-----------------------
Input CSV format (after HEADER_ROWS_SKIP):
    Column names row with: YYYY-MM-DD--HH:MM:SS, GHI, DNI, DHI, Flag_*, etc.
    Data rows follow immediately

First 44 rows:
    Rows 0-42: Metadata headers
    Row 43: Column names
    Row 44+: Data

Notes
-----
- Timestamps assumed to be in correct local time (no conversion applied)
- Missing columns handled gracefully with safe defaults
- Models train on RNN architecture by default
- Predictions preserve existing bad flags (99) by default
- Use preserve_bad_flags=False in run_cycle() to overwrite all flags
- Synthetic augmentation improves model performance on rare failure modes
- Augmentation ratio can be overridden per run_cycle() call
"""

import os
import glob
import pandas as pd
import numpy as np
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta

from solar_features import add_features
from solar_model import SolarHybridModel
from predict_with_saved_model import predict_with_model
from error_injection import ErrorInjectionPipeline
from config import SITE_CONFIG
from io_utils import load_qc_csvs


# ---------------- HELPER FUNCTIONS ----------------
def print_with_timestamp(*args, **kwargs):
    """Print with timestamp prefix."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}]", *args, **kwargs)


# ---------------- CONFIG ----------------
DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'  # Folder to save trained models
HEADER_ROWS_SKIP = 43  # Number of rows to skip when reading (skips rows 0-42, uses row 43 as column names)
HEADER_ROWS_PRESERVE = 44  # Number of rows to preserve when writing back (rows 0-43 including column names)
TS_COL = 'YYYY-MM-DD--HH:MM:SS'

SEQ_WINDOW_MINUTES = 60 # Target window in minutes for RNN sequence length (3 hours)

# Synthetic data augmentation ratio
# Ratio of real training data to synthetic error-injected data
# Default 2:1 means if training on 6 months, sample 3 months for error injection
# Total training data becomes 9 months (6 real + 3 synthetic)
SYNTHETIC_DATA_RATIO = 2.0  # real:synthetic (e.g., 2.0 = 2:1, 3.0 = 3:1, etc.)


def infer_seq_length(df: pd.DataFrame, window_minutes: int) -> int:
    """
    Infer RNN sequence length from median sampling interval.
    """
    # Try to get timestamps from index or column
    if isinstance(df.index, pd.DatetimeIndex):
        times = pd.Series(df.index).sort_values()
    elif 'Timestamp_dt' in df.columns:
        times = pd.to_datetime(df['Timestamp_dt'], errors='coerce').dropna().sort_values()
    else:
        return 60
    
    if len(times) < 2:
        return 60

    deltas = times.diff().dt.total_seconds() / 60.0
    delta_minutes = float(deltas[deltas > 0].median()) if len(deltas) > 0 else 0.0
    if not np.isfinite(delta_minutes) or delta_minutes <= 0:
        return 60

    return max(1, int(round(window_minutes / delta_minutes)))


# ---------------- Main cycle ----------------------------------------------

def run_cycle(train_start: str, train_end: str, pred_start: str, pred_end: str, targets: list[str],
              preserve_bad_flags: bool = True, synthetic_ratio: float = None, num_test_files: int = 2):
    """
    Run complete training and prediction cycle with optional data augmentation.
    
    Parameters
    ----------
    train_start : str
        Training period start date (YYYY-MM-DD HH:MM:SS)
    train_end : str
        Training period end date (YYYY-MM-DD HH:MM:SS)
    pred_start : str
        Prediction period start date (YYYY-MM-DD HH:MM:SS)
    pred_end : str
        Prediction period end date (YYYY-MM-DD HH:MM:SS)
    targets : list[str]
        List of target columns (e.g., ['Flag_GHI', 'Flag_DNI', 'Flag_DHI'])
    preserve_bad_flags : bool, default=True
        If True, preserves existing bad flags (99) during prediction writeback.
        If False, overwrites all flags with model predictions.
    synthetic_ratio : float, optional
        Ratio of real to synthetic data (e.g., 2.0 = 2:1 real:synthetic).
        If None, uses SYNTHETIC_DATA_RATIO from config.
        If 0 or None, no synthetic data augmentation is performed.
    num_test_files : int, default=2
        Number of monthly files to randomly select for testing.
        These files are excluded from training and used for prediction/validation.
    """
    # Get all CSV files
    all_files = sorted(glob.glob(os.path.join(DATA_FOLDER, '**', '*.csv'), recursive=True))
    
    # Randomly select test files
    if len(all_files) > num_test_files:
        random.seed(42)  # For reproducibility
        test_files = random.sample(all_files, num_test_files)
        train_files = [f for f in all_files if f not in test_files]
        print_with_timestamp(f"\n=== File Split (Random Selection) ===")
        print_with_timestamp(f"Total files: {len(all_files)}")
        print_with_timestamp(f"Training files: {len(train_files)}")
        print_with_timestamp(f"Test files: {len(test_files)}")
        print_with_timestamp(f"Test files selected:")
        for tf in sorted(test_files):
            print_with_timestamp(f"  - {os.path.basename(tf)}")
    else:
        print_with_timestamp(f"Warning: Only {len(all_files)} files available, need at least {num_test_files + 1} for train/test split")
        train_files = all_files
        test_files = []
    
    # Load training data
    print_with_timestamp("\nLoading data for training...")
    raw = load_qc_csvs(train_files, header_rows_skip=HEADER_ROWS_SKIP, ts_col=TS_COL)
    if raw.empty:
        print_with_timestamp('No training files found')
        return

    # Add features
    print_with_timestamp("Engineering features...")
    full = add_features(raw, SITE_CONFIG)
    full['Timestamp_dt'] = pd.to_datetime(full['Timestamp_dt'], errors='coerce')
    full = full.set_index('Timestamp_dt', drop=True)
    
    # Extract training data based on date range
    train_mask = (full.index >= pd.to_datetime(train_start)) & (full.index <= pd.to_datetime(train_end))
    train_df = full[train_mask].copy()

    if train_df.empty:
        raise RuntimeError('Training slice is empty')
    
    # Data augmentation with synthetic error injection
    if synthetic_ratio is None:
        synthetic_ratio = SYNTHETIC_DATA_RATIO
    
    if synthetic_ratio > 0:
        print_with_timestamp(f"\n=== Data Augmentation (Ratio {synthetic_ratio}:1 real:synthetic) ===")
        
        # Calculate how much synthetic data to generate
        train_duration_days = (pd.to_datetime(train_end) - pd.to_datetime(train_start)).days
        synthetic_duration_days = int(train_duration_days / synthetic_ratio)
        
        print_with_timestamp(f"Training period: {train_duration_days} days")
        print_with_timestamp(f"Synthetic data target: {synthetic_duration_days} days")
        
        # Sample continuous months from training data
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        
        # Get all available months in training period
        available_months = []
        current = train_start_dt.replace(day=1)
        while current <= train_end_dt:
            available_months.append(current)
            current = current + relativedelta(months=1)
        
        # Determine how many months we need for synthetic data
        days_per_month = 30  # Approximate
        months_needed = max(1, int(synthetic_duration_days / days_per_month))
        months_needed = min(months_needed, len(available_months))  # Can't sample more than available
        
        print_with_timestamp(f"Randomly sampling {months_needed} whole month(s) for error injection...")
        
        # Randomly select individual months (not necessarily continuous)
        if months_needed <= len(available_months):
            sampled_months = random.sample(available_months, months_needed)
            sampled_months.sort()  # Sort for cleaner output
            
            # Extract data for sampled months
            synthetic_data_list = []
            for month_start in sampled_months:
                month_end = month_start + relativedelta(months=1) - pd.Timedelta(seconds=1)
                month_mask = (train_df.index >= month_start) & (train_df.index <= month_end)
                month_data = train_df[month_mask].copy()
                
                if not month_data.empty:
                    print_with_timestamp(f"  Sampled: {month_start.strftime('%Y-%m')} ({len(month_data)} rows)")
                    synthetic_data_list.append(month_data)
            
            if synthetic_data_list:
                # Combine sampled months
                sampled_df = pd.concat(synthetic_data_list, ignore_index=False)
                print_with_timestamp(f"Total sampled data: {len(sampled_df)} rows")
                
                # Inject errors using error injection pipeline
                print_with_timestamp("Injecting synthetic errors...")
                pipeline = ErrorInjectionPipeline()
                
                # Process the sampled data - we need to work with the raw dataframe
                # Extract necessary columns and prepare for error injection
                synthetic_df = sampled_df.copy()
                
                # Run error injection engine directly on the dataframe
                synthetic_errored, error_metadata = pipeline.engine.inject_errors(synthetic_df)
                synthetic_flagged = pipeline.engine.flag_bad_data(synthetic_errored, error_metadata)
                
                # Re-engineer features for the synthetic data to ensure consistency
                print_with_timestamp("Re-engineering features for synthetic data...")
                # We need to restore the _source_file and _raw_ts columns if they were lost
                for col in ['_source_file', '_raw_ts']:
                    if col in sampled_df.columns and col not in synthetic_flagged.columns:
                        synthetic_flagged[col] = sampled_df[col].values
                
                # Re-add features (some may have been affected by error injection)
                synthetic_final = add_features(synthetic_flagged, SITE_CONFIG)
                synthetic_final['Timestamp_dt'] = pd.to_datetime(synthetic_final['Timestamp_dt'], errors='coerce')
                synthetic_final = synthetic_final.set_index('Timestamp_dt', drop=True)
                
                # Combine original training data with synthetic data
                print_with_timestamp(f"\nCombining training data:")
                print_with_timestamp(f"  Original: {len(train_df)} rows")
                print_with_timestamp(f"  Synthetic: {len(synthetic_final)} rows")
                train_df = pd.concat([train_df, synthetic_final], ignore_index=False)
                print_with_timestamp(f"  Combined: {len(train_df)} rows")
                
                # Count bad flags in combined data
                bad_flags_count = 0
                for t in targets:
                    if t in train_df.columns:
                        bad_flags_count += (train_df[t] == 99).sum()
                print_with_timestamp(f"  Total bad flags across all targets: {bad_flags_count}")
            else:
                print_with_timestamp("Warning: No data sampled for synthetic generation")
        else:
            print_with_timestamp(f"Warning: Not enough months available for synthetic data generation")
    else:
        print_with_timestamp("\nSkipping synthetic data augmentation (ratio = 0)")

    # Infer sequence length based on data resolution
    seq_length = infer_seq_length(train_df, SEQ_WINDOW_MINUTES)
    print_with_timestamp(f"\n[info] Inferred seq_length={seq_length} from data resolution")

    # Train models for each target
    for t in targets:
        print_with_timestamp(f"\n=== Training {t} ===")
        # Use RNN model by default for better time-series sensitivity
        model = SolarHybridModel(use_rnn=True, seq_length=seq_length)
        model.fit(train_df, target_col=t)

        # Save the trained model
        model_filename = os.path.join(MODEL_FOLDER, f'model_{t}.pkl')
        model.save_model(model_filename)
        print_with_timestamp(f"Model saved to {model_filename}")
    
    # Use the randomly selected test files for prediction
    if not test_files:
        print_with_timestamp("\nWarning: No test files available for prediction")
        return
    
    pred_files = test_files
    
    # Run predictions using predict_with_saved_model
    print_with_timestamp(f"\n=== Running predictions on {len(pred_files)} test file(s) ===")
    for pf in sorted(pred_files):
        print_with_timestamp(f"  - {os.path.basename(pf)}")
    predict_with_model(
        file_paths=pred_files,
        targets=targets,
        write_back=True,
        preserve_bad_flags=preserve_bad_flags
    )


# ----------------- Entry point -------------------------------------------
if __name__ == '__main__':
    TARGETS = ['Flag_GHI', 'Flag_DNI', 'Flag_DHI']

    TRAIN_START = '2023-01-01 00:00:00'
    TRAIN_END = '2025-06-30 23:59:59'
    PRED_START = '2025-07-01 00:00:00'
    PRED_END = '2025-07-31 23:59:59'

    # Run with default behavior (preserves existing bad flags, uses synthetic augmentation, 2 random test files)
    run_cycle(TRAIN_START, TRAIN_END, PRED_START, PRED_END, TARGETS)
    
    # To overwrite all flags with model predictions:
    # run_cycle(TRAIN_START, TRAIN_END, PRED_START, PRED_END, TARGETS, preserve_bad_flags=False)
    
    # To disable synthetic data augmentation:
    # run_cycle(TRAIN_START, TRAIN_END, PRED_START, PRED_END, TARGETS, synthetic_ratio=0)
    
    # To use a different augmentation ratio (e.g., 3:1 real:synthetic):
    # run_cycle(TRAIN_START, TRAIN_END, PRED_START, PRED_END, TARGETS, synthetic_ratio=3.0)
    
    # To use more test files (e.g., 4 files):
    # run_cycle(TRAIN_START, TRAIN_END, PRED_START, PRED_END, TARGETS, num_test_files=4)