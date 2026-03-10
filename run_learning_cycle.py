"""
run_learning_cycle.py
=====================

Read-Only Training Orchestration with Synthetic Data Augmentation

This script orchestrates the machine learning training workflow for solar irradiance
quality control, from data loading through model training, with optional
synthetic data augmentation using error injection.

NOTE: This script is READ-ONLY and does not modify any existing data files.

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

Key Features
------------
- **Read-Only Operation**: Does NOT modify any existing data files or flags
- **Synthetic Data Augmentation**: Configurable ratio (default 2:1 real:synthetic)
  - Samples whole months from training data (e.g., Feb, Apr, Jun from Jan-Jun)
  - Injects realistic errors to create more bad data examples
  - Increases model robustness to various failure modes
- **RNN Time-Series Models**: Uses fixed-length sequences for temporal awareness
- **Model Persistence**: Saves trained models to models/ folder for later use

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

TRAIN Dates:
    - Define training period
    - Recommendation: 3-4 months minimum
    - Augmentation samples from training period only

Usage
-----
1. Update SITE_CONFIG in config.py with your location details
2. Adjust SYNTHETIC_DATA_RATIO for desired augmentation level
3. Adjust date range for training period in main block
4. Verify DATA_FOLDER contains CSV files with proper structure
5. Run: python run_learning_cycle.py
6. Check models/ folder for saved models

Output Files
------------
- models/model_Flag_GHI.pkl: Trained GHI quality model
- models/model_Flag_DNI.pkl: Trained DNI quality model
- models/model_Flag_DHI.pkl: Trained DHI quality model

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
- Synthetic augmentation improves model performance on rare failure modes
- Augmentation ratio can be overridden per run_cycle() call
- Use predict_with_saved_model.py to run predictions with trained models
"""

import os
import glob
import sys
import pandas as pd
import numpy as np
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

from solar_features import add_features
from solar_model import SolarHybridModel
from error_injection import ErrorInjectionPipeline
from config import SITE_CONFIG, TARGET_CLASS_WEIGHT_MULTIPLIERS, TARGET_FIT_PARAMS
from io_utils import load_qc_csvs


# ---------------- LOGGING UTILITIES ----------------
class TeeLogger:
    """Redirect print statements to both console and log file."""
    def __init__(self, log_file_path):
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        
    def write(self, message):
        self.stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write
        
    def flush(self):
        self.stdout.flush()
        self.log_file.flush()
        
    def close(self):
        sys.stdout = self.stdout
        self.log_file.close()


# ---------------- HELPER FUNCTIONS ----------------
def print_with_timestamp(*args, **kwargs):
    """Print with timestamp prefix."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}]", *args, **kwargs)


# ---------------- CONFIG ----------------
DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'  # Folder to save trained models
LOG_FOLDER = 'log_files'  # Folder to save training logs
HEADER_ROWS_SKIP = 43  # Number of rows to skip when reading (skips rows 0-42, uses row 43 as column names)
HEADER_ROWS_PRESERVE = 44  # Number of rows to preserve when writing back (rows 0-43 including column names)
TS_COL = 'YYYY-MM-DD--HH:MM:SS'

SEQ_WINDOW_MINUTES = 60 # Target window in minutes for RNN sequence length (1 hour)

# Synthetic data augmentation ratio
# Ratio of real training data to synthetic error-injected data
# Default 2:1 means if training on 6 months, sample 3 months for error injection
# Total training data becomes 9 months (6 real + 3 synthetic)
SYNTHETIC_DATA_RATIO = 2.0  # real:synthetic (e.g., 2.0 = 2:1, 3.0 = 3:1, etc.)
MIN_ROWS_PER_SYNTH_MONTH = 1000  # Guardrail to skip clearly incomplete/corrupt months
SYNTHETIC_SAMPLING_SEED = 42  # Deterministic month sampling across runs


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

def run_cycle(train_start: str, train_end: str, targets: list[str],
              synthetic_ratio: float = 2.0, enable_logging: bool = True):
    """
    Run read-only training cycle with optional data augmentation.
    
    NOTE: This function does NOT modify any existing data files.
    
    Parameters
    ----------
    train_start : str
        Training period start date (YYYY-MM-DD HH:MM:SS)
    train_end : str
        Training period end date (YYYY-MM-DD HH:MM:SS)
    targets : list[str]
        List of target columns (e.g., ['Flag_GHI', 'Flag_DNI', 'Flag_DHI'])
    synthetic_ratio : float, optional
        Ratio of real to synthetic data (e.g., 2.0 = 2:1 real:synthetic).
        If None, uses SYNTHETIC_DATA_RATIO from config.
        If 0 or None, no synthetic data augmentation is performed.
    enable_logging : bool, default=True
        If True, writes all output to a timestamped log file in log_files/
    """
    # Setup logging
    logger = None
    if enable_logging:
        Path(LOG_FOLDER).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(LOG_FOLDER) / f'run_learning_cycle_{timestamp}.txt'
        logger = TeeLogger(log_file)
        sys.stdout = logger
        print_with_timestamp(f"Log file created: {log_file}")
        print_with_timestamp(f"{'='*70}")
        print_with_timestamp(f"RUN LEARNING CYCLE - READ-ONLY TRAINING LOG")
        print_with_timestamp(f"{'='*70}")
        print_with_timestamp(f"Training period: {train_start} to {train_end}")
        print_with_timestamp(f"Targets: {', '.join(targets)}")
        print_with_timestamp(f"Synthetic ratio: {synthetic_ratio if synthetic_ratio else 'disabled'}")
        print_with_timestamp(f"Mode: READ-ONLY (no data files will be modified)")
        print_with_timestamp(f"{'='*70}\n")
    
    # Get all CSV files
    all_files = sorted(glob.glob(os.path.join(DATA_FOLDER, '**', '*.csv'), recursive=True))
    
    print_with_timestamp(f"\n=== Loading Data ===")
    print_with_timestamp(f"Total files found: {len(all_files)}")
    
    # Load training data
    print_with_timestamp("\nLoading data for training...")
    raw = load_qc_csvs(all_files, header_rows_skip=HEADER_ROWS_SKIP, ts_col=TS_COL)
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
        random.seed(SYNTHETIC_SAMPLING_SEED)
        
        # Calculate how much synthetic data to generate
        train_duration_days = (pd.to_datetime(train_end) - pd.to_datetime(train_start)).days
        synthetic_duration_days = int(train_duration_days / synthetic_ratio)
        
        print_with_timestamp(f"Training period: {train_duration_days} days")
        print_with_timestamp(f"Synthetic data target: {synthetic_duration_days} days")
        
        # Sample continuous months from training data
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        
        # Build month candidates and drop clearly incomplete months.
        # This prevents pathological samples (e.g., months with 1 row) from skewing augmentation.
        available_months = []
        skipped_months = []
        current = train_start_dt.replace(day=1)
        while current <= train_end_dt:
            month_end = current + relativedelta(months=1) - pd.Timedelta(seconds=1)
            month_mask = (train_df.index >= current) & (train_df.index <= month_end)
            month_rows = int(month_mask.sum())
            if month_rows >= MIN_ROWS_PER_SYNTH_MONTH:
                available_months.append(current)
            elif month_rows > 0:
                skipped_months.append((current, month_rows))
            current = current + relativedelta(months=1)

        if skipped_months:
            print_with_timestamp(
                f"Skipping {len(skipped_months)} underpopulated month(s) "
                f"(<{MIN_ROWS_PER_SYNTH_MONTH} rows) from synthetic sampling."
            )
            for month_start, month_rows in skipped_months[:5]:
                print_with_timestamp(
                    f"  Skipped: {month_start.strftime('%Y-%m')} ({month_rows} rows)"
                )
        
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
        multipliers_cfg = TARGET_CLASS_WEIGHT_MULTIPLIERS.get(t, {'bad': 1.0, 'good': 1.0})
        class_weight_multipliers = {
            0: float(multipliers_cfg.get('bad', 1.0)),
            1: float(multipliers_cfg.get('good', 1.0)),
        }
        print_with_timestamp(
            f"Applying class weight multipliers for {t}: "
            f"BAD x{class_weight_multipliers[0]:.3f}, GOOD x{class_weight_multipliers[1]:.3f}"
        )
        fit_params = TARGET_FIT_PARAMS.get(t, {})
        if fit_params:
            print_with_timestamp(f"Applying fit params for {t}: {fit_params}")

        synthetic_frac_value = float(fit_params.get('synthetic_frac', 0.0))
        if synthetic_ratio and synthetic_ratio > 0 and synthetic_frac_value > 0:
            print_with_timestamp(
                f"Disabling in-model synthetic_frac for {t} because dataset-level "
                f"synthetic_ratio={synthetic_ratio} is already active."
            )
            synthetic_frac_value = 0.0

        model.fit(
            train_df,
            target_col=t,
            class_weight_multipliers=class_weight_multipliers,
            epochs=int(fit_params.get('epochs', 10)),
            batch_size=int(fit_params.get('batch_size', 64)),
            upsample_min_bad=int(fit_params.get('upsample_min_bad', 500)),
            synthetic_frac=synthetic_frac_value,
            max_bad_fraction=float(fit_params.get('max_bad_fraction', 0.10)),
            max_bad_weight=float(fit_params.get('max_bad_weight', 3.0)),
        )

        # Save the trained model
        model_filename = os.path.join(MODEL_FOLDER, f'model_{t}.pkl')
        model.save_model(model_filename)
        print_with_timestamp(f"Model saved to {model_filename}")
    
    print_with_timestamp(f"\n{'='*70}")
    print_with_timestamp("Training cycle complete!")
    print_with_timestamp("NOTE: No data files were modified (read-only mode)")
    print_with_timestamp(f"To run predictions, use: predict_with_saved_model.py")
    print_with_timestamp(f"{'='*70}")
        
    # Close logger and restore stdout
    if logger is not None:
        logger.close()
        print(f"\nLog saved to: {log_file}")


# ----------------- Entry point -------------------------------------------
if __name__ == '__main__':
    TARGETS = ['Flag_GHI', 'Flag_DNI', 'Flag_DHI']

    TRAIN_START = '2023-01-01 00:00:00'
    TRAIN_END = '2025-12-31 23:59:59'

    # Run with default behavior (uses synthetic augmentation at 2:1 ratio)
    run_cycle(TRAIN_START, TRAIN_END, TARGETS)
    
    # To disable synthetic data augmentation:
    # run_cycle(TRAIN_START, TRAIN_END, TARGETS, synthetic_ratio=0)
    
    # To use a different augmentation ratio (e.g., 3:1 real:synthetic):
    # run_cycle(TRAIN_START, TRAIN_END, TARGETS, synthetic_ratio=3.0)