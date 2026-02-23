SOLAR QC PIPELINE - QUICK REFERENCE GUIDE
==========================================

Quick reference for all scripts in the Solar QC system, organized by pipeline importance.

═══════════════════════════════════════════════════════════════════════════════
1. CORE TRAINING & PREDICTION (Primary Pipeline)
═══════════════════════════════════════════════════════════════════════════════

## run_learning_cycle.py - Main Orchestrator

**Purpose**: End-to-end training and prediction workflow with optional synthetic augmentation

**What it does**:
1. Loads CSVs from data/
2. Engineers features (temporal, solar geometry, anomalies)
3. [Optional] Augments training data with synthetic errors
4. Trains hybrid RNN models (one per target: GHI, DNI, DHI)
5. Saves models to models/
6. Runs predictions on specified date range

**Configuration**: NO command-line arguments - edit directly in file before running

**Basic usage**:
```bash
# 1. Edit dates and config in the file:
#    - TRAIN_START/TRAIN_END (training period)
#    - PRED_START/PRED_END (prediction period)
#    - TARGETS (which flags to predict)
#    - SYNTHETIC_DATA_RATIO (augmentation level)
# 2. Run:
python run_learning_cycle.py
```

**Key config knobs** (edit in `__main__` section):
- `TRAIN_START/END`: Training date range (e.g., '2023-01-01 00:00:00' to '2025-06-30 23:59:59')
- `PRED_START/END`: Prediction date range (e.g., '2025-07-01 00:00:00' to '2025-07-31 23:59:59')
- `TARGETS`: Flags to predict (default: `['Flag_GHI', 'Flag_DNI', 'Flag_DHI']`)

**Key config in file body**:
- `SYNTHETIC_DATA_RATIO`: real:synthetic ratio (2.0 = 2:1, 0 = disabled)
- `SITE_CONFIG`: Imported from config.py (see config.py section)

**Synthetic augmentation**: Randomly samples whole months from training data, injects 
realistic errors, combines with real data at specified ratio for more robust models.

─────────────────────────────────────────────────────────────────────────────

## predict_with_saved_model.py - Prediction Only

**Purpose**: Run predictions using saved models without retraining

**IMPORTANT**: By default PRESERVES existing bad flags (99), only updates good flags.
Use `--overwrite-all-flags` to replace all flags (destroys manual QC work!).

**Usage**:
```bash
# Single file (preserve manual flags):
python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv

# Date range (preserve manual flags):
python predict_with_saved_model.py --start "2025-07-01" --end "2025-08-31"

# Overwrite ALL flags (CAUTION - deletes manual QC):
python predict_with_saved_model.py --start "2025-01-01" --end "2025-12-31" --overwrite-all-flags

# Specific targets only:
python predict_with_saved_model.py --start "2025-01-01" --end "2025-12-31" --targets Flag_GHI Flag_DNI
```

**Command-line args**:
- `--file PATH`: Predict single file
- `--start DATE --end DATE`: Predict date range (YYYY-MM-DD)
- `--targets FLAG1 FLAG2`: Specify which flags (default: all three)
- `--overwrite-all-flags`: Replace ALL flags including manual QC (use with caution!)

─────────────────────────────────────────────────────────────────────────────

## prediction_comparison.py - Compare Predictions vs Current Flags

**Purpose**: Load random data files, run predictions without modifying them,
and compare prediction accuracy against current flags

**What it does**:
1. Selects random CSV files from data/ (configurable count)
2. Loads all selected files and engineers features
3. Runs predictions using trained models
4. Compares predictions against current flags (Flag_GHI, Flag_DNI, Flag_DHI)
5. Outputs metrics (accuracy, precision, recall, F1) to terminal and log file

**IMPORTANT**: Does NOT modify any data files - all predictions are made in memory

**Usage**:
```bash
# Compare on 5 random files (default):
python prediction_comparison.py

# Compare on specific number of random files:
python prediction_comparison.py 10

# Compare on ALL files:
python prediction_comparison.py -1

# Compare on specific file(s):
python prediction_comparison.py --file data/STW_2025/STW_2025-07_QC.csv
python prediction_comparison.py --file data/STW_2025/STW_2025-07_QC.csv data/STW_2025/STW_2025-08_QC.csv
```

**Command-line args**:
- `NUM_FILES`: Number of random files to select (default: 5, use -1 for all)
- `--file PATH [PATH ...]`: Specify specific file(s) to compare (takes precedence over random selection)

**Output**:
- **Terminal**: Formatted table with metrics for each feature
- **Log file**: Saved to `log_files/prediction_comparison_YYYYMMDD_HHMMSS.txt`

**Metrics explained**:
- **Accuracy**: % of predictions matching current flags
- **Precision**: Of predicted BAD flags, what % were actually BAD
- **Recall**: Of actual BAD flags, what % did we predict as BAD
- **F1**: Harmonic mean of precision and recall (0.0-1.0, higher is better)

═══════════════════════════════════════════════════════════════════════════════
2. MANUAL REVIEW
═══════════════════════════════════════════════════════════════════════════════

## SRML_ManualQC.py - Interactive QC GUI

**Purpose**: Manual review and correction of automated QC flags

**Features**:
- Time-series visualization (24-hour windows)
- Click-drag box selection for bulk editing
- Multiple x-axis modes (Time/Azimuth/Zenith)
- Day/month navigation
- Undo functionality
- Auto-saves on navigation and edits

**Usage**:
```bash
# Launch GUI (select file in GUI):
python SRML_ManualQC.py

# Or launch with specific file:
python SRML_ManualQC.py path/to/file.csv
```

**Controls**:
- **Load File**: Select CSV to review
- **Previous/Next Day**: Navigate chronologically
- **Previous/Next Month**: Jump by month
- **Select Date**: Direct date selection
- **Mark GOOD/BAD**: Bulk edit selected points
- **Undo**: Revert last edit
- **X-axis**: Switch between Time/AZM/ZEN views

**Color coding**:
- Red = BAD (99)
- Green = GOOD (1)
- Yellow = PROBABLE (intermediate confidence)

**Best practices**:
- Review model predictions systematically (month by month)
- Use box selection for efficiency
- Auto-saves preserve all changes
- Original data backed up in manual_confirmed_data_backup/

═══════════════════════════════════════════════════════════════════════════════
3. SYNTHETIC ERROR INJECTION (Testing & Augmentation)
═══════════════════════════════════════════════════════════════════════════════

## error_injection.py - Realistic Error Injection

**Purpose**: Create realistic synthetic errors in solar radiation data for robustness testing and training augmentation

**Key Features**:
- Six configurable error types (reduce_features, copy_from_day, end_of_day_frost, cleaning_event, water_droplet, broken_tracker)
- SZA-based daytime filtering (solar zenith angle >= 85° excluded)
- Weighted error selection (each error type has configurable probability)
- Automatic bad flag assignment (Flag = 99)
- Optional manifest generation (tracks which samples were modified and how)
- Modular architecture: DataManager, SolarGeometry, ErrorFunctions, ErrorInjectionEngine, OutputHandler

**Error Types**:
1. **reduce_features**: Reduce all irradiance values by 1-50%
2. **copy_from_day**: Copy values from a different day (-30 to -1 days offset)
3. **end_of_day_frost**: Dip 10-40% at day end (frost/dew formation)
4. **cleaning_event**: Brief 4-8% dip lasting 3 minutes (panel cleaning)
5. **water_droplet**: Spike 5-25% lasting 1-5 minutes (water droplet runoff)
6. **broken_tracker**: Sustained low values for 30-240 minutes (tracker failure)

**Configuration** (ERROR_INJECTION_CONFIG in error_injection.py):
- `error_probability`: Likelihood (0.0-1.0) that a file gets errors
- `error_count_range`: (min, max) errors to inject per file
- `daytime_bias`: Bias toward daytime errors (0.0-1.0)
- `sza_threshold`: Solar zenith angle threshold for daytime (85° default)
- Each error type has: `weight` (selection probability), and type-specific parameters

**Usage - Python API**:
```python
from error_injection import ErrorInjectionPipeline

# Default configuration (injected errors automatically flagged as 99)
pipeline = ErrorInjectionPipeline()

# Save errors to file with manifest
pipeline.process_file('data/STW_2023/STW_2023-01_QC.csv', output_mode='save')
# Output: data/injected_error/STW_2023-01_errored_QC.csv + manifest.json

# Return dataframe (no file writeback)
df_synthetic = pipeline.process_file('data/STW_2023/STW_2023-01_QC.csv', output_mode='return')

# Batch processing multiple files
files = [f'data/STW_2023/STW_2023-{m:02d}_QC.csv' for m in range(1, 13)]
pipeline.process_multiple_files(files, output_mode='save')
```

**Custom Configuration**:
```python
from error_injection import ErrorInjectionPipeline, ERROR_INJECTION_CONFIG
import copy

config = copy.deepcopy(ERROR_INJECTION_CONFIG)
config['error_probability'] = 0.9           # 90% of files get errors
config['error_count_range'] = (5, 15)       # 5-15 errors per file
config['daytime_bias'] = 0.95               # Mostly daytime
config['sza_threshold'] = 80.0              # More aggressive daytime filter

# Adjust individual error type weights
config['error_functions']['reduce_features']['weight'] = 0.5  # More common
config['error_functions']['broken_tracker']['weight'] = 0.05  # Less common

pipeline = ErrorInjectionPipeline(config=config)
pipeline.process_file('data/file.csv', output_mode='save')
```

**Command-line Usage**:
```bash
# Single file
python error_injection.py data/STW_2023/STW_2023-01_QC.csv --mode save

# Multiple files (wildcard)
python error_injection.py "data/STW_2023/*.csv" --mode save

# Verbose output
python error_injection.py data/STW_2023/STW_2023-01_QC.csv --mode save --verbose
```

**Command-line Options**:
- `filepath`: Path to file or wildcard pattern (supports glob)
- `--mode save | return`: Save to disk with manifest OR return DataFrame
- `--verbose`: Detailed output about injected errors

**Output Files** (when mode='save'):
- **Data file**: `data/injected_error/{original_name}_errored_QC.csv`
  - Same format as input but with synthetic errors
  - Flag columns set to 99 for modified samples
- **Manifest**: `data/injected_error/{original_name}_errored_manifest.json`
  - JSON tracking: which rows modified, error type, parameters, original values
  - Useful for validation and error analysis

**Training Integration**:
Use synthetically generated data to augment training dataset:
```python
import pandas as pd
from error_injection import ErrorInjectionPipeline

# Generate synthetic errors
pipeline = ErrorInjectionPipeline()
df_synthetic = pipeline.process_file('data/STW_2023/STW_2023-01_QC.csv', output_mode='return')

# Combine with real training data
df_real = pd.read_csv('data/STW_2023/STW_2023-02_QC.csv', skiprows=43)
df_augmented = pd.concat([df_real, df_synthetic], ignore_index=True)

# Train model on augmented data
model.fit(df_augmented, target_col='Flag_GHI')
```

**Reproducibility**:
For reproducible synthetic errors, set random seeds:
```python
import random, numpy as np
random.seed(42)
np.random.seed(42)
df = pipeline.process_file('data/file.csv', output_mode='return')
# Same seed produces identical errors
```

═══════════════════════════════════════════════════════════════════════════════
4. CORE MODULES (Imported by main scripts)
═══════════════════════════════════════════════════════════════════════════════

## config.py - Site Configuration

**Purpose**: Central configuration for site-specific parameters (imported by all scripts)

**Contents**:
```python
SITE_CONFIG = {
    'latitude': 47.654,         # Decimal degrees (edit for your site)
    'longitude': -122.309,      # Decimal degrees (edit for your site)
    'altitude': 70,             # Meters above sea level
    'timezone': 'Etc/GMT+8'     # For solar calculations ONLY (no conversion!)
}
```

**CRITICAL**: Timezone is metadata for pvlib solar calculations, NOT for timestamp conversion.
Timestamps are assumed already correct in local time.

**Usage**: Edit SITE_CONFIG values to match your data collection site, then all scripts 
that import from config.py will use these values.

─────────────────────────────────────────────────────────────────────────────

## solar_features.py - Feature Engineering

**Purpose**: Convert raw measurements into ML-ready features

**Key functions**:
```python
from solar_features import add_features
from config import SITE_CONFIG

df_featured = add_features(df, site_config=SITE_CONFIG)
# Adds 40+ features: solar geometry, clear-sky, ratios, temporal, etc.
```

**Generated features**:
- **Solar geometry**: SZA, azimuth, elevation, hour angle, air mass
- **Clear-sky modeling**: GHI_Clear, DHI_Clear, DNI_Clear (Ineichen model via pvlib)
- **Clear-sky index**: CSI = GHI_measured / GHI_Clear
- **Ratios**: ghi_ratio, ghi_diff (measured vs clear-sky)
- **Temporal**: hour, day-of-year, seasonal cycles (sin/cos encodings)
- **Anomaly features**: Correlation-based features (CorrFeat_GHI, CorrFeat_DNI, CorrFeat_DHI)
- **Deterministic QC**: QC_PhysicalFail (BSRN bounds, closure test)

─────────────────────────────────────────────────────────────────────────────

## solar_model.py - Hybrid QC Model

**Purpose**: Define three-component hybrid model architecture

**Architecture**:
1. **RandomForest**: Supervised baseline on engineered features
2. **IsolationForest**: Unsupervised anomaly detection (trained on GOOD samples only)
3. **RNN (GRU)**: Time-series aware with attention mechanism

**Key features**:
- Temporal sequences (default: 60 samples for 1-minute data)
- Attention mechanism for critical time steps
- Class-weighted training for imbalanced data
- Isotonic calibration for probability reliability
- Dual output: Binary flags (99/1) + probabilities (0.0-1.0)

**Usage** (typically through run_learning_cycle.py):
```python
from solar_model import SolarHybridModel

model = SolarHybridModel(
    use_rnn=True,               # True = RNN, False = Dense
    seq_length=60               # Sequence length for RNN (default: 60)
)

model.fit(X_train, target_col='Flag_GHI')
predictions, probabilities = model.predict(X_test, 'Flag_GHI', do_return_probs=True)
model.save_model('models/model_Flag_GHI.pkl')
```

**Note**: Other hyperparameters (RF trees, RNN layers, dropout) are configured internally 
in the model class, not via __init__ parameters.

─────────────────────────────────────────────────────────────────────────────

## get_solar_data.py - Data Loading

**Purpose**: Load comprehensive-format CSVs and apply automated QC tests

**Format**:
- Rows 0-42: Metadata headers (station info, sensors, etc.)
- Row 43: Column names
- Row 44+: Hourly data

**Key function**:
```python
from get_solar_data import subroutine_main_automated_qc

# Load, apply automated QC, save with flags
subroutine_main_automated_qc('path/to/comprehensive_file.csv')
```

**Automated QC tests**:
- Reasonableness checks (physical bounds)
- Component comparison tests (GHI vs DNI+DHI)
- Missing data detection

─────────────────────────────────────────────────────────────────────────────

## io_utils.py - I/O Utilities

**Purpose**: Helper functions for CSV loading with proper header handling

**Key functions**:
```python
from io_utils import load_qc_csvs

# Load multiple CSVs (preserves 44-row headers)
df = load_qc_csvs(['data/STW_2023/STW_2023-01_QC.csv', ...])
```

═══════════════════════════════════════════════════════════════════════════════
5. ANALYSIS & OPTIMIZATION
═══════════════════════════════════════════════════════════════════════════════

## hyperparameter_selection/selection.py - Hyperparameter Tuning

**Purpose**: Grid search for optimal model hyperparameters with temporal validation

**Features**:
- Block-based train/test split (preserves temporal structure for RNN)
- Grid search over model architecture parameters
- Logs results to feature_performance.log

**Configuration**: NO command-line arguments - edit directly in file before running

**Usage**:
```bash
# 1. Edit config in file:
#    - TEST_MONTHS: Which months to hold out for testing
#    - TARGETS: Which flags to optimize
#    - PARAM_TO_TEST: Which parameter to vary
#    - TEST_VALUES: Values to test for that parameter
# 2. Run:
cd hyperparameter_selection
python selection.py
```

**Key config** (edit in file):
- `TEST_MONTHS`: Which months to use as test set (e.g., [1, 2, 5, 6])
- `TARGETS`: Which flags to optimize (e.g., ['Flag_GHI', 'Flag_DNI'])
- `PARAM_TO_TEST`: Parameter name to vary (e.g., 'rf_n_estimators')
- `TEST_VALUES`: List of values to test

**Output**: Results logged to feature_performance.log with metrics for each parameter value

─────────────────────────────────────────────────────────────────────────────

## performance/compute_feature_importance.py - Feature Importance Analysis

**Purpose**: Analyze which features are most important for model predictions

**What it does**:
1. Loads trained models from models/
2. Loads recent data (2025) for importance computation
3. Computes RF feature importance (direct from model)
4. Computes RNN correlation-based importance
5. Writes report to log_files/feature_importance.log

**Configuration**: NO command-line arguments - just run it

**Usage**:
```bash
cd performance
python compute_feature_importance.py
```

**Requirements**: Trained models must exist in models/ folder

**Output**: log_files/feature_importance.log with ranked feature importance for both 
RF and RNN components, normalized to [0, 1]

═══════════════════════════════════════════════════════════════════════════════
6. TESTING & VALIDATION
═══════════════════════════════════════════════════════════════════════════════

## tests/smoke_tests.py - Minimal Validation

**Purpose**: Lightweight smoke tests for core pipeline

**What it tests**:
- Finds a sample CSV file with sufficient data
- Data loading and feature engineering
- Model training on small sample (Dense NN, 1 epoch)
- Prediction on test set
- Basic sanity checks

**Configuration**: NO command-line arguments - just run it

**Usage**:
```bash
cd tests
python smoke_tests.py
```

**Requirements**: At least one CSV file in data/ with labeled flags

**Example output**:
```
Using sample file: data/STW_2023/STW_2023-01_QC.csv
Predictions: 100 rows
P(GOOD) mean: 0.8542
```

═══════════════════════════════════════════════════════════════════════════════
COMMON WORKFLOWS
═══════════════════════════════════════════════════════════════════════════════

## A. Initial Setup & Training

1. Configure site parameters in config.py
2. Place data files in data/STW_YYYY/ folders (comprehensive format)
3. Edit training dates in run_learning_cycle.py
4. Run: `python run_learning_cycle.py`
5. Models saved to models/ folder

## B. Predict New Data

1. Place new CSV files in data/STW_YYYY/ folders
2. Run: `python predict_with_saved_model.py --start YYYY-MM-DD --end YYYY-MM-DD`
3. Predictions written back to CSV files (preserves manual flags)

## C. Manual Review Cycle

1. Run predictions on data range
2. Launch GUI: `python SRML_ManualQC.py`
3. Load file, review flags, make corrections
4. Auto-saves preserve changes
5. Optionally retrain with corrected data

## D. Testing with Synthetic Errors

1. Generate synthetic errors: `python error_injection.py "data/STW_2023/*.csv" --mode save`
2. Output in data/injected_error/ with manifests
3. Run predictions on synthetic data
4. Compare predictions to known injected errors
5. Evaluate model robustness

## E. Hyperparameter Optimization

1. Ensure manual_confirmed_data_backup/ has ground-truth data
2. Edit TEST_MONTHS, TARGETS, PARAM_TO_TEST, TEST_VALUES in hyperparameter_selection/selection.py
3. Run: `cd hyperparameter_selection && python selection.py`
4. Review feature_performance.log for results
5. Update model architecture in solar_model.py with optimal parameters if needed

═══════════════════════════════════════════════════════════════════════════════
QUICK TIPS
═══════════════════════════════════════════════════════════════════════════════

**Data format**: All CSVs use comprehensive format (43 metadata rows + 1 header)

**Timestamps**: YYYY-MM-DD--HH:MM:SS (local time, NO conversion applied)

**Flags**: 99 = BAD quality, 1 = GOOD quality

**Probabilities**: P(GOOD) from 0.0 (confident BAD) to 1.0 (confident GOOD)

**File naming**: STW_YYYY-MM_QC.csv (e.g., STW_2025-07_QC.csv)

**Model files**: models/model_Flag_{GHI|DNI|DHI}.pkl

**Backup strategy**: manual_confirmed_data_backup/ preserves ground-truth data

**Pipeline safety**: predict_with_saved_model.py preserves manual flags by default

═══════════════════════════════════════════════════════════════════════════════
