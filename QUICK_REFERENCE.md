QUICK REFERENCE GUIDE - Error Injection and Augmentation
========================================================

One-page cheat sheet for the synthetic error injection workflow and training
augmentation options used by this project.

# 1. BASIC USAGE

```python
from error_injection import ErrorInjectionPipeline

# For testing (save to disk):
pipeline = ErrorInjectionPipeline()
pipeline.process_file('data/file.csv', output_mode='save')
# Output: data/injected_error/file_errored.csv + file_manifest.json

# For training (return dataframe):
df_synthetic = pipeline.process_file('data/file.csv', output_mode='return')
# Use directly: model.fit(df_synthetic, labels)
```

# 2. BATCH PROCESSING

```python
from error_injection import ErrorInjectionPipeline

pipeline = ErrorInjectionPipeline()
files = [f'data/STW_2023/STW_2023-{m:02d}_QC.csv' for m in range(1, 13)]
pipeline.process_multiple_files(files, output_mode='save')
# Output: All files in data/injected_error/
```

# 3. CUSTOM PARAMETERS

```python
from error_injection import ErrorInjectionPipeline, ERROR_INJECTION_CONFIG
import copy

config = copy.deepcopy(ERROR_INJECTION_CONFIG)

# Common customizations:
config['error_probability'] = 0.9      # More files get errors
config['error_count_range'] = (2, 8)   # More errors per file
config['daytime_bias'] = 0.95          # Mostly daytime errors
config['sza_threshold'] = 90.0         # SZA < 90 considered daytime

# Per-error parameters (see error_injection.py for defaults)
config['error_functions']['reduce_features']['reduction_percent'] = (10, 40)
config['error_functions']['copy_from_day']['days_offset_range'] = (-14, -1)

pipeline = ErrorInjectionPipeline(config=config)
pipeline.process_file('data/file.csv', output_mode='save')
```

Note: Error duration is controlled by `ErrorInjectionEngine.generate_duration()`
in error_injection.py (not in the config dict). Edit that function if you
want different duration distributions.

# 4. REPRODUCIBLE RESULTS

```python
import random
import numpy as np
from error_injection import ErrorInjectionPipeline

random.seed(42)
np.random.seed(42)

pipeline = ErrorInjectionPipeline()
df1 = pipeline.process_file('data/file.csv', output_mode='return')

random.seed(42)
np.random.seed(42)
df2 = pipeline.process_file('data/file.csv', output_mode='return')
# df1 and df2 are identical
```

# 5. FEATURE CUSTOMIZATION

```python
from error_injection import FEATURE_COLUMNS

print(FEATURE_COLUMNS)  # ['GHI', 'DHI', 'DNI']
```

To customize which features are modified, edit `FEATURE_COLUMNS` inside
error_injection.py and reimport.

# 6. COMMAND LINE

```bash
# Single file:
python error_injection.py data/STW_2023/STW_2023-01_QC.csv --mode save

# Multiple files (wildcard):
python error_injection.py "data/STW_2023/*.csv" --mode save

# Return mode (no writeback):
python error_injection.py data/file.csv --mode return
```

# 7. READ OUTPUT

```python
import json
from error_injection import DataManager

# Load synthetic file
_, df = DataManager.load_data('data/injected_error/file_errored.csv')

# Find flagged rows
bad_rows = df[df['Flag_GHI'] == 99]
print(f"Modified rows: {len(bad_rows)}")

# Read manifest
with open('data/injected_error/file_manifest.json') as f:
        manifest = json.load(f)
        print(f"Errors injected: {manifest['num_errors_injected']}")
        for err in manifest['errors']:
                print(f"  - {err['type']} at rows {err['start_idx']}-{err['end_idx']}")
```

# 8. AUGMENT TRAINING DATA

```python
import pandas as pd
from error_injection import ErrorInjectionPipeline, DataManager

_, df_orig = DataManager.load_data('data/STW_2023/STW_2023-01_QC.csv')

pipeline = ErrorInjectionPipeline()
df_synthetic = pipeline.process_file(
        'data/STW_2023/STW_2023-01_QC.csv',
        output_mode='return'
)

training_data = pd.concat([df_orig, df_synthetic] * 3, ignore_index=True)
# Now 6x more data (original + 5 synthetic versions)
```

# 9. VALIDATION WORKFLOW

```python
import json
from error_injection import DataManager

_, df_orig = DataManager.load_data('original.csv')
_, df_error = DataManager.load_data('original_errored.csv')

ghi_changed = (df_orig['GHI'] != df_error['GHI']).sum()
flagged = (df_error['Flag_GHI'] == 99).sum()

with open('original_manifest.json') as f:
        manifest = json.load(f)
        total_errors = manifest.get('num_errors_injected', 0)

assert ghi_changed > 0, "No values changed!"
assert flagged > 0, "No rows flagged as bad!"
assert total_errors > 0, "Manifest shows no injected errors!"
```

# 10. ERROR TYPES SUMMARY

- reduce_features: 1-50% drop with optional gaussian windowing
    - Config: reduction_percent, window_type, gaussian_sigma_minutes
    - Use case: Partial cloud cover or sensor degradation

- copy_from_day: Replace features with data from another day/time
    - Config: days_offset_range, features_to_copy
    - Use case: Cloud pattern insertion (realistic weather)

- beginning_dip: Deep dip at start that recovers
    - Config: dip_percent
    - Use case: Sunrise anomalies or warm-up issues

- cleaning_event: Sequential short dips per feature
    - Config: dip_percent, duration_minutes
    - Use case: Instrument cleaning/maintenance signature

# CONFIG QUICK REFERENCE

Key parameters in error_injection.py:

```python
FEATURE_COLUMNS = ['GHI', 'DHI', 'DNI']

ERROR_INJECTION_CONFIG = {
        'error_probability': 1.0,
        'error_count_range': (10, 50),
        'daytime_bias': 0.8,
        'sza_threshold': 90.0,
        'error_functions': {
                'reduce_features': {...},
                'copy_from_day': {...},
                'beginning_dip': {...},
                'cleaning_event': {...}
        }
}

OUTPUT_CONFIG = {
        'save_directory': 'data/injected_error',
        'filename_suffix': '_errored',
        'create_manifest': True,
        'manifest_filename_template': '{base_filename}_manifest.json'
}
```

# COMMON TASKS

- Inject many errors: set error_probability = 1.0 and error_count_range = (5, 10)
- Only daytime errors: set daytime_bias = 1.0
- Only nighttime errors: set daytime_bias = 0.0
- Only certain error types: adjust ErrorInjectionEngine.select_random_error_function()
- Get identical results twice: set the same random seed before each run

# FILENAMES REFERENCE

Input:  data/STW_2023/STW_2023-01_QC.csv
Output: data/injected_error/STW_2023-01_QC_errored.csv
                data/injected_error/STW_2023-01_QC_manifest.json

Format: {original_name}_errored.csv
                {original_name}_manifest.json

# TRAINING WITH SYNTHETIC DATA AUGMENTATION

run_learning_cycle.py can augment training data with synthetic errors.

How it works:
1. Loads training data for a date range
2. Randomly samples whole months within that range
3. Injects synthetic errors in memory
4. Combines synthetic data with original data
5. Trains the model on the combined dataset

Controls:
- `SYNTHETIC_DATA_RATIO` in run_learning_cycle.py
    - 2.0 = 2:1 real:synthetic (default)
    - 3.0 = 3:1 real:synthetic (less synthetic)
    - 1.0 = 1:1 real:synthetic (more synthetic)
    - 0.0 = no augmentation

# PREDICTION WITH FLAG PRESERVATION

By default, predictions preserve existing bad flags (99) and only update good
flags (1). Use `--overwrite-all-flags` when you intentionally want to replace
every flag in the file.

Truth Table:
Original Flag | Model Prediction | Result (DEFAULT)      | Result (--overwrite-all-flags)
------------- | ---------------- | --------------------- | ------------------------------
     1        |       1          |         1             |              1
     1        |       99         |         99            |              99
     99       |       1          |         99 ← preserved|              1 ← overwritten
     99       |       99         |         99            |              99

Usage Examples:

# Default behavior (PRESERVES existing bad flags):
python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv

# Overwrite ALL flags (use with caution - deletes manual QC):
python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv --overwrite-all-flags

# Preserved flags for date range (default):
python predict_with_saved_model.py --start "2025-07-01" --end "2025-08-31"

# Dry run (no file modification):
python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv --no-write

# Dry run + see what overwrite would do:
python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv --no-write --overwrite-all-flags

Workflow Example:
1. Train model on historical data
2. Run predictions on new data (creates baseline flags)
3. Manually review problematic days, add 99 flags where needed
4. Retrain model with updated training data
5. Run predictions again - manual flags are AUTOMATICALLY kept! ✓

Key Insight:
With the new default behavior, bad flags act as a "lock" - the model cannot
mark that data as good, even if it thinks it's fine. This preserves expert
domain knowledge and prevents accidentally overwriting careful manual QC work.

When to use --overwrite-all-flags:
- Starting fresh QC on a dataset
- You want to completely replace all existing flags
- You're sure no manual QC needs to be preserved
- Testing model performance on clean slate
"""
