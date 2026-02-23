# Solar QC System

This project provides an end-to-end workflow for solar irradiance quality control (QC):
feature engineering, hybrid model training, prediction with flag preservation, manual
review tools, and synthetic error injection for robustness testing.

## Project Structure

- Core training and prediction
  - run_learning_cycle.py: Orchestrates feature engineering, model training, and prediction
  - predict_with_saved_model.py: Loads saved models and writes predictions back to CSVs
  - solar_model.py: Hybrid QC model (RandomForest + IsolationForest + RNN/Dense)
  - solar_features.py: Feature engineering (solar geometry, clearsky, QC features)

- Data handling and manual review
  - get_solar_data.py: Loads comprehensive-format CSVs and applies automated QC tests
  - SRML_ManualQC.py: GUI tool for manual QC review and flag editing

- Synthetic error injection
  - error_injection.py: Injects realistic synthetic errors for testing/augmentation
  - QUICK_REFERENCE.md: Error injection quick reference

- Hyperparameter tuning and analysis
  - hyperparameter_selection/selection.py: Block-based hyperparameter search driver
  - hyperparameter_selection/solar_model_modular.py: Modular model for tuning
  - performance/compute_feature_importance.py: RF + RNN feature importance report

- Examples
  - examples/example_load_and_predict.py: Load a model and predict
  - examples/example_preserve_flags.py: Flag preservation behavior
  - examples/example_rnn_analysis.py: RNN behavior and analysis
  - examples/error_injection_examples.py: Error injection usage patterns

## Data Format

All primary CSVs are in the SRML comprehensive format:
- Rows 0-42: Metadata headers
- Row 43: Column names
- Row 44+: Data rows

Common columns:
- Timestamp: YYYY-MM-DD--HH:MM:SS (local time, no conversion)
- Irradiance: GHI, DNI, DHI
- Flags: Flag_GHI, Flag_DNI, Flag_DHI
- Optional: Flag_*_prob (model confidence)

## Core Workflow

### 1) Train and Predict (One-Step Cycle)

run_learning_cycle.py is the primary entry point.

What it does:
1. Loads CSVs from data/
2. Builds features via solar_features.add_features
3. Optionally augments training data with synthetic errors
4. Trains models for each target flag
5. Saves models to models/
6. Runs predictions on a date range and writes flags back to CSVs

Example (edit date ranges in the file first):

```bash
python run_learning_cycle.py
```

Key configuration knobs:
- SITE_CONFIG: latitude/longitude/altitude/timezone metadata for solar features
- SYNTHETIC_DATA_RATIO: real:synthetic ratio for augmentation
- TRAIN_START/END, PRED_START/END: date ranges for training and prediction

### 2) Predict Only (Using Saved Models)

Use predict_with_saved_model.py to run predictions on specific files or ranges.
By default, existing bad flags (99) are preserved.

```bash
# Predict a single file
python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv

# Predict a date range
python predict_with_saved_model.py --start "2025-07-01" --end "2025-08-31"

# Overwrite all flags (use with caution)
python predict_with_saved_model.py --start "2025-07-01" --end "2025-08-31" --overwrite-all-flags
```

## Manual Review Workflow

SRML_ManualQC.py provides a GUI for manual flag review and corrections.
It can load CSVs, visualize time series, and apply bulk edits to flags.

```bash
python SRML_ManualQC.py
```

## Synthetic Error Injection

error_injection.py creates realistic synthetic errors and flags them as bad.
This is used for robustness testing and optional training augmentation.

```bash
# Inject errors and save with manifest
python error_injection.py data/STW_2023/STW_2023-01_QC.csv --mode save

# Inject errors and return dataframe (no writeback)
python error_injection.py data/STW_2023/STW_2023-01_QC.csv --mode return
```

See QUICK_REFERENCE.md for additional usage patterns and config notes.

## Hyperparameter Selection

hyperparameter_selection/selection.py runs a block-based grid search using
manual_confirmed_data_backup as labeled data. Each CSV is treated as a monthly
block to preserve temporal structure.

```bash
python hyperparameter_selection/selection.py
```

## Feature Importance

performance/compute_feature_importance.py produces a log with:
- RandomForest feature importances
- RNN correlation-based feature importances

```bash
python performance/compute_feature_importance.py
```

## Flags and Probabilities

- Flag values: 1 = GOOD, 99 = BAD
- Probability columns: Flag_*_prob store P(GOOD) in [0.0, 1.0]
- Default prediction behavior preserves existing bad flags (99)

## Models

Trained models are stored in models/ as model_Flag_*.pkl. Each model file
includes the RF, optional calibration, isolation forest, scaler, NN weights,
and the feature list used for training.

## Configuration Notes

- Timestamps are assumed to already be in local time and are not converted.
- SITE_CONFIG is defined in config.py and imported by scripts that need it.
- For error injection, configuration is defined in error_injection.py.

## Smoke Tests

A lightweight smoke test script is provided to validate core functionality on a
small sample of data.

```bash
python tests/smoke_tests.py
```
