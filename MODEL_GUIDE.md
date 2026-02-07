# Solar QC Model: Save and Load Guide

## Overview

The solar QC model can now be saved after training and loaded later for predictions without retraining. This saves time and ensures consistent predictions using the same trained model.

## Quick Start

### 1. Train and Save Models

```bash
python run_learning_cycle.py
```

This will:
- Train models on your configured date range
- Save three model files to the `models/` directory:
  - `model_Flag_GHI.pkl`
  - `model_Flag_DNI.pkl`
  - `model_Flag_DHI.pkl`

### 2. Use Saved Models

#### Option A: Command Line Tool

```bash
# Predict on a specific file
python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv

# Predict on multiple files
python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv data/STW_2025/STW_2025-08_QC.csv

# Predict on a date range
python predict_with_saved_model.py --start "2025-07-01" --end "2025-08-31"

# Only predict specific targets
python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv --targets Flag_GHI

# Dry run (don't write back to files)
python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv --no-write
```

#### Option B: Python Code

```python
from solar_model import SolarHybridModel
from solar_features import add_features
import pandas as pd

# Load the saved model
model = SolarHybridModel.load_model('models/model_Flag_GHI.pkl')

# Load your data
df = pd.read_csv('data/STW_2025/STW_2025-07_QC.csv', skiprows=43)
df.columns = [c.strip() for c in df.columns]

# Add features
SITE_CONFIG = {
    'latitude': 47.654,
    'longitude': -122.309,
    'altitude': 70,
    'timezone': 'Etc/GMT+8'
}
df_featured = add_features(df, SITE_CONFIG)

# Make predictions
flags, probs = model.predict(df_featured, 'Flag_GHI', do_return_probs=True)

# Use the results
print(f"Good samples: {(flags == 1).sum()}")
print(f"Bad samples: {(flags == 99).sum()}")
print(f"Mean confidence: {probs.mean():.3f}")
```

## What's Included in Saved Models

Each `.pkl` file contains:

1. **RandomForest Classifier** - Base tree model
2. **Calibrated RF** - Probability calibration (if available)
3. **IsolationForest** - Unsupervised anomaly detector
4. **StandardScaler** - Feature normalization parameters
5. **Neural Network Weights** - JAX/Flax DenseNN parameters
6. **Feature List** - Exact features used during training
7. **Version Info** - For compatibility tracking

## Model Files

- **Size**: Each model is typically 10-50 MB
- **Location**: `models/` directory
- **Format**: Python pickle (`.pkl`)
- **Compatibility**: Requires same Python environment (sklearn, JAX, etc.)

## Workflow Examples

### Scenario 1: Train Once, Predict Many Times

```bash
# Train on historical data (once)
python run_learning_cycle.py

# Use the trained model on new data (many times)
python predict_with_saved_model.py --file new_data/STW_2025-10_QC.csv
python predict_with_saved_model.py --file new_data/STW_2025-11_QC.csv
python predict_with_saved_model.py --file new_data/STW_2025-12_QC.csv
```

### Scenario 2: Retrain Periodically

```bash
# Week 1: Train on Jan-Apr data
python run_learning_cycle.py  # Uses config: train_start='2025-01-01', train_end='2025-04-30'

# Weeks 2-4: Use saved models
python predict_with_saved_model.py --start "2025-05-01" --end "2025-05-31"

# Week 5: Retrain with more data
# Edit run_learning_cycle.py to extend train_end to '2025-05-31'
python run_learning_cycle.py  # Overwrites old models with new ones
```

### Scenario 3: Integration into Your Pipeline

```python
# your_pipeline.py
from solar_model import SolarHybridModel
from solar_features import add_features

def qc_new_data(csv_path):
    """Add this function to your existing pipeline."""
    # Load model (do this once at startup, not per-file!)
    model = SolarHybridModel.load_model('models/model_Flag_GHI.pkl')
    
    # Your existing data loading code
    df = load_your_data(csv_path)
    
    # Add QC features
    df = add_features(df, SITE_CONFIG)
    
    # Get QC flags
    flags, probs = model.predict(df, 'Flag_GHI', do_return_probs=True)
    
    # Continue your pipeline
    df['Flag_GHI'] = flags
    df['Flag_GHI_prob'] = probs
    
    return df
```

## Advantages

✅ **Speed**: No retraining needed for new data  
✅ **Consistency**: Same model = reproducible results  
✅ **Portability**: Share models between machines  
✅ **Efficiency**: Load once, predict many times  
✅ **Versioning**: Track which model version produced which results  

## Important Notes

1. **Features Must Match**: New data must have the same features that were used during training
2. **Environment**: Models require the same Python packages (sklearn, JAX, pandas, etc.)
3. **Site Config**: Use the same `SITE_CONFIG` (lat/lon/timezone) for consistency
4. **Overwriting**: Running `run_learning_cycle.py` will overwrite existing model files

## Troubleshooting

**Model file not found**
```
Solution: Run `python run_learning_cycle.py` first to train and save models
```

**Feature mismatch errors**
```
Solution: Ensure add_features() is called with the same SITE_CONFIG
```

**Poor predictions on new data**
```
Solution: Retrain the model with data from a similar time period or conditions
```

**Out of memory errors**
```
Solution: Process files in smaller batches or use --file option for single files
```

## See Also

- `example_load_and_predict.py` - Detailed code examples
- `models/README.md` - Model storage documentation
- `run_learning_cycle.py` - Training script with save functionality
- `predict_with_saved_model.py` - Full-featured prediction script
