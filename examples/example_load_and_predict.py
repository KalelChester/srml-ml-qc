"""
example_load_and_predict.py
============================

Simple example demonstrating how to load a saved model and make predictions.
This is useful if you want to integrate the model into your own workflow.
"""

import pandas as pd
import numpy as np
from solar_model import SolarHybridModel
from solar_features import add_features

# Configuration
SITE_CONFIG = {
    'latitude': 47.654,
    'longitude': -122.309,
    'altitude': 70,
    'timezone': 'Etc/GMT+8'
}


def example_predict_on_file(csv_path: str, target_col: str):
    """
    Example: Load a model and predict on a single file.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file with solar data
    target_col : str
        Target column to predict (e.g., 'Flag_GHI')
    """
    
    print(f"\n{'='*60}")
    print(f"Example: Predicting {target_col} on {csv_path}")
    print(f"{'='*60}\n")
    
    # 1. Load the saved model
    model_path = f'models/model_{target_col}.pkl'
    print(f"Step 1: Loading model from {model_path}...")
    model = SolarHybridModel.load_model(model_path)
    print("✓ Model loaded successfully\n")
    
    # 2. Load your data
    print(f"Step 2: Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, skiprows=43)
    df.columns = [c.strip() for c in df.columns]
    print(f"✓ Loaded {len(df)} rows\n")
    
    # 3. Add features (required for prediction)
    print("Step 3: Engineering features...")
    df_featured = add_features(df, SITE_CONFIG)
    print(f"✓ Added features, now have {len(df_featured.columns)} columns\n")
    
    # 4. Make predictions
    print("Step 4: Making predictions...")
    flags, probs = model.predict(df_featured, target_col, do_return_probs=True)
    print("✓ Predictions complete\n")
    
    # 5. Analyze results
    print("Step 5: Analyzing results...")
    n_good = int((flags == 1).sum())
    n_bad = int((flags == 99).sum())
    
    print(f"  Total samples: {len(flags)}")
    print(f"  GOOD flags (1): {n_good} ({n_good/len(flags)*100:.1f}%)")
    print(f"  BAD flags (99): {n_bad} ({n_bad/len(flags)*100:.1f}%)")
    print(f"  Mean confidence P(GOOD): {np.mean(probs):.3f}")
    print(f"  Median confidence: {np.median(probs):.3f}")
    print(f"  Min confidence: {np.min(probs):.3f}")
    print(f"  Max confidence: {np.max(probs):.3f}")
    
    # 6. Show some examples
    print("\nStep 6: Sample predictions (first 10 rows):")
    print("-" * 60)
    print(f"{'Index':<8} {'Flag':<8} {'Probability':<12} {'Interpretation'}")
    print("-" * 60)
    
    for i in range(min(10, len(flags))):
        flag_str = "GOOD" if flags[i] == 1 else "BAD"
        prob = probs[i]
        
        if prob >= 0.8:
            interp = "High confidence GOOD"
        elif prob >= 0.6:
            interp = "Likely GOOD"
        elif prob >= 0.4:
            interp = "Uncertain"
        elif prob >= 0.2:
            interp = "Likely BAD"
        else:
            interp = "High confidence BAD"
        
        print(f"{i:<8} {flag_str:<8} {prob:<12.4f} {interp}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("="*60 + "\n")
    
    return flags, probs


def example_batch_predict(file_list: list, target_col: str):
    """
    Example: Load a model once and predict on multiple files.
    This is more efficient than loading the model for each file.
    
    Parameters
    ----------
    file_list : list
        List of CSV file paths
    target_col : str
        Target column to predict
    """
    
    print(f"\n{'='*60}")
    print(f"Example: Batch prediction on {len(file_list)} files")
    print(f"{'='*60}\n")
    
    # Load model once
    model_path = f'models/model_{target_col}.pkl'
    print(f"Loading model from {model_path}...")
    model = SolarHybridModel.load_model(model_path)
    print("✓ Model loaded\n")
    
    results = {}
    
    for csv_path in file_list:
        print(f"Processing {csv_path}...")
        
        try:
            # Load and prepare data
            df = pd.read_csv(csv_path, skiprows=43)
            df.columns = [c.strip() for c in df.columns]
            df_featured = add_features(df, SITE_CONFIG)
            
            # Predict
            flags, probs = model.predict(df_featured, target_col, do_return_probs=True)
            
            # Store results
            results[csv_path] = {
                'flags': flags,
                'probs': probs,
                'n_good': int((flags == 1).sum()),
                'n_bad': int((flags == 99).sum())
            }
            
            print(f"  ✓ {results[csv_path]['n_good']} GOOD, {results[csv_path]['n_bad']} BAD\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
    
    print("="*60)
    print("Batch prediction complete!")
    print("="*60 + "\n")
    
    return results


if __name__ == '__main__':
    # Example 1: Predict on a single file
    # Uncomment and modify the path to match your data
    
    # example_predict_on_file(
    #     csv_path='data/STW_2025/STW_2025-07_QC.csv',
    #     target_col='Flag_GHI'
    # )
    
    # Example 2: Batch prediction
    # Uncomment and modify paths to match your data
    
    # file_list = [
    #     'data/STW_2025/STW_2025-07_QC.csv',
    #     'data/STW_2025/STW_2025-08_QC.csv',
    #     'data/STW_2025/STW_2025-09_QC.csv',
    # ]
    # 
    # example_batch_predict(file_list, 'Flag_GHI')
    
    print("="*60)
    print("To use these examples:")
    print("1. Uncomment one of the example calls above")
    print("2. Modify the file paths to match your data")
    print("3. Run: python example_load_and_predict.py")
    print("="*60)
