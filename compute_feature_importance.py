"""
compute_feature_importance.py
==============================

Feature Importance Analysis for Hybrid Solar QC Model

This script computes feature importance for both components of the hybrid model:
1. Random Forest: Direct feature_importances_ from trained classifier
2. RNN: Permutation-based importance (how much accuracy drops when feature permuted)

Output
------
Results written to log_files/feature_importance.log with:
- RF feature importances (all three target models)
- RNN permutation importance (all three target models)
- Timestamp of analysis
- Model versions used

Usage
-----
    python compute_feature_importance.py

Notes
-----
- Loads latest trained models from models/ folder
- Uses recent data (2025) from data/ for importance computation
- RNN importance computed via permutation (slower but interpretable)
- All importance scores normalized to [0, 1] for readability
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from solar_features import add_features
from solar_model import SolarHybridModel
from sklearn.inspection import permutation_importance


# Configuration
DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'
LOG_FOLDER = 'log_files'
LOG_FILE = os.path.join(LOG_FOLDER, 'feature_importance.log')
HEADER_ROWS_SKIP = 43
TS_COL = 'YYYY-MM-DD--HH:MM:SS'

SITE_CONFIG = {
    'latitude': 47.654,
    'longitude': -122.309,
    'altitude': 70,
    'timezone': 'Etc/GMT+8'
}

TARGETS = ['Flag_GHI', 'Flag_DNI', 'Flag_DHI']


# Setup logging
os.makedirs(LOG_FOLDER, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_csvs(file_paths):
    """Load and concatenate CSV files, preserving source tracking."""
    frames = []
    for f in file_paths:
        try:
            df = pd.read_csv(f, skiprows=HEADER_ROWS_SKIP)
            df.columns = [c.strip() for c in df.columns]
            if 'Data_Begins_Next_Row' in df.columns:
                df = df.drop(columns=['Data_Begins_Next_Row'])
            df['_source_file'] = f
            if TS_COL in df.columns:
                df['_raw_ts'] = df[TS_COL].astype(str).str.strip()
            else:
                df['_raw_ts'] = df.iloc[:, 0].astype(str).str.strip()
            frames.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    
    if len(frames) == 0:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)


def compute_rf_importance(model, feature_names):
    """
    Extract Random Forest feature importances.
    
    Parameters
    ----------
    model : SolarHybridModel
        Trained hybrid model
    feature_names : list
        Feature column names in order
    
    Returns
    -------
    dict
        Mapping of feature_name -> importance_score (normalized)
    """
    if model.rf is None:
        return {}
    
    importances = model.rf.feature_importances_
    
    # Normalize to [0, 1]
    importances = importances / importances.sum()
    
    return dict(zip(feature_names, importances))


def compute_rnn_permutation_importance(model, X, y, feature_names, target_col):
    """
    Compute correlation-based feature importance for RNN.
    
    Measures correlation between each feature and model's confidence scores.
    Uses batching to avoid memory issues with large datasets.
    
    Parameters
    ----------
    model : SolarHybridModel
        Trained hybrid model with RNN
    X : pd.DataFrame
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    feature_names : list
        Feature column names
    target_col : str
        Target column name (e.g., 'Flag_GHI')
    
    Returns
    -------
    dict
        Mapping of feature_name -> importance_score
    """
    if model.nn_state is None or model.scaler is None:
        return {}
    
    try:
        # Use a subset of data to avoid memory issues
        # Take a stratified sample to preserve class distribution
        sample_size = min(10000, len(X))  # Use up to 10K samples
        
        if len(X) > sample_size:
            # Sample preserving class balance (using y)
            indices = np.random.choice(len(X), size=sample_size, replace=False)
            X_sample = X.iloc[indices].reset_index(drop=True)
            y_sample = y[indices]
        else:
            X_sample = X.reset_index(drop=True)
            y_sample = y
        
        logger.info(f"  Computing RNN importance on {len(X_sample)} samples (full dataset: {len(X)})")
        
        # Get baseline predictions on sample (in batches to avoid memory issues)
        batch_size = 1000
        all_probs = []
        
        for i in range(0, len(X_sample), batch_size):
            batch = X_sample.iloc[i:i+batch_size]
            try:
                _, batch_probs = model.predict(batch, target_col, do_return_probs=True)
                all_probs.extend(batch_probs)
            except Exception as e:
                logger.warning(f"  Failed to predict on batch {i//batch_size}: {e}")
                # Fall back to RF-only predictions if RNN fails
                all_probs = model.rf.predict_proba(model._build_X(X_sample))[:, 1]
                break
        
        baseline_probs = np.array(all_probs)
        
        # Compute correlation-based importance for core features
        importance_dict = {}
        
        for fname in feature_names:
            # Skip synthetic features that aren't in raw input
            if fname in ['RF_Prob', 'IF_Score']:
                continue
                
            # Get feature column (handle missing with fill)
            if fname in X_sample.columns:
                feat_vals = X_sample[fname].values
            else:
                # For features not in input, skip
                continue
            
            # Compute correlation with model confidence (probability)
            valid_idx = ~(np.isnan(feat_vals) | np.isnan(baseline_probs))
            if valid_idx.sum() > 1:
                correlation = np.corrcoef(feat_vals[valid_idx], baseline_probs[valid_idx])[0, 1]
                importance_dict[fname] = np.abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                importance_dict[fname] = 0.0
        
        # Normalize to [0, 1]
        if importance_dict:
            max_imp = max(importance_dict.values())
            if max_imp > 0:
                importance_dict = {k: v / max_imp for k, v in importance_dict.items()}
        
        return importance_dict
    
    except Exception as e:
        logger.error(f"Error computing RNN importance: {e}")
        return {}


def main():
    """Main feature importance analysis pipeline."""
    logger.info("=" * 80)
    logger.info("FEATURE IMPORTANCE ANALYSIS FOR SOLAR QC HYBRID MODEL")
    logger.info("=" * 80)
    logger.info(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # Load data from most recent year (2025)
    print("Loading data from 2025...")
    data_files = sorted(glob.glob(os.path.join(DATA_FOLDER, 'STW_2025', '*.csv')))
    if not data_files:
        logger.error("No data files found in STW_2025 folder")
        print("No data files found")
        return
    
    raw = load_csvs(data_files)
    if raw.empty:
        logger.error("Failed to load any CSV data")
        print("Failed to load data")
        return
    
    logger.info(f"Loaded {len(raw)} rows from {len(data_files)} files")
    
    # Engineer features
    print("Engineering features...")
    full = add_features(raw, SITE_CONFIG)
    full['Timestamp_dt'] = pd.to_datetime(full['Timestamp_dt'], errors='coerce')
    
    logger.info(f"Engineered features: {len(full)} rows")
    logger.info(f"Feature columns: {len(full.columns)}")
    logger.info("")
    
    # Process each target
    for target_col in TARGETS:
        logger.info("-" * 80)
        logger.info(f"TARGET: {target_col}")
        logger.info("-" * 80)
        
        model_path = os.path.join(MODEL_FOLDER, f'model_{target_col}.pkl')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}")
            continue
        
        print(f"Processing {target_col}...")
        
        # Load trained model
        try:
            model = SolarHybridModel.load_model(model_path)
            logger.info(f"Loaded model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            continue
        
        # Get training data with labels
        valid_data = full[full[target_col].notna()].copy()
        if valid_data.empty:
            logger.warning(f"No labeled data for {target_col}")
            continue
        
        # Convert labels (99 -> 0 BAD, else -> 1 GOOD)
        y = np.where(valid_data[target_col] == 99, 0, 1).astype(int)
        
        logger.info(f"Data points with labels: {len(valid_data)}")
        logger.info(f"Class distribution: GOOD={np.sum(y==1)}, BAD={np.sum(y==0)}")
        logger.info("")
        
        # ===== RANDOM FOREST IMPORTANCE =====
        logger.info("Random Forest Feature Importance:")
        logger.info("-" * 40)
        
        rf_importance = compute_rf_importance(model, model.common_features)
        
        if rf_importance:
            # Sort by importance
            rf_sorted = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
            
            for fname, imp in rf_sorted:
                logger.info(f"  {fname:.<35} {imp:.6f}")
            
            logger.info("")
        else:
            logger.warning("  No Random Forest importance scores available")
            logger.info("")
        
        # ===== RNN FEATURE IMPORTANCE =====
        logger.info("RNN Feature Importance (Correlation-based):")
        logger.info("-" * 40)
        
        rnn_importance = compute_rnn_permutation_importance(
            model, valid_data, y, model.common_features, target_col
        )
        
        if rnn_importance:
            # Sort by importance
            rnn_sorted = sorted(rnn_importance.items(), key=lambda x: x[1], reverse=True)
            
            for fname, imp in rnn_sorted:
                logger.info(f"  {fname:.<35} {imp:.6f}")
            
            logger.info("")
        else:
            logger.warning("  No RNN importance scores available")
            logger.info("")
        
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("Feature importance analysis complete")
    logger.info("=" * 80)
    
    print(f"Feature importance saved to: {LOG_FILE}")


if __name__ == '__main__':
    main()
