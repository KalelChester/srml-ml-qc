"""
prediction_comparison.py
========================

Load a random selection of data files, make predictions without modifying them,
and compare prediction accuracy against current flags for GHI, DNI, and DHI.

Usage:
    # Predict on 5 random files (default)
    python prediction_comparison.py
    
    # Predict on 10 random files
    python prediction_comparison.py 10
    
    # Predict on all files
    python prediction_comparison.py -1
    
    # Predict on specific file(s)
    python prediction_comparison.py --file data/STW_2025/STW_2025-07_QC.csv
    python prediction_comparison.py --file data/STW_2025/STW_2025-07_QC.csv data/STW_2025/STW_2025-08_QC.csv
"""

import os
import sys
import glob
import random
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from solar_features import add_features
from solar_model import SolarHybridModel
from config import SITE_CONFIG
from io_utils import load_qc_csvs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================================
# CONFIG
# ============================================================================
DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'
LOG_FOLDER = 'log_files'
HEADER_ROWS_SKIP = 43
TS_COL = 'YYYY-MM-DD--HH:MM:SS'
TARGETS = ['Flag_GHI', 'Flag_DNI', 'Flag_DHI']


def get_all_data_files():
    """Get all CSV files from data folder (excluding injected_error folder)."""
    all_files = sorted(glob.glob(os.path.join(DATA_FOLDER, '**', '*.csv'), recursive=True))
    # Exclude files in the injected_error folder
    all_files = [f for f in all_files if 'injected_error' not in f]
    return all_files


def select_random_files(num_files):
    """
    Select random files to process.
    
    Parameters
    ----------
    num_files : int
        Number of files to select. If -1, select all files.
    
    Returns
    -------
    list
        List of file paths to process
    """
    all_files = get_all_data_files()
    
    if num_files == -1:
        # Use all files
        return all_files
    
    if num_files > len(all_files):
        print(f"Warning: Requested {num_files} files but only {len(all_files)} available.")
        print(f"Using all {len(all_files)} files instead.")
        return all_files
    
    if num_files < 1:
        raise ValueError("Number of files must be >= 1 or -1 for all files")
    
    return random.sample(all_files, num_files)


def load_and_predict(file_paths, targets):
    """
    Load data files and make predictions without modifying them.
    
    Parameters
    ----------
    file_paths : list
        List of CSV file paths to predict on
    targets : list
        List of target columns (e.g., ['Flag_GHI', 'Flag_DNI', 'Flag_DHI'])
    
    Returns
    -------
    dict
        Dictionary mapping target_col -> DataFrame with predictions and current flags
    """
    print(f"\n{'='*70}")
    print(f"Loading {len(file_paths)} file(s)...")
    print(f"{'='*70}")
    
    # Load data
    raw = load_qc_csvs(file_paths, header_rows_skip=HEADER_ROWS_SKIP, ts_col=TS_COL)
    
    if raw.empty:
        print("No data found!")
        return {}
    
    print(f"Loaded {len(raw)} samples from {len(file_paths)} file(s)")
    
    # Add features
    print("Adding features...")
    full = add_features(raw, SITE_CONFIG)
    full['Timestamp_dt'] = pd.to_datetime(full['Timestamp_dt'], errors='coerce')
    full.index = full['Timestamp_dt']
    
    results = {}
    
    # Load models and predict for each target
    for target in targets:
        model_path = os.path.join(MODEL_FOLDER, f'model_{target}.pkl')
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  Model not found: {model_path}")
            print(f"   Please train the model first by running run_learning_cycle.py")
            continue
        
        print(f"\n--- Predicting {target} ---")
        
        # Load the trained model
        model = SolarHybridModel.load_model(model_path)
        
        # Prepare data for prediction
        pred_df = full.copy()
        
        # Add IF_Score if detector exists
        if getattr(model, 'if_det', None) is not None:
            try:
                pred_df['IF_Score'] = model.if_det.decision_function(
                    pred_df[model.common_features].fillna(0.0)
                )
            except Exception:
                pred_df['IF_Score'] = 0.0
        
        # Make predictions
        print(f"Predicting {len(pred_df)} samples...")
        flags, probs = model.predict(pred_df, target, do_return_probs=True)
        
        # Store results with both predictions and current flags
        pred_df[f'{target}_predicted'] = flags
        pred_df[f'{target}_prob'] = probs
        
        # Get current flags (observations in the data)
        pred_df[f'{target}_current'] = pred_df[target].copy()
        
        results[target] = pred_df
        
        # Print summary
        n_good_pred = int((flags == 1).sum())
        n_bad_pred = int((flags == 99).sum())
        mean_prob = float(np.mean(probs))
        
        print(f"  Predicted: {n_good_pred} GOOD ({n_good_pred/len(flags)*100:.1f}%), "
              f"{n_bad_pred} BAD ({n_bad_pred/len(flags)*100:.1f}%)")
        print(f"  Mean confidence (P(GOOD)): {mean_prob:.3f}")
    
    print(f"\n{'='*70}\n")
    return results


def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, and F1 score.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth flags (1 or 11 for GOOD, 99 for BAD)
    y_pred : array-like
        Predicted flags (1 or 11 for GOOD, 99 for BAD)
    
    Returns
    -------
    dict
        Dictionary with 'accuracy', 'precision', 'recall', 'f1' keys
    """
    # Normalize flags: treat 1 and 11 as GOOD (0), 99 as BAD (1)
    y_true_normalized = (y_true == 99).astype(int)
    y_pred_normalized = (y_pred == 99).astype(int)
    
    # Handle case where all predictions are the same class
    try:
        accuracy = accuracy_score(y_true_normalized, y_pred_normalized)
        # pos_label=1 because we normalized BAD to 1
        precision = precision_score(y_true_normalized, y_pred_normalized, pos_label=1, zero_division=0)
        recall = recall_score(y_true_normalized, y_pred_normalized, pos_label=1, zero_division=0)
        f1 = f1_score(y_true_normalized, y_pred_normalized, pos_label=1, zero_division=0)
    except Exception as e:
        print(f"Warning: Error computing metrics: {e}")
        accuracy = precision = recall = f1 = np.nan
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compare_predictions(results, targets):
    """
    Compare predictions against current flags.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping target_col -> DataFrame with predictions
    targets : list
        List of target columns
    
    Returns
    -------
    dict
        Dictionary with comparison results for each target
    """
    comparison = {}
    
    for target in targets:
        if target not in results:
            continue
        
        df = results[target]
        
        # Get current flags and predictions
        y_true = df[f'{target}_current'].values
        y_pred = df[f'{target}_predicted'].values
        
        # Remove invalid true flags and NaNs
        valid_idx = np.isin(y_true, [1, 11, 99]) & ~np.isnan(y_pred)
        
        y_true_clean = y_true[valid_idx]
        y_pred_clean = y_pred[valid_idx]
        
        # Normalize flags: 1 and 11 are GOOD (0), 99 is BAD (1)
        y_true_normalized = (y_true_clean == 99).astype(int)
        y_pred_normalized = (y_pred_clean == 99).astype(int)
        
        # Compute metrics
        metrics = compute_metrics(y_true_clean, y_pred_clean)
        
        # Count agreements/disagreements
        agree = np.sum(y_true_normalized == y_pred_normalized)
        disagree = np.sum(y_true_normalized != y_pred_normalized)
        
        comparison[target] = {
            'n_samples': len(y_true_clean),
            'n_agree': agree,
            'n_disagree': disagree,
            'pct_agree': 100.0 * agree / len(y_true_clean) if len(y_true_clean) > 0 else 0.0,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }
    
    return comparison


def format_metrics_table(comparison, targets):
    """
    Format comparison results as a readable table.
    
    Parameters
    ----------
    comparison : dict
        Comparison results from compare_predictions()
    targets : list
        List of target columns
    
    Returns
    -------
    str
        Formatted table string
    """
    lines = []
    
    lines.append(f"\n{'='*90}")
    lines.append(f"{'PREDICTION COMPARISON RESULTS':^90}")
    lines.append(f"{'='*90}\n")
    
    # Header
    lines.append(f"{'Feature':<15} {'Samples':<12} {'Agreement':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    lines.append(f"{'-'*90}")
    
    # Data rows
    for target in targets:
        if target not in comparison:
            lines.append(f"{target:<15} {'N/A':^12} {'N/A':^20}")
            continue
        
        stats = comparison[target]
        n_samples = stats['n_samples']
        n_agree = stats['n_agree']
        pct_agree = stats['pct_agree']
        accuracy = stats['accuracy']
        precision = stats['precision']
        recall = stats['recall']
        f1 = stats['f1']
        
        agree_str = f"{n_agree}/{n_samples} ({pct_agree:.1f}%)"
        
        lines.append(f"{target:<15} {n_samples:<12} {agree_str:<20} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    lines.append(f"{'='*90}\n")
    
    return '\n'.join(lines)


def log_to_file(results_dict, file_paths, output_file):
    """
    Write comparison results to a log file.
    
    Parameters
    ----------
    results_dict : dict
        Comparison results from compare_predictions()
    file_paths : list
        List of files that were processed
    output_file : str
        Path to output log file
    """
    with open(output_file, 'w') as f:
        # Header
        f.write(f"{'='*90}\n")
        f.write(f"PREDICTION COMPARISON LOG\n")
        f.write(f"{'='*90}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Files processed
        f.write(f"Files Processed ({len(file_paths)}):\n")
        f.write(f"{'-'*90}\n")
        for i, fp in enumerate(file_paths, 1):
            f.write(f"  {i:2d}. {fp}\n")
        f.write("\n")
        
        # Results table
        f.write(f"\nPREDICTION COMPARISON RESULTS:\n")
        f.write(f"{'-'*90}\n")
        
        # Header row
        f.write(f"{'Feature':<15} {'Samples':<12} {'Agreement':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write(f"{'-'*90}\n")
        
        # Data rows
        for target in TARGETS:
            if target not in results_dict:
                f.write(f"{target:<15} {'N/A':^12} {'Model not found':^20}\n")
                continue
            
            stats = results_dict[target]
            n_samples = stats['n_samples']
            n_agree = stats['n_agree']
            pct_agree = stats['pct_agree']
            accuracy = stats['accuracy']
            precision = stats['precision']
            recall = stats['recall']
            f1_score = stats['f1']
            
            agree_str = f"{n_agree}/{n_samples} ({pct_agree:.1f}%)"
            
            f.write(f"{target:<15} {n_samples:<12} {agree_str:<20} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1_score:<12.4f}\n")
        
        f.write(f"\n{'='*90}\n")
        f.write(f"\nDetailed Metrics Description:\n")
        f.write(f"  Accuracy:  Percentage of predictions that match the current flags\n")
        f.write(f"  Precision: Of predicted BAD flags, what % were actually BAD in current data\n")
        f.write(f"  Recall:    Of actual BAD flags in current data, what % did we predict as BAD\n")
        f.write(f"  F1:        Harmonic mean of Precision and Recall\n")
        f.write(f"\n{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare model predictions against current data flags',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on 5 random files (default)
  python prediction_comparison.py
  
  # Predict on 10 random files
  python prediction_comparison.py 10
  
  # Predict on all files
  python prediction_comparison.py -1
  
  # Predict on specific file(s)
  python prediction_comparison.py --file data/STW_2025/STW_2025-07_QC.csv
  python prediction_comparison.py --file data/STW_2025/STW_2025-07_QC.csv data/STW_2025/STW_2025-08_QC.csv
        """
    )
    
    parser.add_argument('num_files', type=int, nargs='?', default=None,
                        help='Number of random files to predict on (default: 5, use -1 for all files)')
    parser.add_argument('--file', nargs='+', help='Specific file(s) to predict on')
    
    args = parser.parse_args()
    
    # Determine which files to use
    if args.file:
        # Use specified files
        file_paths = args.file
    else:
        # Use random selection
        num_files = args.num_files if args.num_files is not None else 5
        try:
            file_paths = select_random_files(num_files)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    if not file_paths:
        print("No files found to process!")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Selected {len(file_paths)} file(s) for prediction comparison")
    print(f"{'='*70}")
    for fp in file_paths:
        print(f"  - {fp}")
    
    # Load and predict
    results = load_and_predict(file_paths, TARGETS)
    
    if not results:
        print("No predictions were made!")
        sys.exit(1)
    
    # Compare predictions
    comparison = compare_predictions(results, TARGETS)
    
    # Print to terminal
    output_str = format_metrics_table(comparison, TARGETS)
    print(output_str)
    
    # Write to log file
    log_dir = Path(LOG_FOLDER)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'prediction_comparison_{timestamp}.txt'
    
    log_to_file(comparison, file_paths, str(log_file))
    print(f"Log file saved to: {log_file}")


if __name__ == '__main__':
    main()
