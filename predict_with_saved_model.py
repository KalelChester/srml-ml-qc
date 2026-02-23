"""
predict_with_saved_model.py
============================

Demonstrates how to load a saved model and make predictions on new data
without retraining.

IMPORTANT: By default, this script PRESERVES existing bad flags (99) and only
updates good flags (1). This protects manual QC work. Use --overwrite-all-flags
to replace all flags with model predictions.

Usage examples:
    # Predict on a single file (preserves existing bad flags):
    python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv
    
    # Predict on a date range (preserves existing bad flags):
    python predict_with_saved_model.py --start "2025-07-01" --end "2025-08-31"
    
    # Overwrite ALL flags (use with caution - deletes manual QC):
    python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv --overwrite-all-flags
    
    # Specify which models to use:
    python predict_with_saved_model.py --start "2025-07-01" --end "2025-08-31" --targets Flag_GHI Flag_DNI
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np

from solar_features import add_features
from solar_model import SolarHybridModel
from config import SITE_CONFIG
from io_utils import load_qc_csvs


# ---------------- CONFIG ----------------
DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'
HEADER_ROWS_SKIP = 43
HEADER_ROWS_PRESERVE = 44
TS_COL = 'YYYY-MM-DD--HH:MM:SS'

# Confidence thresholds (match training settings)
HIGH_THRESH = 0.51
LOW_THRESH = 0.49




def write_predictions(df_preds: pd.DataFrame, target_col: str, preserve_bad_flags: bool = True):
    """Write predictions back to source files.
    
    Parameters
    ----------
    df_preds : pd.DataFrame
        Dataframe containing predictions with '_source_file', '_raw_ts', target_col, and prob columns
    target_col : str
        Name of the flag column (e.g., 'Flag_GHI')
    preserve_bad_flags : bool, default=True
        If True (DEFAULT), preserves existing bad flags (99) and only updates good flags (1).
        This implements an "OR" operation where bad flags stay bad, protecting manual QC.
        If False, overwrites all flags with model predictions.
    """
    prob_col = f"{target_col}_prob"
    
    for path, g in df_preds.groupby('_source_file'):
        try:
            # Load original file
            orig = pd.read_csv(path, skiprows=HEADER_ROWS_SKIP)
            orig.columns = [c.strip() for c in orig.columns]
            if 'Data_Begins_Next_Row' in orig.columns:
                orig = orig.drop(columns=['Data_Begins_Next_Row'])
            
            # Create mapping for predictions
            pred_map = g.set_index('_raw_ts')[target_col].to_dict()
            prob_map = g.set_index('_raw_ts')[prob_col].to_dict() if prob_col in g.columns else {}
            
            # Update flags
            def update_flag(row):
                ts = str(row[TS_COL]).strip() if TS_COL in row.index else str(row.iloc[0]).strip()
                predicted_flag = pred_map.get(ts, row.get(target_col, 0))
                
                if preserve_bad_flags:
                    # Preserve existing bad flags (99), only update good flags (1)
                    original_flag = row.get(target_col, 1)
                    if original_flag == 99:
                        return 99  # Keep the existing bad flag
                    else:
                        return predicted_flag  # Update with prediction
                else:
                    # Default: overwrite with prediction
                    return predicted_flag
            
            orig[target_col] = orig.apply(update_flag, axis=1)
            
            # Update probabilities
            if prob_map:
                def update_prob(row):
                    ts = str(row[TS_COL]).strip() if TS_COL in row.index else str(row.iloc[0]).strip()
                    return prob_map.get(ts, row.get(prob_col, np.nan))
                
                orig[prob_col] = orig.apply(update_prob, axis=1)
            
            # Write back with header preserved
            with open(path, 'r') as f:
                header_lines = [next(f) for _ in range(HEADER_ROWS_PRESERVE)]
            
            with open(path, 'w', newline='') as f:
                f.writelines(header_lines)
                orig.to_csv(f, index=False, header=False)
            
            print(f"  Updated {path}")
            
        except Exception as e:
            print(f"  Failed to write {path}: {e}")


# ---------------- Main prediction function ----------------
def predict_with_model(file_paths: list, targets: list, write_back: bool = True, 
                       preserve_bad_flags: bool = True):
    """
    Load saved models and predict on specified files.
    
    Parameters
    ----------
    file_paths : list
        List of CSV file paths to predict on
    targets : list
        List of target columns (e.g., ['Flag_GHI', 'Flag_DNI', 'Flag_DHI'])
    write_back : bool
        If True, write predictions back to source files
    preserve_bad_flags : bool, default=True
        If True (DEFAULT), preserves existing bad flags (99) and only updates good flags (1).
        This protects manual QC flags from being overwritten by model predictions.
        If False, all flags are overwritten with model predictions.
    
    Returns
    -------
    dict
        Dictionary mapping target_col -> predictions DataFrame
    """
    
    # Load data
    print(f"\nLoading {len(file_paths)} file(s)...")
    raw = load_qc_csvs(file_paths, header_rows_skip=HEADER_ROWS_SKIP, ts_col=TS_COL)
    
    if raw.empty:
        print("No data found!")
        return {}
    
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
        
        print(f"\n=== Predicting {target} ===")
        
        # Load the trained model
        print(f"Loading model from {model_path}...")
        model = SolarHybridModel.load_model(model_path)
        
        # Add IF_Score if detector exists
        pred_df = full.copy()
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
        
        # Store results
        prob_col = f"{target}_prob"
        pred_df[target] = flags
        pred_df[prob_col] = probs
        
        # Statistics
        n_good = int((flags == 1).sum())
        n_bad = int((flags == 99).sum())
        mean_prob = float(np.mean(probs))
        
        print(f"Results: {n_good} GOOD ({n_good/len(flags)*100:.1f}%), "
              f"{n_bad} BAD ({n_bad/len(flags)*100:.1f}%)")
        print(f"Mean confidence (P(GOOD)): {mean_prob:.3f}")
        
        # Write back if requested
        if write_back:
            mode_msg = " (preserving existing bad flags)" if preserve_bad_flags else " (OVERWRITING ALL FLAGS)"
            print(f"Writing predictions back to source files{mode_msg}...")
            write_predictions(
                pred_df[['_source_file', '_raw_ts', target, prob_col]],
                target,
                preserve_bad_flags=preserve_bad_flags
            )
        
        results[target] = pred_df
    
    print("\n✓ Prediction complete!")
    return results


# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(
        description='Predict quality flags using saved models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on a single file (DEFAULT: preserves existing bad flags)
  python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv
  
  # Predict on multiple files
  python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv data/STW_2025/STW_2025-08_QC.csv
  
  # Predict on a date range (preserves existing bad flags)
  python predict_with_saved_model.py --start "2025-07-01" --end "2025-08-31"
  
  # Overwrite ALL flags (use with caution - deletes manual QC)
  python predict_with_saved_model.py --start "2025-07-01" --end "2025-08-31" --overwrite-all-flags
  
  # Only predict specific targets
  python predict_with_saved_model.py --start "2025-07-01" --end "2025-08-31" --targets Flag_GHI Flag_DNI
  
  # Predict without writing back (dry run)
  python predict_with_saved_model.py --file data/STW_2025/STW_2025-07_QC.csv --no-write

Note: By DEFAULT, existing bad flags (99) are preserved to protect manual QC.
      Use --overwrite-all-flags to replace all flags with model predictions.
        """
    )
    
    parser.add_argument('--file', nargs='+', help='Specific file(s) to predict on')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD) for date range prediction')
    parser.add_argument('--end', help='End date (YYYY-MM-DD) for date range prediction')
    parser.add_argument('--targets', nargs='+', default=['Flag_GHI', 'Flag_DNI', 'Flag_DHI'],
                        help='Target columns to predict (default: Flag_GHI Flag_DNI Flag_DHI)')
    parser.add_argument('--no-write', action='store_true',
                        help='Do not write predictions back to files (dry run)')
    parser.add_argument('--overwrite-all-flags', action='store_true',
                        help='Overwrite ALL flags with model predictions, including existing bad flags (99). '
                             'By DEFAULT, existing bad flags are preserved to protect manual QC. '
                             'Use this flag only when you want to completely replace all flags.')
    
    args = parser.parse_args()
    
    # Determine which files to process
    file_paths = []
    
    if args.file:
        # Use specified files
        file_paths = args.file
    elif args.start and args.end:
        # Find files in date range
        all_files = sorted(glob.glob(os.path.join(DATA_FOLDER, '**', '*.csv'), recursive=True))
        
        for f in all_files:
            # Extract date from filename (assumes format like STW_2025-07_QC.csv)
            try:
                basename = os.path.basename(f)
                # Parse dates from filenames
                if '_' in basename:
                    date_part = basename.split('_')[1].split('.')[0]  # e.g., "2025-07"
                    if date_part >= args.start[:7] and date_part <= args.end[:7]:
                        file_paths.append(f)
            except:
                pass
    else:
        # Default: process all files
        file_paths = sorted(glob.glob(os.path.join(DATA_FOLDER, '**', '*.csv'), recursive=True))
    
    if not file_paths:
        print("No files found to process!")
        print("Usage: python predict_with_saved_model.py --file <path> or --start <date> --end <date>")
        return
    
    # Run prediction
    predict_with_model(
        file_paths=file_paths,
        targets=args.targets,
        write_back=not args.no_write,
        preserve_bad_flags=not args.overwrite_all_flags  # Default True, False if --overwrite-all-flags
    )


if __name__ == '__main__':
    main()
