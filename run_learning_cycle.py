"""
run_learning_cycle.py
=====================

Orchestration with confidence-gated writeback and probability columns.
- Prediction probabilities are written back as Flag_<target>_prob with 1.0 = certain GOOD.
- Auto-write: only commit predicted flags when model confidence is high (>= HIGH_THRESH)
  or low (<= LOW_THRESH). Others are left untouched and recorded in a review CSV.
- Reading is robust to presence or absence of existing Flag_<target>_prob columns.
"""

import os
import glob
import pandas as pd
import numpy as np

from solar_features import add_features
from solar_model import SolarHybridModel


# ---------------- CONFIG ----------------
DATA_FOLDER = 'data'
MODEL_FOLDER = 'models'  # Folder to save trained models
HEADER_ROWS_SKIP = 43  # Number of rows to skip when reading (skips rows 0-42, uses row 43 as column names)
HEADER_ROWS_PRESERVE = 44  # Number of rows to preserve when writing back (rows 0-43 including column names)
TS_COL = 'YYYY-MM-DD--HH:MM:SS'

SITE_CONFIG = {
    'latitude': 47.654,     # <-- SET ME
    'longitude': -122.309,  # <-- SET ME
    'altitude': 70,   # meters (optional)
    'timezone': 'Etc/GMT+8'
}

# confidence thresholds for auto-write (tunable)
HIGH_THRESH = 0.51  # prob >= HIGH_THRESH => auto write GOOD
LOW_THRESH = 0.49   # prob <= LOW_THRESH  => auto write BAD

REVIEW_OUT = 'review_requests.csv'  # appended with rows needing manual review


# ---------------- I/O helpers ----------------
def load_csvs(file_paths):
    frames = []
    for f in file_paths:
        try:
            df = pd.read_csv(f, skiprows=HEADER_ROWS_SKIP)
            df.columns = [c.strip() for c in df.columns]
            # Drop the empty Data_Begins_Next_Row column if present
            if 'Data_Begins_Next_Row' in df.columns:
                df = df.drop(columns=['Data_Begins_Next_Row'])
            df['_source_file'] = f
            # Make sure timestamp column exists and raw ts col
            if TS_COL in df.columns:
                df['_raw_ts'] = df[TS_COL].astype(str).str.strip()
            else:
                # fallback: first column
                df['_raw_ts'] = df.iloc[:, 0].astype(str).str.strip()
            frames.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if len(frames) == 0:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def write_back(df_preds: pd.DataFrame, target_col: str):
    """
    Write predicted flags back to source CSVs based on confidence gating.

    For each grouped file, update only timestamps where auto-write decisions
    were made. Predictions with intermediate confidence are saved to REVIEW_OUT
    for manual labeling and are NOT written back to CSV.

    df_preds is expected to contain columns:
      - _source_file, _raw_ts, target_col, target_prob (Flag_<target>_prob)
      - optional 'AutoWrite' boolean column indicating whether to commit
    """
    review_rows = []

    for path, g in df_preds.groupby('_source_file'):
        try:
            orig = pd.read_csv(path, skiprows=HEADER_ROWS_SKIP)
            orig.columns = [c.strip() for c in orig.columns]
            # Drop the empty trailing column if it exists (from trailing commas in CSV)
            if 'Data_Begins_Next_Row' in orig.columns:
                orig = orig.drop(columns=['Data_Begins_Next_Row'])
        except Exception as e:
            print(f"Failed to open {path} for writing: {e}")
            continue

        # Build mapping only for rows where AutoWrite == True
        if 'AutoWrite' in g.columns:
            to_write = g[g['AutoWrite'] == True]
        else:
            # fallback: write where prob is extreme
            prob_col = f"{target_col}_prob"
            if prob_col in g.columns:
                pw = g[(g[prob_col] >= HIGH_THRESH) | (g[prob_col] <= LOW_THRESH)]
                to_write = pw.copy()
            else:
                to_write = g.copy()

        write_map = to_write.set_index('_raw_ts')[target_col].to_dict()

        # Update original data only for timestamps in write_map
        def upd(row):
            ts = str(row[TS_COL]).strip() if TS_COL in row.index else str(row.iloc[0]).strip()
            if ts in write_map:
                return write_map[ts]
            return row.get(target_col, row.get(target_col, 0))

        orig[target_col] = orig.apply(upd, axis=1)

        # Ensure we preserve any existing prob column; else add it as NaNs
        prob_col = f"{target_col}_prob"
        if prob_col in g.columns:
            prob_map = g.set_index('_raw_ts')[prob_col].to_dict()
            def upd_prob(row):
                ts = str(row[TS_COL]).strip() if TS_COL in row.index else str(row.iloc[0]).strip()
                return prob_map.get(ts, row.get(prob_col, np.nan))
            orig[prob_col] = orig.apply(upd_prob, axis=1)
        else:
            # add NA prob column
            orig[prob_col] = np.nan

        # Drop the empty Data_Begins_Next_Row column if present (caused by trailing comma in CSV)
        if 'Data_Begins_Next_Row' in orig.columns:
            orig = orig.drop(columns=['Data_Begins_Next_Row'])

        # Write the file with original header preserved
        try:
            with open(path, 'r') as f:
                header_lines = [next(f) for _ in range(HEADER_ROWS_PRESERVE)]
            with open(path, 'w', newline='') as f:
                f.writelines(header_lines)
                orig.to_csv(f, index=False, header=False)  # header=False since we already wrote it
        except Exception as e:
            print(f"Failed writing back to {path}: {e}")

        # Record review rows (those not auto-written within this file)
        not_written = g[~g['_raw_ts'].isin(to_write['_raw_ts'])]
        if not not_written.empty:
            review_rows.append(not_written.assign(_source_file_write=path))

    # Append reviews to REVIEW_OUT
    if len(review_rows) > 0:
        df_reviews = pd.concat(review_rows, ignore_index=True)
        if os.path.exists(REVIEW_OUT):
            df_reviews.to_csv(REVIEW_OUT, mode='a', header=False, index=False)
        else:
            df_reviews.to_csv(REVIEW_OUT, index=False)


# ---------------- Main cycle ----------------------------------------------

def run_cycle(train_start: str, train_end: str, pred_start: str, pred_end: str, targets: list[str]):
    files = sorted(glob.glob(os.path.join(DATA_FOLDER, '**', '*.csv'), recursive=True))
    raw = load_csvs(files)
    if raw.empty:
        print('No files found')
        return

    full = add_features(raw, SITE_CONFIG)
    # Ensure Timestamp_dt is datetime type before setting as index
    full['Timestamp_dt'] = pd.to_datetime(full['Timestamp_dt'], errors='coerce')
    full.index = full['Timestamp_dt']
    
    train_mask = (full.index >= pd.to_datetime(train_start)) & (full.index <= pd.to_datetime(train_end))
    pred_mask = (full.index >= pd.to_datetime(pred_start)) & (full.index <= pd.to_datetime(pred_end))

    train_df = full[train_mask].copy()
    pred_df = full[pred_mask].copy()

    if train_df.empty or pred_df.empty:
        raise RuntimeError('Training or prediction slice empty')

    for t in targets:
        print(f"=== Training and predicting {t} ===")
        model = SolarHybridModel()
        # fit model
        model.fit(train_df, target_col=t)

        # Save the trained model
        model_filename = os.path.join(MODEL_FOLDER, f'model_{t}.pkl')
        model.save_model(model_filename)

        # If the unsupervised detector exists, score pred_df and add IF_Score
        if getattr(model, 'if_det', None) is not None:
            try:
                pred_df['IF_Score'] = model.if_det.decision_function(pred_df[model.common_features].fillna(0.0))
            except Exception:
                pred_df['IF_Score'] = 0.0

        # Predict with probabilities
        flags, probs = model.predict(pred_df, t, do_return_probs=True)
        prob_col = f"{t}_prob"
        pred_df[t] = flags
        pred_df[prob_col] = probs  # 1.0 == GOOD

        # Decide auto-write by thresholding
        pred_df['AutoWrite'] = False
        pred_df.loc[pred_df[prob_col] >= HIGH_THRESH, 'AutoWrite'] = True
        pred_df.loc[pred_df[prob_col] <= LOW_THRESH, 'AutoWrite'] = True

        # Ensure _source_file and _raw_ts exist in pred_df for mapping
        if '_source_file' not in pred_df.columns or '_raw_ts' not in pred_df.columns:
            raise RuntimeError('Pred dataframe missing source info')

        # Write back committed flags & probabilities
        write_back(pred_df[['_source_file', '_raw_ts', t, prob_col, 'AutoWrite']], t)


# ----------------- Entry point -------------------------------------------
if __name__ == '__main__':
    TARGETS = ['Flag_GHI', 'Flag_DNI', 'Flag_DHI']

    TRAIN_START = '2025-01-01 00:00:00'
    TRAIN_END = '2025-04-30 23:59:59'
    PRED_START = '2025-05-01 00:00:00'
    PRED_END = '2025-06-30 23:59:59'

    run_cycle(TRAIN_START, TRAIN_END, PRED_START, PRED_END, TARGETS)