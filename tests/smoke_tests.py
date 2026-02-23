"""
Minimal smoke tests for the core pipeline.

This script is intentionally lightweight and uses a small sample of data.
"""

import glob
import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import SITE_CONFIG
from solar_features import add_features
from solar_model import SolarHybridModel


HEADER_ROWS_SKIP = 43
TS_COL = 'YYYY-MM-DD--HH:MM:SS'


def find_sample_file(min_rows: int = 500):
    candidates = glob.glob(os.path.join('data', '**', '*.csv'), recursive=True)
    for path in candidates:
        try:
            df = pd.read_csv(path, skiprows=HEADER_ROWS_SKIP, nrows=min_rows)
            df.columns = [c.strip() for c in df.columns]
            flag_cols = [c for c in df.columns if c.startswith('Flag_')]
            if not flag_cols:
                continue
            flag_col = flag_cols[0]
            labeled = df[df[flag_col].notna()]
            if len(labeled) < 50:
                continue
            y_labels = np.where(labeled[flag_col] == 99, 0, 1).astype(int)
            if len(np.unique(y_labels)) < 2:
                continue
            return path
        except Exception:
            continue
    return None


def main():
    sample_file = find_sample_file()
    if not sample_file:
        print("No CSV files found under data/. Smoke tests skipped.")
        return

    print(f"Using sample file: {sample_file}")

    df = pd.read_csv(sample_file, skiprows=HEADER_ROWS_SKIP, nrows=500)
    df.columns = [c.strip() for c in df.columns]

    df = add_features(df, SITE_CONFIG)
    df['Timestamp_dt'] = pd.to_datetime(df['Timestamp_dt'], errors='coerce')
    df.index = df['Timestamp_dt']

    # Pick a target flag column if available
    target_candidates = [c for c in df.columns if c.startswith('Flag_')]
    if not target_candidates:
        print("No Flag_* columns found. Smoke tests skipped.")
        return

    target_col = target_candidates[0]
    df_labeled = df[df[target_col].notna()].copy()

    if len(df_labeled) < 50:
        print("Not enough labeled data for a quick training pass. Smoke tests skipped.")
        return

    y_labels = np.where(df_labeled[target_col] == 99, 0, 1).astype(int)
    if len(np.unique(y_labels)) < 2:
        print("Only one class present in sample labels. Smoke tests skipped.")
        return

    # Train a lightweight model (Dense, 1 epoch)
    model = SolarHybridModel(use_rnn=False)
    model.fit(
        df_labeled,
        target_col=target_col,
        epochs=1,
        batch_size=32,
        upsample_min_bad=10,
        synthetic_frac=0.0,
        eval_split=0.2
    )

    flags, probs = model.predict(df_labeled.head(100), target_col, do_return_probs=True)
    print(f"Predictions: {len(flags)} rows")
    print(f"P(GOOD) mean: {float(np.mean(probs)):.4f}")


if __name__ == '__main__':
    main()
