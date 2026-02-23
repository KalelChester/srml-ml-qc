"""
Shared I/O helpers for QC CSV files.
"""

import pandas as pd


def load_qc_csvs(file_paths, header_rows_skip=43, ts_col='YYYY-MM-DD--HH:MM:SS'):
    """
    Load and concatenate QC CSV files with source tracking.

    Adds:
    - _source_file: original file path
    - _raw_ts: raw timestamp string for exact writeback matching
    """
    frames = []
    for path in file_paths:
        try:
            df = pd.read_csv(path, skiprows=header_rows_skip)
            df.columns = [c.strip() for c in df.columns]
            if 'Data_Begins_Next_Row' in df.columns:
                df = df.drop(columns=['Data_Begins_Next_Row'])
            df['_source_file'] = path
            if ts_col in df.columns:
                df['_raw_ts'] = df[ts_col].astype(str).str.strip()
            else:
                df['_raw_ts'] = df.iloc[:, 0].astype(str).str.strip()
            frames.append(df)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
