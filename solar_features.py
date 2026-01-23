import pandas as pd
import numpy as np

def correlation_feature(df, col, span=5):
    """
    Calculates the sliding window correlation difference.
    Returns 0.0 for any window that cannot be calculated (too short/constant).
    """
    # Safety: if column missing, return 0s
    if col not in df.columns:
        return np.zeros(len(df))

    x = df[col].fillna(0).to_numpy(dtype=float)
    n = len(x)
    out = np.zeros(n)

    for i in range(n):
        # Adaptive window
        left = max(0, i - span)
        right = min(n, i + span + 1)
        y = x[left:right]
        t = np.arange(left - i, right - i, dtype=float)

        if len(y) < 3:
            out[i] = 0.0
            continue

        # Full window correlation
        y_mean = y.mean()
        t_mean = t.mean()
        y_d = y - y_mean
        t_d = t - t_mean
        var_y = np.dot(y_d, y_d)
        var_t = np.dot(t_d, t_d)

        corr_full = 0.0
        if var_y > 1e-9 and var_t > 1e-9: # Avoid float precision zero
            corr_full = np.dot(y_d, t_d) / np.sqrt(var_y * var_t)

        # Surrounding-only (exclude center)
        mask = t != 0
        y2 = y[mask]
        t2 = t[mask]

        corr_sur = 0.0
        if len(y2) >= 2:
            y2_d = y2 - y2.mean()
            t2_d = t2 - t2.mean()
            var_y2 = np.dot(y2_d, y2_d)
            var_t2 = np.dot(t2_d, t2_d)

            if var_y2 > 1e-9 and var_t2 > 1e-9:
                corr_sur = np.dot(y2_d, t2_d) / np.sqrt(var_y2 * var_t2)

        out[i] = corr_full - corr_sur

    return out

def add_features(df):
    """
    Applies feature transformations for GHI, DNI, and DHI.
    """
    temp_df = df.copy()
    
    # 1. Timestamp Processing
    # We assume 'YYYY-MM-DD--HH:MM:SS' exists or has been renamed to 'Timestamp'
    ts_col = 'YYYY-MM-DD--HH:MM:SS'
    if ts_col not in temp_df.columns and 'Timestamp' in temp_df.columns:
        # If already renamed, use 'Timestamp'
        ts_series = pd.to_datetime(temp_df['Timestamp'])
    elif ts_col in temp_df.columns:
        ts_series = pd.to_datetime(temp_df[ts_col], format='%Y-%m-%d--%H:%M:%S', errors='coerce')
    else:
        # Fallback: try to find the time column
        col_0 = temp_df.columns[0]
        ts_series = pd.to_datetime(temp_df[col_0], errors='coerce')

    temp_df['Timestamp_dt'] = ts_series
    
    # Numeric timestamp for model
    temp_df['Timestamp_Num'] = temp_df['Timestamp_dt'].astype('int64') / 1e9

    # 2. Cyclical Time Features
    # Fill NaT with a safe default (forward fill or 0) to prevent crashes
    hour = temp_df['Timestamp_dt'].dt.hour.fillna(0)
    minute = temp_df['Timestamp_dt'].dt.minute.fillna(0)
    doy = temp_df['Timestamp_dt'].dt.dayofyear.fillna(0)

    temp_df['hour_frac'] = hour + minute / 60.0
    temp_df['hour_sin'] = np.sin(2 * np.pi * temp_df['hour_frac'] / 24.0)
    temp_df['hour_cos'] = np.cos(2 * np.pi * temp_df['hour_frac'] / 24.0)
    temp_df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    temp_df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)
    
    # 3. GHI Features
    if 'GHI' in temp_df.columns:
        temp_df['CorrFeat_GHI'] = correlation_feature(temp_df, 'GHI')
    
    if 'GHI_Calc' in temp_df.columns:
        temp_df['CorrFeat_GHI_Calc'] = correlation_feature(temp_df, 'GHI_Calc')
        # GHI Fraction (Safe Division)
        denom = temp_df['GHI_Calc'].replace(0, np.nan).fillna(1e-3)
        temp_df['GHI_Frac'] = temp_df['GHI'] / denom

    # 4. DNI Features
    if 'DNI' in temp_df.columns:
        temp_df['CorrFeat_DNI'] = correlation_feature(temp_df, 'DNI')

    # 5. DHI Features
    if 'DHI' in temp_df.columns:
        temp_df['CorrFeat_DHI'] = correlation_feature(temp_df, 'DHI')
        
    return temp_df