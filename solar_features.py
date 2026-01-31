"""
solar_features.py
=================

Feature engineering and deterministic-quality-control utilities for
solar irradiance time-series data.

This file is intended to convert raw CSV rows into a stable set of numerical features used by
the supervised / unsupervised models.

Design goals and constraints:
- Backwards compatible with different CSV schemas (it tolerates missing cols)
- Clear-sky features are optional (pvlib used when available)
- Deterministic QC (BSRN-like checks) are produced as FEATURES only,
  not used as final labels automatically.
- Synthetic anomaly injection is available as an explicit training-time
  augmentation step (off by default).
- All outputs are numeric, NaN-free, and safe to feed into ML pipelines.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict

# Optional dependency. If pvlib is not installed, clearsky/solarpos features
# will be filled with safe defaults (zeros or sensible constants).
try:
    import pvlib  # pvlib provides clearsky models and solarposition
    _HAS_PVLIB = True
except Exception:
    pvlib = None
    _HAS_PVLIB = False


# -----------------------------------------------------------------------------
# Helper: sliding-window correlation-based anomaly feature
# -----------------------------------------------------------------------------
def correlation_feature(df: pd.DataFrame, col: str, span: int = 5) -> np.ndarray:
    """
    Compute a sliding-window correlation-difference feature.

    This feature attempts to highlight single-sample or short-window anomalies
    while being robust to smooth diurnal trends.

    For index i:
       - consider window [i-span, i+span]
       - corr_full = correlation(signal, time) across the full window
       - corr_sur  = correlation(signal, time) across the window excluding center
       - output = corr_full - corr_sur

    If not enough samples or constant signals, returns 0 for that index.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column `col`.
    col : str
        Column name to compute the feature on.
    span : int
        Half-window size (total window length = 2*span + 1).

    Returns
    -------
    np.ndarray
        Array of length len(df) with the correlation-difference values.
    """
    if col not in df.columns:
        return np.zeros(len(df), dtype=float)

    x = df[col].fillna(0.0).to_numpy(dtype=float)
    n = len(x)
    out = np.zeros(n, dtype=float)

    for i in range(n):
        left = max(0, i - span)
        right = min(n, i + span + 1)
        y = x[left:right]
        t = np.arange(left - i, right - i, dtype=float)

        if len(y) < 3:
            out[i] = 0.0
            continue

        y_d = y - y.mean()
        t_d = t - t.mean()
        var_y = float(np.dot(y_d, y_d))
        var_t = float(np.dot(t_d, t_d))
        corr_full = 0.0
        if var_y > 1e-9 and var_t > 1e-9:
            corr_full = float(np.dot(y_d, t_d) / np.sqrt(var_y * var_t))

        mask = t != 0
        corr_sur = 0.0
        if mask.sum() >= 2:
            y2 = y[mask]
            t2 = t[mask]
            y2_d = y2 - y2.mean()
            t2_d = t2 - t2.mean()
            var_y2 = float(np.dot(y2_d, y2_d))
            var_t2 = float(np.dot(t2_d, t2_d))
            if var_y2 > 1e-9 and var_t2 > 1e-9:
                corr_sur = float(np.dot(y2_d, t2_d) / np.sqrt(var_y2 * var_t2))

        out[i] = corr_full - corr_sur

    return out


# -----------------------------------------------------------------------------
# Deterministic QC (feature)
# -----------------------------------------------------------------------------
def bs_rn_qc(df: pd.DataFrame,
             ghi_col: str = 'GHI',
             dni_col: str = 'DNI',
             dhi_col: str = 'DHI',
             zenith_col: str = 'SZA') -> np.ndarray:
    """
    Minimal BSRN-style deterministic tests used as a feature.

    Important: this function does NOT claim to replace manual QC. It returns
    a binary indicator (0 = pass, 1 = flagged as physically suspect) which is
    useful as a feature for the ML model and for quick filtering.

    Tests included (conservative):
      - nighttime sanity (sun below horizon => irradiance should be near 0)
      - physically possible limits (simple bounds)
      - diffuse fraction checks (DHI/GHI ratio)
      - three-component closure (GHI ≈ DHI + DNI * cos(zenith))

    Returns
    -------
    np.ndarray (int)
        1 indicates a physically suspicious measurement.
    """
    n = len(df)
    flags = np.zeros(n, dtype=int)

    ghi = df.get(ghi_col, pd.Series(0.0)).fillna(0.0).to_numpy(dtype=float)
    dni = df.get(dni_col, pd.Series(0.0)).fillna(0.0).to_numpy(dtype=float)
    dhi = df.get(dhi_col, pd.Series(0.0)).fillna(0.0).to_numpy(dtype=float)
    zen = df.get(zenith_col, pd.Series(90.0)).fillna(90.0).to_numpy(dtype=float)

    # Night sanity: if sun below horizon (zenith >= 90), expect near-zero GHI
    night_mask = zen >= 90.0
    flags[np.where(night_mask & (np.abs(ghi) > 5.0))[0]] = 1

    # Conservative absolute bounds
    flags[np.where((ghi < -1) | (ghi > 1400))[0]] = 1
    flags[np.where((dni < -10) | (dni > 1400))[0]] = 1
    flags[np.where((dhi < -10) | (dhi > 1400))[0]] = 1

    # Diffuse fraction checks (avoid division by zero)
    valid = (ghi > 50.0) & (zen < 90.0)
    ratio = np.zeros_like(ghi)
    ratio[valid] = dhi[valid] / ghi[valid]
    flags[np.where(valid & (ratio > 1.1))[0]] = 1

    # Three-component closure
    zen_rad = np.deg2rad(np.clip(zen, 0.0, 89.999))
    closure = dhi + dni * np.cos(zen_rad)
    rel_err = np.abs(ghi - closure) / (ghi + 1e-6)
    flags[np.where((ghi > 50.0) & (rel_err > 0.30))[0]] = 1

    return flags


# -----------------------------------------------------------------------------
# Clear-sky features & solar geometry (pvlib - robust)
# -----------------------------------------------------------------------------
def add_clearsky_features(df: pd.DataFrame,
                          latitude: float,
                          longitude: float,
                          altitude: float = 0.0,
                          tz: str = 'UTC') -> pd.DataFrame:
    """
    Add clearsky estimates (GHI_Clear, DNI_Clear, DHI_Clear) and solar geometry
    (zenith, elevation) using pvlib. This function is defensive:
      - If pvlib is absent, fills safe defaults (zeros / 90 degrees)
      - Sanitizes timestamps to avoid pandas/pvlib dtype issues
      - Reindexes results back into original dataframe shape
    """

    # Provide safe defaults if pvlib is not available.
    if not _HAS_PVLIB:
        for c in ['GHI_Clear', 'DNI_Clear', 'DHI_Clear', 'CSI', 'elevation']:
            df[c] = 0.0
        return df

    # Convert timestamp column to a DatetimeIndex (coerce errors -> NaT)
    times = pd.to_datetime(df.get('Timestamp_dt', pd.Series(pd.NaT)), errors='coerce')
    valid_mask = times.notna()
    times_valid = times[valid_mask]

    # Ensure timezone-aware index for pvlib: localize or convert
    try:
        # times_valid is a Series; use .dt accessors
        if times_valid.dt.tz is None:
            times_valid = times_valid.dt.tz_localize(tz)
        else:
            times_valid = times_valid.dt.tz_convert(tz)
    except Exception:
        # Fallback: naive -> interpret as tz
        times_valid = pd.DatetimeIndex(times_valid).tz_localize(tz)

    times_valid = pd.DatetimeIndex(times_valid)

    # Use pvlib Location helper
    loc = pvlib.location.Location(latitude=latitude, longitude=longitude, tz=tz, altitude=altitude)

    # Compute clearsky and solar position for VALID times only
    clearsky = loc.get_clearsky(times_valid)  # columns: ghi, dni, dhi
    solarpos = pvlib.solarposition.get_solarposition(times_valid, latitude, longitude, altitude=altitude)

    # Insert defaults for all rows before filling valid indices
    df['GHI_Clear'] = 0.0
    df['DNI_Clear'] = 0.0
    df['DHI_Clear'] = 0.0
    df['elevation'] = 0.0

    # Fill only valid positions
    df.loc[valid_mask, 'GHI_Clear'] = clearsky['ghi'].values
    df.loc[valid_mask, 'DNI_Clear'] = clearsky['dni'].values
    df.loc[valid_mask, 'DHI_Clear'] = clearsky['dhi'].values
    df.loc[valid_mask, 'elevation'] = solarpos['elevation'].values

    # Clear-sky index (CSI) robustly calculated and clipped
    denom = df['GHI_Clear'].replace(0.0, np.nan)
    df['CSI'] = (df.get('GHI', 0.0) / denom).fillna(0.0).clip(0.0, 3.0)

    return df


# -----------------------------------------------------------------------------
# Synthetic anomaly injection used during training only (opt-in)
# -----------------------------------------------------------------------------
def inject_synthetic_anomalies(df: pd.DataFrame, frac: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic BAD-like samples by perturbing selected GOOD rows.

    This is intentionally conservative: it does not change label columns,
    it only perturbs measurement columns (GHI/DNI/DHI) to create plausible
    failure modes (stuck-zero, spike, clipping, drift). Use only during training.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe (copy will be returned).
    frac : float
        Fraction of rows to perturb (0 disables).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        DataFrame with same shape; a fraction `frac` of rows are perturbed.
    """
    if frac <= 0.0:
        return df

    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    n_sel = max(1, int(frac * n))
    idx = rng.choice(n, size=n_sel, replace=False)

    for i in idx:
        mode = rng.integers(0, 4)
        # stuck-zero: set all irradiances to zero
        if mode == 0:
            for c in ['GHI', 'DNI', 'DHI']:
                if c in out.columns:
                    out.at[out.index[i], c] = 0.0
        # clipping: cap at clearsky value * factor if available
        elif mode == 1:
            if 'GHI' in out.columns and 'GHI_Clear' in out.columns:
                out.at[out.index[i], 'GHI'] = 0.95 * out.at[out.index[i], 'GHI_Clear']
        # spike: add a large positive spike to one channel
        elif mode == 2:
            col = rng.choice(['GHI', 'DNI', 'DHI'])
            if col in out.columns:
                out.at[out.index[i], col] = out.at[out.index[i], col] + float(rng.uniform(300, 800))
        # drift: small offset proportional to zenith
        else:
            if 'GHI' in out.columns and 'SZA' in out.columns:
                out.at[out.index[i], 'GHI'] = out.at[out.index[i], 'GHI'] + (90.0 - out.at[out.index[i], 'SZA']) * 0.1

    return out


# -----------------------------------------------------------------------------
# Master feature-building entry point
# -----------------------------------------------------------------------------
def add_features(df: pd.DataFrame, site_cfg: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convert raw CSV-like dataframe into a feature-rich numeric dataframe.

    Responsibilities:
      - Ensure a Timestamp_dt column exists (coerce parse failures)
      - Generate cyclical time features (hour sine/cos, day-of-year sine/cos)
      - Optionally add clearsky and solar geometry features via pvlib
      - Add correlation features for GHI/DNI/DHI
      - Add deterministic QC indicator (QC_PhysicalFail)
      - Guarantee no NaNs/inf in final output

    Notes:
      - This function is intentionally idempotent and side-effect-free with
        respect to original measurement columns (it returns a new DataFrame).
      - If your CSVs have differently-named timestamp columns, adapt the
        TS_COL constant in run_learning_cycle (default 'YYYY-MM-DD--HH:MM:SS').
    """
    temp = df.copy()

    # Timestamp normalization: if the standard TS column exists, use it; else
    # assume first column is a timestamp (legacy CSVs).
    TS_COL = 'YYYY-MM-DD--HH:MM:SS'
    if TS_COL in temp.columns:
        temp['Timestamp_dt'] = pd.to_datetime(temp[TS_COL], format='%Y-%m-%d--%H:%M:%S', errors='coerce')
    elif 'Timestamp' in temp.columns:
        temp['Timestamp_dt'] = pd.to_datetime(temp['Timestamp'], errors='coerce')
    else:
        # fallback: parse the first column as datetime
        temp['Timestamp_dt'] = pd.to_datetime(temp.iloc[:, 0], errors='coerce')

    # Numeric epoch time (seconds) for numeric models - keep safe if parse failed
    try:
        temp['Timestamp_Num'] = temp['Timestamp_dt'].astype('int64') / 1e9
    except Exception:
        # older pandas may fail on NaT; fallback to zeros
        temp['Timestamp_Num'] = 0.0

    # Cyclical/time-of-day features
    hour = temp['Timestamp_dt'].dt.hour.fillna(0).astype(float)
    minute = temp['Timestamp_dt'].dt.minute.fillna(0).astype(float)
    doy = temp['Timestamp_dt'].dt.dayofyear.fillna(0).astype(float)

    temp['hour_frac'] = hour + minute / 60.0
    temp['hour_sin'] = np.sin(2.0 * np.pi * temp['hour_frac'] / 24.0)
    temp['hour_cos'] = np.cos(2.0 * np.pi * temp['hour_frac'] / 24.0)
    temp['doy_sin'] = np.sin(2.0 * np.pi * doy / 365.25)
    temp['doy_cos'] = np.cos(2.0 * np.pi * doy / 365.25)

    # Clear-sky features (optional)
    if site_cfg is not None:
        temp = add_clearsky_features(temp,
                                     latitude=site_cfg['latitude'],
                                     longitude=site_cfg['longitude'],
                                     altitude=site_cfg.get('altitude', 0.0),
                                     tz=site_cfg.get('timezone', 'UTC'))

    # Correlation features (local)
    for c in ['GHI', 'DNI', 'DHI']:
        if c in temp.columns:
            temp[f'CorrFeat_{c}'] = correlation_feature(temp, c, span=5)

    # Deterministic QC as a feature
    temp['QC_PhysicalFail'] = bs_rn_qc(temp) if 'SZA' in temp.columns else 0

    # Final safety cleanup: no NaN/inf remain
    temp = temp.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return temp
