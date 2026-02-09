"""
solar_features.py
=================

Feature Engineering for Solar Irradiance Quality Control

This module converts raw solar measurement data into a comprehensive set of
engineered features suitable for machine learning models. It handles timestamp
parsing, solar geometry calculations, clear-sky modeling, and anomaly detection
features.

CRITICAL: Timezone Handling Philosophy
---------------------------------------
**Timestamps are assumed to be ALREADY CORRECT in local time.**

The timezone parameter in site configuration is used ONLY to inform pvlib what
timezone the data represents, so it can correctly calculate:
- Solar position (zenith, azimuth, elevation angles)
- Clear-sky irradiance estimates
- Local solar noon timing

The timezone does NOT adjust or convert your timestamp values. Think of it as
metadata: "These times are in Pacific time" or "These times are in UTC+8" so
that pvlib can accurately compute where the sun is in the sky.

Example:
    Your data has timestamp "2025-06-15 12:00:00" which represents local noon
    at your site. When you specify timezone='Etc/GMT+8', you're telling pvlib:
    "This 12:00:00 is in GMT+8 timezone, please calculate sun position for that
    local time." The timestamp value itself (12:00:00) does NOT change.

Main Components
---------------
1. **Timestamp Processing**
   - Parse various timestamp formats
   - Extract time-of-day and seasonal features
   - Create cyclical encodings (hour_sin/cos, doy_sin/cos)

2. **Solar Geometry** (via pvlib)
   - Calculate solar position (zenith, azimuth, elevation)
   - Estimate clear-sky irradiance (Ineichen model)
   - Compute clear-sky index (CSI)

3. **Anomaly Features**
   - Sliding-window correlation features
   - Highlight temporal anomalies while robust to smooth trends

4. **Deterministic QC**
   - BSRN-inspired physical bounds checks
   - Three-component closure test
   - Diffuse fraction validation

5. **Data Augmentation** (Training Only)
   - Synthetic anomaly injection
   - Conservative perturbations for rare failure modes

Design Principles
-----------------
- Backward compatible: Handles missing columns gracefully
- Defensive: Provides safe defaults when pvlib unavailable
- Clean output: No NaN or inf values in returned features
- Side-effect free: Returns new DataFrame, doesn't modify input
- Optional features: Clear-sky calculations only when site_cfg provided

Dependencies
------------
- pandas, numpy: Data manipulation
- pvlib (optional): Solar position and clear-sky modeling
  If not installed, functions gracefully with defaults

Usage
-----
    from solar_features import add_features
    
    site_cfg = {
        'latitude': 47.654,      # Decimal degrees
        'longitude': -122.309,   # Decimal degrees
        'altitude': 70,          # Meters above sea level
        'timezone': 'Etc/GMT+8'  # Timezone for solar calculations (not conversion!)
    }
    
    df_raw = pd.read_csv('solar_data.csv')
    df_featured = add_features(df_raw, site_cfg)

Author: Solar QC Team
Last Updated: February 2026
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

    x = df[col].astype(float, errors='ignore').fillna(0.0).to_numpy(dtype=float)
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

    ghi = df.get(ghi_col, pd.Series(0.0, dtype=float)).fillna(0.0).to_numpy(dtype=float)
    dni = df.get(dni_col, pd.Series(0.0, dtype=float)).fillna(0.0).to_numpy(dtype=float)
    dhi = df.get(dhi_col, pd.Series(0.0, dtype=float)).fillna(0.0).to_numpy(dtype=float)
    zen = df.get(zenith_col, pd.Series(90.0, dtype=float)).fillna(90.0).to_numpy(dtype=float)

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
    (zenith, elevation) using pvlib.
    
    IMPORTANT: Timezone Handling
    -----------------------------
    The 'tz' parameter is used ONLY for:
    - pvlib's solar position calculations (requires timezone-aware timestamps)
    - Clear-sky model calculations (which depend on local solar time)
    
    The timestamps in the input data are assumed to already be in the CORRECT
    local time for the site. This function does NOT adjust or convert the actual
    time values - it only adds timezone awareness so pvlib can calculate solar
    geometry correctly.
    
    Think of it as: "These times are already correct, just tell pvlib what
    timezone they represent so it can calculate sun position accurately."
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Timestamp_dt' column (naive datetime, in local time)
    latitude : float
        Site latitude in decimal degrees
    longitude : float
        Site longitude in decimal degrees
    altitude : float, default=0.0
        Site altitude in meters above sea level
    tz : str, default='UTC'
        Timezone string (e.g., 'Etc/GMT+8', 'America/Los_Angeles')
        Used only to inform pvlib of the timezone for solar calculations
        Does NOT change the actual time values
    
    Returns
    -------
    pd.DataFrame
        Original dataframe with added columns:
        - GHI_Clear, DNI_Clear, DHI_Clear: Clear-sky irradiance estimates
        - elevation: Solar elevation angle
        - CSI: Clear-sky index (GHI / GHI_Clear)
    
    Notes
    -----
    - If pvlib is not available, fills safe defaults (zeros)
    - Handles both timezone-naive and timezone-aware input gracefully
    - If timestamps are already timezone-aware, preserves them as-is
    - If timestamps are naive, assumes they represent local time and adds tz info
    """

    # Provide safe defaults if pvlib is not available
    if not _HAS_PVLIB:
        for c in ['GHI_Clear', 'DNI_Clear', 'DHI_Clear', 'CSI', 'elevation']:
            df[c] = 0.0
        return df

    # Convert timestamp column to a DatetimeIndex (coerce errors -> NaT)
    times = pd.to_datetime(df.get('Timestamp_dt', pd.Series(pd.NaT)), errors='coerce')
    valid_mask = times.notna()
    times_valid = times[valid_mask]

    # Handle timezone awareness for pvlib
    # CRITICAL: We assume times are ALREADY in the correct local time!
    # We only add timezone info so pvlib knows what timezone to use for solar position
    try:
        if times_valid.dt.tz is None:
            # Times are naive - assume they represent local time in timezone 'tz'
            # This does NOT change the time values, just adds timezone metadata
            times_valid = times_valid.dt.tz_localize(tz, ambiguous='NaT', nonexistent='NaT')
        else:
            # Times already have timezone info - preserve them exactly as-is
            # Only convert if they're in a different timezone than expected
            if str(times_valid.dt.tz) != tz:
                # If somehow in wrong timezone, convert (but this shouldn't happen)
                times_valid = times_valid.dt.tz_convert(tz)
    except Exception:
        # Fallback: interpret naive times as being in timezone 'tz'
        times_valid = pd.DatetimeIndex(times_valid).tz_localize(tz, ambiguous='NaT', nonexistent='NaT')

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
    ghi_values = df.get('GHI', pd.Series(0.0, dtype=float))
    df['CSI'] = (ghi_values.astype(float, errors='ignore') / denom).fillna(0.0).clip(0.0, 3.0)

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

    This is the main entry point for feature engineering. It takes raw solar data
    (with timestamps and irradiance measurements) and produces a comprehensive set
    of features suitable for machine learning models.

    Timestamp Handling (IMPORTANT)
    -------------------------------
    Timestamps in the input data are assumed to be ALREADY in the correct local
    time for the measurement site. This function:
    - Parses timestamps to datetime objects
    - Does NOT adjust or convert the actual time values
    - Only adds timezone metadata when needed for pvlib calculations
    
    The timezone specified in site_cfg['timezone'] tells pvlib what timezone the
    data represents (so it can calculate sun position correctly), but does NOT
    change the timestamp values themselves.

    Processing Steps
    ----------------
    1. Parse timestamps (assumes local time, no conversion)
    2. Extract time-based features (hour, day-of-year, cyclical encodings)
    3. Calculate clear-sky irradiance and solar position (via pvlib)
    4. Compute correlation features for anomaly detection
    5. Run deterministic QC checks (BSRN-like rules)
    6. Clean up NaN/inf values
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with at minimum:
        - A timestamp column (YYYY-MM-DD--HH:MM:SS, Timestamp, or first column)
        - Irradiance columns: GHI, DNI, DHI (optional but recommended)
        - Temperature column (optional)
    
    site_cfg : dict, optional
        Site configuration dictionary with keys:
        - latitude : float (decimal degrees, required for clear-sky)
        - longitude : float (decimal degrees, required for clear-sky)
        - altitude : float (meters above sea level, default 0.0)
        - timezone : str (e.g., 'Etc/GMT+8', 'America/Los_Angeles')
                     Used ONLY for pvlib solar position calculations
                     Does NOT adjust timestamp values
        
        If None, skips clear-sky calculations
    
    Returns
    -------
    pd.DataFrame
        Enhanced dataframe with original columns plus:
        
        Time Features:
        - Timestamp_dt : datetime (parsed from input)
        - Timestamp_Num : float (Unix epoch seconds)
        - hour_sin, hour_cos : float (cyclical hour encoding)
        - doy_sin, doy_cos : float (cyclical day-of-year encoding)
        - hour_frac : float (hour with fractional minutes)
        
        Solar Features (if site_cfg provided):
        - GHI_Clear, DNI_Clear, DHI_Clear : float (clear-sky estimates)
        - elevation : float (solar elevation angle, degrees)
        - CSI : float (clear-sky index, GHI/GHI_Clear, clipped to [0,3])
        
        Anomaly Features:
        - CorrFeat_GHI, CorrFeat_DNI, CorrFeat_DHI : float (correlation-based)
        - QC_PhysicalFail : int (0=pass, 1=fail deterministic checks)
    
    Notes
    -----
    - Idempotent: safe to call multiple times (creates new DataFrame)
    - No side effects on input dataframe
    - All NaN and inf values are replaced with 0.0 in output
    - Missing columns filled with safe defaults
    - Timestamp parsing errors become NaT (Not-a-Time), handled gracefully
    
    Example
    -------
        >>> site_config = {
        ...     'latitude': 47.654,
        ...     'longitude': -122.309,
        ...     'altitude': 70,
        ...     'timezone': 'Etc/GMT+8'  # Tells pvlib the timezone, doesn't adjust times
        ... }
        >>> df_raw = pd.read_csv('solar_data.csv', skiprows=43)
        >>> df_featured = add_features(df_raw, site_config)
        >>> print(df_featured.columns)
        # Includes: Timestamp_dt, hour_sin, hour_cos, GHI_Clear, CSI, ...
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
    temp = temp.replace([np.inf, -np.inf], 0.0).astype(float, errors='ignore').fillna(0.0)

    return temp
