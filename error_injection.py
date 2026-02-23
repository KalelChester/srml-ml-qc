"""
error_injection.py
==================

Synthetic Error Injection System for Solar Radiation Data

This module injects realistic artificial errors into solar irradiance data
for testing and model training purposes. The system is highly modular,
allowing easy customization of error types, parameters, and output modes.

Architecture
------------
1. DataManager: Load data while preserving metadata
2. SolarGeometry: Handle SZA-based daytime filtering and sunset detection
3. ErrorFunctions: Six configurable error injection functions
4. ErrorInjectionEngine: Orchestrate error injection workflow
5. OutputHandler: Save or return results

Author: Solar QC Team
"""

from __future__ import annotations
import os
import sys
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional

# Optional pvlib for solar geometry
try:
    import pvlib
    _HAS_PVLIB = True
except ImportError:
    _HAS_PVLIB = False
    warnings.warn("pvlib not installed. SZA calculations may be limited.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

FEATURE_COLUMNS = ['GHI', 'DHI', 'DNI']

ERROR_INJECTION_CONFIG = {
    'error_probability': 1.0,  
    'error_count_range': (10, 30),  
    'daytime_bias': 0.85,  
    'sza_threshold': 85.0,  # Lowered slightly to avoid injecting deep into twilight
    
    'error_functions': {
        'reduce_features': {
            'reduction_percent': (1, 50),  
            'window_type': ['box', 'gaussian'],  
            'gaussian_sigma_minutes': 2,  
            'weight': 0.3 # Probability weight of this error being chosen
        },
        'copy_from_day': {
            'days_offset_range': (-30, -1),  
            'weight': 0.2
        },
        'end_of_day_frost': {
            'dip_percent': (10, 40),  
            'weight': 0.15 
        },
        'cleaning_event': {
            'dip_percent': (4, 8),  
            'duration_minutes': 3,  
            'weight': 0.15
        },
        'water_droplet': {
            'spike_percent': (5, 25), # Magnification effect
            'duration_minutes': (1, 5), # Quick evaporation/runoff
            'weight': 0.1
        },
        'broken_tracker': {
            'duration_minutes': (30, 240), # Trackers usually stay broken until fixed
            'weight': 0.1
        }
    }
}

OUTPUT_CONFIG = {
    'save_directory': 'data/injected_error',  
    'filename_suffix': '_errored',  
    'create_manifest': True,  
    'manifest_filename_template': '{base_filename}_manifest.json',  
}


# ==============================================================================
# CLASS 1: DataManager
# ==============================================================================
class DataManager:
    """Handles loading and saving SRML comprehensive format CSV files.
    
    Preserves metadata headers while processing data rows. Coerces numeric
    columns to appropriate types and handles timestamp columns carefully.
    
    Attributes
    ----------
    METADATA_ROWS : int
        Number of header rows (43) that precede the column name row (row 44)
    """
    
    METADATA_ROWS = 44 
    
    @staticmethod
    def load_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load SRML CSV file, separating metadata and data.
        
        Parameters
        ----------
        filepath : str
            Path to input CSV file
        
        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            (metadata_df, data_df) where:
            - metadata_df: First 44 rows (original headers)
            - data_df: Data rows with columns named and types coerced
        
        Raises
        ------
        FileNotFoundError
            If filepath does not exist
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_full = pd.read_csv(filepath, header=None, dtype=str,
                                  on_bad_lines='skip', engine='c')
        
        metadata = df_full.iloc[:DataManager.METADATA_ROWS].copy()
        data = df_full.iloc[DataManager.METADATA_ROWS:].copy()
        
        if len(metadata) > 43:
            column_names = metadata.iloc[43].values
            data.columns = column_names
            data = data.reset_index(drop=True)
            
        data = DataManager._coerce_to_numeric(data)
        return metadata, data
    
    @staticmethod
    def _coerce_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric columns from strings while preserving timestamps.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with string-typed columns
        
        Returns
        -------
        pd.DataFrame
            DataFrame with numeric columns converted, timestamps preserved as strings
        """
        df_converted = df.copy()
        skip_columns = {'YYYY-MM-DD--HH:MM:SS', 'Timestamp', 'DateTime', 'Time'}
        
        for col in df_converted.columns:
            if col in skip_columns:
                continue
            if isinstance(col, str) and any(pattern in col.lower() for pattern in 
                                           ['time', 'date', 'timestamp', 'datetime']):
                continue
            try:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            except:
                pass 
        return df_converted
    
    @staticmethod
    def save_data(filepath: str, metadata: pd.DataFrame, data: pd.DataFrame) -> None:
        """Save metadata and data back to SRML format CSV file.
        
        Aligns metadata columns with data columns before writing, preserving
        the original SRML format structure (44 header rows + data).
        
        Parameters
        ----------
        filepath : str
            Output file path
        metadata : pd.DataFrame
            First 44 rows from original file
        data : pd.DataFrame
            Data rows with columns and values
        
        Returns
        -------
        None
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data_columns = data.columns
        metadata_aligned = pd.DataFrame(index=range(len(metadata)), columns=data_columns)
        
        metadata_width = metadata.shape[1]
        for col_idx in range(min(metadata_width, len(data_columns))):
            metadata_aligned.iloc[:, col_idx] = metadata.iloc[:, col_idx].values
            
        df_combined = pd.concat([metadata_aligned, data], ignore_index=True)
        df_combined.to_csv(filepath, index=False, header=False)


# ==============================================================================
# CLASS 2: SolarGeometry
# ==============================================================================
class SolarGeometry:
    """Handles solar geometry calculations for daytime filtering and error placement.
    
    Uses Solar Zenith Angle (SZA) to identify daytime periods and sunset proximity.
    Enables context-aware error injection (e.g., frost errors near sunset).
    
    Attributes
    ----------
    sza_column : str
        Name of SZA column in data (default: 'SZA')
    sza_threshold : float
        SZA < threshold is considered daytime (default: 85.0 degrees)
    """
    
    def __init__(self, sza_column: str = 'SZA', sza_threshold: float = 85.0):
        """Initialize solar geometry calculator.
        
        Parameters
        ----------
        sza_column : str, optional
            Name of SZA column in input data (default: 'SZA')
        sza_threshold : float, optional
            Solar zenith angle threshold for daytime classification (default: 85.0)
        """
        self.sza_column = sza_column
        self.sza_threshold = sza_threshold
    
    def get_daytime_indices(self, data: pd.DataFrame) -> np.ndarray:
        """Identify daytime indices based on SZA.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with SZA column
        
        Returns
        -------
        np.ndarray
            Boolean array where True = daytime (SZA < threshold)
        """
        if self.sza_column not in data.columns:
            return np.ones(len(data), dtype=bool)
        sza_values = pd.to_numeric(data[self.sza_column], errors='coerce')
        return (sza_values < self.sza_threshold).fillna(False).values

    def get_sunset_proximity_indices(self, data: pd.DataFrame) -> np.ndarray:
        """Find indices in late afternoon/sunset period.
        
        Identifies times when SZA is between 75-90 degrees and increasing
        (afternoon, not morning). Used to place frost/dew errors near sunset.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with SZA column
        
        Returns
        -------
        np.ndarray
            Array of row indices in sunset proximity window
        """
        if self.sza_column not in data.columns:
            return np.array([])
            
        sza_values = pd.to_numeric(data[self.sza_column], errors='coerce')
        # SZA between 75 and 90 usually represents the hours before sunset
        # We also need to ensure SZA is increasing (afternoon, not morning)
        sza_diff = sza_values.diff()
        is_afternoon = sza_diff > 0 
        is_sunset_window = (sza_values >= 75.0) & (sza_values <= 90.0)
        
        return np.where(is_sunset_window & is_afternoon)[0]

    def select_error_start_times(self, data: pd.DataFrame, num_errors: int,
                                 daytime_bias: float = 0.8) -> np.ndarray:
        """Select random start times for errors with daytime bias.
        
        Splits error start times between daytime and nighttime based on
        daytime_bias parameter. Enables realistic error distribution.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with SZA column for daytime/nighttime classification
        num_errors : int
            Total number of errors to inject
        daytime_bias : float, optional
            Fraction of errors that should occur during daytime (default: 0.8)
        
        Returns
        -------
        np.ndarray
            Array of row indices where errors should start
        """
        daytime_idx = self.get_daytime_indices(data)
        nighttime_idx = ~daytime_idx
        
        num_daytime = max(1, int(num_errors * daytime_bias))
        num_nighttime = num_errors - num_daytime
        
        selected_indices = []
        if daytime_bias > 0:
            daytime_choices = np.where(daytime_idx)[0]
            if len(daytime_choices) > 0:
                selected_indices.extend(np.random.choice(daytime_choices, 
                                       size=min(num_daytime, len(daytime_choices)), replace=False))
                
        if num_nighttime > 0:
            nighttime_choices = np.where(nighttime_idx)[0]
            if len(nighttime_choices) > 0:
                selected_indices.extend(np.random.choice(nighttime_choices,
                                       size=min(num_nighttime, len(nighttime_choices)), replace=False))
        return np.array(sorted(set(selected_indices)))


# ==============================================================================
# CLASS 3: ErrorFunctions
# ==============================================================================
class ErrorFunctions:
    """Implements six realistic solar radiation error injection functions.
    
    Each function simulates a different type of sensor failure or environmental
    effect, with parameters controlling magnitude, duration, and feature impact.
    
    Attributes
    ----------
    feature_columns : list of str
        Irradiance features to inject errors into (default: ['GHI', 'DHI', 'DNI'])
    """
    
    def __init__(self, feature_columns: List[str] = None):
        """Initialize error function generator.
        
        Parameters
        ----------
        feature_columns : list of str, optional
            Irradiance features to work with (default: ['GHI', 'DHI', 'DNI'])
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
    
    def reduce_features_windowed(self, data: pd.DataFrame, start_idx: int, end_idx: int,
                                reduction_percent: float = 25.0, window_type: str = 'box',
                                gaussian_sigma: float = 2.0) -> Tuple[pd.DataFrame, list, list]:
        """Reduce irradiance values over a window (e.g., cloud passage, dust).
        
        Applies a smoothing window (box or Gaussian) to reduce irradiance values
        for one or more features. Simulates gradual obstruction effects.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        start_idx : int
            Start row index for error
        end_idx : int
            End row index for error
        reduction_percent : float, optional
            Percentage reduction (0-100), default: 25.0
        window_type : str, optional
            'box' or 'gaussian' windowing, default: 'box'
        gaussian_sigma : float, optional
            Standard deviation for Gaussian window (default: 2.0)
        
        Returns
        -------
        tuple of (pd.DataFrame, list, list)
            - Modified data
            - List of affected row indices
            - List of (row_idx, feature_name) tuples
        """
        df = data.copy()
        affected_rows, affected_features = [], []
        
        num_features = np.random.randint(1, len(self.feature_columns) + 1)
        features_to_reduce = np.random.choice(self.feature_columns, size=num_features, replace=False)
        duration = end_idx - start_idx + 1
        
        if window_type == 'gaussian':
            x = np.linspace(-3, 3, duration)
            window = np.exp(-(x**2) / (2 * gaussian_sigma**2))
        else:
            window = np.ones(duration)
            
        for feature in features_to_reduce:
            if feature not in df.columns:
                continue
            values = pd.to_numeric(df.iloc[start_idx:end_idx+1][feature], errors='coerce')
            reduction_factor = 1.0 - (reduction_percent / 100.0)
            new_values = np.maximum(values * reduction_factor * window, 0)
            
            df.iloc[start_idx:end_idx+1, df.columns.get_loc(feature)] = new_values
            
            for i in range(start_idx, end_idx + 1):
                affected_rows.append(i)
                affected_features.append((i, feature))
                
        return df, list(set(affected_rows)), affected_features
    
    def copy_from_another_day(self, data: pd.DataFrame, start_idx: int, end_idx: int, 
                              days_offset_range: Tuple[int, int] = (-30, -1)) -> Tuple[pd.DataFrame, list, list]:
        """Copy values from a different day (e.g., sensor stuck with repeated data).
        
        Replaces target period with values from the same time on a different day,
        simulating a sensor that repeats previous values.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        start_idx : int
            Start row index for error
        end_idx : int
            End row index for error
        days_offset_range : tuple of (int, int), optional
            Range of days to offset (negative = past), default: (-30, -1)
        
        Returns
        -------
        tuple of (pd.DataFrame, list, list)
            - Modified data
            - List of affected row indices
            - List of (row_idx, feature_name) tuples
        """
        df = data.copy()
        affected_rows, affected_features = [], []
        
        num_features = np.random.randint(1, len(self.feature_columns) + 1)
        selected_features = np.random.choice(self.feature_columns, size=num_features, replace=False)
        
        timestamp_col = next((col for col in ['YYYY-MM-DD--HH:MM:SS', 'Timestamp', 'timestamp'] if col in df.columns), None)
        if not timestamp_col:
            return df, affected_rows, affected_features
            
        target_timestamps = df.iloc[start_idx:end_idx+1][timestamp_col]
        
        for feature in selected_features:
            if feature not in df.columns: continue
            
            for local_idx, ts in zip(range(start_idx, end_idx+1), target_timestamps):
                day_offset = np.random.randint(days_offset_range[0], days_offset_range[1] + 1)
                try:
                    target_dt = pd.to_datetime(ts) + timedelta(days=day_offset)
                    ts_series = pd.to_datetime(df[timestamp_col], errors='coerce')
                    matching_rows = df[(ts_series.dt.hour == target_dt.hour) & (ts_series.dt.minute == target_dt.minute)]
                    
                    if not matching_rows.empty:
                        source_value = matching_rows.iloc[0][feature]
                    else:
                        source_value = df.iloc[np.random.choice(df.index)][feature]
                        
                    df.iloc[local_idx, df.columns.get_loc(feature)] = source_value
                    affected_rows.append(local_idx)
                    affected_features.append((local_idx, feature))
                except Exception:
                    continue
        return df, list(set(affected_rows)), affected_features
    
    def end_of_day_frost(self, data: pd.DataFrame, start_idx: int, end_idx: int,
                        dip_percent: float = 25.0) -> Tuple[pd.DataFrame, list, list]:
        """Simulate frost/ice accumulation at sunset.
        
        Lowers DNI/GHI (direct radiation blocked) while increasing DHI (diffuse
        radiation from scattered light). Intensifies as time progresses to sunset.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        start_idx : int
            Start row index for error
        end_idx : int
            End row index for error
        dip_percent : float, optional
            Maximum percentage dip for DNI/GHI (default: 25.0)
        
        Returns
        -------
        tuple of (pd.DataFrame, list, list)
            - Modified data
            - List of affected row indices
            - List of (row_idx, feature_name) tuples
        """
        df = data.copy()
        affected_rows, affected_features = [], []
        duration = end_idx - start_idx + 1
        
        # Frost worsens as time progresses toward sunset
        window = np.linspace(0, 1, duration) 
        dip_factor = 1.0 - (dip_percent / 100.0)
        
        for feature in self.feature_columns:
            if feature not in df.columns: continue
            values = pd.to_numeric(df.iloc[start_idx:end_idx+1][feature], errors='coerce')
            
            if feature == 'DHI':
                # Ice diffuses light, often slightly artificially inflating DHI compared to GHI
                new_values = values * (1.0 + (0.1 * window)) 
            else:
                # GHI and DNI drop as ice blocks direct transmittance
                new_values = values * (1.0 - ((1.0 - dip_factor) * window))
                
            new_values = np.maximum(new_values, 0)
            df.iloc[start_idx:end_idx+1, df.columns.get_loc(feature)] = new_values
            
            for i in range(start_idx, end_idx + 1):
                affected_rows.append(i)
                affected_features.append((i, feature))
                
        return df, list(set(affected_rows)), affected_features
    
    def cleaning_event(self, data: pd.DataFrame, start_idx: int, end_idx: int,
                      dip_percent_range: Tuple[float, float] = (4, 8)) -> Tuple[pd.DataFrame, list, list]:
        """Simulate brief dip from panel cleaning (expected maintenance).
        
        Applies small reductions (4-8%) to different features over consecutive
        minutes, simulating the time it takes to clean panels.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        start_idx : int
            Start row index for error
        end_idx : int
            End row index for error
        dip_percent_range : tuple of (float, float), optional
            Range for dip percentage (default: (4, 8))
        
        Returns
        -------
        tuple of (pd.DataFrame, list, list)
            - Modified data
            - List of affected row indices
            - List of (row_idx, feature_name) tuples
        """
        df = data.copy()
        affected_rows, affected_features = [], []
        duration = min(end_idx - start_idx + 1, len(self.feature_columns))
        
        for minute_offset, feature in enumerate(self.feature_columns[:duration]):
            if feature not in df.columns: continue
            target_idx = start_idx + minute_offset
            if target_idx > len(df) - 1: break
            
            dip_percent = np.random.uniform(dip_percent_range[0], dip_percent_range[1])
            dip_factor = 1.0 - (dip_percent / 100.0)
            
            current_value = pd.to_numeric(df.iloc[target_idx][feature], errors='coerce')
            if pd.notna(current_value):
                df.iloc[target_idx, df.columns.get_loc(feature)] = max(current_value * dip_factor, 0)
                affected_rows.append(target_idx)
                affected_features.append((target_idx, feature))
                
        return df, affected_rows, affected_features

    def water_droplet(self, data: pd.DataFrame, start_idx: int, end_idx: int,
                      spike_percent_range: Tuple[float, float] = (5, 25)) -> Tuple[pd.DataFrame, list, list]:
        """Simulate water droplet lensing magnification effect.
        
        Spikes irradiance values (5-25%) on GHI or DNI for 1-5 minutes as
        water droplet focuses sunlight onto sensor dome.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        start_idx : int
            Start row index for error
        end_idx : int
            End row index for error
        spike_percent_range : tuple of (float, float), optional
            Range for spike percentage (default: (5, 25))
        
        Returns
        -------
        tuple of (pd.DataFrame, list, list)
            - Modified data
            - List of affected row indices
            - List of (row_idx, feature_name) tuples
        """
        df = data.copy()
        affected_rows, affected_features = [], []
        
        # Droplets mostly affect instruments with glass domes (GHI/DHI). 
        # Pyrheliometers (DNI) usually have flat windows but can still get droplets.
        target_features = [f for f in ['GHI', 'DNI'] if f in df.columns]
        if not target_features:
            return df, affected_rows, affected_features
            
        feature = np.random.choice(target_features)
        
        spike_percent = np.random.uniform(spike_percent_range[0], spike_percent_range[1])
        spike_factor = 1.0 + (spike_percent / 100.0)
        
        values = pd.to_numeric(df.iloc[start_idx:end_idx+1][feature], errors='coerce')
        df.iloc[start_idx:end_idx+1, df.columns.get_loc(feature)] = values * spike_factor
        
        for i in range(start_idx, end_idx + 1):
            affected_rows.append(i)
            affected_features.append((i, feature))
                
        return df, list(set(affected_rows)), affected_features

    def broken_tracker(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> Tuple[pd.DataFrame, list, list]:
        """Simulate mechanical tracker failure on DNI or DHI sensor.
        
        DNI tracker failure: Drops DNI to near-zero as tracker stops following sun.
        DHI failure: Shading ball moves, causing DHI to mimic GHI values.
        Lasts 30-240 minutes (typical scope of equipment failure).
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        start_idx : int
            Start row index for error
        end_idx : int
            End row index for error
        
        Returns
        -------
        tuple of (pd.DataFrame, list, list)
            - Modified data
            - List of affected row indices
            - List of (row_idx, feature_name) tuples
        """
        df = data.copy()
        affected_rows, affected_features = [], []
        
        target_features = [f for f in ['DNI', 'DHI'] if f in df.columns]
        if not target_features:
            return df, affected_rows, affected_features
            
        broken_sensor = np.random.choice(target_features)
        
        for i in range(start_idx, end_idx + 1):
            if broken_sensor == 'DNI':
                # Tracker stops pointing at sun; direct normal radiation drops to near zero
                df.iloc[i, df.columns.get_loc('DNI')] = np.random.uniform(0, 5) 
            elif broken_sensor == 'DHI' and 'GHI' in df.columns:
                # Shading ball moves off DHI pyranometer, making it act like a GHI pyranometer
                ghi_val = pd.to_numeric(df.iloc[i]['GHI'], errors='coerce')
                if pd.notna(ghi_val):
                    df.iloc[i, df.columns.get_loc('DHI')] = ghi_val
                    
            affected_rows.append(i)
            affected_features.append((i, broken_sensor))
                
        return df, list(set(affected_rows)), affected_features


# ==============================================================================
# CLASS 4: ErrorInjectionEngine
# ==============================================================================
class ErrorInjectionEngine:
    """Orchestrates error injection workflow using configured error functions.
    
    Selects random errors, generates appropriate durations, applies them to data,
    and logs results. Central coordination between ErrorFunctions and output.
    
    Attributes
    ----------
    feature_columns : list of str
        Irradiance features to inject errors into
    config : dict
        Configuration dict with error types, probabilities, and parameters
    solar_geom : SolarGeometry
        Solar geometry calculator for daytime/sunset filtering
    error_funcs : ErrorFunctions
        Error injection function generator
    error_log : list
        Log of all errors applied in current injection session
    """
    
    def __init__(self, feature_columns: List[str] = None, config: Dict = None):
        """Initialize error injection engine.
        
        Parameters
        ----------
        feature_columns : list of str, optional
            Irradiance features (default: ['GHI', 'DHI', 'DNI'])
        config : dict, optional
            Configuration dict (default: ERROR_INJECTION_CONFIG)
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.config = config or ERROR_INJECTION_CONFIG
        self.solar_geom = SolarGeometry(sza_threshold=self.config.get('sza_threshold', 85.0))
        self.error_funcs = ErrorFunctions(self.feature_columns)
        self.error_log = []
    
    def select_random_error_function(self) -> str:
        """Randomly select error function weighted by configured probabilities.
        
        Returns
        -------
        str
            Name of error function to apply (e.g., 'reduce_features')
        """
        funcs_config = self.config['error_functions']
        functions = list(funcs_config.keys())
        weights = [funcs_config[f].get('weight', 1.0) for f in functions]
        
        # Normalize weights
        probs = np.array(weights) / sum(weights)
        return np.random.choice(functions, p=probs)
    
    def inject_errors(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Apply configurable number of realistic errors to data.
        
        Selects error start times based on daytime bias, applies errors with
        context-aware parameters (e.g., frost near sunset), and logs all changes.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with features and timestamps
        
        Returns
        -------
        tuple of (pd.DataFrame, dict)
            - Data with injected errors
            - Metadata dict with 'num_errors', 'errors' list, and 'affected_feature_tuples'
        """
        df = data.copy()
        self.error_log = []
        affected_feature_tuples = [] 
        
        if np.random.random() > self.config.get('error_probability', 1.0):
            return df, {'num_errors': 0, 'errors': []}
        
        min_errors, max_errors = self.config.get('error_count_range', (10, 30))
        num_errors = np.random.randint(min_errors, max_errors + 1)
        
        # Pre-calculate sunset indices for frost events
        sunset_indices = self.solar_geom.get_sunset_proximity_indices(df)
        start_times = list(self.solar_geom.select_error_start_times(
            df, num_errors, self.config.get('daytime_bias', 0.85)
        ))
        
        for error_num in range(num_errors):
            if not start_times: break
            
            error_func = self.select_random_error_function()
            
            # Context-aware start time selection
            if error_func == 'end_of_day_frost' and len(sunset_indices) > 0:
                start_idx = np.random.choice(sunset_indices)
            else:
                start_idx = start_times.pop(0)
                
            if start_idx >= len(df) - 1: continue
            
            # Duration generation based on physical error type
            if error_func == 'water_droplet':
                duration = np.random.randint(*self.config['error_functions']['water_droplet']['duration_minutes'])
            elif error_func == 'broken_tracker':
                duration = np.random.randint(*self.config['error_functions']['broken_tracker']['duration_minutes'])
            elif error_func == 'end_of_day_frost':
                duration = np.random.randint(30, 90) # Lasts through sunset
            elif error_func == 'cleaning_event':
                duration = self.config['error_functions']['cleaning_event']['duration_minutes']
            else:
                duration = np.random.choice([np.random.randint(1, 4), np.random.randint(10, 30)])
                
            end_idx = min(start_idx + duration - 1, len(df) - 1)
            
            try:
                if error_func == 'reduce_features':
                    params = self.config['error_functions']['reduce_features']
                    df, _, error_features = self.error_funcs.reduce_features_windowed(
                        df, start_idx, end_idx,
                        reduction_percent=np.random.uniform(*params['reduction_percent']),
                        window_type=np.random.choice(params['window_type']),
                        gaussian_sigma=params.get('gaussian_sigma_minutes', 2)
                    )
                elif error_func == 'copy_from_day':
                    params = self.config['error_functions']['copy_from_day']
                    df, _, error_features = self.error_funcs.copy_from_another_day(
                        df, start_idx, end_idx, days_offset_range=tuple(params['days_offset_range'])
                    )
                elif error_func == 'end_of_day_frost':
                    params = self.config['error_functions']['end_of_day_frost']
                    df, _, error_features = self.error_funcs.end_of_day_frost(
                        df, start_idx, end_idx, dip_percent=np.random.uniform(*params['dip_percent'])
                    )
                elif error_func == 'cleaning_event':
                    params = self.config['error_functions']['cleaning_event']
                    df, _, error_features = self.error_funcs.cleaning_event(
                        df, start_idx, end_idx, dip_percent_range=tuple(params['dip_percent'])
                    )
                elif error_func == 'water_droplet':
                    params = self.config['error_functions']['water_droplet']
                    df, _, error_features = self.error_funcs.water_droplet(
                        df, start_idx, end_idx, spike_percent_range=tuple(params['spike_percent'])
                    )
                elif error_func == 'broken_tracker':
                    df, _, error_features = self.error_funcs.broken_tracker(df, start_idx, end_idx)

                affected_feature_tuples.extend(error_features)
                unique_features = sorted(set(feat for _, feat in error_features))
                
                self.error_log.append({
                    'error_num': error_num + 1,
                    'type': error_func,
                    'start_idx': int(start_idx),
                    'end_idx': int(end_idx),
                    'duration_minutes': int(end_idx - start_idx + 1),
                    'affected_features': unique_features 
                })
            except Exception as e:
                warnings.warn(f"Error applying {error_func}: {e}")
                continue
                
        return df, {
            'num_errors': len(self.error_log),
            'errors': self.error_log,
            'affected_feature_tuples': affected_feature_tuples
        }
    
    def flag_bad_data(self, data: pd.DataFrame, error_metadata: Dict) -> pd.DataFrame:
        """Set Flag columns to 99 (BAD) for all rows affected by errors.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with injected errors
        error_metadata : dict
            Metadata from inject_errors() containing affected_feature_tuples
        
        Returns
        -------
        pd.DataFrame
            Data with Flag_GHI, Flag_DNI, Flag_DHI set to 99 for affected rows
        """
        df = data.copy()
        affected_feature_tuples = error_metadata.get('affected_feature_tuples', [])
        for row_idx, feature in affected_feature_tuples:
            flag_col = f'Flag_{feature}'
            if flag_col in df.columns and row_idx < len(df):
                df.iloc[row_idx, df.columns.get_loc(flag_col)] = 99
        return df


# ==============================================================================
# CLASS 5: OutputHandler
# ==============================================================================
class OutputHandler:
    """Saves error-injected data and creates JSON manifest tracking changes.
    
    Writes files to output directory with error metadata and detailed manifest
    for validation and error analysis.
    """
    
    @staticmethod
    def save_with_manifest(original_filepath: str, metadata: pd.DataFrame, data: pd.DataFrame,
                          error_metadata: Dict, output_config: Dict = None) -> str:
        """Save data with error metadata manifest.
        
        Parameters
        ----------
        original_filepath : str
            Path to original input file
        metadata : pd.DataFrame
            Metadata rows from original file
        data : pd.DataFrame
            Data with injected errors
        error_metadata : dict
            Dict from inject_errors() with error details
        output_config : dict, optional
            Output configuration (default: OUTPUT_CONFIG)
        
        Returns
        -------
        str
            Path to saved output CSV file
        """
        output_config = output_config or OUTPUT_CONFIG
        output_dir = output_config.get('save_directory', 'data/injected_error')
        os.makedirs(output_dir, exist_ok=True)
        
        orig_filename = Path(original_filepath).stem
        output_filepath = os.path.join(output_dir, f"{orig_filename}{output_config.get('filename_suffix', '_errored')}.csv")
        
        DataManager.save_data(output_filepath, metadata, data)
        
        if output_config.get('create_manifest', True):
            timestamp_col_idx = 2
            timestamp_col_name = data.columns[timestamp_col_idx] if timestamp_col_idx < len(data.columns) else None
            
            errors_with_times = []
            for error in error_metadata.get('errors', []):
                error_copy = error.copy()
                try:
                    start_idx, end_idx = int(error.get('start_idx', 0)), int(error.get('end_idx', 0))
                    if timestamp_col_name and start_idx < len(data) and end_idx < len(data):
                        error_copy['start_time'] = str(data.iloc[start_idx, timestamp_col_idx])
                        error_copy['end_time'] = str(data.iloc[end_idx, timestamp_col_idx])
                except (ValueError, TypeError, KeyError):
                    pass 
                errors_with_times.append(error_copy)
                
            manifest = {
                'timestamp': datetime.now().isoformat(),
                'original_file': original_filepath,
                'output_file': output_filepath,
                'num_errors_injected': error_metadata.get('num_errors', 0),
                'errors': errors_with_times,
            }
            
            manifest_filepath = os.path.join(output_dir, output_config.get('manifest_filename_template', '{base_filename}_manifest.json').format(base_filename=orig_filename))
            with open(manifest_filepath, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
        return output_filepath


# ==============================================================================
# MAIN PIPELINE CLASS
# ==============================================================================
class ErrorInjectionPipeline:
    """High-level API for end-to-end error injection workflow.
    
    Coordinates data loading, error injection, flagging, and output. Provides
    simple methods for single-file and batch processing.
    
    Attributes
    ----------
    feature_columns : list of str
        Irradiance features to inject errors into
    config : dict
        Error injection configuration
    output_config : dict
        Output file save configuration
    engine : ErrorInjectionEngine
        Core error injection engine
    
    Examples
    --------
    Basic usage::
    
        pipeline = ErrorInjectionPipeline()
        pipeline.process_file('data/file.csv', output_mode='save')
    
    Custom configuration::
    
        config = copy.deepcopy(ERROR_INJECTION_CONFIG)
        config['error_count_range'] = (5, 10)
        pipeline = ErrorInjectionPipeline(config=config)
        pipeline.process_multiple_files(filepaths, output_mode='save')
    """
    
    def __init__(self, feature_columns: List[str] = None, config: Dict = None, output_config: Dict = None):
        """Initialize error injection pipeline.
        
        Parameters
        ----------
        feature_columns : list of str, optional
            Irradiance features (default: ['GHI', 'DHI', 'DNI'])
        config : dict, optional
            Error injection config (default: ERROR_INJECTION_CONFIG)
        output_config : dict, optional
            Output config (default: OUTPUT_CONFIG)
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.config = config or ERROR_INJECTION_CONFIG
        self.output_config = output_config or OUTPUT_CONFIG
        self.engine = ErrorInjectionEngine(self.feature_columns, self.config)
    
    def process_file(self, filepath: str, output_mode: str = 'save') -> pd.DataFrame:
        """Process single file: inject errors, flag data, and save or return.
        
        Parameters
        ----------
        filepath : str
            Path to input CSV file
        output_mode : str, optional
            'save' to write files, 'return' to return dataframe (default: 'save')
        
        Returns
        -------
        pd.DataFrame
            Data with injected errors and flags set
        
        Examples
        --------
        Save to disk with manifest::
        
            pipeline.process_file('data/file.csv', output_mode='save')
        
        Return dataframe without saving::
        
            df = pipeline.process_file('data/file.csv', output_mode='return')
        """
        print(f"\n{'='*70}\nProcessing: {filepath}\n{'='*70}")
        metadata, data = DataManager.load_data(filepath)
        data_with_errors, error_metadata = self.engine.inject_errors(data)
        data_flagged = self.engine.flag_bad_data(data_with_errors, error_metadata)
        
        if output_mode == 'save':
            output_path = OutputHandler.save_with_manifest(filepath, metadata, data_flagged, error_metadata, self.output_config)
            print(f"\n✓ Complete! File saved to: {output_path}")
        elif output_mode == 'return':
            print("✓ Returning dataframe (not saving to disk)")
            
        print(f"\nError Summary:\n  - Number of errors injected: {error_metadata.get('num_errors', 0)}")
        for error in error_metadata.get('errors', []):
            print(f"    * Error {error['error_num']}: {error['type']} ({error['duration_minutes']} mins)")
        return data_flagged
    
    def process_multiple_files(self, filepaths: List[str], output_mode: str = 'save') -> List[str]:
        """Process multiple files, applying errors independently to each.
        
        Parameters
        ----------
        filepaths : list of str
            Paths to input CSV files
        output_mode : str, optional
            'save' to write files, 'return' to return dataframes (default: 'save')
        
        Returns
        -------
        list of pd.DataFrame
            List of dataframes with injected errors
        
        Examples
        --------
        Batch process monthly files::
        
            files = [f'data/STW_2023/STW_2023-{m:02d}_QC.csv' for m in range(1, 13)]
            pipeline.process_multiple_files(files, output_mode='save')
        """
        return [self.process_file(f, output_mode) for f in filepaths]


if __name__ == '__main__':
    import argparse
    from glob import glob
    
    parser = argparse.ArgumentParser(description='Inject synthetic errors into solar radiation data files.')
    parser.add_argument('filepath', help='Path to input CSV file or wildcard pattern')
    parser.add_argument('--mode', '-m', choices=['save', 'return'], default='save')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    filepaths = glob(args.filepath)
    
    if not filepaths:
        sys.exit(f"Error: No files found matching '{args.filepath}'")
        
    pipeline = ErrorInjectionPipeline()
    try:
        if len(filepaths) == 1:
            pipeline.process_file(filepaths[0], args.mode)
        else:
            pipeline.process_multiple_files(filepaths, args.mode)
    except Exception as e:
        sys.exit(f"Error: {e}")