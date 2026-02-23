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
2. SolarGeometry: Handle SZA-based daytime filtering
3. ErrorFunctions: Four configurable error injection functions
4. ErrorInjectionEngine: Orchestrate error injection workflow
5. OutputHandler: Save or return results

Usage Examples
--------------
# Load data, inject errors, save to disk with manifest
from error_injection import ErrorInjectionPipeline

pipeline = ErrorInjectionPipeline()
df_errored = pipeline.process_file(
    'data/STW_2023/STW_2023-01_QC.csv',
    output_mode='save'
)

# Or inject errors and return dataframe for training
df_errored = pipeline.process_file(
    'data/STW_2023/STW_2023-01_QC.csv',
    output_mode='return'
)

Author: Solar QC Team
"""

from __future__ import annotations
from config import SITE_CONFIG

# ==============================================================================
# CONFIGURATION - Customize error injection parameters
# ==============================================================================

# Feature columns to inject errors into
# Customize this list for different datasets with different column names
FEATURE_COLUMNS = ['GHI', 'DHI', 'DNI']

# Main error injection configuration dictionary
ERROR_INJECTION_CONFIG = {
    # Overall probability of injecting errors into a file
    # Set to 1.0 to always inject errors (use error_count_range for variation)
    'error_probability': 1.0,  # 100% - always inject errors
    
    # Number of distinct error events per file
    # For ~30 day files:
    #   - Min 10 errors = ~1 error per 3 days
    #   - Max 50 errors = ~1.67 errors per day  
    #   - Average ~30 errors = ~1 error per day
    'error_count_range': (10, 50),  # Random 10-50 separate error events
    
    # Bias toward daytime hours for error injection
    'daytime_bias': 0.8,  # 80% of errors during daytime (SZA < 90)
    'sza_threshold': 90.0,  # Solar Zenith Angle threshold for daytime
    
    # Error function parameters
    'error_functions': {
        # 1. Reduce features with window (box or gaussian)
        'reduce_features': {
            'reduction_percent': (1, 50),  # Random 1-50% reduction
            'window_type': ['box', 'gaussian'],  # Box or gaussian window
            'gaussian_sigma_minutes': 2,  # Sigma for gaussian window
            'affected_features': FEATURE_COLUMNS,  # Which features to reduce
        },
        
        # 2. Copy data from same hour on different day
        'copy_from_day': {
            'features_to_copy': ['GHI', 'DHI'],  # Default (can be overridden)
            'days_offset_range': (-30, -1),  # Use data from 1-30 days prior
            'feature_subset': ['GHI', 'DHI'],  # Only copy these features
        },
        
        # 3. Beginning/sunrise dip
        'beginning_dip': {
            'time_offset_minutes': (-30, 30),  # -30 min before to +30 min after sunrise
            'dip_percent': (10, 50),  # Deep dip 10-50%
            'affected_features': FEATURE_COLUMNS,
        },
        
        # 4. Cleaning event - sequential dips (1 feature per minute, 3 minutes total)
        'cleaning_event': {
            'dip_percent': (4, 8),  # Each feature dips 4-8%
            'duration_minutes': 3,  # Total 3 minutes (1 feature per minute)
            'affected_features': FEATURE_COLUMNS,  # 3 features = 3 minutes
        }
    }
}

# Output configuration
OUTPUT_CONFIG = {
    'save_directory': 'data/injected_error',  # Where to save modified files
    'filename_suffix': '_errored',  # Append this to original filename
    'create_manifest': True,  # Create manifest file when saving
    'manifest_filename_template': '{base_filename}_manifest.json',  # Manifest file naming
}

# Site configuration (for SZA calculations if needed)
# Imported from config.py for consistency

# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional, Union
import json
import random

# Optional pvlib for solar geometry
try:
    import pvlib
    _HAS_PVLIB = True
except ImportError:
    _HAS_PVLIB = False
    warnings.warn("pvlib not installed. SZA calculations may be limited.")


# ==============================================================================
# CLASS 1: DataManager - Load and Save Data with Metadata
# ==============================================================================
class DataManager:
    """
    Handles loading and saving solar data files while preserving metadata.
    
    The solar data CSV format has 44 rows of metadata (rows 0-43) followed
    by the actual data rows (row 44+). This class preserves that structure
    when loading and saving.
    """
    
    METADATA_ROWS = 44  # First 44 rows are metadata
    
    @staticmethod
    def load_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load solar data CSV file, separating metadata and data.
        
        Parameters
        ----------
        filepath : str
            Path to solar data CSV file
        
        Returns
        -------
        tuple
            (metadata_df, data_df) where metadata is rows 0-43 and data is row 44+
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_full = pd.read_csv(filepath, header=None, dtype=str,
                                  on_bad_lines='skip', engine='c')
        
        # Separate metadata and data
        metadata = df_full.iloc[:DataManager.METADATA_ROWS].copy()
        data = df_full.iloc[DataManager.METADATA_ROWS:].copy()
        
        # Set column names from row 43 (header row)
        if len(metadata) > 43:
            column_names = metadata.iloc[43].values
            data.columns = column_names
            data = data.reset_index(drop=True)
        
        # Convert data columns to numeric where appropriate
        data = DataManager._coerce_to_numeric(data)
        
        return metadata, data
    
    @staticmethod
    def _coerce_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert numeric columns to appropriate types.
        
        Skips timestamp and other string-based columns that should not be
        converted to numeric (they would become NaN).
        
        Parameters
        ----------
        df : pd.DataFrame
            Data frame with string dtype columns
        
        Returns
        -------
        pd.DataFrame
            Data frame with numeric columns converted
        """
        df_converted = df.copy()
        
        # Columns to skip conversion (keep as strings)
        skip_columns = {'YYYY-MM-DD--HH:MM:SS', 'Timestamp', 'DateTime', 'Time'}
        
        for col in df_converted.columns:
            # Skip string columns that shouldn't be converted to numeric
            if col in skip_columns:
                continue
            
            # Skip columns with obvious timestamp/date patterns
            if isinstance(col, str) and any(pattern in col.lower() for pattern in 
                                           ['time', 'date', 'timestamp', 'datetime']):
                continue
            
            try:
                df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
            except:
                pass  # Keep as string if conversion fails
        
        return df_converted
    
    @staticmethod
    def save_data(filepath: str, metadata: pd.DataFrame, data: pd.DataFrame) -> None:
        """
        Save solar data with metadata preserved.
        
        Combines metadata and data, ensuring proper column alignment.
        
        Parameters
        ----------
        filepath : str
            Output file path
        metadata : pd.DataFrame
            Metadata rows (rows 0-43)
        data : pd.DataFrame
            Data to save (will be appended after metadata)
        """
        # Create output directory if doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Ensure metadata has same column structure as data
        # Get the column names from data (these are the proper solar data columns)
        data_columns = data.columns
        
        # Create a new metadata dataframe with the same columns as data
        # This preserves the structure while keeping all metadata intact
        metadata_aligned = pd.DataFrame(index=range(len(metadata)), columns=data_columns)
        
        # Fill in the metadata values into the first columns (up to metadata width)
        metadata_width = metadata.shape[1]
        for col_idx in range(min(metadata_width, len(data_columns))):
            metadata_aligned.iloc[:, col_idx] = metadata.iloc[:, col_idx].values
        
        # Combine metadata and data
        df_combined = pd.concat([metadata_aligned, data], ignore_index=True)
        
        # Save without index and header
        df_combined.to_csv(filepath, index=False, header=False)
        print(f"✓ Saved: {filepath}")


# ==============================================================================
# CLASS 2: SolarGeometry - SZA-based Hour Selection
# ==============================================================================
class SolarGeometry:
    """
    Handle solar geometry calculations and daytime/nighttime filtering.
    """
    
    def __init__(self, sza_column: str = 'SZA', sza_threshold: float = 90.0):
        """
        Initialize solar geometry handler.
        
        Parameters
        ----------
        sza_column : str
            Name of Solar Zenith Angle column in data
        sza_threshold : float
            SZA threshold for daytime (default 90 = horizon)
        """
        self.sza_column = sza_column
        self.sza_threshold = sza_threshold
    
    def get_daytime_indices(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get indices of daytime rows (SZA < threshold).
        
        Parameters
        ----------
        data : pd.DataFrame
            Solar data with SZA column
        
        Returns
        -------
        np.ndarray
            Boolean array indicating daytime rows
        """
        if self.sza_column not in data.columns:
            warnings.warn(f"SZA column '{self.sza_column}' not found. "
                         "Using all hours as daytime candidates.")
            return np.ones(len(data), dtype=bool)
        
        sza_values = pd.to_numeric(data[self.sza_column], errors='coerce')
        return (sza_values < self.sza_threshold).fillna(False).values
    
    def select_error_start_times(self, data: pd.DataFrame, num_errors: int,
                                 daytime_bias: float = 0.8) -> np.ndarray:
        """
        Select random start times for error injection, biased toward daytime.
        
        Parameters
        ----------
        data : pd.DataFrame
            Solar data
        num_errors : int
            Number of error events to inject
        daytime_bias : float
            Probability of selecting daytime hours (0.0-1.0)
        
        Returns
        -------
        np.ndarray
            Indices of selected start times
        """
        daytime_idx = self.get_daytime_indices(data)
        nighttime_idx = ~daytime_idx
        
        num_daytime = max(1, int(num_errors * daytime_bias))
        num_nighttime = num_errors - num_daytime
        
        selected_indices = []
        
        # Select daytime starts
        if daytime_bias > 0:
            daytime_choices = np.where(daytime_idx)[0]
            if len(daytime_choices) > 0:
                selected_indices.extend(
                    np.random.choice(daytime_choices, 
                                   size=min(num_daytime, len(daytime_choices)),
                                   replace=False)
                )
        
        # Select nighttime starts
        if num_nighttime > 0:
            nighttime_choices = np.where(nighttime_idx)[0]
            if len(nighttime_choices) > 0:
                selected_indices.extend(
                    np.random.choice(nighttime_choices,
                                   size=min(num_nighttime, len(nighttime_choices)),
                                   replace=False)
                )
        
        return np.array(sorted(set(selected_indices)))


# ==============================================================================
# CLASS 3: ErrorFunctions - Four Types of Error Injection
# ==============================================================================
class ErrorFunctions:
    """
    Implements four distinct error injection functions for solar radiation data.
    """
    
    def __init__(self, feature_columns: List[str] = None):
        """
        Initialize error functions.
        
        Parameters
        ----------
        feature_columns : list
            List of feature column names (GHI, DHI, DNI, etc.)
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
    
    def reduce_features_windowed(self, data: pd.DataFrame, start_idx: int,
                                end_idx: int,
                                reduction_percent: float = 25.0,
                                window_type: str = 'box',
                                gaussian_sigma: float = 2.0) -> Tuple[pd.DataFrame, list, list]:
        """
        Reduce feature values between 1-50% with optional windowing.
        
        Randomly selects which features to affect (e.g., could be GHI+DNI,
        or just DHI, or all 3, etc.).
        
        Parameters
        ----------
        data : pd.DataFrame
            Solar data
        start_idx : int
            Starting index for error
        end_idx : int
            Ending index for error
        reduction_percent : float
            Percentage to reduce (1-50)
        window_type : str
            'box' for uniform reduction or 'gaussian' for smooth window
        gaussian_sigma : float
            Sigma parameter for gaussian window
        
        Returns
        -------
        tuple
            (modified_data, affected_row_indices, affected_feature_tuples)
            affected_feature_tuples: list of (row_idx, feature) tuples for flagging
        """
        df = data.copy()
        affected_rows = []
        affected_features = []  # List of (row_idx, feature) for flagging
        
        # Randomly select which features to affect (1 to len(feature_columns))
        num_features = np.random.randint(1, len(self.feature_columns) + 1)
        features_to_reduce = np.random.choice(self.feature_columns, 
                                             size=num_features, 
                                             replace=False)
        
        duration = end_idx - start_idx + 1
        
        # Create window
        if window_type == 'gaussian':
            # Gaussian window centered on error period
            x = np.linspace(-3, 3, duration)
            window = np.exp(-(x**2) / (2 * gaussian_sigma**2))
        else:  # box
            window = np.ones(duration)
        
        # Apply reduction to selected features only
        for feature in features_to_reduce:
            if feature not in df.columns:
                continue
            
            # Extract values, convert to numeric
            values = pd.to_numeric(df.iloc[start_idx:end_idx+1][feature],
                                  errors='coerce')
            
            # Apply reduction with window
            reduction_factor = 1.0 - (reduction_percent / 100.0)
            new_values = values * reduction_factor * window
            
            # Prevent negative values (except nighttime which are already near 0)
            new_values = np.maximum(new_values, 0)
            
            # Update dataframe
            df.iloc[start_idx:end_idx+1, df.columns.get_loc(feature)] = new_values
            
            # Track affected (row, feature) pairs
            for i in range(start_idx, end_idx + 1):
                affected_rows.append(i)
                affected_features.append((i, feature))
        
        return df, list(set(affected_rows)), affected_features
    
    def copy_from_another_day(self, data: pd.DataFrame, start_idx: int,
                              end_idx: int, days_offset_range: Tuple[int, int] = (-30, -1),
                              features_to_copy: List[str] = None) -> Tuple[pd.DataFrame, list, list]:
        """
        Copy values from same hour on a different day.
        
        Randomly selects which features to copy (could be just GHI,
        just DHI, GHI+DHI, all 3, etc.). This simulates cloud cover
        from a different day appearing in current data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Solar data
        start_idx : int
            Starting index for error
        end_idx : int
            Ending index for error
        days_offset_range : tuple
            Range of days to look back (negative = past days)
        features_to_copy : list
            Available features to copy from (default: GHI, DHI, DNI)
        
        Returns
        -------
        tuple
            (modified_data, affected_row_indices, affected_feature_tuples)
        """
        # Use all available features if not specified
        available_features = features_to_copy or list(self.feature_columns)
        
        df = data.copy()
        affected_rows = []
        affected_features = []  # List of (row_idx, feature) for flagging
        
        # Randomly select which features to copy (1 to all available)
        num_features = np.random.randint(1, len(available_features) + 1)
        selected_features = np.random.choice(available_features,
                                            size=num_features,
                                            replace=False)
        
        # Get timestamp column (handle various naming conventions)
        timestamp_col = None
        for col_name in ['YYYY-MM-DD--HH:MM:SS', 'Timestamp', 'timestamp']:
            if col_name in df.columns:
                timestamp_col = col_name
                break
        
        if timestamp_col is None:
            warnings.warn("Timestamp column not found. Skipping copy_from_day.")
            return df, affected_rows, affected_features
        
        # Get the target hour timestamps
        target_timestamps = df.iloc[start_idx:end_idx+1][timestamp_col]
        
        for feature in selected_features:
            if feature not in df.columns:
                continue
            
            for local_idx, ts in zip(range(start_idx, end_idx+1), target_timestamps):
                
                # Try to find same time on different day
                day_offset = np.random.randint(days_offset_range[0], days_offset_range[1] + 1)
                
                try:
                    # Parse timestamp and offset by days
                    orig_dt = pd.to_datetime(ts)
                    target_dt = orig_dt + timedelta(days=day_offset)
                    
                    # Try to find matching row
                    matching_rows = df[
                        (pd.to_datetime(df[timestamp_col], errors='coerce').dt.hour == target_dt.hour) &
                        (pd.to_datetime(df[timestamp_col], errors='coerce').dt.minute == target_dt.minute)
                    ]
                    
                    if not matching_rows.empty:
                        # Use random matching row if multiple exist
                        source_idx = np.random.choice(matching_rows.index)
                        source_value = matching_rows.iloc[0][feature]
                        df.iloc[local_idx, df.columns.get_loc(feature)] = source_value
                        affected_rows.append(local_idx)
                        affected_features.append((local_idx, feature))
                    else:
                        # If no matching time found, use any random row from the period
                        random_idx = np.random.choice(df.index)
                        random_value = df.iloc[random_idx][feature]
                        df.iloc[local_idx, df.columns.get_loc(feature)] = random_value
                        affected_rows.append(local_idx)
                        affected_features.append((local_idx, feature))
                        
                except Exception as e:
                    warnings.warn(f"Error in copy_from_another_day: {e}")
                    continue
        
        return df, list(set(affected_rows)), affected_features
    
    def beginning_of_day_dip(self, data: pd.DataFrame, start_idx: int,
                            end_idx: int,
                            dip_percent: float = 25.0) -> Tuple[pd.DataFrame, list, list]:
        """
        Create a dip in solar features at beginning of day (sunrise area).
        
        Parameters
        ----------
        data : pd.DataFrame
            Solar data
        start_idx : int
            Starting index for error
        end_idx : int
            Ending index for error
        dip_percent : float
            Depth of dip (10-50%)
        
        Returns
        -------
        tuple
            (modified_data, affected_row_indices, affected_feature_tuples)
        """
        df = data.copy()
        affected_rows = []
        affected_features = []
        
        duration = end_idx - start_idx + 1
        
        # Create ramp-up window (dip is deep at start, recovers toward end)
        window = np.linspace(0, 1, duration)  # 0 at start, 1 at end
        
        dip_factor = 1.0 - (dip_percent / 100.0)
        
        for feature in self.feature_columns:
            if feature not in df.columns:
                continue
            
            values = pd.to_numeric(df.iloc[start_idx:end_idx+1][feature],
                                  errors='coerce')
            
            # Apply dip - worst at start, recovers by end
            new_values = values * (dip_factor + (1 - dip_factor) * window)
            new_values = np.maximum(new_values, 0)
            
            df.iloc[start_idx:end_idx+1, df.columns.get_loc(feature)] = new_values
            
            for i in range(start_idx, end_idx + 1):
                affected_rows.append(i)
                affected_features.append((i, feature))
        
        return df, list(set(affected_rows)), affected_features
    
    def cleaning_event(self, data: pd.DataFrame, start_idx: int,
                      end_idx: int,
                      dip_percent_range: Tuple[float, float] = (4, 8)) -> Tuple[pd.DataFrame, list, list]:
        """
        Simulate a cleaning event: sequential 1-minute dips per feature (3 minutes total).
        
        A cleaning event shows as:
        - Minute 1: GHI dips 4-8%
        - Minute 2: DHI dips 4-8%
        - Minute 3: DNI dips 4-8%
        
        Parameters
        ----------
        data : pd.DataFrame
            Solar data
        start_idx : int
            Starting index (should span at least 3 minutes)
        end_idx : int
            Ending index
        dip_percent_range : tuple
            Range for dip percentage (4-8%)
        
        Returns
        -------
        tuple
            (modified_data, affected_row_indices, affected_feature_tuples)
        """
        df = data.copy()
        affected_rows = []
        affected_features = []
        
        # Ensure we have at least 3 minutes
        duration = min(end_idx - start_idx + 1, len(self.feature_columns))
        
        # Apply sequential dips, one feature per minute
        for minute_offset, feature in enumerate(self.feature_columns[:duration]):
            if feature not in df.columns:
                continue
            
            target_idx = start_idx + minute_offset
            if target_idx > len(df) - 1:
                break
            
            # Random dip for this feature
            dip_percent = np.random.uniform(dip_percent_range[0], dip_percent_range[1])
            dip_factor = 1.0 - (dip_percent / 100.0)
            
            current_value = pd.to_numeric(df.iloc[target_idx][feature], errors='coerce')
            
            if pd.notna(current_value):
                new_value = current_value * dip_factor
                df.iloc[target_idx, df.columns.get_loc(feature)] = max(new_value, 0)
                affected_rows.append(target_idx)
                affected_features.append((target_idx, feature))
        
        return df, affected_rows, affected_features


# ==============================================================================
# CLASS 4: ErrorInjectionEngine - Orchestrate Error Injection
# ==============================================================================
class ErrorInjectionEngine:
    """
    Main orchestrator for error injection workflow.
    
    Manages:
    - Random error event generation
    - Duration calculation
    - Error function selection
    - Flagging of modified data
    """
    
    def __init__(self, feature_columns: List[str] = None,
                 config: Dict = None):
        """
        Initialize error injection engine.
        
        Parameters
        ----------
        feature_columns : list
            List of feature columns to inject errors into
        config : dict
            Configuration dictionary (uses ERROR_INJECTION_CONFIG if not provided)
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.config = config or ERROR_INJECTION_CONFIG
        self.solar_geom = SolarGeometry(
            sza_threshold=self.config.get('sza_threshold', 90.0)
        )
        self.error_funcs = ErrorFunctions(self.feature_columns)
        self.error_log = []
    
    def generate_duration(self) -> int:
        """
        Generate error duration from a mixture distribution.
        
        Creates varied error durations:
        - 33% chance: 1-2 minutes (very brief, sensor glitch)
        - 50% chance: 10-30 minutes (moderate, cloud shadow or cleaning)
        - 17% chance: 60-120 minutes (long, extended anomaly)
        
        Returns
        -------
        int
            Duration in minutes
        """
        rand = np.random.random()
        
        if rand < 0.33:
            # Very brief errors: 1-2 minutes
            duration = np.random.randint(1, 3)
        elif rand < 0.83:
            # Moderate errors: 10-30 minutes
            duration = np.random.randint(10, 31)
        else:
            # Long errors: 1-2 hours
            duration = np.random.randint(60, 121)
        
        return duration
    
    def select_random_error_function(self) -> str:
        """
        Randomly select an error function type.
        
        Returns
        -------
        str
            Error function name: 'reduce_features', 'copy_from_day',
            'beginning_dip', or 'cleaning_event'
        """
        functions = list(self.config['error_functions'].keys())
        return np.random.choice(functions)
    
    def inject_errors(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Main error injection orchestration.
        
        Parameters
        ----------
        data : pd.DataFrame
            Solar data to inject errors into
        
        Returns
        -------
        tuple
            (modified_data, error_metadata)
        """
        df = data.copy()
        self.error_log = []
        affected_feature_tuples = []  # List of (row_idx, feature) pairs
        
        # Decide if this file gets errors
        if np.random.random() > self.config.get('error_probability', 0.5):
            return df, {'num_errors': 0, 'errors': []}
        
        # Determine number of error events
        min_errors, max_errors = self.config.get('error_count_range', (1, 5))
        num_errors = np.random.randint(min_errors, max_errors + 1)
        
        # Select start times
        start_times = self.solar_geom.select_error_start_times(
            df,
            num_errors,
            self.config.get('daytime_bias', 0.8)
        )
        
        # Apply each error
        for error_num, start_idx in enumerate(start_times):
            if start_idx >= len(df) - 1:
                continue
            
            # Generate duration
            duration = self.generate_duration()
            end_idx = min(start_idx + duration - 1, len(df) - 1)
            
            # Select error function
            error_func = self.select_random_error_function()
            
            # Apply error
            try:
                error_features = []  # Track which (row, feature) pairs were affected
                
                if error_func == 'reduce_features':
                    params = self.config['error_functions']['reduce_features']
                    reduction = np.random.uniform(*params['reduction_percent'])
                    window = np.random.choice(params['window_type'])
                    df, affected_rows, error_features = self.error_funcs.reduce_features_windowed(
                        df, start_idx, end_idx,
                        reduction_percent=reduction,
                        window_type=window,
                        gaussian_sigma=params.get('gaussian_sigma_minutes', 2)
                    )
                
                elif error_func == 'copy_from_day':
                    params = self.config['error_functions']['copy_from_day']
                    df, affected_rows, error_features = self.error_funcs.copy_from_another_day(
                        df, start_idx, end_idx,
                        days_offset_range=tuple(params['days_offset_range']),
                        features_to_copy=self.feature_columns
                    )
                
                elif error_func == 'beginning_dip':
                    params = self.config['error_functions']['beginning_dip']
                    dip = np.random.uniform(*params['dip_percent'])
                    df, affected_rows, error_features = self.error_funcs.beginning_of_day_dip(
                        df, start_idx, end_idx,
                        dip_percent=dip
                    )
                
                elif error_func == 'cleaning_event':
                    params = self.config['error_functions']['cleaning_event']
                    df, affected_rows, error_features = self.error_funcs.cleaning_event(
                        df, start_idx, end_idx,
                        dip_percent_range=tuple(params['dip_percent'])
                    )
                
                affected_feature_tuples.extend(error_features)
                
                # Extract unique feature names from error_features (list of (row, feature) tuples)
                unique_features = sorted(set(feat for _, feat in error_features))
                
                # Log error
                self.error_log.append({
                    'error_num': error_num + 1,
                    'type': error_func,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'duration_minutes': end_idx - start_idx + 1,
                    'affected_features': unique_features  # Just feature names, no indices
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
        """
        Flag all affected rows with 99 in their Flag_* columns.
        
        Marks ALL modified data points as bad (Flag_* = 99), regardless of
        their original flag status. Modified points are synthetic errors and
        should always be marked as bad.
        
        Parameters
        ----------
        data : pd.DataFrame
            Modified data with errors injected
        error_metadata : dict
            Metadata from error injection (contains affected feature tuples)
        
        Returns
        -------
        pd.DataFrame
            Data with Flag_* columns updated
        """
        df = data.copy()
        
        # Get affected (row, feature) tuples
        affected_feature_tuples = error_metadata.get('affected_feature_tuples', [])
        
        # Flag each affected (row, feature) pair as bad (99)
        # This marks all injected errors regardless of original flag status
        for row_idx, feature in affected_feature_tuples:
            flag_col = f'Flag_{feature}'
            if flag_col not in df.columns:
                continue
            
            if row_idx >= len(df):
                continue
            
            # Always mark affected points as bad
            df.iloc[row_idx, df.columns.get_loc(flag_col)] = 99
        
        return df


# ==============================================================================
# CLASS 5: OutputHandler - Save or Return Results
# ==============================================================================
class OutputHandler:
    """
    Handle output: save to disk with manifest or return dataframe.
    """
    
    @staticmethod
    def save_with_manifest(original_filepath: str, metadata: pd.DataFrame,
                          data: pd.DataFrame,
                          error_metadata: Dict,
                          output_config: Dict = None) -> str:
        """
        Save modified data and create manifest file linked to data file.
        
        Parameters
        ----------
        original_filepath : str
            Original file path (for naming)
        metadata : pd.DataFrame
            Metadata rows
        data : pd.DataFrame
            Modified data
        error_metadata : dict
            Metadata from error injection (includes errors with start_idx, end_idx)
        output_config : dict
            Output configuration
        
        Returns
        -------
        str
            Path to saved file
        """
        output_config = output_config or OUTPUT_CONFIG
        
        # Create output directory
        output_dir = output_config.get('save_directory', 'data/injected_error')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        orig_filename = Path(original_filepath).stem
        suffix = output_config.get('filename_suffix', '_errored')
        output_filename = f"{orig_filename}{suffix}.csv"
        output_filepath = os.path.join(output_dir, output_filename)
        
        # Save data
        DataManager.save_data(output_filepath, metadata, data)
        
        # Create and save manifest with linked filename
        if output_config.get('create_manifest', True):
            # Get timestamp column - typically at index 2 (after fraction columns)
            timestamp_col_idx = 2
            timestamp_col_name = data.columns[timestamp_col_idx] if timestamp_col_idx < len(data.columns) else None
            
            # Convert error indices to timestamps
            errors_with_times = []
            for error in error_metadata.get('errors', []):
                error_copy = error.copy()
                
                # Convert start_idx and end_idx to int (might be strings from JSON)
                try:
                    start_idx = int(error.get('start_idx', 0))
                    end_idx = int(error.get('end_idx', 0))
                    
                    # Add timestamps if indices are valid
                    if timestamp_col_name and start_idx < len(data) and end_idx < len(data):
                        start_time_val = data.iloc[start_idx, timestamp_col_idx]
                        end_time_val = data.iloc[end_idx, timestamp_col_idx]
                        
                        # Safely convert to string
                        error_copy['start_time'] = str(start_time_val) if pd.notna(start_time_val) else 'N/A'
                        error_copy['end_time'] = str(end_time_val) if pd.notna(end_time_val) else 'N/A'
                except (ValueError, TypeError, KeyError):
                    pass  # Keep original indices if conversion fails
                
                errors_with_times.append(error_copy)
            
            manifest = {
                'timestamp': datetime.now().isoformat(),
                'original_file': original_filepath,
                'output_file': output_filepath,
                'num_errors_injected': error_metadata.get('num_errors', 0),
                'errors': errors_with_times,
            }
            
            # Use template to create manifest name linked to data file
            manifest_template = output_config.get('manifest_filename_template', 
                                                  '{base_filename}_manifest.json')
            manifest_filename = manifest_template.format(base_filename=orig_filename)
            manifest_filepath = os.path.join(output_dir, manifest_filename)
            
            with open(manifest_filepath, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            print(f"✓ Manifest saved: {manifest_filepath}")
        
        return output_filepath

# ==============================================================================
# MAIN PIPELINE CLASS
# ==============================================================================
class ErrorInjectionPipeline:
    """
    High-level interface for error injection workflow.
    
    Combines all components into a simple API for injecting errors.
    """
    
    def __init__(self, feature_columns: List[str] = None,
                 config: Dict = None,
                 output_config: Dict = None):
        """
        Initialize error injection pipeline.
        
        Parameters
        ----------
        feature_columns : list
            Feature columns to modify (default from config)
        config : dict
            Error injection configuration (default from error_injection_config.py)
        output_config : dict
            Output configuration (default from error_injection_config.py)
        """
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.config = config or ERROR_INJECTION_CONFIG
        self.output_config = output_config or OUTPUT_CONFIG
        self.engine = ErrorInjectionEngine(self.feature_columns, self.config)
    
    def process_file(self, filepath: str,
                    output_mode: str = 'save') -> pd.DataFrame:
        """
        Process a single solar data file: inject errors and save or return.
        
        Parameters
        ----------
        filepath : str
            Path to input CSV file
        output_mode : str
            'save' to save to disk with manifest, 'return' to return dataframe
        
        Returns
        -------
        pd.DataFrame
            Modified data (will also save to disk if output_mode='save')
        """
        print(f"\n{'='*70}")
        print(f"Processing: {filepath}")
        print(f"{'='*70}")
        
        # Load data
        print("Loading data...")
        metadata, data = DataManager.load_data(filepath)
        
        # Inject errors
        print("Injecting errors...")
        data_with_errors, error_metadata = self.engine.inject_errors(data)
        
        # Flag bad data
        print("Flagging bad data...")
        data_flagged = self.engine.flag_bad_data(data_with_errors, error_metadata)
        
        # Output
        if output_mode == 'save':
            print("Saving to disk...")
            output_path = OutputHandler.save_with_manifest(
                filepath, metadata, data_flagged, error_metadata, self.output_config
            )
            print(f"\n✓ Complete! File saved to: {output_path}")
        elif output_mode == 'return':
            print("✓ Returning dataframe (not saving to disk)")
        else:
            raise ValueError(f"Unknown output_mode: {output_mode}")
        
        # Print summary
        print(f"\nError Summary:")
        print(f"  - Number of errors injected: {error_metadata.get('num_errors', 0)}")
        for error in error_metadata.get('errors', []):
            print(f"    * Error {error['error_num']}: {error['type']}")
            print(f"      Duration: {error['duration_minutes']} minutes")
            affected_features = error.get('affected_features', [])
            if affected_features:
                print(f"      Affected features: {', '.join(affected_features)}")
        
        return data_flagged
    
    def process_multiple_files(self, filepaths: List[str],
                              output_mode: str = 'save') -> List[str]:
        """
        Process multiple files.
        
        Parameters
        ----------
        filepaths : list
            List of input file paths
        output_mode : str
            'save' or 'return'
        
        Returns
        -------
        list
            List of output paths (if output_mode='save') or dataframes
        """
        results = []
        for filepath in filepaths:
            result = self.process_file(filepath, output_mode)
            results.append(result)
        return results


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================
if __name__ == '__main__':
    import argparse
    from glob import glob
    
    parser = argparse.ArgumentParser(
        description='Inject synthetic errors into solar radiation data files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Inject errors into a single file with linked manifest
  python error_injection.py data/STW_2023/STW_2023-01_QC.csv
  
  # Inject errors into multiple files using wildcard
  python error_injection.py "data/STW_2023/*.csv"
  
  # Save errored files to output folder
  python error_injection.py data/STW_2023/STW_2023-01_QC.csv --mode save
  
  # Return DataFrame instead of saving (for direct processing)
  python error_injection.py data/STW_2023/STW_2023-01_QC.csv --mode return
  
  # Process entire year with verbose output
  python error_injection.py "data/STW_2024/*.csv" --mode save --verbose
        '''
    )
    
    parser.add_argument(
        'filepath',
        help='Path to input CSV file or wildcard pattern (e.g., "data/STW_2023/*.csv")'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['save', 'return'],
        default='save',
        help='Output mode: save to disk with manifest, or return DataFrame (default: save)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed processing information'
    )
    
    args = parser.parse_args()
    
    # Expand wildcards if present
    filepaths = glob(args.filepath)
    
    if not filepaths:
        print(f"Error: No files found matching '{args.filepath}'")
        sys.exit(1)
    
    if args.verbose:
        print(f"Processing {len(filepaths)} file(s) in {args.mode} mode...\n")
    
    # Run pipeline
    pipeline = ErrorInjectionPipeline()
    try:
        if len(filepaths) == 1:
            if args.verbose:
                print(f"Processing: {filepaths[0]}")
            pipeline.process_file(filepaths[0], args.mode)
        else:
            if args.verbose:
                print(f"Processing batch of {len(filepaths)} files...")
            pipeline.process_multiple_files(filepaths, args.mode)
        
        if args.verbose:
            print("\n✓ Completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
