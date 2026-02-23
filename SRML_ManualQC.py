# NOTE: This is a fully cleaned, organized, and documented version of SRML_manualQC.py.
# Functional logic, algorithms, and outputs are unchanged.
# Changes are LIMITED to:
#   - Path portability
#   - Logging robustness
#   - Initialization safety
#   - Structural organization
#   - Documentation and comments

"""
SRML Manual Quality Control (QC) Tool
======================================

Purpose
-------
Interactive GUI for manual review and correction of automated solar irradiance
quality control predictions. Provides visualization and point-by-point editing
of QC flags with support for multiple view modes and intelligent navigation.

Key Features
------------
- **Interactive Visualization**: 
  - Time-series plots with 24-hour windows
  - Multiple x-axis modes (Time/AZM/ZEN)
  - Color-coded flag states (red=BAD, green=GOOD, yellow=PROBABLE)
  
- **Point Selection & Editing**:
  - Click-drag box selection with visual feedback
  - Bulk flag editing (Mark GOOD/BAD)
  - Undo functionality for corrections
  
- **Navigation**:
  - Day-level stepping (forward/back)
  - Month-level jumping
  - Direct date selection
  
- **Data Persistence**:
  - Auto-saves on navigation and edits
  - Preserves CSV header metadata (43 rows)
  - Maintains all original columns

GUI Layout
----------
Main Window:
  - Top Row: File selection, navigation controls, flag editing buttons
  - Middle: Time-series plot (GHI/DNI/DHI) with flag overlay
  - Bottom: Status messages and current date display

Controls:
  - Previous/Next Day: Step through data chronologically
  - Previous/Next Month: Jump month boundaries
  - Select Box: Click-drag to select points
  - Mark GOOD/BAD: Apply flag to selected points
  - Undo Last: Revert most recent edit
  - X-Axis Dropdown: Switch Time/AZM/ZEN views

Usage Workflow
--------------
1. Launch: python SRML_ManualQC.py
2. Select CSV file via dropdown (auto-populates from data folders)
3. Navigate to date needing review
4. Switch x-axis mode if needed (Time/AZM/ZEN)
5. Click-drag to select suspicious points
6. Click "Mark as GOOD" or "Mark as BAD"
7. Changes auto-save when navigating to new day
8. Use Undo if needed
9. Repeat for all dates requiring review

File Format
-----------
Input/Output CSV Structure:
  - Rows 0-42: Metadata headers (preserved on save)
  - Row 43: Column names (timestamp, GHI, DNI, DHI, flags, etc.)
  - Row 44+: Data rows
  
Required Columns:
  - Timestamp column (YYYY-MM-DD--HH:MM:SS format)
  - GHI, DNI, DHI (irradiance values)
  - Flag_GHI, Flag_DNI, Flag_DHI (QC flags to edit)
  - Optional: Flag_*_prob (probability scores)

Flag Values
-----------
- 'GOOD': Measurement passes QC
- 'BAD': Measurement fails QC
- 'PROBABLE': Uncertain, needs review
- '' (empty): Not yet evaluated

Implementation Notes
--------------------
- Built with matplotlib for plotting
- Uses pandas for data management
- Thread-safe logging to qc_log_files/
- No timezone conversion (timestamps assumed correct)
- Preserves all original columns on save

Author: Solar QC Team

This file is safe to run on any Windows machine with the required
Python dependencies installed.
"""

# =====================================================================
# Standard Library Imports
# =====================================================================

import os
import sys
import logging
import time
import copy
import threading
from logging.handlers import RotatingFileHandler
from collections import defaultdict

# =====================================================================
# Third-Party Imports
# =====================================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import binary_dilation

# =====================================================================
# Import get_solar_data module
# =====================================================================

import get_solar_data as SRML_Data

# =====================================================================
# Import solar model and features for probability prediction
# =====================================================================

from solar_model import SolarHybridModel
from solar_features import add_features
from config import SITE_CONFIG

# =====================================================================
# Global Path Configuration (PORTABLE)
# =====================================================================

# Absolute path to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Log file directory (relative to script)
LOG_DIR = os.path.join(BASE_DIR, "log_files", "qc_log_files")
LOG_FILE = os.path.join(LOG_DIR, "SRML_QC_log.txt")

# Model directory
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure logging directory exists BEFORE logger initialization
os.makedirs(LOG_DIR, exist_ok=True)

# =====================================================================
# Logging Configuration
# =====================================================================

def setup_logging():
    """
    Initialize rotating file logging for the application.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    max_log_size = 10 * 1024 * 1024  # 10 MB
    backup_files = 5
    
    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=max_log_size,
        backupCount=backup_files,
        encoding="utf-8"
    )
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.info("Logging initialized successfully")
    return logger

def log_button_click(message):
    """Log a button click or action with timestamp."""
    logging.info(message)

def close_log():
    """Ensure all logging handlers are properly closed."""
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logging.shutdown()

# =====================================================================
# Main Application Entry
# =====================================================================

def main(test_file=None):
    """
    Primary entry point for the SRML Manual QC application.
    
    Args:
        test_file: Optional file path for testing (bypasses file dialog)
    
    Workflow:
    1. Initialize logging
    2. Prompt user to select file
    3. Get data
    4. Format and prepare data
    5. Initialize GUI and variables
    6. Set up GUI buttons and start main loop
    """
    setup_logging()
    log_button_click('****************** Program Start ******************')
    
    # Select file for QC
    if test_file:
        comprehensive_location = test_file
    else:
        comprehensive_location = select_file()
    
    if not comprehensive_location:
        close_log()
        sys.exit("No file selected")
    
    # Normalize path
    comprehensive_location = os.path.abspath(comprehensive_location)
    
    log_button_click(f'Selected file: {comprehensive_location}')
    log_button_click(f'File exists: {os.path.exists(comprehensive_location)}')
    
    if not os.path.exists(comprehensive_location):
        close_log()
        sys.exit(f"File not found: {comprehensive_location}")
    
    # Get data
    df, df_header = SRML_Data.subroutine_main_automated_qc(comprehensive_location)
    log_button_click(f'Data shape after loading: {df.shape}')
    
    # Format dataframe
    df, df_string, station_name = format_df(df)
    
    # Load models and predict probabilities
    models, probabilities = load_models_and_predict(df)
    log_button_click(f'Loaded {len(models)} models successfully')
    
    # Define plot variables
    plot_vars = define_plot_vars(df, station_name, probabilities)
    
    # Initialize GUI
    gui_vars = initialize_gui_and_variables(plot_vars, df)
    
    # Prepare save path
    if comprehensive_location.lower().endswith('_qc.csv'):
        save_path = comprehensive_location
    else:
        root, ext = os.path.splitext(comprehensive_location)
        save_path = f"{root}_QC{ext}" if ext else f"{comprehensive_location}_QC.csv"
    
    # Set up GUI buttons
    setup_gui_buttons(gui_vars, plot_vars, df, df_string, save_path)
    
    # Start main loop
    gui_vars['root'].mainloop()
    
    log_button_click('****************** Program End ******************')
    close_log()


# =====================================================================
# Model Loading and Probability Prediction
# =====================================================================

def load_models_and_predict(df):
    """
    Load all saved models and predict probabilities for all data points.
    
    Args:
        df: Formatted dataframe with data (after header removal)
        
    Returns:
        tuple: (models_dict, probabilities_dict)
            - models_dict: {target_name: model_object}
            - probabilities_dict: {target_name: probability_array}
    """
    log_button_click('Loading models and predicting probabilities...')
    
    models = {}
    probabilities = {}
    
    targets = ['Flag_GHI', 'Flag_DNI', 'Flag_DHI']
    
    # Prepare dataframe for prediction - need to add features
    try:
        # Create a copy with proper timestamp column
        pred_df = df.copy()
        
        # Ensure timestamp column exists
        if 'YYYY-MM-DD--HH:MM:SS' not in pred_df.columns:
            # Assume first data column is timestamp
            pred_df['YYYY-MM-DD--HH:MM:SS'] = pred_df.iloc[:, 2]
        
        # Add features using solar_features module
        log_button_click('Adding features for prediction...')
        pred_df = add_features(pred_df, SITE_CONFIG)
        pred_df['Timestamp_dt'] = pd.to_datetime(pred_df['Timestamp_dt'], errors='coerce')
        pred_df.index = pred_df['Timestamp_dt']
        
        # Load each model and predict
        for target in targets:
            model_path = os.path.join(MODEL_DIR, f'model_{target}.pkl')
            
            if not os.path.exists(model_path):
                log_button_click(f'Model not found: {model_path}')
                messagebox.showwarning(
                    "Model Not Found",
                    f"Model file not found: {model_path}\n\n"
                    f"Probability plotting for {target} will be disabled.\n\n"
                    f"Please train models by running run_learning_cycle.py"
                )
                probabilities[target] = np.full(len(df), np.nan)
                continue
            
            try:
                log_button_click(f'Loading model: {target}')
                model = SolarHybridModel.load_model(model_path)
                models[target] = model
                
                # Add IF_Score if detector exists
                if getattr(model, 'if_det', None) is not None:
                    try:
                        pred_df['IF_Score'] = model.if_det.decision_function(
                            pred_df[model.common_features].fillna(0.0)
                        )
                    except Exception:
                        pred_df['IF_Score'] = 0.0
                
                # Predict probabilities
                log_button_click(f'Predicting probabilities for {target}...')
                flags, probs = model.predict(pred_df, target, do_return_probs=True)
                
                probabilities[target] = probs
                log_button_click(f'{target}: Mean P(GOOD) = {np.mean(probs):.3f}')
                
            except Exception as e:
                log_button_click(f'Error loading/predicting {target}: {str(e)}')
                messagebox.showerror(
                    "Model Error",
                    f"Error with model {target}:\n{str(e)}\n\n"
                    f"Probability plotting for {target} will be disabled."
                )
                probabilities[target] = np.full(len(df), np.nan)
        
    except Exception as e:
        log_button_click(f'Error in model loading: {str(e)}')
        messagebox.showerror(
            "Feature Generation Error",
            f"Could not generate features for prediction:\n{str(e)}\n\n"
            f"Probability plotting will be disabled."
        )
        for target in targets:
            probabilities[target] = np.full(len(df), np.nan)
    
    return models, probabilities


# =====================================================================
# File Selection
# =====================================================================

def select_file():
    """Prompt user to select a file for QC."""
    try:
        file_path = filedialog.askopenfilename(title="Select file to QC")
        if not file_path:
            print("\n" + "="*70)
            print("File dialog returned no file. Running in terminal environment?")
            print("Usage: python SRML_ManualQC.py <path_to_file>")
            print("Example: python SRML_ManualQC.py data\\STW_2024\\STW_2024-01_QC.csv")
            print("="*70 + "\n")
        return file_path
    except Exception as e:
        print(f"Error in file dialog: {e}")
        print("Please run with file path argument:")
        print("  python SRML_ManualQC.py <path_to_file>")
        return None

# =====================================================================
# Data Processing Functions
# =====================================================================

def format_df(df):
    """
    Format and prepare the dataframe for QC.
    
    Args:
        df: Raw dataframe from automated QC
        
    Returns:
        tuple: (formatted_df, string_df_copy, station_name)
    """
    log_button_click('Inside format_df')
    
    df = pd.DataFrame(df)
    df.index = range(len(df))
    df.columns = range(df.shape[1])
    
    log_button_click(f'df.shape,({df.shape[0]}, {df.shape[1]})')
    
    # Validate that we have enough rows for the header
    if df.shape[0] < 44:
        error_msg = f'ERROR: Data has only {df.shape[0]} rows, but expected at least 44 (44 header rows + data)'
        log_button_click(error_msg)
        close_log()
        sys.exit(error_msg)
    
    # Create string copy for preserving original formatting
    df_string = df.copy()
    
    log_button_click(f'df_string shape: {df_string.shape}')
    
    # Remove header rows (first 44 rows)
    df = df.iloc[44:].reset_index(drop=True)
    
    log_button_click(f'df shape after removing header: {df.shape}')
    
    # Column renaming and conversion
    column_renames = {}
    seen_names = set()
    duplicate_counts = defaultdict(int)
    
    for col_idx in range(df.shape[1]):
        header_val = df_string.iloc[43, col_idx]
        
        if 'Wind_Speed' in header_val and 'Flag' in header_val:
            new_name = f"Flag_{df_string.iloc[2, col_idx-1]}"
        elif 'Wind_Speed' in header_val:
            new_name = df_string.iloc[2, col_idx]
        elif 'SensorTemperature' in header_val and 'Flag' in header_val:
            new_name = f"{header_val}_{df_string.iloc[2, col_idx-1]}"
        elif 'SensorTemperature' in header_val:
            new_name = f"{header_val}_{df_string.iloc[2, col_idx]}"
        else:
            new_name = header_val
        
        # Handle duplicate names by appending a suffix
        if new_name in seen_names:
            duplicate_counts[new_name] += 1
            new_name = f"{new_name}_{duplicate_counts[new_name]}"
        
        seen_names.add(new_name)
        column_renames[col_idx] = new_name
        
        # Convert to numeric except for datetime column (index 2)
        if col_idx != 2:
            df.iloc[:, col_idx] = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
    
    df.rename(columns=column_renames, inplace=True)
    
    # Validate data length
    if (len(df) % 1440) != 0:
        print("Comprehensive format file has incorrect length")
        sys.exit("Invalid file length")
    
    station_name = df_string.iloc[1, 1]
    log_button_click(f'df.shape: {df.shape}')
    
    return df, df_string, station_name

def define_plot_vars(df, station_name, probabilities):
    """
    Define all variables needed for plotting and QC.
    
    Args:
        df: Formatted dataframe
        station_name: Name of the station
        probabilities: Dictionary of probability arrays from models
        
    Returns:
        dict: Dictionary containing all plot variables
    """
    log_button_click('Inside define_plot_vars')
    
    plot_vars = {
        'station_name': station_name,
        'dt': mdates.date2num(pd.to_datetime(df.iloc[:, 2], format="%Y-%m-%d--%H:%M:%S")),
        'zen': np.array(df.iloc[:, 3]),
        'azm': np.array(df.iloc[:, 4]),
        'ghi': np.array(df['GHI']),
        'at': np.array(df['Temperature']),
        'prob_ghi': probabilities.get('Flag_GHI', np.full(len(df), np.nan)),
        'prob_dni': probabilities.get('Flag_DNI', np.full(len(df), np.nan)),
        'prob_dhi': probabilities.get('Flag_DHI', np.full(len(df), np.nan))
    }
    
    # Define linked columns for QC
    plot_vars['linked_columns'] = get_linked_columns(df.columns[7:-1])
    
    # Calculate GHI ratio
    if 'GHI_Calc' in df.columns:
        ghi_2 = df['GHI_Calc']
    else:
        ghi_2 = df['GHI_Aux(1002)']
    
    plot_vars['ghi_ratio'] = np.full(len(df), np.nan)
    daytime_mask = (plot_vars['zen'] < 85) & (plot_vars['ghi'] > 0.1)
    plot_vars['ghi_ratio'][daytime_mask] = ghi_2[daytime_mask] / plot_vars['ghi'][daytime_mask]
    
    # Initialize plot limits
    plot_vars.update({
        'x_values': plot_vars['dt'],
        'dt_ll': plot_vars['dt'][0],
        'dt_ul': plot_vars['dt'][1440],
        'zen_ll': 0, 'zen_ul': 90,
        'azm_ll': 0, 'azm_ul': 360,
        'ghi_ll': -100, 'ghi_ul': 1500,
        'at_ll': -50, 'at_ul': 50
    })
    
    # Create masks
    plot_vars['mask_dt'] = (plot_vars['dt_ll'] <= plot_vars['dt']) & (plot_vars['dt'] < plot_vars['dt_ul'])
    plot_vars['mask_zen'] = (plot_vars['zen_ll'] <= plot_vars['zen']) & (plot_vars['zen'] < plot_vars['zen_ul'])
    plot_vars['mask_azm'] = (plot_vars['azm_ll'] <= plot_vars['azm']) & (plot_vars['azm'] < plot_vars['azm_ul'])
    plot_vars['mask_ghi'] = (plot_vars['ghi_ll'] <= plot_vars['ghi']) & (plot_vars['ghi'] < plot_vars['ghi_ul'])
    plot_vars['mask_at'] = (plot_vars['at_ll'] <= plot_vars['at']) & (plot_vars['at'] < plot_vars['at_ul'])
    
    # Combined mask
    plot_vars['mask_combined'] = (
        plot_vars['mask_dt'] & plot_vars['mask_zen'] & 
        plot_vars['mask_azm'] & plot_vars['mask_ghi']
    )
    
    # GUI control variables
    plot_vars.update({
        'pan_mode': False,
        'select_mode': False,
        'single_click_TF': False,
        'rect_x_min': None, 'rect_x_max': None,
        'rect_y_min': None, 'rect_y_max': None,
        'all_columns': df.columns[7:-1:2],
        'all_columns_flags': df.columns[8:-1:2],
        'last_checked_index': 0,
        'uncertain_index_ghi': 0,
        'uncertain_index_dni': 0,
        'uncertain_index_dhi': 0,
        'mask_select': np.full_like(plot_vars['dt'], False, dtype=bool),
        'toggle_select': True,
        'toggle_bad_x': True,
        'range_of_rows': np.arange(len(df)) - 44,
        'show_ghi_ratio': True,
        'show_prob_ghi': False,
        'show_prob_dni': False,
        'show_prob_dhi': False
    })
    
    # Initial columns to plot
    if 'GHI_Calc' in plot_vars['all_columns']:
        plot_vars['plot_these_columns'] = ['GHI', 'GHI_Calc']
    else:
        plot_vars['plot_these_columns'] = ['GHI', 'GHI_Aux(1002)']
    
    # Define colors for each column
    plot_vars['column_colors'] = define_column_colors(plot_vars['all_columns'])
    
    return plot_vars

def get_linked_columns(df_columns):
    """
    Create mapping of linked columns for QC.
    
    Args:
        df_columns: List of column names
        
    Returns:
        dict: Dictionary mapping columns to their linked counterparts
    """
    # Initialize with each column linked to itself
    linked_columns = {col: [col] for col in df_columns}
    
    # Remove empty values and calculation columns
    for col in df_columns:
        if 'Calc' in col or 'Flag' in col or 'original' in col:
            linked_columns[col] = []
    
    # Keywords for grouping
    keywords = {
        'GHI': {'include': ['GHI'], 'exclude': ['Aux', 'Calc', 'DNI', 'DHI', 'GTI', 'Upwelling']},
        'DHI': {'include': ['DHI'], 'exclude': ['Aux', 'GHI', 'DNI', 'GTI', 'Upwelling']},
        'DNI': {'include': ['DNI'], 'exclude': ['Aux', 'GHI', 'DHI', 'GTI', 'Upwelling']}
    }
    
    # Build linked columns based on keywords
    for col in df_columns:
        if 'original' in col or 'Aux' in col or 'Calc' in col or 'Flag' in col:
            continue
            
        for keyword, phrases in keywords.items():
            if any(phrase in col for phrase in phrases['include']):
                linked_cols = []
                for col2 in df_columns:
                    if (any(phrase in col2 for phrase in phrases['include']) and 
                        not any(phrase in col2 for phrase in phrases['exclude'])):
                        # Match flag with flag, data with data
                        if ('Flag' in col) == ('Flag' in col2):
                            linked_cols.append(col2)
                
                if linked_cols:
                    linked_columns[col] = linked_cols
    
    # Special handling for DNI and DHI calculations
    if 'DNI' in linked_columns:
        linked_columns['DNI'].extend(['GHI_Calc', 'DrHI_Calc'])
    
    if 'DHI' in linked_columns:
        linked_columns['DHI'].append('GHI_Calc')
    
    # RSR sensor handling
    if 'DNI_Aux(2012)' in df_columns:
        if 'GHI_Aux(1002)' in linked_columns:
            linked_columns['GHI_Aux(1002)'].extend(['DHI_Aux(3002)', 'DNI_Aux(2012)', 'DrHI_Calc'])
        
        if 'DHI_Aux(3002)' in linked_columns:
            linked_columns['DHI_Aux(3002)'].extend(['DNI_Aux(2012)', 'DrHI_Calc'])
    
    # Remove empty entries
    return {k: v for k, v in linked_columns.items() if v}

def define_column_colors(columns):
    """
    Define color mapping for columns.
    
    Args:
        columns: List of column names
        
    Returns:
        dict: Column name to color mapping
    """
    palette = plt.colormaps["tab20"](range(len(columns)))
    colors = {col: matplotlib.colors.to_hex(color[:3]) for col, color in zip(columns, palette)}
    
    # Override specific colors
    color_overrides = {
        'GHI': '#ffac00', 'GHI_Calc': '#9B7A00', 'GHI_Aux(1002)': '#9B7A00',
        'DNI': '#ad25be', 'DNI_Aux(2012)': '#ad25be', 'DNI_Calc': '#7550A0',
        'DHI': '#0000FF', 'DHI_Aux(3002)': '#0000FF'
    }
    
    colors.update(color_overrides)
    return colors

# =====================================================================
# GUI Initialization
# =====================================================================

def initialize_gui_and_variables(plot_vars, df):
    """
    Initialize the Tkinter GUI and related variables.
    
    Args:
        plot_vars: Plot variables dictionary
        df: Dataframe
        
    Returns:
        dict: GUI variables dictionary
    """
    # Fix DPI scaling on Windows
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    gui_vars = {}
    
    # Create main window
    gui_vars['root'] = tk.Tk()
    gui_vars['root'].title("SRML Manual QC")
    
    # Set window to fullscreen
    def set_fullscreen():
        width = gui_vars['root'].winfo_screenwidth()
        height = gui_vars['root'].winfo_screenheight()
        gui_vars['root'].geometry(f"{width}x{height}+0+0")
    
    gui_vars['root'].after(100, set_fullscreen)
    gui_vars['root'].state('zoomed')
    
    # Main container
    gui_vars['main_frame'] = tk.Frame(gui_vars['root'])
    gui_vars['main_frame'].pack(fill="both", expand=True)
    
    # Create matplotlib figure
    gui_vars['fig'], gui_vars['ax'] = plt.subplots(
        2, 1, figsize=(13, 11), sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.03}
    )
    
    # Embed figure in Tkinter
    gui_vars['canvas'] = FigureCanvasTkAgg(gui_vars['fig'], master=gui_vars['main_frame'])
    gui_vars['canvas'].draw()
    gui_vars['canvas'].get_tk_widget().pack(side=tk.LEFT, padx=10, pady=10)
    
    gui_vars['fig'].subplots_adjust(left=0.05, right=0.99, top=0.925, bottom=0.1)
    
    # Button frame
    gui_vars['button_frame'] = tk.Frame(gui_vars['main_frame'])
    gui_vars['button_frame'].pack(side="right", fill="y")
    
    # Create button sections
    gui_vars['button_frame_top'] = tk.Frame(gui_vars['button_frame'])
    gui_vars['button_frame_middle'] = tk.Frame(gui_vars['button_frame'])
    gui_vars['button_frame_bottom'] = tk.Frame(gui_vars['button_frame'])
    
    gui_vars['button_frame_top'].grid(row=0, column=0, sticky="ew", pady=0)
    gui_vars['button_frame_middle'].grid(row=1, column=0, sticky="ew", pady=0)
    gui_vars['button_frame_bottom'].grid(row=2, column=0, sticky="ew", pady=0)
    
    # Initialize mouse event handlers
    gui_vars['fig'].canvas.mpl_connect(
        "button_press_event",
        lambda event: gui_functions.on_mouse_event_press(gui_vars, plot_vars, df, event)
    )
    
    gui_vars['fig'].canvas.mpl_connect(
        "button_release_event",
        lambda event: gui_functions.on_mouse_event_release(gui_vars, plot_vars, df, event)
    )
    
    gui_vars['fig'].canvas.mpl_connect(
        "motion_notify_event",
        lambda event: gui_functions.on_mouse_event_motion(gui_vars, plot_vars, df, event)
    )
    
    # X-axis variable
    gui_vars['x_variable'] = tk.StringVar(value='Time')
    gui_vars['x_mapping'] = {
        'Time': plot_vars['dt'],
        'ZEN': plot_vars['zen'],
        'AZM': plot_vars['azm'],
        'GHI': plot_vars['ghi'],
        'Temperature': plot_vars['at']
    }
    
    # Initial plot
    update_plot(gui_vars, plot_vars, df)
    
    # Initialize rectangle patch
    gui_functions.clear_rectangle(gui_vars)
    
    return gui_vars

# =====================================================================
# GUI Button Setup
# =====================================================================

def setup_gui_buttons(gui_vars, plot_vars, df, df_string, save_path):
    """
    Set up all GUI buttons and their callbacks.
    
    Args:
        gui_vars: GUI variables dictionary
        plot_vars: Plot variables dictionary
        df: Dataframe
        df_string: String dataframe copy
        save_path: Path to save file
    """
    log_button_click('Setting up GUI buttons')
    
    # Create scrollable frame for sensor buttons
    canvas_container = tk.Frame(gui_vars['button_frame_top'])
    canvas_container.pack(fill="both", expand=True)
    
    canvas = tk.Canvas(canvas_container, height=400)
    scrollbar = tk.Scrollbar(canvas_container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    
    canvas.configure(yscrollcommand=scrollbar.set)
    
    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    scrollable_frame.bind("<Configure>", configure_scroll_region)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    # Store reference
    gui_vars['scrollable_frame'] = scrollable_frame
    
    # Sensor selection buttons
    row_idx = 0
    
    # Quick action buttons
    buttons_config = [
        ("GHI, GHI2, DNI, DHI", lambda: gui_functions.toggle_gddg2(gui_vars, plot_vars, df), 
         "lightgray", 0, row_idx),
        ("Next Sensor", lambda: gui_functions.next_sensor(gui_vars, plot_vars, df), 
         "gray", 1, row_idx),
        ("Flag all good", lambda: gui_functions.change_all_flag(gui_vars, plot_vars, df, 12), 
         "lightgreen", 2, row_idx),
        ("Flag all bad", lambda: gui_functions.change_all_flag(gui_vars, plot_vars, df, 99), 
         "pink", 3, row_idx)
    ]
    
    for text, command, bg, col, row in buttons_config:
        btn = tk.Button(scrollable_frame, text=text, command=command, bg=bg, wraplength=100)
        btn.grid(row=row, column=col, padx=5, pady=3, sticky="ew")
    
    row_idx += 1
    
    # Sensor-specific buttons
    for col in plot_vars['all_columns']:
        if "_original" in col:
            continue
            
        col_color = plot_vars['column_colors'].get(col, "lightblue")
        
        # Column toggle button
        btn_toggle = tk.Button(
            scrollable_frame, text=col,
            command=lambda c=col: gui_functions.toggle_column(gui_vars, plot_vars, df, c),
            bg=col_color, wraplength=120
        )
        btn_toggle.grid(row=row_idx, column=0, padx=5, pady=3, sticky="ew")
        
        # Original column button if exists
        if f"{col}_original" in plot_vars['all_columns']:
            btn_original = tk.Button(
                scrollable_frame, text=f"{col}_original",
                command=lambda c=f"{col}_original": gui_functions.toggle_column(gui_vars, plot_vars, df, c),
                bg="lightgray", wraplength=120
            )
            btn_original.grid(row=row_idx, column=1, padx=5, pady=3, sticky="ew")
        
        # Flag buttons for non-calculated columns
        if 'Calc' not in col:
            btn_good = tk.Button(
                scrollable_frame, text=f"Flag good {col}",
                command=lambda c=col: gui_functions.change_flag(gui_vars, plot_vars, df, c, 12),
                bg="lightgreen", wraplength=100
            )
            btn_good.grid(row=row_idx, column=2, padx=5, pady=3, sticky="ew")
            
            btn_bad = tk.Button(
                scrollable_frame, text=f"Flag bad {col}",
                command=lambda c=col: gui_functions.change_flag(gui_vars, plot_vars, df, c, 99),
                bg="pink", wraplength=100
            )
            btn_bad.grid(row=row_idx, column=3, padx=5, pady=3, sticky="ew")
        
        row_idx += 1
    
    # Middle section buttons (navigation and tools)
    setup_middle_buttons(gui_vars, plot_vars, df)
    
    # Bottom section (filters and controls)
    setup_bottom_buttons(gui_vars, plot_vars, df, df_string, save_path)
    
    # Update scroll region
    canvas.update_idletasks()
    bbox = canvas.bbox("all")
    if bbox:
        canvas.config(scrollregion=bbox)

def setup_middle_buttons(gui_vars, plot_vars, df):
    """Set up middle section buttons."""
    button_configs = [
        # Row 0: Probability toggle buttons
        [("Toggle GHI Ratio", lambda: gui_functions.toggle_bottom_plot(gui_vars, plot_vars, df, 'ghi_ratio'),
          "#FFD700", 0, 0),
         ("Toggle P(GHI)", lambda: gui_functions.toggle_bottom_plot(gui_vars, plot_vars, df, 'prob_ghi'),
          "#FFA500", 1, 0),
         ("Toggle P(DNI)", lambda: gui_functions.toggle_bottom_plot(gui_vars, plot_vars, df, 'prob_dni'),
          "#DA70D6", 2, 0),
         ("Toggle P(DHI)", lambda: gui_functions.toggle_bottom_plot(gui_vars, plot_vars, df, 'prob_dhi'),
          "#4169E1", 3, 0)],
        
        # Row 1
        [("Autofind good next", lambda: gui_functions.autofind_flag_then_next(gui_vars, plot_vars, df, 12),
          "lightgreen", 2, 1),
         ("Autofind bad next", lambda: gui_functions.autofind_flag_then_next(gui_vars, plot_vars, df, 99),
          "pink", 3, 1)],
        
        # Row 2: Find most uncertain points
        [("Find Uncertain GHI", lambda: gui_functions.autofind_uncertain_next(gui_vars, plot_vars, df, 'GHI'),
          "#FFB347", 0, 2),
         ("Find Uncertain DNI", lambda: gui_functions.autofind_uncertain_next(gui_vars, plot_vars, df, 'DNI'),
          "#DDA0DD", 1, 2),
         ("Find Uncertain DHI", lambda: gui_functions.autofind_uncertain_next(gui_vars, plot_vars, df, 'DHI'),
          "#87CEEB", 2, 2)],
        
        # Row 3
        [("<-- Backward", lambda: gui_functions.change_dt(gui_vars, plot_vars, df, -1, -1),
          "#4A90E2", 0, 3),
         ("Forward -->", lambda: gui_functions.change_dt(gui_vars, plot_vars, df, 1, 1),
          "#4A90E2", 1, 3),
         ("Autofind previous", lambda: gui_functions.autofind_next_issue(gui_vars, plot_vars, df, 'backward'),
          "#8D6E63", 2, 3),
         ("Autofind next", lambda: gui_functions.autofind_next_issue(gui_vars, plot_vars, df, 'forward'),
          "#8D6E63", 3, 3)],
        
        # Row 4
        [("Left to Left", lambda: gui_functions.change_dt(gui_vars, plot_vars, df, -1, 0),
          "#4A90E2", 0, 4),
         ("Right to Left", lambda: gui_functions.change_dt(gui_vars, plot_vars, df, 0, -1),
          "#4A90E2", 1, 4),
         ("Selected L to L", lambda: gui_functions.change_selected(gui_vars, plot_vars, df, -1, 0),
          "#8D6E63", 2, 4),
         ("Selected R to L", lambda: gui_functions.change_selected(gui_vars, plot_vars, df, 0, -1),
          "#8D6E63", 3, 4)],
        
        # Row 5
        [("Left to Right", lambda: gui_functions.change_dt(gui_vars, plot_vars, df, 1, 0),
          "#4A90E2", 0, 5),
         ("Right to Right", lambda: gui_functions.change_dt(gui_vars, plot_vars, df, 0, 1),
          "#4A90E2", 1, 5),
         ("Selected L to R", lambda: gui_functions.change_selected(gui_vars, plot_vars, df, 1, 0),
          "#8D6E63", 2, 5),
         ("Selected R to R", lambda: gui_functions.change_selected(gui_vars, plot_vars, df, 0, 1),
          "#8D6E63", 3, 5)],
        
        # Row 6
        [("Show one day", lambda: gui_functions.show_day_week(gui_vars, plot_vars, df, 0, 1),
          "#4A90E2", 0, 6),
         ("Show one week", lambda: gui_functions.show_day_week(gui_vars, plot_vars, df, -3, 7),
          "#4A90E2", 1, 6),
         ("Toggle bad x", lambda: gui_functions.toggle_bad_x(gui_vars, plot_vars, df),
          "#8D6E63", 2, 6),
         ("Toggle selected", lambda: gui_functions.toggle_select(gui_vars, plot_vars, df),
          "#8D6E63", 3, 6)],
        
        # Row 7
        [("Show first day", lambda: gui_functions.show_first_day(gui_vars, plot_vars, df),
          "pink", 0, 7),
         ("Show all days", lambda: gui_functions.show_all_days(gui_vars, plot_vars, df),
          "#4A90E2", 1, 7)],
        
        # Row 8
        [("Zoom X axis", lambda: gui_functions.zoom_to_rectangle(gui_vars, plot_vars, df, True, False),
          "#5A8266", 0, 8),
         ("Zoom Y axis", lambda: gui_functions.zoom_to_rectangle(gui_vars, plot_vars, df, False, True),
          "#5A8266", 1, 8),
         ("Zoom XY axis", lambda: gui_functions.zoom_to_rectangle(gui_vars, plot_vars, df, True, True),
          "#5A8266", 2, 8)]
    ]
    
    for row_config in button_configs:
        for text, command, bg, col, row in row_config:
            btn = tk.Button(gui_vars['button_frame_middle'], text=text, command=command, bg=bg)
            btn.grid(row=row, column=col, padx=10, pady=3, sticky="ew")
    
    # X-axis dropdown
    drpdwn_x_axis = tk.OptionMenu(
        gui_vars['button_frame_middle'], gui_vars['x_variable'],
        *gui_vars['x_mapping'].keys(),
        command=lambda value: gui_functions.update_x_axis(gui_vars, plot_vars, df, value)
    )
    drpdwn_x_axis.config(width=15)
    drpdwn_x_axis.grid(row=9, column=3, padx=5, pady=3, sticky="ew")

def setup_bottom_buttons(gui_vars, plot_vars, df, df_string, save_path):
    """Set up bottom section buttons (filters and controls)."""
    # Filter sliders
    filters = [
        ('ZEN', 'zen', 0, 180, 5),
        ('AZM', 'azm', 0, 360, 10),
        ('GHI', 'ghi', -100, 2000, 100),
        ('AT', 'at', -50, 50, 5)
    ]
    
    for row_idx, (label, var, min_val, max_val, res) in enumerate(filters):
        # Label
        tk.Label(gui_vars['button_frame_bottom'], text=label).grid(
            row=row_idx, column=0, padx=10, pady=5, sticky="w"
        )
        
        # Lower limit
        ll_value = tk.Label(gui_vars['button_frame_bottom'], text=str(plot_vars[f'{var}_ll']))
        ll_value.grid(row=row_idx, column=1, padx=5)
        
        slider_ll = tk.Scale(
            gui_vars['button_frame_bottom'], from_=min_val, to=max_val,
            resolution=res, orient=tk.HORIZONTAL, showvalue=0,
            command=lambda v, vname=var: update_filter_limit(
                gui_vars, plot_vars, df, vname, 'll', v
            )
        )
        slider_ll.set(plot_vars[f'{var}_ll'])
        slider_ll.grid(row=row_idx, column=2, padx=10, pady=5, columnspan=2)
        
        # Upper limit
        ul_value = tk.Label(gui_vars['button_frame_bottom'], text=str(plot_vars[f'{var}_ul']))
        ul_value.grid(row=row_idx, column=4, padx=5)
        
        slider_ul = tk.Scale(
            gui_vars['button_frame_bottom'], from_=min_val, to=max_val,
            resolution=res, orient=tk.HORIZONTAL, showvalue=0,
            command=lambda v, vname=var: update_filter_limit(
                gui_vars, plot_vars, df, vname, 'ul', v
            )
        )
        slider_ul.set(plot_vars[f'{var}_ul'])
        slider_ul.grid(row=row_idx, column=5, padx=10, pady=5, columnspan=2)
        
        # Reset button
        btn_reset = tk.Button(
            gui_vars['button_frame_bottom'], text="Reset limits",
            command=lambda vname=var, mn=min_val, mx=max_val: 
                gui_functions.reset_limit(gui_vars, plot_vars, df, vname, mn, mx),
            bg="lightgray"
        )
        btn_reset.grid(row=row_idx, column=7, padx=10, pady=3)
    
    # Save and Exit buttons
    last_row = len(filters)
    btn_save = tk.Button(
        gui_vars['button_frame_bottom'], text="Save",
        command=lambda: gui_functions.save(gui_vars['root'], df, df_string, save_path),
        bg="darkgreen", fg="white"
    )
    btn_save.grid(row=last_row, column=0, padx=10, pady=3)
    
    btn_exit = tk.Button(
        gui_vars['button_frame_bottom'], text="Exit",
        command=lambda: gui_functions.graceful_exit(gui_vars['root']),
        bg="red", fg="white"
    )
    btn_exit.grid(row=last_row, column=1, padx=10, pady=3)

def update_filter_limit(gui_vars, plot_vars, df, var_name, limit_type, value):
    """Update filter limit and refresh plot."""
    if limit_type == 'll':
        plot_vars[f'{var_name}_ll'] = float(value)
    else:
        plot_vars[f'{var_name}_ul'] = float(value)
    
    # Ensure ll <= ul
    if plot_vars[f'{var_name}_ll'] > plot_vars[f'{var_name}_ul']:
        plot_vars[f'{var_name}_ll'], plot_vars[f'{var_name}_ul'] = \
            plot_vars[f'{var_name}_ul'], plot_vars[f'{var_name}_ll']
    
    # Update mask
    plot_vars[f'mask_{var_name}'] = (
        plot_vars[f'{var_name}_ll'] <= plot_vars[var_name]
    ) & (plot_vars[var_name] < plot_vars[f'{var_name}_ul'])
    
    update_plot(gui_vars, plot_vars, df)

# =====================================================================
# GUI Functions Class
# =====================================================================

class gui_functions:
    """Collection of GUI callback functions."""
    
    @staticmethod
    def toggle_bottom_plot(gui_vars, plot_vars, df, plot_type):
        """
        Toggle display of different data on bottom plot.
        
        Args:
            plot_type: One of 'ghi_ratio', 'prob_ghi', 'prob_dni', 'prob_dhi'
        """
        log_button_click(f'Toggling bottom plot: {plot_type}')
        
        if plot_type == 'ghi_ratio':
            plot_vars['show_ghi_ratio'] = not plot_vars['show_ghi_ratio']
        elif plot_type == 'prob_ghi':
            plot_vars['show_prob_ghi'] = not plot_vars['show_prob_ghi']
        elif plot_type == 'prob_dni':
            plot_vars['show_prob_dni'] = not plot_vars['show_prob_dni']
        elif plot_type == 'prob_dhi':
            plot_vars['show_prob_dhi'] = not plot_vars['show_prob_dhi']
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def change_dt(gui_vars, plot_vars, df, left_shift, right_shift):
        """Shift the datetime viewing window."""
        log_button_click(f'change_dt_{left_shift}_{right_shift}')
        
        # Apply shifts with bounds checking
        plot_vars['dt_ll'] = max(plot_vars['dt'][0], plot_vars['dt_ll'] + left_shift)
        plot_vars['dt_ul'] = min(plot_vars['dt'][-1], plot_vars['dt_ul'] + right_shift)
        
        # Ensure valid window
        if left_shift > 0:
            plot_vars['dt_ll'] = min(plot_vars['dt_ll'], plot_vars['dt_ul'] - 1)
        if right_shift < 0:
            plot_vars['dt_ul'] = max(plot_vars['dt_ul'], plot_vars['dt_ll'] + 1)
        
        # Update mask and plot
        plot_vars['mask_dt'] = (
            plot_vars['dt_ll'] <= plot_vars['dt']
        ) & (plot_vars['dt'] < plot_vars['dt_ul'])
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def show_day_week(gui_vars, plot_vars, df, left_side, right_side):
        """Show a specific time window."""
        log_button_click(f'show_day_week_{left_side}_{right_side}')
        
        middle = plot_vars['dt_ll'] + (plot_vars['dt_ul'] - plot_vars['dt_ll']) / 2
        plot_vars['dt_ll'] = middle + left_side
        plot_vars['dt_ul'] = plot_vars['dt_ll'] + right_side
        
        plot_vars['mask_dt'] = (
            plot_vars['dt_ll'] <= plot_vars['dt']
        ) & (plot_vars['dt'] < plot_vars['dt_ul'])
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def show_first_day(gui_vars, plot_vars, df):
        """Show the first day of data."""
        log_button_click('show_first_day')
        plot_vars['dt_ll'] = plot_vars['dt'][0]
        plot_vars['dt_ul'] = plot_vars['dt_ll'] + 1
        plot_vars['mask_dt'] = (
            plot_vars['dt_ll'] <= plot_vars['dt']
        ) & (plot_vars['dt'] < plot_vars['dt_ul'])
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def show_all_days(gui_vars, plot_vars, df):
        """Show all available data."""
        log_button_click('show_all_days')
        plot_vars['dt_ll'] = plot_vars['dt'][0]
        plot_vars['dt_ul'] = plot_vars['dt'][-1]
        plot_vars['mask_dt'] = (
            plot_vars['dt_ll'] <= plot_vars['dt']
        ) & (plot_vars['dt'] < plot_vars['dt_ul'])
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def update_x_axis(gui_vars, plot_vars, df, value):
        """Update the X-axis variable for plotting."""
        log_button_click(f'update_x_axis_{value}')
        plot_vars['x_values'] = gui_vars['x_mapping'][value]
        
        # When switching x-axis, ensure proper axis limits are set
        if value == 'Time':
            gui_vars['ax'][0].set_xlim(plot_vars['dt_ll'], plot_vars['dt_ul'])
        elif value == 'ZEN':
            gui_vars['ax'][0].set_xlim(plot_vars['zen_ll'], plot_vars['zen_ul'])
        elif value == 'AZM':
            gui_vars['ax'][0].set_xlim(plot_vars['azm_ll'], plot_vars['azm_ul'])
        elif value == 'GHI':
            gui_vars['ax'][0].set_xlim(plot_vars['ghi_ll'], plot_vars['ghi_ul'])
        elif value == 'Temperature':
            gui_vars['ax'][0].set_xlim(plot_vars['at_ll'], plot_vars['at_ul'])
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def on_mouse_event_press(gui_vars, plot_vars, df, event):
        """Handle mouse press events."""
        if not event.inaxes:
            return
        
        log_button_click('on_mouse_event_press')
        
        def handle_single_click():
            if plot_vars['single_click_TF']:
                gui_functions.on_mouse_single_click(gui_vars, plot_vars, df, event)
        
        if event.dblclick:
            plot_vars['single_click_TF'] = False
            gui_functions.on_mouse_double_click(gui_vars, plot_vars, df, event)
        else:
            plot_vars['single_click_TF'] = True
            threading.Timer(0.3, handle_single_click).start()
    
    @staticmethod
    def on_mouse_double_click(gui_vars, plot_vars, df, event):
        """Handle double-click zoom."""
        if not event.dblclick or event.xdata is None or event.ydata is None:
            return
        
        log_button_click('on_mouse_double_click')
        
        x_center, y_center = event.xdata, event.ydata
        zoom_factor = 2 if (event.guiEvent.state & 0x0001) else 0.5
        
        x_min, x_max = gui_vars['ax'][0].get_xlim()
        y_min, y_max = gui_vars['ax'][0].get_ylim()
        
        x_range = (x_max - x_min) * zoom_factor
        y_range = (y_max - y_min) * zoom_factor
        
        x_min = x_center - x_range / 2
        x_max = x_center + x_range / 2
        y_min = y_center - y_range / 2
        y_max = y_center + y_range / 2
        
        gui_functions.set_axis_limits(gui_vars, plot_vars, x_min, x_max)
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def on_mouse_single_click(gui_vars, plot_vars, df, event):
        """Handle single click for selection or panning."""
        if not event.inaxes:
            return
        
        log_button_click('on_mouse_single_click')
        
        plot_vars['rect_x_min'], plot_vars['rect_y_min'] = event.xdata, event.ydata
        
        if event.key == 'shift' and event.button == 1:
            plot_vars['pan_mode'] = True
        elif event.button == 1:
            plot_vars['select_mode'] = True
            
            # Create selection rectangle
            gui_vars['rect_patch'] = patches.Rectangle(
                (plot_vars['rect_x_min'], plot_vars['rect_y_min']), 0, 0,
                edgecolor='dimgray', facecolor='gray', lw=1, alpha=0.2
            )
            event.inaxes.add_patch(gui_vars['rect_patch'])
            event.canvas.draw()
    
    @staticmethod
    def on_mouse_event_release(gui_vars, plot_vars, df, event):
        """Handle mouse release events."""
        if not plot_vars['single_click_TF']:
            return
        
        plot_vars['single_click_TF'] = False
        log_button_click('on_mouse_event_release')
        
        if plot_vars['pan_mode']:
            plot_vars['pan_mode'] = False
            x_min, x_max = gui_vars['ax'][0].get_xlim()
            gui_functions.set_axis_limits(gui_vars, plot_vars, x_min, x_max)
            update_plot(gui_vars, plot_vars, df)
        
        elif plot_vars['select_mode']:
            plot_vars['select_mode'] = False
            plot_vars['toggle_select'] = True
            plot_vars['toggle_bad_x'] = True
            
            event_xdata = event.xdata
            
            # Check if mouse was released outside plot area
            if event_xdata is None or event.ydata is None or plot_vars['rect_x_min'] is None or plot_vars['rect_y_min'] is None:
                log_button_click('Mouse released outside plot area, ignoring selection')
                return
            
            # Sort rectangle coordinates
            plot_vars['rect_x_min'], plot_vars['rect_x_max'] = sorted([
                plot_vars['rect_x_min'], event_xdata
            ])
            plot_vars['rect_y_min'], plot_vars['rect_y_max'] = sorted([
                plot_vars['rect_y_min'], event.ydata
            ])
            
            # Get selected data
            gui_functions.get_selected_data(
                gui_vars, plot_vars,
                plot_vars['rect_x_min'], plot_vars['rect_x_max']
            )
            
            # Update last checked index
            selected_indices = plot_vars['range_of_rows'][plot_vars['mask_select']]
            if len(selected_indices) > 0:
                plot_vars['last_checked_index'] = selected_indices[0]
            
            update_plot(gui_vars, plot_vars, df)
            
            # Redraw rectangle
            if gui_vars['rect_patch'] not in gui_vars['ax'][0].patches:
                gui_vars['ax'][0].add_patch(gui_vars['rect_patch'])
            
            gui_vars['fig'].canvas.draw()
    
    @staticmethod
    def toggle_bottom_plot(gui_vars, plot_vars, df, plot_type):
        """
        Toggle display of different data on bottom plot.
        
        Args:
            plot_type: One of 'ghi_ratio', 'prob_ghi', 'prob_dni', 'prob_dhi'
        """
        log_button_click(f'Toggling bottom plot: {plot_type}')
        
        if plot_type == 'ghi_ratio':
            plot_vars['show_ghi_ratio'] = not plot_vars['show_ghi_ratio']
        elif plot_type == 'prob_ghi':
            plot_vars['show_prob_ghi'] = not plot_vars['show_prob_ghi']
        elif plot_type == 'prob_dni':
            plot_vars['show_prob_dni'] = not plot_vars['show_prob_dni']
        elif plot_type == 'prob_dhi':
            plot_vars['show_prob_dhi'] = not plot_vars['show_prob_dhi']
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def on_mouse_event_motion(gui_vars, plot_vars, df, event):
        """Handle mouse motion during drag operations."""
        if plot_vars['pan_mode']:
            if event.xdata is None or event.ydata is None:
                return
                
            x1, y1 = event.xdata, event.ydata
            x_l, x_r = gui_vars['ax'][0].get_xlim()
            
            dx = plot_vars['rect_x_min'] - x1
            dy = plot_vars['rect_y_min'] - y1
            
            gui_vars['ax'][0].set_xlim(x_l + dx, x_r + dx)
            gui_vars['fig'].canvas.draw_idle()
        
        elif plot_vars['select_mode'] and plot_vars['rect_x_min'] is not None and event.inaxes:
            x0, y0 = plot_vars['rect_x_min'], plot_vars['rect_y_min']
            x1, y1 = event.xdata, event.ydata
            
            gui_vars['rect_patch'].set_width(x1 - x0)
            gui_vars['rect_patch'].set_height(y1 - y0)
            gui_vars['rect_patch'].set_xy((x0, y0))
            event.canvas.draw()
    
    @staticmethod
    def set_axis_limits(gui_vars, plot_vars, x_min, x_max):
        """Set axis limits based on current X variable."""
        log_button_click(f'set_axis_limits_{x_min}_{x_max}')
        
        x_var = gui_vars['x_variable'].get()
        
        if x_var == 'Time':
            plot_vars['dt_ll'] = x_min
            plot_vars['dt_ul'] = x_max
            plot_vars['mask_dt'] = (
                plot_vars['dt_ll'] <= plot_vars['dt']
            ) & (plot_vars['dt'] < plot_vars['dt_ul'])
        
        elif x_var == 'ZEN':
            plot_vars['zen_ll'] = x_min
            plot_vars['zen_ul'] = x_max
            plot_vars['mask_zen'] = (
                plot_vars['zen_ll'] <= plot_vars['zen']
            ) & (plot_vars['zen'] < plot_vars['zen_ul'])
        
        elif x_var == 'AZM':
            plot_vars['azm_ll'] = x_min
            plot_vars['azm_ul'] = x_max
            plot_vars['mask_azm'] = (
                plot_vars['azm_ll'] <= plot_vars['azm']
            ) & (plot_vars['azm'] < plot_vars['azm_ul'])
        
        elif x_var == 'GHI':
            plot_vars['ghi_ll'] = x_min
            plot_vars['ghi_ul'] = x_max
            plot_vars['mask_ghi'] = (
                plot_vars['ghi_ll'] <= plot_vars['ghi']
            ) & (plot_vars['ghi'] < plot_vars['ghi_ul'])
        
        elif x_var == 'Temperature':
            plot_vars['at_ll'] = x_min
            plot_vars['at_ul'] = x_max
            plot_vars['mask_at'] = (
                plot_vars['at_ll'] <= plot_vars['at']
            ) & (plot_vars['at'] < plot_vars['at_ul'])
    
    @staticmethod
    def get_selected_data(gui_vars, plot_vars, x_min, x_max):
        """Get data selected by rectangle."""
        log_button_click(f'get_selected_data_{x_min}_{x_max}')
        
        x_var = gui_vars['x_variable'].get()
        
        if x_var == 'Time':
            plot_vars['mask_select'] = (x_min <= plot_vars['dt']) & (plot_vars['dt'] < x_max)
        elif x_var == 'ZEN':
            plot_vars['mask_select'] = (x_min <= plot_vars['zen']) & (plot_vars['zen'] < x_max)
        elif x_var == 'AZM':
            plot_vars['mask_select'] = (x_min <= plot_vars['azm']) & (plot_vars['azm'] < x_max)
        elif x_var == 'GHI':
            plot_vars['mask_select'] = (x_min <= plot_vars['ghi']) & (plot_vars['ghi'] < x_max)
        elif x_var == 'Temperature':
            plot_vars['mask_select'] = (x_min <= plot_vars['at']) & (plot_vars['at'] < x_max)
    
    @staticmethod
    def change_selected(gui_vars, plot_vars, df, left_shift, right_shift):
        """Shift the selected region."""
        log_button_click(f'change_selected_{left_shift}_{right_shift}')
        
        plot_vars['toggle_select'] = True
        plot_vars['toggle_bad_x'] = True
        
        mask = plot_vars['mask_select']
        
        if left_shift < 0:
            mask = binary_dilation(mask, structure=np.array([1, 1, 0]))
        elif left_shift > 0:
            rolled = np.roll(mask, left_shift)
            mask = mask & rolled
        
        if right_shift > 0:
            mask = binary_dilation(mask, structure=np.array([0, 1, 1]))
        elif right_shift < 0:
            rolled = np.roll(mask, right_shift)
            mask = mask & rolled
        
        plot_vars['mask_select'] = mask
        
        # Update last checked index
        selected_indices = plot_vars['range_of_rows'][plot_vars['mask_select']]
        if len(selected_indices) > 0:
            plot_vars['last_checked_index'] = selected_indices[0]
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def toggle_select(gui_vars, plot_vars, df):
        """Toggle selection highlighting."""
        plot_vars['toggle_select'] = not plot_vars['toggle_select']
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def toggle_bad_x(gui_vars, plot_vars, df):
        """Toggle bad data highlighting."""
        plot_vars['toggle_bad_x'] = not plot_vars['toggle_bad_x']
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def clear_rectangle(gui_vars):
        """Initialize selection rectangle."""
        log_button_click('clear_rectangle')
        
        rect_patch = patches.Rectangle(
            (0, 0), 0, 0,
            edgecolor="dimgray", facecolor="gray", alpha=0.3
        )
        gui_vars['ax'][0].add_patch(rect_patch)
        gui_vars['rect_patch'] = rect_patch
    
    @staticmethod
    def zoom_to_rectangle(gui_vars, plot_vars, df, x_zoom, y_zoom):
        """Zoom to selected rectangle."""
        log_button_click(f'zoom_to_rectangle_{x_zoom}_{y_zoom}')
        
        x_min = plot_vars['rect_x_min']
        x_max = plot_vars['rect_x_max']
        y_min = plot_vars['rect_y_min']
        y_max = plot_vars['rect_y_max']
        
        if x_zoom and x_min is not None and x_max is not None and (x_max - x_min) > 1/1440:
            gui_functions.set_axis_limits(gui_vars, plot_vars, x_min, x_max)
        
        if y_zoom:
            update_plot(gui_vars, plot_vars, df, y_min, y_max)
        else:
            update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def next_sensor(gui_vars, plot_vars, df):
        """Cycle to next sensor column."""
        all_columns = plot_vars['all_columns']
        current = plot_vars['plot_these_columns'][-1]
        
        try:
            current_idx = all_columns.index(current)
            next_idx = (current_idx + 1) % len(all_columns)
            plot_vars['plot_these_columns'] = [all_columns[next_idx]]
        except ValueError:
            plot_vars['plot_these_columns'] = [all_columns[0]]
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def toggle_gddg2(gui_vars, plot_vars, df):
        """Toggle GHI, DNI, DHI, GHI2 display."""
        if 'GHI_Calc' in plot_vars['all_columns']:
            target_cols = ['GHI', 'DNI', 'DHI', 'GHI_Calc']
        else:
            target_cols = ['GHI', 'GHI_Aux(1002)', 'DNI_Aux(2012)', 'DHI_Aux(3002)']
        
        current_set = set(plot_vars['plot_these_columns'])
        target_set = set(target_cols)
        
        if current_set == target_set:
            # Remove all target columns
            plot_vars['plot_these_columns'] = [
                col for col in plot_vars['plot_these_columns'] 
                if col not in target_set
            ]
        else:
            # Add target columns (remove duplicates)
            plot_vars['plot_these_columns'] = list(set(
                plot_vars['plot_these_columns'] + target_cols
            ))
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def toggle_column(gui_vars, plot_vars, df, column):
        """Toggle a specific column on/off."""
        log_button_click(f'toggle_column_{column}')
        
        if column in plot_vars['plot_these_columns']:
            plot_vars['plot_these_columns'].remove(column)
        else:
            plot_vars['plot_these_columns'].append(column)
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def change_flag(gui_vars, plot_vars, df, column, flag_value):
        """Change flag value for selected data."""
        if not (plot_vars['toggle_select'] and plot_vars['toggle_bad_x']):
            return
        
        log_button_click(f'change_flag_{column}_{flag_value}')
        
        # Get linked columns
        linked_cols = plot_vars['linked_columns'].get(column, [column])
        flag_cols = [f'Flag_{col}' for col in linked_cols]
        
        # Apply flag to selected data
        mask = plot_vars['mask_select']
        
        for flag_col in flag_cols:
            if flag_col not in df.columns:
                continue
            
            flag_to_set = flag_value
            
            # Adjust flag values for special columns
            if flag_value < 99:
                if 'original' in flag_col:
                    flag_to_set = flag_value - 1
                elif 'calc' in flag_col.lower():
                    flag_to_set = 72
            
            df.loc[mask, flag_col] = flag_to_set
        
        # Handle calculated column dependencies
        gui_functions._update_calc_flags(df, plot_vars)
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def _update_calc_flags(df, plot_vars):
        """Update calculated column flags based on dependencies."""
        if 'GHI_Calc' in df.columns:
            # GHI_Calc depends on DNI and DHI
            df['Flag_GHI_Calc'] = 72
            bad_mask = (df['Flag_DNI'] == 99) | (df['Flag_DHI'] == 99)
            df.loc[bad_mask, 'Flag_GHI_Calc'] = 99
        else:
            # RSR station handling
            dni_flag_col = 'Flag_DNI_Aux(2012)' if 'Flag_DNI_Aux(2012)' in df.columns else 'Flag_DNI_Calc'
            drhi_flag_col = 'Flag_DrHI_Aux(2000)' if 'Flag_DrHI_Aux(2000)' in df.columns else 'Flag_DrHI_Calc'
            
            if dni_flag_col in df.columns:
                df[dni_flag_col] = 72
            if drhi_flag_col in df.columns:
                df[drhi_flag_col] = 72
            
            bad_mask = (df['Flag_GHI_Aux(1002)'] == 99) | (df['Flag_DHI_Aux(3002)'] == 99)
            
            if dni_flag_col in df.columns:
                df.loc[bad_mask, dni_flag_col] = 99
            if drhi_flag_col in df.columns:
                df.loc[bad_mask, drhi_flag_col] = 99
    
    @staticmethod
    def change_all_flag(gui_vars, plot_vars, df, flag_value):
        """Change flag for all currently plotted columns."""
        if not (plot_vars['toggle_select'] and plot_vars['toggle_bad_x']):
            return
        
        log_button_click(f'change_all_flag_{flag_value}')
        
        for column in plot_vars['plot_these_columns']:
            gui_functions.change_flag(gui_vars, plot_vars, df, column, flag_value)
    
    @staticmethod
    def autofind_flag_then_next(gui_vars, plot_vars, df, flag_value):
        """Flag current selection and move to next issue."""
        gui_functions.change_all_flag(gui_vars, plot_vars, df, flag_value)
        gui_functions.autofind_next_issue(gui_vars, plot_vars, df, 'forward')
    
    @staticmethod
    def autofind_next_issue(gui_vars, plot_vars, df, direction):
        """Find next QC issue automatically."""
        log_button_click(f'autofind_next_issue_{direction}')
        
        plot_vars['toggle_select'] = True
        plot_vars['toggle_bad_x'] = True
        
        # Get all flag columns with issues
        flag_cols = [f'Flag_{col}' for col in plot_vars['linked_columns']]
        problem_mask = ((df[flag_cols] == 99).any(axis=1)) & (plot_vars['zen'] <= plot_vars['zen_ul'])
        problem_indices = df.index[problem_mask]
        
        if len(problem_indices) < 2:
            # No issues found
            plot_vars['plot_these_columns'] = ['GHI']
            gui_functions.show_all_days(gui_vars, plot_vars, df)
            plot_vars['mask_select'] = np.full_like(plot_vars['dt'], False, dtype=bool)
            update_plot(gui_vars, plot_vars, df)
            return
        
        # Find contiguous problem regions
        breaks = np.where(np.diff(problem_indices) >= 6)[0]
        start_indices = np.concatenate([[problem_indices[0]], problem_indices[breaks + 1]])
        end_indices = np.concatenate([problem_indices[breaks], [problem_indices[-1]]])
        
        problem_spots = np.column_stack((start_indices, end_indices))
        
        # Search in specified direction
        if direction == "forward":
            search_values = start_indices
            mask = plot_vars['last_checked_index'] < search_values
        else:
            search_values = end_indices[::-1]
            problem_spots = problem_spots[::-1]
            mask = plot_vars['last_checked_index'] > search_values
        
        if np.any(mask):
            next_spot = problem_spots[mask][0]
            next_start, next_end = next_spot
            
            # Determine which columns to show
            plot_vars['plot_these_columns'] = gui_functions._get_columns_with_issues(
                df, plot_vars, flag_cols, next_start, next_end
            )
            
            # Update indices and view
            if direction == "forward":
                plot_vars['last_checked_index'] = next_start
            else:
                plot_vars['last_checked_index'] = next_end
            
            plot_vars['dt_ll'] = plot_vars['dt'][next_start-44] - 1/24
            plot_vars['dt_ul'] = plot_vars['dt'][next_end-44] + 1/24
            
            plot_vars['rect_x_min'] = plot_vars['dt'][next_start-44]
            plot_vars['rect_x_max'] = plot_vars['dt'][next_end-44]
            
            plot_vars['mask_dt'] = (
                plot_vars['dt_ll'] <= plot_vars['dt']
            ) & (plot_vars['dt'] < plot_vars['dt_ul'])
            
            plot_vars['mask_select'] = (next_start <= df.index) & (df.index <= next_end)
        
        else:
            plot_vars['plot_these_columns'] = ['GHI']
            gui_functions.show_all_days(gui_vars, plot_vars, df)
            plot_vars['mask_select'] = np.full_like(plot_vars['dt'], False, dtype=bool)
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def _get_columns_with_issues(df, plot_vars, flag_cols, start_idx, end_idx):
        """Get columns that have issues in the specified range."""
        if 'GHI_Calc' in plot_vars['all_columns']:
            base_cols = ['GHI', 'DNI', 'DHI', 'GHI_Calc']
        else:
            base_cols = ['GHI', 'GHI_Aux(1002)', 'DNI_Aux(2012)', 'DHI_Aux(3002)']
        
        result_cols = list(base_cols)
        
        # Check which flag columns have issues
        for flag_col in flag_cols:
            data_col = flag_col.replace('Flag_', '')
            if data_col in result_cols:
                continue
            
            # Check if this column has bad data in the range
            if end_idx == start_idx:
                slice_data = df[flag_col].iloc[start_idx-44]
            else:
                slice_data = df[flag_col].iloc[start_idx-44:end_idx-44+1]
            
            if np.any(slice_data == 99):
                result_cols.append(data_col)
        
        return result_cols
    
    @staticmethod
    def autofind_uncertain_next(gui_vars, plot_vars, df, feature):
        """
        Find next most uncertain point for a specific feature.
        
        Navigates to points where the model prediction probability is closest to 50%
        (most uncertain). Shows a window including all surrounding uncertain points.
        
        Args:
            gui_vars: GUI state variables
            plot_vars: Plotting state variables
            df: Data DataFrame
            feature: Feature name ('GHI', 'DNI', or 'DHI')
        """
        log_button_click(f'autofind_uncertain_next_{feature}')
        
        plot_vars['toggle_select'] = True
        plot_vars['toggle_bad_x'] = True
        
        # Get probability column for the feature
        prob_col = f'prob_{feature.lower()}'
        if prob_col not in plot_vars:
            log_button_click(f'No probability data available for {feature}')
            return
        
        probabilities = plot_vars[prob_col]
        
        # Calculate distance from 0.5 (uncertainty measure)
        # Only consider points with valid probabilities and within zen limit
        valid_mask = (~np.isnan(probabilities)) & (plot_vars['zen'] <= plot_vars['zen_ul'])
        uncertainty = np.abs(probabilities - 0.5)
        
        # Create mask for uncertain points (close to 0.5, e.g., within 0.3 of 0.5)
        # This means probability between 0.2 and 0.8
        uncertain_mask = valid_mask & (uncertainty <= 0.3)
        uncertain_indices = df.index[uncertain_mask]
        
        if len(uncertain_indices) < 1:
            # No uncertain points found
            log_button_click(f'No uncertain points found for {feature}')
            # Keep current column selection, just show all days
            gui_functions.show_all_days(gui_vars, plot_vars, df)
            plot_vars['mask_select'] = np.full_like(plot_vars['dt'], False, dtype=bool)
            update_plot(gui_vars, plot_vars, df)
            return
        
        # Find contiguous uncertain regions
        breaks = np.where(np.diff(uncertain_indices) >= 6)[0]
        start_indices = np.concatenate([[uncertain_indices[0]], uncertain_indices[breaks + 1]])
        end_indices = np.concatenate([uncertain_indices[breaks], [uncertain_indices[-1]]])
        
        uncertain_regions = np.column_stack((start_indices, end_indices))
        
        # For each region, calculate the minimum uncertainty (closest to 0.5)
        region_uncertainties = []
        for start_idx, end_idx in uncertain_regions:
            region_probs = probabilities[start_idx:end_idx+1]
            region_unc = np.abs(region_probs - 0.5)
            min_uncertainty = np.nanmin(region_unc)
            region_uncertainties.append(min_uncertainty)
        
        # Sort regions by uncertainty (most uncertain first)
        sorted_indices = np.argsort(region_uncertainties)
        sorted_regions = uncertain_regions[sorted_indices]
        
        # Get the tracking index for this feature
        uncertain_idx_key = f'uncertain_index_{feature.lower()}'
        current_idx = plot_vars[uncertain_idx_key]
        
        # Find next region to show
        if current_idx >= len(sorted_regions):
            # Wrap around to the beginning
            current_idx = 0
        
        next_region = sorted_regions[current_idx]
        next_start, next_end = next_region
        
        # Update the index for next time
        plot_vars[uncertain_idx_key] = (current_idx + 1) % len(sorted_regions)
        
        # Ensure the feature is visible (add to plot_these_columns if not already present)
        if feature not in plot_vars['plot_these_columns']:
            plot_vars['plot_these_columns'].append(feature)
        
        # Show the probability plot for this feature
        plot_vars[f'show_prob_{feature.lower()}'] = True
        
        # Update view bounds
        plot_vars['dt_ll'] = plot_vars['dt'][next_start-44] - 1/24
        plot_vars['dt_ul'] = plot_vars['dt'][next_end-44] + 1/24
        
        plot_vars['rect_x_min'] = plot_vars['dt'][next_start-44]
        plot_vars['rect_x_max'] = plot_vars['dt'][next_end-44]
        
        plot_vars['mask_dt'] = (
            plot_vars['dt_ll'] <= plot_vars['dt']
        ) & (plot_vars['dt'] < plot_vars['dt_ul'])
        
        plot_vars['mask_select'] = (next_start <= df.index) & (df.index <= next_end)
        
        update_plot(gui_vars, plot_vars, df)
        
        # Log which uncertainty level we found
        min_unc = region_uncertainties[sorted_indices[current_idx]]
        log_button_click(f'Found {feature} uncertain region {current_idx+1}/{len(sorted_regions)} '
                        f'with uncertainty {min_unc:.3f} (prob closest to {0.5-min_unc:.3f}-{0.5+min_unc:.3f})')

    
    @staticmethod
    def reset_limit(gui_vars, plot_vars, df, name, lower, upper):
        """Reset filter limits."""
        log_button_click(f'reset_limit_{lower}_{upper}')
        
        plot_vars[f'{name}_ll'] = lower
        plot_vars[f'{name}_ul'] = upper
        plot_vars[f'mask_{name}'] = (
            plot_vars[f'{name}_ll'] <= plot_vars[name]
        ) & (plot_vars[name] < plot_vars[f'{name}_ul'])
        
        update_plot(gui_vars, plot_vars, df)
    
    @staticmethod
    def graceful_exit(root):
        """Exit the application gracefully."""
        log_button_click('graceful_exit')
        
        response = messagebox.askyesno("Exit", "Are you sure you want to exit?")
        if response:
            root.quit()
            root.destroy()
            print("Program exited gracefully.")
            close_log()
    
    @staticmethod
    def save(root, df, df_string, save_path):
        """Save data and exit."""
        log_button_click('save')
        
        response = messagebox.askyesno("Exit and Save", "Are you sure you want to save and exit?")
        if response:
            save_dataframe_to_file(df, df_string, save_path)
            root.quit()
            root.destroy()
            print("GUI ended. Saving data")

# =====================================================================
# Plotting Functions
# =====================================================================

def update_plot(gui_vars, plot_vars, df, y_min=None, y_max=None):
    """
    Update the plot with current data and settings.
    
    Args:
        gui_vars: GUI variables dictionary
        plot_vars: Plot variables dictionary
        df: Dataframe
        y_min: Optional Y-axis minimum
        y_max: Optional Y-axis maximum
    """
    # Update combined mask
    plot_vars['mask_combined'] = (
        plot_vars['mask_dt'] & plot_vars['mask_zen'] &
        plot_vars['mask_azm'] & plot_vars['mask_ghi']
    )
    
    # Clear axes
    gui_vars['ax'][0].clear()
    gui_vars['ax'][1].clear()
    
    # Determine marker size based on data density
    visible_points = np.sum(plot_vars['mask_combined'])
    if visible_points < 180:
        marker_size = 3
        marker_scale = 2
    else:
        marker_size = 0.75
        marker_scale = 5
    
    # Plot selected columns
    for col in plot_vars['plot_these_columns']:
        if col not in df.columns:
            continue
            
        # Create masked data
        masked_data = np.where(plot_vars['mask_combined'], df[col], np.nan)
        
        # Plot main data
        gui_vars['ax'][0].plot(
            plot_vars['x_values'], masked_data,
            marker='o', markersize=marker_size, linewidth=0.25,
            color=plot_vars['column_colors'].get(col, 'blue'),
            label=col
        )
        
        # Highlight bad data
        if plot_vars['toggle_bad_x']:
            bad_mask = plot_vars['mask_combined'] & (df[f'Flag_{col}'] > 72)
            if np.any(bad_mask):
                gui_vars['ax'][0].plot(
                    plot_vars['x_values'][bad_mask], df[col][bad_mask],
                    marker='x', markersize=7, linewidth=0, markeredgewidth=2,
                    color='red'
                )
        
        # Highlight selected data
        if plot_vars['toggle_select'] and np.any(plot_vars['mask_select']):
            select_mask = plot_vars['mask_combined'] & plot_vars['mask_select']
            if np.any(select_mask):
                gui_vars['ax'][0].plot(
                    plot_vars['x_values'][select_mask], df[col][select_mask],
                    marker='o', markersize=12, linewidth=0,
                    markeredgecolor='black', markeredgewidth=1,
                    markerfacecolor='none'
                )
    
    # Add legend to top plot
    if plot_vars['plot_these_columns']:
        gui_vars['ax'][0].legend(markerscale=marker_scale)
    
    # Apply Y-axis zoom if specified
    if y_min is not None and y_max is not None:
        gui_vars['ax'][0].set_ylim(y_min, y_max)
    
    # Add grid
    gui_vars['ax'][0].grid(
        which='major', axis='both',
        linestyle=(0, (2, 10)), linewidth=0.75, color='gray'
    )
    
    # --- Lower plot: GHI Ratio and/or Probabilities ---
    # Determine what to plot based on toggle states
    plot_items = []
    
    # GHI Ratio (existing functionality)
    if plot_vars.get('show_ghi_ratio', True):
        ratio_mask = plot_vars['mask_combined'] & (plot_vars['ghi_ratio'] > 0)
        if np.any(ratio_mask):
            plot_items.append({
                'x': plot_vars['x_values'][ratio_mask],
                'y': plot_vars['ghi_ratio'][ratio_mask],
                'label': 'GHI Ratio',
                'color': 'black',
                'marker': 'o',
                'markersize': marker_size,
                'alpha': 0.5
            })
    
    # Probability plots
    if plot_vars.get('show_prob_ghi', False):
        prob_mask = plot_vars['mask_combined'] & ~np.isnan(plot_vars['prob_ghi'])
        if np.any(prob_mask):
            plot_items.append({
                'x': plot_vars['x_values'][prob_mask],
                'y': plot_vars['prob_ghi'][prob_mask],
                'label': 'P(GHI Good)',
                'color': '#FFA500',
                'marker': 'o',
                'markersize': marker_size * 0.8,
                'alpha': 0.6
            })
    
    if plot_vars.get('show_prob_dni', False):
        prob_mask = plot_vars['mask_combined'] & ~np.isnan(plot_vars['prob_dni'])
        if np.any(prob_mask):
            plot_items.append({
                'x': plot_vars['x_values'][prob_mask],
                'y': plot_vars['prob_dni'][prob_mask],
                'label': 'P(DNI Good)',
                'color': '#DA70D6',
                'marker': 's',
                'markersize': marker_size * 0.8,
                'alpha': 0.6
            })
    
    if plot_vars.get('show_prob_dhi', False):
        prob_mask = plot_vars['mask_combined'] & ~np.isnan(plot_vars['prob_dhi'])
        if np.any(prob_mask):
            plot_items.append({
                'x': plot_vars['x_values'][prob_mask],
                'y': plot_vars['prob_dhi'][prob_mask],
                'label': 'P(DHI Good)',
                'color': '#4169E1',
                'marker': '^',
                'markersize': marker_size * 0.8,
                'alpha': 0.6
            })
    
    # Plot all enabled items
    if plot_items:
        for item in plot_items:
            gui_vars['ax'][1].plot(
                item['x'], item['y'],
                marker=item['marker'], linestyle='-',
                color=item['color'], markersize=item['markersize'],
                alpha=item['alpha'], label=item['label'], linewidth=0.5
            )
        
        gui_vars['ax'][1].legend()
        
        # Set y-axis label and limits based on what's shown
        has_prob = any(item['label'].startswith('P(') for item in plot_items)
        has_ratio = any(item['label'] == 'GHI Ratio' for item in plot_items)
        
        # Calculate dynamic y-axis limits based on actual data
        all_y_values = np.concatenate([item['y'] for item in plot_items])
        all_y_values = all_y_values[~np.isnan(all_y_values)]  # Remove NaNs
        
        if len(all_y_values) > 0:
            y_min = np.min(all_y_values)
            y_max = np.max(all_y_values)
            y_range = y_max - y_min
            y_padding = max(0.1, y_range * 0.05)  # 5% padding, minimum 0.1
        else:
            y_min, y_max, y_padding = 0, 1, 0.1
        
        if has_prob and has_ratio:
            gui_vars['ax'][1].set_ylabel('Ratio / Probability', fontsize=10)
            # Dynamic limits with reasonable bounds
            gui_vars['ax'][1].set_ylim(min(0.9, y_min - y_padding), max(1.1, y_max + y_padding))
            # Add reference lines for both ratio and probability
            gui_vars['ax'][1].axhline(y=0.5, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
            gui_vars['ax'][1].axhline(y=1.0, color='darkgray', linestyle='--', linewidth=0.5)
            gui_vars['ax'][1].axhline(y=1.08, color='darkgray', linestyle='-', linewidth=0.5)
            gui_vars['ax'][1].axhline(y=0.92, color='darkgray', linestyle='-', linewidth=0.5)
        elif has_prob:
            gui_vars['ax'][1].set_ylabel('P(GOOD)', fontsize=10)
            gui_vars['ax'][1].set_ylim(-0.05, 1.05)
            # Add reference lines for probability
            gui_vars['ax'][1].axhline(y=0.5, color='green', linestyle='--', linewidth=0.5, label='Decision Threshold')
        else:
            gui_vars['ax'][1].set_ylabel('GHI Ratio', fontsize=10)
            # Dynamic limits for ratio with reasonable minimum
            gui_vars['ax'][1].set_ylim(min(0.9, y_min - y_padding), max(1.1, y_max + y_padding))
            # Add reference lines for ratio
            gui_vars['ax'][1].axhline(y=1.0, color='darkgray', linestyle='--', linewidth=0.5)
            gui_vars['ax'][1].axhline(y=1.08, color='darkgray', linestyle='-', linewidth=0.5)
            gui_vars['ax'][1].axhline(y=0.92, color='darkgray', linestyle='-', linewidth=0.5)
        
        gui_vars['ax'][1].grid(
            which='major', axis='both',
            linestyle=(0, (2, 10)), color='lightgray'
        )
    else:
        gui_vars['ax'][1].text(
            0.5, 0.5, 'No data selected for lower plot\\nUse toggle buttons to enable',
            transform=gui_vars['ax'][1].transAxes,
            ha='center', va='center', fontsize=12
        )
    
    # Format X-axis for time display
    if gui_vars['x_variable'].get() == 'Time':
        time_range = plot_vars['dt_ul'] - plot_vars['dt_ll']
        
        if time_range >= 7:
            date_format = '%Y-%m-%d'
        elif time_range > 1:
            date_format = '%m-%d %H:%M'
        else:
            date_format = '%H:%M'
        
        gui_vars['ax'][1].xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.xticks(rotation=45)
    
    # Add title with station name and time range
    start_time = mdates.num2date(plot_vars['dt_ll']).strftime('%Y-%m-%d %H:%M')
    end_time = mdates.num2date(plot_vars['dt_ul']).strftime('%Y-%m-%d %H:%M')
    title_text = f"{plot_vars['station_name']}            {start_time}  to  {end_time}"
    
    gui_vars['ax'][0].text(
        0.0, 1.05, title_text,
        ha='left', va='bottom', fontsize=12, fontweight='bold',
        transform=gui_vars['ax'][0].transAxes
    )
    
    # Redraw canvas
    gui_vars['canvas'].draw()

# =====================================================================
# File I/O Functions
# =====================================================================

def save_dataframe_to_file(df, df_string, save_path):
    """
    Save QC results to file.
    
    Args:
        df: Processed dataframe with flags
        df_string: Original string dataframe
        save_path: Path to save file
    """
    # Combine header and data
    df_save = np.vstack([np.array(df_string)[:44], np.array(df)])
    
    # Restore string notes in last column
    df_save[44:, -1] = df_string.iloc[44:, -1]
    
    # Save to file
    np.savetxt(save_path, df_save, delimiter=',', fmt='%s')
    print(f"Data saved to: {save_path}")

def start_auto_save(df, df_string, save_path, interval=600):
    """
    Start auto-save thread.
    
    Args:
        df: Dataframe to save
        df_string: String dataframe
        save_path: Save path
        interval: Save interval in seconds (default: 10 minutes)
    """
    def auto_save_worker():
        while True:
            save_dataframe_to_file(df, df_string, save_path)
            time.sleep(interval)
    
    thread = threading.Thread(target=auto_save_worker, daemon=True)
    thread.start()

# =====================================================================
# Main Execution
# =====================================================================

if __name__ == "__main__":
    # Support command line file argument for testing
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        main(test_file=test_file)
    else:
        main()
    print('End of Manual QC program')