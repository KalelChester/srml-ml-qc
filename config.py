"""
Project-wide configuration values.

Update SITE_CONFIG to match the data collection site.
"""

SITE_CONFIG = {
    'latitude': 47.654,
    'longitude': -122.309,
    'altitude': 70,
    'timezone': 'Etc/GMT+8'
}

# Target-specific decision thresholds on P(GOOD).
# Lower values make BAD prediction harder (more conservative), which reduces false positives.
PREDICTION_THRESHOLDS = {
    'Flag_GHI': 0.50,
    'Flag_DNI': 0.50,
    'Flag_DHI': 0.50,
}

# Target-specific class-weight multipliers applied after automatic class balancing.
# Class convention in training: 0=BAD, 1=GOOD.
# Conservative setting: downweight BAD emphasis to reduce false positives.
TARGET_CLASS_WEIGHT_MULTIPLIERS = {
    'Flag_GHI': {'bad': 0.80, 'good': 1.00},
    'Flag_DNI': {'bad': 0.65, 'good': 1.00},
    'Flag_DHI': {'bad': 0.65, 'good': 1.00},
}

# No-retrain ensemble control: blend RF probability with NN stacked probability.
# final_prob = (1 - rf_blend) * nn_prob + rf_blend * rf_prob
TARGET_RF_BLEND_WEIGHTS = {
    'Flag_GHI': 0.70,
    'Flag_DNI': 0.70,
    'Flag_DHI': 0.70,
}

# Target-specific fit parameters used by run_learning_cycle during retraining.
TARGET_FIT_PARAMS = {
    # max_bad_fraction caps BAD prevalence in per-target fit data.
    # max_bad_weight caps the final BAD class weight used by RF/NN.
    'Flag_GHI': {'epochs': 14, 'batch_size': 64, 'upsample_min_bad': 1200, 'synthetic_frac': 0.03, 'max_bad_fraction': 0.08, 'max_bad_weight': 3.0},
    'Flag_DNI': {'epochs': 18, 'batch_size': 64, 'upsample_min_bad': 2200, 'synthetic_frac': 0.08, 'max_bad_fraction': 0.06, 'max_bad_weight': 2.5},
    'Flag_DHI': {'epochs': 18, 'batch_size': 64, 'upsample_min_bad': 2200, 'synthetic_frac': 0.08, 'max_bad_fraction': 0.06, 'max_bad_weight': 2.5},
}
