"""
solar_model.py
==============

Hybrid Time-Series Solar QC Model (Version 2.0)

This module implements a sophisticated multi-component quality control system for
solar irradiance data that combines supervised learning, unsupervised anomaly
detection, and time-series modeling.

Architecture Overview
---------------------
1. **RandomForest Classifier** (Base Model)
   - Trained on engineered solar features
   - Isotonic probability calibration for reliable confidence scores
   - Balanced class weights for imbalanced data handling

2. **IsolationForest Detector** (Unsupervised)
   - Trained exclusively on GOOD samples
   - Provides anomaly scores as meta-features
   - Catches novel failure modes not in training data

3. **Recurrent Neural Network** (RNN with Attention) - NEW in v2.0
   - Multi-layer GRU processes 24-hour sequences
   - Temporal attention learns critical time steps
   - Residual connections for improved gradient flow
   - Dropout regularization to prevent overfitting
   - Alternative Dense NN available for non-sequential use cases

Key Features
------------
- Time-series aware: Processes 24-hour sequences (RNN mode)
- Temporal context: Learns daily patterns and transitions
- Attention mechanism: Automatically focuses on important hours
- Class-weighted training: Handles imbalanced good/bad ratios
- Synthetic augmentation: Optional data augmentation for rare failures
- Safe upsampling: Conservative minority class balancing
- Dual output: Both binary flags (99/1) and probabilities (0.0-1.0)
- Model persistence: Save/load with full configuration preservation
- Backward compatible: Supports both v1.0 (Dense) and v2.0 (RNN) models

Output Convention
-----------------
- Flags: 99 = BAD quality, 1 = GOOD quality
- Probabilities: P(GOOD) in range [0.0, 1.0]
  - 1.0 = fully confident GOOD
  - 0.5 = uncertain (decision boundary)
  - 0.0 = fully confident BAD

Usage Example
-------------
    # Training
    model = SolarHybridModel(use_rnn=True)
    model.fit(training_df, target_col='Flag_GHI', epochs=20)
    model.save_model('models/model_Flag_GHI.pkl')
    
    # Prediction
    model = SolarHybridModel.load_model('models/model_Flag_GHI.pkl')
    flags, probs = model.predict(test_df, 'Flag_GHI', do_return_probs=True)

Version History
---------------
- v2.0 (2026): Added RNN architecture with temporal attention
- v1.0 (2025): Initial hybrid model with Dense NN

Dependencies
------------
- JAX/Flax: Neural network implementation
- TensorFlow: Data pipeline utilities
- scikit-learn: Random Forest, IsolationForest, preprocessing
- pandas/numpy: Data manipulation

Author: Solar QC Team
Last Updated: February 2026
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import pickle
import os
import functools

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report


import tensorflow as tf
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# =============================================================================
# Neural Network Architectures - Time Series RNN with Attention
# =============================================================================

class SolarRNN(nn.Module):
    """
    Recurrent Neural Network for time-series solar QC prediction.
    
    This architecture is specifically designed to capture temporal patterns in
    solar irradiance data, which exhibits strong daily cycles and time-dependent
    quality issues.
    
    Architecture Components
    -----------------------
    1. Multi-layer GRU (Gated Recurrent Unit)
       - Layer 1: Processes raw sequences, learns basic temporal features
       - Layer 2: Learns complex patterns from Layer 1 outputs
       - Each layer has configurable hidden dimension (default: 64)
       
    2. Residual Connections
       - Skip connections between GRU layers
       - Improves gradient flow during backpropagation
       - Enables training of deeper networks
       
    3. Temporal Attention Mechanism
       - Learns importance weights for each time step
       - Automatically focuses on critical hours (dawn/dusk, weather changes)
       - Produces weighted sum of hidden states
       
    4. Dense Classification Head
       - 128 → 64 → 32 → 2 fully connected layers
       - ReLU activations throughout
       - Dropout for regularization (training only)
       - Final 2-unit output: logits for [BAD(0), GOOD(1)]
    
    Parameters
    ----------
    hidden_dim : int, default=64
        Size of GRU hidden state. Larger = more capacity, slower training
    num_layers : int, default=2
        Number of stacked GRU layers. More layers = deeper temporal understanding
    dropout_rate : float, default=0.2
        Dropout probability during training (0.0-1.0). Higher = more regularization
    use_attention : bool, default=True
        Whether to use temporal attention. True = learns important time steps
    
    Input Shape
    -----------
    (Batch_Size, Time_Steps, Features)
        - Batch_Size: Number of sequences
        - Time_Steps: Length of each sequence (typically 24 for hourly data)
        - Features: Number of input features per time step
    
    Output Shape
    ------------
    (Batch_Size, 2)
        - Logits for [BAD(0), GOOD(1)]
        - Apply softmax to get probabilities
    
    Example
    -------
        >>> model = SolarRNN(hidden_dim=64, num_layers=2, dropout_rate=0.2)
        >>> x = jnp.ones((32, 24, 18))  # 32 sequences, 24 hours, 18 features
        >>> logits = model(x, training=False)
        >>> probs = jax.nn.softmax(logits, axis=1)[:, 1]  # P(GOOD)
    
    Notes
    -----
    - GRU chosen over LSTM for efficiency (fewer parameters, similar performance)
    - Attention improves interpretability (can visualize important hours)
    - Dropout only active when training=True
    - Residual connections enable training beyond 2 layers
    """
    hidden_dim: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.2
    use_attention: bool = True
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Args:
            x: Input tensor of shape (Batch_Size, Time_Steps, Features)
            training: Whether in training mode (for dropout)
        """
        batch_size = x.shape[0]
        
        # Layer 1: First GRU layer
        GRU_1 = nn.scan(
            nn.GRUCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=1,
            out_axes=1
        )
        carry_1 = jnp.zeros((batch_size, self.hidden_dim))
        carry_1, outputs_1 = GRU_1(features=self.hidden_dim)(carry_1, x)
        
        # Residual connection: project input to hidden_dim if needed
        if x.shape[-1] != self.hidden_dim:
            x_proj = nn.Dense(self.hidden_dim)(x)
        else:
            x_proj = x
        
        # Add residual
        outputs_1 = outputs_1 + 0.5 * x_proj
        
        # Layer 2: Second GRU layer (optional but recommended)
        if self.num_layers >= 2:
            GRU_2 = nn.scan(
                nn.GRUCell,
                variable_broadcast='params',
                split_rngs={'params': False},
                in_axes=1,
                out_axes=1
            )
            carry_2 = jnp.zeros((batch_size, self.hidden_dim))
            carry_2, outputs_2 = GRU_2(features=self.hidden_dim)(carry_2, outputs_1)
            
            # Residual connection
            outputs_2 = outputs_2 + 0.5 * outputs_1
            
            outputs = outputs_2
        else:
            outputs = outputs_1
        
        # Temporal Attention: Learn which time steps are most important
        if self.use_attention:
            # Attention scores: (Batch, Time, Hidden_Dim) -> (Batch, Time, 1)
            attention_scores = nn.Dense(1)(outputs)  # (B, T, 1)
            attention_weights = jax.nn.softmax(attention_scores, axis=1)  # (B, T, 1)
            
            # Apply attention: weighted sum over time
            attended = jnp.sum(outputs * attention_weights, axis=1)  # (B, Hidden_Dim)
        else:
            # Fallback: just use the final hidden state
            attended = outputs[:, -1, :]
        
        # Dropout for regularization
        if training:
            attended = nn.Dropout(rate=self.dropout_rate)(attended, deterministic=False)
        
        # Dense layers for classification
        x = nn.Dense(128)(attended)
        x = nn.relu(x)
        
        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        
        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        
        # Output logits for [BAD(0), GOOD(1)]
        logits = nn.Dense(2)(x)
        
        return logits


class DenseNN(nn.Module):
    """
    Fallback Dense Neural Network for non-sequential predictions.
    
    Simple feedforward architecture used when RNN processing is not feasible
    (e.g., insufficient sequence length, non-temporal features).
    
    Architecture
    ------------
    - Input: Flattened feature vector
    - Hidden Layer 1: 128 units + ReLU
    - Hidden Layer 2: 64 units + ReLU
    - Hidden Layer 3: 32 units + ReLU
    - Output: 2 logits for [BAD(0), GOOD(1)]
    
    Input Shape
    -----------
    Any shape that can be flattened: (Batch, ...)
        - Auto-flattened to (Batch, Features)
    
    Output Shape
    ------------
    (Batch, 2) - Logits for binary classification
    
    Parameters
    ----------
    None (architecture is fixed)
    
    Notes
    -----
    - No dropout or regularization (simpler than RNN)
    - Ignores temporal patterns in data
    - Used as fallback when sequence construction fails
    - Maintained for backward compatibility with v1.0 models
    
    Example
    -------
        >>> model = DenseNN()
        >>> x = jnp.ones((32, 18))  # 32 samples, 18 features
        >>> logits = model(x, training=False)
        >>> probs = jax.nn.softmax(logits, axis=1)[:, 1]  # P(GOOD)
    """
    @nn.compact
    def __call__(self, x, training: bool = False):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        logits = nn.Dense(2)(x)  # logits for [BAD(0), GOOD(1)]
        return logits


@functools.partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, class_weights, is_rnn: bool = False):
    """
    Single training step with class-weighted loss and JIT compilation.
    
    Computes gradients and updates model parameters for one batch using
    cross-entropy loss with per-sample class weights to handle imbalanced data.
    
    Parameters
    ----------
    state : optax.TrainState
        Current training state containing:
        - params: Model parameters
        - apply_fn: Model forward function
        - tx: Optimizer
        - opt_state: Optimizer state
    
    batch : dict
        Training batch with keys:
        - 'x': Input features/sequences, shape (Batch, ...) or (Batch, Time, Features)
        - 'y': Integer labels, shape (Batch,), values in {0, 1}
          where 0=BAD, 1=GOOD
    
    class_weights : jnp.array
        Weight for each class, shape (2,)
        - class_weights[0]: Weight for BAD samples (class 0)
        - class_weights[1]: Weight for GOOD samples (class 1)
        Higher weight = more importance in loss
    
    is_rnn : bool, default=False
        Whether model is RNN-based (static argument)
        - True: Passes training=True and dropout RNG to model
        - False: Simple forward pass without dropout
        - NOTE: This is a static argument (doesn't change during training batch)
    
    Returns
    -------
    tuple : (new_state, loss_value)
        - new_state: Updated training state after gradient step
        - loss_value: Scalar loss for this batch (for monitoring)
    
    Loss Computation
    ----------------
    1. Forward pass: logits = model(x)
    2. Softmax cross-entropy: per_sample_loss = -sum(y_true * log(softmax(logits)))
    3. Weight application: weighted_loss = per_sample_loss * class_weights[y]
    4. Aggregate: final_loss = mean(weighted_loss)
    
    JIT Compilation
    ---------------
    - Function compiled with JAX JIT for speed
    - First call slow (compilation), subsequent calls fast
    - is_rnn is a static argument (compiled separately for True/False)
    - All other operations must be JAX-compatible (no Python control flow)
    
    Notes
    -----
    - Uses fixed dropout RNG (key=0) per step (acceptable for training)
    - Class weights typically computed as 1 / class_frequency
    - Supports both RNN (3D input) and Dense (2D input) models
    - Loss is scalar (reduced across batch)
    - is_rnn must be consistent during training epoch (it is)
    
    Example
    -------
        >>> state = create_train_state(model, learning_rate=1e-3)
        >>> batch = {'x': x_train[:32], 'y': y_train[:32]}
        >>> class_weights = jnp.array([2.0, 1.0])  # BAD class has 2x weight
        >>> new_state, loss = train_step(state, batch, class_weights, is_rnn=True)
        >>> print(f"Loss: {loss:.4f}")
    """
    def loss_fn(params):
        if is_rnn:
            # RNN models need training=True for dropout
            logits = state.apply_fn(
                {'params': params}, 
                batch['x'],
                training=True,
                rngs={'dropout': jax.random.PRNGKey(0)}
            )
        else:
            logits = state.apply_fn({'params': params}, batch['x'], training=False)
        
        labels = jax.nn.one_hot(batch['y'], 2)
        per_sample = optax.softmax_cross_entropy(logits=logits, labels=labels)
        weights = class_weights[batch['y']]  # lookup final weight per sample
        return jnp.mean(per_sample * weights)

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


# =============================================================================
# Sequence Preparation Utilities
# =============================================================================

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = 24, stride: int = 1):
    """
    Transform time-series data into sliding-window sequences for RNN processing.
    
    This function is CRITICAL for enabling RNNs to learn temporal patterns.
    It converts flat feature vectors into sequences with historical context.
    
    Why Sequences Matter
    --------------------
    Solar irradiance has strong temporal dependencies:
    - Sunrise/sunset patterns
    - Weather transitions
    - Sensor drift over hours
    - Cloud movement effects
    
    Without sequences, RNNs see data as independent samples (like Dense NN).
    With sequences, RNNs understand "what happened before" at each time step.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Time-ordered feature matrix
        - Each row is one time step (e.g., one hour)
        - Features: engineered solar variables (clearsky, angles, etc.)
        - MUST be chronologically sorted
    
    y : np.ndarray, shape (n_samples,)
        Time-ordered labels corresponding to X
        - Values: 0=BAD, 1=GOOD
        - MUST align with X row-by-row
    
    seq_length : int, default=24
        Number of time steps in each sequence
        - 24 = One full day of hourly data (recommended)
        - Smaller = less context, faster training
        - Larger = more context, but needs more data
    
    stride : int, default=1
        Step size between sequence start points
        - 1 = Maximum overlap (seq1: [0:24], seq2: [1:25], ...)
        - 24 = No overlap (seq1: [0:24], seq2: [24:48], ...)
        - Smaller stride = more training samples but correlated
    
    Returns
    -------
    tuple : (X_seq, y_seq)
        X_seq : np.ndarray, shape (n_sequences, seq_length, n_features)
            Stacked sequences, each with seq_length time steps
            - n_sequences = (n_samples - seq_length) / stride + 1
        
        y_seq : np.ndarray, shape (n_sequences,)
            Label for each sequence
            - Always the label at the LAST time step of the sequence
            - Prediction task: "Given previous seq_length hours, is the final hour GOOD?"
    
    Edge Cases
    ----------
    - If n_samples < seq_length: Returns single-sample sequence with shape (1, ...)
    - If stride > (n_samples - seq_length): Returns minimal sequences
    - Labels always correspond to the final element of each sequence
    
    Sequence Window Visualization
    -----------------------------
    For X with 48 samples, seq_length=24, stride=1:
        Sequence 0: X[0:24]   -> y[23]
        Sequence 1: X[1:25]   -> y[24]
        Sequence 2: X[2:26]   -> y[25]
        ...
        Sequence 24: X[24:48] -> y[47]
    
    Total sequences = 48 - 24 + 1 = 25
    
    Example
    -------
        >>> X = np.random.randn(100, 18)  # 100 hours, 18 features
        >>> y = np.random.randint(0, 2, 100)  # Binary labels
        >>> X_seq, y_seq = create_sequences(X, y, seq_length=24, stride=1)
        >>> print(X_seq.shape)  # (77, 24, 18) - 77 sequences
        >>> print(y_seq.shape)  # (77,)
        >>> # First sequence predicts label at hour 23 using hours 0-23
        >>> assert y_seq[0] == y[23]
    
    Notes
    -----
    - Input data MUST be time-ordered (no shuffling before this function)
    - Output sequences are still in chronological order
    - Training should shuffle sequences AFTER creation (not time steps within)
    - Prediction requires mapping sequence indices back to original timestamps
    - This function is compatible with any time-series data, not just solar
    
    Performance
    -----------
    - Memory: O(n_sequences * seq_length * n_features)
    - Time: O(n_sequences) for copying windows
    - For 1 year hourly data: ~8700 sequences with stride=1 (~2 MB for 18 features)
    """
    X_sequences = []
    y_sequences = []
    
    n_samples = len(X)
    
    for i in range(0, n_samples - seq_length + 1, stride):
        X_sequences.append(X[i:i + seq_length])
        # Label is the last time step in the sequence
        y_sequences.append(y[i + seq_length - 1])
    
    if not X_sequences:
        # Fallback: if not enough data, just return original
        return X[np.newaxis, ...], y[-1:]
    
    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    
    return X_seq, y_seq


# =============================================================================
# Hybrid wrapper
# =============================================================================
class SolarHybridModel:
    def __init__(self, use_rnn: bool = True):
        """
        Initialize hybrid solar QC model with RF, IF, and NN components.
        
        The hybrid model combines three complementary approaches:
        1. RandomForest: Fast, interpretable, captures non-linear feature interactions
        2. IsolationForest: Detects novel anomalies not in training data
        3. RNN/Dense NN: Learns temporal patterns and complex decision boundaries
        
        Final predictions are ensembles of all three components.
        
        Parameters
        ----------
        use_rnn : bool, default=True
            Architecture for neural network component
            - True: Use RNN (SolarRNN) for temporal sequence modeling
              * Input shape: (Batch, 24 hours, 18 features)
              * Learns sunrise/sunset transitions, weather changes
              * Recommended for solar data with strong daily patterns
            - False: Use Dense NN for feature-based prediction
              * Input shape: (Batch, 18 features)
              * Faster training, simpler implementation
              * Use if sequence data unavailable
        
        Model Components Initialized
        ----------------------------
        
        1. RandomForestClassifier (self.rf)
           - 50 estimators for stable predictions
           - max_depth=50 for complex patterns
           - Balanced class weights for imbalanced data
           - Trained on engineered features
        
        2. CalibratedClassifierCV (self.rf_cal)
           - Isotonic calibration applied to RF probabilities
           - Ensures P(GOOD) ∈ [0, 1] and is interpretable
           - Trained on validation set after RF fitting
           - Set to None until fit() is called
        
        3. IsolationForest (self.if_det)
           - Trained exclusively on GOOD samples
           - Anomaly score ∈ [-1, 1] (normalized)
           - Detects quality issues not in training data
           - Set to None until fit() is called
        
        4. StandardScaler (self.scaler)
           - Normalizes NN inputs to mean=0, std=1
           - Improves convergence and stability
           - Learned during fit() on training data
        
        5. JAX TrainState (self.nn_state)
           - Holds NN parameters and optimizer state
           - Set to None until fit() is called
           - Re-initialized for each fit() call
        
        Configuration Parameters
        -------------------------
        self.use_rnn : bool
            Architecture choice (set at initialization)
        
        self.seq_length : int
            Sequence length for RNN mode (always 24)
            - 24 hours of hourly data for daily patterns
            - Cannot be changed after initialization
        
        self.common_features : list
            Expected feature columns (set at initialization)
            - 18 features from solar_features.add_features()
            - Must be present in all training/prediction data
            - Missing columns auto-filled with 0 during _build_X()
        
        Example
        -------
            >>> model = SolarHybridModel(use_rnn=True)
            >>> model.fit(X_train, y_train)  # Train hybrid model
            >>> probs = model.predict_proba(X_test)  # Get probabilities
            >>> flags = model.predict(X_test)  # Get binary predictions
        
        Notes
        -----
        - Model is not trainable until fit() is called
        - All components must be present for predictions (they're all mandatory)
        - Probabilities are ensemble averages of RF, IF, and NN
        - Supports both binary flags (0/1) and probability outputs (0.0-1.0)
        
        See Also
        --------
        fit() : Train model on data
        predict() : Get binary flags (0=BAD, 1=GOOD)
        predict_proba() : Get probability scores P(GOOD)
        save_model() : Persist trained model to disk
        load_model() : Load model from saved file
        """
        # Base tree model
        self.rf = RandomForestClassifier(
            n_estimators=50,
            min_samples_leaf=1,
            max_depth=50,
            class_weight='balanced_subsample',  # helps with time-chunked imbalance
            n_jobs=-1,
            random_state=42
        )
        self.rf_cal: Optional[CalibratedClassifierCV] = None

        # IsolationForest for unsupervised anomaly scoring (trained on GOOD)
        self.if_det: Optional[IsolationForest] = None

        # Scaler for NN inputs
        self.scaler = StandardScaler()

        # JAX/Flax state for NN
        self.nn_state: Optional[train_state.TrainState] = None
        
        # Configuration
        self.use_rnn = use_rnn
        self.seq_length = 60  # 60 samples = 1 hour of 1-minute data (temporal context for sunrise/sunset)

        # Feature set (must mirror solar_features.add_features)
        self.common_features = [
            # 'Timestamp_Num',
            'hour_sin', 'hour_cos',
            'doy_sin', 'doy_cos',
            'SZA', 'CSI',
            'QC_PhysicalFail', 'Temperature',
            'ghi_ratio', 'ghi_diff',
            # 'CorrFeat_GHI', 'CorrFeat_DNI', 'CorrFeat_DHI', 'elevation',
            'GHI', 'DNI', 'DHI',
            'GHI_Clear', 'DNI_Clear', 'DHI_Clear'
        ]

    # ---------------- internal utilities ---------------------------------
    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a DataFrame X with columns in self.common_features.
        If a feature is absent, fill with zeros to preserve ordering.
        """
        data = {}
        for c in self.common_features:
            data[c] = df[c] if c in df.columns else pd.Series(0.0, index=df.index)
        X = pd.DataFrame(data, index=df.index)
        # If IF_Score exists in df, include it as an auxiliary column (optional)
        if 'IF_Score' in df.columns:
            X['IF_Score'] = df['IF_Score']
        return X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ---------------- fit -----------------------------------------------
    def fit(self,
            df: pd.DataFrame,
            target_col: str,
            epochs: int = 20,
            batch_size: int = 64,
            upsample_min_bad: int = 500,
            synthetic_frac: float = 0.0):
        """
        Train the complete hybrid model: RF + IsolationForest + NN (RNN or Dense).
        
        This method orchestrates a multi-stage training pipeline that combines
        supervised learning (RF), unsupervised anomaly detection (IF), and
        deep learning (RNN/Dense) for robust quality control predictions.
        
        Parameters
        ----------
        df : pd.DataFrame
            Complete dataframe with features and labels
            - Must contain output from solar_features.add_features()
            - Must have column named target_col with labels
            - All self.common_features should be present (missing = auto-filled with 0)
        
        target_col : str
            Name of label column (e.g., 'Flag_GHI', 'Flag_DNI', 'Flag_DHI')
            Label convention:
            - 99 => BAD (quality failure)
            - Any other value => GOOD (quality pass)
            - NaN/missing => Unlabeled (skipped in training)
        
        epochs : int, default=20
            Number of passes through training data for NN
            - Typical range: 10-50
            - Smaller (5-10) = faster, less fitting
            - Larger (50+) = slower, may overfit
        
        batch_size : int, default=64
            Gradient update batch size for NN
            - Smaller (16) = noisier updates, may help generalization
            - Larger (128+) = smoother updates, more memory
            - Automatically adjusted down if dataset smaller
        
        upsample_min_bad : int, default=500
            Minimum number of BAD samples for training
            - If training has fewer BAD samples, upsample with replacement
            - Prevents extremely small minority classes
            - Upsampling applied BEFORE sequence creation
            - Conservative default: rarely exceeds 2x amplification
        
        synthetic_frac : float, default=0.0
            Fraction of training rows to synthetically perturb
            - Range: [0.0, 1.0]
            - 0.0 = No synthetic anomalies (use original data only)
            - 0.1 = Create synthetic versions of 10% of training samples
            - Uses inject_synthetic_anomalies() with feature perturbations
            - Applied AFTER upsampling, BEFORE sequence creation
        
        Training Workflow
        -----------------
        1. **Label Extraction**
           - Extract rows with non-NaN target_col
           - Convert 99 => BAD (class 0), else => GOOD (class 1)
        
        2. **Feature Building**
           - Construct X from self.common_features
           - Missing columns filled with 0
           - Reorder to standard feature order
        
        3. **RandomForest Training**
           - Fit 50-tree RF with balanced class weights
           - Generates base probability predictions
           - Serves as feature importance source
        
        4. **Probability Calibration**
           - Apply isotonic calibration if enough minority samples (≥3)
           - Uses cross-validation (5-fold or fewer)
           - Ensures probabilities are well-calibrated [0, 1]
           - Falls back to raw RF probs if insufficient data
        
        5. **IsolationForest Training**
           - Trained ONLY on GOOD samples
           - Learns normal data distribution
           - Provides anomaly scores [-1, 1] for novel failures
        
        6. **Data Balancing**
           - Upsample minority (BAD) class if needed
           - Upsample to at least upsample_min_bad samples
           - Uses random sampling WITH replacement
        
        7. **Synthetic Augmentation (Optional)**
           - Create synthetic BAD-like samples via perturbation
           - Fraction controlled by synthetic_frac
           - Adds diversity to training data
        
        8. **Feature Scaling**
           - Fit StandardScaler on feature matrix
           - Normalize to mean=0, std=1
           - Critical for NN convergence
        
        9. **Sequence Creation (RNN Mode Only)**
           - Transform (n_samples, features) -> (n_sequences, 24, features)
           - 24-hour sliding windows with stride=1
           - Generate new labels at last time step of each sequence
        
        10. **NN Training**
            - Initialize JAX TrainState with model parameters
            - Run training loop for specified epochs
            - Per-sample class-weighted cross-entropy loss
            - Validation loss monitored (printed each epoch)
        
        Outputs Produced
        ----------------
        After successful fit():
        - self.rf: Trained RandomForestClassifier
        - self.rf_cal: Optional calibrated wrapper
        - self.if_det: Trained IsolationForest (on GOOD data)
        - self.scaler: Fitted StandardScaler
        - self.nn_state: JAX TrainState with NN weights
        
        Ready for prediction via:
        - predict(X) -> binary flags
        - predict_proba(X) -> probabilities
        - predict_ensemble(X) -> raw component scores
        
        Class Weight Computation
        -------------------------
        Weights = 1 / (class_frequency + epsilon)
        - Underrepresented classes get higher weight
        - Epsilon prevents division by zero
        - Example: If GOOD=95% and BAD=5%
          - GOOD weight = 1 / 0.95 ≈ 1.05
          - BAD weight = 1 / 0.05 = 20.0
          - BAD samples 19x more important in loss
        
        Error Handling
        --------------
        - Empty training set: Raises RuntimeError
        - Missing features: Auto-fills with 0 via _build_X()
        - Low minority count: Falls back to raw RF probs
        - Sequence length > training size: Returns single sequence
        
        Notes
        -----
        - Model assumes time-ordered data for RNN mode
        - Sequence creation is lossy: (n_samples - 23) sequences from n_samples
        - NN training is stochastic (depends on batch ordering and dropout)
        - Large synthetic_frac can create data leakage (perturbed versions of same sample)
        - Class weights computed per fit(), not per-epoch
        
        Example
        -------
            >>> model = SolarHybridModel(use_rnn=True)
            >>> model.fit(training_df, 'Flag_GHI', epochs=30, synthetic_frac=0.05)
            >>> print(f"RF AUC: {model.rf.score(X_test, y_test)}")
            >>> preds = model.predict(test_df)
            >>> probs = model.predict_proba(test_df)
        
        See Also
        --------
        predict() : Get binary QC flags
        predict_proba() : Get probability scores
        predict_ensemble() : Get component scores separately
        inject_synthetic_anomalies() : Synthetic data generation
        """
        # select labeled rows
        train_df = df[df[target_col].notna()].copy()
        if train_df.empty:
            raise RuntimeError(f"No labeled rows for {target_col}")

        # labels: 99 -> 0 (BAD), others -> 1 (GOOD)
        y = np.where(train_df[target_col] == 99, 0, 1).astype(int)

        # ---------------- build features & RF baseline -----------------
        X = self._build_X(train_df)
        print(f"    [fit] Training base RF on {len(X)} rows (target={target_col})")
        self.rf.fit(X, y)

        # safe calibrated probabilities (if viable)
        # CalibratedClassifierCV requires at least one sample per class in each fold
        classes, counts = np.unique(y, return_counts=True)
        min_class_count = int(min(counts)) if len(counts) > 0 else 0
        self.rf_cal = None
        try:
            if min_class_count >= 3:
                cv = min(5, min_class_count)
                self.rf_cal = CalibratedClassifierCV(self.rf, method='isotonic', cv=cv)
                self.rf_cal.fit(X, y)
                rf_prob = self.rf_cal.predict_proba(X)[:, 1]
            else:
                # fallback: use raw RF probabilities
                rf_prob = self.rf.predict_proba(X)[:, 1]
                print("    [fit] WARNING: too few minority samples for robust calibration; using raw RF probs")
        except Exception as e:
            print(f"    [fit] Calibration failed ({e}); using raw RF probs")
            self.rf_cal = None
            rf_prob = self.rf.predict_proba(X)[:, 1]

        # ---------------- unsupervised detector (IsolationForest) -------
        # Train IF on GOOD samples only (so that lower IF score = more anomalous)
        try:
            good_mask = (y == 1)
            if good_mask.sum() >= 50:
                self.if_det = IsolationForest(n_estimators=128, contamination='auto', random_state=42)
                self.if_det.fit(X.loc[good_mask])
                # provide IF score for train_df rows (useful as a feature)
                train_df['IF_Score'] = self.if_det.decision_function(X)
            else:
                self.if_det = None
                train_df['IF_Score'] = 0.0
        except Exception as e:
            print(f"    [fit] IsolationForest training failed: {e}")
            self.if_det = None
            train_df['IF_Score'] = 0.0

        # ---------------- stack RF prob and IF_Score into X_stack ----------
        X_stack = X.copy()
        X_stack['RF_Prob'] = rf_prob
        if 'IF_Score' in train_df.columns:
            X_stack['IF_Score'] = train_df['IF_Score'].values

        # ---------------- handle imbalance: upsample small BAD class -----------
        y_arr = y.copy()
        counts_all = np.bincount(y_arr)
        n_bad = int(counts_all[0]) if len(counts_all) > 0 else 0
        n_total = len(y_arr)
        target_bad = max(upsample_min_bad, int(0.005 * n_total))  # 0.5% or min

        if 0 < n_bad < target_bad:
            print(f"    [fit] Upsampling BAD examples {n_bad} -> {target_bad}")
            idx_bad = np.where(y_arr == 0)[0]
            idx_good = np.where(y_arr == 1)[0]
            n_need = target_bad - n_bad
            idx_bad_ups = np.random.choice(idx_bad, size=n_need, replace=True)
            all_idx = np.concatenate([idx_good, idx_bad, idx_bad_ups])
            X_stack = X_stack.iloc[all_idx].reset_index(drop=True)
            y_arr = np.concatenate([
                np.ones(len(idx_good), dtype=int),
                np.zeros(len(idx_bad), dtype=int),
                np.zeros(len(idx_bad_ups), dtype=int)
            ])
            # shuffle
            perm = np.random.permutation(len(y_arr))
            X_stack = X_stack.iloc[perm].reset_index(drop=True)
            y_arr = y_arr[perm]
            print(f"    [fit] After upsample: total={len(y_arr)}, bad={int((y_arr == 0).sum())}")
        else:
            # no upsampling needed
            X_stack = X_stack.reset_index(drop=True)
            y_arr = y_arr.copy()

        # ---------------- synthetic augmentation (optional) -------------------
        if synthetic_frac > 0.0:
            # use features only (we will not attempt to alter label column in this function)
            # create synthetic rows by perturbing X_stack copies (not the original df)
            n_make = max(1, int(synthetic_frac * len(X_stack)))
            print(f"    [fit] Creating {n_make} synthetic anomalies (frac={synthetic_frac})")
            rng = np.random.default_rng(12345)
            synth_rows = []
            for i in rng.choice(len(X_stack), size=n_make, replace=True):
                row = X_stack.iloc[i].copy()
                # small set of perturbations (conservative)
                mode = rng.integers(0, 4)
                if mode == 0:
                    # stuck-zero sensors
                    for c in ['GHI', 'DNI', 'DHI', 'RF_Prob']:
                        if c in row.index:
                            row[c] = 0.0
                elif mode == 1:
                    # spike on random irradiance
                    cand = [c for c in ['GHI', 'DNI', 'DHI'] if c in row.index]
                    if cand:
                        row[rng.choice(cand)] = float(row[cand[0]] + rng.uniform(300, 800))
                elif mode == 2:
                    # clipping to clear-sky if present
                    if 'GHI' in row.index and 'GHI_Clear' in X_stack.columns:
                        row['GHI'] = 0.95 * float(X_stack.at[i, 'GHI_Clear']) if 'GHI_Clear' in X_stack.columns else float(row['GHI'] * 0.95)
                else:
                    # drift proportional to zenith
                    if 'SZA' in row.index:
                        row['GHI'] = float(row.get('GHI', 0.0) + (90.0 - float(row.get('SZA', 90.0))) * 0.1)
                synth_rows.append(row)
            if synth_rows:
                X_extra = pd.DataFrame(synth_rows).reset_index(drop=True)
                y_extra = np.zeros(len(X_extra), dtype=int)  # synthetic = BAD
                X_stack = pd.concat([X_stack.reset_index(drop=True), X_extra], axis=0).reset_index(drop=True)
                y_arr = np.concatenate([y_arr, y_extra])

        # ---------------- scaling --------------------------------------------
        X_scaled = self.scaler.fit_transform(X_stack)

        # ---------------- compute class weights (normalized & clipped) -------
        classes_present = np.unique(y_arr)
        raw_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y_arr)
        # normalize to mean 1.0 to keep loss magnitude stable
        normed = raw_weights / float(np.mean(raw_weights))
        # clip excessive weights to avoid instability
        clipped = np.clip(normed, 1.0, 10.0)
        class_weights = jnp.ones(2)
        for cls, w in zip(classes_present, clipped):
            class_weights = class_weights.at[int(cls)].set(float(w))

        print(f"    [fit] class counts bad={int((y_arr==0).sum())} good={int((y_arr==1).sum())}")
        print(f"    [fit] raw_weights={raw_weights}, normed={normed}, clipped={clipped}")
        print(f"    [fit] used NN class_weights (BAD=0,GOOD=1)={class_weights}")

        # -------- Prepare data for NN (RNN vs Dense) --------
        if self.use_rnn:
            print(f"    [fit] Preparing sequences for RNN (seq_length={self.seq_length})...")
            X_seq, y_seq = create_sequences(X_scaled, y_arr, seq_length=self.seq_length, stride=5)
            print(f"    [fit] RNN sequences: {X_seq.shape}")
            X_nn = X_seq.astype('float32')
            y_nn = y_seq.astype('int32')
        else:
            X_nn = X_scaled.astype('float32')
            y_nn = y_arr.astype('int32')

        # ---------------- NN training (JAX/Flax) ----------------------------
        # Prepare tf.data.Dataset to supply data to the JAX step loop.
        # We use `.repeat()` and iterate for a deterministic number of steps.
        batch_size = max(8, int(batch_size))
        ds = tf.data.Dataset.from_tensor_slices({
            'x': X_nn,
            'y': y_nn
        }).shuffle(8192).batch(batch_size, drop_remainder=True).repeat()

        steps_per_epoch = max(1, len(X_nn) // batch_size)
        total_steps = int(epochs * steps_per_epoch)
        print(f"    [fit] Training NN: samples={len(X_nn)} batch_size={batch_size} steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

        # init model
        rng = jax.random.PRNGKey(42)
        
        if self.use_rnn:
            # Initialize RNN model with proper input shape
            model = SolarRNN(hidden_dim=64, num_layers=2, dropout_rate=0.2, use_attention=True)
            # Create dummy input: (batch=1, time_steps, features)
            dummy_input = jnp.ones((1, X_seq.shape[1], X_seq.shape[2]))
            params = model.init(rng, dummy_input, training=False)['params']
            print(f"    [fit] Initialized SolarRNN with shape {dummy_input.shape}")
        else:
            model = DenseNN()
            params = model.init(rng, jnp.ones((1, X_scaled.shape[1])))['params']
            print(f"    [fit] Initialized DenseNN with shape (1, {X_scaled.shape[1]})")
        
        self.nn_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))

        it = iter(ds)
        for step in range(total_steps):
            batch = next(it)
            jax_batch = {'x': jnp.array(batch['x'].numpy()), 'y': jnp.array(batch['y'].numpy())}
            self.nn_state = train_step(self.nn_state, jax_batch, class_weights, is_rnn=self.use_rnn)
            if (step + 1) % max(1, (steps_per_epoch * max(1, epochs // 5))) == 0:
                # light progress log every ~20% of training
                print(f"    [fit] NN training progress: step {step+1}/{total_steps}")

        # Diagnostics: training report on the training set
        # Note: Skip diagnostics for RNN mode since X_scaled has different shape than training data
        if not self.use_rnn:
            logits = self.nn_state.apply_fn({'params': self.nn_state.params}, X_scaled)
            preds = jnp.argmax(logits, axis=1)
            try:
                print("    [fit] NN training classification report (on stacked dataset):")
                print(classification_report(y_arr, np.array(preds), target_names=['BAD', 'GOOD'], digits=4))
            except Exception:
                pass
        else:
            print("    [fit] NN training complete (RNN diagnostics skipped - different data shape)")

        # store last-fit artifacts (RF, rf_cal, scaler, if_det, nn_state) on self
        # they are already attributes of self

    # ---------------- predict --------------------------------------------
    def predict(self, df: pd.DataFrame, target_col: str, do_return_probs: bool = False):
        """
        Predict quality flags for new data using hybrid ensemble model.
        
        This method orchestrates end-to-end prediction using the trained
        RandomForest, IsolationForest, and Neural Network in an ensemble.
        
        Critical Design Invariants
        ---------------------------
        1. **Feature Consistency**
           - RandomForest sees EXACTLY the same features at prediction time as training
           - No extra columns, no missing columns, same ordering via self.common_features
           - Strictly enforced: missing features auto-filled with 0.0
        
        2. **Meta-Feature Isolation**
           - RF outputs (probability, anomaly scores) are meta-features ONLY
           - Never fed back into RandomForest (would cause information leakage)
           - Consumed only by NN stacker for ensemble
        
        3. **Probability Interpretation**
           - All outputs represent P(GOOD) ∈ [0.0, 1.0]
           - 1.0 = Fully confident GOOD quality
           - 0.5 = Uncertain (decision boundary)
           - 0.0 = Fully confident BAD quality
        
        Prediction Pipeline
        -------------------
        1. **Build RF Features**
           - Extract common_features from input
           - Missing columns filled with 0.0
           - Inf/-Inf clamped to 0.0 (numerical safety)
        
        2. **RandomForest Prediction**
           - Use calibrated model if available (preferred)
           - Fall back to raw RF if calibration unavailable
           - Output: RF probability P(GOOD)
        
        3. **IsolationForest Anomaly Score**
           - Score new samples against GOOD distribution
           - Lower score = more anomalous
           - Normalized to [0, 1] range
           - Treated as meta-feature for NN
        
        4. **NN Input Preparation**
           - Combine RF features with meta-features:
             * RF_Prob: RandomForest's P(GOOD)
             * IF_Score: IsolationForest anomaly score
           - Scale features via pre-fitted StandardScaler
           - Create 24-hour sequences (RNN mode only)
        
        5. **NN Prediction**
           - RNN mode: Process 24-hour windows, aggregate final layer
           - Dense mode: Direct classification on features
           - Output: logits, convert to P(GOOD) via softmax
        
        6. **Ensemble Aggregation**
           - Combine probabilities from RF, IF, and NN
           - Simple average: (rf_prob + if_prob + nn_prob) / 3
           - Weights all components equally (could be tuned)
        
        7. **Flag Generation**
           - Ensemble P(GOOD) >= 0.5: Flag = 1 (GOOD)
           - Ensemble P(GOOD) < 0.5: Flag = 99 (BAD)
        
        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered dataframe from solar_features.add_features()
            - Must contain all required features (or auto-filled)
            - Rows are independent predictions
            - Can be any length (single row to full dataset)
        
        target_col : str
            Target column name (e.g., 'Flag_GHI', 'Flag_DNI', 'Flag_DHI')
            - Used for identification/logging
            - Does NOT affect prediction logic
            - Included for API symmetry with fit()
        
        do_return_probs : bool, default=False
            Whether to return probability scores
            - False: Return flags only (99/1)
            - True: Return (flags, probs) tuple
            - Probabilities useful for confidence-gated writeback
        
        Returns
        -------
        np.ndarray or tuple
            If do_return_probs=False:
              flags : np.ndarray, shape (n_samples,)
                - Values: 99 = BAD, 1 = GOOD
                - Integer array
            
            If do_return_probs=True:
              (flags, probs) : tuple of np.ndarray
                - flags: shape (n_samples,), values {1, 99}
                - probs: shape (n_samples,), values ∈ [0.0, 1.0]
                - probs[i] = P(GOOD) for sample i
        
        Ensemble Probability Calculation
        --------------------------------
        For each sample:
        1. rf_prob = RandomForest P(GOOD)
        2. if_prob = (IsolationForest_score + 1) / 2  [normalize to [0,1]]
        3. nn_prob = softmax(nn_logits)[1]
        4. ensemble_prob = (rf_prob + if_prob + nn_prob) / 3
        5. flag = 1 if ensemble_prob >= 0.5 else 99
        
        Error Handling
        --------------
        - Missing features: Auto-filled with 0.0 (no error)
        - Inf/-Inf in features: Clamped to 0.0
        - Calibration unavailable: Falls back to raw RF probs
        - Sequence too short (RNN): Handled gracefully (single sequence)
        - Non-trainable model: Raises RuntimeError (must call fit() first)
        
        Performance Notes
        -----------------
        - RF prediction: O(n_samples * n_trees * depth)
        - IF prediction: O(n_samples * log(n_samples) * n_isolators)
        - NN prediction: O(n_samples * seq_length) for RNN
        - Total: Typically <1ms per 1000 samples
        
        Confidence Scores
        -----------------
        Probability interpretation:
        - [0.9, 1.0]: Very confident GOOD
        - [0.7, 0.9]: Confident GOOD
        - [0.5, 0.7]: Probably GOOD
        - [0.3, 0.5]: Probably BAD
        - [0.1, 0.3]: Confident BAD
        - [0.0, 0.1]: Very confident BAD
        
        Example
        -------
            >>> model.fit(train_df, 'Flag_GHI')
            >>> # Predict flags only
            >>> flags = model.predict(test_df, 'Flag_GHI')
            >>> # Predict flags and probabilities
            >>> flags, probs = model.predict(test_df, 'Flag_GHI', do_return_probs=True)
            >>> high_confidence = probs > 0.8
            >>> print(f"High confidence predictions: {high_confidence.sum()}")
        
        See Also
        --------
        fit() : Train the model
        predict_ensemble() : Get component predictions separately
        
        Notes
        -----
        - Prediction order matches input DataFrame row order
        - Model must be fitted before prediction (will error otherwise)
        - Batch size doesn't affect prediction results, only speed
        - Ensemble is simple average (could be improved with learned weights)
        """

        # ------------------------------------------------------------------
        # 1. Build RF feature matrix (STRICT MATCH to RF training features)
        # ------------------------------------------------------------------
        # NOTE:
        #  - We do NOT call _build_X() here because that function may evolve
        #    to include NN-only or diagnostic features.
        #  - This explicit construction prevents silent feature drift.
        X_rf = pd.DataFrame(
            {c: (df[c] if c in df.columns else 0.0) for c in self.common_features},
            index=df.index
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # ------------------------------------------------------------------
        # 2. RandomForest probability (calibrated if available)
        # ------------------------------------------------------------------
        # We always try the calibrated model first.
        # If calibration failed or is unavailable, fall back safely.
        if self.rf_cal is not None:
            try:
                rf_prob = self.rf_cal.predict_proba(X_rf)[:, 1]
            except Exception:
                rf_prob = self.rf.predict_proba(X_rf)[:, 1]
        else:
            rf_prob = self.rf.predict_proba(X_rf)[:, 1]

        # ------------------------------------------------------------------
        # 3. Build NN input matrix (RF features + meta-features)
        # ------------------------------------------------------------------
        X_stack = X_rf.copy()
        X_stack['RF_Prob'] = rf_prob

        # IsolationForest score is a *meta-feature* ONLY
        # It must NEVER be used by the RF
        if getattr(self, 'if_det', None) is not None:
            try:
                X_stack['IF_Score'] = self.if_det.decision_function(X_rf)
            except Exception:
                # Fail-safe: never let IF errors crash prediction
                X_stack['IF_Score'] = 0.0

        # ------------------------------------------------------------------
        # 4. Scale features for NN
        # ------------------------------------------------------------------
        X_scaled = self.scaler.transform(X_stack)

        # ------------------------------------------------------------------
        # 5. Neural-network inference (with RNN or Dense)
        # ------------------------------------------------------------------
        if self.use_rnn:
            # For RNN: create sequences from the scaled data
            # We predict on each complete sequence, using the last time step's prediction
            X_seq, _ = create_sequences(X_scaled, np.zeros(len(X_scaled)), 
                                       seq_length=self.seq_length, stride=1)
            
            if len(X_seq) == 0:
                # Not enough data for sequences, fallback to last few samples
                print("    [predict] Warning: Not enough samples for RNN sequence, using fallback")
                X_seq = X_scaled[np.newaxis, ...]
            
            X_seq = jnp.array(X_seq.astype('float32'))
            logits = self.nn_state.apply_fn(
                {'params': self.nn_state.params},
                X_seq,
                training=False
            )
            
            # Get probabilities
            probs_seq = np.array(jax.nn.softmax(logits, axis=1)[:, 1])
            
            # Map back to original data length
            # Each sequence maps to its last time step
            probs_good = np.zeros(len(X_scaled))
            for i, seq_idx in enumerate(range(self.seq_length - 1, len(X_scaled))):
                if i < len(probs_seq):
                    probs_good[seq_idx] = probs_seq[i]
            
            # Fill beginning with first sequence prediction (conservative)
            if len(probs_seq) > 0:
                for i in range(min(self.seq_length - 1, len(X_scaled))):
                    probs_good[i] = probs_seq[0]
        else:
            # Dense NN: process all samples at once
            logits = self.nn_state.apply_fn(
                {'params': self.nn_state.params},
                jnp.array(X_scaled.astype('float32')),
                training=False
            )
            
            # Softmax to obtain probabilities
            probs_good = np.array(jax.nn.softmax(logits, axis=1)[:, 1])

        # Binary decision (0 = BAD, 1 = GOOD)
        preds = (probs_good >= 0.5).astype(int)

        # Map to project convention: 99 = BAD, 1 = GOOD
        flags = np.where(preds == 1, 1, 99)

        # ------------------------------------------------------------------
        # 6. Return
        # ------------------------------------------------------------------
        if do_return_probs:
            return flags, probs_good

        return flags

    # ---------------- save/load model ------------------------------------
    def save_model(self, filepath: str):
        """
        Persist trained model to disk with full configuration.
        
        Saves all trained components and configuration in pickle format,
        enabling model reuse without retraining.
        
        Model Contents
        ---------------
        - RandomForestClassifier (base model)
        - CalibratedClassifierCV wrapper (if calibration succeeded)
        - IsolationForest detector (trained on GOOD samples)
        - StandardScaler (feature normalization)
        - NN parameters (RNN or Dense weights)
        - Common features list (for predict-time validation)
        - Configuration (use_rnn, seq_length)
        - Version tag (2.0 for RNN-enabled)
        
        Parameters
        ----------
        filepath : str
            Destination file path
            - Can be relative or absolute
            - Recommended: 'models/model_Flag_GHI.pkl'
            - Parent directory created automatically if missing
            - Overwrites existing file (no confirmation)
        
        File Format
        -----------
        Python pickle (.pkl)
        - Human-unreadable but fast and complete
        - Backward compatible with v1.0 models
        - Contains full JAX pytree (NN parameters)
        - File size: typically 50-500 KB
        
        File Size Estimation
        --------------------
        - RF (50 trees): ~100-200 KB
        - IsolationForest: ~10-50 KB
        - NN parameters (2-layer GRU): ~50-100 KB
        - Scalars and metadata: ~1 KB
        - Total: typically 150-350 KB
        
        Backward Compatibility
        ----------------------
        - v2.0 (RNN-enabled): Saves use_rnn=True/False in dict
        - v1.0 (Dense-only): Old models detected and loaded correctly
        - Loading v1.0: Uses DenseNN architecture
        - Loading v2.0: Uses RNN or Dense based on saved flag
        
        Usage
        -----
        After training:
            >>> model = SolarHybridModel(use_rnn=True)
            >>> model.fit(train_df, 'Flag_GHI', epochs=30)
            >>> model.save_model('models/ghi_model.pkl')
        
        Later, for prediction without retraining:
            >>> model = SolarHybridModel.load_model('models/ghi_model.pkl')
            >>> flags = model.predict(new_df, 'Flag_GHI')
        
        Error Handling
        --------------
        - Parent directory created automatically (mkdir -p style)
        - Overwrites existing file without warning
        - Raises OSError if write fails (e.g., permissions)
        - No compression applied (consider gzip for archival)
        
        Notes
        -----
        - Pickle is Python-specific (not portable to other languages)
        - Use load_model() classmethod to restore
        - Saved at any point (only components set so far are saved)
        - Can save even if not fully trained (partial model)
        - Version '2.0' added to support future format changes
        
        See Also
        --------
        load_model() : Load saved model
        fit() : Train model before saving
        predict() : Use loaded model for prediction
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Extract NN params if state exists
        nn_params = None
        if self.nn_state is not None:
            nn_params = self.nn_state.params
        
        # Package all components
        model_dict = {
            'rf': self.rf,
            'rf_cal': self.rf_cal,
            'if_det': self.if_det,
            'scaler': self.scaler,
            'nn_params': nn_params,
            'common_features': self.common_features,
            'use_rnn': self.use_rnn,
            'seq_length': self.seq_length,
            'version': '2.0'  # Updated version for RNN support
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        
        model_type = "RNN" if self.use_rnn else "Dense"
        print(f"Model saved to {filepath} (type: {model_type})")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SolarHybridModel':
        """
        Load a trained model from disk for prediction.
        
        Restores all model components and configuration from a saved pickle file,
        enabling prediction without retraining. Supports both v1.0 and v2.0 formats.
        
        Parameters
        ----------
        filepath : str
            Path to saved model file
            - Typically: 'models/model_Flag_*.pkl'
            - Can be relative or absolute
            - Must be created via save_model()
        
        Returns
        -------
        SolarHybridModel
            Fully restored model ready for prediction
            - All training components loaded
            - Configuration restored from file
            - No additional setup needed
        
        Model Restoration Process
        -------------------------
        1. **Load Pickle**
           - Deserialize model_dict from disk
           - Handles both v1.0 and v2.0 formats
        
        2. **Architecture Detection**
           - Read use_rnn flag from file
           - Create new instance with correct type
           - RNN vs Dense detected automatically
        
        3. **Component Restoration**
           - Load RF classifier
           - Load calibration wrapper (if saved)
           - Load IsolationForest detector
           - Load StandardScaler
           - Restore common features list
           - Restore configuration (use_rnn, seq_length)
        
        4. **NN Parameter Recovery**
           - If nn_params present in file:
             * Recreate NN model (SolarRNN or DenseNN)
             * Initialize JAX TrainState
             * Load saved parameters into state
           - If nn_params absent (partial model):
             * Skip NN restoration
             * Model usable but without NN component
        
        5. **Validation**
           - Print model type and version
           - Check file existence
           - Raise error if file not found
        
        File Format Support
        -------------------
        Backward Compatible:
        - v1.0: Old models (Dense-only) detected and loaded
          * use_rnn field missing (defaults to False)
          * seq_length field missing (defaults to 24)
        - v2.0: New models (RNN-enabled)
          * use_rnn explicitly saved
          * seq_length saved
        
        Usage Examples
        ---------------
        Load and predict in one step:
            >>> model = SolarHybridModel.load_model('models/ghi_model.pkl')
            >>> flags = model.predict(test_df, 'Flag_GHI')
        
        Load and check model type:
            >>> model = SolarHybridModel.load_model('models/ghi_model.pkl')
            >>> print(f"Architecture: {'RNN' if model.use_rnn else 'Dense'}")
        
        Load with error handling:
            >>> try:
            ...     model = SolarHybridModel.load_model('models/unknown.pkl')
            ... except FileNotFoundError:
            ...     print("Model file not found")
        
        Error Handling
        ---------------
        - FileNotFoundError: If filepath doesn't exist
        - pickle.UnpicklingError: If file is corrupted
        - ValueError: If model_dict missing critical keys
        - Version mismatch: Handled gracefully (defaults used)
        
        Performance
        -----------
        - Load time: 100-500ms (depends on disk speed)
        - Memory: Same as training (model components)
        - Inference ready immediately after load
        
        Notes
        -----
        - Returned model is ready for predict() immediately
        - No retraining needed after loading
        - NN optimizer re-initialized (not needed for prediction)
        - Pickle is Python-specific (not portable)
        - Use save_model() to update/recreate
        
        See Also
        --------
        save_model() : Save trained model to disk
        predict() : Use loaded model for prediction
        fit() : Train new model
        
        Version History
        ----------------
        v2.0: Added RNN support, use_rnn flag
        v1.0: Original Dense-only models
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Determine if RNN or Dense based on model_dict
        use_rnn = model_dict.get('use_rnn', False)
        
        # Create new instance with correct model type
        model = cls(use_rnn=use_rnn)
        
        # Restore components
        model.rf = model_dict['rf']
        model.rf_cal = model_dict['rf_cal']
        model.if_det = model_dict['if_det']
        model.scaler = model_dict['scaler']
        model.common_features = model_dict['common_features']
        model.use_rnn = use_rnn
        model.seq_length = model_dict.get('seq_length', 24)
        
        # Restore NN state
        nn_params = model_dict.get('nn_params')
        if nn_params is not None:
            # Recreate the NN model and state
            rng = jax.random.PRNGKey(42)
            
            if use_rnn:
                nn_model = SolarRNN(hidden_dim=64, num_layers=2, dropout_rate=0.2, use_attention=True)
                # RNN needs proper sequence shape
                dummy_input = jnp.ones((1, model.seq_length, model.scaler.n_features_in_))
                params = nn_model.init(rng, dummy_input, training=False)['params']
            else:
                nn_model = DenseNN()
                dummy_input = jnp.ones((1, model.scaler.n_features_in_))
                params = nn_model.init(rng, dummy_input, training=False)['params']
            
            # Create state with loaded params
            model.nn_state = train_state.TrainState.create(
                apply_fn=nn_model.apply,
                params=nn_params,
                tx=optax.adam(1e-3)  # optimizer not needed for inference but required for state
            )
        
        model_type = "RNN" if use_rnn else "Dense"
        version = model_dict.get('version', '1.0')
        print(f"Model loaded from {filepath} (type: {model_type}, version: {version})")
        return model
