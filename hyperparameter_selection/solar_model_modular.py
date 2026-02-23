"""
solar_model_modular.py
=======================

Modular Hybrid Solar QC Model (RNN Architecture)

This is a modular version of solar_model.py adapted for hyperparameter selection.
It implements the same RNN-based architecture with customizable hyperparameters
for tuning sequence length, hidden dimensions, learning rates, and epochs.

Architecture
------------
1. **RandomForest Classifier** - Base model for feature importance
2. **IsolationForest Detector** - Anomaly detection on GOOD samples
3. **SolarRNN** - Recurrent neural network with temporal attention for sequences
4. **Probability Calibration** - Isotonic calibration for reliable confidences

Key Differences from solar_model.py
-----------------------------------
- Hyperparameters extracted to a 'params' dict for grid search
- Support for variable sequence lengths and hidden dimensions
- Simplified for cleaner hyperparameter experimentation
- All three model components (RF, IF, RNN) included

Usage
-----
    params = {
        'rf_n_estimators': 50,
        'rf_max_depth': 50,
        'seq_length': 240,
        'hidden_dim': 64,
        'num_layers': 2,
        'nn_epochs': 20,
        'nn_batch_size': 64,
        'nn_learning_rate': 0.001
    }
    
    model = SolarHybridModel(params=params)
    model.fit(df_train, target_col='Flag_GHI')
    flags, probs = model.predict(df_test, 'Flag_GHI', do_return_probs=True)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


# =============================================================================
# RNN Architecture - Temporal Attention
# =============================================================================

class SolarRNN(nn.Module):
    """
    Recurrent Neural Network for time-series solar QC prediction.
    
    Parameters
    ----------
    hidden_dim : int, default=64
        Size of GRU hidden state
    num_layers : int, default=2
        Number of stacked GRU layers
    dropout_rate : float, default=0.2
        Dropout probability during training
    use_attention : bool, default=True
        Whether to use temporal attention mechanism
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
        
        # Residual connection
        if x.shape[-1] != self.hidden_dim:
            x_proj = nn.Dense(self.hidden_dim)(x)
        else:
            x_proj = x
        
        outputs_1 = outputs_1 + 0.5 * x_proj
        
        # Layer 2: Second GRU layer (optional)
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
            outputs_2 = outputs_2 + 0.5 * outputs_1
            outputs = outputs_2
        else:
            outputs = outputs_1
        
        # Temporal Attention
        if self.use_attention:
            attention_scores = nn.Dense(1)(outputs)
            attention_weights = jax.nn.softmax(attention_scores, axis=1)
            attended = jnp.sum(outputs * attention_weights, axis=1)
        else:
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
        
        # Output logits
        logits = nn.Dense(2)(x)
        
        return logits


class DenseNN(nn.Module):
    """Fallback dense neural network for non-sequential use."""
    
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        logits = nn.Dense(2)(x)
        return logits


# =============================================================================
# Sequence Creation Utility
# =============================================================================

def create_sequences(X: np.ndarray, y: np.ndarray, seq_length: int = 24, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for RNN training.
    
    Parameters
    ----------
    X : np.ndarray
        Feature array of shape (n_samples, n_features)
    y : np.ndarray
        Labels array of shape (n_samples,)
    seq_length : int
        Length of sliding window (adjust for your data resolution)
    stride : int
        Step size between windows (default: 1)
    
    Returns
    -------
    X_seq : np.ndarray
        Sequences of shape (n_sequences, seq_length, n_features)
    y_seq : np.ndarray
        Labels at end of each sequence, shape (n_sequences,)
    """
    X_sequences = []
    y_sequences = []
    
    n_samples = len(X)
    
    for i in range(0, n_samples - seq_length + 1, stride):
        X_sequences.append(X[i:i + seq_length])
        y_sequences.append(y[i + seq_length - 1])
    
    if not X_sequences:
        # Fallback if insufficient data
        return X[np.newaxis, ...], y[-1:]
    
    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    
    return X_seq, y_seq


# =============================================================================
# Training Step Function
# =============================================================================

def train_step_fn(state, batch, class_weights, rng):
    """Single training step with class-weighted loss."""
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch['x'],
            training=True,
            rng=rng  # Pass RNG for dropout
        )
        labels = jax.nn.one_hot(batch['y'], 2)
        per_sample = optax.softmax_cross_entropy(logits=logits, labels=labels)
        weights = class_weights[batch['y']]
        return jnp.mean(per_sample * weights)
    
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

# =============================================================================
# Hybrid Model Wrapper
# =============================================================================

class SolarHybridModel:
    def __init__(self, params: Optional[Dict] = None, use_rnn: bool = True):
        """
        Initialize hybrid solar QC model.
        
        Parameters
        ----------
        params : dict, optional
            Hyperparameters:
            - rf_n_estimators (default: 50)
            - rf_max_depth (default: 50)
            - seq_length (default: 24)
            - hidden_dim (default: 64)
            - num_layers (default: 2)
            - dropout_rate (default: 0.2)
            - nn_epochs (default: 20)
            - nn_batch_size (default: 64)
            - nn_learning_rate (default: 0.001)
        use_rnn : bool
            Use RNN (True) or DenseNN (False)
        """
        if params is None:
            params = {}
        
        self.params = params
        self.use_rnn = use_rnn
        
        # Extract hyperparameters with defaults
        self.rf_n_estimators = params.get('rf_n_estimators', 50)
        self.rf_max_depth = params.get('rf_max_depth', 50)
        self.seq_length = params.get('seq_length', 240)  # 4 hours instead of 24 hours (memory efficiency)
        self.hidden_dim = params.get('hidden_dim', 64)
        self.num_layers = params.get('num_layers', 2)
        self.dropout_rate = params.get('dropout_rate', 0.2)
        self.nn_epochs = params.get('nn_epochs', 20)
        self.nn_batch_size = params.get('nn_batch_size', 64)
        self.nn_learning_rate = params.get('nn_learning_rate', 0.001)
        
        # Model components
        self.rf: Optional[RandomForestClassifier] = None
        self.rf_cal: Optional[CalibratedClassifierCV] = None
        self.if_det: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        self.nn_state: Optional[train_state.TrainState] = None
        
        # Feature list (must match solar_features.add_features output)
        self.common_features = [
            'Timestamp_Num',
            'hour_sin', 'hour_cos',
            'doy_sin', 'doy_cos',
            'SZA', 'elevation',
            'CSI',
            'QC_PhysicalFail',
            'CorrFeat_GHI', 'CorrFeat_DNI', 'CorrFeat_DHI',
            'GHI', 'DNI', 'DHI',
            'GHI_Clear', 'DNI_Clear', 'DHI_Clear'
        ]
    
    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from dataframe."""
        data = {}
        for c in self.common_features:
            data[c] = df[c].values if c in df.columns else np.zeros(len(df))
        X = np.column_stack([data[c] for c in self.common_features])
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    def fit(self, df: pd.DataFrame, target_col: str, upsample_min_bad: int = 500):
        """
        Fit the hybrid model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with features and labels
        target_col : str
            Name of label column (99=BAD, else=GOOD)
        upsample_min_bad : int
            Minimum number of BAD samples
        """
        # Extract labeled data
        train_df = df[df[target_col].notna()].copy()
        if train_df.empty:
            raise RuntimeError(f"No labeled rows for {target_col}")
        
        # Convert labels: 99 -> 0 (BAD), else -> 1 (GOOD)
        y = np.where(train_df[target_col] == 99, 0, 1).astype(int)
        
        # Build feature matrix
        X = self._build_X(train_df)
        print(f"    [fit] Training on {len(X)} rows (target={target_col})")
        
        # Train RandomForest
        self.rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=42
        )
        self.rf.fit(X, y)
        
        # Calibrate RF probabilities
        classes, counts = np.unique(y, return_counts=True)
        min_class_count = int(min(counts)) if len(counts) > 0 else 0
        
        try:
            if min_class_count >= 3:
                cv = min(5, min_class_count)
                self.rf_cal = CalibratedClassifierCV(self.rf, method='isotonic', cv=cv)
                self.rf_cal.fit(X, y)
                rf_prob = self.rf_cal.predict_proba(X)[:, 1]
            else:
                rf_prob = self.rf.predict_proba(X)[:, 1]
                print("    [fit] WARNING: Too few minority samples; using raw RF probs")
        except Exception as e:
            print(f"    [fit] Calibration failed ({e}); using raw RF probs")
            self.rf_cal = None
            rf_prob = self.rf.predict_proba(X)[:, 1]
        
        # Train IsolationForest on GOOD samples only
        try:
            good_mask = (y == 1)
            if good_mask.sum() >= 50:
                self.if_det = IsolationForest(
                    n_estimators=128,
                    contamination='auto',
                    random_state=42
                )
                self.if_det.fit(X[good_mask])
                if_score = self.if_det.decision_function(X)
            else:
                self.if_det = None
                if_score = np.zeros(len(X))
        except Exception as e:
            print(f"    [fit] IsolationForest training failed: {e}")
            self.if_det = None
            if_score = np.zeros(len(X))
        
        # Build augmented feature matrix (RF_Prob + IF_Score)
        X_stack = np.column_stack([X, rf_prob, if_score])
        
        # Upsample minority class
        y_arr = y.copy()
        counts_all = np.bincount(y_arr)
        n_bad = int(counts_all[0]) if len(counts_all) > 0 else 0
        n_total = len(y_arr)
        target_bad = max(upsample_min_bad, int(0.005 * n_total))
        
        if 0 < n_bad < target_bad:
            idx_bad = np.where(y_arr == 0)[0]
            idx_good = np.where(y_arr == 1)[0]
            n_need = target_bad - n_bad
            idx_bad_ups = np.random.choice(idx_bad, size=n_need, replace=True)
            all_idx = np.concatenate([idx_good, idx_bad, idx_bad_ups])
            
            X_stack = X_stack[all_idx]
            y_arr = np.concatenate([
                np.ones(len(idx_good), dtype=int),
                np.zeros(len(idx_bad), dtype=int),
                np.zeros(len(idx_bad_ups), dtype=int)
            ])
            # Shuffle
            perm = np.random.permutation(len(y_arr))
            X_stack = X_stack[perm]
            y_arr = y_arr[perm]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_stack)
        
        # Compute class weights
        classes_present = np.unique(y_arr)
        raw_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes_present,
            y=y_arr
        )
        normed = raw_weights / float(np.mean(raw_weights))
        clipped = np.clip(normed, 1.0, 10.0)
        
        class_weights = jnp.ones(2)
        for cls, w in zip(classes_present, clipped):
            class_weights = class_weights.at[int(cls)].set(float(w))
        
        # Create sequences if using RNN
        if self.use_rnn:
            X_seq, y_seq = create_sequences(
                X_scaled,
                y_arr,
                seq_length=self.seq_length,
                stride=1
            )
            print(f"    [fit] Created {len(X_seq)} sequences of length {self.seq_length}")
        else:
            X_seq = X_scaled
            y_seq = y_arr
        
        # Create TensorFlow dataset
        batch_size = max(8, int(self.nn_batch_size))
        if self.use_rnn:
            ds = tf.data.Dataset.from_tensor_slices({
                'x': X_seq.astype('float32'),
                'y': y_seq.astype('int32')
            }).shuffle(8192).batch(batch_size, drop_remainder=True).repeat()
        else:
            ds = tf.data.Dataset.from_tensor_slices({
                'x': X_scaled.astype('float32'),
                'y': y_arr.astype('int32')
            }).shuffle(8192).batch(batch_size, drop_remainder=True).repeat()
        
        steps_per_epoch = max(1, len(y_seq if self.use_rnn else y_arr) // batch_size)
        total_steps = int(self.nn_epochs * steps_per_epoch)
        
        # Initialize NN
        rng = jax.random.PRNGKey(42)
        
        if self.use_rnn:
            model = SolarRNN(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate,
                use_attention=True
            )
            dummy_input = jnp.ones((1, self.seq_length, X_scaled.shape[1]))
        else:
            model = DenseNN()
            dummy_input = jnp.ones((1, X_scaled.shape[1]))
        
        params_init = model.init(rng, dummy_input)['params']
        tx = optax.adam(learning_rate=self.nn_learning_rate)
        self.nn_state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params_init,
            tx=tx
        )
        
        # Training loop
        it = iter(ds)
        for step in range(total_steps):
            batch = next(it)
            jax_batch = {
                'x': jnp.array(batch['x'].numpy()),
                'y': jnp.array(batch['y'].numpy())
            }
            # Generate fresh RNG for this step's dropout
            rng, subkey = jax.random.split(rng)
            self.nn_state = train_step_fn(self.nn_state, jax_batch, class_weights, subkey)
            
            if (step + 1) % max(10, steps_per_epoch) == 0:
                epoch = (step + 1) // steps_per_epoch
                print(f"    [fit] Epoch {epoch}/{self.nn_epochs}")
    
    def predict(self, df: pd.DataFrame, target_col: str, do_return_probs: bool = False):
        """
        Predict quality flags.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with features
        target_col : str
            Target column name (for naming only)
        do_return_probs : bool
            Return probabilities along with flags
        
        Returns
        -------
        flags : np.ndarray
            Quality flags (99=BAD, 1=GOOD)
        [probs] : np.ndarray
            Probability of GOOD (if do_return_probs=True)
        """
        # Build feature matrix
        X = self._build_X(df)
        
        # RF probability
        if self.rf_cal is not None:
            try:
                rf_prob = self.rf_cal.predict_proba(X)[:, 1]
            except:
                rf_prob = self.rf.predict_proba(X)[:, 1]
        else:
            rf_prob = self.rf.predict_proba(X)[:, 1]
        
        # IF score
        if self.if_det is not None:
            try:
                if_score = self.if_det.decision_function(X)
            except:
                if_score = np.zeros(len(X))
        else:
            if_score = np.zeros(len(X))
        
        # Stack and scale
        X_stack = np.column_stack([X, rf_prob, if_score])
        X_scaled = self.scaler.transform(X_stack)
        
        # NN inference
        if self.use_rnn:
            # For RNN, we need sequences. Use last seq_length points.
            if len(X_scaled) < self.seq_length:
                pad_len = self.seq_length - len(X_scaled)
                X_seq = np.vstack([
                    np.zeros((pad_len, X_scaled.shape[1])),
                    X_scaled
                ])
                X_seq = X_seq[np.newaxis, ...]
            else:
                X_seq = X_scaled[-self.seq_length:][np.newaxis, ...]
            
            logits = self.nn_state.apply_fn(
                {'params': self.nn_state.params},
                X_seq,
                training=False
            )
            probs_good = np.array(jax.nn.softmax(logits, axis=1)[0, 1])
        else:
            logits = self.nn_state.apply_fn(
                {'params': self.nn_state.params},
                X_scaled,
                training=False
            )
            probs_good = np.array(jax.nn.softmax(logits, axis=1)[:, 1])
        
        # Convert to flags
        if isinstance(probs_good, (float, np.floating)):
            preds = 1 if probs_good >= 0.5 else 0
            flags = np.array([1 if preds == 1 else 99])
            probs_good = np.array([probs_good])
        else:
            preds = (probs_good >= 0.5).astype(int)
            flags = np.where(preds == 1, 1, 99)
        
        if do_return_probs:
            return flags, probs_good
        return flags