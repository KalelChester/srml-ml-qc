"""
solar_model.py -- MODIFIED
MODIFIED FOR HYPERPARAMETER SELECTION
==============

Hybrid supervised + unsupervised solar QC model.

This module implements:
 - RandomForest base classifier (calibrated) to produce RF probabilities
 - IsolationForest unsupervised detector trained on GOOD examples (exposed as .if_det)
 - Optional synthetic augmentation and safe upsampling for scarce BAD labels
 - A small JAX/Flax neural network that stacks RF_prob + features to produce a
   final probability P(GOOD). The NN uses class-weighted loss (weights are
   normalized and clipped to avoid destabilization).
 - A predict() method that returns flags and optionally probabilities:
       flags: array of {99 (bad), 1 (good)}
       probs: P(GOOD) in [0,1] (1.0 means confident GOOD)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Sequence, Dict

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

# -----------------------------------------------------------------------------
# Neural network (Modular Architecture)
# -----------------------------------------------------------------------------
class DenseNN(nn.Module):
    # Modular layer sizes, defaulting to original architecture
    layer_sizes: Sequence[int] = (128, 64, 32)

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        
        # Dynamically create layers based on layer_sizes
        for size in self.layer_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
            
        logits = nn.Dense(2)(x)  # logits for [BAD(0), GOOD(1)]
        return logits


@jax.jit
def train_step(state, batch, class_weights):
    """
    Single JAX training step with manual per-sample class weights.

    - batch: dict with 'x' (float32 array) and 'y' (int array)
    - class_weights: jnp.array shape (2,) giving weight for classes [0,1]
    """
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['x'])
        labels = jax.nn.one_hot(batch['y'], 2)
        per_sample = optax.softmax_cross_entropy(logits=logits, labels=labels)
        weights = class_weights[batch['y']]  # lookup final weight per sample
        return jnp.mean(per_sample * weights)

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


# -----------------------------------------------------------------------------
# Hybrid wrapper
# -----------------------------------------------------------------------------
class SolarHybridModel:
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the model components.
        
        Parameters
        ----------
        params : dict, optional
            Dictionary of hyperparameters. Supported keys:
            - rf_n_estimators (default: 200)
            - rf_max_depth (default: None)
            - rf_min_samples_leaf (default: 4)
            - nn_layers (default: (128, 64, 32))
            - nn_learning_rate (default: 0.001)
            - nn_batch_size (default: 128)
            - nn_epochs (default: 10)
        """
        if params is None:
            params = {}
        self.params = params

        # Extract RF Hyperparameters
        rf_n_est = self.params.get('rf_n_estimators', 200)
        rf_depth = self.params.get('rf_max_depth', None)
        rf_leaf = self.params.get('rf_min_samples_leaf', 4)

        # Base tree model
        self.rf = RandomForestClassifier(
            n_estimators=rf_n_est,
            max_depth=rf_depth,
            min_samples_leaf=rf_leaf,
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

        # Feature set (must mirror solar_features.add_features)
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
            upsample_min_bad: int = 500,
            synthetic_frac: float = 0.0):
        """
        Fit the hybrid model on labeled rows in df.

        Parameters
        ----------
        df : DataFrame
            Full dataframe with features (output from add_features) and the label
            column target_col. Label convention: 99 => BAD, others => GOOD.
        target_col : str
            Column name for labels (Flag_GHI, Flag_DNI, ...).
        upsample_min_bad : int
            Minimal number of BAD samples used by upsampling (with replacement).
            This prevents extremely small minority classes from producing huge
            class-weight multipliers.
        synthetic_frac : float
            Fraction (0..1) of training rows to perturb synthetically to create
            additional BAD-like examples (calls inject_synthetic_anomalies).
        """
        # Extract NN Hyperparameters from self.params
        nn_epochs = self.params.get('nn_epochs', 10)
        nn_batch_size = self.params.get('nn_batch_size', 128)
        nn_lr = self.params.get('nn_learning_rate', 0.001)
        nn_layers = self.params.get('nn_layers', (128, 64, 32))

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
        target_bad = max(upsample_min_bad, int(0.005 * n_total)) # 0.5% or min

        if 0 < n_bad < target_bad:
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
        else:
            # no upsampling needed
            X_stack = X_stack.reset_index(drop=True)
            y_arr = y_arr.copy()

        # ---------------- synthetic augmentation (optional) -------------------
        if synthetic_frac > 0.0:
            # Placeholder for future implementation logic
            # For now, we assume X_stack and y_arr are final
            pass

        # ---------------- prepare for NN -------------------------------------
        X_scaled = self.scaler.fit_transform(X_stack)

        # compute class weights (weighted loss)
        classes_present = np.unique(y_arr)
        raw_weights = compute_class_weight(class_weight='balanced', classes=classes_present, y=y_arr)
        # normalize and clip weights to prevent instability
        normed = raw_weights / float(np.mean(raw_weights))
        clipped = np.clip(normed, 1.0, 10.0)

        class_weights = jnp.ones(2)
        for cls, w in zip(classes_present, clipped):
            class_weights = class_weights.at[int(cls)].set(float(w))

        # create tf.data pipeline
        batch_size = max(8, int(nn_batch_size))
        ds = tf.data.Dataset.from_tensor_slices({
            'x': X_scaled.astype('float32'),
            'y': y_arr.astype('int32')
        }).shuffle(8192).batch(batch_size, drop_remainder=True).repeat()

        steps_per_epoch = max(1, X_scaled.shape[0] // batch_size)
        total_steps = int(nn_epochs * steps_per_epoch)

        # ---------------- JAX Training Loop ----------------------------------
        rng = jax.random.PRNGKey(42)
        
        # Initialize DenseNN with dynamic layer sizes from params
        model = DenseNN(layer_sizes=nn_layers)
        
        dummy_input = jnp.ones((1, X_scaled.shape[1]))
        params_init = model.init(rng, dummy_input)['params']

        # Use dynamic learning rate from params
        tx = optax.adam(learning_rate=nn_lr)
        self.nn_state = train_state.TrainState.create(apply_fn=model.apply, params=params_init, tx=tx)

        it = iter(ds)
        for step in range(total_steps):
            batch = next(it)
            jax_batch = {'x': jnp.array(batch['x'].numpy()), 'y': jnp.array(batch['y'].numpy())}
            self.nn_state = train_step(self.nn_state, jax_batch, class_weights)

    def predict(self, df: pd.DataFrame, target_col: str, do_return_probs: bool = False):
        """
        Predict quality flags for new data.

        Parameters
        ----------
        df : DataFrame
            Data with features.
        target_col : str
            Only used for naming the output prob column if needed.
        do_return_probs : bool
            If True, returns (flags, probs).

        Returns
        -------
        flags : np.ndarray
            99 (BAD) or 1 (GOOD)
        [probs] : np.ndarray
            Probability of GOOD class (0..1)
        """
        # 1. Build X (strictly features)
        X_rf = pd.DataFrame(
            {c: (df[c] if c in df.columns else 0.0) for c in self.common_features},
            index=df.index
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # 2. RF Probability
        if self.rf_cal is not None:
            try:
                rf_prob = self.rf_cal.predict_proba(X_rf)[:, 1]
            except Exception:
                rf_prob = self.rf.predict_proba(X_rf)[:, 1]
        else:
            rf_prob = self.rf.predict_proba(X_rf)[:, 1]

        # 3. Stack IF Score
        X_stack = X_rf.copy()
        X_stack['RF_Prob'] = rf_prob

        # If IF model exists, generate score. IF is strictly a *meta-feature* ONLY
        # It must NEVER be used by the RF
        if getattr(self, 'if_det', None) is not None:
            try:
                X_stack['IF_Score'] = self.if_det.decision_function(X_rf)
            except Exception:
                # Fail-safe: never let IF errors crash prediction
                X_stack['IF_Score'] = 0.0
        else:
             if 'IF_Score' in X_stack.columns:
                 X_stack['IF_Score'] = 0.0

        # 4. Scale features for NN
        X_scaled = self.scaler.transform(X_stack)

        # 5. Neural-network inference
        logits = self.nn_state.apply_fn(
            {'params': self.nn_state.params},
            X_scaled
        )

        # Softmax to obtain probabilities
        probs_good = np.array(jax.nn.softmax(logits, axis=1)[:, 1])

        # Binary decision (0 = BAD, 1 = GOOD)
        preds = (probs_good >= 0.5).astype(int)

        # Map to project convention: 99 = BAD, 1 = GOOD
        flags = np.where(preds == 1, 1, 99)

        # 6. Return
        if do_return_probs:
            return flags, probs_good
        return flags