"""
solar_model.py
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
from typing import Optional, Tuple

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
# Neural network (small, intentionally shallow)
# -----------------------------------------------------------------------------
class DenseNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
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
    def __init__(self):
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

        # Feature set (must mirror solar_features.add_features)
        self.common_features = [
            'Timestamp_Num',
            'hour_sin', 'hour_cos',
            'doy_sin', 'doy_cos',
            'SZA', 'elevation', 'CSI',
            'QC_PhysicalFail', 'Temperature',
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
            epochs: int = 20,
            batch_size: int = 64,
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
        epochs : int
            Number of epochs to run the NN training loop (small by default).
        batch_size : int
            NN batch size; will be adjusted to dataset size if necessary.
        upsample_min_bad : int
            Minimal number of BAD samples used by upsampling (with replacement).
            This prevents extremely small minority classes from producing huge
            class-weight multipliers.
        synthetic_frac : float
            Fraction (0..1) of training rows to perturb synthetically to create
            additional BAD-like examples (calls inject_synthetic_anomalies).
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

        # ---------------- NN training (JAX/Flax) ----------------------------
        # Prepare tf.data.Dataset to supply data to the JAX step loop.
        # We use `.repeat()` and iterate for a deterministic number of steps.
        batch_size = max(8, int(batch_size))
        ds = tf.data.Dataset.from_tensor_slices({
            'x': X_scaled.astype('float32'),
            'y': y_arr.astype('int32')
        }).shuffle(8192).batch(batch_size, drop_remainder=True).repeat()

        steps_per_epoch = max(1, X_scaled.shape[0] // batch_size)
        total_steps = int(epochs * steps_per_epoch)
        print(f"    [fit] Training NN: samples={X_scaled.shape[0]} batch_size={batch_size} steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

        # init model
        rng = jax.random.PRNGKey(42)
        model = DenseNN()
        params = model.init(rng, jnp.ones((1, X_scaled.shape[1])))['params']
        self.nn_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))

        it = iter(ds)
        for step in range(total_steps):
            batch = next(it)
            jax_batch = {'x': jnp.array(batch['x'].numpy()), 'y': jnp.array(batch['y'].numpy())}
            self.nn_state = train_step(self.nn_state, jax_batch, class_weights)
            if (step + 1) % max(1, (steps_per_epoch * max(1, epochs // 5))) == 0:
                # light progress log every ~20% of training
                print(f"    [fit] NN training progress: step {step+1}/{total_steps}")

        # Diagnostics: training report on the training set
        logits = self.nn_state.apply_fn({'params': self.nn_state.params}, X_scaled)
        preds = jnp.argmax(logits, axis=1)
        try:
            print("    [fit] NN training classification report (on stacked dataset):")
            print(classification_report(y_arr, np.array(preds), target_names=['BAD', 'GOOD'], digits=4))
        except Exception:
            pass

        # store last-fit artifacts (RF, rf_cal, scaler, if_det, nn_state) on self
        # they are already attributes of self

    # ---------------- predict --------------------------------------------
    def predict(self, df: pd.DataFrame, target_col: str, do_return_probs: bool = False):
        """
        Predict quality flags for new data.

        Design invariants (VERY IMPORTANT):
        -----------------------------------
        1. The RandomForest (and its calibrated wrapper) MUST see *exactly*
        the same feature set at predict-time as it did at fit-time.
        No extra columns. No missing columns. Same names.

        2. Meta-features (RF probability, IsolationForest score, etc.)
        are *never* fed back into the RF.
        They are consumed ONLY by the neural-network stacker.

        3. Probabilities returned represent P(GOOD).
        - 1.0 => fully confident GOOD
        - 0.0 => fully confident BAD

        Parameters
        ----------
        df : pandas.DataFrame
            Feature-engineered dataframe (output of add_features).
        target_col : str
            One of {'Flag_GHI', 'Flag_DNI', 'Flag_DHI'}.
            Included for symmetry and future per-target specialization.
        do_return_probs : bool
            If True, return (flags, probs).
            If False, return flags only.

        Returns
        -------
        np.ndarray or (np.ndarray, np.ndarray)
            flags: 99 = BAD, 1 = GOOD
            probs: probability of GOOD (if requested)
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
        # 5. Neural-network inference
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 6. Return
        # ------------------------------------------------------------------
        if do_return_probs:
            return flags, probs_good

        return flags
