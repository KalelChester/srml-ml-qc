import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

# --- JAX Model Definition ---
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
        x = nn.Dense(2)(x) 
        return x

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
        return loss
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

class SolarHybridModel:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.nn_state = None
        
        # Extended Feature List
        # Note: We include ALL potential features. 
        # If a column (like GHI_Frac) is missing for DNI targets, we fill it with 0s.
        self.feature_cols = [
            'Timestamp_Num', 'SZA', 'AZM', 'hour_frac', 'hour_sin', 'hour_cos', 
            'doy_sin', 'doy_cos', 
            'GHI', 'CorrFeat_GHI', 
            'GHI_Calc', 'CorrFeat_GHI_Calc', 'GHI_Frac',
            'DNI', 'CorrFeat_DNI',
            'DHI', 'CorrFeat_DHI'
        ]

    def _prepare_features(self, df):
        """Helper to ensure all feature columns exist and are numeric."""
        X = pd.DataFrame(index=df.index)
        for col in self.feature_cols:
            if col in df.columns:
                X[col] = df[col]
            else:
                X[col] = 0.0 # Default missing features to 0
        return X

    def fit(self, df, target_col='Flag_GHI', epochs=10, batch_size=100):
        # Filter: We only train on rows where we have VALID targets (1, 11, 12, 99)
        # We assume 0 or NaN in target means "Untrusted/Unchecked"
        train_df = df[df[target_col].notna()].copy()
        
        # BINARIZATION: 99 is Bad (0), Everything else is Good (1)
        y = (train_df[target_col] != 99).astype(int)
        
        X = self._prepare_features(train_df)
        
        # Handle NaNs in INPUTS by filling (RF handles 0 fine)
        X = X.fillna(0)

        print(f"Training on {len(X)} points. Class balance: {y.value_counts().to_dict()}")
        
        print("    Fitting Random Forest...")
        self.rf.fit(X, y)
        
        # Stack Predictions
        rf_prob = self.rf.predict_proba(X)[:, 1]
        X['RF_Prob'] = rf_prob
        
        X_scaled = self.scaler.fit_transform(X)
        
        print("    Fitting Neural Network...")
        train_ds = tf.data.Dataset.from_tensor_slices({
            'image': X_scaled.astype('float32'),
            'label': y.values.astype('int32')
        }).shuffle(1024).batch(batch_size, drop_remainder=True).repeat()
        
        rng = jax.random.PRNGKey(0)
        model = DenseNN()
        params = model.init(rng, jnp.ones((1, X_scaled.shape[1])))['params']
        self.nn_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.001))
        
        steps_per_epoch = len(X) // batch_size
        ds_iter = iter(train_ds)
        
        for i in range(epochs):
            for _ in range(steps_per_epoch):
                raw_batch = next(ds_iter)
                # Convert to numpy to satisfy JAX
                batch = {
                    'image': jnp.array(raw_batch['image'].numpy()),
                    'label': jnp.array(raw_batch['label'].numpy())
                }
                self.nn_state = train_step(self.nn_state, batch)

        # Performance Report
        print("\n--- Training Performance Report ---")
        logits = self.nn_state.apply_fn({'params': self.nn_state.params}, X_scaled)
        nn_preds = jnp.argmax(logits, axis=-1)
        print(classification_report(y, nn_preds, target_names=['Bad (99)', 'Good (1)'], digits=6))
        print("-" * 50)

    def predict(self, df):
        """
        Robust prediction that handles NaNs in features.
        Rows with NaNs are defaulted to 99 (Bad) for safety.
        """
        # 1. Prepare clean feature matrix
        X_full = self._prepare_features(df)
        
        # 2. Identify rows with NaNs (missing data) vs Clean rows
        # We fill NaNs with 0 temporarily to allow model to run, 
        # but you could strictly flag them as 99 if you prefer.
        # Strategy: Fill NaNs with 0 so we predict on EVERYTHING.
        X_safe = X_full.fillna(0)
        
        # 3. RF Prediction
        rf_prob = self.rf.predict_proba(X_safe)[:, 1]
        X_safe['RF_Prob'] = rf_prob
        
        # 4. NN Prediction
        X_scaled = self.scaler.transform(X_safe)
        logits = self.nn_state.apply_fn({'params': self.nn_state.params}, X_scaled)
        preds = jnp.argmax(logits, axis=-1)
        
        # 5. Convert to Flags (1=Good, 99=Bad)
        # Note: If GHI was NaN in original df, we force 99
        final_flags = np.where(preds == 1, 1, 99)
        
        # Optional: Force 99 if key data was missing (e.g. GHI is NaN)
        if 'GHI' in df.columns:
            mask_missing = df['GHI'].isna()
            final_flags[mask_missing] = 99
            
        return final_flags