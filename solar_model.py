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
from sklearn.utils.class_weight import compute_class_weight

# --- JAX Model Definition ---
class DenseNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Increased capacity slightly for fine-tuning
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x) 
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x) # Logits for [Bad(0), Good(1)]
        return x

@jax.jit
def train_step(state, batch, class_weights):
    """
    Training step with CLASS WEIGHTING to fix the imbalance issue.
    """
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        
        # Calculate unweighted loss per sample
        one_hot = jax.nn.one_hot(batch['label'], 2)
        loss_per_sample = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
        
        # Apply manual class weights
        # We look up the weight for each label in the batch
        weights = class_weights[batch['label']] 
        weighted_loss = jnp.mean(loss_per_sample * weights)
        
        return weighted_loss
        
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

class SolarHybridModel:
    def __init__(self):
        # Tuned RF to be less "noisy" but still sensitive
        self.rf = RandomForestClassifier(
            n_estimators=150, 
            class_weight='balanced_subsample', # Better for time-series chunks
            min_samples_leaf=4, # Reduces overfitting to single noise points (helps GHI)
            random_state=42, 
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.nn_state = None
        
        # --- FEATURE GROUPS ---
        # We define relevant features per target to prevent noise leakage.
        self.common_feats = ['Timestamp_Num', 'hour_frac', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 
                             'CorrFeat_DNI', 'CorrFeat_DHI', 'CorrFeat_GHI', 'CorrFeat_GHI_Calc', 'GHI_Frac',
                             'DNI', 'DHI', 'GHI', 'GHI_Calc']
        
        self.features_map = {
            'Flag_DNI': self.common_feats + [],
            'Flag_DHI': self.common_feats + ['Flag_DNI'],
            'Flag_GHI': self.common_feats + ['Flag_DNI', 'Flag_DHI'],
        }
        
        self.current_features = []

    def _prepare_features(self, df, target_name):
        """Prepares features specifically for the requested target."""
        # Determine which features to use
        if target_name in self.features_map:
            cols = self.features_map[target_name]
        else:
            # Fallback for unknown targets
            cols = self.common_feats + [c for c in df.columns if 'GHI' in c or 'DNI' in c]
            
        self.current_features = cols # Save for prediction time
        
        X = pd.DataFrame(index=df.index)
        for col in cols:
            if col in df.columns:
                X[col] = df[col]
            else:
                X[col] = 0.0
        
        # Safety fill
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        return X

    def fit(self, df, target_col='Flag_GHI', epochs=15, batch_size=128):
        print(f"    Target: {target_col}")
        train_df = df[df[target_col].notna()].copy()
        
        # MAPPING: 99 -> 0 (Bad), Else -> 1 (Good)
        y = np.where(train_df[target_col] == 99, 0, 1)
        
        # 1. Prepare Target-Specific Features
        X = self._prepare_features(train_df, target_col)
        
        # 2. RF Training
        print(f"    Fitting Random Forest on {len(X)} rows...")
        self.rf.fit(X, y)
        
        # 3. Stack RF Probability
        # The NN gets the RF's opinion as a feature
        rf_prob = self.rf.predict_proba(X)[:, 1] # Prob of "Good"
        X_stacked = X.copy()
        X_stacked['RF_Prob'] = rf_prob
        
        # 4. NN Training Prep
        X_scaled = self.scaler.fit_transform(X_stacked)
        
        # --- CRITICAL: CALCULATE CLASS WEIGHTS ---
        # This tells the NN that "0" (Bad) is much more important than "1" (Good)
        # Because we only have 2 classes (0 and 1), we get an array of size 2.
        classes = np.unique(y)
        weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        
        # Map these to a JAX array [Weight_for_0, Weight_for_1]
        # We ensure index 0 corresponds to class 0, etc.
        class_weights_tensor = jnp.zeros(2)
        for cls, w in zip(classes, weights_array):
            class_weights_tensor = class_weights_tensor.at[int(cls)].set(w)
            
        print(f"    NN Class Weights (0=Bad, 1=Good): {class_weights_tensor}")

        # TF Dataset
        train_ds = tf.data.Dataset.from_tensor_slices({
            'image': X_scaled.astype('float32'),
            'label': y.astype('int32')
        }).shuffle(2048).batch(batch_size, drop_remainder=True).repeat()
        
        # Init JAX
        rng = jax.random.PRNGKey(42)
        model = DenseNN()
        params = model.init(rng, jnp.ones((1, X_scaled.shape[1])))['params']
        self.nn_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.001))
        
        steps_per_epoch = len(X) // batch_size
        ds_iter = iter(train_ds)
        
        print("    Fitting Neural Network (with weighted loss)...")
        for i in range(epochs):
            for _ in range(steps_per_epoch):
                raw_batch = next(ds_iter)
                batch = {
                    'image': jnp.array(raw_batch['image'].numpy()),
                    'label': jnp.array(raw_batch['label'].numpy())
                }
                self.nn_state = train_step(self.nn_state, batch, class_weights_tensor)

        # Performance Check
        logits = self.nn_state.apply_fn({'params': self.nn_state.params}, X_scaled)
        nn_preds = jnp.argmax(logits, axis=-1)
        print(classification_report(y, nn_preds, target_names=['Bad (99)', 'Good (1)'], digits=4))
        print("-" * 50)

    def predict(self, df, target_col):
        # 1. Prepare clean features (using SAME columns as fit)
        X = self._prepare_features(df, target_col)
        
        # 2. RF Prediction (Probabilities)
        rf_prob = self.rf.predict_proba(X)[:, 1]
        
        # 3. Stack
        X_stacked = X.copy()
        X_stacked['RF_Prob'] = rf_prob
        
        # 4. NN Prediction
        X_scaled = self.scaler.transform(X_stacked)
        logits = self.nn_state.apply_fn({'params': self.nn_state.params}, X_scaled)
        preds = jnp.argmax(logits, axis=-1)
        
        # 5. Convert to Flags (0->99, 1->1)
        final_flags = np.where(preds == 0, 99, 1)
            
        return final_flags