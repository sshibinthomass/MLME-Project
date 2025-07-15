#!/usr/bin/env python3
"""
Shibin Paul
Nishitkumar Karkar
Sankar Nair
Aadhithya Krishnakumar


End-to-end training script for the SFC project (Machine Learning Methods for
Engineers, SS 25).

Key features
------------
* Per-cluster NARX models
* Input order:  [y_t , u_t , y_{t-1} , u_{t-1} , … , y_{t-LAG}]
* Units: **metres** for d10/d50/d90 everywhere  ➜  no µm scaling headaches
* Early-Stopping + Checkpoint to avoid over-/under-fitting
* Weighted MSE  (PSD columns get higher loss weight)
* Quantile Regression nets for τ = 0.1 & 0.9  +  Conformal deltas for 90 % PI
  ─→  calibrated CQR intervals
* Deterministic seeds  ➜  reproducible runs

 spec (see CrysID_MLME25.pdf).
"""

#%% ──  Imports  ──────────────────────────────────────────────────────────────
from __future__ import annotations

import os, random, json, pickle, shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, optimizers, Sequential, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#%% --- Unit configuration ---------------------------------------------------
#USE_MICRONS = False        # True  ➜ internally work in µm  (recommended by you)
PSD_COLS    = ('d10', 'd50', 'd90')


# ──  Reproducibility  ──────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ──  User paths & global constants  ────────────────────────────────────────
RAW_ROOT   = Path(r"Data/RAW DATA")
MODEL_DIR  = Path(r"model_5files18") 
# clean slate (avoids shape mismatches when you change LAG etc.)
if MODEL_DIR.exists():
    shutil.rmtree(MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR/"narx").mkdir()
(MODEL_DIR/"qr").mkdir()

LAG             = 20                # number of past steps
N_CLUSTERS      = 2
EPOCHS_NARX     = 300
EPOCHS_QR       = 100
BATCH_SIZE_NARX      = 32
BATCH_SIZE_QR       = 32
QUANTILES       = [0.1, 0.9]
OUTPUT_WEIGHTS  = np.array([6, 8, 30, 30, 50, 10], dtype="float32") # higher weight to PSD (6 outputs)

# Add early stopping configuration for NARX
ES_PATIENCE_NARX = 8        # Reduced from 20
ES_MIN_DELTA = 1e-6         # Minimum improvement threshold
ES_RESTORE_BEST = True

# Column layout  (matches report)
STATE_COLS = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
EXOG_COLS  = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal',
              'c_in', 'T_PM_in', 'T_TM_in'] # Removed constant variables
OUTPUT_COLS = STATE_COLS
CLUST_COLS  = STATE_COLS + EXOG_COLS

# ──  Helper functions  ─────────────────────────────────────────────────────
def read_txt(path: Path) -> pd.DataFrame:
    """Read TAB-separated text file into DataFrame (all numeric)."""
    return pd.read_csv(path, sep='\t', engine='python'
                      ).apply(pd.to_numeric, errors='coerce')

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaNs + obvious sensor out-of-range artefacts."""
    df = df.dropna(subset=CLUST_COLS)
    df = df[(df.T_PM.between(250, 400)) & (df.T_TM.between(250, 400))
            & (df.d10>0) & (df.d50>0) & (df.d90>0)
            & (df.mf_PM>=0) & (df.mf_TM>=0) & (df.Q_g>=0)]
    return df.reset_index(drop=True)

def clean_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Improved: 
      - Log-transform d10/d50/d90 before outlier handling
      - Use stricter IQR (1.5x) for d10/d50/d90
      - Add engineered features after cleaning
    """
    available_cols = [col for col in CLUST_COLS if col in df.columns]
    df = df.dropna(subset=available_cols) #Drop empty rows

    # Log-transform d10/d50/d90 before outlier handling
    for col in ['d10', 'd50', 'd90']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    for column in df.columns:
        if column in available_cols:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            if column in ['T_PM', 'T_TM']:
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                vals = df[column].values.copy()
                for i in range(len(vals)):
                    if not (lower_bound <= vals[i] <= upper_bound):
                        prev_idx = i - 1
                        while prev_idx >= 0 and not (lower_bound <= vals[prev_idx] <= upper_bound):
                            prev_idx -= 1
                        next_idx = i + 1
                        while next_idx < len(vals) and not (lower_bound <= vals[next_idx] <= upper_bound):
                            next_idx += 1
                        if prev_idx >= 0 and next_idx < len(vals):
                            vals[i] = 0.5 * (vals[prev_idx] + vals[next_idx])
                        elif prev_idx >= 0:
                            vals[i] = vals[prev_idx]
                        elif next_idx < len(vals):
                            vals[i] = vals[next_idx]
                df[column] = vals
            elif column == 'c':
                lower_bound = Q1 - 6.0 * IQR
                upper_bound = Q3 + 6.0 * IQR
                vals = df[column].values.copy()
                mask = ~((lower_bound <= vals) & (vals <= upper_bound))
                i = 0
                n = len(vals)
                while i < n:
                    if mask[i]:
                        run_start = i
                        while i < n and mask[i]:
                            i += 1
                        run_end = i
                        prev_idx = run_start - 1
                        next_idx = run_end
                        prev_val = vals[prev_idx] if prev_idx >= 0 else None
                        next_val = vals[next_idx] if next_idx < n else None
                        if prev_val is not None and next_val is not None:
                            for j in range(run_start, run_end):
                                alpha = (j - run_start + 1) / (run_end - run_start + 1)
                                vals[j] = (1 - alpha) * prev_val + alpha * next_val
                        elif prev_val is not None:
                            vals[run_start:run_end] = prev_val
                        elif next_val is not None:
                            vals[run_start:run_end] = next_val
                    else:
                        i += 1
                df[column] = vals
            elif column in ['d10', 'd50', 'd90']:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            else:
                lower_bound = Q1 - 2 * IQR
                upper_bound = Q3 + 2 * IQR
                df[column] = df[column].apply(
                    lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
                )

    # Add engineered features (span, ratios) in original scale
    if all(col in df.columns for col in ['d10', 'd50', 'd90']):
        d10 = np.expm1(df['d10'])
        d50 = np.expm1(df['d50'])
        d90 = np.expm1(df['d90'])
        df['span_d90_d10'] = d90 - d10
        df['ratio_d90_d50'] = d90 / d50.replace(0, np.nan)
        df['ratio_d50_d10'] = d50 / d10.replace(0, np.nan)

    return df


def remove_trash_files(file_path_list):
    """
    Move the trash files with extreme d-values to different directory 
    """
    for path in file_path_list:
        df = read_txt(path)
        for column in df.columns:
            if column in ['d10', 'd90', 'd50'] and df[column].median() > 1:
                shutil.copy(path, trash_dir)


def preprocess(path: Path) -> pd.DataFrame:
    df = clean_iqr(read_txt(path))
    return df


# ----------  Lag-matrix creation (newest-to-oldest)  ----------------------
def make_xy(df: pd.DataFrame, lag=LAG
           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build NARX design-matrix X and target Y.

    X row format (newest-first):
        [ y_t, u_t, y_{t-1}, u_{t-1}, … , y_{t-lag}, u_{t-lag} ]
    """
    hist_size = len(STATE_COLS) + len(EXOG_COLS)
    X, Y = [], []
    for i in range(lag, len(df)-1):
        # newest-to-oldest slice
        hist = []
        for l in range(0, lag+1):                          # 0 … LAG
            idx = i - l
            hist.extend(df[STATE_COLS + EXOG_COLS].iloc[idx].values)
        X.append(hist)
        Y.append(df[STATE_COLS].iloc[i+1].values)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)

# ----------  Custom losses / builders  ------------------------------------
def weighted_mse(y_true, y_pred):
    """MSE with per-output weights (PSD columns matter more)."""
    w = tf.constant(OUTPUT_WEIGHTS, dtype=y_true.dtype)
    return tf.reduce_mean(tf.reduce_mean(tf.square(y_true - y_pred) * w, axis=-1))

def build_narx(input_dim: int, output_dim: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,))
    
    # First block
    x = layers.Dense(1024, activation='selu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Second block with residual connection (same size)
    shortcut = x
    x = layers.Dense(1024, activation='selu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Add()([shortcut, x])
    
    # Third block with projection shortcut (1024 -> 768)
    shortcut = x
    x = layers.Dense(768, activation='selu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    shortcut_proj = layers.Dense(768)(shortcut)
    x = layers.Add()([shortcut_proj, x])
    
    # Fourth block with projection shortcut (768 -> 512)
    shortcut = x
    x = layers.Dense(512, activation='selu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    shortcut_proj = layers.Dense(512)(shortcut)
    x = layers.Add()([shortcut_proj, x])
    
    # Fifth block with projection shortcut (512 -> 256)
    shortcut = x
    x = layers.Dense(256, activation='selu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    shortcut_proj = layers.Dense(256)(shortcut)
    x = layers.Add()([shortcut_proj, x])
    
    # Sixth block with projection shortcut (256 -> 128)
    shortcut = x
    x = layers.Dense(128, activation='selu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    shortcut_proj = layers.Dense(128)(shortcut)
    x = layers.Add()([shortcut_proj, x])
    
    outputs = layers.Dense(output_dim)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
    

def build_advanced_qr(input_dim: int) -> tf.keras.Model:
    """Advanced QR model with attention mechanism and better regularization."""
    inputs = layers.Input(shape=(input_dim,))
    
    # Feature extraction with attention
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Self-attention mechanism for feature importance
    attention = layers.Dense(512, activation='softmax')(x)
    x = layers.Multiply()([x, attention])
    
    # Residual blocks with skip connections
    for i in range(3):
        residual = x
        x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        
        # Skip connection if dimensions match
        if residual.shape[-1] == x.shape[-1]:
            x = layers.Add()([residual, x])
    
    # Final layers with adaptive activation
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Output with uncertainty estimation
    outputs = layers.Dense(1, activation='linear')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_ensemble_qr(input_dim: int) -> tf.keras.Model:
    """Ensemble QR model that combines multiple predictions."""
    inputs = layers.Input(shape=(input_dim,))
    
    # Multiple parallel paths
    paths = []
    for i in range(3):
        path = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
        path = layers.BatchNormalization()(path)
        path = layers.Dropout(0.3)(path)
        
        path = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(path)
        path = layers.BatchNormalization()(path)
        path = layers.Dropout(0.2)(path)
        
        path = layers.Dense(64, activation='relu')(path)
        paths.append(path)
    
    # Combine paths
    combined = layers.Concatenate()(paths)
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_deep_qr(input_dim: int) -> tf.keras.Model:
    """Enhanced deep QR model with better architecture."""
    inputs = layers.Input(shape=(input_dim,))
    
    # Initial feature extraction
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Dense blocks with residual connections
    for i in range(4):
        residual = x
        x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Residual connection
        if residual.shape[-1] == x.shape[-1]:
            x = layers.Add()([residual, x])
    
    # Final layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def pinball_loss(tau: float):
    """Enhanced pinball / quantile loss for τ with regularization."""
    def loss(y, y_hat):
        e = y - y_hat
        pinball = tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
        
        # Add Huber loss component for robustness
        huber_delta = 1.0
        huber_loss = tf.reduce_mean(tf.where(
            tf.abs(e) <= huber_delta,
            0.5 * tf.square(e),
            huber_delta * tf.abs(e) - 0.5 * huber_delta ** 2
        ))
        
        # Combine pinball and Huber loss
        return pinball + 0.1 * huber_loss
    return loss

def adaptive_pinball_loss(tau: float):
    """Adaptive pinball loss that adjusts based on prediction uncertainty."""
    def loss(y, y_hat):
        e = y - y_hat
        
        # Standard pinball loss
        pinball = tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
        
        # Adaptive component based on error magnitude
        error_magnitude = tf.abs(e)
        adaptive_weight = tf.exp(-error_magnitude / tf.reduce_mean(error_magnitude))
        
        # Weighted pinball loss
        weighted_pinball = tf.reduce_mean(adaptive_weight * tf.maximum(tau * e, (tau - 1) * e))
        
        return 0.7 * pinball + 0.3 * weighted_pinball
    return loss

def quantile_huber_loss(tau: float, delta: float = 1.0):
    """Quantile Huber loss combining pinball and Huber loss."""
    def loss(y, y_hat):
        e = y - y_hat
        
        # Pinball component
        pinball = tf.maximum(tau * e, (tau - 1) * e)
        
        # Huber component
        huber = tf.where(
            tf.abs(e) <= delta,
            0.5 * tf.square(e),
            delta * tf.abs(e) - 0.5 * delta ** 2
        )
        
        # Combine both losses
        return tf.reduce_mean(pinball + 0.1 * huber)
    return loss

class PrintMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics_str = " ".join([f"{k}={v:.6f}" for k, v in logs.items()])
        print(f"Epoch {epoch+1}: {metrics_str}")

class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=5, initial_lr=1e-6, target_lr=1e-4):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * epoch / self.warmup_epochs
            self.model.optimizer.learning_rate.assign(lr)

class ConvergenceCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=3):
        super().__init__()
        self.patience = patience
        self.wait = 0
        self.best_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get('val_loss', float('inf'))
        
        if current_loss < self.best_loss - ES_MIN_DELTA:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs (no improvement for {self.patience} epochs)")
            self.model.stop_training = True

# ──────────────────────────────────────────────────────────────────────────
# 1. Split RAW files once into *train* and *calibration* sub-folders.
# --------------------------------------------------------------------------
train_dir, calib_dir = RAW_ROOT/'train', RAW_ROOT/'calib'
trash_dir = RAW_ROOT/'trash'


## Create trash_dir and copy the trash files from dataset to the trash_dir
if not trash_dir.exists(): 
    trash_dir.mkdir()
    remove_trash_files(RAW_ROOT.glob("*.txt"))   

if not train_dir.exists():
    train_dir.mkdir(); calib_dir.mkdir()
    files = list(RAW_ROOT.glob("*.txt"))
    files = [f_path for f_path in files if f_path not in list(trash_dir.glob("*.txt")) ] # Ignore the trash files from the train dataset
    random.shuffle(files)
    n_cal = int(0.2 * len(files))
    for p in files[n_cal:]: shutil.copy(p, train_dir/p.name)
    for p in files[:n_cal]: shutil.copy(p, calib_dir/p.name)

# ──────────────────────────────────────────────────────────────────────────
#%% 2.5. Data Visualization Before and After Preprocessing
# --------------------------------------------------------------------------
print("\n" + "="*60)
print("📊 DATA VISUALIZATION BEFORE AND AFTER PREPROCESSING")
print("="*60)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create visualization directory
viz_dir = MODEL_DIR / 'data_visualization'
viz_dir.mkdir(exist_ok=True)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Get train files for visualization
train_files = sorted(train_dir.glob("*.txt"))

# Sample a few files for visualization (to avoid overwhelming plots)
sample_files = train_files[:5]  # First 5 files
print(f"\n📁 Analyzing {len(sample_files)} sample files for visualization...")

# Collect data for visualization
raw_data = []
cleaned_data = []
iqr_cleaned_data = []

for p in sample_files:
    # Raw data
    raw_df = read_txt(p)
    raw_data.append(raw_df)
    
    # Cleaned data (original method)
    cleaned_df = clean_df(raw_df)
    cleaned_data.append(cleaned_df)
    
    # IQR cleaned data (current method)
    iqr_df = clean_iqr(raw_df)
    iqr_cleaned_data.append(iqr_df)

# Combine all data for analysis
raw_combined = pd.concat(raw_data, ignore_index=True)
cleaned_combined = pd.concat(cleaned_data, ignore_index=True)
iqr_combined = pd.concat(iqr_cleaned_data, ignore_index=True)

print(f"\n📈 Data Statistics Summary:")
print(f"Raw data: {len(raw_combined)} rows")
print(f"Cleaned (original): {len(cleaned_combined)} rows")
print(f"Cleaned (IQR): {len(iqr_combined)} rows")

# 1. Data Loss Analysis
print(f"\n📉 Data Loss Analysis:")
for col in CLUST_COLS:
    raw_count = raw_combined[col].notna().sum()
    cleaned_count = cleaned_combined[col].notna().sum()
    iqr_count = iqr_combined[col].notna().sum()
    
    print(f"  {col}:")
    print(f"    Raw: {raw_count} valid values")
    print(f"    Cleaned (original): {cleaned_count} ({cleaned_count/raw_count*100:.1f}%)")
    print(f"    Cleaned (IQR): {iqr_count} ({iqr_count/raw_count*100:.1f}%)")

# 2. Distribution Comparison Plots
print(f"\n🎨 Creating distribution comparison plots...")

# Create subplots for each column - adjust layout for 14 columns
n_cols = 4
n_rows = (len(CLUST_COLS) + n_cols - 1) // n_cols  # Ceiling division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(CLUST_COLS):
    ax = axes[i]
    
    # Plot histograms
    ax.hist(raw_combined[col].dropna(), bins=50, alpha=0.5, label='Raw', density=True, color='red')
    ax.hist(cleaned_combined[col].dropna(), bins=50, alpha=0.5, label='Cleaned (Original)', density=True, color='blue')
    ax.hist(iqr_combined[col].dropna(), bins=50, alpha=0.5, label='Cleaned (IQR)', density=True, color='green')
    
    ax.set_title(f'{col} Distribution')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Remove extra subplots
for i in range(len(CLUST_COLS), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(viz_dir / 'distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Box Plots for Outlier Detection
print(f"📦 Creating box plots for outlier analysis...")

# Create subplots for box plots - adjust layout for 14 columns
n_cols = 4
n_rows = (len(CLUST_COLS) + n_cols - 1) // n_cols  # Ceiling division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(CLUST_COLS):
    ax = axes[i]
    
    # Prepare data for box plot
    data_to_plot = []
    labels = []
    
    if len(raw_combined[col].dropna()) > 0:
        data_to_plot.append(raw_combined[col].dropna())
        labels.append('Raw')
    
    if len(cleaned_combined[col].dropna()) > 0:
        data_to_plot.append(cleaned_combined[col].dropna())
        labels.append('Cleaned (Original)')
    
    if len(iqr_combined[col].dropna()) > 0:
        data_to_plot.append(iqr_combined[col].dropna())
        labels.append('Cleaned (IQR)')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    ax.set_title(f'{col} Box Plot')
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45)

# Remove extra subplots
for i in range(len(CLUST_COLS), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig(viz_dir / 'box_plots.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Statistical Summary Table
print(f"📊 Creating statistical summary...")

stats_summary = {}
for col in CLUST_COLS:
    stats_summary[col] = {
        'raw': {
            'count': raw_combined[col].notna().sum(),
            'mean': raw_combined[col].mean(),
            'std': raw_combined[col].std(),
            'min': raw_combined[col].min(),
            'max': raw_combined[col].max(),
            'q25': raw_combined[col].quantile(0.25),
            'q75': raw_combined[col].quantile(0.75)
        },
        'cleaned_original': {
            'count': cleaned_combined[col].notna().sum(),
            'mean': cleaned_combined[col].mean(),
            'std': cleaned_combined[col].std(),
            'min': cleaned_combined[col].min(),
            'max': cleaned_combined[col].max(),
            'q25': cleaned_combined[col].quantile(0.25),
            'q75': cleaned_combined[col].quantile(0.75)
        },
        'cleaned_iqr': {
            'count': iqr_combined[col].notna().sum(),
            'mean': iqr_combined[col].mean(),
            'std': iqr_combined[col].std(),
            'min': iqr_combined[col].min(),
            'max': iqr_combined[col].max(),
            'q25': iqr_combined[col].quantile(0.25),
            'q75': iqr_combined[col].quantile(0.75)
        }
    }

# Save statistical summary
import json
with open(viz_dir / 'statistical_summary.json', 'w') as f:
    json.dump(stats_summary, f, indent=2, default=str)

# 5. Correlation Analysis
print(f"🔗 Creating correlation analysis...")

# Calculate correlations for each dataset
raw_corr = raw_combined[CLUST_COLS].corr()
cleaned_corr = cleaned_combined[CLUST_COLS].corr()
iqr_corr = iqr_combined[CLUST_COLS].corr()

# Plot correlation matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Raw data correlation
im1 = axes[0].imshow(raw_corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_title('Raw Data Correlation')
axes[0].set_xticks(range(len(CLUST_COLS)))
axes[0].set_yticks(range(len(CLUST_COLS)))
axes[0].set_xticklabels(CLUST_COLS, rotation=45)
axes[0].set_yticklabels(CLUST_COLS)

# Add correlation values
for i in range(len(CLUST_COLS)):
    for j in range(len(CLUST_COLS)):
        text = axes[0].text(j, i, f'{raw_corr.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)

# Cleaned data correlation
im2 = axes[1].imshow(cleaned_corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[1].set_title('Cleaned (Original) Correlation')
axes[1].set_xticks(range(len(CLUST_COLS)))
axes[1].set_yticks(range(len(CLUST_COLS)))
axes[1].set_xticklabels(CLUST_COLS, rotation=45)
axes[1].set_yticklabels(CLUST_COLS)

for i in range(len(CLUST_COLS)):
    for j in range(len(CLUST_COLS)):
        text = axes[1].text(j, i, f'{cleaned_corr.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)

# IQR cleaned data correlation
im3 = axes[2].imshow(iqr_corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[2].set_title('Cleaned (IQR) Correlation')
axes[2].set_xticks(range(len(CLUST_COLS)))
axes[2].set_yticks(range(len(CLUST_COLS)))
axes[2].set_xticklabels(CLUST_COLS, rotation=45)
axes[2].set_yticklabels(CLUST_COLS)

for i in range(len(CLUST_COLS)):
    for j in range(len(CLUST_COLS)):
        text = axes[2].text(j, i, f'{iqr_corr.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im3, ax=axes, shrink=0.8)
plt.tight_layout()
plt.savefig(viz_dir / 'correlation_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# 6. Time Series Visualization (for one file)
print(f"⏰ Creating time series visualization...")

if len(sample_files) > 0:
    sample_file = sample_files[0]
    raw_sample = read_txt(sample_file)
    cleaned_sample = clean_iqr(raw_sample)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot first 6 columns as time series
    for i, col in enumerate(CLUST_COLS[:6]):
        row = i // 2
        col_idx = i % 2
        
        axes[row, col_idx].plot(raw_sample.index, raw_sample[col], 
                               label='Raw', alpha=0.7, color='red')
        axes[row, col_idx].plot(cleaned_sample.index, cleaned_sample[col], 
                               label='Cleaned', alpha=0.7, color='blue')
        axes[row, col_idx].set_title(f'{col} Time Series')
        axes[row, col_idx].set_xlabel('Time Step')
        axes[row, col_idx].set_ylabel(col)
        axes[row, col_idx].legend()
        axes[row, col_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'time_series_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

# 7. Outlier Analysis
print(f"🔍 Creating outlier analysis...")

outlier_analysis = {}
for col in CLUST_COLS:
    raw_data_col = raw_combined[col].dropna()
    iqr_data_col = iqr_combined[col].dropna()
    
    # Calculate IQR for raw data
    Q1 = raw_data_col.quantile(0.25)
    Q3 = raw_data_col.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers
    outliers = ((raw_data_col < lower_bound) | (raw_data_col > upper_bound)).sum()
    total_points = len(raw_data_col)
    
    outlier_analysis[col] = {
        'total_points': total_points,
        'outliers_removed': outliers,
        'outlier_percentage': (outliers / total_points) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR
    }

# Save outlier analysis
with open(viz_dir / 'outlier_analysis.json', 'w') as f:
    json.dump(outlier_analysis, f, indent=2, default=str)

# Print outlier summary
print(f"\n📊 Outlier Analysis Summary:")
for col, stats in outlier_analysis.items():
    print(f"  {col}:")
    print(f"    Total points: {stats['total_points']}")
    print(f"    Outliers removed: {stats['outliers_removed']} ({stats['outlier_percentage']:.1f}%)")
    print(f"    IQR bounds: [{stats['lower_bound']:.2e}, {stats['upper_bound']:.2e}]")

# 8. Data Quality Report
print(f"\n📋 Creating data quality report...")

quality_report = {
    'preprocessing_summary': {
        'files_analyzed': len(sample_files),
        'raw_total_rows': len(raw_combined),
        'cleaned_total_rows': len(iqr_combined),
        'data_retention_rate': len(iqr_combined) / len(raw_combined) * 100
    },
    'column_analysis': {}
}

for col in CLUST_COLS:
    raw_nulls = raw_combined[col].isna().sum()
    cleaned_nulls = iqr_combined[col].isna().sum()
    
    quality_report['column_analysis'][col] = {
        'raw_null_count': int(raw_nulls),
        'cleaned_null_count': int(cleaned_nulls),
        'null_reduction': int(raw_nulls - cleaned_nulls),
        'outlier_analysis': outlier_analysis[col]
    }

# Save quality report
with open(viz_dir / 'data_quality_report.json', 'w') as f:
    json.dump(quality_report, f, indent=2, default=str)

print(f"\n✅ Data visualization complete!")
print(f"📁 Visualizations saved to: {viz_dir}")
print(f"📊 Data Quality Summary:")
print(f"   Raw data rows: {len(raw_combined)}")
print(f"   Cleaned data rows: {len(iqr_combined)}")
print(f"   Data retention: {len(iqr_combined)/len(raw_combined)*100:.1f}%")

print("\n" + "="*60)

# ──────────────────────────────────────────────────────────────────────────
#%% 2.5. Clustering Analysis and Scoring
# --------------------------------------------------------------------------
print("\n" + "="*60)
print("🔍 CLUSTERING ANALYSIS AND SCORING")
print("="*60)

feat = []
for p in train_files:
    df  = preprocess(p)
    arr = df[CLUST_COLS].values
    feat.append(np.concatenate([arr.mean(0), arr.std(0), arr.min(0), arr.max(0)]))
feat = np.vstack(feat)

sc_feat = StandardScaler().fit(feat)
feat_s  = sc_feat.transform(feat)
kmeans  = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit(feat_s)

pickle.dump(sc_feat, (MODEL_DIR/'feature_scaler.pkl').open('wb'))
pickle.dump(kmeans , (MODEL_DIR/'kmeans_model.pkl' ).open('wb'))

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Get cluster assignments
cluster_labels = kmeans.labels_
unique_clusters = np.unique(cluster_labels)

print(f"\n📊 Clustering Results:")
print(f"Number of clusters: {len(unique_clusters)}")
print(f"Total files: {len(train_files)}")

# 1. Cluster Distribution Analysis
print(f"\n📈 Cluster Distribution:")
cluster_counts = np.bincount(cluster_labels)
for cid in unique_clusters:
    count = cluster_counts[cid]
    percentage = (count / len(train_files)) * 100
    print(f"  Cluster {cid}: {count} files ({percentage:.1f}%)")

# 2. Clustering Quality Metrics
print(f"\n🎯 Clustering Quality Metrics:")
silhouette = silhouette_score(feat_s, cluster_labels)
calinski = calinski_harabasz_score(feat_s, cluster_labels)
davies = davies_bouldin_score(feat_s, cluster_labels)

print(f"  Silhouette Score: {silhouette:.4f} (Higher is better, range: -1 to 1)")
print(f"  Calinski-Harabasz Score: {calinski:.2f} (Higher is better)")
print(f"  Davies-Bouldin Score: {davies:.4f} (Lower is better)")

# 3. Feature Importance Analysis
print(f"\n🔍 Feature Importance Analysis:")
feature_names = []
for col in CLUST_COLS:
    feature_names.extend([f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max'])

feature_importance = np.zeros(len(feature_names))
for i in range(len(feature_names)):
    if i < len(feat_s[0]):
        overall_mean = np.mean(feat_s[:, i])
        between_cluster_var = 0
        within_cluster_var = 0
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_mean = np.mean(feat_s[cluster_mask, i])
            cluster_size = np.sum(cluster_mask)
            
            between_cluster_var += cluster_size * (cluster_mean - overall_mean)**2
            within_cluster_var += np.sum((feat_s[cluster_mask, i] - cluster_mean)**2)
        
        # Add small epsilon to prevent division by zero and handle numerical instability
        epsilon = 1e-10
        if within_cluster_var > epsilon:
            feature_importance[i] = between_cluster_var / (within_cluster_var + epsilon)
        else:
            # If within-cluster variance is too small, use a different metric
            feature_importance[i] = between_cluster_var / epsilon

# Show top features
top_features = 10
top_indices = np.argsort(feature_importance)[-top_features:]
print(f"  Top {top_features} features driving cluster separation:")
for i, idx in enumerate(top_indices):
    print(f"    {i+1}. {feature_names[idx]}: {feature_importance[idx]:.3f}")

# Diagnostic: Why certain features might not appear
print(f"\n🔍 Diagnostic: Feature Variance Analysis:")
for col in CLUST_COLS:
    col_idx = CLUST_COLS.index(col)
    mean_idx = col_idx * 4
    std_idx = col_idx * 4 + 1
    min_idx = col_idx * 4 + 2
    max_idx = col_idx * 4 + 3
    
    print(f"  {col}:")
    print(f"    Mean importance: {feature_importance[mean_idx]:.3f}")
    print(f"    Std importance:  {feature_importance[std_idx]:.3f}")
    print(f"    Min importance:  {feature_importance[min_idx]:.3f}")
    print(f"    Max importance:  {feature_importance[max_idx]:.3f}")
    
    # Check if this feature has low variance
    if feature_importance[std_idx] < 0.1:
        print(f"    ⚠️  Low std importance - may indicate low variance in {col}")
    
    # Show actual variance values for debugging
    feature_values = feat[:, std_idx]  # Get std values for this column
    print(f"    Actual std values - min: {np.min(feature_values):.2e}, max: {np.max(feature_values):.2e}")

# 4. Cluster Separation Analysis
print(f"\n📊 Cluster Separation Analysis:")
for cid in unique_clusters:
    cluster_mask = cluster_labels == cid
    cluster_data = feat_s[cluster_mask]
    cluster_center = kmeans.cluster_centers_[cid]
    
    # Calculate average distance to cluster center
    distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    print(f"  Cluster {cid}:")
    print(f"    Size: {np.sum(cluster_mask)} files")
    print(f"    Avg distance to center: {avg_distance:.4f} ± {std_distance:.4f}")

# 5. Inter-cluster Distance Analysis
print(f"\n📏 Inter-cluster Distances:")
for i in range(len(unique_clusters)):
    for j in range(i+1, len(unique_clusters)):
        cid1, cid2 = unique_clusters[i], unique_clusters[j]
        center1 = kmeans.cluster_centers_[cid1]
        center2 = kmeans.cluster_centers_[cid2]
        distance = np.linalg.norm(center1 - center2)
        print(f"  Cluster {cid1} ↔ Cluster {cid2}: {distance:.4f}")

# 6. Create Clustering Visualization
print(f"\n🎨 Creating clustering visualizations...")
viz_dir = MODEL_DIR / 'clustering_analysis'
viz_dir.mkdir(exist_ok=True)

# PCA visualization
pca = PCA(n_components=2)
feat_pca = pca.fit_transform(feat_s)

plt.figure(figsize=(10, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

for i, cid in enumerate(unique_clusters):
    mask = cluster_labels == cid
    plt.scatter(feat_pca[mask, 0], feat_pca[mask, 1], 
               c=[colors[i]], label=f'Cluster {cid}',
               s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

# Plot cluster centers
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
           c='red', s=200, marker='x', linewidths=3, 
           label='Cluster Centers', zorder=5)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Cluster Visualization using PCA')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(viz_dir / 'pca_clusters.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance plot
plt.figure(figsize=(12, 8))
y_pos = np.arange(top_features)
plt.barh(y_pos, feature_importance[top_indices], color='skyblue')
plt.yticks(y_pos, [feature_names[i] for i in top_indices])
plt.xlabel('Feature Importance (Between/Within Cluster Variance)')
plt.title('Top Features Contributing to Cluster Separation')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(viz_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Cluster distribution plot
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(unique_clusters)), cluster_counts, color=colors[:len(unique_clusters)])
plt.xlabel('Cluster ID')
plt.ylabel('Number of Files')
plt.title('Distribution of Files Across Clusters')
plt.xticks(range(len(unique_clusters)))

# Add count labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(viz_dir / 'cluster_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 7. Save Detailed Analysis Report
analysis_report = {
    'clustering_metrics': {
        'silhouette_score': float(silhouette),
        'calinski_harabasz_score': float(calinski),
        'davies_bouldin_score': float(davies)
    },
    'cluster_distribution': {
        str(cid): int(cluster_counts[cid]) for cid in unique_clusters
    },
    'feature_importance': {
        feature_names[i]: float(feature_importance[i]) for i in range(len(feature_names))
    },
    'top_features': [feature_names[i] for i in top_indices],
    'cluster_centers': kmeans.cluster_centers_.tolist(),
    'n_clusters': len(unique_clusters),
    'total_files': len(train_files)
}

import json
with open(viz_dir / 'clustering_analysis_report.json', 'w') as f:
    json.dump(analysis_report, f, indent=2)

print(f"\n✅ Clustering analysis complete!")
print(f"📁 Analysis saved to: {viz_dir}")
print(f"📊 Quality Summary:")
print(f"   Silhouette: {silhouette:.4f} ({'Good' if silhouette > 0.3 else 'Fair' if silhouette > 0.1 else 'Poor'})")
print(f"   Balance: {min(cluster_counts)/max(cluster_counts):.3f} ({'Good' if min(cluster_counts)/max(cluster_counts) > 0.5 else 'Fair' if min(cluster_counts)/max(cluster_counts) > 0.3 else 'Poor'})")

print("\n" + "="*60)

# ──────────────────────────────────────────────────────────────────────────
#%% 3. NARX Data Preprocessing and Preparation (Save/Load approach)
# --------------------------------------------------------------------------
print("\n🔧 Phase 1: NARX Data Preprocessing and Preparation …")

# Check if preprocessed data already exists
preprocessed_data_file = MODEL_DIR / 'narx' / 'preprocessed_data.pkl'

if preprocessed_data_file.exists():
    print("📂 Loading existing preprocessed NARX data...")
    with open(preprocessed_data_file, 'rb') as f:
        cluster_data = pickle.load(f)
    print(f"✅ Loaded preprocessed data for {len(cluster_data)} clusters.")
else:
    print("🔄 Creating new preprocessed NARX data...")
    # Store preprocessed data for each cluster
    cluster_data = {}

    for cid in range(N_CLUSTERS):
        print(f"\n📊 Preprocessing data for cluster {cid}...")
        Xc, Yc = [], []
        for idx, p in enumerate(train_files):
            if kmeans.labels_[idx] != cid:
                continue
            x, y = make_xy(preprocess(p))
            if len(x):
                Xc.append(x); Yc.append(y)
        
        if not Xc:
            print(f"   ⚠️  No data found for cluster {cid}, skipping...")
            continue

        Xc = np.vstack(Xc);  Yc = np.vstack(Yc)
        scX = StandardScaler().fit(Xc)
        scY = StandardScaler().fit(Yc)

        # train/val split 80/20
        split = int(0.8 * len(Xc))
        Xtr, Ytr = Xc[:split], Yc[:split]
        Xvl, Yvl = Xc[split:], Yc[split:]

        # Store preprocessed data for training phase
        cluster_data[cid] = {
            'Xtr': Xtr, 'Ytr': Ytr,
            'Xvl': Xvl, 'Yvl': Yvl,
            'scX': scX, 'scY': scY,
            'input_dim': Xc.shape[1],
            'output_dim': Yc.shape[1]
        }
        
        print(f"   ✅ Cluster {cid} preprocessing complete:")
        print(f"      Training samples: {len(Xtr)}")
        print(f"      Validation samples: {len(Xvl)}")
        print(f"      Input dimension: {Xc.shape[1]}")
        print(f"      Output dimension: {Yc.shape[1]}")

    # Save preprocessed data for future use
    preprocessed_data_file.parent.mkdir(parents=True, exist_ok=True)
    with open(preprocessed_data_file, 'wb') as f:
        pickle.dump(cluster_data, f)
    print(f"\n✅ Phase 1 complete! Preprocessed data saved for {len(cluster_data)} clusters.")
    print(f"   Data saved to: {preprocessed_data_file}")

# Save scalers for each cluster (needed for inference)
for cid, data in cluster_data.items():
    pickle.dump(data['scX'], (MODEL_DIR/f'narx/scaler_X_{cid}.pkl').open('wb'))
    pickle.dump(data['scY'], (MODEL_DIR/f'narx/scaler_Y_{cid}.pkl').open('wb'))

#%%
print("\n🚀 Phase 2: NARX Model Training …")
for cid, data in cluster_data.items():
    print(f"\n🚀 Training NARX model for cluster {cid}")
    print(f"   Training samples: {len(data['Xtr'])}")
    print(f"   Validation samples: {len(data['Xvl'])}")
    print(f"   Early stopping patience: {ES_PATIENCE_NARX} epochs")
    print(f"   Min improvement threshold: {ES_MIN_DELTA}")

    model = build_narx(data['input_dim'], data['output_dim'])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=1e-4, 
            weight_decay=1e-4,
            clipnorm=1.0  # Gradient clipping
        ), 
        loss=weighted_mse
    )

    # Optimized early stopping callbacks
    es = callbacks.EarlyStopping(
        patience=ES_PATIENCE_NARX, 
        restore_best_weights=ES_RESTORE_BEST,
        min_delta=ES_MIN_DELTA,
        verbose=1
    )
    ck = callbacks.ModelCheckpoint(
        filepath=MODEL_DIR/f'narx/cluster_{cid}.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0  # Reduced verbosity
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3,  # Reduced from 4
        min_lr=1e-6, 
        verbose=0,   # Reduced verbosity
        min_delta=ES_MIN_DELTA
    )
    warmup_scheduler = WarmUpLearningRateScheduler(warmup_epochs=5)
    
    # Add convergence callback
    convergence_callback = ConvergenceCallback(patience=ES_PATIENCE_NARX)
    
    history_narx = model.fit(
        data['scX'].transform(data['Xtr']), data['scY'].transform(data['Ytr']),
        validation_data=(data['scX'].transform(data['Xvl']), data['scY'].transform(data['Yvl'])),
        epochs=EPOCHS_NARX,
        batch_size=BATCH_SIZE_NARX,
        verbose=1,
        callbacks=[es, ck, lr_scheduler, warmup_scheduler, convergence_callback]
    )

    # Print training summary
    best_epoch = np.argmin(history_narx.history['val_loss']) + 1
    best_val_loss = min(history_narx.history['val_loss'])
    final_epochs = len(history_narx.history['val_loss'])
    
    print(f"\n✅ Cluster {cid} training complete!")
    print(f"   Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"   Total epochs trained: {final_epochs}")
    print(f"   Time saved: {EPOCHS_NARX - final_epochs} epochs")
    
    # Export detailed NARX training report
    narx_report = {
        'cluster_id': cid,
        'training_summary': {
            'total_training_samples': len(data['Xtr']),
            'total_validation_samples': len(data['Xvl']),
            'input_dimension': data['input_dim'],
            'output_dimension': data['output_dim'],
            'batch_size': BATCH_SIZE_NARX,
            'max_epochs': EPOCHS_NARX,
            'actual_epochs_trained': final_epochs,
            'epochs_saved': EPOCHS_NARX - final_epochs,
            'early_stopping_patience': ES_PATIENCE_NARX,
            'min_improvement_threshold': ES_MIN_DELTA
        },
        'performance_metrics': {
            'best_validation_loss': float(best_val_loss),
            'best_epoch': best_epoch,
            'final_training_loss': float(history_narx.history['loss'][-1]),
            'final_validation_loss': float(history_narx.history['val_loss'][-1]),
            'loss_improvement': float(history_narx.history['val_loss'][0] - best_val_loss),
            'convergence_rate': float((history_narx.history['val_loss'][0] - best_val_loss) / best_val_loss * 100)
        },
        'training_history': {
            'train_loss': [float(x) for x in history_narx.history['loss']],
            'val_loss': [float(x) for x in history_narx.history['val_loss']],
            'epochs': list(range(1, final_epochs + 1))
        },
        'model_architecture': {
            'input_shape': data['input_dim'],
            'output_shape': data['output_dim'],
            'layers': [
                {'type': 'Dense', 'units': 1024, 'activation': 'selu', 'dropout': 0.4},
                {'type': 'Dense', 'units': 512, 'activation': 'selu', 'dropout': 0.4},
                {'type': 'Dense', 'units': 256, 'activation': 'selu', 'dropout': 0.3},
                {'type': 'Dense', 'units': 128, 'activation': 'selu', 'dropout': 0.2},
                {'type': 'Dense', 'units': data['output_dim'], 'activation': 'linear'}
            ],
            'regularization': 'L2 (5e-4)',
            'batch_normalization': True,
            'residual_connections': True
        },
        'optimizer_config': {
            'optimizer': 'AdamW',
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'clipnorm': 1.0
        },
        'callbacks_used': [
            'EarlyStopping',
            'ModelCheckpoint', 
            'ReduceLROnPlateau',
            'WarmUpLearningRateScheduler',
            'ConvergenceCallback'
        ],
        'data_preprocessing': {
            'scaler_type': 'StandardScaler',
            'train_samples': len(data['Xtr']),
            'val_samples': len(data['Xvl']),
            'input_features': len(STATE_COLS + EXOG_COLS) * (LAG + 1),
            'output_features': len(STATE_COLS)
        },
        'convergence_analysis': {
            'epochs_to_best': best_epoch,
            'epochs_after_best': final_epochs - best_epoch,
            'improvement_rate': float((history_narx.history['val_loss'][0] - best_val_loss) / best_epoch),
            'plateau_detected': final_epochs - best_epoch > ES_PATIENCE_NARX,
            'early_stopping_triggered': final_epochs < EPOCHS_NARX
        }
    }
    
    # Save detailed report
    report_dir = MODEL_DIR / 'narx' / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / f'cluster_{cid}_narx_report.json'
    with open(report_file, 'w') as f:
        json.dump(narx_report, f, indent=2, default=str)
    
    # Create training curves plot
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 2, 1)
    plt.plot(history_narx.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history_narx.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.axvline(x=best_epoch-1, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted MSE Loss')
    plt.title(f'NARX Training Curves - Cluster {cid}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss improvement over time
    plt.subplot(2, 2, 2)
    initial_loss = history_narx.history['val_loss'][0]
    loss_improvement = [initial_loss - loss for loss in history_narx.history['val_loss']]
    plt.plot(loss_improvement, color='purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Improvement')
    plt.title('Validation Loss Improvement')
    plt.grid(True, alpha=0.3)
    
    # Learning rate schedule (if available)
    plt.subplot(2, 2, 3)
    if 'lr' in history_narx.history:
        plt.plot(history_narx.history['lr'], color='orange', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nNot Tracked', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    
    # Convergence analysis
    plt.subplot(2, 2, 4)
    epochs = range(1, final_epochs + 1)
    plt.bar(epochs, history_narx.history['val_loss'], alpha=0.7, color='lightcoral')
    plt.axhline(y=best_val_loss, color='green', linestyle='--', linewidth=2, label=f'Best Loss: {best_val_loss:.6f}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(report_dir / f'cluster_{cid}_narx_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\n📊 Detailed Report Generated:")
    print(f"   Report file: {report_file}")
    print(f"   Analysis plot: {report_dir}/cluster_{cid}_narx_analysis.png")
    print(f"   Loss improvement: {narx_report['performance_metrics']['loss_improvement']:.6f}")
    print(f"   Convergence rate: {narx_report['performance_metrics']['convergence_rate']:.2f}%")
    print(f"   Early stopping triggered: {narx_report['convergence_analysis']['early_stopping_triggered']}")
    if narx_report['convergence_analysis']['plateau_detected']:
        print(f"   ⚠️  Plateau detected - model stopped improving")
    else:
        print(f"   ✅ Model converged naturally")


    plt.figure()
    plt.plot(history_narx.history['loss'], label='Train Loss')
    plt.plot(history_narx.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted MSE Loss')
    plt.title(f'NARX Loss Curve (Cluster {cid})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / f'narx/cluster_{cid}_loss_curve.png', dpi=150)
    plt.close()
    print(f"Saved NARX loss curve for cluster {cid} to narx/cluster_{cid}_loss_curve.png")

print("✅ NARX training complete.")

# Collect loss histories for all clusters
narx_histories = {}
for cid in cluster_data.keys():
    report_dir = MODEL_DIR / 'narx' / 'reports'
    report_file = report_dir / f'cluster_{cid}_narx_report.json'
    if report_file.exists():
        with open(report_file, 'r') as f:
            narx_report = json.load(f)
        narx_histories[cid] = {
            'train_loss': narx_report['training_history']['train_loss'],
            'val_loss': narx_report['training_history']['val_loss'],
            'epochs': narx_report['training_history']['epochs']
        }

# Plot side-by-side loss curves
if len(narx_histories) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for idx, cid in enumerate(sorted(narx_histories.keys())):
        ax = axes[idx]
        hist = narx_histories[cid]
        ax.plot(hist['epochs'], hist['train_loss'], label='Train Loss', color='blue')
        ax.plot(hist['epochs'], hist['val_loss'], label='Validation Loss', color='red')
        ax.set_title(f'NARX Loss Curve (Cluster {cid})')
        ax.set_xlabel('Epoch')
        if idx == 0:
            ax.set_ylabel('Weighted MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'narx/narx_loss_curves_side_by_side.png', dpi=150)
    plt.close()
    print("Saved side-by-side NARX loss curves to narx/narx_loss_curves_side_by_side.png")
else:
    print("Side-by-side plot only supported for exactly 2 clusters.")



#-- store global metadata (helps inference script remain agnostic) --------
json.dump({'state_cols': STATE_COLS, 'exog_cols': EXOG_COLS, 'lag': LAG},
          (MODEL_DIR/'metadata.json').open('w'))




#%%
# ──────────────────────────────────────────────────────────────────────────
## 4. Collect NARX validation residuals across *all* clusters  ➜  training
#    set for Quantile-Regression nets.
# --------------------------------------------------------------------------
print("\nCollecting NARX validation residuals for QR training ...")

# Check if QR training data already exists
qr_data_file = MODEL_DIR / 'qr' / 'qr_training_data.pkl'

if qr_data_file.exists():
    print("📂 Loading existing QR training data...")
    with open(qr_data_file, 'rb') as f:
        qr_data = pickle.load(f)
    val_X_qr = qr_data['val_X_qr']
    val_E_qr = qr_data['val_E_qr']
    print(f"✅ Loaded QR training data: {val_X_qr.shape[0]} samples")
else:
    print("🔄 Creating new QR training data...")
    val_X, val_E = [], []

    # Map each file to its cluster using the same logic as in clustering
    file_to_cluster = {}
    for idx, p in enumerate(train_files):
        # Use the same feature extraction as in clustering
        arr = preprocess(p)[CLUST_COLS].values
        feat_vec = np.concatenate([arr.mean(0), arr.std(0), arr.min(0), arr.max(0)])
        sig = sc_feat.transform([feat_vec])
        cluster_id = int(kmeans.predict(sig)[0])
        file_to_cluster[p] = cluster_id

    # For each cluster, collect residuals from files assigned to that cluster
    for cid in range(N_CLUSTERS):
        scaler_x = pickle.loads((MODEL_DIR/f'narx/scaler_X_{cid}.pkl').read_bytes())
        scaler_y = pickle.loads((MODEL_DIR/f'narx/scaler_Y_{cid}.pkl').read_bytes())
        narx     = tf.keras.models.load_model(MODEL_DIR/f'narx/cluster_{cid}.keras',
                                              compile=False)
        files_in_cluster = [p for p, c in file_to_cluster.items() if c == cid]
        print(f"  Cluster {cid}: {len(files_in_cluster)} files")
        for p in files_in_cluster:
            X, Y = make_xy(preprocess(p))
            if not len(X):
                continue
            y_hat = scaler_y.inverse_transform(
                        narx.predict(scaler_x.transform(X), verbose=0))
            val_X.append(scaler_x.transform(X))
            val_E.append(Y - y_hat)

    if len(val_X) == 0 or len(val_E) == 0:
        raise RuntimeError("No validation data collected for QR training. Check preprocessing and data files.")

    val_X = np.vstack(val_X)
    val_E = np.vstack(val_E)
    print(f"Collected {val_X.shape[0]} validation samples for QR training.")

    # Limit samples for QR training
    MAX_QR_SAMPLES = 70000
    val_X_qr = val_X[:MAX_QR_SAMPLES]
    val_E_qr = val_E[:MAX_QR_SAMPLES]

    # Save QR training data for future use
    qr_data_file.parent.mkdir(parents=True, exist_ok=True)
    qr_data = {
        'val_X_qr': val_X_qr,
        'val_E_qr': val_E_qr,
        'total_samples': len(val_X),
        'used_samples': len(val_X_qr)
    }
    with open(qr_data_file, 'wb') as f:
        pickle.dump(qr_data, f)
    print(f"✅ QR training data saved to: {qr_data_file}")


# ──────────────────────────────────────────────────────────────────────────
#%% 5. Enhanced Quantile-Regression Training (τ = 0.1 & 0.9) on residuals.
# --------------------------------------------------------------------------

print("\n Training enhanced QR nets …")

loss_dir = MODEL_DIR / 'qr/loss_curves'
loss_dir.mkdir(parents=True, exist_ok=True)

# Model selection based on variable characteristics
model_configs = {
    'T_PM': {'model_type': 'advanced', 'loss_type': 'adaptive'},
    'c': {'model_type': 'ensemble', 'loss_type': 'quantile_huber'},
    'd10': {'model_type': 'deep', 'loss_type': 'enhanced_pinball'},
    'd50': {'model_type': 'deep', 'loss_type': 'adaptive'},
    'd90': {'model_type': 'deep', 'loss_type': 'quantile_huber'},
    'T_TM': {'model_type': 'advanced', 'loss_type': 'enhanced_pinball'}
}

# Loss function mapping
loss_functions = {
    'enhanced_pinball': pinball_loss,
    'adaptive': adaptive_pinball_loss,
    'quantile_huber': lambda tau: quantile_huber_loss(tau, delta=1.0)
}

# Model building functions
model_builders = {
    'advanced': build_advanced_qr,
    'ensemble': build_ensemble_qr,
    'deep': build_deep_qr
}

qr_loss_histories = {}

for j, col in enumerate(STATE_COLS):
    y_err = val_E_qr[:, j:j+1]  # residual for that state
    config = model_configs.get(col, {'model_type': 'deep', 'loss_type': 'enhanced_pinball'})
    
    for q in QUANTILES:
        print(f"  Training {col} τ={q:.1f} with {config['model_type']} model...")
        
        # Build model based on configuration
        model_builder = model_builders[config['model_type']]
        qr = model_builder(val_X_qr.shape[1])
        
        # Select loss function
        loss_func = loss_functions[config['loss_type']](q)
        
        # Enhanced optimizer with better scheduling
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=1e-4,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999
        )
        
        qr.compile(optimizer=optimizer, loss=loss_func)
        
        # Advanced callbacks
        es_qr = callbacks.EarlyStopping(
            patience=20,
            restore_best_weights=True,
            monitor="val_loss",
            min_delta=1e-6
        )
        
        lr_scheduler_qr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=8, 
            min_lr=1e-7, 
            verbose=1
        )
        
        warmup_scheduler_qr = WarmUpLearningRateScheduler(warmup_epochs=5)
        
        # Model checkpoint for best model
        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / f'qr/{col}_{q:.1f}_best.keras'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
        
        # Learning rate finder callback
        class LRFinder(callbacks.Callback):
            def __init__(self, start_lr=1e-7, end_lr=1e-2, num_iterations=100):
                super().__init__()
                self.start_lr = start_lr
                self.end_lr = end_lr
                self.num_iterations = num_iterations
                self.lr_multiplier = (end_lr / start_lr) ** (1 / num_iterations)
                self.best_loss = float('inf')
                self.lr_history = []
                self.loss_history = []
            
            def on_batch_begin(self, batch, logs=None):
                if len(self.lr_history) < self.num_iterations:
                    lr = self.start_lr * (self.lr_multiplier ** len(self.lr_history))
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
                    self.lr_history.append(lr)
            
            def on_batch_end(self, batch, logs=None):
                if len(self.loss_history) < self.num_iterations:
                    self.loss_history.append(logs.get('loss', 0))
                    if logs.get('loss', 0) < self.best_loss:
                        self.best_loss = logs.get('loss', 0)
        
        # Train with learning rate finder first (optional)
        if col in ['d50', 'd90']:  # Use LR finder for harder variables
            lr_finder = LRFinder()
            qr.fit(
                val_X_qr, y_err,
                validation_split=0.2,
                epochs=1,
                batch_size=BATCH_SIZE_QR,
                verbose=0,
                callbacks=[lr_finder]
            )
            # Find optimal learning rate
            if len(lr_finder.loss_history) > 10:
                min_loss_idx = np.argmin(lr_finder.loss_history[10:]) + 10
                optimal_lr = lr_finder.lr_history[min_loss_idx]
                print(f"    Optimal LR for {col}: {optimal_lr:.2e}")
                # Update optimizer with optimal LR
                tf.keras.backend.set_value(optimizer.learning_rate, optimal_lr)
        
        # Main training
        history_qr = qr.fit(
            val_X_qr, y_err,
            validation_split=0.2,
            epochs=EPOCHS_QR,
            batch_size=BATCH_SIZE_QR,
            verbose=0,
            callbacks=[es_qr, lr_scheduler_qr, warmup_scheduler_qr, checkpoint, PrintMetricsCallback()]
        )
        
        # Save final model
        qr.save(MODEL_DIR / f'qr/{col}_{q:.1f}.keras')
        
        # Store history for plotting
        qr_loss_histories[(col, q)] = {
            'loss': history_qr.history['loss'],
            'val_loss': history_qr.history['val_loss'],
            'model_type': config['model_type'],
            'loss_type': config['loss_type']
        }
        
        # Plot individual loss curves
        plt.figure(figsize=(8, 4))
        plt.plot(history_qr.history['loss'], label='train', linewidth=2)
        plt.plot(history_qr.history['val_loss'], label='val', linewidth=2)
        plt.title(f"{col} τ={q:.1f} ({config['model_type']} model)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(loss_dir / f"{col}_{q:.1f}_loss.png", dpi=150)
        plt.close()

print(" Enhanced QR nets training completed.")

# --- Plot all QR loss curves in a single image ---
try:
    n_rows = len(STATE_COLS)
    n_cols = len(QUANTILES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    for i, col in enumerate(STATE_COLS):
        for j, q in enumerate(QUANTILES):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            hist = qr_loss_histories.get((col, q))
            if hist:
                ax.plot(hist['loss'], label='Train Loss', color='blue')
                ax.plot(hist['val_loss'], label='Validation Loss', color='red')
                ax.set_title(f"{col}  τ={q:.1f}")
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Pinball Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'qr/qr_loss_curves_all.png', dpi=150)
    plt.close()
    print("Saved all QR loss curves to qr/qr_loss_curves_all.png")
except Exception as e:
    print(f"Could not create combined QR loss plot: {e}")


# ──────────────────────────────────────────────────────────────────────────
#%% 6. Enhanced Conformal Calibration  (90% Prediction-Interval)  in **metres**.
# --------------------------------------------------------------------------
print("\n Computing enhanced conformal deltas …")

def compute_adaptive_conformal_deltas(val_X, val_E, alpha=0.1):
    """Compute adaptive conformal deltas with local calibration."""
    deltas = {}
    
    for j, col in enumerate(STATE_COLS):
        errors = np.abs(val_E[:, j])
        
        # Basic conformal delta
        k = int(np.ceil((1-alpha) * (len(errors)+1))) - 1
        basic_delta = float(np.sort(errors)[k])
        
        # Adaptive delta based on error distribution
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        
        # Adjust delta based on error statistics
        adaptive_factor = 1.0 + 0.5 * (error_std / error_mean) if error_mean > 0 else 1.0
        adaptive_delta = basic_delta * adaptive_factor
        
        # Local calibration using quantile regression
        local_errors = errors[errors <= np.percentile(errors, 95)]
        if len(local_errors) > 10:
            local_k = int(np.ceil((1-alpha) * (len(local_errors)+1))) - 1
            local_delta = float(np.sort(local_errors)[local_k])
            # Combine global and local deltas
            final_delta = 0.7 * adaptive_delta + 0.3 * local_delta
        else:
            final_delta = adaptive_delta
        
        deltas[col] = final_delta
    
    return deltas

def compute_multiple_quantile_deltas(val_X, val_E, quantiles=[0.05, 0.1, 0.15]):
    """Compute deltas for multiple quantiles for better coverage."""
    deltas = {}
    
    for j, col in enumerate(STATE_COLS):
        errors = np.abs(val_E[:, j])
        col_deltas = {}
        
        for alpha in quantiles:
            k = int(np.ceil((1-alpha) * (len(errors)+1))) - 1
            delta = float(np.sort(errors)[k])
            col_deltas[f'q{int((1-alpha)*100)}'] = delta
        
        deltas[col] = col_deltas
    
    return deltas

# Compute both basic and enhanced deltas
alpha = 0.1  # 90% coverage target
basic_deltas = {}
enhanced_deltas = compute_adaptive_conformal_deltas(val_X_qr, val_E_qr, alpha)
multiple_deltas = compute_multiple_quantile_deltas(val_X_qr, val_E_qr)

# Store all delta types
delta_results = {
    'basic': basic_deltas,
    'enhanced': enhanced_deltas,
    'multiple': multiple_deltas
}

pickle.dump(delta_results, (MODEL_DIR/'enhanced_conformal_deltas.pkl').open('wb'))
pickle.dump(enhanced_deltas, (MODEL_DIR/'conformal_deltas.pkl').open('wb'))  # Backward compatibility

print(" Saved enhanced deltas:", enhanced_deltas)
print(" Multiple quantile deltas available for advanced calibration")

print("\n Finished training. Model at:", MODEL_DIR)

# %%
