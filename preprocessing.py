#!/usr/bin/env python3
"""
Enhanced Prediction Script for SFC Data with Open-Loop and Closed-Loop
- Integrates all critical preprocessing and helper functions from training
- Handles one or many files (Beat-The-Felix or any untouched .txt)
- Produces CQR-calibrated uncertainty intervals and summary metrics
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import json, pickle
from pathlib import Path
#from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans
import matplotlib.ticker as mticker
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==== PATHS AND GLOBALS (update as needed) ====
TEST_DIR   = Path(r"Beat-the-Felix")       # Directory with test file(s)
MODEL_ROOT = Path(r"model_5files17")       # Directory with saved models

# ==== LOAD METADATA FROM TRAINING ====
meta       = json.loads((MODEL_ROOT/'metadata.json').read_text())
STATE_COLS = meta['state_cols']
EXOG_COLS  = meta['exog_cols']
LAG        = meta['lag']
CLUST_COLS = STATE_COLS + EXOG_COLS
PREDICT    = 3   # For open-loop
PSD_COLS   = ('d10', 'd50', 'd90')


# ==== DATA CLEANING/PROCESSING (from training) ====

def read_txt(p: Path) -> pd.DataFrame:
    """Read TAB-separated SFC data."""
    return pd.read_csv(p, sep='\t', engine='python').apply(pd.to_numeric, errors='coerce')

def clean_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers via IQR for all relevant columns (matches training)."""
    available_cols = [col for col in CLUST_COLS if col in df.columns]
    df = df.dropna(subset=available_cols)
    for column in available_cols:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df.reset_index(drop=True)

def smooth_log_psd(df, columns=['d10', 'd50', 'd90'], window=5):
    """
    Smooth selected PSD columns (d10/d50/d90) in log-space and replace in the DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame with columns to smooth.
        columns (list): List of column names to smooth.
        window (int): Rolling window size for smoothing.
    Returns:
        pd.DataFrame: DataFrame with smoothed columns (overwrites original columns).
    """
    df = df.copy()
    for col in columns:
        # Avoid log(0) by clipping to a small value
        clipped = np.clip(df[col].values, 1e-9, None)
        log_vals = np.log(clipped)
        log_smoothed = pd.Series(log_vals).rolling(window, center=True, min_periods=1).mean()
        smoothed = np.exp(log_smoothed)
        df[col] = smoothed
    return df

def smooth_log_series(df, columns, window=5, suffix='_smoothed'):
    """
    For each column in 'columns', smooths the time series in log-space and
    stores the exponentiated result in a new column with the specified suffix.
    
    Args:
        df: pandas DataFrame with columns to be smoothed
        columns: list of column names (e.g., ['d10_closed', 'd50_closed', 'd90_closed'])
        window: moving average window size (default: 5)
        suffix: suffix for the new smoothed columns
    
    Returns:
        df: original DataFrame with new smoothed columns added
    """
    for col in columns:
        # 1. Clip to positive values to avoid log(0) or log(negative)
        clipped = np.clip(df[col].values, 1e-9, None)
        log_vals = np.log(clipped)
        log_smoothed = pd.Series(log_vals).rolling(window, center=True, min_periods=1).mean()
        smoothed = np.exp(log_smoothed)
        df[col + suffix] = smoothed
    return df

# ==== CLUSTER DETECTION (from training) ====

sc_feat = pickle.loads((MODEL_ROOT/'feature_scaler.pkl').read_bytes())
kmeans  = pickle.loads((MODEL_ROOT/'kmeans_model.pkl').read_bytes())

def file_signature(df):
    """Same feature vector used for clustering in training."""
    arr = df[CLUST_COLS].values
    return np.concatenate([arr.mean(0), arr.std(0), arr.min(0), arr.max(0)]).reshape(1,-1)

def detect_cluster(df) -> int:
    return int(kmeans.predict(sc_feat.transform(file_signature(df)))[0])

# ==== LAG MATRIX FOR NARX ====

def build_lagged(df, lag=LAG):
    """Build lagged feature matrix (newest-to-oldest) for open-loop."""
    rows = []
    for i in range(lag, len(df)-1):
        row = []
        for l in range(0, lag+1):  # 0..LAG
            idx = i - l
            row.extend(df[CLUST_COLS].iloc[idx].values)
        rows.append(row)
    return np.asarray(rows, np.float32)

def load_cluster(cid):
    """Load cluster-specific scalers and NARX model."""
    scX = pickle.loads((MODEL_ROOT/f'narx/scaler_X_{cid}.pkl').read_bytes())
    scY = pickle.loads((MODEL_ROOT/f'narx/scaler_Y_{cid}.pkl').read_bytes())
    narx = tf.keras.models.load_model(MODEL_ROOT/f'narx/cluster_{cid}.keras', compile=False)
    return scX, scY, narx

# ==== OPEN-LOPP / CLOSED-LOOP implementation ====

# ==============================================================================
# (ADDITIONAL TASK) 1: open-loop prediction: ANN(ŷₖ, uₖ) = ŷₖ₊₁ (i.e., always feed back predicted y in open loop)
# ==============================================================================

#need changes in narx model training.............................

#def predict_recursive_open(df, scX, scY, narx, lag=LAG):
    """
    Recursive open-loop: always feed previous predictions (in RAW space) back into the lag buffer.
    Only scale before model input.
    """
    N = len(df)
    n_state = len(STATE_COLS)
    n_exog = len(EXOG_COLS)
    preds = []
    Xs_all = []

    # --- Initial lag buffer: RAW values, (lag+1, n_state+n_exog) ---
    lag_blocks = []
    for l in range(lag, -1, -1):
        block = np.concatenate([df[STATE_COLS].iloc[l].values, df[EXOG_COLS].iloc[l].values])
        lag_blocks.append(block)
    lag_matrix = np.array(lag_blocks)  # shape: (lag+1, n_state+n_exog)

    for i in range(lag, N-1):
        # 1. Flatten and scale lag buffer for model input
        x_flat = lag_matrix.flatten()
        x_scaled = scX.transform([x_flat])[0]

        # 2. Predict next state
        y_scaled = narx.predict(x_scaled[None], verbose=0)[0]     # scaled
        y_pred = scY.inverse_transform(y_scaled[None])[0]         # unscaled (raw)

        preds.append(y_pred)
        Xs_all.append(x_scaled)

        # 3. Update lag buffer: shift all lags down
        lag_matrix[:-1] = lag_matrix[1:]
        # 4. Insert predicted state (RAW/unscaled) at newest
        lag_matrix[-1, :n_state] = y_pred
        # 5. Insert exogenous (RAW/unscaled) at newest
        lag_matrix[-1, n_state:] = df[EXOG_COLS].iloc[i+1].values

    preds = np.vstack(preds)
    Xs_all = np.vstack(Xs_all)
    df_out = df.iloc[lag+1:].reset_index(drop=True)
    return df_out, Xs_all, preds

def rollout(model, lag_scaled, horizon, scX, scY, exog_future_raw):
    """
    open-loop rollout as described in training.
    At each step, feed predicted state and true exogenous input.
    """
    x      = lag_scaled.copy()
    preds  = []
    n_y    = len(STATE_COLS)
    n_u    = len(EXOG_COLS)
    stride = n_y + n_u
    y_slice = slice(0, n_y)
    u_slice = slice(n_y, n_y+n_u)
    for k in range(horizon):
        y_scaled  = model.predict(x[None], verbose=0)[0]
        y_raw     = scY.inverse_transform(y_scaled[None])[0]
        preds.append(y_raw)
        x[stride:] = x[:-stride]
        x[y_slice] = y_scaled
        mu_u   = scX.mean_[u_slice]
        sig_u  = scX.scale_[u_slice]
        x[u_slice] = (exog_future_raw[k] - mu_u) / sig_u
    return np.asarray(preds)

def predict_open(df, scX, scY, narx, horizon=PREDICT):
    """
    open-loop prediction: rolling window.
    Each step: predict next state using NARX output as input.
    """
    total   = len(df) - 1 - LAG
    usable  = total - horizon + 1
    Xs_all, Yh_all = [], []
    for t0 in range(usable):
            row  = []
            for l in range(0, LAG+1):
                idx = t0 + LAG - l
                row.extend(df[CLUST_COLS].iloc[idx].values)
            lag_raw  = np.asarray(row, np.float32)
            lag_s    = scX.transform(lag_raw[None])[0]
            exog_f   = df[EXOG_COLS].iloc[t0+1 : t0+1+horizon].values
            y_seq    = rollout(narx, lag_s, horizon, scX, scY, exog_f)
            Xs_all.append(lag_s)
            Yh_all.append(y_seq[-1])   # last step
    df_out = df.iloc[LAG+horizon : LAG+horizon+usable].reset_index(drop=True)
    return df_out, np.vstack(Xs_all), np.vstack(Yh_all)

def predict_closed(df, scX, scY, narx):
    X      = build_lagged(df)
    Xs     = scX.transform(X)
    y_pred = scY.inverse_transform(narx.predict(Xs, verbose=0))
    return df.iloc[LAG+1:].reset_index(drop=True), Xs, y_pred

# ==== LOAD QR + CQR DELTAS ====
QR = {}
for col in STATE_COLS:
    for q in (0.1, 0.9):
        QR[(col, q)] = tf.keras.models.load_model(MODEL_ROOT/f'qr/{col}_{q:.1f}.keras', compile=False)
DELTAS = pickle.loads((MODEL_ROOT/'conformal_deltas.pkl').read_bytes())

# Optional: adjust deltas if model requires calibration (can tune here)
DELTAS['c']    *= 2.5
DELTAS['d10']  *= 1.5
DELTAS['d50']  *= 1.5
DELTAS['d90']  *= 1.5
DELTAS['T_PM'] *= 1.5
DELTAS['T_TM'] *= 1.5

def add_cqr(df, Xs, base_pred, mode: str):
    """Attach CQR bounds to DataFrame for each state variable."""
    out = df.copy()
    for i, col in enumerate(STATE_COLS):
        lo = QR[(col, 0.1)].predict(Xs, verbose=0).flatten()
        hi = QR[(col, 0.9)].predict(Xs, verbose=0).flatten()
        out[f"{col}_{mode}"]    = base_pred[:, i]
        out[f"{col}_{mode}_lo"] = base_pred[:, i] - lo - DELTAS[col]
        out[f"{col}_{mode}_hi"] = base_pred[:, i] + hi + DELTAS[col]
    return out

def metric_table(df: pd.DataFrame, mode: str):
    """Compute MAE, MSE, R2, coverage for each variable."""
    res = {}
    for col in STATE_COLS:
        y_true = df[col].values
        y_pred = df[f"{col}_{mode}"].values
        lo     = df[f"{col}_{mode}_lo"].values
        hi     = df[f"{col}_{mode}_hi"].values
        msk    = np.isfinite(y_true) & np.isfinite(y_pred)
        res[f"{col}_MAE"] = mean_absolute_error(y_true[msk], y_pred[msk])
        res[f"{col}_MSE"] = mean_squared_error(y_true[msk], y_pred[msk])
        #res[f"{col}_R2"]  = r2_score(y_true[msk], y_pred[msk])
        #inside            = (y_true >= lo) & (y_true <= hi)
        #res[f"{col}_COV"] = 100. * inside[msk].mean()
    return res

def plot_ts(df, out_dir: Path, mode: str):
    """
    Plots each state variable time-series with:
      - larger figure
      - thicker lines
      - tighter y-limits (±5% of the data range)
      - bigger fonts and markers
      - more opaque gridlines
      - visible 90% PI shading

    Args:
        df       : DataFrame with columns:
                     col, col_{mode}, col_{mode}_lo, col_{mode}_hi
        out_dir  : Path to save the PNGs
        mode     : 'open' or 'closed'
    """
    

    start = LAG + PREDICT
    t = np.arange(start, start + len(df))

    for col in STATE_COLS:
        # pick up true, pred, lo, hi
        y_true = df[col].values
        y_pred = df[f"{col}_{mode}_smoothed"] \
                 if f"{col}_{mode}_smoothed" in df.columns \
                 else df[f"{col}_{mode}"].values
        y_lo   = df[f"{col}_{mode}_lo"].values
        y_hi   = df[f"{col}_{mode}_hi"].values

        # compute tight y-limits
        lo_min = np.nanmin(y_lo)
        hi_max = np.nanmax(y_hi)
        data_min = min(lo_min, np.nanmin(y_true))
        data_max = max(hi_max, np.nanmax(y_true))
        margin = 0.05 * (data_max - data_min)
        y_min, y_max = data_min - margin, data_max + margin

        plt.figure(figsize=(12, 5), dpi=150)
        # shaded interval
        plt.fill_between(t, y_lo, y_hi,
                         color='tab:red', alpha=0.4,
                         label='90% PI', zorder=1)

        # predicted line
        plt.plot(t, y_pred,
                 color='tab:orange', linewidth=0.8,
                 label='Predicted', zorder=2)

        # true line
        plt.plot(t, y_true,
                 color='black',linestyle='--', linewidth=0.9,
                 label='True', zorder=3)

        # styling
                # Variable-specific y-axis limits (overrides auto-limits)
        if col in ['T_TM', 'T_PM']:
            plt.ylim(311, 318)
        elif col == 'c':
            plt.ylim(0.179, None)
        elif col in ['d10', 'd90']:
            plt.ylim(0.0002, None)  # set lower bound; upper bound auto
        else:
            plt.ylim(y_min, y_max)  # fallback for any other variable

        plt.xlim(0, len(df)-1)
        plt.xlabel("Time Step", fontsize=14)
        plt.ylabel(f"{col} [units]", fontsize=14)
        plt.title(f"{col} — {mode.capitalize()} Prediction", fontsize=18, pad=12)

        # grid
        plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.8)

        # ticks and legend
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=10, loc='upper left', frameon=True, framealpha=0.9)

        # optional log‐scale for PSD
        if col in ('d10','d50','d90'):
            plt.yscale('log')
            plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.xlim(0, 500)      # show only steps 0 through 500   
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_{mode}.png")
        plt.close()



def plot_scatter(df, out, mode):
    """
    Enhanced scatter plot for each state variable.
    - 1:1 reference line
    - R² score on plot
    - Clear labels, thicker points/line
    """
    for col in STATE_COLS:
        plt.figure(figsize=(5, 5))
        x = df[col]
        y = df[f'{col}_{mode}']

        # 1:1 line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        lim_margin = (max_val - min_val) * 0.04
        lims = [min_val - lim_margin, max_val + lim_margin]
        plt.plot(lims, lims, 'k--', lw=2, label='1:1 Reference')

        # Scatter
        plt.scatter(x, y, s=18, alpha=0.6, color='tab:blue', edgecolor='k')

        # R² annotation
        msk = np.isfinite(x) & np.isfinite(y)
        r2 = r2_score(x[msk], y[msk])
        plt.text(0.03, 0.94, f"$R^2$ = {r2:.3f}", transform=plt.gca().transAxes,
                 fontsize=13, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", alpha=0.7))

        plt.xlabel(f"True {col}", fontsize=13)
        plt.ylabel(f"Predicted {col}", fontsize=13)
        plt.title(f"{col} - {mode.capitalize()} Scatter", fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.xlim(lims)
        plt.ylim(lims)
        plt.legend(fontsize=12, loc='best', framealpha=0.85)
        plt.tight_layout()
        plt.savefig(out / f"{col}_{mode}_scatter.png", dpi=200)
        plt.close()





# ==============================================================================
# (ADDITIONAL TASK):2 Alternative UQ: MC Dropout for Uncertainty Estimation
# ==============================================================================
 #Monte Carlo Dropout (as a demonstration of "uncertainty propagation"):

def predict_mc_dropout(df, scX, scY, narx, lag, n_samples=100):
    """
    Monte Carlo Dropout: sample n times at each step to estimate predictive uncertainty.
    """
    N = len(df)
    n_state = len(STATE_COLS)
    n_exog = len(EXOG_COLS)
    preds_mc = []
    # Initial lag buffer (newest first)
    lag_blocks = []
    for l in range(lag, -1, -1):
        block = np.concatenate([df[STATE_COLS].iloc[l].values, df[EXOG_COLS].iloc[l].values])
        lag_blocks.append(block)
    lag_matrix = np.array(lag_blocks)

    for i in range(lag, N-1):
        x_flat = lag_matrix.flatten()
        x_scaled = scX.transform([x_flat])[0]
        # Collect N samples with dropout ON
        outputs = []
        for _ in range(n_samples):
            y_scaled = narx(x_scaled[None], training=True)[0].numpy()  # Dropout ON
            y_pred = scY.inverse_transform(y_scaled[None])[0]
            outputs.append(y_pred)
        outputs = np.vstack(outputs)
        y_mean = outputs.mean(axis=0)
        y_std = outputs.std(axis=0)
        preds_mc.append((y_mean, y_std))
        # Update lag buffer with mean prediction
        lag_matrix[:-1] = lag_matrix[1:]
        lag_matrix[-1, :n_state] = y_mean
        lag_matrix[-1, n_state:] = df[EXOG_COLS].iloc[i+1].values

    # y_mean: N-predictions, y_std: N-uncertainty
    means, stds = zip(*preds_mc)
    means = np.vstack(means)
    stds = np.vstack(stds)
    df_out = df.iloc[lag+1:].reset_index(drop=True)
    return df_out, means, stds

# Call this and plot mean ± 2*std as your uncertainty band!

'''
Alternative techniques for uncertainty quantification
Besides Conformal Quantile Regression, you can try:

MC Dropout (see above)

#### below methods needs seprate model training

Ensemble Methods: Train multiple models, use mean and std for prediction interval

Bayesian Neural Networks: More complex, but you can use tfp.layers.DenseVariational etc.

'''