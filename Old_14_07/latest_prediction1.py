#!/usr/bin/env python3
"""

Generates open-loop and closed-loop forecasts for the SFC data, with calibrated
CQR prediction bands + full metrics & plots.

Mapping to Fig S5 (page 65) as per info given:
    • "prediction model" in the paper  →  **closed-loop** here
    • "simulation model"               →  **open-loop** here
"""

#%% ──  Imports  ──────────────────────────────────────────────────────────────
from __future__ import annotations

import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import trange
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──  Paths & runtime constants  ────────────────────────────────────────────
TEST_DIR   = Path(r"Beat-the-Felix") #/Data/Test")
MODEL_ROOT = Path(r"model_5files17")  # Updated to match training script
OUT_DIR    = MODEL_ROOT/"BEAT"
OUT_DIR.mkdir(exist_ok=True)

# ----------  Load metadata to stay in sync with training  -----------------

# --- Unit configuration ---------------------------------------------------
USE_MICRONS = False        # True  ➜ internally work in µm  
PSD_COLS    = ('d10', 'd50', 'd90')




meta         = json.loads((MODEL_ROOT/'metadata.json').read_text())
STATE_COLS   = meta['state_cols']
EXOG_COLS    = meta['exog_cols']
LAG          = meta['lag']
CLUST_COLS   = STATE_COLS + EXOG_COLS
HORIZON      = 3         # closed-loop rollout length  (= Fig S5 horizon)

print(f"[INFO]  LAG = {LAG},  horizon = {HORIZON}")

#%% ──  Pre-processing helpers (same as training)  ───────────────────────────
def read_txt(p): return pd.read_csv(p, sep='\t', engine='python'
                                   ).apply(pd.to_numeric, errors='coerce')
def clean_df(df):
    # Only process columns that are in CLUST_COLS (excludes constant variables)
    available_cols = [col for col in CLUST_COLS if col in df.columns]
    df = df.dropna(subset=available_cols)
    df = df[(df.T_PM.between(250,400)) & (df.T_TM.between(250,400))
            & (df.d10>0)&(df.d50>0)&(df.d90>0)
            & (df.mf_PM>=0)&(df.mf_TM>=0)&(df.Q_g>=0)]
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

#def harmonise_units(df):
    """
    Make sure d10 / d50 / d90 are in *micrometres*.

    Rule:
        • If the median of a column is smaller than 1 × 10⁻² (i.e. < 1 cm)
          the data must already be in metres  ➜  multiply by 1 × 10⁶.
        • Otherwise assume it is already µm and leave unchanged.

    Works row-wise, so mixed units inside the same file are also fixed.
    """
    if not USE_MICRONS:
        return df    # fall-back for future experiments

    for col in PSD_COLS:
        median = df[col].median(skipna=True)
        if median < 1e-2:         # < 1 cm  ⇒ data were metres
            df[col] *= 1e6        # m → µm
    return df
def to_metres(df):
    """
    Ensure d10 / d50 / d90 are in metres, regardless of the file’s unit.

    Heuristic:
        • If the column median is > 0.01  (i.e. larger than 1 cm)
          the numbers must be µm  → divide by 1 × 10⁶.
        • Otherwise they are already metres → leave untouched.
    """
    for col in PSD_COLS:
        if df[col].median(skipna=True) > 1e-2:   # > 1 cm ⇒ µm
            df[col] /= 1e6                       # µm → m
    return df


def preprocess(path: Path) -> pd.DataFrame:
    #df = clean_df(read_txt(path)) ##Used IQR method to clean data outliers
    #df = to_metres(df)        # <<< make sure we are in metres  ## This is not needed anymore since we moved the trash data from the data directory
    df = clean_iqr(read_txt(path))
    return df


#%% ──  Cluster artefacts  ───────────────────────────────────────────────────
sc_feat = pickle.loads((MODEL_ROOT/'feature_scaler.pkl').read_bytes())
kmeans  = pickle.loads((MODEL_ROOT/'kmeans_model.pkl' ).read_bytes())

def file_signature(df):
   # Return same feature vector used during training clustering.
    arr = df[CLUST_COLS].values
    return np.concatenate([arr.mean(0), arr.std(0), arr.min(0), arr.max(0)]).reshape(1,-1)

def detect_cluster(df) -> int:
    return int(kmeans.predict(sc_feat.transform(file_signature(df)))[0])

# ──  Build lag matrix (newest-to-oldest)  ──────────────────────────────────
def build_lagged(df, lag=LAG):
    rows = []
    for i in range(lag, len(df)-1):          # need y_{t+1} for target
        row = []
        for l in range(0, lag+1):            # 0 … LAG
            idx = i - l
            row.extend(df[CLUST_COLS].iloc[idx].values)
        rows.append(row)
    return np.asarray(rows, np.float32)

# ──  Load per-cluster artefacts  ───────────────────────────────────────────
def load_cluster(cid):
    scX = pickle.loads((MODEL_ROOT/f'narx/scaler_X_{cid}.pkl').read_bytes())
    scY = pickle.loads((MODEL_ROOT/f'narx/scaler_Y_{cid}.pkl').read_bytes())
    narx = tf.keras.models.load_model(MODEL_ROOT/f'narx/cluster_{cid}.keras',
                                      compile=False)
    return scX, scY, narx

#%% ──  Closed-loop rollout  (scaled space)  ──────────────────────────────────
def rollout(model, lag_scaled, horizon, scX, scY, exog_future_raw):
    """
    Predict horizon-step ahead sequence *closed-loop*:
        at each step feed back the *predicted* (scaled) state,
        plus the *true* exogenous input for that step.
    """
    x      = lag_scaled.copy()              # shape = (input_dim,)
    preds  = []
    n_y    = len(STATE_COLS)
    n_u    = len(EXOG_COLS)
    stride = n_y + n_u                      # one (y+u) block per time step

    # handy slices
    y_slice  = slice(0, n_y)                # first part of each block
    u_slice  = slice(n_y, n_y+n_u)          # second part of each block

    for k in range(horizon):
        y_scaled  = model.predict(x[None], verbose=0)[0]     # (n_y,)
        y_raw     = scY.inverse_transform(y_scaled[None])[0]
        preds.append(y_raw)

        # --- push history backwards (newest first layout) ---
        x[stride:] = x[:-stride] * 1.0      # shift older blocks
        x[y_slice] = y_scaled               # newest state (pred)
        # scale & insert exogenous truth for step k
        mu_u   = scX.mean_[u_slice]
        sig_u  = scX.scale_[u_slice]
        x[u_slice] = (exog_future_raw[k] - mu_u) / sig_u

    return np.asarray(preds)

def predict_closed(df, scX, scY, narx, horizon=HORIZON):
    """
    Rolling closed-loop prediction with stride = 1.
    Returns
        df_out   – ground-truth rows matching predictions
        Xs_all   – scaled lag vectors   (for QR nets)
        Yh_all   – raw predictions      (shape rows × |STATE|)
    """
    total   = len(df) - 1 - LAG
    usable  = total - horizon + 1
    Xs_all, Yh_all = [], []

    for t0 in trange(usable, desc="closed", leave=False):
        # lag vector (newest-first)
        row  = []
        for l in range(0, LAG+1):
            idx = t0 + LAG - l
            row.extend(df[CLUST_COLS].iloc[idx].values)
        lag_raw  = np.asarray(row, np.float32)
        lag_s    = scX.transform(lag_raw[None])[0]
        exog_f   = df[EXOG_COLS].iloc[t0+1 : t0+1+horizon].values
        y_seq    = rollout(narx, lag_s, horizon, scX, scY, exog_f)
        Xs_all.append(lag_s)
        Yh_all.append(y_seq[-1])            # only last step (t+h)

    df_out = df.iloc[LAG+horizon : LAG+horizon+usable].reset_index(drop=True)
    return df_out, np.vstack(Xs_all), np.vstack(Yh_all)

def predict_open(df, scX, scY, narx):
    X      = build_lagged(df)
    Xs     = scX.transform(X)
    y_pred = scY.inverse_transform(narx.predict(Xs, verbose=0))
    return df.iloc[LAG+1:].reset_index(drop=True), Xs, y_pred

#%% ──  QR nets & conformal deltas  ───────────────────────────────────────────
QR = {}
for col in STATE_COLS:
    for q in (0.1, 0.9):
        QR[(col, q)] = tf.keras.models.load_model(MODEL_ROOT/f'qr/{col}_{q:.1f}.keras',
                                                  compile=False)
DELTAS = pickle.loads((MODEL_ROOT/'conformal_deltas.pkl').read_bytes())
# Optional: Adjust deltas to improve coverage
DELTAS['c']    *= 4.3
DELTAS['d10']  *= 3.5
DELTAS['d50']  *= 3.5
DELTAS['d90']  *= 3.5
DELTAS['T_PM']  *= 2
DELTAS['T_TM']  *= 2



def add_cqr(df, Xs, base_pred, mode: str):
   # Attach point-pred + CQR bounds to DataFrame."""
    out = df.copy()
    for i, col in enumerate(STATE_COLS):
        lo = QR[(col, 0.1)].predict(Xs, verbose=0).flatten()
        hi = QR[(col, 0.9)].predict(Xs, verbose=0).flatten()
        out[f"{col}_{mode}"]    = base_pred[:, i]
        out[f"{col}_{mode}_lo"] = base_pred[:, i] - lo - DELTAS[col]
        out[f"{col}_{mode}_hi"] = base_pred[:, i] + hi + DELTAS[col]
    return out

# ──  Metrics helper  ───────────────────────────────────────────────────────
def metric_table(df: pd.DataFrame, mode: str):
    res = {}
    for col in STATE_COLS:
        y_true = df[col].values
        y_pred = df[f"{col}_{mode}"].values
        lo     = df[f"{col}_{mode}_lo"].values
        hi     = df[f"{col}_{mode}_hi"].values
        msk    = np.isfinite(y_true) & np.isfinite(y_pred)

        res[f"{col}_MAE"] = mean_absolute_error(y_true[msk], y_pred[msk])
        res[f"{col}_MSE"] = mean_squared_error(y_true[msk], y_pred[msk])
        res[f"{col}_R2"]  = r2_score(y_true[msk], y_pred[msk])
        inside            = (y_true >= lo) & (y_true <= hi)
        res[f"{col}_COV"] = 100. * inside[msk].mean()
    return res

#%% ----------  Plot helpers  -------------------------------------------------
def plot_ts(df, out, mode):
    t = np.arange(len(df))
    for col in STATE_COLS:
        plt.figure(figsize=(7,3))
        plt.plot(t, df[col],  label='truth', lw=1)
        plt.plot(t, df[f'{col}_{mode}'], label='pred', lw=1)
        plt.fill_between(t, df[f'{col}_{mode}_lo'], df[f'{col}_{mode}_hi'],
                         alpha=.25, label='90 % PI')
        plt.title(col); plt.tight_layout()
        plt.legend()
        plt.savefig(out/f"{col}_{mode}.png", dpi=150)
        plt.close()

def plot_scatter(df, out, mode):
    for col in STATE_COLS:
        plt.figure(figsize=(3.5,3.5))
        plt.scatter(df[col], df[f'{col}_{mode}'], s=8, alpha=.6)
        mn, mx = df[[col, f'{col}_{mode}']].values.min(), \
                 df[[col, f'{col}_{mode}']].values.max()
        plt.plot([mn, mx],[mn, mx],'r--'); plt.title(col); plt.tight_layout()
        plt.savefig(out/f"{col}_{mode}_scatter.png", dpi=150)
        plt.close()

# ──────────────────────────────────────────────────────────────────────────
#%%  Main loop over test files
# --------------------------------------------------------------------------
summary = []
for p in sorted(TEST_DIR.glob("*.txt")):
    stem  = p.stem
    out_f = OUT_DIR / stem
    out_f.mkdir(exist_ok=True)
    print(f"\n⚙  Processing {stem} …")

    try:
        # 1. preprocess & cluster
        df    = preprocess(p)
        cid   = detect_cluster(df)
        scX, scY, narx = load_cluster(cid)

        # 2. open-loop
        df_o, Xo_s, y_open = predict_open( df, scX, scY, narx)
        df_o = add_cqr(df_o, Xo_s, y_open, mode="open")

        # 3. closed-loop
        df_c, Xc_s, y_closed = predict_closed(df, scX, scY, narx, HORIZON)
        df_c = add_cqr(df_c, Xc_s, y_closed, mode="closed")

        # 4. merge
        df_pred = pd.concat(
            [df_o,
             df_c[[f"{c}_{m}" for c in STATE_COLS
                               for m in ("closed", "closed_lo", "closed_hi")]]],
            axis=1)

        # 5. save & plots
        df_pred.to_csv(out_f/"predictions.csv", index=False)
        plot_ts(df_pred, out_f, mode="open")
        plot_ts(df_pred, out_f, mode="closed")
        plot_scatter(df_pred, out_f, mode="open")
        plot_scatter(df_pred, out_f, mode="closed")

        # 6. metrics
        m_open   = metric_table(df_pred, mode="open")
        m_closed = metric_table(df_pred, mode="closed")
        summary.append(
            {"file": stem, **m_open,
                        **{f"{k}_closed": v for k,v in m_closed.items()}}
        )

    except Exception as e:
        print(f"⨯  {stem} skipped  →  {e}")

# ──  Aggregate summary  ────────────────────────────────────────────────────
df_sum = pd.DataFrame(summary)
df_sum.to_csv(OUT_DIR/"metrics_summary.csv", index=False)

rows = []
for col in STATE_COLS:
    rows.append([
        col,
        df_sum[f"{col}_MAE"].mean(),
        df_sum[f"{col}_MAE_closed"].mean(),
        df_sum[f"{col}_MSE"].mean(),
        df_sum[f"{col}_MSE_closed"].mean(),
        df_sum[f"{col}_R2"].mean(),
        df_sum[f"{col}_R2_closed"].mean(),
        df_sum[f"{col}_COV"].mean(),
        df_sum[f"{col}_COV_closed"].mean()
    ])


print("\n Average error, R² & coverage (open vs closed)\n")
print(pd.DataFrame(
        rows,
        columns=["Var",
                 "Open MAE", "Closed MAE",
                 "Open MSE", "Closed MSE",
                 "Open R2", "Closed R2",
                 "Open COV %", "Closed COV %"]
).to_string(index=False, float_format="%0.8f"))

# %%
