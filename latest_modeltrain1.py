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
MODEL_DIR  = Path(r"model_5files4") 
# clean slate (avoids shape mismatches when you change LAG etc.)
if MODEL_DIR.exists():
    shutil.rmtree(MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR/"narx").mkdir()
(MODEL_DIR/"qr").mkdir()

LAG             = 20                # number of past steps
N_CLUSTERS      = 2
EPOCHS_NARX     = 200
EPOCHS_QR       = 100
BATCH_SIZE      = 32
QUANTILES       = [0.1, 0.9]
OUTPUT_WEIGHTS  = np.array([10, 6, 15, 15, 15, 10], dtype="float32")  # higher weight to PSD

# Column layout  (matches report)
STATE_COLS = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
EXOG_COLS  = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal',
              'c_in', 'T_PM_in', 'T_TM_in']
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
    Clear out outliers from data using IQR (Interquantile Range) method
    (Also includes extra steps to drop empty rows)
    """
    df = df.dropna(subset=CLUST_COLS) #Drop empty rows

    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR

        df[column] = df[column].apply(
            lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
        )

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
    #df = clean_df(read_txt(path)) ##Used IQR method to clean data outliers
    #df = to_metres(df)        # <<< make sure we are in metres  ## This is not needed anymore since we moved the trash data from the data directory
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
    return Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim)
    ])

def build_qr(input_dim: int) -> tf.keras.Model:
    return Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64 , activation='relu'),
        layers.Dense(1)
    ])

def pinball_loss(tau: float):
    """Pinball / quantile loss for τ."""
    def loss(y, y_hat):
        e = y - y_hat
        return tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
    return loss

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
# 2. Unsupervised clustering on summary stats  ➜  k-means labels.
# --------------------------------------------------------------------------
train_files = sorted(train_dir.glob("*.txt"))

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

# ──────────────────────────────────────────────────────────────────────────
#%% 3. Train a separate NARX per cluster (with scaler per cluster).
# --------------------------------------------------------------------------
print("\n Training per-cluster NARX models …")
for cid in range(N_CLUSTERS):
    Xc, Yc = [], []
    for idx, p in enumerate(train_files):
        if kmeans.labels_[idx] != cid:
            continue
        x, y = make_xy(preprocess(p))
        if len(x):
            Xc.append(x); Yc.append(y)
    if not Xc:
        continue

    Xc = np.vstack(Xc);  Yc = np.vstack(Yc)
    scX = StandardScaler().fit(Xc)
    scY = StandardScaler().fit(Yc)

    pickle.dump(scX, (MODEL_DIR/f'narx/scaler_X_{cid}.pkl').open('wb'))
    pickle.dump(scY, (MODEL_DIR/f'narx/scaler_Y_{cid}.pkl').open('wb'))

    # train/val split 80/20
    split = int(0.8 * len(Xc))
    Xtr, Ytr = Xc[:split], Yc[:split]
    Xvl, Yvl = Xc[split:], Yc[split:]

    model = build_narx(Xc.shape[1], Yc.shape[1])
    model.compile(optimizer=optimizers.Adam(1e-3), loss=weighted_mse)

    es = callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    ck = callbacks.ModelCheckpoint(
        filepath=MODEL_DIR/f'narx/cluster_{cid}.keras',
        monitor='val_loss',
        save_best_only=True
    )

    model.fit(scX.transform(Xtr), scY.transform(Ytr),
              validation_data=(scX.transform(Xvl), scY.transform(Yvl)),
              epochs=EPOCHS_NARX,
              batch_size=BATCH_SIZE,
              verbose=0,
              callbacks=[es, ck])

print(" NARX training done.")

#%% -- store global metadata (helps inference script remain agnostic) --------
json.dump({'state_cols': STATE_COLS, 'exog_cols': EXOG_COLS, 'lag': LAG},
          (MODEL_DIR/'metadata.json').open('w'))

# ──────────────────────────────────────────────────────────────────────────
# 4. Collect NARX validation residuals across *all* clusters  ➜  training
#    set for Quantile-Regression nets.
# --------------------------------------------------------------------------
val_X, val_E = [], []
for cid in range(N_CLUSTERS):
    scaler_x = pickle.loads((MODEL_DIR/f'narx/scaler_X_{cid}.pkl').read_bytes())
    scaler_y = pickle.loads((MODEL_DIR/f'narx/scaler_Y_{cid}.pkl').read_bytes())
    narx     = tf.keras.models.load_model(MODEL_DIR/f'narx/cluster_{cid}.keras',
                                          compile=False)

    for p in train_files:
        # predict cluster id directly from already-computed feature → faster
        sig = sc_feat.transform([feat[kmeans.labels_ == cid][0]])  # re-use stats
        if kmeans.predict(sig)[0] != cid:
            continue

        X, Y = make_xy(preprocess(p))
        if not len(X):
            continue

        y_hat = scaler_y.inverse_transform(
                    narx.predict(scaler_x.transform(X), verbose=0))
        val_X.append(scaler_x.transform(X))
        val_E.append(Y - y_hat)

val_X = np.vstack(val_X)
val_E = np.vstack(val_E)

# ──────────────────────────────────────────────────────────────────────────
#%% 5. Train Quantile-Regression nets (τ = 0.1 & 0.9) on residuals.
# --------------------------------------------------------------------------


print("\n Training QR nets …")

MAX_QR_SAMPLES=70000
val_X_qr=val_X[:MAX_QR_SAMPLES]
val_E_qr=val_E[:MAX_QR_SAMPLES]

loss_dir = MODEL_DIR / 'qr/loss_curves'
loss_dir.mkdir(parents=True, exist_ok=True)

for j, col in enumerate(STATE_COLS):
    y_err = val_E_qr[:, j:j+1]      # residual for that state
    for q in QUANTILES:
        qr = Sequential([
             layers.Input(shape=(val_X_qr.shape[1],)),
             layers.Dense(128, activation='relu'),
             layers.Dropout(0.2),
             layers.Dense(64, activation='relu'),
             layers.Dense(1)
         ])

        
        qr.compile(optimizer=optimizers.Adam(1e-4), loss=pinball_loss(q))

        es_qr = callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor="val_loss"
        )

        history = qr.fit(val_X_qr, y_err,
                         validation_split=0.2,
                         epochs=EPOCHS_QR,
                         batch_size=BATCH_SIZE,
                         verbose=0,
                         callbacks=[es_qr])

        # Save model
        qr.save(MODEL_DIR / f'qr/{col}_{q:.1f}.keras')

        # Plot loss curves
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='val')
        plt.title(f"{col}  τ={q:.1f}")
        plt.xlabel("Epoch")
        plt.ylabel("Pinball loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_dir / f"{col}_{q:.1f}_loss.png", dpi=150)
        plt.close()

print(" QR nets done.")


# ──────────────────────────────────────────────────────────────────────────
#%% 6. Conformal deltas  (90 % Prediction-Interval)  in **metres**.
# --------------------------------------------------------------------------
print("\n Computing conformal deltas …")
alpha  = 0.1                     # 90 % coverage target
deltas = {}
for j, col in enumerate(STATE_COLS):
    nonconf = np.abs(val_E_qr[:, j])          # |error|
    k       = int(np.ceil((1-alpha) * (len(nonconf)+1))) - 1
    deltas[col] = float(np.sort(nonconf)[k])

pickle.dump(deltas, (MODEL_DIR/'conformal_deltas.pkl').open('wb'))
print(" Saved deltas:", deltas)

print("\n Finished training. Model at:", MODEL_DIR)

# %%
