#%%!/usr/bin/env python3  #%%!usr/bin/env python3
"""
SFC NARX + CQR Training Pipeline (with cluster and training visualizations)
Row-based filtering: keep only rows with d10, d50, d90 <= 0.0001
"""

import os, random, pickle, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Sequential, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Paths and constants ---
RAW_ROOT     = Path(r"C:/Users/nishi/Desktop/MLME Project/Data/RAW DATA")
MODEL_DIR    = Path(r"C:/Users/nishi/Desktop/MLME Project/model_newlogic")
PLOT_DIR     = MODEL_DIR / "plots"
COMBINED_CSV = RAW_ROOT / "all_cleaned_combined.csv"

LAG             = 10
#N_CLUSTERS      = 2
EPOCHS_NARX     = 70
EPOCHS_QR       = 40
BATCH_SIZE      = 32
QUANTILES       = [0.1, 0.9]
OUTPUT_WEIGHTS  = np.array([10, 6, 15, 15, 15, 10], dtype="float32")

STATE_COLS = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
EXOG_COLS  = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal',
              'c_in', 'T_PM_in', 'T_TM_in']
OUTPUT_COLS = STATE_COLS
CLUST_COLS  = STATE_COLS + EXOG_COLS
PSD_COLS    = ['d10', 'd50', 'd90']

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Create directories
for d in [MODEL_DIR, PLOT_DIR, MODEL_DIR/"narx", MODEL_DIR/"qr"]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Data loading and cleaning ----------------------------------------------
all_dfs = []
for txt_file in RAW_ROOT.glob("*.txt"):
    try:
        df = pd.read_csv(txt_file, sep="\t", engine="python").apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=PSD_COLS)
        # Keep only rows where all d10, d50, d90 <= 0.0001
        df = df[(df['d10'] <= 0.001) & (df['d50'] <= 0.001) & (df['d90'] <= 0.001)]
        if len(df) > 0:
            all_dfs.append(df)
    except Exception as e:
        print(f"Error processing {txt_file.name}: {e}")

if not all_dfs:
    raise RuntimeError("No data met the d10/d50/d90 <= 0.0001 criteria.")

df_all = pd.concat(all_dfs, axis=0, ignore_index=True)
df_all.to_csv(COMBINED_CSV, index=False)
print(f"Combined clean data shape: {df_all.shape} (saved to {COMBINED_CSV})")

# 2. Split combined data into train/calibration -----------------------------
shuffled_idx = np.random.permutation(len(df_all))
n_cal = int(0.2 * len(df_all))
cal_idx = shuffled_idx[:n_cal]
train_idx = shuffled_idx[n_cal:]
df_train = df_all.iloc[train_idx].reset_index(drop=True)
df_calib = df_all.iloc[cal_idx].reset_index(drop=True)

# 3. Clustering on training data -------------------------------------------
# Cluster on all CLUST_COLS for each row
arr = df_train[CLUST_COLS].values
sc_feat = StandardScaler().fit(arr)
arr_s = sc_feat.transform(arr)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit(arr_s)
pickle.dump(sc_feat, (MODEL_DIR/'feature_scaler.pkl').open('wb'))
pickle.dump(kmeans , (MODEL_DIR/'kmeans_model.pkl' ).open('wb'))

train_labels = kmeans.predict(arr_s)
df_train['cluster'] = train_labels

# --- Visualization: PCA plot of clusters ----------------------------------
pca = PCA(n_components=2)
arr_pca = pca.fit_transform(arr_s)
plt.figure(figsize=(7,5))
for cid in range(N_CLUSTERS):
    plt.scatter(arr_pca[train_labels==cid, 0], arr_pca[train_labels==cid, 1], 
                s=8, alpha=0.7, label=f"Cluster {cid}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KMeans clusters (PCA projection)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "clusters_pca.png", dpi=150)
plt.close()
print(f"Saved PCA cluster plot to {PLOT_DIR/'clusters_pca.png'}")

# --- Visualization: Distribution of each cluster --------------------------
for col in STATE_COLS+EXOG_COLS:
    plt.figure(figsize=(7,4))
    for cid in range(N_CLUSTERS):
        data = df_train[df_train['cluster']==cid][col]
        plt.hist(data, bins=40, alpha=0.6, label=f"Cluster {cid}")
    plt.title(f"Distribution of {col} per cluster")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"hist_{col}.png", dpi=150)
    plt.close()

#%% 4. Helper: NARX sequence builder -----------------------------------------
def make_xy(df, lag=LAG):
    X, Y = [], []
    for i in range(lag, len(df)-1):
        hist = []
        for l in range(0, lag+1):  # 0 ... LAG
            idx = i - l
            hist.extend(df[STATE_COLS + EXOG_COLS].iloc[idx].values)
        X.append(hist)
        Y.append(df[STATE_COLS].iloc[i+1].values)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)

# 5. Loss functions and model builders -------------------------------------
def weighted_mse(y_true, y_pred):
    w = tf.constant(OUTPUT_WEIGHTS, dtype=y_true.dtype)
    return tf.reduce_mean(tf.reduce_mean(tf.square(y_true - y_pred) * w, axis=-1))

def build_narx(input_dim, output_dim):
    return Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim)
    ])

def pinball_loss(tau):
    def loss(y, y_hat):
        e = y - y_hat
        return tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
    return loss

#%% 6. Train NARX models per cluster -----------------------------------------
print("\nTraining per-cluster NARX models ...")
for cid in range(N_CLUSTERS):
    sub = df_train[df_train['cluster'] == cid].reset_index(drop=True)
    Xc, Yc = make_xy(sub)
    if len(Xc) == 0:
        print(f"Cluster {cid}: No data after lag trimming. Skipping.")
        continue
    scX = StandardScaler().fit(Xc)
    scY = StandardScaler().fit(Yc)
    pickle.dump(scX, (MODEL_DIR/f'narx/scaler_X_{cid}.pkl').open('wb'))
    pickle.dump(scY, (MODEL_DIR/f'narx/scaler_Y_{cid}.pkl').open('wb'))
    # Train/val split
    split = int(0.8 * len(Xc))
    Xtr, Ytr = Xc[:split], Yc[:split]
    Xvl, Yvl = Xc[split:], Yc[split:]
    model = build_narx(Xc.shape[1], Yc.shape[1])
    model.compile(optimizer=optimizers.Adam(1e-3), loss=weighted_mse)
    es = callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    ck = callbacks.ModelCheckpoint(
        filepath=MODEL_DIR/f'narx/cluster_{cid}',
        monitor='val_loss',
        save_best_only=True,
        save_format='tf'
    )
    history = model.fit(
        scX.transform(Xtr), scY.transform(Ytr),
        validation_data=(scX.transform(Xvl), scY.transform(Yvl)),
        epochs=EPOCHS_NARX, batch_size=BATCH_SIZE,
        verbose=0, callbacks=[es, ck]
    )
    # Plot training and validation loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted MSE')
    plt.title(f'NARX Loss (Cluster {cid})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f'narx_loss_cluster_{cid}.png', dpi=150)
    plt.close()
    print(f"  Cluster {cid}: Trained NARX ({len(Xc)} samples)")

print("NARX training done.")

# Save global metadata
json.dump({'state_cols': STATE_COLS, 'exog_cols': EXOG_COLS, 'lag': LAG},
          (MODEL_DIR/'metadata.json').open('w'))

# 7. Residual dataset for QR -----------------------------------------------
val_X, val_E = [], []
for cid in range(N_CLUSTERS):
    scX = pickle.loads((MODEL_DIR/f'narx/scaler_X_{cid}.pkl').read_bytes())
    scY = pickle.loads((MODEL_DIR/f'narx/scaler_Y_{cid}.pkl').read_bytes())
    narx = tf.keras.models.load_model(MODEL_DIR/f'narx/cluster_{cid}', compile=False)
    sub = df_train[df_train['cluster'] == cid].reset_index(drop=True)
    X, Y = make_xy(sub)
    if not len(X):
        continue
    y_hat = scY.inverse_transform(narx.predict(scX.transform(X), verbose=0))
    val_X.append(scX.transform(X))
    val_E.append(Y - y_hat)
val_X = np.vstack(val_X)
val_E = np.vstack(val_E)

# 8. Quantile Regression nets ----------------------------------------------
print("\nTraining QR nets ...")
MAX_QR_SAMPLES = 70000
val_X_qr = val_X[:MAX_QR_SAMPLES]
val_E_qr = val_E[:MAX_QR_SAMPLES]
loss_dir = MODEL_DIR / 'qr/loss_curves'
loss_dir.mkdir(parents=True, exist_ok=True)

for j, col in enumerate(STATE_COLS):
    y_err = val_E_qr[:, j:j+1]
    for q in QUANTILES:
        qr = Sequential([
             layers.Input(shape=(val_X_qr.shape[1],)),
             layers.Dense(128, activation='relu'),
             layers.Dropout(0.2),
             layers.Dense(64, activation='relu'),
             layers.Dense(1)
        ])
        qr.compile(optimizer=optimizers.Adam(1e-4), loss=pinball_loss(q))
        es_qr = callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
        history = qr.fit(val_X_qr, y_err, validation_split=0.2,
                         epochs=EPOCHS_QR, batch_size=BATCH_SIZE, verbose=0,
                         callbacks=[es_qr])
        qr.save(MODEL_DIR / f'qr/{col}_{q:.1f}')
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
print("QR nets done.")

# 9. Conformal calibration -------------------------------------------------
print("\nComputing conformal deltas ...")
alpha  = 0.1
deltas = {}
for j, col in enumerate(STATE_COLS):
    nonconf = np.abs(val_E_qr[:, j])
    k       = int(np.ceil((1-alpha) * (len(nonconf)+1))) - 1
    deltas[col] = float(np.sort(nonconf)[k])
pickle.dump(deltas, (MODEL_DIR/'conformal_deltas.pkl').open('wb'))
print("Saved deltas:", deltas)

print("\nFinished training. Model at:", MODEL_DIR)
print(f"All cluster, histogram, NARX, and QR loss plots are in: {PLOT_DIR}")
