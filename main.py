import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import json, pickle
from pathlib import Path
from preprocessing import (clean_iqr, read_txt,predict_closed,
                                add_cqr,detect_cluster,plot_scatter,
                                plot_ts,metric_table, smooth_log_series,predict_open,smooth_log_psd) #,file_signature ,smooth_log_psd



"""
main script for computing MSE/MAE metrics per file with PLOTS & Predictions saved in file .
Requires separate preprocessing module (preprocessing.py) for data cleaning, plotting,etc.
"""


# ==== PATHS AND GLOBALS (update as needed) ====
file_path   = Path(r"Beat-the-Felix")   # Directory with test file(s)
MODEL_ROOT = Path(r"model_5files17")    # Directory with saved models
OUT_DIR    = MODEL_ROOT/"BEAT_smooth"     # Where to save outputs
OUT_DIR.mkdir(exist_ok=True)

# ==== LOAD METADATA FROM TRAINING ====
meta       = json.loads((MODEL_ROOT/'metadata.json').read_text())
STATE_COLS = meta['state_cols']
EXOG_COLS  = meta['exog_cols']
LAG        = meta['lag']
CLUST_COLS = STATE_COLS + EXOG_COLS
PREDICT=3
PSD_COLS   = ('d10', 'd50', 'd90')

# --- DATA PROCESSING ---
def process_data(file_path):
    """Read, clean, and preprocess a single data file."""
    df = read_txt(file_path)
    df = clean_iqr(df)
    df = smooth_log_psd(df, columns=['d10', 'd50', 'd90'], window=3)
    return df

# --- MODEL LOADING ---
def load_cluster_models(cid, model_root=MODEL_ROOT):
    """Load the scalers and trained model for a specific cluster."""
    scX = pickle.loads((model_root/f'narx/scaler_X_{cid}.pkl').read_bytes())
    scY = pickle.loads((model_root/f'narx/scaler_Y_{cid}.pkl').read_bytes())
    narx = tf.keras.models.load_model(model_root/f'narx/cluster_{cid}.keras', compile=False)
    return scX, scY, narx

# --- EVALUATION/PREDICTION ---
def evaluate_on_file(file_path, out_dir, model_root=MODEL_ROOT):
    """Run full prediction pipeline on a single file, save results and plots."""
    stem = file_path.stem
    out_f = out_dir / stem
    out_f.mkdir(exist_ok=True)

    # 1. Data Processing
    df = process_data(file_path)

    # 2. Cluster Detection & Model Loading
    cid = detect_cluster(df)
    scX, scY, narx = load_cluster_models(cid, model_root)

    print(f"\nProcessing {stem} [Cluster {cid}]")
    print(f"LAG={LAG}, STATE_COLS={STATE_COLS}, EXOG_COLS={EXOG_COLS}")
    print(f"Input dimension expected by model: {narx.input_shape}")
    print('wait process starting now... .  .   .')
    # 3. Prediction & Uncertainty
    df_closed, X_closed, y_closed = predict_closed(df, scX, scY, narx)
    df_c = add_cqr(df_closed, X_closed, y_closed, mode="closed")

    df_open, X_open, y_open = predict_open(df, scX, scY, narx,PREDICT)
    df_o = add_cqr(df_open, X_open, y_open, mode="open")

    #df_open, X_open, y_open = predict_recursive_open(df, scX, scY, narx, lag=LAG)

    # 4. Merge and Save
    df_pred = pd.concat(
        [df_c, 
         df_o[[f"{c}_{m}" for c in STATE_COLS
                              for m in ("open", "open_lo", "open_hi")]]],
        axis=1)
    
    # Smooth d10, d50, d90 predictions in log-space
    df_pred = smooth_log_series(
    df_pred,
    columns=[
        'd10_closed', 'd50_closed', 'd90_closed',
        'd10_open', 'd50_open', 'd90_open'
    ],
    window=5,    # You can try 3, 5, 7, etc. 
    suffix='_smoothed'
      )

    df_pred.to_csv(out_f/"predictions.csv", index=False)

    # 5. Plots
    plot_ts(df_pred, out_f, mode="closed")
    plot_ts(df_pred, out_f, mode="open")
    plot_scatter(df_pred, out_f, mode="closed")
    plot_scatter(df_pred, out_f, mode="open")

    # 6. Metrics
    m_closed = metric_table(df_pred, mode="closed")
    m_open = metric_table(df_pred, mode="open")
    result_row = {"file": stem, **m_closed, **{f"{k}_open": v for k, v in m_open.items()}}
    return result_row

# --- MAIN PIPELINE ---
def run_prediction_on_dir(test_dir=file_path, out_dir=OUT_DIR):
    summary = []
    for p in sorted(test_dir.glob("*.txt")):
        try:
            result_row = evaluate_on_file(p, out_dir)
            summary.append(result_row)
        except Exception as e:
            print(f"  {p.stem} skipped  →  {e}")

    # Summarize results
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(out_dir/"metrics_summary.csv", index=False)
    states = STATE_COLS
    table_rows = []
    for col in states:
        row = [
        
            col,
            df_sum[f"{col}_MSE"].mean(),
            df_sum[f"{col}_MAE"].mean(),
            df_sum[f"{col}_MSE_open"].mean(),
            df_sum[f"{col}_MAE_open"].mean()
        ]
        table_rows.append(row)

    header = (
        f"{'':<6} {'Closed loop':^25} {'Open Loop':^25}\n"
        f"{'State':<6} {'MSE':>10} {'MAE':>10}  {'MSE':>10} {'MAE':>10}"
    )
    print(header)
    for row in table_rows:
        print(f"{row[0]:<6} {row[1]:10.3e} {row[2]:10.3e}  {row[3]:10.3e} {row[4]:10.3e}")

# --- RUN MAIN ---
if __name__ == "__main__":
    run_prediction_on_dir()