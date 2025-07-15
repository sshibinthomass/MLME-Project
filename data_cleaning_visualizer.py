import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# --- Configuration (copy from latest_modeltrain13.py) ---
PSD_COLS    = ('d10', 'd50', 'd90')
STATE_COLS = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
EXOG_COLS  = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
CLUST_COLS  = STATE_COLS + EXOG_COLS

# --- Cleaning function (copy from latest_modeltrain13.py) ---
def clean_iqr(df: pd.DataFrame) -> pd.DataFrame:
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

def read_txt(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t', engine='python').apply(pd.to_numeric, errors='coerce')

def plot_raw_vs_cleaned_all(df_raw, df_cleaned, file_stem, out_dir):
    cols_to_plot = [col for col in CLUST_COLS if col in df_raw.columns and col in df_cleaned.columns]
    n_cols = 2
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), squeeze=False)
    for idx, col in enumerate(cols_to_plot):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.plot(df_raw.index, df_raw[col], label='Raw', color='red', alpha=0.7)
        ax.plot(df_cleaned.index, df_cleaned[col], label='Cleaned', color='blue', alpha=0.7)
        ax.set_title(f'{col} (Raw vs Cleaned)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)
    # Hide any unused subplots
    for idx in range(len(cols_to_plot), n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols, idx % n_cols])
    plt.tight_layout()
    plt.savefig(out_dir / f'{file_stem}_raw_vs_cleaned_all.png', dpi=150)
    plt.close()

def main():
    if len(sys.argv) < 2:
        print('Usage: python data_cleaning_visualizer.py <input_folder> [output_folder]')
        sys.exit(1)
    input_folder = "Beat-the-Felix"
    output_folder = "Data_visuals"
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    txt_files = list(input_folder.glob('*.txt'))
    if not txt_files:
        print(f'No .txt files found in {input_folder}')
        sys.exit(1)
    for txt_file in txt_files:
        df_raw = read_txt(txt_file)
        df_cleaned = clean_iqr(df_raw.copy())
        plot_raw_vs_cleaned_all(df_raw, df_cleaned, txt_file.stem, output_folder)
        print(f'Processed {txt_file.name}')
    print(f'All plots saved to {output_folder}')

if __name__ == '__main__':
    main() 

# %%