# NARX Training and Preprocessing Pipeline

This document details the data preprocessing and NARX (Nonlinear AutoRegressive with eXogenous inputs) model training workflow for the SFC project (Machine Learning Methods for Engineers, SS 25).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Preprocessing](#data-preprocessing)
   - [Data Sources](#data-sources)
   - [Cleaning Steps](#cleaning-steps)
   - [Data Splitting](#data-splitting)
3. [Feature Engineering](#feature-engineering)
4. [Clustering](#clustering)
5. [NARX Model Training](#narx-model-training)
   - [Per-Cluster Data Preparation](#per-cluster-data-preparation)
   - [Model Architecture](#model-architecture)
   - [Training Procedure](#training-procedure)
6. [Artifacts and Outputs](#artifacts-and-outputs)
7. [Reproducibility](#reproducibility)
8. [Directory Structure](#directory-structure)
9. [How to Run](#how-to-run)
10. [Troubleshooting & Tips](#troubleshooting--tips)
11. [References](#references)

---

## Project Overview

This pipeline is designed for robust, cluster-specific time series modeling of process data using NARX neural networks. The workflow includes:
- Data cleaning and outlier handling
- Clustering of process conditions
- Per-cluster NARX model training
- Quantile regression and conformal prediction (for uncertainty estimation)

The goal is to build accurate, interpretable, and robust predictive models for process engineering data, with a focus on reproducibility and data quality.

---

## Data Preprocessing

### Data Sources

- **Location:** `Data/RAW DATA/`
- **Format:** Tab-separated `.txt` files, each representing a time series from a process run.
- **Columns:**
  - **State Columns:** `['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']`
  - **Exogenous Columns:** `['mf_PM', 'mf_TM', 'Q_g', 'w_crystal']`

### Cleaning Steps

1. **NaN and Out-of-Range Removal:**
   - Drops rows with missing values in any of the key columns.
   - Removes rows with physically implausible values (e.g., negative mass flows, temperatures outside [250, 400] K, non-positive particle sizes).
   - This ensures only valid, meaningful data is used for modeling.

2. **Outlier Handling (IQR Method):**
   - For each column, compute the Interquartile Range (IQR).
   - Cap values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` to the respective bounds.
   - This method preserves more data than outright row removal, while reducing the influence of extreme outliers.
   - Outlier statistics and bounds are saved for transparency.

3. **Trash File Filtering:**
   - Files with extreme median d-values (d10, d50, d90) are moved to a `trash` directory and excluded from further analysis.
   - This prevents corrupted or anomalous runs from biasing the model.

4. **Data Quality Reporting:**
   - The script generates visualizations and JSON reports on data loss, outlier removal, and column-wise statistics.
   - Plots include histograms, boxplots, time series, and correlation matrices for before/after cleaning.

### Data Splitting

- **Train/Calibration Split:**
  - 80% of files for training, 20% for calibration/validation.
  - The split is random but reproducible (fixed seed).
  - Trash files are excluded from both sets.

---

## Feature Engineering

### Lagged Matrix Construction

- For each file, a lagged design matrix is built:
  - **Input (X):**
    - Concatenation of `[y_t, u_t, y_{t-1}, u_{t-1}, ..., y_{t-LAG}, u_{t-LAG}]` for each time step.
    - Where `y` = state variables, `u` = exogenous inputs.
    - This captures the temporal dynamics and dependencies in the process.
  - **Output (Y):**
    - Next-step state variables (`y_{t+1}`), i.e., the target for prediction.
- **LAG:**
  - The number of past steps used (default: 40).
  - This is a key hyperparameter for capturing process memory.

---

## Clustering

### Purpose
- To group similar process conditions and train specialized models for each group (cluster).
- This can improve model accuracy and interpretability by reducing heterogeneity.

### Method
- For each file, compute summary statistics (mean, std, min, max) for all relevant columns.
- Standardize features using `StandardScaler`.
- Apply KMeans clustering (`N_CLUSTERS` = 2 by default).
- Save the scaler and KMeans model for inference and reproducibility.

### Quality Metrics & Visualization
- **Silhouette Score:** Measures how well samples are clustered (higher is better).
- **Calinski-Harabasz Score:** Ratio of between-cluster to within-cluster dispersion (higher is better).
- **Davies-Bouldin Score:** Measures average similarity between clusters (lower is better).
- **PCA Visualization:** 2D scatter plot of clusters using principal components.
- **Feature Importance:** Bar chart showing which features drive cluster separation.
- **Cluster Distribution:** Bar chart of file counts per cluster.

---

## NARX Model Training

### Per-Cluster Data Preparation
- For each cluster:
  - Collect all files assigned to the cluster.
  - Build lagged X/Y matrices for each file and concatenate.
  - Standardize X and Y using `StandardScaler` (scalers saved per cluster).
  - Split into training (80%) and validation (20%) sets.

### Model Architecture
- **Input:**
  - Flattened lagged features (dimension: `(LAG+1) * (num_state + num_exog)`).
- **Network:**
  - Deep feedforward neural network with:
    - Multiple dense layers (sizes: 1024 → 768 → 512 → 256 → 128).
    - SELU activations for robust learning.
    - Batch normalization for stable training.
    - Dropout for regularization.
    - Residual and projection shortcuts for improved gradient flow.
    - Final dense output layer (linear activation).
- **Loss:**
  - Weighted MSE, with higher weights for PSD columns (d10, d50, d90) to emphasize their importance.
- **Optimizer:**
  - AdamW with learning rate scheduling, gradient clipping, and weight decay for robust optimization.

### Training Procedure
- **Callbacks:**
  - Early stopping (with patience and min-delta) to prevent overfitting.
  - Model checkpointing (save best model based on validation loss).
  - ReduceLROnPlateau (adaptive learning rate reduction on plateau).
  - Warm-up learning rate scheduler for smooth start.
  - Custom convergence callback for additional stopping criteria.
- **Reporting:**
  - Training and validation loss curves are plotted and saved.
  - Detailed JSON reports are generated per cluster, including:
    - Training/validation sizes, best epoch, loss improvements, convergence analysis, and model configuration.
- **Artifacts Saved:**
  - Trained model (`.keras`), scalers (`.pkl`), training curves (`.png`), and reports (`.json`) for each cluster.

#### Why NARX?
- NARX models are well-suited for time series with exogenous inputs and can capture complex, nonlinear dependencies with memory.
- The per-cluster approach allows the model to specialize for different process regimes.

---

## Artifacts and Outputs

- **Models:**
  - One NARX model per cluster (`.keras`), each with its own input/output scalers (`.pkl`).
- **Reports:**
  - Training curves (`.png`), loss improvements, convergence diagnostics, and detailed JSON summaries.
- **Metadata:**
  - Model input/output column names, lag value, and clustering configuration (`metadata.json`).
- **Visualizations:**
  - Data quality, clustering, and training process plots for transparency and debugging.

---

## Reproducibility

- **Deterministic Seeds:**
  - All random seeds are set for Python, NumPy, and TensorFlow to ensure reproducible results.
- **Clean Slate:**
  - Model output directories are cleared before each run to avoid shape mismatches or stale artifacts.
- **Saved Artifacts:**
  - All scalers, models, and reports are saved with clear naming for traceability.

---

## Directory Structure

```
model_5files19/
  narx/
    cluster_0.keras
    cluster_1.keras
    scaler_X_0.pkl
    scaler_Y_0.pkl
    scaler_X_1.pkl
    scaler_Y_1.pkl
    reports/
      cluster_0_narx_report.json
      cluster_1_narx_report.json
      cluster_0_narx_analysis.png
      cluster_1_narx_analysis.png
  metadata.json
  data_visualization/
  clustering_analysis/
```

---

## How to Run

1. **Install dependencies:**
   - Python 3.8+
   - Install required packages:
     ```
     pip install -r requirements.txt
     ```
     (Ensure TensorFlow, scikit-learn, pandas, matplotlib, seaborn, etc. are included)

2. **Prepare data:**
   - Place raw `.txt` files in `Data/RAW DATA/`.

3. **Run the script:**
   ```
   python latest_modeltrain13.py
   ```

4. **Outputs:**
   - Models, scalers, reports, and visualizations will be saved in the `model_5files19/` directory.

---

## Troubleshooting & Tips

- **Shape Mismatches:**
  - The script deletes the model output directory at the start to avoid shape mismatches from previous runs.
- **Data Quality:**
  - Check the `data_visualization/` and `clustering_analysis/` folders for plots and JSON reports on data quality and clustering.
- **Customizing Clusters:**
  - Change `N_CLUSTERS` in the script to adjust the number of clusters.
- **Adjusting LAG:**
  - Modify the `LAG` variable to change the history window for the NARX model.
- **Performance:**
  - Training deep NARX models can be computationally intensive. Use a machine with sufficient RAM and a GPU if possible.
- **Extending the Pipeline:**
  - The script is modular; you can add new preprocessing steps, change the model architecture, or add new evaluation metrics as needed.

---

## References

- See `latest_modeltrain13.py` for full implementation details.
- For further information on the NARX model and clustering rationale, refer to the project report.
- [NARX Neural Networks - Wikipedia](https://en.wikipedia.org/wiki/Nonlinear_autoregressive_exogenous_model)
- [KMeans Clustering - scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

---

**This pipeline ensures robust, cluster-specific NARX models with thorough data cleaning, feature engineering, and reproducible training.** 