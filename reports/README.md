# SFC Project Training Pipeline Reports

## Overview
This directory contains comprehensive reports for each stage of the SFC (Crystallization Process) machine learning pipeline implemented in `latest_modeltrain1.py`. Each report provides detailed analysis of the methodology, implementation, and results for specific components of the training process.

## Report Index

### 1. [Data Preprocessing Report](01_Data_Preprocessing_Report.md)
**Focus:** Data cleaning, visualization, and quality assessment
- **Key Topics:**
  - IQR-based outlier detection and treatment
  - Comprehensive data visualization pipeline
  - Unit standardization for particle sizes
  - Data splitting strategy (train/calibration)
  - Quality metrics and statistical analysis
- **Files Generated:** Distribution plots, correlation matrices, statistical summaries
- **Data Retention:** Typically 85-95% depending on data quality

### 2. [Data Clustering Report](02_Data_Clustering_Report.md)
**Focus:** K-means clustering analysis and operational regime identification
- **Key Topics:**
  - Feature engineering (56-dimensional feature vectors)
  - K-means clustering with 2 clusters
  - Quality assessment metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
  - Feature importance analysis
  - Cluster separation and visualization
- **Files Generated:** PCA visualizations, feature importance plots, cluster distribution charts
- **Quality Metrics:** Silhouette score, cluster balance, feature discrimination

### 3. [Preparing Data for NARX Training Report](03_Preparing_Data_for_NARX_Training_Report.md)
**Focus:** Data preparation for NARX model training
- **Key Topics:**
  - Lag matrix creation (LAG=25, 260 input features)
  - Per-cluster data preparation
  - Data scaling and normalization
  - Train-validation split (80/20)
  - Model architecture preparation
- **Data Structure:** 260 input features, 6 output variables
- **Scaling Strategy:** Per-cluster StandardScaler for inputs and outputs

### 4. [NARX Training Report](04_NARX_Training_Report.md)
**Focus:** Nonlinear AutoRegressive with eXogenous inputs model training
- **Key Topics:**
  - Deep neural network architecture with residual connections
  - Per-cluster model training (2 clusters)
  - Weighted MSE loss function
  - Advanced training callbacks (early stopping, LR scheduling, warm-up)
  - Training monitoring and visualization
- **Architecture:** 4 hidden layers (1024→512→256→128→6)
- **Training Features:** SELU activation, batch normalization, dropout, L2 regularization

### 5. [Preparing for QR Training Report](05_Preparing_for_QR_Training_Report.md)
**Focus:** Data preparation for Quantile Regression training
- **Key Topics:**
  - Cross-cluster residual collection
  - NARX model prediction and error calculation
  - Data aggregation and sampling (70,000 samples max)
  - QR model architecture preparation
  - Quality assurance and validation
- **Data Structure:** 260 input features, 6 residual variables
- **Sampling Strategy:** First-come-first-served with 70K sample limit

### 6. [QR Training Report](06_QR_Training_Report.md)
**Focus:** Quantile Regression model training for uncertainty quantification
- **Key Topics:**
  - Variable-specific QR architectures (standard vs deep)
  - Pinball loss function for quantile estimation
  - Per-variable training (6 variables × 2 quantiles = 12 models)
  - Conformal delta calculation for calibration
  - Training monitoring and model persistence
- **Quantiles:** τ = 0.1 (lower bound), τ = 0.9 (upper bound)
- **Coverage Target:** 90% prediction intervals with conformal calibration

## Pipeline Summary

### Overall Architecture
```
Raw Data → Preprocessing → Clustering → NARX Training → QR Training → Conformal Calibration
```

### Key Components
1. **Data Preprocessing:** IQR cleaning, visualization, unit standardization
2. **Clustering:** K-means with 2 clusters for operational regime identification
3. **NARX Training:** Per-cluster deep neural networks with residual connections
4. **QR Training:** Variable-specific quantile regression for uncertainty quantification
5. **Conformal Calibration:** Delta calculation for proper coverage guarantee

### Model Inventory
- **NARX Models:** 2 cluster-specific models
- **QR Models:** 12 models (6 variables × 2 quantiles)
- **Scalers:** 4 scalers (2 input + 2 output per cluster)
- **Metadata:** Configuration and model information

### Performance Characteristics
- **Data Volume:** 70,000 samples for QR training
- **Input Dimensions:** 260 features (26 time steps × 10 variables)
- **Output Dimensions:** 6 state variables
- **Coverage:** 90% prediction intervals with conformal calibration
- **Architecture:** Deep networks with residual connections and comprehensive regularization

### Quality Assurance
- **Reproducibility:** Fixed random seed (42)
- **Validation:** Early stopping and comprehensive monitoring
- **Visualization:** Extensive plotting and analysis
- **Persistence:** Complete model and metadata storage

## File Organization
```
reports/
├── README.md                           # This overview file
├── 01_Data_Preprocessing_Report.md     # Data preprocessing analysis
├── 02_Data_Clustering_Report.md        # Clustering analysis
├── 03_Preparing_Data_for_NARX_Training_Report.md  # NARX data preparation
├── 04_NARX_Training_Report.md          # NARX model training
├── 05_Preparing_for_QR_Training_Report.md         # QR data preparation
└── 06_QR_Training_Report.md            # QR model training
```

## Usage
Each report can be read independently or as part of the complete pipeline analysis. The reports provide:
- **Methodology:** Detailed explanation of approaches used
- **Implementation:** Code snippets and configuration details
- **Results:** Performance metrics and quality assessments
- **Recommendations:** Suggestions for improvement and optimization

## Technical Details
- **Framework:** TensorFlow/Keras for neural networks
- **Language:** Python 3
- **Data Format:** TAB-separated text files
- **Units:** Metres for particle sizes
- **Precision:** Float32 for memory efficiency
- **Visualization:** Matplotlib and Seaborn

This comprehensive reporting system provides complete documentation of the SFC project's machine learning pipeline, enabling reproducibility, understanding, and future improvements. 