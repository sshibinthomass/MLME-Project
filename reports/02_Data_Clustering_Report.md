# Data Clustering Analysis Report

## Overview
This report details the comprehensive clustering analysis implemented in the SFC project. The clustering stage is essential for identifying distinct operational regimes in the crystallization process, enabling the development of cluster-specific NARX models for improved prediction accuracy.

## Table of Contents
1. [Clustering Objectives and Strategy](#clustering-objectives-and-strategy)
2. [Feature Engineering](#feature-engineering)
3. [Clustering Algorithm and Configuration](#clustering-algorithm-and-configuration)
4. [Quality Assessment Metrics](#quality-assessment-metrics)
5. [Cluster Analysis Results](#cluster-analysis-results)
6. [Feature Importance Analysis](#feature-importance-analysis)
7. [Visualization and Diagnostics](#visualization-and-diagnostics)
8. [Cluster Separation Analysis](#cluster-separation-analysis)
9. [Summary and Recommendations](#summary-and-recommendations)

## Clustering Objectives and Strategy

### Primary Goals
1. **Operational Regime Identification:** Identify distinct operating conditions in the crystallization process
2. **Model Specialization:** Enable cluster-specific NARX models for improved prediction accuracy
3. **Process Understanding:** Gain insights into different operational modes
4. **Data Organization:** Structure data for targeted model training

### Clustering Strategy
- **Algorithm:** K-means clustering
- **Number of Clusters:** 2 (configurable via `N_CLUSTERS`)
- **Feature Space:** Statistical features derived from time series data
- **Scalability:** StandardScaler for feature normalization

## Feature Engineering

### Feature Extraction Process
```python
feat = []
for p in train_files:
    df = preprocess(p)
    arr = df[CLUST_COLS].values
    feat.append(np.concatenate([arr.mean(0), arr.std(0), arr.min(0), arr.max(0)]))
feat = np.vstack(feat)
```

### Feature Vector Composition
For each file, a 56-dimensional feature vector is created:

**Base Columns (14):**
- State columns: `T_PM`, `c`, `d10`, `d50`, `d90`, `T_TM`
- Exogenous columns: `mf_PM`, `mf_TM`, `Q_g`, `w_crystal`

**Statistical Features (4 per column):**
- Mean: Central tendency measure
- Standard deviation: Variability measure
- Minimum: Lower bound of values
- Maximum: Upper bound of values

**Total Feature Dimensions:**
- 14 columns × 4 statistics = 56 features per file

### Feature Normalization
```python
sc_feat = StandardScaler().fit(feat)
feat_s = sc_feat.transform(feat)
```

**Normalization Benefits:**
- Equal weight to all features regardless of scale
- Improved clustering performance
- Numerical stability

## Clustering Algorithm and Configuration

### K-means Implementation
```python
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED).fit(feat_s)
```

**Configuration Parameters:**
- **Algorithm:** K-means clustering
- **Number of clusters:** 2
- **Random state:** 42 (for reproducibility)
- **Convergence:** Standard K-means convergence criteria

### Model Persistence
```python
pickle.dump(sc_feat, (MODEL_DIR/'feature_scaler.pkl').open('wb'))
pickle.dump(kmeans, (MODEL_DIR/'kmeans_model.pkl').open('wb'))
```

**Saved Components:**
- Feature scaler for consistent preprocessing
- Trained K-means model for inference

## Quality Assessment Metrics

### Clustering Quality Metrics

#### 1. Silhouette Score
```python
silhouette = silhouette_score(feat_s, cluster_labels)
```
- **Range:** -1 to 1
- **Interpretation:** Higher values indicate better-defined clusters
- **Thresholds:**
  - > 0.3: Good clustering
  - 0.1-0.3: Fair clustering
  - < 0.1: Poor clustering

#### 2. Calinski-Harabasz Score
```python
calinski = calinski_harabasz_score(feat_s, cluster_labels)
```
- **Range:** 0 to ∞
- **Interpretation:** Higher values indicate better cluster separation
- **Formula:** Between-cluster variance / Within-cluster variance

#### 3. Davies-Bouldin Score
```python
davies = davies_bouldin_score(feat_s, cluster_labels)
```
- **Range:** 0 to ∞
- **Interpretation:** Lower values indicate better clustering
- **Formula:** Average similarity measure between clusters

### Cluster Balance Metrics
- **Cluster distribution:** Number of files per cluster
- **Balance ratio:** Min cluster size / Max cluster size
- **Balance assessment:**
  - > 0.5: Good balance
  - 0.3-0.5: Fair balance
  - < 0.3: Poor balance

## Cluster Analysis Results

### Cluster Distribution
```python
cluster_counts = np.bincount(cluster_labels)
for cid in unique_clusters:
    count = cluster_counts[cid]
    percentage = (count / len(train_files)) * 100
    print(f"  Cluster {cid}: {count} files ({percentage:.1f}%)")
```

**Expected Results:**
- Cluster 0: ~50% of files
- Cluster 1: ~50% of files
- Balanced distribution for optimal model training

### Cluster Characteristics
For each cluster, the analysis provides:
- **Size:** Number of files assigned
- **Percentage:** Proportion of total dataset
- **Center coordinates:** Mean feature values
- **Dispersion:** Average distance to cluster center

## Feature Importance Analysis

### Feature Importance Calculation
```python
feature_importance = np.zeros(len(feature_names))
for i in range(len(feature_names)):
    overall_mean = np.mean(feat_s[:, i])
    between_cluster_var = 0
    within_cluster_var = 0
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_mean = np.mean(feat_s[cluster_mask, i])
        cluster_size = np.sum(cluster_mask)
        
        between_cluster_var += cluster_size * (cluster_mean - overall_mean)**2
        within_cluster_var += np.sum((feat_s[cluster_mask, i] - cluster_mean)**2)
    
    feature_importance[i] = between_cluster_var / (within_cluster_var + epsilon)
```

### Top Features Identification
```python
top_features = 10
top_indices = np.argsort(feature_importance)[-top_features:]
```

**Feature Importance Interpretation:**
- Higher values indicate features that better separate clusters
- Ratio of between-cluster to within-cluster variance
- Identifies most discriminative features

### Feature Categories Analysis
For each base column, importance is calculated for:
- **Mean importance:** How well the mean separates clusters
- **Std importance:** How well the variability separates clusters
- **Min importance:** How well the minimum values separate clusters
- **Max importance:** How well the maximum values separate clusters

## Visualization and Diagnostics

### 1. PCA Visualization
```python
pca = PCA(n_components=2)
feat_pca = pca.fit_transform(feat_s)
```

**Visualization Features:**
- 2D projection of high-dimensional feature space
- Color-coded cluster assignments
- Cluster centers marked with 'x'
- Explained variance ratios displayed

### 2. Feature Importance Plot
- Horizontal bar chart of top features
- Feature names on y-axis
- Importance scores on x-axis
- Sorted by importance (descending)

### 3. Cluster Distribution Plot
- Bar chart showing file counts per cluster
- Cluster IDs on x-axis
- File counts on y-axis
- Count labels on bars

### 4. Diagnostic Analysis
```python
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
```

## Cluster Separation Analysis

### Inter-cluster Distance Analysis
```python
for i in range(len(unique_clusters)):
    for j in range(i+1, len(unique_clusters)):
        cid1, cid2 = unique_clusters[i], unique_clusters[j]
        center1 = kmeans.cluster_centers_[cid1]
        center2 = kmeans.cluster_centers_[cid2]
        distance = np.linalg.norm(center1 - center2)
        print(f"  Cluster {cid1} ↔ Cluster {cid2}: {distance:.4f}")
```

### Intra-cluster Dispersion
```python
for cid in unique_clusters:
    cluster_mask = cluster_labels == cid
    cluster_data = feat_s[cluster_mask]
    cluster_center = kmeans.cluster_centers_[cid]
    
    distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    print(f"  Cluster {cid}:")
    print(f"    Size: {np.sum(cluster_mask)} files")
    print(f"    Avg distance to center: {avg_distance:.4f} ± {std_distance:.4f}")
```

## Summary and Recommendations

### Key Achievements
1. **Robust Feature Engineering:** Comprehensive statistical feature extraction
2. **Quality Assessment:** Multiple metrics for clustering evaluation
3. **Visualization:** Comprehensive diagnostic plots
4. **Feature Analysis:** Identification of discriminative features
5. **Reproducible Results:** Deterministic clustering with fixed seed

### Clustering Quality Assessment
Based on the implemented metrics:

**Silhouette Score Interpretation:**
- **Good (> 0.3):** Well-separated clusters
- **Fair (0.1-0.3):** Moderately separated clusters
- **Poor (< 0.1):** Poorly separated clusters

**Balance Assessment:**
- **Good (> 0.5):** Well-balanced cluster sizes
- **Fair (0.3-0.5):** Moderately balanced clusters
- **Poor (< 0.3):** Imbalanced clusters

### Feature Insights
The analysis reveals which features are most important for cluster separation:

**High-Importance Features:**
- Features with high between-cluster variance
- Low within-cluster variance
- Strong discriminative power

**Low-Importance Features:**
- Features with similar distributions across clusters
- High within-cluster variance
- Limited discriminative power

### Recommendations for Improvement
1. **Feature Selection:** Consider removing low-importance features
2. **Cluster Number Optimization:** Experiment with different numbers of clusters
3. **Domain Knowledge Integration:** Incorporate process-specific constraints
4. **Dynamic Clustering:** Consider adaptive clustering based on data characteristics

### Performance Metrics
- **Clustering Quality:** Measured by silhouette, Calinski-Harabasz, and Davies-Bouldin scores
- **Cluster Balance:** Ratio of smallest to largest cluster
- **Feature Discrimination:** Importance scores for feature selection
- **Computational Efficiency:** Scalable to large datasets

This clustering analysis provides the foundation for cluster-specific model training, enabling more accurate predictions by accounting for different operational regimes in the crystallization process. 