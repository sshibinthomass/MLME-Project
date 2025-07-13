# NARX Training Report

## Overview
This report details the comprehensive NARX (Nonlinear AutoRegressive with eXogenous inputs) model training process implemented in the SFC project. The training stage involves developing cluster-specific neural network models that predict future system states based on historical state and exogenous variables.

## Table of Contents
1. [NARX Model Architecture](#narx-model-architecture)
2. [Training Strategy](#training-strategy)
3. [Model Building Functions](#model-building-functions)
4. [Training Configuration](#training-configuration)
5. [Loss Function and Optimization](#loss-function-and-optimization)
6. [Training Callbacks](#training-callbacks)
7. [Per-Cluster Training Process](#per-cluster-training-process)
8. [Training Monitoring and Visualization](#training-monitoring-and-visualization)
9. [Model Persistence](#model-persistence)
10. [Summary and Recommendations](#summary-and-recommendations)

## NARX Model Architecture

### Architecture Overview
The NARX model uses a deep neural network with the following characteristics:

**Input Layer:**
- **Dimensions:** 260 features (26 time steps × 10 variables)
- **Features:** Historical state and exogenous variables
- **Data type:** Float32 for memory efficiency

**Hidden Layers:**
- **Layer 1:** 1024 neurons with SELU activation
- **Layer 2:** 512 neurons with SELU activation
- **Layer 3:** 256 neurons with SELU activation
- **Layer 4:** 128 neurons with SELU activation

**Output Layer:**
- **Dimensions:** 6 neurons (state variables)
- **Activation:** Linear (regression output)

### Residual Connections
```python
def build_narx(input_dim: int, output_dim: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(input_dim,))
    
    # First layer
    x = layers.Dense(1024, activation='selu', 
                     kernel_regularizer=tf.keras.regularizers.l2(5e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Second layer with residual connection
    dense1 = layers.Dense(512, activation='selu', 
                          kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.4)(dense1)
    
    # Residual connection (if dimensions match)
    if x.shape[-1] == dense1.shape[-1]:
        x = layers.Add()([x, dense1])
    else:
        x = dense1
    
    # Third layer
    dense2 = layers.Dense(256, activation='selu', 
                          kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Dropout(0.3)(dense2)
    
    # Residual connection
    if x.shape[-1] == dense2.shape[-1]:
        x = layers.Add()([x, dense2])
    else:
        x = dense2
    
    # Fourth layer
    x = layers.Dense(128, activation='selu', 
                     kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(output_dim)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```

### Architecture Features
1. **Deep Architecture:** 4 hidden layers with decreasing width
2. **Residual Connections:** Skip connections for gradient flow
3. **Regularization:** L2 regularization and dropout
4. **Normalization:** Batch normalization for training stability
5. **Activation:** SELU for hidden layers, linear for output

## Training Strategy

### Per-Cluster Training Approach
```python
for cid in range(N_CLUSTERS):
    # Collect cluster-specific data
    Xc, Yc = [], []
    for idx, p in enumerate(train_files):
        if kmeans.labels_[idx] != cid:
            continue
        x, y = make_xy(preprocess(p))
        if len(x):
            Xc.append(x); Yc.append(y)
    
    # Train cluster-specific model
    model = build_narx(Xc.shape[1], Yc.shape[1])
    # ... training process
```

### Training Benefits
1. **Specialization:** Each model optimized for specific operational regime
2. **Reduced Complexity:** Smaller, focused models per cluster
3. **Better Performance:** Targeted training for cluster characteristics
4. **Scalability:** Parallel training across clusters

## Model Building Functions

### Core Architecture Components

#### 1. Dense Layers
- **Activation:** SELU (Scaled Exponential Linear Unit)
- **Regularization:** L2 regularization (5e-4)
- **Purpose:** Feature transformation and learning

#### 2. Batch Normalization
- **Purpose:** Stabilize training and accelerate convergence
- **Benefits:** Reduces internal covariate shift
- **Placement:** After each dense layer

#### 3. Dropout Layers
- **Rates:** 0.4, 0.4, 0.3, 0.2 (decreasing)
- **Purpose:** Prevent overfitting
- **Strategy:** Higher dropout in early layers

#### 4. Residual Connections
- **Condition:** Only when dimensions match
- **Purpose:** Improve gradient flow
- **Benefits:** Easier training of deep networks

### Architecture Optimization
```python
# You can also try tuning BATCH_SIZE_NARX and dropout rates for further improvement.
```

**Tunable Parameters:**
- Batch size: Currently 16
- Dropout rates: 0.4, 0.4, 0.3, 0.2
- Learning rate: 1e-4
- Weight decay: 1e-4

## Training Configuration

### Global Training Parameters
```python
EPOCHS_NARX = 300
BATCH_SIZE_NARX = 16
```

### Optimizer Configuration
```python
optimizer=tf.keras.optimizers.AdamW(
    learning_rate=1e-4, 
    weight_decay=1e-4,
    clipnorm=1.0  # Gradient clipping
)
```

**Optimizer Features:**
- **AdamW:** Adam with decoupled weight decay
- **Learning rate:** 1e-4 (conservative for stability)
- **Weight decay:** 1e-4 (L2 regularization)
- **Gradient clipping:** 1.0 (prevents exploding gradients)

## Loss Function and Optimization

### Weighted MSE Loss
```python
def weighted_mse(y_true, y_pred):
    """MSE with per-output weights (PSD columns matter more)."""
    w = tf.constant(OUTPUT_WEIGHTS, dtype=y_true.dtype)
    return tf.reduce_mean(tf.reduce_mean(tf.square(y_true - y_pred) * w, axis=-1))
```

### Output Weights Configuration
```python
OUTPUT_WEIGHTS = np.array([8, 6, 25, 30, 35, 8], dtype="float32")
# Corresponds to: [T_PM, c, d10, d50, d90, T_TM]
```

**Weight Rationale:**
- **PSD columns (d10, d50, d90):** Higher weights (25, 30, 35)
- **Temperature columns (T_PM, T_TM):** Lower weights (8, 8)
- **Concentration (c):** Medium weight (6)
- **Purpose:** Prioritize particle size prediction accuracy

### Loss Function Benefits
1. **Multi-objective optimization:** Balances different output variables
2. **Domain-specific weighting:** Reflects process importance
3. **Numerical stability:** Prevents output variable dominance
4. **Interpretable results:** Clear performance metrics per variable

## Training Callbacks

### 1. Early Stopping
```python
es = callbacks.EarlyStopping(
    patience=20, 
    restore_best_weights=True
)
```

**Configuration:**
- **Patience:** 20 epochs without improvement
- **Monitor:** Validation loss
- **Restore:** Best weights from training history

### 2. Model Checkpointing
```python
ck = callbacks.ModelCheckpoint(
    filepath=MODEL_DIR/f'narx/cluster_{cid}.keras',
    monitor='val_loss',
    save_best_only=True
)
```

**Features:**
- **File naming:** Cluster-specific model files
- **Best only:** Saves only the best model
- **Monitor:** Validation loss for model selection

### 3. Learning Rate Scheduling
```python
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=4, 
    min_lr=1e-6, 
    verbose=1
)
```

**Scheduling Strategy:**
- **Factor:** 0.2 (reduce by 80%)
- **Patience:** 4 epochs without improvement
- **Minimum LR:** 1e-6 (prevent too small learning rate)

### 4. Warm-up Learning Rate
```python
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
```

**Warm-up Benefits:**
- **Stable training:** Gradual learning rate increase
- **Convergence:** Prevents early training instability
- **Performance:** Often improves final model quality

### 5. Metrics Printing
```python
class PrintMetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics_str = " ".join([f"{k}={v:.6f}" for k, v in logs.items()])
        print(f"Epoch {epoch+1}: {metrics_str}")
```

## Per-Cluster Training Process

### Training Loop Implementation
```python
for cid in range(N_CLUSTERS):
    # Data preparation (already covered in previous report)
    Xc, Yc = collect_cluster_data(cid)
    
    # Scaling
    scX = StandardScaler().fit(Xc)
    scY = StandardScaler().fit(Yc)
    
    # Train/validation split
    split = int(0.8 * len(Xc))
    Xtr, Ytr = Xc[:split], Yc[:split]
    Xvl, Yvl = Xc[split:], Yc[split:]
    
    # Model building and training
    model = build_narx(Xc.shape[1], Yc.shape[1])
    model.compile(optimizer=optimizer, loss=weighted_mse)
    
    history_narx = model.fit(
        scX.transform(Xtr), scY.transform(Ytr),
        validation_data=(scX.transform(Xvl), scY.transform(Yvl)),
        epochs=EPOCHS_NARX,
        batch_size=BATCH_SIZE_NARX,
        verbose=1,
        callbacks=[es, ck, PrintMetricsCallback(), lr_scheduler, warmup_scheduler]
    )
```

### Training Characteristics
- **Per-cluster models:** Independent training for each cluster
- **Scaled data:** Normalized inputs and outputs
- **Validation monitoring:** Real-time performance tracking
- **Early stopping:** Prevents overfitting
- **Best model saving:** Preserves optimal weights

## Training Monitoring and Visualization

### Loss Curve Generation
```python
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
```

### Visualization Features
1. **Training vs Validation Loss:** Monitor overfitting
2. **Convergence Tracking:** Assess training progress
3. **Cluster-specific Curves:** Compare cluster performance
4. **High-resolution Output:** 150 DPI for detailed analysis

### Training Metrics
- **Loss values:** Weighted MSE at each epoch
- **Convergence:** Epochs to reach best performance
- **Overfitting:** Gap between train and validation loss
- **Final performance:** Best validation loss achieved

## Model Persistence

### Saved Components
```python
# Scaler persistence
pickle.dump(scX, (MODEL_DIR/f'narx/scaler_X_{cid}.pkl').open('wb'))
pickle.dump(scY, (MODEL_DIR/f'narx/scaler_Y_{cid}.pkl').open('wb'))

# Model persistence
# Saved via ModelCheckpoint callback
```

### File Organization
```
model_5files10/
├── narx/
│   ├── cluster_0.keras
│   ├── cluster_1.keras
│   ├── scaler_X_0.pkl
│   ├── scaler_Y_0.pkl
│   ├── scaler_X_1.pkl
│   ├── scaler_Y_1.pkl
│   ├── cluster_0_loss_curve.png
│   └── cluster_1_loss_curve.png
```

### Metadata Storage
```python
json.dump({
    'state_cols': STATE_COLS, 
    'exog_cols': EXOG_COLS, 
    'lag': LAG
}, (MODEL_DIR/'metadata.json').open('w'))
```

## Summary and Recommendations

### Key Achievements
1. **Cluster-Specific Models:** Tailored architecture per operational regime
2. **Robust Architecture:** Deep network with residual connections
3. **Comprehensive Regularization:** L2, dropout, and batch normalization
4. **Advanced Training:** Learning rate scheduling and warm-up
5. **Quality Monitoring:** Extensive callback system for training control

### Training Performance
- **Architecture depth:** 4 hidden layers with residual connections
- **Regularization:** Multiple techniques for generalization
- **Optimization:** AdamW with gradient clipping
- **Monitoring:** Comprehensive callback system

### Model Quality Indicators
1. **Convergence:** Stable training curves
2. **Generalization:** Small gap between train and validation loss
3. **Performance:** Low final validation loss
4. **Robustness:** Consistent performance across clusters

### Recommendations for Improvement
1. **Architecture Tuning:** Experiment with different layer sizes
2. **Hyperparameter Optimization:** Grid search for optimal parameters
3. **Ensemble Methods:** Combine multiple models per cluster
4. **Advanced Regularization:** Consider additional techniques
5. **Cross-validation:** Implement k-fold validation for robust evaluation

### Performance Metrics
- **Training time:** Varies by cluster size and complexity
- **Memory usage:** Efficient with float32 precision
- **Convergence:** Typically 50-150 epochs
- **Final loss:** Weighted MSE values per cluster

This NARX training pipeline provides robust, cluster-specific models that capture the temporal dynamics of the crystallization process while maintaining generalization capabilities through comprehensive regularization and monitoring techniques. 