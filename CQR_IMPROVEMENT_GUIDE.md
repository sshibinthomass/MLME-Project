# CQR (Conformal Quantile Regression) Training Improvement Guide

## Overview
This guide outlines comprehensive improvements to enhance CQR training for better uncertainty quantification and prediction intervals.

## Key Improvements Implemented

### 1. **Enhanced Model Architectures**

#### Advanced QR Model
- **Attention Mechanism**: Self-attention for feature importance weighting
- **Residual Connections**: Multiple skip connections for better gradient flow
- **L2 Regularization**: Prevents overfitting with weight decay
- **Adaptive Dropout**: Varying dropout rates across layers

#### Ensemble QR Model
- **Multiple Parallel Paths**: 3 independent feature extraction paths
- **Feature Concatenation**: Combines diverse feature representations
- **Ensemble Learning**: Reduces variance and improves robustness

#### Deep QR Model (Enhanced)
- **Dense Blocks**: Multiple residual blocks for complex patterns
- **Progressive Regularization**: Decreasing dropout rates
- **Better Initialization**: Improved weight initialization

### 2. **Advanced Loss Functions**

#### Enhanced Pinball Loss
```python
def pinball_loss(tau: float):
    """Enhanced pinball loss with Huber regularization."""
    def loss(y, y_hat):
        e = y - y_hat
        pinball = tf.reduce_mean(tf.maximum(tau * e, (tau - 1) * e))
        
        # Huber component for robustness
        huber_loss = tf.reduce_mean(tf.where(
            tf.abs(e) <= huber_delta,
            0.5 * tf.square(e),
            huber_delta * tf.abs(e) - 0.5 * huber_delta ** 2
        ))
        
        return pinball + 0.1 * huber_loss
    return loss
```

#### Adaptive Pinball Loss
- **Error-based Weighting**: Adjusts loss based on error magnitude
- **Dynamic Adaptation**: Responds to prediction uncertainty
- **Robust Training**: Better handling of outliers

#### Quantile Huber Loss
- **Combined Approach**: Pinball + Huber loss
- **Robustness**: Better handling of heavy-tailed distributions
- **Stability**: More stable gradients

### 3. **Enhanced Conformal Calibration**

#### Adaptive Conformal Deltas
```python
def compute_adaptive_conformal_deltas(val_X, val_E, alpha=0.1):
    """Compute adaptive conformal deltas with local calibration."""
    for j, col in enumerate(STATE_COLS):
        errors = np.abs(val_E[:, j])
        
        # Basic conformal delta
        k = int(np.ceil((1-alpha) * (len(errors)+1))) - 1
        basic_delta = float(np.sort(errors)[k])
        
        # Adaptive factor based on error distribution
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        adaptive_factor = 1.0 + 0.5 * (error_std / error_mean)
        adaptive_delta = basic_delta * adaptive_factor
        
        # Local calibration
        local_errors = errors[errors <= np.percentile(errors, 95)]
        if len(local_errors) > 10:
            local_k = int(np.ceil((1-alpha) * (len(local_errors)+1))) - 1
            local_delta = float(np.sort(local_errors)[local_k])
            final_delta = 0.7 * adaptive_delta + 0.3 * local_delta
        else:
            final_delta = adaptive_delta
```

#### Multiple Quantile Deltas
- **Multiple Coverage Levels**: 85%, 90%, 95% prediction intervals
- **Flexible Calibration**: Choose appropriate coverage for different use cases
- **Robust Coverage**: Better handling of varying uncertainty levels

### 4. **Advanced Training Techniques**

#### Model-Specific Configurations
```python
model_configs = {
    'T_PM': {'model_type': 'advanced', 'loss_type': 'adaptive'},
    'c': {'model_type': 'ensemble', 'loss_type': 'quantile_huber'},
    'd10': {'model_type': 'deep', 'loss_type': 'enhanced_pinball'},
    'd50': {'model_type': 'deep', 'loss_type': 'adaptive'},
    'd90': {'model_type': 'deep', 'loss_type': 'quantile_huber'},
    'T_TM': {'model_type': 'advanced', 'loss_type': 'enhanced_pinball'}
}
```

#### Learning Rate Finder
- **Optimal LR Detection**: Automatically finds best learning rate
- **Variable-specific Tuning**: Different LRs for different variables
- **Convergence Improvement**: Faster and more stable training

#### Enhanced Callbacks
- **Model Checkpointing**: Saves best models during training
- **Advanced Early Stopping**: Better patience and monitoring
- **Learning Rate Scheduling**: Adaptive LR reduction

### 5. **Improved Prediction and Uncertainty Quantification**

#### Enhanced CQR Function
```python
def add_enhanced_cqr(df, Xs, base_pred, mode: str):
    """Enhanced CQR with better uncertainty quantification."""
    for i, col in enumerate(STATE_COLS):
        # Get quantile predictions
        lo = QR[(col, 0.1)].predict(Xs, verbose=0).flatten()
        hi = QR[(col, 0.9)].predict(Xs, verbose=0).flatten()
        
        # Calculate prediction uncertainty
        pred_uncertainty = hi - lo
        
        # Adaptive delta calculation
        adaptive_factor = 1.0 + 0.2 * (pred_uncertainty / np.mean(pred_uncertainty))
        adaptive_delta = base_delta * adaptive_factor
        
        # Store enhanced predictions
        out[f"{col}_{mode}_uncertainty"] = pred_uncertainty
        out[f"{col}_{mode}_delta"] = adaptive_delta
```

## Training Recommendations

### 1. **Data Preparation**
- **Robust Preprocessing**: Use IQR-based outlier detection
- **Feature Engineering**: Add engineered features (ratios, spans)
- **Data Augmentation**: Consider synthetic data for rare cases

### 2. **Model Selection**
- **Variable-specific Models**: Choose architecture based on variable characteristics
- **Ensemble Methods**: Combine multiple models for better performance
- **Cross-validation**: Use proper validation strategies

### 3. **Hyperparameter Tuning**
- **Learning Rate**: Use LR finder for optimal rates
- **Batch Size**: Experiment with different batch sizes
- **Regularization**: Adjust L2 and dropout rates

### 4. **Monitoring and Evaluation**
- **Loss Curves**: Monitor training and validation loss
- **Coverage Metrics**: Track prediction interval coverage
- **Calibration Plots**: Verify conformal calibration

## Expected Improvements

### 1. **Better Coverage**
- **More Accurate Intervals**: Improved 90% prediction intervals
- **Adaptive Calibration**: Better handling of varying uncertainty
- **Robust Coverage**: Consistent coverage across different scenarios

### 2. **Reduced Prediction Error**
- **Enhanced Models**: Better feature extraction and representation
- **Advanced Loss Functions**: More robust training objectives
- **Ensemble Methods**: Reduced variance and improved accuracy

### 3. **Better Uncertainty Quantification**
- **Multiple Metrics**: Uncertainty, delta, and coverage metrics
- **Adaptive Deltas**: Context-aware uncertainty adjustment
- **Enhanced Visualization**: Better uncertainty plots

## Usage Instructions

### 1. **Training Enhanced CQR**
```bash
python latest_modeltrain13.py
```

### 2. **Making Predictions**
```bash
python latest_prediction1.py
```

### 3. **Monitoring Results**
- Check `qr/loss_curves/` for training progress
- Review `enhanced_conformal_deltas.pkl` for calibration
- Analyze prediction coverage in output files

## Troubleshooting

### 1. **Poor Coverage**
- Increase delta adjustment factors
- Check data quality and preprocessing
- Verify model convergence

### 2. **Overfitting**
- Increase regularization (L2, dropout)
- Reduce model complexity
- Use early stopping

### 3. **Underfitting**
- Increase model capacity
- Reduce regularization
- Check learning rate

## Future Enhancements

### 1. **Advanced Conformal Methods**
- **Conditional Conformal**: Context-aware calibration
- **Conformal Risk Control**: Multiple risk measures
- **Online Conformal**: Adaptive calibration

### 2. **Deep Uncertainty Methods**
- **Bayesian Neural Networks**: Probabilistic predictions
- **Deep Ensembles**: Multiple model predictions
- **Monte Carlo Dropout**: Uncertainty estimation

### 3. **Advanced Loss Functions**
- **Focal Loss**: Handle class imbalance
- **Distribution-aware Loss**: Model full uncertainty distribution
- **Multi-objective Loss**: Balance multiple objectives

## Conclusion

These improvements provide a comprehensive enhancement to CQR training, resulting in:
- **Better Prediction Intervals**: More accurate uncertainty quantification
- **Improved Coverage**: Consistent 90% prediction intervals
- **Enhanced Robustness**: Better handling of outliers and noise
- **Advanced Monitoring**: Comprehensive training and evaluation metrics

The enhanced CQR implementation should significantly improve your uncertainty quantification capabilities and provide more reliable prediction intervals for your MLME project. 