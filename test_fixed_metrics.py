#!/usr/bin/env python3
"""
Test script to verify the fixed metric formatting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from predict_and_test import NARXPredictor, read_txt

def test_fixed_metrics():
    """Test the fixed metric formatting."""
    
    # Initialize predictor
    model_dir = Path("model_5files7")
    predictor = NARXPredictor(model_dir)
    
    # Test on a sample file
    test_dir = Path("Beat-the-Felix")
    if test_dir.exists():
        test_files = list(test_dir.glob("*.txt"))
        if test_files:
            test_file = test_files[0]
            print(f"Testing on: {test_file}")
            
            # Read data
            df = read_txt(test_file)
            
            # Make predictions
            Y_pred, Y_true, cluster_id = predictor.predict(df, return_cluster_info=True)
            
            if len(Y_pred) > 0:
                print(f"\nCluster assigned: {cluster_id}")
                print(f"Number of predictions: {len(Y_pred)}")
                
                # Evaluate using the fixed method
                metrics = predictor.evaluate_predictions(Y_true, Y_pred)
                
                print(f"\n=== Fixed Metrics Output ===")
                print(f"Overall R²: {metrics.get('r2', 0):.3f}")
                print(f"Overall RMSE: {metrics.get('rmse', 0):.6f}")
                print(f"Overall MAE: {metrics.get('mae', 0):.6f}")
                
                # Print per-column metrics with proper formatting
                state_cols = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
                for col in state_cols:
                    r2_key = f'{col}_r2'
                    rmse_key = f'{col}_rmse'
                    mae_key = f'{col}_mae'
                    
                    if r2_key in metrics:
                        rmse_val = metrics[rmse_key]
                        mae_val = metrics[mae_key]
                        
                        # Use scientific notation for very small values
                        if rmse_val < 1e-6:
                            rmse_str = f"{rmse_val:.2e}"
                        else:
                            rmse_str = f"{rmse_val:.6f}"
                            
                        if mae_val < 1e-6:
                            mae_str = f"{mae_val:.2e}"
                        else:
                            mae_str = f"{mae_val:.6f}"
                            
                        print(f"  {col}: R² = {metrics[r2_key]:.3f}, RMSE = {rmse_str}, MAE = {mae_str}")
                
                print(f"\n=== Summary ===")
                print("The fix should now show actual RMSE and MAE values instead of 0.000")
                print("Very small values will be displayed in scientific notation (e.g., 1.23e-08)")
                
            else:
                print("No valid predictions made")
    else:
        print("Test directory not found")

if __name__ == "__main__":
    test_fixed_metrics() 