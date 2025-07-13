#!/usr/bin/env python3
"""
Raw Data Analysis Script
Analyzes the original data files to understand preprocessing needs
"""
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Define columns
STATE_COLS = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
EXOG_COLS  = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
CLUST_COLS = STATE_COLS + EXOG_COLS

def read_txt(path: Path) -> pd.DataFrame:
    """Read TAB-separated text file into DataFrame."""
    return pd.read_csv(path, sep='\t', engine='python').apply(pd.to_numeric, errors='coerce')

def analyze_raw_data():
    """Comprehensive analysis of raw data files."""
    
    raw_dir = Path("Data/RAW DATA")
    train_dir = raw_dir / "train"
    
    # Get all training files
    train_files = list(train_dir.glob("*.txt"))
    print(f"📁 Found {len(train_files)} training files")
    
    # Analyze a sample of files
    sample_files = train_files[:10]  # First 10 files for detailed analysis
    
    # Collect statistics
    all_stats = []
    constant_vars = {}
    outlier_stats = {}
    
    for i, file_path in enumerate(sample_files):
        print(f"\n📊 Analyzing {file_path.name} ({i+1}/{len(sample_files)})")
        
        try:
            df = read_txt(file_path)
            
            # Basic stats
            file_stats = {
                'file': file_path.name,
                'rows': len(df),
                'columns': len(df.columns),
                'missing_total': df.isna().sum().sum(),
                'missing_pct': (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
            }
            
            # Column-wise analysis
            for col in CLUST_COLS:
                if col in df.columns:
                    col_data = df[col].dropna()
                    
                    if len(col_data) > 0:
                        # Check for constant values
                        unique_vals = col_data.nunique()
                        if unique_vals == 1:
                            if col not in constant_vars:
                                constant_vars[col] = []
                            constant_vars[col].append({
                                'file': file_path.name,
                                'value': col_data.iloc[0]
                            })
                        
                        # Outlier analysis using IQR
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                        outlier_pct = (outliers / len(col_data)) * 100
                        
                        if col not in outlier_stats:
                            outlier_stats[col] = []
                        
                        outlier_stats[col].append({
                            'file': file_path.name,
                            'outliers': outliers,
                            'outlier_pct': outlier_pct,
                            'mean': col_data.mean(),
                            'std': col_data.std(),
                            'min': col_data.min(),
                            'max': col_data.max(),
                            'Q1': Q1,
                            'Q3': Q3,
                            'IQR': IQR
                        })
            
            all_stats.append(file_stats)
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
    
    # Generate comprehensive report
    print("\n" + "="*80)
    print("📋 RAW DATA ANALYSIS REPORT")
    print("="*80)
    
    # 1. File Statistics Summary
    print(f"\n📊 File Statistics Summary:")
    if all_stats:
        avg_rows = np.mean([s['rows'] for s in all_stats])
        avg_missing = np.mean([s['missing_pct'] for s in all_stats])
        print(f"  Average rows per file: {avg_rows:.1f}")
        print(f"  Average missing data: {avg_missing:.2f}%")
        print(f"  Total files analyzed: {len(all_stats)}")
    
    # 2. Constant Variables Analysis
    print(f"\n🔍 Constant Variables Analysis:")
    for col, files in constant_vars.items():
        print(f"  {col}:")
        print(f"    Found in {len(files)} files")
        unique_values = set([f['value'] for f in files])
        print(f"    Unique values: {unique_values}")
        if len(unique_values) == 1:
            print(f"    ⚠️  CONSTANT VALUE: {list(unique_values)[0]}")
    
    # 3. Outlier Analysis Summary
    print(f"\n📈 Outlier Analysis Summary:")
    for col, stats in outlier_stats.items():
        if stats:
            avg_outlier_pct = np.mean([s['outlier_pct'] for s in stats])
            max_outlier_pct = np.max([s['outlier_pct'] for s in stats])
            print(f"  {col}:")
            print(f"    Average outlier percentage: {avg_outlier_pct:.2f}%")
            print(f"    Maximum outlier percentage: {max_outlier_pct:.2f}%")
            
            # Check for extreme outliers
            if avg_outlier_pct > 10:
                print(f"    ⚠️  HIGH OUTLIER RATE (>10%)")
            elif avg_outlier_pct > 5:
                print(f"    ⚠️  MODERATE OUTLIER RATE (>5%)")
    
    # 4. Scale Analysis
    print(f"\n📏 Scale Analysis:")
    for col, stats in outlier_stats.items():
        if stats:
            # Calculate typical ranges
            means = [s['mean'] for s in stats]
            stds = [s['std'] for s in stats]
            max_vals = [s['max'] for s in stats]
            min_vals = [s['min'] for s in stats]
            
            avg_mean = np.mean(means)
            avg_std = np.mean(stds)
            max_range = np.max(max_vals) - np.min(min_vals)
            
            print(f"  {col}:")
            print(f"    Average mean: {avg_mean:.2e}")
            print(f"    Average std: {avg_std:.2e}")
            print(f"    Total range: {max_range:.2e}")
            
            # Identify scale issues
            if avg_mean < 1e-6:
                print(f"    ⚠️  VERY SMALL SCALE (<1e-6)")
            elif avg_mean > 1e6:
                print(f"    ⚠️  VERY LARGE SCALE (>1e6)")
    
    # 5. Data Quality Issues
    print(f"\n🚨 Data Quality Issues Identified:")
    
    # Constant variables
    constant_cols = [col for col, files in constant_vars.items() if len(files) > 0]
    if constant_cols:
        print(f"  ❌ Constant variables (no predictive value): {constant_cols}")
    
    # High outlier rates
    high_outlier_cols = []
    for col, stats in outlier_stats.items():
        if stats:
            avg_outlier_pct = np.mean([s['outlier_pct'] for s in stats])
            if avg_outlier_pct > 5:
                high_outlier_cols.append((col, avg_outlier_pct))
    
    if high_outlier_cols:
        print(f"  ⚠️  High outlier rates:")
        for col, pct in high_outlier_cols:
            print(f"    {col}: {pct:.1f}%")
    
    # Scale mismatches
    scale_issues = []
    for col, stats in outlier_stats.items():
        if stats:
            means = [s['mean'] for s in stats]
            avg_mean = np.mean(means)
            if avg_mean < 1e-6 or avg_mean > 1e6:
                scale_issues.append((col, avg_mean))
    
    if scale_issues:
        print(f"  ⚠️  Scale issues:")
        for col, mean_val in scale_issues:
            print(f"    {col}: {mean_val:.2e}")
    
    # 6. Recommendations
    print(f"\n💡 Preprocessing Recommendations:")
    
    if constant_cols:
        print(f"  1. Remove constant variables: {constant_cols}")
        print(f"     - These provide no predictive information")
        print(f"     - Will reduce model complexity and improve training")
    
    if high_outlier_cols:
        print(f"  2. Implement adaptive outlier detection:")
        for col, pct in high_outlier_cols:
            print(f"     - {col}: Use more conservative IQR threshold (1.2 instead of 1.5)")
    
    if scale_issues:
        print(f"  3. Address scale issues:")
        for col, mean_val in scale_issues:
            if mean_val < 1e-6:
                print(f"     - {col}: Consider log transformation")
            elif mean_val > 1e6:
                print(f"     - {col}: Consider scaling or normalization")
    
    print(f"  4. Feature engineering opportunities:")
    print(f"     - Add PSD ratios (d90/d10, d50/d10, d90/d50)")
    print(f"     - Add temperature differences (T_PM - T_TM)")
    print(f"     - Add mass flow ratios")
    print(f"     - Add rate of change features for time series")
    
    # Save detailed report
    report = {
        'file_statistics': all_stats,
        'constant_variables': constant_vars,
        'outlier_statistics': outlier_stats,
        'recommendations': {
            'remove_constant': constant_cols,
            'high_outlier_cols': high_outlier_cols,
            'scale_issues': scale_issues
        }
    }
    
    with open('raw_data_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete! Detailed report saved to 'raw_data_analysis_report.json'")

if __name__ == "__main__":
    analyze_raw_data() 
# %%
