#!/usr/bin/env python3
"""
快速补齐：历史均值、移动平均、Trident v1/v2 (前50样本)
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v2.txt'
RESULTS_DIR = f'{BASE_DIR}/results/baseline_50_remaining'

os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, f'{BASE_DIR}/src')
from baseline_491_experiment import (
    load_sample_data, split_data,
    historical_mean_baseline, moving_average_baseline, 
    trident_rolling_anchor
)
from trident_v2 import trident_v2_baseline

with open(SAMPLE_LIST_FILE, 'r') as f:
    sample_list = [int(float(line.strip())) for line in f if line.strip()]

test_samples = sample_list[:50]

methods = [
    ('historical_mean', historical_mean_baseline, '历史均值'),
    ('ma', moving_average_baseline, '移动平均'),
    ('trident_v1', trident_rolling_anchor, 'Trident v1'),
    ('trident_v2', trident_v2_baseline, 'Trident v2'),
]

results = []

print("补齐剩余方法 (历史均值、移动平均、v1、v2)...")
print(f"{'样本':<8} {'历史均值':<10} {'移动平均':<10} {'v1':<8} {'v2':<8}")
print("-" * 50)

for i, mile in enumerate(test_samples, 1):
    df = load_sample_data(mile)
    if df is None or len(df) < 50:
        continue
    
    train_df, val_df, test_df = split_data(df)
    result = {'tqi_mile': mile}
    
    vals = []
    for key, func, name in methods:
        try:
            r = func(train_df, test_df)
            result[f'{key}_mae'] = r['mae']
            vals.append(f"{r['mae']:.3f}")
        except Exception as e:
            result[f'{key}_mae'] = None
            vals.append("ERR")
    
    results.append(result)
    print(f"{mile:<8} {vals[0]:<10} {vals[1]:<10} {vals[2]:<8} {vals[3]:<8}")

# 保存
df_results = pd.DataFrame(results)
df_results.to_csv(f'{RESULTS_DIR}/remaining_methods.csv', index=False)

print(f"\n统计:")
print(f"{'方法':<20} {'平均MAE':<12} {'标准差':<12}")
print("-" * 50)
for key, _, name in methods:
    col = f'{key}_mae'
    if col in df_results.columns:
        valid = df_results[col].dropna()
        if len(valid) > 0:
            print(f"{name:<20} {valid.mean():<12.4f} {valid.std():<12.4f}")

print(f"\n结果保存: {RESULTS_DIR}/remaining_methods.csv")
