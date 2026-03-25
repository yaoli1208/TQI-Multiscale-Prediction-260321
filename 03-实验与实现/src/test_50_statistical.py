#!/usr/bin/env python3
"""
前50样本基线对比 - 统计方法版 (跳过深度学习)
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 配置路径
BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v2.txt'
RESULTS_DIR = f'{BASE_DIR}/results/baseline_50_statistical'

os.makedirs(RESULTS_DIR, exist_ok=True)

# 导入函数
sys.path.insert(0, f'{BASE_DIR}/src')
from baseline_491_experiment import (
    load_sample_data, split_data,
    historical_mean_baseline, moving_average_baseline, 
    holt_winters_baseline,
    trident_rolling_anchor
)
from trident_v2 import trident_v2_baseline
from trident_v21 import trident_v21_baseline

# 加载样本列表
with open(SAMPLE_LIST_FILE, 'r') as f:
    sample_list = [int(float(line.strip())) for line in f if line.strip()]

print(f"总样本数: {len(sample_list)}")
print(f"\n{'='*70}")
print(f"前50样本基线对比 - 统计方法版")
print(f"方法: 历史均值、移动平均、Holt-Winters、Trident v1/v2/v2.1")
print(f"跳过: TimeMixer、MLP、LSTM (已跑过或效率低)")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")

# 只取前50个样本
test_samples = sample_list[:50]

# 定义方法 (只包含统计方法)
methods = [
    ('historical_mean', historical_mean_baseline, '历史均值'),
    ('ma', moving_average_baseline, '移动平均'),
    ('holt_winters', holt_winters_baseline, 'Holt-Winters'),
    ('trident_v1', trident_rolling_anchor, 'Trident v1'),
    ('trident_v2', trident_v2_baseline, 'Trident v2'),
    ('trident_v21', trident_v21_baseline, 'Trident v2.1')
]

results = []
method_times = {k: [] for k, _, _ in methods}

print(f"\n{'样本':<8} {'历史均值':<10} {'移动平均':<10} {'Holt-W':<10} {'v1':<8} {'v2':<8} {'v2.1':<8} {'最佳':<10}")
print("="*75)

for i, mile in enumerate(test_samples, 1):
    df = load_sample_data(mile)
    if df is None or len(df) < 50:
        print(f"{mile:<8} 跳过:数据不足")
        continue
    
    train_df, val_df, test_df = split_data(df)
    result = {
        'tqi_mile': mile,
        'record_count': len(df),
        'train_count': len(train_df),
        'test_count': len(test_df)
    }
    
    maes = {}
    for key, func, name in methods:
        try:
            start = time.time()
            r = func(train_df, test_df)
            elapsed = time.time() - start
            result[f'{key}_mae'] = r['mae']
            maes[name] = r['mae']
            method_times[key].append(elapsed)
        except Exception as e:
            result[f'{key}_mae'] = None
            maes[name] = 999
    
    results.append(result)
    
    # 找出最佳方法
    valid_maes = {k: v for k, v in maes.items() if v < 900}
    if valid_maes:
        best_method = min(valid_maes, key=valid_maes.get)
        best_mae = valid_maes[best_method]
        
        print(f"{mile:<8} "
              f"{maes.get('历史均值', 0):<10.3f} "
              f"{maes.get('移动平均', 0):<10.3f} "
              f"{maes.get('Holt-Winters', 0):<10.3f} "
              f"{maes.get('Trident v1', 0):<8.3f} "
              f"{maes.get('Trident v2', 0):<8.3f} "
              f"{maes.get('Trident v2.1', 0):<8.3f} "
              f"{best_method}({best_mae:.3f})")

# 统计
print(f"\n{'='*70}")
print("统计结果")
print(f"{'='*70}")

df_results = pd.DataFrame(results)
df_results.to_csv(f'{RESULTS_DIR}/statistical_comparison.csv', index=False)

# 各方法平均MAE
print(f"\n各方法MAE统计:")
print(f"{'方法':<20} {'平均MAE':<12} {'标准差':<12} {'有效样本':<10}")
print("-" * 60)

for key, _, name in methods:
    col = f'{key}_mae'
    if col in df_results.columns:
        valid = df_results[col].dropna()
        if len(valid) > 0:
            print(f"{name:<20} {valid.mean():<12.4f} {valid.std():<12.4f} {len(valid):<10}")

# 各样本最佳方法统计
print(f"\n各样本最佳方法统计:")
best_method_counts = {}
for _, row in df_results.iterrows():
    mae_cols = [f'{k}_mae' for k, _, _ in methods if f'{k}_mae' in row]
    valid_maes = {col: row[col] for col in mae_cols if pd.notna(row[col])}
    if valid_maes:
        best_col = min(valid_maes, key=valid_maes.get)
        best_key = best_col.replace('_mae', '')
        best_method_counts[best_key] = best_method_counts.get(best_key, 0) + 1

print(f"{'方法':<20} {'最佳次数':<10} {'占比':<10}")
print("-" * 45)
for key, _, name in methods:
    count = best_method_counts.get(key, 0)
    pct = count / len(df_results) * 100 if len(df_results) > 0 else 0
    print(f"{name:<20} {count:<10} {pct:<10.1f}%")

# v2.1 vs 各方法对比
print(f"\nTrident v2.1 vs 其他方法 (胜/负/平):")
print(f"{'对比方法':<20} {'v2.1胜':<10} {'v2.1负':<10} {'平':<10}")
print("-" * 55)

v21_key = 'trident_v21'
for key, _, name in methods:
    if key != v21_key and f'{key}_mae' in df_results.columns:
        v21_col = f'{v21_key}_mae'
        other_col = f'{key}_mae'
        
        valid = df_results[[v21_col, other_col]].dropna()
        if len(valid) > 0:
            wins = (valid[v21_col] < valid[other_col]).sum()
            losses = (valid[v21_col] > valid[other_col]).sum()
            ties = (valid[v21_col] == valid[other_col]).sum()
            print(f"{name:<20} {wins:<10} {losses:<10} {ties:<10}")

# 平均运行时间
print(f"\n平均运行时间 (秒/样本):")
print(f"{'方法':<20} {'平均时间':<15}")
print("-" * 40)
for key, _, name in methods:
    times = method_times[key]
    if times:
        print(f"{name:<20} {np.mean(times):<15.4f}")

print(f"\n{'='*70}")
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"结果保存: {RESULTS_DIR}/statistical_comparison.csv")
print(f"{'='*70}")
