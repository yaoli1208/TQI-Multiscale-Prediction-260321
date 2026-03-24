#!/usr/bin/env python3
"""
前50样本完整基线对比 - 使用v3清洗数据
跳过LSTM(慢)和Holt-Winters(不稳定)
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v3.txt'
RESULTS_DIR = f'{BASE_DIR}/results/baseline_50_v3_cleaned'

os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, f'{BASE_DIR}/src')
from baseline_491_experiment import (
    load_sample_data, split_data,
    historical_mean_baseline, moving_average_baseline,
    timemixer_baseline, trident_rolling_anchor
)
from trident_v21 import trident_v21_baseline

# 读取v3样本列表
with open(SAMPLE_LIST_FILE, 'r') as f:
    sample_list = [int(line.strip()) for line in f if line.strip()]

test_samples = sample_list[:50]

print("="*70)
print("前50样本对比实验 (v3清洗数据)")
print("="*70)
print(f"样本数: {len(test_samples)}")
print(f"方法: 历史均值、移动平均、TimeMixer、Trident v1/v2.1")
print(f"跳过: LSTM(慢)、Holt-Winters(不稳定)")
print("="*70)

methods = [
    ('historical_mean', historical_mean_baseline, '历史均值'),
    ('ma', moving_average_baseline, '移动平均'),
    ('timemixer', timemixer_baseline, 'TimeMixer'),
    ('trident_v1', trident_rolling_anchor, 'Trident v1'),
    ('trident_v21', trident_v21_baseline, 'Trident v2.1'),
]

results = []

print(f"\n{'样本':<8} {'历史均值':<10} {'移动平均':<10} {'TimeMixer':<12} {'v1':<8} {'v2.1':<8} {'最佳':<12}")
print("-" * 75)

for i, mile in enumerate(test_samples, 1):
    print(f"\n[{i}/50] 样本 {mile}", end='', flush=True)
    
    df = load_sample_data(mile)
    if df is None or len(df) < 50:
        print(f" - 跳过(数据不足)")
        continue
    
    train_df, val_df, test_df = split_data(df)
    result = {'tqi_mile': mile}
    
    maes = {}
    for key, func, name in methods:
        try:
            r = func(train_df, test_df)
            result[f'{key}_mae'] = r['mae']
            maes[name] = r['mae']
            print(f" {name}={r['mae']:.3f}", end='', flush=True)
        except Exception as e:
            result[f'{key}_mae'] = None
            maes[name] = 999
            print(f" {name}=ERR", end='', flush=True)
    
    results.append(result)
    
    # 找出最佳方法
    valid_maes = {k: v for k, v in maes.items() if v < 900}
    if valid_maes:
        best_method = min(valid_maes, key=valid_maes.get)
        print(f" [最佳:{best_method}]")
    else:
        print(f" [无有效结果]")

# 统计
print(f"\n\n{'='*70}")
print("统计结果")
print(f"{'='*70}")

df_results = pd.DataFrame(results)
df_results.to_csv(f'{RESULTS_DIR}/comparison_v3.csv', index=False)

print(f"\n各方法MAE统计:")
print(f"{'方法':<20} {'平均MAE':<12} {'标准差':<12} {'有效样本':<10}")
print("-" * 60)

for key, _, name in methods:
    col = f'{key}_mae'
    if col in df_results.columns:
        valid = df_results[col].dropna()
        if len(valid) > 0:
            print(f"{name:<20} {valid.mean():<12.4f} {valid.std():<12.4f} {len(valid):<10}")

# 胜负统计
print(f"\nv2.1 vs 其他方法:")
print(f"{'对比方法':<20} {'v2.1胜':<10} {'v2.1负':<10} {'胜率':<10}")
print("-" * 55)

v21_col = 'trident_v21_mae'
for key, _, name in methods:
    if key != 'trident_v21' and f'{key}_mae' in df_results.columns:
        other_col = f'{key}_mae'
        valid = df_results[[v21_col, other_col]].dropna()
        if len(valid) > 0:
            wins = (valid[v21_col] < valid[other_col]).sum()
            losses = (valid[v21_col] > valid[other_col]).sum()
            win_rate = wins / (wins + losses) * 100
            print(f"{name:<20} {wins:<10} {losses:<10} {win_rate:<10.1f}%")

print(f"\n结果保存: {RESULTS_DIR}/comparison_v3.csv")
print(f"{'='*70}")
