#!/usr/bin/env python3
"""
v2.1 vs TimeMixer 对比实验 (前50样本)
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
RESULTS_DIR = f'{BASE_DIR}/results/v21_vs_timemixer'

os.makedirs(RESULTS_DIR, exist_ok=True)

# 导入函数
sys.path.insert(0, f'{BASE_DIR}/src')
from baseline_491_experiment import (
    load_sample_data, split_data,
    timemixer_baseline
)
from trident_v21 import trident_v21_baseline

# 加载样本列表
with open(SAMPLE_LIST_FILE, 'r') as f:
    sample_list = [int(float(line.strip())) for line in f if line.strip()]

print(f"总样本数: {len(sample_list)}")
print(f"\n{'='*60}")
print(f"v2.1 vs TimeMixer 对比实验 (前50个样本)")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

# 只取前50个样本
test_samples = sample_list[:50]
print(f"\n测试样本: 前50个")

results = []

print(f"\n{'样本':>8} {'TimeMixer':>12} {'Trident v2.1':>14} {'差值':>10} {'更好':>8}")
print("="*60)

for i, mile in enumerate(test_samples, 1):
    df = load_sample_data(mile)
    if df is None or len(df) < 50:
        continue
    
    train_df, val_df, test_df = split_data(df)
    
    # TimeMixer
    try:
        tm_start = time.time()
        tm_result = timemixer_baseline(train_df, test_df)
        tm_mae = tm_result['mae']
        tm_time = time.time() - tm_start
    except Exception as e:
        print(f"[{i}/50] {mile}: TimeMixer错误 - {e}")
        tm_mae = None
        tm_time = None
    
    # Trident v2.1
    try:
        v21_start = time.time()
        v21_result = trident_v21_baseline(train_df, test_df)
        v21_mae = v21_result['mae']
        v21_time = time.time() - v21_start
    except Exception as e:
        print(f"[{i}/50] {mile}: v2.1错误 - {e}")
        v21_mae = None
        v21_time = None
    
    # 比较
    if tm_mae is not None and v21_mae is not None:
        diff = v21_mae - tm_mae
        better = "v2.1" if v21_mae < tm_mae else "TM"
        print(f"{mile:>8} {tm_mae:>12.4f} {v21_mae:>14.4f} {diff:>+10.4f} {better:>8}")
    else:
        print(f"{mile:>8} {'ERROR':>12} {'ERROR':>14} {'N/A':>10} {'N/A':>8}")
    
    results.append({
        'tqi_mile': mile,
        'timemixer_mae': tm_mae,
        'v21_mae': v21_mae,
        'timemixer_time': tm_time,
        'v21_time': v21_time
    })

# 统计
print(f"\n{'='*60}")
print("统计结果:")
print(f"{'='*60}")

df_results = pd.DataFrame(results)
df_valid = df_results.dropna()

if len(df_valid) > 0:
    print(f"有效样本数: {len(df_valid)}/50")
    print(f"\nMAE均值:")
    print(f"  TimeMixer:   {df_valid['timemixer_mae'].mean():.4f} ± {df_valid['timemixer_mae'].std():.4f}")
    print(f"  Trident v2.1: {df_valid['v21_mae'].mean():.4f} ± {df_valid['v21_mae'].std():.4f}")
    
    v21_better = (df_valid['v21_mae'] < df_valid['timemixer_mae']).sum()
    tm_better = (df_valid['timemixer_mae'] < df_valid['v21_mae']).sum()
    tie = len(df_valid) - v21_better - tm_better
    
    print(f"\nv2.1胜: {v21_better}  |  TimeMixer胜: {tm_better}  |  平: {tie}")
    
    # 保存结果
    output_file = f'{RESULTS_DIR}/v21_vs_timemixer_50samples.csv'
    df_valid.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")
else:
    print("无有效结果")

print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
