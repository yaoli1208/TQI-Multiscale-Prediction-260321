#!/usr/bin/env python3
"""
前50样本: Trident v2.1 vs 历史均值 快速对比
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

os.makedirs(f'{BASE_DIR}/results', exist_ok=True)

sys.path.insert(0, f'{BASE_DIR}/src')
from baseline_491_experiment import load_sample_data, split_data, historical_mean_baseline
from trident_v21 import trident_v21_baseline

# 读取v3样本列表
with open(SAMPLE_LIST_FILE, 'r') as f:
    sample_list = [int(line.strip()) for line in f if line.strip()]

test_samples = sample_list[:50]

print("="*80)
print("前50样本: Trident v2.1 vs 历史均值 对比 (v3.1清洗数据)")
print("="*80)
print(f"样本数: {len(test_samples)}")
print("="*80)

results = []

print(f"\n{'#':<4} {'里程':<10} {'历史均值MAE':<12} {'v2.1 MAE':<12} {'差值':<10} {'胜者':<10}")
print("-" * 80)

for i, mile in enumerate(test_samples, 1):
    df = load_sample_data(mile)
    if df is None or len(df) < 50:
        print(f"{i:<4} {mile:<10} 跳过(数据不足)")
        continue
    
    train_df, val_df, test_df = split_data(df)
    
    try:
        hm_result = historical_mean_baseline(train_df, test_df)
        hm_mae = hm_result['mae']
    except:
        hm_mae = None
    
    try:
        v21_result = trident_v21_baseline(train_df, test_df)
        v21_mae = v21_result['mae']
    except:
        v21_mae = None
    
    results.append({
        'mile': mile,
        'historical_mean_mae': hm_mae,
        'trident_v21_mae': v21_mae
    })
    
    if hm_mae is not None and v21_mae is not None:
        diff = v21_mae - hm_mae
        winner = '历史均值' if hm_mae < v21_mae else 'v2.1'
        print(f"{i:<4} {mile:<10} {hm_mae:<12.4f} {v21_mae:<12.4f} {diff:<+10.4f} {winner:<10}")
    else:
        print(f"{i:<4} {mile:<10} {hm_mae if hm_mae else 'ERR':<12} {v21_mae if v21_mae else 'ERR':<12} 计算错误")

# 统计
df_results = pd.DataFrame(results)
valid_results = df_results.dropna()

print(f"\n{'='*80}")
print("统计汇总")
print(f"{'='*80}")

print(f"\n有效样本数: {len(valid_results)}/50")

if len(valid_results) > 0:
    hm_wins = (valid_results['historical_mean_mae'] < valid_results['trident_v21_mae']).sum()
    v21_wins = (valid_results['historical_mean_mae'] > valid_results['trident_v21_mae']).sum()
    ties = (valid_results['historical_mean_mae'] == valid_results['trident_v21_mae']).sum()
    
    print(f"\n胜负统计:")
    print(f"  历史均值胜: {hm_wins} 个 ({hm_wins/len(valid_results)*100:.1f}%)")
    print(f"  v2.1胜: {v21_wins} 个 ({v21_wins/len(valid_results)*100:.1f}%)")
    print(f"  平局: {ties} 个")
    
    print(f"\nMAE统计:")
    print(f"  历史均值平均MAE: {valid_results['historical_mean_mae'].mean():.4f} ± {valid_results['historical_mean_mae'].std():.4f}")
    print(f"  v2.1平均MAE: {valid_results['trident_v21_mae'].mean():.4f} ± {valid_results['trident_v21_mae'].std():.4f}")
    
    hm_better = valid_results[valid_results['historical_mean_mae'] < valid_results['trident_v21_mae']]
    v21_better = valid_results[valid_results['historical_mean_mae'] > valid_results['trident_v21_mae']]
    
    print(f"\n历史均值大胜的样本 (差距 > 0.5):")
    hm_big_wins = hm_better[hm_better['trident_v21_mae'] - hm_better['historical_mean_mae'] > 0.5]
    if len(hm_big_wins) > 0:
        for _, row in hm_big_wins.iterrows():
            diff = row['trident_v21_mae'] - row['historical_mean_mae']
            print(f"  {int(row['mile'])}: 历史均值={row['historical_mean_mae']:.4f}, v2.1={row['trident_v21_mae']:.4f}, 差距={diff:.4f}")
    else:
        print("  无")
    
    print(f"\nv2.1大胜的样本 (差距 > 0.5):")
    v21_big_wins = v21_better[v21_better['historical_mean_mae'] - v21_better['trident_v21_mae'] > 0.5]
    if len(v21_big_wins) > 0:
        for _, row in v21_big_wins.iterrows():
            diff = row['historical_mean_mae'] - row['trident_v21_mae']
            print(f"  {int(row['mile'])}: v2.1={row['trident_v21_mae']:.4f}, 历史均值={row['historical_mean_mae']:.4f}, 差距={diff:.4f}")
    else:
        print("  无")
    
    # 保存结果
    df_results.to_csv(f'{BASE_DIR}/results/hm_vs_v21_top50.csv', index=False)
    print(f"\n结果保存: {BASE_DIR}/results/hm_vs_v21_top50.csv")

print(f"{'='*80}")
