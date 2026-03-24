#!/usr/bin/env python3
"""
前50样本: Trident v2.2 vs 历史均值 对比
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v3.txt'

sys.path.insert(0, f'{BASE_DIR}/src')
from baseline_491_experiment import load_sample_data, split_data, historical_mean_baseline
from trident_v22 import trident_v22_baseline

# 读取v3样本列表
with open(SAMPLE_LIST_FILE, 'r') as f:
    sample_list = [int(line.strip()) for line in f if line.strip()]

test_samples = sample_list[:50]

print("="*80)
print("前50样本: Trident v2.2 vs 历史均值 对比 (v3.1清洗数据)")
print("="*80)
print(f"v2.2策略：")
print(f"  - 逐月检测7-9月维修突变")
print(f"  - 有维修：用9月值锚定")
print(f"  - 无维修：找最后维修，算劣化趋势")
print(f"  - 明显劣化→用趋势预测 | 无明显劣化→用上一年同月均值")
print("="*80)

results = []

print(f"\n{'#':<4} {'里程':<10} {'历史均值MAE':<12} {'v2.2 MAE':<12} {'差值':<10} {'胜者':<10} {'策略':<20}")
print("-" * 90)

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
        v22_result = trident_v22_baseline(train_df, test_df)
        v22_mae = v22_result['mae']
        strategy_a = v22_result['metadata'].get('strategy_a_count', 0)
        strategy_b = v22_result['metadata'].get('strategy_b_count', 0)
        strategy_str = f"趋势:{strategy_a}/均值:{strategy_b}"
    except Exception as e:
        v22_mae = None
        strategy_str = f"错误:{str(e)[:15]}"
    
    results.append({
        'mile': mile,
        'historical_mean_mae': hm_mae,
        'trident_v22_mae': v22_mae
    })
    
    if hm_mae is not None and v22_mae is not None:
        diff = v22_mae - hm_mae
        winner = '历史均值' if hm_mae < v22_mae else 'v2.2'
        print(f"{i:<4} {mile:<10} {hm_mae:<12.4f} {v22_mae:<12.4f} {diff:<+10.4f} {winner:<10} {strategy_str:<20}")
    else:
        print(f"{i:<4} {mile:<10} {hm_mae if hm_mae else 'ERR':<12} {v22_mae if v22_mae else 'ERR':<12} 计算错误")

# 统计
df_results = pd.DataFrame(results)
valid_results = df_results.dropna()

print(f"\n{'='*80}")
print("统计汇总")
print(f"{'='*80}")

print(f"\n有效样本数: {len(valid_results)}/50")

if len(valid_results) > 0:
    hm_wins = (valid_results['historical_mean_mae'] < valid_results['trident_v22_mae']).sum()
    v22_wins = (valid_results['historical_mean_mae'] > valid_results['trident_v22_mae']).sum()
    ties = (valid_results['historical_mean_mae'] == valid_results['trident_v22_mae']).sum()
    
    print(f"\n胜负统计:")
    print(f"  历史均值胜: {hm_wins} 个 ({hm_wins/len(valid_results)*100:.1f}%)")
    print(f"  v2.2胜: {v22_wins} 个 ({v22_wins/len(valid_results)*100:.1f}%)")
    print(f"  平局: {ties} 个")
    
    print(f"\nMAE统计:")
    print(f"  历史均值平均MAE: {valid_results['historical_mean_mae'].mean():.4f} ± {valid_results['historical_mean_mae'].std():.4f}")
    print(f"  v2.2平均MAE: {valid_results['trident_v22_mae'].mean():.4f} ± {valid_results['trident_v22_mae'].std():.4f}")
    
    # 大胜样本
    df_results['diff'] = df_results['trident_v22_mae'] - df_results['historical_mean_mae']
    
    print(f"\nv2.2大胜的样本 (历史均值MAE - v2.2MAE > 0.3):")
    v22_big_wins = df_results[df_results['diff'] < -0.3].sort_values('diff')
    if len(v22_big_wins) > 0:
        print(f"{'里程':<10} {'v2.2':<12} {'历史均值':<12} {'差距':<10}")
        for _, row in v22_big_wins.iterrows():
            gap = -row['diff']
            print(f"{int(row['mile']):<10} {row['trident_v22_mae']:<12.4f} {row['historical_mean_mae']:<12.4f} {gap:<+10.4f}")
    else:
        print("  无")
    
    print(f"\n历史均值大胜的样本 (v2.2MAE - 历史均值MAE > 0.3):")
    hm_big_wins = df_results[df_results['diff'] > 0.3].sort_values('diff', ascending=False)
    if len(hm_big_wins) > 0:
        print(f"{'里程':<10} {'历史均值':<12} {'v2.2':<12} {'差距':<10}")
        for _, row in hm_big_wins.iterrows():
            print(f"{int(row['mile']):<10} {row['historical_mean_mae']:<12.4f} {row['trident_v22_mae']:<12.4f} {row['diff']:<+10.4f}")
    else:
        print("  无")

print(f"{'='*80}")
