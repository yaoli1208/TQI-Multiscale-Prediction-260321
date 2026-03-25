#!/usr/bin/env python3
"""
TimeMixer 场景深度分析 - 使用正确的实验数据
基于 v21_vs_timemixer_50samples.csv 和 full_comparison_50_v3.csv
"""
import pandas as pd
import numpy as np
import os

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'

# 加载正确的 TimeMixer 数据（v21 vs timemixer 实验）
tm_v21_df = pd.read_csv(f'{BASE_DIR}/results/v21_vs_timemixer/v21_vs_timemixer_50samples.csv')

# 加载完整对比数据（包含历史均值、v2.3）
full_df = pd.read_csv(f'{BASE_DIR}/results/baseline_comparison_50_v3.csv')

print("="*100)
print("TimeMixer 场景深度分析 - 正确数据")
print("="*100)

# 合并数据集（以里程为键）
merged_df = pd.merge(
    full_df[['mile', 'historical_mean', 'moving_average', 'v21', 'v23', 'best']], 
    tm_v21_df[['tqi_mile', 'timemixer_mae']], 
    left_on='mile', 
    right_on='tqi_mile',
    how='inner'
)

# 重命名列以便统一
merged_df = merged_df.rename(columns={
    'timemixer_mae': 'timemixer'
})

n_samples = len(merged_df)
print(f"\n合并后有效样本数: {n_samples}")

# ============ 1. TimeMixer 整体表现 ============
print("\n" + "="*100)
print("1. TimeMixer 整体表现")
print("="*100)

print(f"\n各方法平均MAE:")
for method in ['historical_mean', 'moving_average', 'timemixer', 'v21', 'v23']:
    if method in merged_df.columns:
        mae = merged_df[method].mean()
        std = merged_df[method].std()
        print(f"  {method:20s}: {mae:.4f} ± {std:.4f}")

# ============ 2. TimeMixer 击败历史均值的情况 ============
print("\n" + "="*100)
print("2. TimeMixer vs 历史均值")
print("="*100)

tm_vs_hm = merged_df['timemixer'] - merged_df['historical_mean']
tm_wins = (merged_df['timemixer'] < merged_df['historical_mean']).sum()
tm_losses = (merged_df['timemixer'] > merged_df['historical_mean']).sum()
tm_ties = n_samples - tm_wins - tm_losses

print(f"\nTimeMixer 击败历史均值: {tm_wins}/{n_samples} ({tm_wins/n_samples*100:.1f}%)")
print(f"TimeMixer 输给历史均值: {tm_losses}/{n_samples} ({tm_losses/n_samples*100:.1f}%)")
print(f"平局: {tm_ties}/{n_samples}")

# TimeMixer 大胜/大败样本
big_wins_tm = merged_df[merged_df['historical_mean'] - merged_df['timemixer'] > 0.2]
big_losses_tm = merged_df[merged_df['timemixer'] - merged_df['historical_mean'] > 0.2]

print(f"\nTimeMixer 大胜样本 (>0.2): {len(big_wins_tm)} 个")
if len(big_wins_tm) > 0:
    print(f"  平均领先: {(big_wins_tm['historical_mean'] - big_wins_tm['timemixer']).mean():.4f}")
    print(f"  里程: {big_wins_tm['mile'].tolist()}")

print(f"\nTimeMixer 大败样本 (>0.2): {len(big_losses_tm)} 个")
if len(big_losses_tm) > 0:
    print(f"  平均落后: {(big_losses_tm['timemixer'] - big_losses_tm['historical_mean']).mean():.4f}")
    print(f"  里程: {big_losses_tm['mile'].tolist()}")

# ============ 3. 场景分析 ============
print("\n" + "="*100)
print("3. 场景分析 - TimeMixer 在什么情况下表现好？")
print("="*100)

# 按历史均值MAE分层
merged_df['hm_bin'] = pd.cut(merged_df['historical_mean'], 
                              bins=[0, 0.5, 0.8, 1.0, 1.5, 10], 
                              labels=['极低(0-0.5)', '低(0.5-0.8)', '中(0.8-1.0)', '高(1.0-1.5)', '极高(>1.5)'])

print("\n按历史均值MAE分层:")
print(f"{'分层':<15} {'样本数':<8} {'TimeMixer MAE':<15} {'历史均值 MAE':<15} {'TimeMixer胜率':<15}")
print("-" * 70)
for bin_name in merged_df['hm_bin'].cat.categories:
    bin_df = merged_df[merged_df['hm_bin'] == bin_name]
    if len(bin_df) == 0:
        continue
    tm_mae = bin_df['timemixer'].mean()
    hm_mae = bin_df['historical_mean'].mean()
    tm_win_rate = (bin_df['timemixer'] < bin_df['historical_mean']).mean() * 100
    print(f"{bin_name:<15} {len(bin_df):<8} {tm_mae:<15.4f} {hm_mae:<15.4f} {tm_win_rate:<14.1f}%")

# ============ 4. TimeMixer vs v2.1 ============
print("\n" + "="*100)
print("4. TimeMixer vs v2.1")
print("="*100)

v21_wins = (merged_df['v21'] < merged_df['timemixer']).sum()
tm_wins_vs_v21 = (merged_df['timemixer'] < merged_df['v21']).sum()

print(f"\nv2.1 击败 TimeMixer: {v21_wins}/{n_samples} ({v21_wins/n_samples*100:.1f}%)")
print(f"TimeMixer 击败 v2.1: {tm_wins_vs_v21}/{n_samples} ({tm_wins_vs_v21/n_samples*100:.1f}%)")

# TimeMixer 击败 v2.1 的样本特征
tm_better_v21 = merged_df[merged_df['timemixer'] < merged_df['v21']]
print(f"\nTimeMixer 击败 v2.1 的 {len(tm_better_v21)} 个样本特征:")
if len(tm_better_v21) > 0:
    print(f"  平均历史均值MAE: {tm_better_v21['historical_mean'].mean():.4f}")
    print(f"  平均 TimeMixer MAE: {tm_better_v21['timemixer'].mean():.4f}")
    print(f"  平均 v2.1 MAE: {tm_better_v21['v21'].mean():.4f}")
    print(f"  里程: {tm_better_v21['mile'].tolist()}")

# v2.1 击败 TimeMixer 的样本特征
v21_better = merged_df[merged_df['v21'] < merged_df['timemixer']]
print(f"\nv2.1 击败 TimeMixer 的 {len(v21_better)} 个样本特征:")
if len(v21_better) > 0:
    print(f"  平均历史均值MAE: {v21_better['historical_mean'].mean():.4f}")
    print(f"  平均 TimeMixer MAE: {v21_better['timemixer'].mean():.4f}")
    print(f"  平均 v2.1 MAE: {v21_better['v21'].mean():.4f}")

# ============ 5. TimeMixer vs v2.3 ============
print("\n" + "="*100)
print("5. TimeMixer vs v2.3")
print("="*100)

v23_wins_vs_tm = (merged_df['v23'] < merged_df['timemixer']).sum()
tm_wins_vs_v23 = (merged_df['timemixer'] < merged_df['v23']).sum()

print(f"\nv2.3 击败 TimeMixer: {v23_wins_vs_tm}/{n_samples} ({v23_wins_vs_tm/n_samples*100:.1f}%)")
print(f"TimeMixer 击败 v2.3: {tm_wins_vs_v23}/{n_samples} ({tm_wins_vs_v23/n_samples*100:.1f}%)")

# ============ 6. 融合建议 ============
print("\n" + "="*100)
print("6. 融合策略建议")
print("="*100)

# 找 TimeMixer 是唯一最优的样本（击败历史均值、v2.1、v2.3）
tm_best_unique = merged_df[
    (merged_df['timemixer'] < merged_df['historical_mean']) &
    (merged_df['timemixer'] < merged_df['v21']) &
    (merged_df['timemixer'] < merged_df['v23'])
]
print(f"\nTimeMixer 同时击败历史均值、v2.1、v2.3 的样本数: {len(tm_best_unique)}")

# 找 TimeMixer 和 v2.3 互补的样本
tm_good_v23_bad = merged_df[
    (merged_df['timemixer'] < merged_df['historical_mean']) &
    (merged_df['v23'] > merged_df['historical_mean'])
]
v23_good_tm_bad = merged_df[
    (merged_df['v23'] < merged_df['historical_mean']) &
    (merged_df['timemixer'] > merged_df['historical_mean'])
]

print(f"\nTimeMixer 强 + v2.3 弱的样本数: {len(tm_good_v23_bad)}")
if len(tm_good_v23_bad) > 0:
    print(f"  里程: {tm_good_v23_bad['mile'].tolist()}")

print(f"\nv2.3 强 + TimeMixer 弱的样本数: {len(v23_good_tm_bad)}")
if len(v23_good_tm_bad) > 0:
    print(f"  里程: {v23_good_tm_bad['mile'].tolist()}")

# ============ 7. 总结 ============
print("\n" + "="*100)
print("7. 总结与融合建议")
print("="*100)

print(f"\n【TimeMixer 优势场景】")
print(f"  - 击败历史均值的样本: {tm_wins} 个 ({tm_wins/n_samples*100:.1f}%)")
if len(big_wins_tm) > 0:
    print(f"  - 大胜样本特征: 历史均值MAE较低 ({big_wins_tm['historical_mean'].mean():.4f})")
    print(f"  - 大胜样本里程: {big_wins_tm['mile'].tolist()}")

print(f"\n【TimeMixer 劣势场景】")
print(f"  - 输给历史均值的样本: {tm_losses} 个 ({tm_losses/n_samples*100:.1f}%)")
if len(big_losses_tm) > 0:
    print(f"  - 大败样本特征: 历史均值MAE极高 ({big_losses_tm['historical_mean'].mean():.4f})")

print(f"\n【与 v2.3 的互补性】")
print(f"  - v2.3 击败 TimeMixer: {v23_wins_vs_tm} 次")
print(f"  - TimeMixer 击败 v2.3: {tm_wins_vs_v23} 次")
print(f"  - 互补样本数: {len(tm_good_v23_bad) + len(v23_good_tm_bad)}")

print(f"\n【融合建议】")
if len(tm_good_v23_bad) > 0 or len(v23_good_tm_bad) > 0:
    print(f"  ✅ TimeMixer 和 v2.3 有互补性，可以考虑融合")
    print(f"  - 策略: 在 v2.3 预测的样本上使用 v2.3，在 TimeMixer 强的样本上使用 TimeMixer")
    print(f"  - 需要设计一个选择机制（如基于训练集表现或分布偏移检测）")
else:
    print(f"  ⚠️ TimeMixer 和 v2.3 互补性不强，融合价值有限")

print("\n" + "="*100)
