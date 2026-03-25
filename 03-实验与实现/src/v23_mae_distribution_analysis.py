#!/usr/bin/env python3
"""
v2.3_no_seasonal MAE 分布统计分析
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
RESULTS_FILE = f'{BASE_DIR}/results/v25_full_optimization_results.csv'

# 加载数据
df = pd.read_csv(RESULTS_FILE)

print("="*80)
print("v2.3_no_seasonal MAE 分布统计分析")
print("="*80)
print(f"样本总数: {len(df)}")
print()

# 提取v23 MAE
v23_mae = df['v23_no_seasonal'].values
hm_mae = df['历史均值'].values

# ============ MAE 统计描述 ============
print("="*80)
print("1. MAE 基本统计")
print("="*80)

stats_desc = {
    '样本数': len(v23_mae),
    '平均值': np.mean(v23_mae),
    '中位数': np.median(v23_mae),
    '标准差': np.std(v23_mae),
    '最小值': np.min(v23_mae),
    '最大值': np.max(v23_mae),
    '25%分位数': np.percentile(v23_mae, 25),
    '75%分位数': np.percentile(v23_mae, 75),
    '90%分位数': np.percentile(v23_mae, 90),
    '95%分位数': np.percentile(v23_mae, 95),
}

for key, val in stats_desc.items():
    print(f"  {key:<12}: {val:.4f}")

# ============ MAE 分布区间 ============
print()
print("="*80)
print("2. MAE 分布区间统计")
print("="*80)

bins = [0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 10.0]
labels = ['0-0.3', '0.3-0.5', '0.5-0.8', '0.8-1.0', '1.0-1.5', '1.5-2.0', '>2.0']

counts, _ = np.histogram(v23_mae, bins=bins)
percentages = counts / len(v23_mae) * 100

print(f"{'区间':<10} {'数量':<8} {'占比':<8}")
print("-" * 30)
for i, (label, count, pct) in enumerate(zip(labels, counts, percentages)):
    print(f"{label:<10} {count:<8} {pct:<7.1f}%")

# ============ 大胜/大败样本分析 ============
print()
print("="*80)
print("3. 大胜/大败样本分析 (vs 历史均值)")
print("="*80)

# 计算改善幅度
improvement = hm_mae - v23_mae

# 大胜样本 (v23显著优于历史均值)
great_win = improvement > 0.3
win = improvement > 0
tie = improvement == 0
lose = improvement < 0
great_lose = improvement < -0.3

print(f"大胜 (改善>0.3):    {sum(great_win):>3} 样本 ({sum(great_win)/len(df)*100:.1f}%)")
print(f"小胜 (0<改善≤0.3):  {sum(win & ~great_win):>3} 样本 ({sum(win & ~great_win)/len(df)*100:.1f}%)")
print(f"平局 (改善=0):      {sum(tie):>3} 样本 ({sum(tie)/len(df)*100:.1f}%)")
print(f"小败 (-0.3≤改善<0): {sum(lose & ~great_lose):>3} 样本 ({sum(lose & ~great_lose)/len(df)*100:.1f}%)")
print(f"大败 (改善<-0.3):   {sum(great_lose):>3} 样本 ({sum(great_lose)/len(df)*100:.1f}%)")

# 大胜样本详情
print()
print("大胜样本 (改善>0.3) 详情:")
print(f"{'里程':<10} {'历史均值MAE':<12} {'v23_MAE':<12} {'改善':<12}")
print("-" * 50)
great_win_df = df[great_win].copy()
great_win_df['improvement'] = hm_mae[great_win] - v23_mae[great_win]
great_win_df = great_win_df.sort_values('improvement', ascending=False)
for _, row in great_win_df.head(15).iterrows():
    print(f"{int(row['mile']):<10} {row['历史均值']:<12.4f} {row['v23_no_seasonal']:<12.4f} {row['improvement']:<12.4f}")

# 大败样本详情
print()
print("大败样本 (改善<-0.3) 详情:")
print(f"{'里程':<10} {'历史均值MAE':<12} {'v23_MAE':<12} {'恶化':<12}")
print("-" * 50)
great_lose_df = df[great_lose].copy()
great_lose_df['worsen'] = v23_mae[great_lose] - hm_mae[great_lose]
great_lose_df = great_lose_df.sort_values('worsen', ascending=False)
for _, row in great_lose_df.head(10).iterrows():
    print(f"{int(row['mile']):<10} {row['历史均值']:<12.4f} {row['v23_no_seasonal']:<12.4f} {row['worsen']:<12.4f}")

# ============ 箱线图数据 ============
print()
print("="*80)
print("4. 箱线图统计")
print("="*80)

q1 = np.percentile(v23_mae, 25)
q2 = np.percentile(v23_mae, 50)
q3 = np.percentile(v23_mae, 75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print(f"Q1 (25%):     {q1:.4f}")
print(f"Q2 (中位数):  {q2:.4f}")
print(f"Q3 (75%):     {q3:.4f}")
print(f"IQR:          {iqr:.4f}")
print(f"下界:         {lower_fence:.4f}")
print(f"上界:         {upper_fence:.4f}")

outliers = v23_mae[(v23_mae < lower_fence) | (v23_mae > upper_fence)]
print(f"异常值数量:   {len(outliers)} ({len(outliers)/len(v23_mae)*100:.1f}%)")

# ============ 与历史均值对比分布 ============
print()
print("="*80)
print("5. MAE 对比分布 (v23 vs 历史均值)")
print("="*80)

print(f"{'统计量':<15} {'v23_no_seasonal':<15} {'历史均值':<15} {'差值':<15}")
print("-" * 60)
metrics = ['平均值', '中位数', '标准差', '最小值', '最大值']
v23_stats = [np.mean(v23_mae), np.median(v23_mae), np.std(v23_mae), np.min(v23_mae), np.max(v23_mae)]
hm_stats = [np.mean(hm_mae), np.median(hm_mae), np.std(hm_mae), np.min(hm_mae), np.max(hm_mae)]

for metric, v23, hm in zip(metrics, v23_stats, hm_stats):
    diff = v23 - hm
    print(f"{metric:<15} {v23:<15.4f} {hm:<15.4f} {diff:+.4f}")

print()
print("="*80)
print("分析完成!")
print("="*80)
