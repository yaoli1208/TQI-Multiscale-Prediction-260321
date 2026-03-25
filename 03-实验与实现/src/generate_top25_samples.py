#!/usr/bin/env python3
"""
生成v2.3_no_seasonal MAE前15%的测试样本列表
"""
import pandas as pd

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
RESULTS_FILE = f'{BASE_DIR}/results/v25_full_optimization_results.csv'
OUTPUT_FILE = f'{BASE_DIR}/data/processed/qualified_miles_top15_v23.txt'

# 读取结果数据
df = pd.read_csv(RESULTS_FILE)

print("="*70)
print("生成v2.3_no_seasonal MAE前15%测试样本")
print("="*70)
print(f"总样本数: {len(df)}")

# 按v23_no_seasonal MAE升序排序（MAE越小越好）
df_sorted = df.sort_values('v23_no_seasonal', ascending=True)

# 计算前15%的样本数
top15_count = int(len(df) * 0.15)
print(f"前15%样本数: {top15_count}")

# 取前15%
top15_df = df_sorted.head(top15_count)

# 显示统计信息
print(f"\n前15%样本MAE统计:")
print(f"  最小MAE: {top15_df['v23_no_seasonal'].min():.4f}")
print(f"  最大MAE: {top15_df['v23_no_seasonal'].max():.4f}")
print(f"  平均MAE: {top15_df['v23_no_seasonal'].mean():.4f}")
print(f"  中位数MAE: {top15_df['v23_no_seasonal'].median():.4f}")

# 保存样本列表
top15_miles = top15_df['mile'].astype(int).tolist()
with open(OUTPUT_FILE, 'w') as f:
    for mile in top15_miles:
        f.write(f"{mile}\n")

print(f"\n样本列表已保存: {OUTPUT_FILE}")

# 显示前10个样本详情
print(f"\n前10个最佳样本详情:")
print(f"{'排名':<6} {'里程':<12} {'v23_MAE':<12} {'历史均值MAE':<12} {'改善':<12}")
print("-" * 60)
for i, (_, row) in enumerate(top15_df.head(10).iterrows(), 1):
    improvement = row['历史均值'] - row['v23_no_seasonal']
    print(f"{i:<6} {int(row['mile']):<12} {row['v23_no_seasonal']:<12.4f} {row['历史均值']:<12.4f} {improvement:+.4f}")

print("\n" + "="*70)
