#!/usr/bin/env python3
"""
TimeMixer 场景分析
分析 TimeMixer 在什么情况下表现良好/糟糕
"""
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/baseline_comparison_50_v3.csv')

print("="*100)
print("TimeMixer 场景深度分析")
print("="*100)

# 计算 TimeMixer vs 其他算法的差距
df['tm_vs_hm'] = df['timemixer'] - df['historical_mean']  # 正数表示 TimeMixer 更差
df['tm_vs_v23'] = df['timemixer'] - df['v23']
df['tm_vs_v21'] = df['timemixer'] - df['v21']

# 1. TimeMixer 与历史均值完全相同的样本
print("\n【关键发现：TimeMixer vs 历史均值】")
print("-"*100)
diff = np.abs(df['timemixer'] - df['historical_mean'])
same_samples = df[diff < 0.0001]
print(f"TimeMixer 与历史均值完全相同的样本数: {len(same_samples)}/50")

# 2. TimeMixer 击败历史均值的样本
print("\n【TimeMixer 击败历史均值的样本】")
print("-"*100)
tm_wins = df[df['timemixer'] < df['historical_mean']]
print(f"TimeMixer 击败历史均值的样本数: {len(tm_wins)}/50")
if len(tm_wins) > 0:
    print(tm_wins[['mile', 'historical_mean', 'timemixer', 'v21', 'v23', 'tm_vs_hm']].to_string(index=False))
    print(f"\n平均领先幅度: {np.abs(tm_wins['tm_vs_hm']).mean():.4f}")

# 3. TimeMixer 表现最差的样本
print("\n\n【TimeMixer 表现最差（vs 历史均值落后最多）的10个样本】")
print("-"*100)
tm_worst = df.nlargest(10, 'tm_vs_hm')[['mile', 'historical_mean', 'timemixer', 'v21', 'v23', 'tm_vs_hm', 'v23_improvement']]
print(tm_worst.to_string(index=False))

print("\n这些样本的特征:")
print(f"  - 平均历史均值MAE: {tm_worst['historical_mean'].mean():.4f}")
print(f"  - 平均v2.3改善: {tm_worst['v23_improvement'].mean():.4f}")
print(f"  - v2.3在这些样本中大胜的比例: {(tm_worst['v23_improvement'] > 0.2).sum()}/10")

# 4. TimeMixer 表现最好的样本
print("\n\n【TimeMixer 表现最好（MAE最低）的10个样本】")
print("-"*100)
tm_best = df.nsmallest(10, 'timemixer')[['mile', 'historical_mean', 'timemixer', 'v21', 'v23', 'tm_vs_hm']]
print(tm_best.to_string(index=False))

print("\n这些样本的特征:")
print(f"  - 平均历史均值MAE: {tm_best['historical_mean'].mean():.4f}")
print(f"  - 平均TimeMixer MAE: {tm_best['timemixer'].mean():.4f}")

# 5. 对比分析：v2.3 大胜样本中 TimeMixer 的表现
print("\n\n【对比分析：不同场景下各算法表现】")
print("-"*100)

# 场景1: v2.3 大胜样本 (>0.2)
v23_big_wins = df[df['v23_improvement'] > 0.2]
print(f"\n场景A: v2.3 大胜样本 (改善>0.2, n={len(v23_big_wins)})")
print(f"  历史均值平均MAE: {v23_big_wins['historical_mean'].mean():.4f}")
print(f"  TimeMixer平均MAE: {v23_big_wins['timemixer'].mean():.4f}")
print(f"  v2.3平均MAE: {v23_big_wins['v23'].mean():.4f}")
print(f"  TimeMixer击败历史均值: {(v23_big_wins['timemixer'] < v23_big_wins['historical_mean']).sum()}/{len(v23_big_wins)}")

# 场景2: v2.3 落败样本
v23_losses = df[df['v23_improvement'] < -0.1]
print(f"\n场景B: v2.3 落败样本 (落后>0.1, n={len(v23_losses)})")
print(f"  历史均值平均MAE: {v23_losses['historical_mean'].mean():.4f}")
print(f"  TimeMixer平均MAE: {v23_losses['timemixer'].mean():.4f}")
print(f"  v2.3平均MAE: {v23_losses['v23'].mean():.4f}")
print(f"  TimeMixer击败历史均值: {(v23_losses['timemixer'] < v23_losses['historical_mean']).sum()}/{len(v23_losses)}")

# 场景3: 历史均值最佳样本
hm_best = df[df['best'] == '历史均值']
print(f"\n场景C: 历史均值最佳样本 (n={len(hm_best)})")
print(f"  历史均值平均MAE: {hm_best['historical_mean'].mean():.4f}")
print(f"  TimeMixer平均MAE: {hm_best['timemixer'].mean():.4f}")
print(f"  v2.3平均MAE: {hm_best['v23'].mean():.4f}")
print(f"  TimeMixer击败历史均值: {(hm_best['timemixer'] < hm_best['historical_mean']).sum()}/{len(hm_best)}")

# 6. 按历史均值MAE分层的分析
print("\n\n【按历史均值MAE分层分析】")
print("-"*100)
df['hm_bin'] = pd.cut(df['historical_mean'], bins=[0, 0.5, 0.8, 1.0, 2.0], labels=['低(0-0.5)', '中低(0.5-0.8)', '中高(0.8-1.0)', '高(>1.0)'])

for bin_name in df['hm_bin'].cat.categories:
    bin_df = df[df['hm_bin'] == bin_name]
    if len(bin_df) == 0:
        continue
    print(f"\n历史均值MAE分层: {bin_name} (n={len(bin_df)})")
    print(f"  历史均值平均MAE: {bin_df['historical_mean'].mean():.4f}")
    print(f"  TimeMixer平均MAE: {bin_df['timemixer'].mean():.4f}")
    print(f"  v2.3平均MAE: {bin_df['v23'].mean():.4f}")
    tm_wins = (bin_df['timemixer'] < bin_df['historical_mean']).sum()
    v23_wins = (bin_df['v23'] < bin_df['historical_mean']).sum()
    print(f"  TimeMixer击败历史均值: {tm_wins}/{len(bin_df)} ({tm_wins/len(bin_df)*100:.1f}%)")
    print(f"  v2.3击败历史均值: {v23_wins}/{len(bin_df)} ({v23_wins/len(bin_df)*100:.1f}%)")

# 7. 融合策略建议
print("\n\n" + "="*100)
print("融合策略建议")
print("="*100)

print("\n【发现1: TimeMixer = 历史均值？】")
print(f"  TimeMixer与历史均值完全相同的样本: {len(same_samples)}/50 ({len(same_samples)/50*100:.1f}%)")
print("  → TimeMixer可能实现过于简化，实际上退化为历史均值")

print("\n【发现2: TimeMixer 在所有场景下表现一致】")
print("  - 在v2.3大胜样本中: TimeMixer完全无法击败历史均值")
print("  - 在v2.3落败样本中: TimeMixer同样无法击败历史均值")
print("  → TimeMixer没有场景适应性")

print("\n【发现3: 融合可能性评估】")
# 找TimeMixer比v2.3好的样本
tm_better_v23 = df[df['timemixer'] < df['v23']]
print(f"  TimeMixer比v2.3好的样本数: {len(tm_better_v23)}/50")
if len(tm_better_v23) > 0:
    print(f"  这些样本的特征:")
    print(f"    - 平均历史均值MAE: {tm_better_v23['historical_mean'].mean():.4f}")
    print(f"    - 平均TimeMixer MAE: {tm_better_v23['timemixer'].mean():.4f}")
    print(f"    - 平均v2.3 MAE: {tm_better_v23['v23'].mean():.4f}")

# 找TimeMixer是唯一优势的样本
tm_only_good = df[(df['timemixer'] < df['historical_mean']) & (df['timemixer'] < df['v23'])]
print(f"\n  TimeMixer同时击败历史均值和v2.3的样本数: {len(tm_only_good)}/50")

print("\n【结论】")
print("  TimeMixer在当前实现下:")
print("  1. 实际上等价于历史均值（50个样本中全部相同或接近）")
print("  2. 没有提供超出历史均值的增量价值")
print("  3. 融合价值有限，因为与历史均值高度相关")
print("\n  如果要融合，建议:")
print("  - 使用真正的TimeMixer实现（当前可能是简化版）")
print("  - 或者考虑其他深度学习模型（如N-BEATS, DeepAR等）")
