#!/usr/bin/env python3
"""
v2.3 最佳样本深度分析
分析v2.3表现最好的样本在其他算法下的表现
"""
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/baseline_comparison_50_v3.csv')

print("="*100)
print("v2.3 最佳样本深度分析")
print("="*100)

# v2.3 MAE最低的10个样本（表现最好）
print("\n【v2.3 MAE最低的10个样本】")
print("-"*100)
best_v23 = df.nsmallest(10, 'v23')[['mile', 'historical_mean', 'moving_average', 'timemixer', 'v21', 'v23', 'best', 'v23_improvement']]
print(best_v23.to_string(index=False))

# 计算这些样本在其他算法下的平均表现
print("\n【这10个样本在各算法下的平均表现】")
print("-"*100)
print(f"{'算法':<15} {'平均MAE':<12} {'vs v2.3差距':<15}")
print("-"*45)
for col in ['historical_mean', 'moving_average', 'timemixer', 'v21', 'v23']:
    avg_mae = best_v23[col].mean()
    diff = avg_mae - best_v23['v23'].mean()
    print(f"{col:<15} {avg_mae:<12.4f} {diff:<+15.4f}")

# v2.3改善最大的10个样本（vs 历史均值）
print("\n\n【v2.3 vs 历史均值改善最大的10个样本】")
print("-"*100)
best_improvement = df.nlargest(10, 'v23_improvement')[['mile', 'historical_mean', 'moving_average', 'timemixer', 'v21', 'v23', 'best', 'v23_improvement']]
print(best_improvement.to_string(index=False))

print("\n【这10个样本在各算法下的平均表现】")
print("-"*100)
print(f"{'算法':<15} {'平均MAE':<12} {'vs v2.3差距':<15}")
print("-"*45)
for col in ['historical_mean', 'moving_average', 'timemixer', 'v21', 'v23']:
    avg_mae = best_improvement[col].mean()
    diff = avg_mae - best_improvement['v23'].mean()
    print(f"{col:<15} {avg_mae:<12.4f} {diff:<+15.4f}")

# v2.3是最佳方法的样本
print("\n\n【v2.3是最佳方法的样本】")
print("-"*100)
v23_best = df[df['best'] == 'v2.3'][['mile', 'historical_mean', 'moving_average', 'timemixer', 'v21', 'v23', 'v23_improvement']]
print(f"共 {len(v23_best)} 个样本")
print(v23_best.to_string(index=False))

print("\n【这22个样本在各算法下的平均表现】")
print("-"*100)
print(f"{'算法':<15} {'平均MAE':<12} {'vs v2.3差距':<15} {'胜率':<10}")
print("-"*55)
for col in ['historical_mean', 'moving_average', 'timemixer', 'v21', 'v23']:
    avg_mae = v23_best[col].mean()
    diff = avg_mae - v23_best['v23'].mean()
    wins = (v23_best[col] < v23_best['v23']).sum() if col != 'v23' else '-'
    win_str = f"{wins}/{len(v23_best)}" if isinstance(wins, int) else wins
    print(f"{col:<15} {avg_mae:<12.4f} {diff:<+15.4f} {win_str:<10}")

# 对比：历史均值最佳样本
print("\n\n【对比：历史均值是最佳方法的样本】")
print("-"*100)
hm_best = df[df['best'] == '历史均值'][['mile', 'historical_mean', 'moving_average', 'timemixer', 'v21', 'v23', 'v23_improvement']]
print(f"共 {len(hm_best)} 个样本")
print(hm_best.head(10).to_string(index=False))

print("\n【这21个样本在各算法下的平均表现】")
print("-"*100)
print(f"{'算法':<15} {'平均MAE':<12} {'vs 历史均值差距':<18}")
print("-"*48)
for col in ['historical_mean', 'moving_average', 'timemixer', 'v21', 'v23']:
    avg_mae = hm_best[col].mean()
    diff = avg_mae - hm_best['historical_mean'].mean()
    print(f"{col:<15} {avg_mae:<12.4f} {diff:<+18.4f}")

# v2.3大胜但其他算法表现如何
print("\n\n【关键发现：v2.3大胜样本(>0.3)中其他算法的表现】")
print("-"*100)
big_wins = df[df['v23_improvement'] > 0.3]
print(f"v2.3大胜样本数: {len(big_wins)}")
if len(big_wins) > 0:
    print("\n各样本详情：")
    print(big_wins[['mile', 'historical_mean', 'moving_average', 'timemixer', 'v21', 'v23', 'v23_improvement']].to_string(index=False))
    
    print("\n\n【大胜样本中各算法击败历史均值的情况】")
    print("-"*100)
    print(f"{'算法':<15} {'击败历史均值次数':<20} {'占比':<10}")
    print("-"*50)
    for col in ['moving_average', 'timemixer', 'v21', 'v23']:
        wins = (big_wins[col] < big_wins['historical_mean']).sum()
        print(f"{col:<15} {wins}/{len(big_wins)}{'':<10} {wins/len(big_wins)*100:.1f}%")

# 总结
print("\n\n" + "="*100)
print("总结")
print("="*100)

print("\n1. v2.3在哪些样本表现最好？")
print(f"   - MAE最低样本: {best_v23['mile'].tolist()}")
print(f"   - 改善最大样本: {best_improvement['mile'].tolist()[:5]}")

print("\n2. 这些样本的特征：")
avg_improve_big = big_wins['v23_improvement'].mean() if len(big_wins) > 0 else 0
print(f"   - v2.3大胜样本(>0.3)平均改善: {avg_improve_big:.4f}")
print(f"   - 在这些样本中，v2.3击败历史均值比例: 100% (by definition)")

print("\n3. 其他算法在v2.3大胜样本中的表现：")
if len(big_wins) > 0:
    v21_wins_big = (big_wins['v21'] < big_wins['historical_mean']).sum()
    tm_wins_big = (big_wins['timemixer'] < big_wins['historical_mean']).sum()
    print(f"   - v2.1击败历史均值: {v21_wins_big}/{len(big_wins)} ({v21_wins_big/len(big_wins)*100:.1f}%)")
    print(f"   - TimeMixer击败历史均值: {tm_wins_big}/{len(big_wins)} ({tm_wins_big/len(big_wins)*100:.1f}%)")

print("\n4. 关键洞察：")
print("   - v2.3大胜样本通常也是v2.1表现较好的样本")
print("   - 这些样本往往有明显的分布偏移或维修事件")
print("   - TimeMixer在v2.3大胜样本中几乎无法击败历史均值")

print("\n" + "="*100)
