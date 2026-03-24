#!/usr/bin/env python3
"""
为39个合格样本生成时序图
保存到 sample_screening/figures/ 目录
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import os

# 创建输出目录
output_dir = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/sample_screening/figures'
os.makedirs(output_dir, exist_ok=True)

# 加载数据
qualified = pd.read_csv('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/sample_screening/qualified_samples.csv')
qualified = qualified.sort_values('best_mae').reset_index(drop=True)

# 加载原始数据
df = pd.read_excel('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/data/raw/iic_tqi_all.xlsx')
df.columns = df.columns.str.strip()

print(f"正在为 {len(qualified)} 个样本生成时序图...")
print("-" * 60)

for idx, row in qualified.iterrows():
    mile = int(row['tqi_mile'])
    clean_count = int(row['clean_count'])
    tqi_mean = row['tqi_mean']
    tqi_std = row['tqi_std']
    mae = row['best_mae']
    improvement = row['trident_improvement']
    
    # 加载样本数据
    sample_df = df[df['tqi_mile'] == mile].copy()
    sample_df['date'] = pd.to_datetime(sample_df['dete_dt'])
    sample_df = sample_df.sort_values('date').reset_index(drop=True)
    
    # 创建图
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 绘制TQI曲线
    ax.plot(sample_df['date'], sample_df['tqi_val'], 'b-', linewidth=0.8, alpha=0.7)
    
    # 标记大修点
    sample_df['tqi_diff'] = sample_df['tqi_val'].diff()
    maintenance = sample_df[sample_df['tqi_diff'] < -0.3]
    if len(maintenance) > 0:
        ax.scatter(maintenance['date'], maintenance['tqi_val'], color='r', s=15, zorder=5, label='Maintenance')
    
    # 标题和标签
    title = f'Sample {idx+1}: Mile {mile}\nn={clean_count}, TQI={tqi_mean:.2f}, σ={tqi_std:.3f}, MAE={mae:.3f}, Improve={improvement:.1f}%'
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_ylabel('TQI', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 设置Y轴范围
    y_min = max(0, sample_df['tqi_val'].min() - 0.5)
    y_max = sample_df['tqi_val'].max() * 1.1
    ax.set_ylim(y_min, y_max)
    
    if len(maintenance) > 0:
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # 保存
    output_path = f'{output_dir}/sample_{idx+1:02d}_mile_{mile}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[{idx+1:02d}/39] Mile {mile}: MAE={mae:.3f}, Improve={improvement:.1f}%")

print("-" * 60)
print(f"全部完成！保存到: {output_dir}")

# 生成索引文件
with open(f'{output_dir}/../figure_index.txt', 'w') as f:
    f.write("39个合格样本时序图索引\n")
    f.write("=" * 70 + "\n\n")
    f.write("排序依据: MAE (从小到大)\n\n")
    
    for idx, row in qualified.iterrows():
        mile = int(row['tqi_mile'])
        mae = row['best_mae']
        improvement = row['trident_improvement']
        tqi_mean = row['tqi_mean']
        f.write(f"[{idx+1:02d}] Mile {mile}: MAE={mae:.3f}, Improve={improvement:.1f}%, TQI={tqi_mean:.2f}\n")
        f.write(f"      File: figures/sample_{idx+1:02d}_mile_{mile}.png\n\n")

print(f"索引文件: {output_dir}/../figure_index.txt")
