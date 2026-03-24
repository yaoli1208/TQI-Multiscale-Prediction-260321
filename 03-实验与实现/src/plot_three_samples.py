#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 三个候选样本
samples = [
    (1184400, "Sample 0 (1184400): Stable Baseline\nn=509, TQI=2.55, σ=0.50, MAE=0.127"),
    (1190400, "Sample 1 (1190400): Volatile Type\nn=508, TQI=2.24, σ=0.77, MAE=0.190, Improve=72%"),
    (1191400, "Sample 2 (1191400): Good Performance\nn=508, TQI=2.27, σ=0.51, MAE=0.157, Improve=57%"),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 加载原始数据
df = pd.read_excel('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/data/raw/iic_tqi_all.xlsx')
df.columns = df.columns.str.strip()

for idx, (mile, title) in enumerate(samples):
    sample_df = df[df['tqi_mile'] == mile].copy()
    sample_df['date'] = pd.to_datetime(sample_df['dete_dt'])
    sample_df = sample_df.sort_values('date').reset_index(drop=True)
    
    ax = axes[idx]
    ax.plot(sample_df['date'], sample_df['tqi_val'], 'b-', linewidth=0.8, alpha=0.7)
    
    # 检测并标记大修点
    sample_df['tqi_diff'] = sample_df['tqi_val'].diff()
    maintenance = sample_df[sample_df['tqi_diff'] < -0.3]
    if len(maintenance) > 0:
        ax.scatter(maintenance['date'], maintenance['tqi_val'], color='r', s=20, zorder=5, label='Maintenance')
    
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_ylabel('TQI', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, sample_df['tqi_val'].max() * 1.1)
    
    if idx == 0:
        ax.legend(loc='upper right')
    
    if idx == 2:
        ax.set_xlabel('Date', fontsize=10)

plt.tight_layout()
output_path = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/preview_three_samples.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved: {output_path}")
