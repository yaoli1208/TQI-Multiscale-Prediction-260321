#!/usr/bin/env python3
"""
为39个合格样本生成时序图（剔除异常值后）
保存到 sample_screening/figures_cleaned/ 目录
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import os

def detect_outliers_iqr(data, k=1.5):
    """使用IQR方法检测异常值"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return (data >= lower_bound) & (data <= upper_bound)

# 创建输出目录
output_dir = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/sample_screening/figures_cleaned'
os.makedirs(output_dir, exist_ok=True)

# 加载合格样本数据
qualified = pd.read_csv('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/sample_screening/qualified_samples.csv')
qualified = qualified.sort_values('best_mae').reset_index(drop=True)

# 加载原始数据
df = pd.read_excel('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/data/raw/iic_tqi_all.xlsx')
df.columns = df.columns.str.strip()

print(f"正在为 {len(qualified)} 个样本生成时序图（剔除异常值）...")
print("-" * 70)

cleaning_stats = []

for idx, row in qualified.iterrows():
    mile = int(row['tqi_mile'])
    
    # 加载样本数据
    sample_df = df[df['tqi_mile'] == mile].copy()
    sample_df['date'] = pd.to_datetime(sample_df['dete_dt'])
    sample_df = sample_df.sort_values('date').reset_index(drop=True)
    
    original_count = len(sample_df)
    original_mean = sample_df['tqi_val'].mean()
    original_std = sample_df['tqi_val'].std()
    
    # 剔除异常值（IQR方法，k=1.5）
    mask = detect_outliers_iqr(sample_df['tqi_val'], k=1.5)
    sample_clean = sample_df[mask].copy()
    
    cleaned_count = len(sample_clean)
    outliers_count = original_count - cleaned_count
    
    # 重新计算统计量
    new_mean = sample_clean['tqi_val'].mean()
    new_std = sample_clean['tqi_val'].std()
    
    cleaning_stats.append({
        'mile': mile,
        'original_count': original_count,
        'cleaned_count': cleaned_count,
        'outliers_removed': outliers_count,
        'original_mean': original_mean,
        'new_mean': new_mean,
        'original_std': original_std,
        'new_std': new_std
    })
    
    # 创建图
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 上图：原始数据
    ax = axes[0]
    ax.plot(sample_df['date'], sample_df['tqi_val'], 'b-', linewidth=0.8, alpha=0.7, label='Original')
    # 标记异常值
    outliers = sample_df[~mask]
    if len(outliers) > 0:
        ax.scatter(outliers['date'], outliers['tqi_val'], color='red', s=30, zorder=5, label=f'Outliers ({len(outliers)})')
    # 标记大修点
    sample_df['tqi_diff'] = sample_df['tqi_val'].diff()
    maintenance = sample_df[sample_df['tqi_diff'] < -0.3]
    if len(maintenance) > 0:
        ax.scatter(maintenance['date'], maintenance['tqi_val'], color='orange', s=15, zorder=4, label='Maintenance')
    ax.set_title(f'Sample {idx+1}: Mile {mile} - ORIGINAL\nn={original_count}, TQI={original_mean:.2f}, σ={original_std:.3f}', fontsize=10)
    ax.set_ylabel('TQI', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 下图：清洗后数据
    ax = axes[1]
    ax.plot(sample_clean['date'], sample_clean['tqi_val'], 'g-', linewidth=0.8, alpha=0.7, label='Cleaned')
    # 标记大修点
    sample_clean['tqi_diff'] = sample_clean['tqi_val'].diff()
    maintenance_clean = sample_clean[sample_clean['tqi_diff'] < -0.3]
    if len(maintenance_clean) > 0:
        ax.scatter(maintenance_clean['date'], maintenance_clean['tqi_val'], color='orange', s=15, zorder=4, label='Maintenance')
    ax.set_title(f'CLEANED - n={cleaned_count}, TQI={new_mean:.2f}, σ={new_std:.3f}, Removed: {outliers_count}', fontsize=10)
    ax.set_ylabel('TQI', fontsize=9)
    ax.set_xlabel('Date', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = f'{output_dir}/sample_{idx+1:02d}_mile_{mile}_cleaned.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[{idx+1:02d}/39] Mile {mile}: {original_count}→{cleaned_count} (剔除{outliers_count}个异常值)")

print("-" * 70)
print(f"全部完成！保存到: {output_dir}")

# 保存清洗统计
stats_df = pd.DataFrame(cleaning_stats)
stats_df.to_csv(f'{output_dir}/../cleaning_stats.csv', index=False)

# 生成报告
with open(f'{output_dir}/../cleaning_report.txt', 'w') as f:
    f.write("39个样本异常值剔除报告\n")
    f.write("=" * 70 + "\n\n")
    f.write("方法: IQR (k=1.5)\n")
    f.write(f"总计剔除异常值: {stats_df['outliers_removed'].sum()} 个\n")
    f.write(f"平均每个样本剔除: {stats_df['outliers_removed'].mean():.1f} 个\n\n")
    f.write("详细统计:\n")
    f.write(stats_df.to_string(index=False))

print(f"清洗统计: {output_dir}/../cleaning_stats.csv")
print(f"清洗报告: {output_dir}/../cleaning_report.txt")
