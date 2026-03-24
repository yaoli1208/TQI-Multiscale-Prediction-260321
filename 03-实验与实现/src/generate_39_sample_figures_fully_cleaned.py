#!/usr/bin/env python3
"""
为39个合格样本进行完整数据清洗并生成时序图
完整清洗规则:
1. IQR异常值检测 (k=1.5)
2. 频率规则: 删除间隔<3天的过密检测
3. 标记间隔>45天的过疏检测
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import os

def full_cleaning(df):
    """完整清洗流程"""
    original_count = len(df)
    
    # 1. IQR异常值检测
    Q1 = df['tqi_val'].quantile(0.25)
    Q3 = df['tqi_val'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask_iqr = (df['tqi_val'] >= lower_bound) & (df['tqi_val'] <= upper_bound)
    df = df[mask_iqr].copy()
    removed_iqr = original_count - len(df)
    
    # 2. 频率规则 - 删除过密检测(<3天)
    df = df.sort_values('dete_dt').reset_index(drop=True)
    df['days_since_last'] = df['dete_dt'].diff().dt.days
    mask_dense = ~(df['days_since_last'] < 3)  # 保留不满足过密条件的
    df = df[mask_dense | (df['days_since_last'].isna())].copy()  # 保留第一条（NA）
    removed_dense = original_count - removed_iqr - len(df)
    
    # 3. 标记过疏检测(>45天)
    df['is_sparse'] = df['days_since_last'] > 45
    
    return df, removed_iqr, removed_dense

# 创建输出目录
output_dir = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/sample_screening/figures_fully_cleaned'
os.makedirs(output_dir, exist_ok=True)

# 加载合格样本
qualified = pd.read_csv('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/sample_screening/qualified_samples.csv')
qualified = qualified.sort_values('best_mae').reset_index(drop=True)

# 加载原始数据
df = pd.read_excel('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/data/raw/iic_tqi_all.xlsx')
df.columns = df.columns.str.strip()

print(f"正在为 {len(qualified)} 个样本进行完整数据清洗...")
print("-" * 70)

cleaning_stats = []

for idx, row in qualified.iterrows():
    mile = int(row['tqi_mile'])
    
    # 加载样本数据
    sample_df = df[df['tqi_mile'] == mile].copy()
    sample_df['dete_dt'] = pd.to_datetime(sample_df['dete_dt'])
    sample_df = sample_df.sort_values('dete_dt').reset_index(drop=True)
    
    original_count = len(sample_df)
    original_mean = sample_df['tqi_val'].mean()
    original_std = sample_df['tqi_val'].std()
    
    # 完整清洗
    sample_clean, removed_iqr, removed_dense = full_cleaning(sample_df)
    
    cleaned_count = len(sample_clean)
    total_removed = original_count - cleaned_count
    
    # 新统计量
    new_mean = sample_clean['tqi_val'].mean()
    new_std = sample_clean['tqi_val'].std()
    
    cleaning_stats.append({
        'mile': mile,
        'original_count': original_count,
        'cleaned_count': cleaned_count,
        'removed_iqr': removed_iqr,
        'removed_dense': removed_dense,
        'total_removed': total_removed,
        'original_mean': original_mean,
        'new_mean': new_mean,
        'original_std': original_std,
        'new_std': new_std
    })
    
    # 创建对比图
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 上图: 原始数据
    ax = axes[0]
    ax.plot(sample_df['dete_dt'], sample_df['tqi_val'], 'b-', linewidth=0.8, alpha=0.7, label='Original')
    # IQR异常值
    Q1 = sample_df['tqi_val'].quantile(0.25)
    Q3 = sample_df['tqi_val'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = sample_df[(sample_df['tqi_val'] < Q1-1.5*IQR) | (sample_df['tqi_val'] > Q3+1.5*IQR)]
    if len(outliers) > 0:
        ax.scatter(outliers['dete_dt'], outliers['tqi_val'], color='red', s=30, zorder=5, label=f'IQR outliers ({len(outliers)})')
    # 大修点
    sample_df['tqi_diff'] = sample_df['tqi_val'].diff()
    maintenance = sample_df[sample_df['tqi_diff'] < -0.3]
    if len(maintenance) > 0:
        ax.scatter(maintenance['dete_dt'], maintenance['tqi_val'], color='orange', s=15, zorder=4, label='Maintenance')
    
    title = f'Sample {idx+1}: Mile {mile} - ORIGINAL\nn={original_count}, TQI={original_mean:.2f}, σ={original_std:.3f}'
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('TQI', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 下图: 清洗后数据
    ax = axes[1]
    ax.plot(sample_clean['dete_dt'], sample_clean['tqi_val'], 'g-', linewidth=0.8, alpha=0.7, label='After cleaning')
    # 标记过疏检测
    sparse = sample_clean[sample_clean['is_sparse'] == True]
    if len(sparse) > 0:
        ax.scatter(sparse['dete_dt'], sparse['tqi_val'], color='purple', s=15, zorder=4, label=f'Sparse (>45 days, {len(sparse)})')
    # 大修点
    sample_clean['tqi_diff'] = sample_clean['tqi_val'].diff()
    maintenance_clean = sample_clean[sample_clean['tqi_diff'] < -0.3]
    if len(maintenance_clean) > 0:
        ax.scatter(maintenance_clean['dete_dt'], maintenance_clean['tqi_val'], color='orange', s=15, zorder=4, label='Maintenance')
    
    title = f'FULLY CLEANED - n={cleaned_count}, TQI={new_mean:.2f}, σ={new_std:.3f}\nRemoved: IQR={removed_iqr}, Dense={removed_dense}'
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('TQI', fontsize=9)
    ax.set_xlabel('Date', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = f'{output_dir}/sample_{idx+1:02d}_mile_{mile}_fully_cleaned.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[{idx+1:02d}/39] Mile {mile}: {original_count}→{cleaned_count} (IQR:{removed_iqr}, Dense:{removed_dense})")

print("-" * 70)
print(f"全部完成！保存到: {output_dir}")

# 保存统计
stats_df = pd.DataFrame(cleaning_stats)
stats_df.to_csv(f'{output_dir}/../full_cleaning_stats.csv', index=False)

# 生成报告
with open(f'{output_dir}/../full_cleaning_report.txt', 'w') as f:
    f.write("39个样本完整数据清洗报告\n")
    f.write("=" * 70 + "\n\n")
    f.write("清洗规则:\n")
    f.write("1. IQR异常值检测 (k=1.5)\n")
    f.write("2. 删除间隔<3天的过密检测\n")
    f.write("3. 标记间隔>45天的过疏检测\n\n")
    f.write(f"总计剔除: {stats_df['total_removed'].sum()} 个\n")
    f.write(f"  - IQR异常值: {stats_df['removed_iqr'].sum()} 个\n")
    f.write(f"  - 过密检测: {stats_df['removed_dense'].sum()} 个\n")
    f.write(f"平均每个样本剔除: {stats_df['total_removed'].mean():.1f} 个\n\n")
    f.write("详细统计:\n")
    f.write(stats_df.to_string(index=False))

print(f"统计文件: {output_dir}/../full_cleaning_stats.csv")
print(f"报告文件: {output_dir}/../full_cleaning_report.txt")
