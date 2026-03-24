#!/usr/bin/env python3
"""
为12个选定样本运行完整实验流程
样本列表: 02, 03, 08, 10, 15, 16, 19, 27, 29, 30, 32, 38
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import os
from datetime import datetime

# 选定样本（按MAE排序的索引）
SELECTED_INDICES = [2, 3, 8, 10, 15, 16, 19, 27, 29, 30, 32, 38]  # 实际索引减1

def full_cleaning(df):
    """完整清洗流程"""
    # 1. IQR异常值检测
    Q1 = df['tqi_val'].quantile(0.25)
    Q3 = df['tqi_val'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask_iqr = (df['tqi_val'] >= lower_bound) & (df['tqi_val'] <= upper_bound)
    df = df[mask_iqr].copy()
    
    # 2. 删除过密检测(<3天)
    df = df.sort_values('dete_dt').reset_index(drop=True)
    df['days_since_last'] = df['dete_dt'].diff().dt.days
    mask_dense = ~(df['days_since_last'] < 3)
    df = df[mask_dense | (df['days_since_last'].isna())].copy()
    
    return df

def detect_maintenance_points(df):
    """检测大修点"""
    df['tqi_diff'] = df['tqi_val'].diff()
    return df[df['tqi_diff'] < -0.3].copy()

def rolling_anchor_prediction(train_df, test_df):
    """滚动锚定策略预测"""
    # 使用训练集最后12个月均值作为基准
    anchor = train_df['tqi_val'].tail(12).mean()
    
    predictions = []
    for _, row in test_df.iterrows():
        month = row['dete_dt'].month
        # 计算月度季节性调整
        seasonal = train_df[train_df['dete_dt'].dt.month == month]['tqi_val'].mean() - train_df['tqi_val'].mean()
        pred = anchor + seasonal
        predictions.append(pred)
    
    return np.array(predictions)

def data_driven_prediction(train_df, test_df):
    """数据驱动基线 - 历史均值"""
    mean_val = train_df['tqi_val'].mean()
    return np.full(len(test_df), mean_val)

def moving_average_prediction(train_df, test_df):
    """移动平均预测 - 最后12个月均值"""
    ma_val = train_df['tqi_val'].tail(12).mean()
    return np.full(len(test_df), ma_val)

def calculate_mae(y_true, y_pred):
    """计算MAE"""
    return np.mean(np.abs(y_true - y_pred))

# 创建输出目录
output_dir = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/selected_12_samples'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)

print("="*70)
print("12个选定样本完整实验")
print("="*70)

# 加载数据
qualified = pd.read_csv('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/sample_screening/qualified_samples.csv')
qualified = qualified.sort_values('best_mae').reset_index(drop=True)

df_raw = pd.read_excel('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/data/raw/iic_tqi_all.xlsx')
df_raw.columns = df_raw.columns.str.strip()

results = []

for idx in SELECTED_INDICES:
    row = qualified.iloc[idx-1]  # 索引减1
    mile = int(row['tqi_mile'])
    
    print(f"\n【样本 {idx}】Mile {mile}")
    print("-"*50)
    
    # 加载并清洗数据
    sample_df = df_raw[df_raw['tqi_mile'] == mile].copy()
    sample_df['dete_dt'] = pd.to_datetime(sample_df['dete_dt'])
    sample_df = sample_df.sort_values('dete_dt').reset_index(drop=True)
    
    # 完整清洗
    sample_clean = full_cleaning(sample_df)
    n_clean = len(sample_clean)
    
    print(f"  清洗后记录数: {n_clean}")
    
    if n_clean < 450:
        print(f"  ⚠️ 警告: 记录数不足450 ({n_clean})")
    
    # 划分训练/验证/测试集 (70/15/15)
    n = len(sample_clean)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = sample_clean.iloc[:train_end].copy()
    val_df = sample_clean.iloc[train_end:val_end].copy()
    test_df = sample_clean.iloc[val_end:].copy()
    
    print(f"  数据集划分: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 各方法预测
    y_test = test_df['tqi_val'].values
    
    # 1. 数据驱动基线
    pred_dd = data_driven_prediction(train_df, test_df)
    mae_dd = calculate_mae(y_test, pred_dd)
    
    # 2. 移动平均
    pred_ma = moving_average_prediction(train_df, test_df)
    mae_ma = calculate_mae(y_test, pred_ma)
    
    # 3. 滚动锚定 (Trident)
    pred_ra = rolling_anchor_prediction(train_df, test_df)
    mae_ra = calculate_mae(y_test, pred_ra)
    
    # 计算改善幅度
    improvement = (mae_dd - mae_ra) / mae_dd * 100
    
    print(f"  MAE - DD: {mae_dd:.3f}, MA: {mae_ma:.3f}, RA: {mae_ra:.3f}")
    print(f"  改善幅度: {improvement:.1f}%")
    
    # 保存结果
    results.append({
        'sample_idx': idx,
        'mile': mile,
        'n_clean': n_clean,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
        'tqi_mean': sample_clean['tqi_val'].mean(),
        'tqi_std': sample_clean['tqi_val'].std(),
        'mae_data_driven': mae_dd,
        'mae_moving_avg': mae_ma,
        'mae_rolling_anchor': mae_ra,
        'improvement': improvement
    })
    
    # 生成预测对比图
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_df['dete_dt'], y_test, 'ko-', markersize=4, label='Actual', zorder=3)
    ax.plot(test_df['dete_dt'], pred_dd, 'r--', alpha=0.7, label=f'Data Driven (MAE={mae_dd:.3f})')
    ax.plot(test_df['dete_dt'], pred_ma, 'g:', alpha=0.7, label=f'Moving Avg (MAE={mae_ma:.3f})')
    ax.plot(test_df['dete_dt'], pred_ra, 'b-', alpha=0.7, label=f'Rolling Anchor (MAE={mae_ra:.3f})')
    
    ax.set_title(f'Sample {idx} (Mile {mile}): Prediction Comparison\nImprovement: {improvement:.1f}%', fontsize=11)
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('TQI', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/sample_{idx:02d}_mile_{mile}_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()

# 保存总结果
results_df = pd.DataFrame(results)
results_df.to_csv(f'{output_dir}/experiment_results.csv', index=False)

print("\n" + "="*70)
print("实验完成！")
print("="*70)
print(f"\n结果保存到: {output_dir}/")
print(f"  - experiment_results.csv: 实验结果汇总")
print(f"  - figures/: 预测对比图")

print("\n【12个样本实验结果汇总】")
print(results_df[['sample_idx', 'mile', 'n_clean', 'mae_data_driven', 'mae_rolling_anchor', 'improvement']].to_string(index=False))

# 统计
print(f"\n【统计摘要】")
print(f"平均MAE (Data Driven): {results_df['mae_data_driven'].mean():.3f}")
print(f"平均MAE (Rolling Anchor): {results_df['mae_rolling_anchor'].mean():.3f}")
print(f"平均改善幅度: {results_df['improvement'].mean():.1f}%")
print(f"改善>50%的样本数: {(results_df['improvement'] > 50).sum()} / 12")
