#!/usr/bin/env python3
"""
数据清洗 - v3.0 (修复版)
增加: 全局TQI上限 + IQR异常值检测
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
OUTPUT_DIR = f'{BASE_DIR}/data/processed'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("TQI数据清洗 v3.0 (修复版)")
print("="*70)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Step 1: 加载数据
print("\n[Step 1] 加载原始数据...")
df = pd.read_excel(DATA_FILE)
print(f"  原始记录数: {len(df)}")

# Step 2: 列名标准化
print("\n[Step 2] 列名标准化...")
df.columns = df.columns.str.strip()
df = df.rename(columns={
    'dete_dt': 'date',
    'tqi_mile': 'mile',
    'tqi_val': 'tqi_value',
    'tqi_lprf': 'left_profile',
    'tqi_rprf': 'right_profile',
    'tqi_laln': 'left_alignment',
    'tqi_raln': 'right_alignment',
    'tqi_gage': 'gauge',
    'tqi_warp1': 'warp',
    'tqi_xlvl': 'level'
})

# Step 3: 数据类型转换
print("\n[Step 3] 数据类型转换...")
df['mile'] = pd.to_numeric(df['mile'], errors='coerce')
df['tqi_value'] = pd.to_numeric(df['tqi_value'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['mile', 'tqi_value', 'date'])
print(f"  有效记录数: {len(df)}")

# Step 4: 全局TQI上限 - 删除极端异常值 (v3.1: 上限改为9)
print("\n[Step 4] 全局TQI上限 (TQI > 9 删除)...")
high_tqi_count = (df['tqi_value'] > 9).sum()
df = df[df['tqi_value'] <= 9].copy()
print(f"  删除高TQI记录: {high_tqi_count} 条")
print(f"  剩余记录数: {len(df)}")

# Step 5: IQR异常值检测 - 按里程分组 (新增!)
print("\n[Step 5] IQR异常值检测 (每个样本内部)...")
initial_count = len(df)

# 对每个里程分别做IQR检测
all_data = []
for mile in df['mile'].unique():
    mile_data = df[df['mile'] == mile].copy()
    if len(mile_data) < 10:
        all_data.append(mile_data)
        continue
    
    Q1 = mile_data['tqi_value'].quantile(0.25)
    Q3 = mile_data['tqi_value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 3 * IQR)  # 下限不低于0
    upper_bound = Q3 + 3 * IQR
    
    # 标记异常值
    outliers = (mile_data['tqi_value'] < lower_bound) | (mile_data['tqi_value'] > upper_bound)
    mile_data = mile_data[~outliers].copy()
    all_data.append(mile_data)

df = pd.concat(all_data, ignore_index=True)
iqr_removed = initial_count - len(df)
print(f"  IQR删除异常值: {iqr_removed} 条")
print(f"  剩余记录数: {len(df)}")

# Step 6: 按里程+日期去重
print("\n[Step 6] 按里程+日期去重...")
df = df.groupby(['mile', 'date'])['tqi_value'].mean().reset_index()
print(f"  去重后记录数: {len(df)}")

# Step 7: 月度频率限制 (每月≤4条)
print("\n[Step 7] 月度频率限制 (每月≤4条)...")
df['year_month'] = df['date'].dt.to_period('M')
df['month_rank'] = df.groupby(['mile', 'year_month'])['date'].rank(method='first')
df = df[df['month_rank'] <= 4].copy()
print(f"  限制后记录数: {len(df)}")

# Step 8: 按里程分组统计
print("\n[Step 8] 按里程分组统计...")
mile_stats = df.groupby('mile').agg({
    'tqi_value': ['count', 'mean', 'std', 'min', 'max'],
    'date': ['min', 'max'],
    'year_month': 'nunique'
}).reset_index()

# 展平列名
mile_stats.columns = ['_'.join(col).strip('_') for col in mile_stats.columns.values]
mile_stats = mile_stats.rename(columns={
    'mile_': 'mile',
    'tqi_value_count': 'record_count',
    'tqi_value_mean': 'tqi_mean',
    'tqi_value_std': 'tqi_std',
    'tqi_value_min': 'tqi_min',
    'tqi_value_max': 'tqi_max',
    'date_min': 'date_min',
    'date_max': 'date_max',
    'year_month_nunique': 'unique_months'
})

# 计算时间跨度
mile_stats['date_min'] = pd.to_datetime(mile_stats['date_min'])
mile_stats['date_max'] = pd.to_datetime(mile_stats['date_max'])
mile_stats['time_span_days'] = (mile_stats['date_max'] - mile_stats['date_min']).dt.days
mile_stats['expected_months'] = (mile_stats['time_span_days'] / 30).astype(int) + 1
mile_stats['monthly_coverage'] = mile_stats['unique_months'] / mile_stats['expected_months']

print(f"  总里程数: {len(mile_stats)}")

# Step 9: 应用筛选条件
print("\n[Step 9] 应用筛选条件...")
print(f"  条件1: 记录数 ≥ 400")
qualified = mile_stats[mile_stats['record_count'] >= 400]
print(f"    通过后: {len(qualified)}")

print(f"  条件2: 平均TQI ≤ 6")
qualified = qualified[qualified['tqi_mean'] <= 6]
print(f"    通过后: {len(qualified)}")

print(f"  条件3: 平均TQI ≥ 1.5 (新增)")
qualified = qualified[qualified['tqi_mean'] >= 1.5]
print(f"    通过后: {len(qualified)}")

print(f"  条件4: 时间跨度 ≥ 10年 (3650天)")
qualified = qualified[qualified['time_span_days'] >= 3650]
print(f"    通过后: {len(qualified)}")

print(f"  条件5: 月度覆盖度 ≥ 95%")
qualified = qualified[qualified['monthly_coverage'] >= 0.95]
print(f"    通过后: {len(qualified)}")

# Step 10: 保存结果
print("\n[Step 10] 保存结果...")

# 保存合格样本列表
qualified_miles = qualified['mile'].astype(int).tolist()
with open(f'{OUTPUT_DIR}/qualified_miles_v3.txt', 'w') as f:
    for mile in qualified_miles:
        f.write(f"{mile}\n")

# 保存详细统计
qualified.to_csv(f'{OUTPUT_DIR}/qualified_samples_v3.csv', index=False)

print(f"\n{'='*70}")
print("清洗完成!")
print(f"{'='*70}")
print(f"合格样本数: {len(qualified)}")
print(f"输出文件:")
print(f"  - {OUTPUT_DIR}/qualified_miles_v3.txt")
print(f"  - {OUTPUT_DIR}/qualified_samples_v3.csv")

print(f"\n合格样本统计:")
print(f"  平均记录数: {qualified['record_count'].mean():.0f}")
print(f"  平均TQI: {qualified['tqi_mean'].mean():.2f}")
print(f"  TQI最大值范围: {qualified['tqi_max'].min():.2f} ~ {qualified['tqi_max'].max():.2f}")
print(f"  平均时间跨度: {qualified['time_span_days'].mean()/365:.1f} 年")

print(f"\n{'='*70}")
