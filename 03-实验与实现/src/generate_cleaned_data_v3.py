#!/usr/bin/env python3
"""
生成清洗后的全量详细数据表
保存每个样本清洗后的详细记录（时间点 + TQI值）
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v3.txt'
OUTPUT_FILE = f'{BASE_DIR}/data/processed/cleaned_data_v3.csv'

print("="*70)
print("生成清洗后的全量详细数据表")
print("="*70)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Step 1: 加载原始数据
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
})

# Step 3: 数据类型转换
print("\n[Step 3] 数据类型转换...")
df['mile'] = pd.to_numeric(df['mile'], errors='coerce')
df['tqi_value'] = pd.to_numeric(df['tqi_value'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['mile', 'tqi_value', 'date'])
print(f"  有效记录数: {len(df)}")

# Step 4: 全局TQI上限
print("\n[Step 4] 全局TQI上限 (TQI > 9 删除)...")
high_tqi_count = (df['tqi_value'] > 9).sum()
df = df[df['tqi_value'] <= 9].copy()
print(f"  删除高TQI记录: {high_tqi_count} 条")
print(f"  剩余记录数: {len(df)}")

# Step 5: IQR异常值检测 - 按里程分组
print("\n[Step 5] IQR异常值检测 (每个样本内部)...")
initial_count = len(df)

all_data = []
for mile in df['mile'].unique():
    mile_data = df[df['mile'] == mile].copy()
    if len(mile_data) < 10:
        all_data.append(mile_data)
        continue
    
    Q1 = mile_data['tqi_value'].quantile(0.25)
    Q3 = mile_data['tqi_value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 3 * IQR)
    upper_bound = Q3 + 3 * IQR
    
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

# Step 8: 只保留合格样本
print("\n[Step 8] 筛选合格样本...")
with open(SAMPLE_LIST_FILE, 'r') as f:
    qualified_miles = [int(line.strip()) for line in f if line.strip()]
print(f"  合格样本数: {len(qualified_miles)}")

df_cleaned = df[df['mile'].isin(qualified_miles)].copy()
print(f"  清洗后记录数: {len(df_cleaned)}")

# Step 9: 保存清洗后的详细数据
print("\n[Step 9] 保存清洗后的详细数据...")
df_cleaned = df_cleaned[['mile', 'date', 'tqi_value']].copy()
df_cleaned = df_cleaned.sort_values(['mile', 'date']).reset_index(drop=True)
df_cleaned.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'='*70}")
print("完成!")
print(f"{'='*70}")
print(f"输出文件: {OUTPUT_FILE}")
print(f"总记录数: {len(df_cleaned)}")
print(f"样本数: {df_cleaned['mile'].nunique()}")
print(f"时间范围: {df_cleaned['date'].min()} ~ {df_cleaned['date'].max()}")
print(f"{'='*70}")
