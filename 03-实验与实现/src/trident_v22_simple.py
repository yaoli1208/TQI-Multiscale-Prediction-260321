#!/usr/bin/env python3
"""
Trident v2.2 - 简化版（修复bug）
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def trident_v22_baseline(train_df, test_df):
    """v2.2简化版"""
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_df['year'] = train_df['date'].dt.year
    train_df['month'] = train_df['date'].dt.month
    
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    test_df['year'] = test_df['date'].dt.year
    test_df['month'] = test_df['date'].dt.month
    
    # 步骤1: 检测7-9月是否发生维修
    summer_df = train_df[train_df['month'].isin([7, 8, 9])]
    yearly_summer = summer_df.groupby('year')['tqi_value'].mean()
    
    # 计算年比年变化
    yearly_change = yearly_summer.diff()
    change_std = yearly_change.dropna().std()
    
    # 检测维修（TQI显著下降）
    maintenance_years = []
    for year in yearly_change.dropna().index:
        if yearly_change[year] < -2 * change_std:
            maintenance_years.append(year)
    
    last_maint_year = max(maintenance_years) if maintenance_years else None
    
    # 步骤2: 计算劣化趋势（维修后数据）
    if last_maint_year:
        post_maint = yearly_summer[yearly_summer.index >= last_maint_year]
    else:
        post_maint = yearly_summer
    
    if len(post_maint) >= 2:
        years = post_maint.index.values
        values = post_maint.values
        # 简单线性拟合
        n = len(years)
        x_mean = years.mean()
        y_mean = values.mean()
        slope = ((years - x_mean) * (values - y_mean)).sum() / ((years - x_mean)**2).sum()
    else:
        slope = 0
    
    # 步骤3: 获取上一年各月均值
    last_train_year = train_df['year'].max()
    prev_year_data = train_df[train_df['year'] == last_train_year]
    monthly_avg = prev_year_data.groupby('month')['tqi_value'].mean().to_dict()
    
    # 步骤4: 预测
    predictions = []
    actuals = []
    
    for _, row in test_df.iterrows():
        month = row['month']
        year_gap = row['year'] - last_train_year
        
        # 基础值：上一年同月均值
        base = monthly_avg.get(month, train_df['tqi_value'].mean())
        
        # 加上劣化趋势
        pred = base + slope * year_gap
        predictions.append(pred)
        actuals.append(row['tqi_value'])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 安全检查
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    predictions = np.clip(predictions, max(0, train_mean - 5*train_std), train_mean + 5*train_std)
    
    mae = mean_absolute_error(actuals, predictions)
    
    return {
        'mae': mae,
        'rmse': np.sqrt(np.mean((actuals - predictions)**2)),
        'mape': np.mean(np.abs((actuals - predictions) / actuals)) * 100,
        'metadata': {
            'last_maint_year': last_maint_year,
            'slope': slope,
            'maintenance_years': maintenance_years
        }
    }
