#!/usr/bin/env python3
"""
Trident v2.2 - 精细化维修检测与预测策略

核心改进：
1. 维修检测：逐月检查7-9月是否发生突变（不只是年均值比较）
2. 发生维修：用9月值作为锚定（维修后的最新状态）
3. 未发生维修：
   - 找到最后一次维修时间
   - 计算年比年劣化趋势
   - 有明显劣化 → 用劣化趋势预测
   - 无明显劣化 → 直接用上一年各月均值预测

作者: Kimi Claw
日期: 2026-03-24
版本: v2.2
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def detect_monthly_maintenance(train_df, threshold_factor=2.0):
    """
    逐月检测7-9月是否发生维修（突变）
    
    逻辑：比较同月份年比年的变化，如果某年7/8/9月任一月突变，认为该年夏季有维修
    
    返回: {
        'maintenance_years': [发生维修的年份列表],
        'last_maintenance_year': 最后一次维修年份,
        'monthly_changes': DataFrame[year, month, value, change_from_prev]
    }
    """
    df = train_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # 只取7-9月数据
    summer_df = df[df['month'].isin([7, 8, 9])].copy()
    
    if len(summer_df) == 0:
        return {'maintenance_years': [], 'last_maintenance_year': None, 'monthly_changes': None}
    
    # 按年月聚合（取均值）
    monthly = summer_df.groupby(['year', 'month'])['tqi_value'].mean().reset_index()
    
    # 计算年比年变化
    monthly['change'] = np.nan
    monthly['is_maintenance'] = False
    
    for month in [7, 8, 9]:
        month_data = monthly[monthly['month'] == month].sort_values('year')
        if len(month_data) >= 2:
            month_data = month_data.copy()
            month_data['change'] = month_data['tqi_value'].diff()
            
            # 计算该月的历史变化标准差
            changes = month_data['change'].dropna()
            if len(changes) > 0:
                change_std = changes.std()
                if change_std > 0:
                    # 变化超过 -2σ 认为是维修（TQI显著下降）
                    month_data['is_maintenance'] = month_data['change'] < -threshold_factor * change_std
                else:
                    # 标准差为0，用绝对阈值
                    month_data['is_maintenance'] = month_data['change'] < -0.5
            
            # 更新回原表（使用values避免类型问题）
            monthly.loc[month_data.index, 'change'] = month_data['change'].values
            monthly.loc[month_data.index, 'is_maintenance'] = month_data['is_maintenance'].values
    
    # 按年汇总：如果7/8/9月任一月份发生维修，认为该年夏季有维修
    yearly_maintenance = monthly.groupby('year')['is_maintenance'].any().reset_index()
    maintenance_years = yearly_maintenance[yearly_maintenance['is_maintenance']]['year'].tolist()
    
    last_maintenance_year = max(maintenance_years) if maintenance_years else None
    
    return {
        'maintenance_years': maintenance_years,
        'last_maintenance_year': last_maintenance_year,
        'monthly_changes': monthly
    }


def calculate_yearly_deterioration(train_df, last_maintenance_year=None, min_years=2):
    """
    计算年比年劣化趋势
    
    从最后一次维修后，计算每年的平均劣化量
    
    返回: (annual_deterioration, is_reliable)
    """
    df = train_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    # 只取7-9月数据（最稳定的时期）
    summer_df = df[df['date'].dt.month.isin([7, 8, 9])]
    yearly_summer = summer_df.groupby('year')['tqi_value'].mean().reset_index()
    
    if last_maintenance_year is not None:
        # 只用维修后的数据
        yearly_summer = yearly_summer[yearly_summer['year'] >= last_maintenance_year]
    
    if len(yearly_summer) < min_years + 1:
        return 0.0, False  # 数据不足，假设无劣化
    
    # 计算年比年变化
    yearly_summer['change'] = yearly_summer['tqi_value'].diff()
    changes = yearly_summer['change'].dropna()
    
    if len(changes) == 0:
        return 0.0, False
    
    # 使用中位数作为劣化趋势（排除异常值）
    annual_deterioration = changes.median()
    
    # 判断是否有明显劣化（变化大于噪声）
    change_std = changes.std()
    is_reliable = abs(annual_deterioration) > 0.5 * change_std
    
    return annual_deterioration, is_reliable


def trident_v22_predict(train_df, test_df):
    """
    Trident v2.2 主预测函数
    
    策略：
    1. 检测7-9月是否发生维修
    2. 有维修 → 用9月值作为锚定
    3. 无维修 → 找到最后一次维修，计算劣化趋势
       - 有明显劣化 → 用劣化趋势指导预测
       - 无明显劣化 → 直接用上一年各月均值预测
    """
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # 步骤1: 检测维修
    maint_info = detect_monthly_maintenance(train_df)
    last_maint_year = maint_info['last_maintenance_year']
    
    # 步骤2: 计算劣化趋势
    annual_deterioration, is_reliable = calculate_yearly_deterioration(
        train_df, last_maintenance_year=last_maint_year
    )
    
    # 步骤3: 确定预测策略
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    last_train_year = df['year'].max()
    
    # 获取上一年各月均值（用于无劣化时的预测）
    prev_year_data = df[df['year'] == last_train_year]
    if len(prev_year_data) == 0:
        prev_year_data = df[df['year'] == df['year'].max()]
    
    monthly_avg_last_year = prev_year_data.groupby('month')['tqi_value'].mean()
    
    # 步骤4: 生成预测
    test_df['year'] = test_df['date'].dt.year
    test_df['month'] = test_df['date'].dt.month
    
    predictions = []
    actuals = []
    months_predicted = []
    strategy_used = []
    
    for _, row in test_df.iterrows():
        target_month = row['month']
        actual_value = row['tqi_value']
        target_year = row['year']
        
        # 计算与上一年的差距
        years_gap = target_year - last_train_year
        
        if is_reliable and abs(annual_deterioration) > 0.01:
            # 策略A: 有明显劣化，用劣化趋势
            base_value = monthly_avg_last_year.get(target_month, df['tqi_value'].mean())
            prediction = base_value + annual_deterioration * years_gap
            strategy = 'deterioration_trend'
        else:
            # 策略B: 无明显劣化，直接用上一年同月均值
            prediction = monthly_avg_last_year.get(target_month, df['tqi_value'].mean())
            strategy = 'last_year_same_month'
        
        predictions.append(prediction)
        actuals.append(actual_value)
        months_predicted.append(target_month)
        strategy_used.append(strategy)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 安全检查
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    metadata = {
        'last_maintenance_year': last_maint_year,
        'annual_deterioration': annual_deterioration,
        'is_reliable': is_reliable,
        'maintenance_years': maint_info['maintenance_years'],
        'strategy_a_count': strategy_used.count('deterioration_trend'),
        'strategy_b_count': strategy_used.count('last_year_same_month'),
        'num_predictions': len(predictions)
    }
    
    return predictions, actuals, metadata


def trident_v22_baseline(train_df, test_df):
    """v2.2基线包装函数"""
    predictions, actuals, metadata = trident_v22_predict(train_df, test_df)
    
    if predictions is None or len(predictions) == 0:
        return {'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'metadata': metadata}
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'metadata': metadata,
        'predictions': predictions,
        'actuals': actuals
    }


if __name__ == '__main__':
    # 简单测试
    print("Trident v2.2 加载成功")
    print("核心逻辑：精细化维修检测 + 劣化趋势/上年均值双策略")
