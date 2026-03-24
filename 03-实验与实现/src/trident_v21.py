#!/usr/bin/env python3
"""
Trident v2.1 - 7-9月基准期 + 月度预测框架

核心设计：
1. 7-9月是基准期，不做预测，只用于年比年比较
2. 检测7-9月中是否发生维修（突变识别）
3. 只预测10月-次年6月（9月后的月份）
4. 每月预测一个值，与月度均值比较

预测公式：
   月度预测 = (最近7-9月均值 + 年趋势 × 年数) × (目标月季节因子 / 7-9月平均季节因子)

作者: Kimi Claw
日期: 2026-03-24
"""
import pandas as pd
import numpy as np
from scipy import fft
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def detect_seasonality_fft(tqi_values, sample_rate=1):
    """使用FFT检测季节性周期"""
    tqi_values = np.array(tqi_values, dtype=np.float64)
    
    if len(tqi_values) < 24:
        return None, 0.0
    
    yf = fft.fft(tqi_values - np.mean(tqi_values))
    xf = fft.fftfreq(len(tqi_values), sample_rate)
    
    power = np.abs(yf)**2
    power[0] = 0
    
    peak_indices = np.argsort(power)[-3:]
    
    seasonal_period = None
    seasonal_strength = 0.0
    
    for idx in peak_indices:
        if power[idx] > np.mean(power) * 2:
            period = 1 / abs(xf[idx]) if xf[idx] != 0 else float('inf')
            if 10 <= period <= 14:
                seasonal_period = 12
                seasonal_strength = power[idx] / np.sum(power)
                break
    
    return seasonal_period, seasonal_strength


def calculate_seasonal_factors(train_df):
    """计算各月季节性因子"""
    train_df = train_df.copy()
    train_df['month'] = pd.to_datetime(train_df['date']).dt.month
    
    monthly_avg = train_df.groupby('month')['tqi_value'].mean()
    overall_avg = train_df['tqi_value'].mean()
    
    seasonal_factors = {}
    for month in range(1, 13):
        if month in monthly_avg.index:
            seasonal_factors[month] = monthly_avg[month] / overall_avg
        else:
            seasonal_factors[month] = 1.0
    
    # 7-9月平均季节因子
    summer_factors = [seasonal_factors.get(m, 1.0) for m in [7, 8, 9]]
    summer_factor_mean = np.mean(summer_factors)
    
    return seasonal_factors, summer_factor_mean


def extract_yearly_summer_stats(train_df):
    """
    提取每年7-9月统计数据
    
    返回: DataFrame [year, mean, std, min, max, count]
    """
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_df['month'] = train_df['date'].dt.month
    train_df['year'] = train_df['date'].dt.year
    
    # 筛选7-9月
    summer_df = train_df[train_df['month'].isin([7, 8, 9])]
    
    if len(summer_df) == 0:
        return None
    
    # 按年统计
    yearly_stats = summer_df.groupby('year').agg({
        'tqi_value': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()
    yearly_stats.columns = ['year', 'mean', 'std', 'min', 'max', 'count']
    
    return yearly_stats.sort_values('year')


def detect_maintenance(yearly_stats, threshold_std=2.0):
    """
    检测维修事件
    
    逻辑：如果某年7-9月均值比前一年下降超过阈值，认为发生了维修
    
    返回: list [(year, change_amount, is_maintenance)]
    """
    if yearly_stats is None or len(yearly_stats) < 2:
        return []
    
    maintenance_events = []
    
    for i in range(1, len(yearly_stats)):
        prev_year = yearly_stats.iloc[i-1]
        curr_year = yearly_stats.iloc[i]
        
        year = curr_year['year']
        change = curr_year['mean'] - prev_year['mean']
        
        # 如果TQI显著下降（变好），可能是维修
        # 阈值：变化超过历史标准差的2倍
        historical_std = yearly_stats['mean'].std()
        is_maintenance = (change < -threshold_std * historical_std)
        
        maintenance_events.append({
            'year': year,
            'change': change,
            'is_maintenance': is_maintenance,
            'prev_mean': prev_year['mean'],
            'curr_mean': curr_year['mean']
        })
    
    return maintenance_events


def find_last_maintenance(yearly_stats, threshold_std=2.0, recent_years=5):
    """
    找到最后一次维修年份（只考虑近期）
    
    参数:
        recent_years: 只考虑最近N年内的维修事件
    
    返回: (last_maintenance_year, has_maintenance) 或 (None, False)
    """
    if yearly_stats is None or len(yearly_stats) < 2:
        return None, False
    
    maint_events = detect_maintenance(yearly_stats, threshold_std)
    
    # 只考虑近期的维修
    latest_year = yearly_stats['year'].max()
    cutoff_year = latest_year - recent_years
    
    maint_years = [e['year'] for e in maint_events 
                   if e['is_maintenance'] and e['year'] >= cutoff_year]
    
    if len(maint_years) == 0:
        return None, False
    
    return max(maint_years), True


def calculate_annual_trend(yearly_stats, last_maintenance_year=None):
    """
    计算年均劣化趋势
    
    修复: 维修后是新基础设施，只用维修后的数据计算趋势
    
    参数:
        yearly_stats: 7-9月年度统计
        last_maintenance_year: 最后一次维修年份，如果None表示无维修
    """
    if yearly_stats is None or len(yearly_stats) < 2:
        return 0.0
    
    if last_maintenance_year is not None:
        # 维修后是新基础设施，只用维修后的数据
        post_maint = yearly_stats[yearly_stats['year'] >= last_maintenance_year]
        if len(post_maint) >= 2:
            # 有足够维修后数据，用维修后趋势
            years = post_maint['year'].values.astype(np.float64)
            values = post_maint['mean'].values.astype(np.float64)
            slope = np.polyfit(years, values, 1)[0]
            return slope, True  # True表示是维修后趋势
        else:
            # 维修后数据不足，假设趋势为0（新基础设施稳定期）
            return 0.0, True
    else:
        # 无维修，用全部历史数据
        years = yearly_stats['year'].values.astype(np.float64)
        values = yearly_stats['mean'].values.astype(np.float64)
        slope = np.polyfit(years, values, 1)[0]
        return slope, False


def aggregate_to_monthly(df):
    """
    将日数据聚合为月度数据
    
    返回: DataFrame [year, month, tqi_value_mean]
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    monthly = df.groupby(['year', 'month'])['tqi_value'].mean().reset_index()
    monthly.columns = ['year', 'month', 'tqi_value']
    
    return monthly


def trident_v21_predict(train_df, test_df):
    """
    Trident v2.1主预测函数
    
    只预测最后一年（测试集最后一个周期）的10月-次年6月
    7-9月作为基准期，不预测
    """
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    test_df['year'] = test_df['date'].dt.year
    test_df['month'] = test_df['date'].dt.month
    
    # 步骤1: 计算季节性因子
    seasonal_factors, summer_factor_mean = calculate_seasonal_factors(train_df)
    seasonal_period, seasonal_strength = detect_seasonality_fft(train_df['tqi_value'].values)
    
    # 步骤2: 提取7-9月年度统计
    yearly_summer = extract_yearly_summer_stats(train_df)
    
    if yearly_summer is None or len(yearly_summer) == 0:
        return None, None, {'error': 'no_summer_data', 'fallback_to_v1': True}
    
    # 步骤3: 检测维修事件并找到最后一次维修（近5年）
    maint_events = detect_maintenance(yearly_summer)
    last_maint_year, has_maintenance = find_last_maintenance(yearly_summer, recent_years=5)
    
    # 步骤4: 计算年趋势（维修后是新基础设施）
    annual_trend, is_post_maint_trend = calculate_annual_trend(yearly_summer, last_maint_year)
    
    # 确定锚定值
    if has_maintenance and last_maint_year in yearly_summer['year'].values:
        anchor_data = yearly_summer[yearly_summer['year'] == last_maint_year].iloc[0]
        anchor_year = last_maint_year
        anchor_value = anchor_data['mean']
    else:
        latest_summer = yearly_summer.iloc[-1]
        anchor_year = latest_summer['year']
        anchor_value = latest_summer['mean']
    
    # 步骤5: 聚合测试集为月度
    test_monthly = aggregate_to_monthly(test_df)
    
    # 只保留10月-次年6月的数据
    test_monthly = test_monthly[test_monthly['month'].isin([10, 11, 12, 1, 2, 3, 4, 5, 6])]
    
    if len(test_monthly) == 0:
        return None, None, {'error': 'no_non_summer_test_data'}
    
    # 步骤6: 只预测最后一年（测试集最后一个周期）
    # 找到测试集中最大的年份（考虑跨年：如果是1-6月，属于次年的周期）
    test_monthly['cycle_year'] = test_monthly.apply(
        lambda x: x['year'] if x['month'] >= 10 else x['year'] - 1, axis=1
    )
    
    last_cycle_year = test_monthly['cycle_year'].max()
    last_cycle_data = test_monthly[test_monthly['cycle_year'] == last_cycle_year]
    
    if len(last_cycle_data) == 0:
        return None, None, {'error': 'no_last_cycle_data'}
    
    # 步骤7: 预测最后一个周期的各月
    predictions = []
    actuals = []
    months_predicted = []
    
    for _, row in last_cycle_data.iterrows():
        target_month = row['month']
        actual_value = row['tqi_value']
        
        # 计算距离锚定7-9月的时间
        if target_month >= 10:
            months_from_anchor = target_month - 8
        else:
            months_from_anchor = 12 + target_month - 8
        
        years_from_anchor = months_from_anchor / 12
        
        # 趋势增量
        trend_delta = annual_trend * years_from_anchor
        
        # 基础预测
        base_prediction = anchor_value + trend_delta
        
        # 季节性调整
        target_seasonal = seasonal_factors.get(target_month, 1.0)
        if summer_factor_mean > 0:
            prediction = base_prediction * (target_seasonal / summer_factor_mean)
        else:
            prediction = base_prediction * target_seasonal
        
        predictions.append(prediction)
        actuals.append(actual_value)
        months_predicted.append(target_month)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 安全检查
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - 6 * train_std)
    safe_max = train_mean + 6 * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    metadata = {
        'anchor_year': anchor_year,
        'anchor_value': anchor_value,
        'annual_trend': annual_trend,
        'is_post_maint_trend': is_post_maint_trend,
        'has_maintenance': has_maintenance,
        'last_maintenance_year': last_maint_year,
        'summer_factor_mean': summer_factor_mean,
        'seasonal_strength': seasonal_strength,
        'num_summer_years': len(yearly_summer),
        'maintenance_events': maint_events,
        'num_predictions': len(predictions),
        'last_cycle_year': last_cycle_year,
        'months_predicted': months_predicted
    }
    
    return predictions, actuals, metadata


def trident_v1_predict(train_df, test_df):
    """Trident v1预测 (回退用)"""
    try:
        anchor_window = min(3, len(train_df) // 10)
        anchor_window = max(anchor_window, 1)
        anchor = train_df['tqi_value'].tail(anchor_window).mean()
        
        seasonal_factors = {
            1: 1.0, 2: 1.0, 3: 1.02, 4: 1.05, 5: 1.08, 6: 1.10,
            7: 1.12, 8: 1.10, 9: 1.05, 10: 1.02, 11: 1.0, 12: 1.0
        }
        
        train_df_copy = train_df.copy()
        train_df_copy['days'] = (pd.to_datetime(train_df_copy['date']) - pd.to_datetime(train_df_copy['date']).min()).dt.days
        slope = np.polyfit(train_df_copy['days'].values.astype(np.float64), 
                          train_df_copy['tqi_value'].values.astype(np.float64), 1)[0]
        
        predictions = []
        start_days = train_df_copy['days'].max()
        
        for i, row in test_df.iterrows():
            month = pd.to_datetime(row['date']).month
            day = start_days + (i + 1) * 30
            
            seasonal = seasonal_factors.get(month, 1.0)
            degradation = slope * day
            pred = anchor * seasonal + degradation
            predictions.append(pred)
        
        return np.array(predictions)
    except:
        return np.full(len(test_df), train_df['tqi_value'].mean())


def trident_v21_baseline(train_df, test_df):
    """Trident v2.1基线接口"""
    try:
        result = trident_v21_predict(train_df, test_df)
        
        if result[0] is None:
            # 回退到v1
            v1_pred = trident_v1_predict(train_df, test_df)
            y_true = test_df['tqi_value'].values
            return {
                'mae': mean_absolute_error(y_true, v1_pred),
                'rmse': np.sqrt(np.mean((y_true - v1_pred)**2)),
                'mape': np.mean(np.abs((y_true - v1_pred) / y_true)) * 100,
                'metadata': {**result[1], 'fallback_to_v1': True}
            }
        
        predictions, actuals, metadata = result
        
        return {
            'mae': mean_absolute_error(actuals, predictions),
            'rmse': np.sqrt(np.mean((actuals - predictions)**2)),
            'mape': np.mean(np.abs((actuals - predictions) / actuals)) * 100,
            'metadata': metadata
        }
    except Exception as e:
        print(f"      Trident v2.1错误: {e}")
        # 回退到v1
        try:
            v1_pred = trident_v1_predict(train_df, test_df)
            y_true = test_df['tqi_value'].values
            return {
                'mae': mean_absolute_error(y_true, v1_pred),
                'rmse': np.sqrt(np.mean((y_true - v1_pred)**2)),
                'mape': np.mean(np.abs((y_true - v1_pred) / y_true)) * 100,
                'metadata': {'error': str(e), 'fallback_to_v1': True}
            }
        except:
            y_true = test_df['tqi_value'].values
            fallback = np.full(len(test_df), train_df['tqi_value'].mean())
            return {
                'mae': mean_absolute_error(y_true, fallback),
                'rmse': np.sqrt(np.mean((y_true - fallback)**2)),
                'mape': np.mean(np.abs((y_true - fallback) / y_true)) * 100,
                'metadata': {'error': str(e), 'fallback_to_mean': True}
            }


if __name__ == '__main__':
    print("Trident v2.1 模块加载成功")
    print("核心设计: 7-9月基准期 + 维修检测 + 月度预测(10-6月)")
