#!/usr/bin/env python3
"""
Trident v2.0 - 优化版滚动锚定框架

优化内容:
1. 季节性参数自适应 (FFT检测季节性周期)
2. 非线性劣化趋势自适应 (三种方法+自动选择)
3. 锚定值优化 (使用7-9月稳定期作为锚定)
4. 异常样本检测与回退

作者: Kimi Claw
日期: 2026-03-24
"""
import pandas as pd
import numpy as np
from scipy import fft
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def detect_seasonality_fft(tqi_values, sample_rate=1):
    """
    使用FFT检测季节性周期
    
    Returns:
        seasonal_period: 检测到的周期长度(月)，如果无明显季节性返回None
        seasonal_strength: 季节性强度(0-1)
    """
    # 确保数值类型
    tqi_values = np.array(tqi_values, dtype=np.float64)
    
    if len(tqi_values) < 24:  # 至少需要2年数据
        return None, 0.0
    
    # FFT变换
    yf = fft.fft(tqi_values - np.mean(tqi_values))
    xf = fft.fftfreq(len(tqi_values), sample_rate)
    
    # 找主峰 (排除直流分量)
    power = np.abs(yf)**2
    power[0] = 0  # 排除直流分量
    
    # 找前3个峰值
    peak_indices = np.argsort(power)[-3:]
    
    # 检查是否有12个月周期
    seasonal_period = None
    seasonal_strength = 0.0
    
    for idx in peak_indices:
        if power[idx] > np.mean(power) * 2:  # 峰值显著高于平均
            period = 1 / abs(xf[idx]) if xf[idx] != 0 else float('inf')
            if 10 <= period <= 14:  # 接近12个月
                seasonal_period = 12
                seasonal_strength = power[idx] / np.sum(power)
                break
    
    return seasonal_period, seasonal_strength


def calculate_adaptive_seasonal_factors(train_df):
    """
    自适应计算季节性因子
    
    策略:
    1. 用FFT检测是否有12个月季节性
    2. 如有，计算各月平均TQI相对于年均值的比率
    3. 如无明显季节性，返回均匀因子(全1)
    """
    tqi_values = np.array(train_df['tqi_value'].values, dtype=np.float64)
    
    # 从date提取month
    train_df = train_df.copy()
    train_df['month'] = pd.to_datetime(train_df['date']).dt.month
    
    # 检测季节性
    seasonal_period, seasonal_strength = detect_seasonality_fft(tqi_values)
    
    if seasonal_period is None or seasonal_strength < 0.1:
        # 无明显季节性，返回中性因子
        return {m: 1.0 for m in range(1, 13)}, 0.0
    
    # 计算各月平均TQI
    monthly_avg = train_df.groupby('month')['tqi_value'].mean()
    overall_avg = train_df['tqi_value'].mean()
    
    # 计算季节性因子 (确保平均为1)
    seasonal_factors = {}
    for month in range(1, 13):
        if month in monthly_avg.index:
            seasonal_factors[month] = monthly_avg[month] / overall_avg
        else:
            seasonal_factors[month] = 1.0
    
    # 归一化使平均为1
    mean_factor = np.mean(list(seasonal_factors.values()))
    seasonal_factors = {k: v/mean_factor for k, v in seasonal_factors.items()}
    
    return seasonal_factors, seasonal_strength


def linear_trend(x, a, b):
    """线性趋势: a * x + b"""
    return a * x + b


def log_trend(x, a, b, c):
    """对数趋势: a * log(x+1) + b * x + c"""
    return a * np.log(x + 1) + b * x + c


def exp_trend(x, a, b, c):
    """指数趋势: a * exp(b * x) + c (带稳定性限制)"""
    # 限制指数增长，防止爆炸
    max_exp = 10  # 最大允许exp值
    exp_val = np.exp(np.clip(b * x, -max_exp, max_exp))
    return a * exp_val + c


def fit_trend_models(train_df):
    """
    拟合三种劣化趋势模型，选择最佳
    
    Returns:
        best_model: 最佳模型名称
        best_params: 模型参数
        trend_func: 趋势函数
    """
    days = (pd.to_datetime(train_df['date']) - pd.to_datetime(train_df['date']).min()).dt.days.values.astype(np.float64)
    tqi = train_df['tqi_value'].values.astype(np.float64)
    
    models = {}
    errors = {}
    
    # 1. 线性趋势
    try:
        popt, _ = curve_fit(linear_trend, days, tqi, maxfev=10000)
        pred = linear_trend(days, *popt)
        errors['linear'] = mean_absolute_error(tqi, pred)
        models['linear'] = (popt, linear_trend)
    except:
        errors['linear'] = float('inf')
    
    # 2. 对数趋势
    try:
        popt, _ = curve_fit(log_trend, days, tqi, maxfev=10000)
        pred = log_trend(days, *popt)
        errors['log'] = mean_absolute_error(tqi, pred)
        models['log'] = (popt, log_trend)
    except:
        errors['log'] = float('inf')
    
    # 3. 指数趋势 (带更严格的限制)
    try:
        # 指数趋势容易爆炸，严格限制参数范围
        # b必须<=0 (递减趋势) 或 很小的正值
        max_day = days.max()
        max_b = 5.0 / max_day if max_day > 0 else 0.001  # 确保exp(b*max_day) <= e^5 ≈ 148
        popt, _ = curve_fit(exp_trend, days, tqi, maxfev=10000, 
                           bounds=([-50, -0.01, -50], [50, max_b, 50]))
        pred = exp_trend(days, *popt)
        errors['exp'] = mean_absolute_error(tqi, pred)
        models['exp'] = (popt, exp_trend)
    except:
        errors['exp'] = float('inf')
    
    # 4. 平滑样条趋势
    try:
        spline = UnivariateSpline(days, tqi, s=len(days))
        pred = spline(days)
        errors['spline'] = mean_absolute_error(tqi, pred)
        models['spline'] = (spline, lambda x, s: s(x))
    except:
        errors['spline'] = float('inf')
    
    # 选择MAE最低的模型
    best_model = min(errors, key=errors.get)
    
    return best_model, models[best_model], errors[best_model]


def calculate_stable_anchor(train_df, test_year):
    """
    使用7-9月稳定期计算锚定值
    
    策略:
    1. 查找训练集中7-9月的数据
    2. 取最近一年的7-9月平均值作为锚定
    3. 预测目标是下一年7月前的数据
    """
    # 从date提取month和year
    train_df = train_df.copy()
    train_df['month'] = pd.to_datetime(train_df['date']).dt.month
    train_df['year'] = pd.to_datetime(train_df['date']).dt.year
    
    # 查找训练集中7-9月的数据
    summer_data = train_df[train_df['month'].isin([7, 8, 9])]
    
    if len(summer_data) == 0:
        # 没有7-9月数据，退回到最后3个月均值
        return train_df['tqi_value'].tail(3).mean(), False
    
    # 取最近一年的7-9月数据
    last_year = int(train_df['year'].max())
    last_summer = summer_data[summer_data['year'] == last_year]
    
    if len(last_summer) == 0:
        # 取所有7-9月数据的平均
        anchor = summer_data['tqi_value'].mean()
    else:
        # 加权平均 (8月权重最高)
        weights = last_summer['month'].map({7: 0.25, 8: 0.5, 9: 0.25})
        anchor = np.average(last_summer['tqi_value'], weights=weights)
    
    return anchor, True


def detect_anomaly_sample(train_df):
    """
    检测样本是否适合Trident模型
    
    返回: (is_anomaly, anomaly_score, reason)
    """
    tqi_values = np.array(train_df['tqi_value'].values, dtype=np.float64)
    
    scores = []
    reasons = []
    
    # 检查1: 数据量不足
    if len(tqi_values) < 30:
        scores.append(1.0)
        reasons.append("insufficient_data")
    
    # 检查2: 方差过大 (过于波动)
    cv = tqi_values.std() / tqi_values.mean() if tqi_values.mean() > 0 else float('inf')
    if cv > 0.5:  # 变异系数>50%
        scores.append(min(cv, 1.0))
        reasons.append("high_variance")
    
    # 检查3: 无明显趋势 (不适合劣化建模)
    days = np.arange(len(tqi_values))
    slope = np.polyfit(days, tqi_values, 1)[0]
    if abs(slope) < 0.001:  # 几乎无趋势
        scores.append(0.5)
        reasons.append("no_trend")
    
    # 检查4: 异常值过多
    q1, q3 = np.percentile(tqi_values, [25, 75])
    iqr = q3 - q1
    outliers = np.sum((tqi_values < q1 - 1.5*iqr) | (tqi_values > q3 + 1.5*iqr))
    if outliers > len(tqi_values) * 0.1:  # >10%异常值
        scores.append(outliers / len(tqi_values))
        reasons.append("too_many_outliers")
    
    anomaly_score = np.mean(scores) if scores else 0.0
    is_anomaly = anomaly_score > 0.5
    
    return is_anomaly, anomaly_score, reasons


def trident_v2_predict(train_df, test_df):
    """
    Trident v2.0 主预测函数
    
    Returns:
        predictions: 测试集预测值数组
        metadata: 包含各组件信息的字典
    """
    # 步骤1: 异常检测
    is_anomaly, anomaly_score, reasons = detect_anomaly_sample(train_df)
    
    # 准备测试集时间信息
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    test_dates = test_df['date']
    test_year = test_dates.dt.year
    test_month = test_dates.dt.month
    test_days = (test_dates - pd.to_datetime(train_df['date']).min()).dt.days.values.astype(np.float64)
    
    # 步骤2: 计算锚定值 (使用7-9月稳定期)
    anchor, used_summer = calculate_stable_anchor(train_df, test_year.min())
    
    # 步骤3: 自适应季节性
    seasonal_factors, seasonal_strength = calculate_adaptive_seasonal_factors(train_df)
    
    # 步骤4: 拟合劣化趋势
    best_trend_model, (trend_params, trend_func), trend_mae = fit_trend_models(train_df)
    
    # 步骤5: 预测
    predictions = []
    
    # 训练数据范围 (用于限制预测值)
    train_min = train_df['tqi_value'].min()
    train_max = train_df['tqi_value'].max()
    train_mean = float(train_df['tqi_value'].mean())
    train_std = train_df['tqi_value'].std()
    # 允许的范围: 均值 ± 5倍标准差
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    
    for i, (date, month, days) in enumerate(zip(test_dates, test_month, test_days)):
        # 基础锚定
        base = anchor
        
        # 季节性调整
        seasonal_factor = seasonal_factors.get(month, 1.0)
        
        # 劣化趋势
        try:
            if best_trend_model == 'spline':
                trend_val = float(trend_func(days, trend_params))
            else:
                trend_val = float(trend_func(days, *trend_params))
        except:
            # 趋势计算失败，使用均值
            trend_val = train_mean
        
        # 组合预测
        # 策略: 锚定值 * 季节性因子 + 趋势调整
        pred = base * seasonal_factor + (trend_val - train_mean)
        
        # 限制预测值在合理范围
        pred = np.clip(pred, safe_min, safe_max)
        
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # 元数据
    metadata = {
        'is_anomaly': is_anomaly,
        'anomaly_score': anomaly_score,
        'anomaly_reasons': reasons,
        'anchor': anchor,
        'used_summer_anchor': used_summer,
        'seasonal_factors': seasonal_factors,
        'seasonal_strength': seasonal_strength,
        'trend_model': best_trend_model,
        'trend_mae': trend_mae
    }
    
    return predictions, metadata


def trident_v2_baseline(train_df, test_df):
    """
    Trident v2.0 基线接口 (适配原实验框架)
    """
    try:
        predictions, metadata = trident_v2_predict(train_df, test_df)
        y_true = test_df['tqi_value'].values
        
        # 检查预测值是否合理
        train_mean = train_df['tqi_value'].mean()
        train_std = train_df['tqi_value'].std()
        
        # 如果预测值超出合理范围，使用回退
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        # 判断预测是否异常 (预测均值偏离训练均值超过3倍标准差)
        if abs(pred_mean - train_mean) > 3 * train_std or metadata['is_anomaly']:
            # 使用Historical Mean作为回退
            fallback_pred = np.full(len(test_df), train_mean)
            predictions = fallback_pred
            metadata['used_fallback'] = True
        else:
            metadata['used_fallback'] = False
        
        return {
            'mae': mean_absolute_error(y_true, predictions),
            'rmse': np.sqrt(np.mean((y_true - predictions)**2)),
            'mape': np.mean(np.abs((y_true - predictions) / y_true)) * 100,
            'metadata': metadata
        }
    except Exception as e:
        print(f"      Trident v2错误: {e}")
        # 回退到简单均值
        y_true = test_df['tqi_value'].values
        fallback = np.full(len(test_df), train_df['tqi_value'].mean())
        return {
            'mae': mean_absolute_error(y_true, fallback),
            'rmse': np.sqrt(np.mean((y_true - fallback)**2)),
            'mape': np.mean(np.abs((y_true - fallback) / y_true)) * 100,
            'metadata': {'error': str(e), 'used_fallback': True}
        }


# 向后兼容: Trident v1 (原始版本)
def trident_rolling_anchor(train_df, test_df):
    """
    Trident v1.0 - 原始滚动锚定 (保持兼容)
    """
    try:
        from scipy.interpolate import interp1d
        
        # 计算锚定值 (训练集最后3个月平均)
        anchor_window = min(3, len(train_df) // 10)
        anchor_window = max(anchor_window, 1)
        anchor = train_df['tqi_value'].tail(anchor_window).mean()
        
        # 季节性因子 (固定模式)
        seasonal_factors = {
            1: 1.0, 2: 1.0, 3: 1.02, 4: 1.05, 5: 1.08, 6: 1.10,
            7: 1.12, 8: 1.10, 9: 1.05, 10: 1.02, 11: 1.0, 12: 1.0
        }
        
        # 线性劣化趋势
        days = (train_df['dete_dt'] - train_df['dete_dt'].min()).dt.days.values
        tqi = train_df['tqi_value'].values
        slope = np.polyfit(days, tqi, 1)[0] if len(days) > 1 else 0
        
        # 预测
        predictions = []
        for _, row in test_df.iterrows():
            month = row['month']
            day = (row['dete_dt'] - train_df['dete_dt'].min()).days
            
            seasonal = seasonal_factors.get(month, 1.0)
            degradation = slope * day
            
            pred = anchor * seasonal + degradation
            predictions.append(pred)
        
        y_true = test_df['tqi_value'].values
        y_pred = np.array(predictions)
        
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(np.mean((y_true - y_pred)**2)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    except Exception as e:
        print(f"      Trident v1错误: {e}")
        return {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}


if __name__ == '__main__':
    # 测试代码
    print("Trident v2.0 模块加载成功")
    print("优化功能:")
    print("  1. 自适应季节性检测 (FFT)")
    print("  2. 自适应劣化趋势 (4种模型自动选择)")
    print("  3. 7-9月稳定期锚定")
    print("  4. 异常样本检测")
