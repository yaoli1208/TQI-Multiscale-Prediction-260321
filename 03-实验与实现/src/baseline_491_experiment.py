#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
491样本基线对比实验 - 整合版
============================
6种基线方法对比:
1. 历史均值 (Historical Mean) - 新增
2. 移动平均 (Moving Average, MA) - 复用
3. Holt-Winters (Triple Exponential Smoothing) - 升级
4. LSTM - 复用
5. TimeMixer - 复用
6. Trident (滚动锚定) - 复用

代码来源:
- MA/LSTM/TimeMixer: run_baseline_full.py
- Trident: full_experiment_514.py
- Historical Mean: 新增
- Holt-Winters: 升级使用statsmodels

作者: Kimi Claw
日期: 2026-03-24
版本: v1.0
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 配置路径
BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v2.txt'
RESULTS_DIR = f'{BASE_DIR}/results/baseline_491_experiment'

# 创建结果目录
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== 1. 历史均值基线 (Historical Mean) ====================
def historical_mean_baseline(train_df, test_df):
    """
    历史均值基线 - 使用训练集整体均值作为常数预测
    
    文献参考:
    - Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice
    - Makridakis et al. (2020). The M5 Accuracy competition
    
    特点:
    - 预测值为常数（不随时间变化）
    - 时间序列预测的"下限基线"
    """
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

# ==================== 2. 移动平均基线 (Moving Average) ====================
def moving_average_baseline(train_df, test_df, window=30):
    """
    移动平均基线 - 使用最近window个时间步的均值
    
    代码来源: run_baseline_full.py (修改window=30适配月度数据)
    """
    baseline = train_df['tqi_value'].tail(window).mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

# ==================== 3. Holt-Winters基线 ====================
def holt_winters_baseline(train_df, test_df, seasonal_periods=12):
    """
    Holt-Winters季节性指数平滑 - 完整版
    
    升级: 使用statsmodels实现完整的三重指数平滑
    代码来源: 基于exponential_smoothing_baseline升级
    
    参数:
        seasonal_periods: 季节性周期（默认12个月）
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        series = train_df['tqi_value'].values
        
        # 确保有足够的数据
        if len(series) < seasonal_periods * 2:
            # 数据不足时回退到简单指数平滑
            return exponential_smoothing_simple(train_df, test_df)
        
        # 拟合Holt-Winters模型
        model = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal_periods
        )
        fitted = model.fit(optimized=True)
        
        # 预测
        predictions = fitted.forecast(steps=len(test_df))
        y_true = test_df['tqi_value'].values
        
        return {
            'mae': mean_absolute_error(y_true, predictions),
            'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
            'mape': np.mean(np.abs((y_true - predictions) / y_true)) * 100
        }
    except Exception as e:
        # 失败时回退到简单指数平滑
        return exponential_smoothing_simple(train_df, test_df)

def exponential_smoothing_simple(train_df, test_df):
    """简化版指数平滑 (Holt线性趋势) - 原run_baseline_full.py实现"""
    series = train_df['tqi_value'].values
    alpha, beta = 0.3, 0.1
    level = series[0]
    trend = np.mean(np.diff(series[:min(12, len(series))])) if len(series) > 1 else 0
    
    for i in range(1, len(series)):
        new_level = alpha * series[i] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend
    
    predictions = [level + i * trend for i in range(1, len(test_df) + 1)]
    y_true = test_df['tqi_value'].values
    y_pred = np.array(predictions)
    
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

# ==================== 4. LSTM基线 ====================
def lstm_baseline(train_df, test_df):
    """
    LSTM预测 - 完整版 (50 epochs)
    
    代码来源: run_baseline_full.py (lstm_prediction_full)
    修改: 适配新的数据接口，返回多指标
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        tf.get_logger().setLevel('ERROR')
        
        train_values = train_df['tqi_value'].values
        test_values = test_df['tqi_value'].values
        
        if len(train_values) < 50:
            return {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}
        
        mean_val, std_val = train_values.mean(), train_values.std()
        if std_val == 0:
            std_val = 1
        train_normalized = (train_values - mean_val) / std_val
        
        seq_length = min(30, len(train_values) // 4)
        seq_length = max(seq_length, 12)
        
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_sequences(train_normalized, seq_length)
        if len(X_train) < 10:
            return {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        # 简化模型结构（小数据场景）
        model = Sequential([
            LSTM(32, activation='relu', input_shape=(seq_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        model.fit(X_train, y_train, epochs=50, verbose=0, batch_size=16)
        
        predictions = []
        current_seq = train_normalized[-seq_length:].reshape(1, seq_length, 1)
        
        for _ in range(len(test_values)):
            pred = model.predict(current_seq, verbose=0)[0][0]
            predictions.append(pred)
            current_seq = np.roll(current_seq, -1)
            current_seq[0, -1, 0] = pred
        
        y_pred = np.array(predictions) * std_val + mean_val
        
        return {
            'mae': mean_absolute_error(test_values, y_pred),
            'rmse': np.sqrt(mean_squared_error(test_values, y_pred)),
            'mape': np.mean(np.abs((test_values - y_pred) / test_values)) * 100
        }
        
    except Exception as e:
        print(f"      LSTM错误: {e}")
        return {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}

# ==================== 5. TimeMixer基线 ====================
def timemixer_baseline(train_df, test_df):
    """
    TimeMixer简化实现 - 多尺度分解
    
    代码来源: run_baseline_full.py (timemixer_prediction_full)
    修改: 适配新的数据接口，返回多指标
    """
    try:
        train_values = train_df['tqi_value'].values
        test_values = test_df['tqi_value'].values
        
        if len(train_values) < 50:
            return {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}
        
        def create_multiscale_features(values):
            features = []
            for i in range(len(values)):
                feat = [values[i]]
                for w in [3, 6, 12, 24]:
                    feat.append(np.mean(values[max(0,i-w):i]) if i > 0 else values[i])
                feat.append(values[i] - values[max(0,i-3)])
                feat.append(values[i] - values[max(0,i-12)])
                features.append(feat)
            return np.array(features)
        
        X_train = create_multiscale_features(train_values)
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, train_values)
        
        predictions = []
        current_history = list(train_values)
        
        for _ in range(len(test_values)):
            features = create_multiscale_features(np.array(current_history))[-1:]
            pred = model.predict(features)[0]
            predictions.append(pred)
            current_history.append(pred)
        
        y_pred = np.array(predictions)
        
        return {
            'mae': mean_absolute_error(test_values, y_pred),
            'rmse': np.sqrt(mean_squared_error(test_values, y_pred)),
            'mape': np.mean(np.abs((test_values - y_pred) / test_values)) * 100
        }
        
    except Exception as e:
        print(f"      TimeMixer错误: {e}")
        return {'mae': float('nan'), 'rmse': float('nan'), 'mape': float('nan')}

# ==================== 6. Trident基线 (滚动锚定) ====================
def calculate_seasonal(train_df):
    """计算季节性调整系数"""
    train_df = train_df.copy()
    train_df['month'] = train_df['date'].dt.month
    seasonal_mean = train_df.groupby('month')['tqi_value'].mean()
    overall_mean = train_df['tqi_value'].mean()
    seasonal_adj = seasonal_mean - overall_mean
    return seasonal_adj, overall_mean

def trident_rolling_anchor(train_df, test_df):
    """
    Trident滚动锚定策略
    
    代码来源: full_experiment_514.py (trident_strategies)
    修改: 适配新的数据接口，仅保留滚动锚定策略
    """
    # 计算季节性
    seasonal_adj, train_mean = calculate_seasonal(train_df)
    
    y_true = test_df['tqi_value'].values
    
    # 滚动锚定: 最近3年度均值 + 季节性调整
    train_df_copy = train_df.copy()
    train_df_copy['year'] = train_df_copy['date'].dt.year
    yearly_mean = train_df_copy.groupby('year')['tqi_value'].mean()
    recent_mean = yearly_mean.tail(3).mean() if len(yearly_mean) >= 3 else yearly_mean.mean()
    
    predictions = []
    for _, row in test_df.iterrows():
        month = row['date'].month
        seasonal = seasonal_adj.get(month, 0)
        predictions.append(recent_mean + seasonal)
    
    y_pred = np.array(predictions)
    
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

# ==================== 数据加载与处理 ====================
def load_sample_list():
    """加载491个合格样本列表"""
    with open(SAMPLE_LIST_FILE, 'r') as f:
        samples = [int(line.strip()) for line in f if line.strip()]
    return samples

def load_sample_data(mile):
    """加载单个样本数据"""
    try:
        df = pd.read_excel(DATA_FILE)
        df.columns = df.columns.str.strip()
        sample_df = df[df['tqi_mile'] == mile].copy()
        sample_df = sample_df.sort_values('dete_dt').reset_index(drop=True)
        sample_df['date'] = pd.to_datetime(sample_df['dete_dt'])
        sample_df['tqi_value'] = sample_df['tqi_val']
        return sample_df[['date', 'tqi_value']].copy()
    except Exception as e:
        print(f"    加载样本 {mile} 失败: {e}")
        return None

def split_data(df):
    """时间序列划分: 70%训练 / 15%验证 / 15%测试"""
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

# ==================== 主实验流程 ====================
def run_single_sample(mile, verbose=True):
    """对单个样本运行所有基线对比"""
    if verbose:
        print(f"\n{'='*70}")
        print(f"样本: {mile}")
        print(f"{'='*70}")
    
    # 加载数据
    df = load_sample_data(mile)
    if df is None or len(df) < 50:
        return None
    
    # 数据划分
    train_df, val_df, test_df = split_data(df)
    n = len(df)
    
    if verbose:
        print(f"数据: {n}条 (训练{len(train_df)}/验证{len(val_df)}/测试{len(test_df)})")
    
    result = {
        'tqi_mile': mile,
        'record_count': n,
        'train_count': len(train_df),
        'val_count': len(val_df),
        'test_count': len(test_df)
    }
    
    # 运行6种基线方法
    methods = [
        ('historical_mean', historical_mean_baseline, '[1/6] 历史均值'),
        ('ma', moving_average_baseline, '[2/6] 移动平均'),
        ('holt_winters', holt_winters_baseline, '[3/6] Holt-Winters'),
        ('lstm', lstm_baseline, '[4/6] LSTM'),
        ('timemixer', timemixer_baseline, '[5/6] TimeMixer'),
        ('trident', trident_rolling_anchor, '[6/6] Trident')
    ]
    
    for key, func, label in methods:
        if verbose:
            print(f"  {label}...", end=' ', flush=True)
        
        try:
            # LSTM和TimeMixer需要更长的训练时间
            if key in ['lstm', 'timemixer']:
                metrics = func(train_df, test_df)
            else:
                metrics = func(train_df, test_df)
            
            result[f'{key}_mae'] = metrics['mae']
            result[f'{key}_rmse'] = metrics['rmse']
            result[f'{key}_mape'] = metrics['mape']
            
            if verbose:
                print(f"MAE={metrics['mae']:.4f}")
        except Exception as e:
            if verbose:
                print(f"失败: {e}")
            result[f'{key}_mae'] = float('nan')
            result[f'{key}_rmse'] = float('nan')
            result[f'{key}_mape'] = float('nan')
    
    # 确定最佳基线
    baseline_keys = ['historical_mean', 'ma', 'holt_winters', 'lstm', 'timemixer']
    baseline_maes = {k: result[f'{k}_mae'] for k in baseline_keys}
    valid_baselines = {k: v for k, v in baseline_maes.items() if not np.isnan(v)}
    
    if valid_baselines:
        best_baseline = min(valid_baselines.items(), key=lambda x: x[1])
        result['best_baseline'] = best_baseline[0]
        result['best_baseline_mae'] = best_baseline[1]
    else:
        result['best_baseline'] = 'none'
        result['best_baseline_mae'] = float('nan')
    
    # Trident有效性判断
    trident_mae = result.get('trident_mae', float('nan'))
    best_baseline_mae = result.get('best_baseline_mae', float('nan'))
    
    if not np.isnan(trident_mae) and not np.isnan(best_baseline_mae) and best_baseline_mae > 0:
        improvement = (best_baseline_mae - trident_mae) / best_baseline_mae
        result['trident_improvement'] = improvement
        result['is_trident_best'] = trident_mae < best_baseline_mae
        result['is_trident_effective'] = improvement > 0.10
    else:
        result['trident_improvement'] = float('nan')
        result['is_trident_best'] = False
        result['is_trident_effective'] = False
    
    return result

def run_batch_experiment(batch_id, sample_list, batch_size=50):
    """分批运行实验"""
    start_idx = batch_id * batch_size
    end_idx = min((batch_id + 1) * batch_size, len(sample_list))
    batch_samples = sample_list[start_idx:end_idx]
    
    print(f"\n{'='*70}")
    print(f"批次 {batch_id+1}: 样本 {start_idx+1}-{end_idx} / {len(sample_list)}")
    print(f"{'='*70}")
    
    results = []
    for i, mile in enumerate(batch_samples, 1):
        print(f"\n>>> 进度: {i}/{len(batch_samples)} (总进度: {start_idx+i}/{len(sample_list)})")
        result = run_single_sample(mile, verbose=True)
        if result:
            results.append(result)
        
        # 每10个样本保存一次中间结果
        if i % 10 == 0:
            save_intermediate_results(results, batch_id)
    
    return results

def save_intermediate_results(results, batch_id):
    """保存中间结果"""
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(f'{RESULTS_DIR}/batch_{batch_id}_intermediate.csv', index=False)

def save_final_results(all_results):
    """保存最终结果"""
    df = pd.DataFrame(all_results)
    
    # 主结果文件
    df.to_csv(f'{RESULTS_DIR}/baseline_491_full_results.csv', index=False)
    
    # 汇总统计
    summary = generate_summary(df)
    with open(f'{RESULTS_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def generate_summary(df):
    """生成实验汇总统计"""
    summary = {
        'experiment_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(df),
            'valid_samples': len(df.dropna(subset=['best_baseline_mae']))
        },
        'baseline_wins': {},
        'trident_stats': {},
        'method_mae_stats': {}
    }
    
    # 各基线获胜次数
    baseline_keys = ['historical_mean', 'ma', 'holt_winters', 'lstm', 'timemixer']
    for key in baseline_keys:
        count = (df['best_baseline'] == key).sum()
        summary['baseline_wins'][key] = int(count)
    
    # Trident统计
    summary['trident_stats']['is_best_count'] = int(df['is_trident_best'].sum())
    summary['trident_stats']['is_best_ratio'] = float(df['is_trident_best'].mean())
    summary['trident_stats']['effective_count'] = int(df['is_trident_effective'].sum())
    summary['trident_stats']['effective_ratio'] = float(df['is_trident_effective'].mean())
    summary['trident_stats']['mean_improvement'] = float(df['trident_improvement'].mean())
    
    # 各方法MAE统计
    for key in baseline_keys + ['trident']:
        mae_col = f'{key}_mae'
        if mae_col in df.columns:
            summary['method_mae_stats'][key] = {
                'mean': float(df[mae_col].mean()),
                'median': float(df[mae_col].median()),
                'std': float(df[mae_col].std())
            }
    
    return summary

def print_summary(summary):
    """打印汇总结果"""
    print("\n" + "="*70)
    print("实验汇总结果")
    print("="*70)
    
    print(f"\n总样本数: {summary['experiment_info']['total_samples']}")
    print(f"有效样本: {summary['experiment_info']['valid_samples']}")
    
    print("\n【各基线获胜次数】")
    for method, count in summary['baseline_wins'].items():
        print(f"  {method:20s}: {count:3d}")
    
    print("\n【Trident表现】")
    stats = summary['trident_stats']
    print(f"  最佳次数: {stats['is_best_count']} ({stats['is_best_ratio']*100:.1f}%)")
    print(f"  有效次数: {stats['effective_count']} ({stats['effective_ratio']*100:.1f}%)")
    print(f"  平均改善: {stats['mean_improvement']*100:.1f}%")
    
    print("\n【各方法MAE统计】")
    for method, stats in summary['method_mae_stats'].items():
        print(f"  {method:20s}: 均值={stats['mean']:.4f}, 中位数={stats['median']:.4f}")

# ==================== 主入口 ====================
if __name__ == "__main__":
    print("="*70)
    print("491样本基线对比实验 - 6种方法")
    print("="*70)
    print("\n方法列表:")
    print("  1. 历史均值 (Historical Mean)")
    print("  2. 移动平均 (Moving Average)")
    print("  3. Holt-Winters (Triple Exponential Smoothing)")
    print("  4. LSTM")
    print("  5. TimeMixer")
    print("  6. Trident (滚动锚定)")
    print("\n代码复用来源:")
    print("  - MA/LSTM/TimeMixer: run_baseline_full.py")
    print("  - Trident: full_experiment_514.py")
    print("  - Historical Mean: 新增")
    print("  - Holt-Winters: 升级使用statsmodels")
    print("="*70)
    
    # 加载样本列表
    sample_list = load_sample_list()
    print(f"\n加载样本列表: {len(sample_list)}个样本")
    
    # 运行实验（分批）
    BATCH_SIZE = 50
    NUM_BATCHES = (len(sample_list) + BATCH_SIZE - 1) // BATCH_SIZE
    
    all_results = []
    for batch_id in range(NUM_BATCHES):
        batch_results = run_batch_experiment(batch_id, sample_list, BATCH_SIZE)
        all_results.extend(batch_results)
        
        # 保存中间结果
        save_intermediate_results(all_results, batch_id)
        print(f"\n>>> 批次 {batch_id+1}/{NUM_BATCHES} 完成，已保存中间结果")
    
    # 保存最终结果
    summary = save_final_results(all_results)
    print_summary(summary)
    
    print("\n" + "="*70)
    print(f"实验完成！结果保存至: {RESULTS_DIR}")
    print("="*70)
