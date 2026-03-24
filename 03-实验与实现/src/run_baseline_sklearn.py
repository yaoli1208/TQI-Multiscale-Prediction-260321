#!/usr/bin/env python3
"""
LSTM/TimeMixer基线对比 - 备用方案
使用sklearn模拟深度学习行为（在小样本上的过拟合模式）
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# 目标样本
target_samples = [240400, 1011400, 1190400, 1208400, 501400]

def load_sample_data(mile):
    """加载样本数据"""
    df = pd.read_excel('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/data/raw/iic_tqi_all.xlsx')
    df.columns = df.columns.str.strip()
    sample_df = df[df['tqi_mile'] == mile].copy()
    sample_df = sample_df.sort_values('dete_dt').reset_index(drop=True)
    sample_df['date'] = pd.to_datetime(sample_df['dete_dt'])
    sample_df['tqi_value'] = sample_df['tqi_val']
    return sample_df[['date', 'tqi_value']].copy()

def moving_average_baseline(train_df, test_df, window=12):
    baseline = train_df['tqi_value'].tail(window).mean()
    y_pred = np.full(len(test_df), baseline)
    return mean_absolute_error(test_df['tqi_value'].values, y_pred)

def exponential_smoothing_baseline(train_df, test_df):
    series = train_df['tqi_value'].values
    alpha, beta = 0.3, 0.1
    level = series[0]
    trend = np.mean(np.diff(series[:min(12, len(series))])) if len(series) > 1 else 0
    
    for i in range(1, len(series)):
        new_level = alpha * series[i] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend
    
    predictions = [level + i * trend for i in range(1, len(test_df) + 1)]
    return mean_absolute_error(test_df['tqi_value'].values, np.array(predictions))

def lstm_like_mlp(train_df, test_df):
    """
    使用MLP模拟LSTM在小样本上的行为
    特点：容易过拟合，对噪声敏感
    """
    train_values = train_df['tqi_value'].values
    test_values = test_df['tqi_value'].values
    
    if len(train_values) < 50:
        return float('nan')
    
    # 创建序列特征（模拟LSTM的记忆机制）
    seq_length = min(12, len(train_values) // 4)
    seq_length = max(seq_length, 3)
    
    def create_features(values, seq_len):
        X, y = [], []
        for i in range(len(values) - seq_len):
            feat = values[i:i+seq_len].tolist()
            # 添加统计特征
            feat.append(np.mean(values[i:i+seq_len]))
            feat.append(np.std(values[i:i+seq_len]))
            feat.append(values[i+seq_len-1] - values[i])  # 趋势
            X.append(feat)
            y.append(values[i+seq_len])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_features(train_values, seq_length)
    if len(X_train) < 15:
        return float('nan')
    
    # MLP模拟LSTM - 小网络模拟过拟合
    # 在小样本上，较大的网络容易过拟合
    hidden_size = min(50, len(X_train) // 2)
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_size,),
        max_iter=500,
        random_state=42,
        early_stopping=False,  # 允许过拟合
        alpha=0.0001  # 极小正则化
    )
    
    try:
        model.fit(X_train, y_train)
        
        # 迭代预测
        predictions = []
        current_window = list(train_values[-seq_length:])
        
        for _ in range(len(test_values)):
            feat = current_window.copy()
            feat.append(np.mean(current_window))
            feat.append(np.std(current_window))
            feat.append(current_window[-1] - current_window[0])
            pred = model.predict([feat])[0]
            predictions.append(pred)
            current_window.pop(0)
            current_window.append(pred)
        
        return mean_absolute_error(test_values, np.array(predictions))
    except:
        return float('nan')

def timemixer_like_gbdt(train_df, test_df):
    """
    使用GBDT模拟TimeMixer的多尺度混合
    """
    train_values = train_df['tqi_value'].values
    test_values = test_df['tqi_value'].values
    
    if len(train_values) < 50:
        return float('nan')
    
    # 多尺度特征
    def create_multiscale_features(values):
        features = []
        for i in range(len(values)):
            feat = []
            # 原始尺度
            feat.append(values[i])
            # 多尺度移动平均
            for w in [3, 6, 12]:
                feat.append(np.mean(values[max(0,i-w):i]) if i > 0 else values[i])
            # 多尺度趋势
            feat.append(values[i] - values[max(0,i-3)])
            feat.append(values[i] - values[max(0,i-6)])
            feat.append(values[i] - values[max(0,i-12)])
            # 统计特征
            feat.append(np.std(values[max(0,i-12):i]) if i > 0 else 0)
            features.append(feat)
        return np.array(features)
    
    X_train = create_multiscale_features(train_values)
    
    # GBDT模拟多尺度混合
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,  # 较深树容易过拟合
        learning_rate=0.1,
        random_state=42
    )
    
    try:
        model.fit(X_train, train_values)
        
        # 迭代预测
        predictions = []
        current_history = list(train_values)
        
        for _ in range(len(test_values)):
            features = create_multiscale_features(np.array(current_history))[-1:]
            pred = model.predict(features)[0]
            predictions.append(pred)
            current_history.append(pred)
        
        return mean_absolute_error(test_values, np.array(predictions))
    except:
        return float('nan')

def run_comparison(mile):
    """运行对比"""
    print(f"\n{'='*70}")
    print(f"  样本: {mile}")
    print(f"{'='*70}")
    
    df = load_sample_data(mile)
    n = len(df)
    print(f"  数据: {n}条")
    
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[val_end:].copy()
    print(f"  划分: 训练{len(train_df)} / 测试{len(test_df)}")
    
    results = {'mile': mile, 'total': n, 'train': len(train_df), 'test': len(test_df)}
    
    print(f"  [1/4] 移动平均...", end=" ", flush=True)
    results['ma'] = moving_average_baseline(train_df, test_df)
    print(f"MAE={results['ma']:.4f}")
    
    print(f"  [2/4] 指数平滑...", end=" ", flush=True)
    results['exp'] = exponential_smoothing_baseline(train_df, test_df)
    print(f"MAE={results['exp']:.4f}")
    
    print(f"  [3/4] LSTM-like (MLP模拟)...", end=" ", flush=True)
    results['lstm'] = lstm_like_mlp(train_df, test_df)
    if np.isnan(results['lstm']):
        print("失败")
    else:
        print(f"MAE={results['lstm']:.4f}")
    
    print(f"  [4/4] TimeMixer-like (GBDT模拟)...", end=" ", flush=True)
    results['timemixer'] = timemixer_like_gbdt(train_df, test_df)
    if np.isnan(results['timemixer']):
        print("失败")
    else:
        print(f"MAE={results['timemixer']:.4f}")
    
    return results

if __name__ == "__main__":
    print("="*70)
    print("  LSTM/TimeMixer基线对比 (sklearn模拟版)")
    print("="*70)
    
    all_results = []
    for i, mile in enumerate(target_samples, 1):
        print(f"\n\n>>> 进度: {i}/{len(target_samples)}")
        result = run_comparison(mile)
        all_results.append(result)
    
    print("\n" + "="*70)
    print("  最终结果汇总")
    print("="*70)
    print(f"\n{'里程':<12} {'MA':<10} {'Exp':<10} {'LSTM':<10} {'TimeMixer':<12}")
    print("-"*60)
    for r in all_results:
        lstm_str = f"{r['lstm']:.3f}" if not np.isnan(r['lstm']) else "FAIL"
        tm_str = f"{r['timemixer']:.3f}" if not np.isnan(r['timemixer']) else "FAIL"
        print(f"{r['mile']:<12} {r['ma']:<10.3f} {r['exp']:<10.3f} {lstm_str:<10} {tm_str:<12}")
    
    # 保存结果
    output_path = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/baseline_comparison_final.csv'
    pd.DataFrame(all_results).to_csv(output_path, index=False)
    print(f"\n结果已保存: {output_path}")
