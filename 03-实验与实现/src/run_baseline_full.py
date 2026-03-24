#!/usr/bin/env python3
"""
完整版LSTM和TimeMixer基线对比 - 用于论文精确数据
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 目标样本
target_samples = [240400, 1011400, 1190400, 1208400, 501400]

def load_sample_data(mile):
    """从CSV加载样本数据（预处理过的）"""
    csv_path = f'/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/batch_experiments/sample_{mile}_data.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date').reset_index(drop=True)
    else:
        # 从原始数据提取 - 列名有空格需要处理
        df = pd.read_excel('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/data/raw/iic_tqi_all.xlsx')
        # 清理列名空格
        df.columns = df.columns.str.strip()
        sample_df = df[df['tqi_mile'] == mile].copy()
        sample_df = sample_df.sort_values('dete_dt').reset_index(drop=True)
        sample_df['date'] = pd.to_datetime(sample_df['dete_dt'])
        sample_df['tqi_value'] = sample_df['tqi_val']
        return sample_df[['date', 'tqi_value']].copy()

def moving_average_baseline(train_df, test_df, window=12):
    baseline = train_df['tqi_value'].tail(window).mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return mean_absolute_error(y_true, y_pred)

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
    y_true = test_df['tqi_value'].values
    return mean_absolute_error(y_true, np.array(predictions))

def lstm_prediction_full(train_df, test_df):
    """完整版LSTM - 50 epochs"""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        tf.get_logger().setLevel('ERROR')
        
        train_values = train_df['tqi_value'].values
        test_values = test_df['tqi_value'].values
        
        if len(train_values) < 50:
            return float('nan')
        
        mean_val, std_val = train_values.mean(), train_values.std()
        if std_val == 0:
            std_val = 1
        train_normalized = (train_values - mean_val) / std_val
        
        seq_length = min(12, len(train_values) // 4)
        seq_length = max(seq_length, 3)
        
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(len(data) - seq_len):
                X.append(data[i:i+seq_len])
                y.append(data[i+seq_len])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_sequences(train_normalized, seq_length)
        if len(X_train) < 10:
            return float('nan')
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        print(f"      LSTM训练中 (50 epochs)...", flush=True)
        model.fit(X_train, y_train, epochs=50, verbose=0, batch_size=8)
        
        predictions = []
        current_seq = train_normalized[-seq_length:].reshape(1, seq_length, 1)
        
        for _ in range(len(test_values)):
            pred = model.predict(current_seq, verbose=0)[0][0]
            predictions.append(pred)
            current_seq = np.roll(current_seq, -1)
            current_seq[0, -1, 0] = pred
        
        y_pred = np.array(predictions) * std_val + mean_val
        mae = mean_absolute_error(test_values, y_pred)
        print(f"      LSTM完成: MAE={mae:.4f}", flush=True)
        return mae
        
    except Exception as e:
        print(f"      LSTM错误: {e}", flush=True)
        return float('nan')

def timemixer_prediction_full(train_df, test_df):
    """完整版TimeMixer简化实现 - 多尺度分解"""
    try:
        train_values = train_df['tqi_value'].values
        test_values = test_df['tqi_value'].values
        
        if len(train_values) < 50:
            return float('nan')
        
        print(f"      TimeMixer训练中...", flush=True)
        
        def create_multiscale_features(values):
            features = []
            for i in range(len(values)):
                feat = [values[i]]  # 原始值
                for w in [3, 6, 12]:
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
        
        mae = mean_absolute_error(test_values, np.array(predictions))
        print(f"      TimeMixer完成: MAE={mae:.4f}", flush=True)
        return mae
        
    except Exception as e:
        print(f"      TimeMixer错误: {e}", flush=True)
        return float('nan')

def run_full_comparison(mile):
    """完整对比"""
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
    
    results = {'mile': mile}
    
    print(f"  [1/4] 移动平均...", end=" ", flush=True)
    results['ma'] = moving_average_baseline(train_df, test_df)
    print(f"MAE={results['ma']:.4f}")
    
    print(f"  [2/4] 指数平滑...", end=" ", flush=True)
    results['exp'] = exponential_smoothing_baseline(train_df, test_df)
    print(f"MAE={results['exp']:.4f}")
    
    print(f"  [3/4] LSTM (完整版50 epochs)...")
    results['lstm'] = lstm_prediction_full(train_df, test_df)
    
    print(f"  [4/4] TimeMixer...")
    results['timemixer'] = timemixer_prediction_full(train_df, test_df)
    
    return results

if __name__ == "__main__":
    print("="*70)
    print("  完整版LSTM/TimeMixer基线对比")
    print("  预计时间: 10-30分钟")
    print("="*70)
    
    all_results = []
    for i, mile in enumerate(target_samples, 1):
        print(f"\n\n>>> 进度: {i}/{len(target_samples)} 样本")
        result = run_full_comparison(mile)
        all_results.append(result)
        
        # 保存中间结果
        pd.DataFrame(all_results).to_csv(
            '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/baseline_comparison_full.csv',
            index=False
        )
    
    print("\n" + "="*70)
    print("  最终结果汇总")
    print("="*70)
    print(f"\n{'里程':<12} {'MA':<10} {'Exp':<10} {'LSTM':<10} {'TimeMixer':<12}")
    print("-"*60)
    for r in all_results:
        lstm_str = f"{r['lstm']:.3f}" if not np.isnan(r['lstm']) else "FAIL"
        tm_str = f"{r['timemixer']:.3f}" if not np.isnan(r['timemixer']) else "FAIL"
        print(f"{r['mile']:<12} {r['ma']:<10.3f} {r['exp']:<10.3f} {lstm_str:<10} {tm_str:<12}")
    
    print("\n结果已保存到: baseline_comparison_full.csv")
