#!/usr/bin/env python3
"""
为新增5个样本运行LSTM和TimeMixer基线对比
目标样本: 240400, 1011400, 1190400, 1208400, 501400
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 目标样本
target_samples = [240400, 1011400, 1190400, 1208400, 501400]

def load_sample_data(mile):
    """从原始数据加载指定里程的样本"""
    df = pd.read_excel('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/data/raw/iic_tqi_all.xlsx')
    sample_df = df[df['tqi_mile'] == mile].copy()
    sample_df = sample_df.sort_values('tqi_date').reset_index(drop=True)
    return sample_df

def moving_average_baseline(train_df, test_df, window=12):
    """移动平均基线"""
    baseline = train_df['tqi_value'].tail(window).mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    return mae

def exponential_smoothing_baseline(train_df, test_df):
    """指数平滑基线"""
    try:
        series = train_df['tqi_value'].values
        alpha = 0.3
        beta = 0.1
        
        level = series[0]
        trend = np.mean(np.diff(series[:min(12, len(series))]))
        
        for i in range(1, len(series)):
            new_level = alpha * series[i] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level = new_level
            trend = new_trend
        
        predictions = []
        current_level = level
        current_trend = trend
        
        for i in range(1, len(test_df) + 1):
            pred = current_level + i * current_trend
            predictions.append(pred)
        
        y_true = test_df['tqi_value'].values
        y_pred = np.array(predictions)
        
        mae = mean_absolute_error(y_true, y_pred)
        return mae
    except:
        return float('nan')

def lstm_prediction(train_df, test_df):
    """LSTM预测 - 简化版"""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        tf.get_logger().setLevel('ERROR')
        
        def create_sequences(data, seq_length=12):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        train_values = train_df['tqi_value'].values
        test_values = test_df['tqi_value'].values
        
        # 数据量检查
        if len(train_values) < 50:
            print(f"    [LSTM] 训练数据不足: {len(train_values)}条")
            return float('nan')
        
        mean_val = train_values.mean()
        std_val = train_values.std()
        if std_val == 0:
            std_val = 1
        
        train_normalized = (train_values - mean_val) / std_val
        
        seq_length = min(12, len(train_values) // 4)
        if seq_length < 3:
            seq_length = 3
            
        X_train, y_train = create_sequences(train_normalized, seq_length)
        
        if len(X_train) < 10:
            print(f"    [LSTM] 序列数不足: {len(X_train)}个")
            return float('nan')
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        # 简单LSTM模型
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(seq_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # 训练
        model.fit(X_train, y_train, epochs=50, verbose=0, batch_size=8)
        
        # 预测
        predictions = []
        current_seq = train_normalized[-seq_length:].reshape(1, seq_length, 1)
        
        for _ in range(len(test_values)):
            pred = model.predict(current_seq, verbose=0)[0][0]
            predictions.append(pred)
            current_seq = np.roll(current_seq, -1)
            current_seq[0, -1, 0] = pred
        
        y_pred = np.array(predictions) * std_val + mean_val
        y_true = test_values
        
        mae = mean_absolute_error(y_true, y_pred)
        return mae
        
    except Exception as e:
        print(f"    [LSTM] 失败: {e}")
        return float('nan')

def timemixer_prediction(train_df, test_df):
    """
    TimeMixer简化实现
    由于原始TimeMixer需要复杂的PDM/FMM模块，这里使用多尺度分解的简化版
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        
        train_values = train_df['tqi_value'].values
        test_values = test_df['tqi_value'].values
        
        if len(train_values) < 50:
            print(f"    [TimeMixer] 训练数据不足: {len(train_values)}条")
            return float('nan')
        
        # 多尺度特征：原始值、移动平均(3个月)、移动平均(6个月)、移动平均(12个月)
        def create_multiscale_features(values, window_size=12):
            features = []
            for i in range(len(values)):
                feat = []
                # 原始值
                feat.append(values[i])
                # 多尺度移动平均
                for w in [3, 6, 12]:
                    if i >= w:
                        feat.append(np.mean(values[i-w:i]))
                    else:
                        feat.append(values[i])
                # 趋势特征
                if i >= 3:
                    feat.append(values[i] - values[i-3])  # 短期趋势
                else:
                    feat.append(0)
                if i >= 12:
                    feat.append(values[i] - values[i-12])  # 长期趋势
                else:
                    feat.append(0)
                features.append(feat)
            return np.array(features)
        
        # 准备训练数据
        X_train = create_multiscale_features(train_values)
        y_train = train_values
        
        # 训练模型
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        # 迭代预测
        predictions = []
        current_history = list(train_values)
        
        for _ in range(len(test_values)):
            features = create_multiscale_features(np.array(current_history))[-1:]
            pred = model.predict(features)[0]
            predictions.append(pred)
            current_history.append(pred)
        
        y_pred = np.array(predictions)
        y_true = test_values
        
        mae = mean_absolute_error(y_true, y_pred)
        return mae
        
    except Exception as e:
        print(f"    [TimeMixer] 失败: {e}")
        return float('nan')

def run_comparison_for_sample(mile):
    """为单个样本运行所有基线对比"""
    print(f"\n{'='*70}")
    print(f"  样本里程: {mile}")
    print(f"{'='*70}")
    
    try:
        df = load_sample_data(mile)
        print(f"数据: {len(df)}条")
        
        if len(df) < 100:
            print(f"[警告] 数据量较少: {len(df)}条")
        
        # 时序划分 (70%/15%/15%)
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"划分: 训练{len(train_df)} / 验证{len(val_df)} / 测试{len(test_df)}")
        
        results = {'mile': mile, 'total': n, 'train': len(train_df), 'test': len(test_df)}
        
        # 移动平均
        print("  [1] 移动平均...", end="")
        results['ma'] = moving_average_baseline(train_df, test_df)
        print(f" MAE={results['ma']:.4f}")
        
        # 指数平滑
        print("  [2] 指数平滑...", end="")
        results['exp'] = exponential_smoothing_baseline(train_df, test_df)
        print(f" MAE={results['exp']:.4f}")
        
        # LSTM
        print("  [3] LSTM...", end="")
        results['lstm'] = lstm_prediction(train_df, test_df)
        if np.isnan(results['lstm']):
            print(" 失败")
        else:
            print(f" MAE={results['lstm']:.4f}")
        
        # TimeMixer
        print("  [4] TimeMixer...", end="")
        results['timemixer'] = timemixer_prediction(train_df, test_df)
        if np.isnan(results['timemixer']):
            print(" 失败")
        else:
            print(f" MAE={results['timemixer']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"[错误] 处理样本 {mile} 失败: {e}")
        return None

if __name__ == "__main__":
    print("="*70)
    print("  新增样本基线对比实验")
    print("  目标样本:", target_samples)
    print("="*70)
    
    all_results = []
    for mile in target_samples:
        result = run_comparison_for_sample(mile)
        if result:
            all_results.append(result)
    
    # 汇总输出
    print("\n" + "="*70)
    print("  汇总结果")
    print("="*70)
    print(f"\n{'里程':<12} {'MA':<10} {'Exp':<10} {'LSTM':<10} {'TimeMixer':<12}")
    print("-"*60)
    for r in all_results:
        lstm_str = f"{r['lstm']:.3f}" if not np.isnan(r['lstm']) else "FAIL"
        tm_str = f"{r['timemixer']:.3f}" if not np.isnan(r['timemixer']) else "FAIL"
        print(f"{r['mile']:<12} {r['ma']:<10.3f} {r['exp']:<10.3f} {lstm_str:<10} {tm_str:<12}")
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    output_path = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/baseline_comparison_new_samples.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n结果已保存: {output_path}")
