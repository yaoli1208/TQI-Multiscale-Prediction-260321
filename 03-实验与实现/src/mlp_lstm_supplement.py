#!/usr/bin/env python3
"""
MLP/LSTM 补充实验 - 6个精选样本
样本: 709400, 734400, 739400, 746400, 747400, 749400
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/processed/cleaned_data_v3.csv'

# 目标样本
TARGET_MILES = [709400, 734400, 739400, 746400, 747400, 749400]

def load_sample_data(df, mile):
    """加载指定样本数据"""
    sample_df = df[df['mile'] == mile].copy()
    sample_df = sample_df.sort_values('date').reset_index(drop=True)
    return sample_df[['date', 'tqi_value']].copy()

def split_data(df):
    """70% 训练 / 15% 验证 / 15% 测试"""
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy()
    )

def create_sequences(data, lookback=12):
    """创建时间序列样本"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def mlp_baseline(train_df, test_df, lookback=12):
    """MLP神经网络基线"""
    try:
        train_values = train_df['tqi_value'].values
        test_values = test_df['tqi_value'].values
        
        # 创建训练序列
        X_train, y_train = create_sequences(train_values, lookback)
        
        if len(X_train) < 10:
            return None, None
        
        # 标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # 训练MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train_scaled)
        
        # 预测测试集
        # 使用训练集最后lookback个点作为初始输入
        predictions = []
        current_seq = train_values[-lookback:].copy()
        
        for _ in range(len(test_values)):
            X_pred = scaler_X.transform(current_seq.reshape(1, -1))
            y_pred_scaled = mlp.predict(X_pred)[0]
            y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0][0]
            predictions.append(y_pred)
            # 滑动窗口更新
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = y_pred
        
        predictions = np.array(predictions)
        errors = test_values - predictions
        return np.mean(np.abs(errors)), np.std(errors)
    except Exception as e:
        print(f"  MLP错误: {e}")
        return None, None

def lstm_baseline(train_df, test_df, lookback=12):
    """LSTM神经网络基线"""
    try:
        train_values = train_df['tqi_value'].values
        test_values = test_df['tqi_value'].values
        
        # 创建训练序列
        X_train, y_train = create_sequences(train_values, lookback)
        
        if len(X_train) < 10:
            return None, None
        
        # 尝试导入tensorflow，如果没有则返回None
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            print("  TensorFlow未安装，跳过LSTM")
            return None, None
        
        # 标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        
        # 重塑为LSTM输入格式 (samples, timesteps, features)
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        
        # 构建LSTM模型
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(30, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # 早停
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # 训练
        model.fit(
            X_train_lstm, y_train_scaled,
            epochs=100,
            batch_size=16,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        # 预测测试集
        predictions = []
        current_seq = train_values[-lookback:].copy()
        
        for _ in range(len(test_values)):
            X_pred = scaler_X.transform(current_seq.reshape(1, -1))
            X_pred_lstm = X_pred.reshape((1, lookback, 1))
            y_pred_scaled = model.predict(X_pred_lstm, verbose=0)[0][0]
            y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0][0]
            predictions.append(y_pred)
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = y_pred
        
        predictions = np.array(predictions)
        errors = test_values - predictions
        return np.mean(np.abs(errors)), np.std(errors)
    except Exception as e:
        print(f"  LSTM错误: {e}")
        return None, None

def historical_mean_baseline(train_df, test_df):
    """历史均值基线"""
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred)), np.std(y_true - y_pred)

def main():
    print("="*80)
    print("MLP/LSTM 补充实验")
    print("="*80)
    print(f"样本: {TARGET_MILES}")
    print(f"数据: {DATA_FILE}")
    print("="*80)
    
    # 加载清洗后的数据
    print("\n[1] 加载数据...")
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  总记录数: {len(df)}")
    print(f"  样本数: {df['mile'].nunique()}")
    
    # 存储结果
    results = []
    
    print("\n[2] 开始实验...")
    print(f"{'里程':<10} {'记录数':<8} {'历史均值MAE':<12} {'MLP_MAE':<12} {'LSTM_MAE':<12}")
    print("-"*60)
    
    for mile in TARGET_MILES:
        try:
            sample_df = load_sample_data(df, mile)
            if len(sample_df) < 50:
                print(f"{mile:<10} {len(sample_df):<8} 数据不足")
                continue
            
            train_df, val_df, test_df = split_data(sample_df)
            
            # 历史均值
            hm_mae, _ = historical_mean_baseline(train_df, test_df)
            
            # MLP
            mlp_mae, _ = mlp_baseline(train_df, test_df)
            mlp_str = f"{mlp_mae:.4f}" if mlp_mae else "N/A"
            
            # LSTM
            lstm_mae, _ = lstm_baseline(train_df, test_df)
            lstm_str = f"{lstm_mae:.4f}" if lstm_mae else "N/A"
            
            print(f"{mile:<10} {len(sample_df):<8} {hm_mae:<12.4f} {mlp_str:<12} {lstm_str:<12}")
            
            results.append({
                'mile': mile,
                'records': len(sample_df),
                'historical_mean': hm_mae,
                'mlp': mlp_mae,
                'lstm': lstm_mae
            })
            
        except Exception as e:
            print(f"{mile:<10} 错误: {e}")
    
    # 汇总
    print("\n" + "="*80)
    print("结果汇总")
    print("="*80)
    
    valid_results = [r for r in results if r['mlp'] is not None]
    if valid_results:
        hm_avg = np.mean([r['historical_mean'] for r in valid_results])
        mlp_avg = np.mean([r['mlp'] for r in valid_results])
        
        print(f"\n平均MAE:")
        print(f"  历史均值: {hm_avg:.4f}")
        print(f"  MLP:      {mlp_avg:.4f}")
        if any(r['lstm'] for r in valid_results):
            lstm_avg = np.mean([r['lstm'] for r in valid_results if r['lstm']])
            print(f"  LSTM:     {lstm_avg:.4f}")
        
        # 胜率
        mlp_wins = sum(1 for r in valid_results if r['mlp'] < r['historical_mean'])
        print(f"\nMLP vs 历史均值: {mlp_wins}/{len(valid_results)} 胜")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    output_file = f'{BASE_DIR}/results/mlp_lstm_supplement_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")
    
    print("="*80)

if __name__ == '__main__':
    main()
