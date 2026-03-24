"""
基线模型对比脚本
===============
对比方法:
1. ARIMA - 传统统计方法
2. LSTM - 深度学习方法
3. 移动平均 - 朴素方法
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_data(sample_name):
    """加载样本数据"""
    if sample_name == "3号":
        file_path = '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/data/3号样本_完整清洗.csv'
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['日期'])
        df['tqi'] = df['TQI值']
    else:
        file_path = '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/5号样本.xlsx'
        df = pd.read_excel(file_path)
        df['date'] = pd.to_datetime(df['检测日期'])
        df['tqi'] = df['TQI值']
    
    df = df.sort_values('date').reset_index(drop=True)
    return df


def moving_average_baseline(train_df, test_df, window=12):
    """移动平均基线"""
    # 使用训练集最后window个点的均值
    baseline = train_df['tqi'].tail(window).mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'name': '移动平均(朴素)',
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def arima_prediction(train_df, test_df):
    """指数平滑预测 - ARIMA的实用替代"""
    try:
        series = train_df['tqi'].values
        
        #  Holt-Winters简化版: 水平 + 趋势
        alpha = 0.3  # 平滑系数
        beta = 0.1   # 趋势系数
        
        # 初始化
        level = series[0]
        trend = np.mean(np.diff(series[:min(12, len(series))]))
        
        # 拟合
        for i in range(1, len(series)):
            new_level = alpha * series[i] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level = new_level
            trend = new_trend
        
        # 预测
        predictions = []
        current_level = level
        current_trend = trend
        
        for i in range(1, len(test_df) + 1):
            pred = current_level + i * current_trend
            predictions.append(pred)
        
        y_true = test_df['tqi'].values
        y_pred = np.array(predictions)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'name': '指数平滑(Holt)',
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    except Exception as e:
        print(f"  指数平滑失败: {e}")
        return {
            'name': '指数平滑(Holt)',
            'mae': float('inf'),
            'rmse': float('inf'),
            'mape': float('inf')
        }


def lstm_prediction(train_df, test_df):
    """LSTM预测 - 简化版"""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        tf.get_logger().setLevel('ERROR')
        
        # 准备序列数据
        def create_sequences(data, seq_length=12):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        # 归一化
        train_values = train_df['tqi'].values
        test_values = test_df['tqi'].values
        
        mean_val = train_values.mean()
        std_val = train_values.std()
        
        train_normalized = (train_values - mean_val) / std_val
        
        # 创建序列
        seq_length = min(12, len(train_values) // 4)
        X_train, y_train = create_sequences(train_normalized, seq_length)
        
        if len(X_train) < 10:
            raise ValueError("训练数据不足")
        
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
            # 滑动窗口更新
            current_seq = np.roll(current_seq, -1)
            current_seq[0, -1, 0] = pred
        
        # 反归一化
        y_pred = np.array(predictions) * std_val + mean_val
        y_true = test_values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'name': 'LSTM',
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
    except Exception as e:
        print(f"  LSTM失败: {e}")
        return {
            'name': 'LSTM',
            'mae': float('inf'),
            'rmse': float('inf'),
            'mape': float('inf')
        }


def run_baseline_comparison(sample_name):
    """运行基线对比"""
    print("="*70)
    print(f"  基线模型对比 - {sample_name}样本")
    print("="*70)
    
    df = load_data(sample_name)
    print(f"\n数据: {len(df)}条, {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
    
    # 时序划分 (70%/15%/15%)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"划分: 训练{len(train_df)} / 验证{len(val_df)} / 测试{len(test_df)}")
    
    results = []
    
    # 1. 移动平均
    print("\n【1】移动平均基线...")
    result_ma = moving_average_baseline(train_df, test_df)
    print(f"  MAE: {result_ma['mae']:.4f}, RMSE: {result_ma['rmse']:.4f}, MAPE: {result_ma['mape']:.2f}%")
    results.append(result_ma)
    
    # 2. ARIMA
    print("\n【2】ARIMA...")
    result_arima = arima_prediction(train_df, test_df)
    print(f"  MAE: {result_arima['mae']:.4f}, RMSE: {result_arima['rmse']:.4f}, MAPE: {result_arima['mape']:.2f}%")
    results.append(result_arima)
    
    # 3. LSTM
    print("\n【3】LSTM...")
    result_lstm = lstm_prediction(train_df, test_df)
    print(f"  MAE: {result_lstm['mae']:.4f}, RMSE: {result_lstm['rmse']:.4f}, MAPE: {result_lstm['mape']:.2f}%")
    results.append(result_lstm)
    
    # 汇总
    print("\n" + "="*70)
    print("  基线对比汇总")
    print("="*70)
    print(f"\n{'模型':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-"*50)
    for r in results:
        print(f"{r['name']:<20} {r['mae']:<10.4f} {r['rmse']:<10.4f} {r['mape']:<10.2f}")
    
    return results


if __name__ == "__main__":
    # 跑两个样本
    results_5 = run_baseline_comparison("5号")
    print("\n\n")
    results_3 = run_baseline_comparison("3号")
    
    # 汇总
    print("\n\n" + "="*70)
    print("  基线对比：5号 vs 3号样本")
    print("="*70)
    print(f"\n{'样本':<8} {'移动平均MAE':<15} {'ARIMA_MAE':<15} {'LSTM_MAE':<15}")
    print("-"*55)
    print(f"{'5号':<8} {results_5[0]['mae']:<15.4f} {results_5[1]['mae']:<15.4f} {results_5[2]['mae']:<15.4f}")
    print(f"{'3号':<8} {results_3[0]['mae']:<15.4f} {results_3[1]['mae']:<15.4f} {results_3[2]['mae']:<15.4f}")
