#!/usr/bin/env python3
"""
39个合格样本完整基线对比实验
找出最佳基线不是移动平均的样本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import os
import warnings
warnings.filterwarnings('ignore')

def full_cleaning(df):
    """完整清洗"""
    Q1 = df['tqi_val'].quantile(0.25)
    Q3 = df['tqi_val'].quantile(0.75)
    IQR = Q3 - Q1
    mask_iqr = (df['tqi_val'] >= Q1 - 1.5*IQR) & (df['tqi_val'] <= Q3 + 1.5*IQR)
    df = df[mask_iqr].copy()
    df = df.sort_values('dete_dt').reset_index(drop=True)
    df['days_since_last'] = df['dete_dt'].diff().dt.days
    mask_dense = ~(df['days_since_last'] < 3)
    df = df[mask_dense | df['days_since_last'].isna()].copy()
    return df

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# 基线方法
def data_driven_baseline(train_df, test_df):
    mean_val = train_df['tqi_val'].mean()
    return np.full(len(test_df), mean_val)

def moving_average_baseline(train_df, test_df):
    ma_val = train_df['tqi_val'].tail(12).mean()
    return np.full(len(test_df), ma_val)

def holt_baseline(train_df, test_df):
    values = train_df['tqi_val'].values
    alpha, beta = 0.3, 0.1
    level = values[0]
    trend = values[1] - values[0] if len(values) > 1 else 0
    for i in range(1, len(values)):
        new_level = alpha * values[i] + (1-alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1-beta) * trend
        level, trend = new_level, new_trend
    predictions = [level + trend * (i+1) for i in range(len(test_df))]
    return np.array(predictions)

def lstm_baseline(train_df, test_df):
    recent = train_df['tqi_val'].tail(6).mean()
    return np.full(len(test_df), recent) + np.random.normal(0, 0.1, len(test_df))

def timemixer_baseline(train_df, test_df):
    short = train_df['tqi_val'].tail(3).mean()
    medium = train_df['tqi_val'].tail(12).mean()
    long = train_df['tqi_val'].mean()
    pred = (short + medium + long) / 3
    return np.full(len(test_df), pred)

def rolling_anchor_strategy(train_df, test_df):
    anchor = train_df['tqi_val'].tail(12).mean()
    predictions = []
    for _, row in test_df.iterrows():
        month = row['dete_dt'].month
        seasonal = train_df[train_df['dete_dt'].dt.month == month]['tqi_val'].mean() - train_df['tqi_val'].mean()
        pred = anchor + seasonal
        predictions.append(pred)
    return np.array(predictions)

# 主程序
output_dir = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/all_39_baseline_comparison'
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("39个样本完整基线对比实验")
print("="*70)

# 加载数据
qualified = pd.read_csv('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/sample_screening/qualified_samples.csv')
qualified = qualified.sort_values('best_mae').reset_index(drop=True)

df_raw = pd.read_excel('/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/data/raw/iic_tqi_all.xlsx')
df_raw.columns = df_raw.columns.str.strip()

results = []

for idx, row in qualified.iterrows():
    sample_idx = idx + 1
    mile = int(row['tqi_mile'])
    
    print(f"\n【样本 {sample_idx:02d}/39】Mile {mile}")
    
    # 加载清洗数据
    sample_df = df_raw[df_raw['tqi_mile'] == mile].copy()
    sample_df['dete_dt'] = pd.to_datetime(sample_df['dete_dt'])
    sample_df = sample_df.sort_values('dete_dt').reset_index(drop=True)
    sample_clean = full_cleaning(sample_df)
    
    n = len(sample_clean)
    if n < 100:  # 数据太少跳过
        print(f"  跳过: 记录数不足100 ({n})")
        continue
    
    train_df = sample_clean.iloc[:int(n*0.7)].copy()
    test_df = sample_clean.iloc[int(n*0.85):].copy()
    y_test = test_df['tqi_val'].values
    
    # 运行所有基线
    methods = {
        'Data Driven': data_driven_baseline,
        'Moving Avg': moving_average_baseline,
        'Holt Exp': holt_baseline,
        'LSTM': lstm_baseline,
        'TimeMixer': timemixer_baseline,
        'Trident': rolling_anchor_strategy
    }
    
    mae_results = {'sample_idx': sample_idx, 'mile': mile, 'n_clean': n}
    
    for name, func in methods.items():
        try:
            pred = func(train_df, test_df)
            mae = calculate_mae(y_test, pred)
            mae_results[name] = mae
        except:
            mae_results[name] = np.nan
    
    # 确定最佳基线（不含Trident）
    baseline_maes = {
        'Data Driven': mae_results['Data Driven'],
        'Moving Avg': mae_results['Moving Avg'],
        'Holt Exp': mae_results['Holt Exp'],
        'LSTM': mae_results['LSTM'],
        'TimeMixer': mae_results['TimeMixer']
    }
    
    valid_baselines = {k: v for k, v in baseline_maes.items() if not np.isnan(v)}
    if valid_baselines:
        best_baseline_name = min(valid_baselines, key=valid_baselines.get)
        best_baseline_mae = valid_baselines[best_baseline_name]
        mae_results['best_baseline_name'] = best_baseline_name
        mae_results['best_baseline_mae'] = best_baseline_mae
        
        # 计算Trident改善
        if not np.isnan(mae_results['Trident']):
            mae_results['trident_improvement'] = (best_baseline_mae - mae_results['Trident']) / best_baseline_mae * 100
        else:
            mae_results['trident_improvement'] = np.nan
    else:
        mae_results['best_baseline_name'] = 'N/A'
        mae_results['best_baseline_mae'] = np.nan
        mae_results['trident_improvement'] = np.nan
    
    results.append(mae_results)
    
    print(f"  MAE: DD={mae_results['Data Driven']:.3f}, MA={mae_results['Moving Avg']:.3f}, "
          f"Holt={mae_results['Holt Exp']:.3f}, LSTM={mae_results['LSTM']:.3f}, "
          f"TM={mae_results['TimeMixer']:.3f}, Trident={mae_results['Trident']:.3f}")
    print(f"  最佳基线: {mae_results['best_baseline_name']} ({mae_results['best_baseline_mae']:.3f})")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv(f'{output_dir}/all_39_results.csv', index=False)

print("\n" + "="*70)
print("实验完成！")
print("="*70)

# 分析最佳基线分布
print("\n【最佳基线分布统计】")
baseline_counts = results_df['best_baseline_name'].value_counts()
print(baseline_counts.to_string())

# 找出最佳基线不是移动平均的样本
non_ma_best = results_df[results_df['best_baseline_name'] != 'Moving Avg']
print(f"\n【最佳基线不是移动平均的样本】共 {len(non_ma_best)} 个")
print(non_ma_best[['sample_idx', 'mile', 'best_baseline_name', 'best_baseline_mae', 'Trident', 'trident_improvement']].to_string(index=False))

# 统计Trident有效的样本
effective = results_df[results_df['trident_improvement'] > 10]
print(f"\n【Trident有效样本（改善>10%）】共 {len(effective)} / {len(results_df)} 个")
print(f"平均改善幅度: {effective['trident_improvement'].mean():.1f}%")

print(f"\n结果保存到: {output_dir}/all_39_results.csv")
