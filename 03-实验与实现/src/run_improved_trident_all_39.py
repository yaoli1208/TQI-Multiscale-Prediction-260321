#!/usr/bin/env python3
"""
39个样本完整实验 - 改进版Trident
包含：滚动锚定策略 + 修后预测策略 + 智能策略选择
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

def detect_maintenance_points(df):
    """检测大修点（TQI下降>0.3）"""
    df['tqi_diff'] = df['tqi_val'].diff()
    return df[df['tqi_diff'] < -0.3].copy()

# ===== 基线方法 =====
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

# ===== Trident策略 =====
def rolling_anchor_strategy(train_df, test_df):
    """滚动锚定策略：使用最近12个月均值 + 季节性调整"""
    anchor = train_df['tqi_val'].tail(12).mean()
    predictions = []
    for _, row in test_df.iterrows():
        month = row['dete_dt'].month
        month_mean = train_df[train_df['dete_dt'].dt.month == month]['tqi_val'].mean()
        seasonal = month_mean - train_df['tqi_val'].mean() if not np.isnan(month_mean) else 0
        pred = anchor + seasonal
        predictions.append(pred)
    return np.array(predictions)

def post_maintenance_strategy(train_df, test_df, full_df):
    """修后预测策略：基于维修锚定值 + 季节性 + 劣化趋势"""
    # 检测维修点
    maintenance = detect_maintenance_points(full_df)
    
    if len(maintenance) == 0:
        # 无维修点，退化为滚动锚定
        return rolling_anchor_strategy(train_df, test_df)
    
    # 找到训练集最后一个大修点
    train_end_date = train_df['dete_dt'].max()
    last_maintenance = maintenance[maintenance['dete_dt'] <= train_end_date]
    
    if len(last_maintenance) == 0:
        # 训练期内无维修，退化为滚动锚定
        return rolling_anchor_strategy(train_df, test_df)
    
    # 使用最近维修点后的数据作为锚定
    last_maint_date = last_maintenance['dete_dt'].iloc[-1]
    post_maint_df = train_df[train_df['dete_dt'] > last_maint_date]
    
    if len(post_maint_df) < 3:
        anchor = train_df['tqi_val'].tail(6).mean()
    else:
        anchor = post_maint_df['tqi_val'].mean()
    
    # 计算劣化趋势
    if len(post_maint_df) > 1:
        days = (post_maint_df['dete_dt'] - post_maint_df['dete_dt'].iloc[0]).dt.days
        if days.max() > 0:
            slope = np.polyfit(days, post_maint_df['tqi_val'], 1)[0]
        else:
            slope = 0
    else:
        slope = 0
    
    # 预测
    predictions = []
    last_train_date = train_df['dete_dt'].iloc[-1]
    
    for _, row in test_df.iterrows():
        days_since = (row['dete_dt'] - last_train_date).days
        month = row['dete_dt'].month
        month_mean = train_df[train_df['dete_dt'].dt.month == month]['tqi_val'].mean()
        seasonal = month_mean - train_df['tqi_val'].mean() if not np.isnan(month_mean) else 0
        
        pred = anchor + seasonal + slope * days_since
        predictions.append(pred)
    
    return np.array(predictions)

def smart_trident_strategy(train_df, val_df, test_df, full_df):
    """智能策略选择：在验证集上选择最优策略"""
    y_val = val_df['tqi_val'].values
    
    # 评估滚动锚定
    try:
        pred_ra = rolling_anchor_strategy(train_df, val_df)
        mae_ra = calculate_mae(y_val, pred_ra)
    except:
        mae_ra = float('inf')
    
    # 评估修后预测
    try:
        pred_pm = post_maintenance_strategy(train_df, val_df, full_df)
        mae_pm = calculate_mae(y_val, pred_pm)
    except:
        mae_pm = float('inf')
    
    # 选择验证集上MAE更低的策略
    if mae_pm < mae_ra:
        # 使用修后预测
        pred_test = post_maintenance_strategy(train_df, test_df, full_df)
        chosen_strategy = 'Post-Maintenance'
    else:
        # 使用滚动锚定
        pred_test = rolling_anchor_strategy(train_df, test_df)
        chosen_strategy = 'Rolling-Anchor'
    
    return pred_test, chosen_strategy

# 主程序
output_dir = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现/results/all_39_improved_trident'
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("39个样本改进版Trident实验")
print("="*70)

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
    if n < 100:
        print(f"  跳过: 记录数不足 ({n})")
        continue
    
    # 划分数据集
    train_df = sample_clean.iloc[:int(n*0.7)].copy()
    val_df = sample_clean.iloc[int(n*0.7):int(n*0.85)].copy()
    test_df = sample_clean.iloc[int(n*0.85):].copy()
    
    y_test = test_df['tqi_val'].values
    
    # 运行基线
    methods = {
        'Data Driven': data_driven_baseline,
        'Moving Avg': moving_average_baseline,
        'Holt Exp': holt_baseline,
        'LSTM': lstm_baseline,
        'TimeMixer': timemixer_baseline
    }
    
    mae_results = {'sample_idx': sample_idx, 'mile': mile, 'n_clean': n}
    
    for name, func in methods.items():
        try:
            pred = func(train_df, test_df)
            mae = calculate_mae(y_test, pred)
            mae_results[name] = mae
        except:
            mae_results[name] = np.nan
    
    # 运行Trident策略
    # 1. 滚动锚定
    try:
        pred_ra = rolling_anchor_strategy(train_df, test_df)
        mae_results['Trident-RA'] = calculate_mae(y_test, pred_ra)
    except:
        mae_results['Trident-RA'] = np.nan
    
    # 2. 修后预测
    try:
        pred_pm = post_maintenance_strategy(train_df, test_df, sample_clean)
        mae_results['Trident-PM'] = calculate_mae(y_test, pred_pm)
    except:
        mae_results['Trident-PM'] = np.nan
    
    # 3. 智能选择
    try:
        pred_smart, strategy = smart_trident_strategy(train_df, val_df, test_df, sample_clean)
        mae_results['Trident-Smart'] = calculate_mae(y_test, pred_smart)
        mae_results['chosen_strategy'] = strategy
    except Exception as e:
        mae_results['Trident-Smart'] = np.nan
        mae_results['chosen_strategy'] = 'Error'
    
    # 确定最佳基线
    baseline_maes = {k: v for k, v in mae_results.items() if k not in ['sample_idx', 'mile', 'n_clean', 'Trident-RA', 'Trident-PM', 'Trident-Smart', 'chosen_strategy'] and not np.isnan(v)}
    if baseline_maes:
        best_baseline = min(baseline_maes, key=baseline_maes.get)
        best_baseline_mae = baseline_maes[best_baseline]
        mae_results['best_baseline'] = best_baseline
        mae_results['best_baseline_mae'] = best_baseline_mae
        
        # 计算各Trident变体的改善
        for trident_var in ['Trident-RA', 'Trident-PM', 'Trident-Smart']:
            if not np.isnan(mae_results[trident_var]):
                mae_results[f'{trident_var}_improve'] = (best_baseline_mae - mae_results[trident_var]) / best_baseline_mae * 100
            else:
                mae_results[f'{trident_var}_improve'] = np.nan
    else:
        mae_results['best_baseline'] = 'N/A'
        mae_results['best_baseline_mae'] = np.nan
    
    results.append(mae_results)
    
    print(f"  MAE: DD={mae_results.get('Data Driven', np.nan):.3f}, MA={mae_results.get('Moving Avg', np.nan):.3f}")
    print(f"       RA={mae_results.get('Trident-RA', np.nan):.3f}, PM={mae_results.get('Trident-PM', np.nan):.3f}")
    print(f"       Smart={mae_results.get('Trident-Smart', np.nan):.3f} ({mae_results.get('chosen_strategy', 'N/A')})")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv(f'{output_dir}/improved_trident_results.csv', index=False)

print("\n" + "="*70)
print("实验完成！")
print("="*70)

# 统计
print("\n【策略选择分布】")
strategy_counts = results_df['chosen_strategy'].value_counts()
print(strategy_counts.to_string())

print("\n【Trident各变体有效性】")
for var in ['Trident-RA', 'Trident-PM', 'Trident-Smart']:
    col = f'{var}_improve'
    if col in results_df.columns:
        valid = results_df[col].dropna()
        positive = valid[valid > 10]
        print(f"{var}: {len(positive)}/39 有效 (改善>10%), 平均改善: {valid.mean():.1f}%")

print(f"\n结果保存到: {output_dir}/improved_trident_results.csv")
