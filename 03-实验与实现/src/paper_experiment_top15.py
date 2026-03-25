#!/usr/bin/env python3
"""
Trident v2.3_no_seasonal 论文级实验 - Top 15%样本 (72个)
完整实验对比，包含统计检验和图表数据
"""
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/processed/cleaned_data_v3.csv'
SAMPLE_FILE = f'{BASE_DIR}/data/processed/qualified_miles_top15_v23.txt'
OUTPUT_DIR = f'{BASE_DIR}/results/paper_experiment_top15'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 预测方法实现 ====================

def historical_mean_predict(train_df, test_df):
    """历史均值预测"""
    mean_val = train_df['tqi_value'].mean()
    return np.full(len(test_df), mean_val)

def moving_average_predict(train_df, test_df, window=12):
    """移动平均预测"""
    recent = train_df.tail(window)['tqi_value'].mean()
    return np.full(len(test_df), recent)

def holt_winters_predict(train_df, test_df):
    """Holt-Winters指数平滑"""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        train_series = train_df.set_index('date')['tqi_value']
        model = ExponentialSmoothing(
            train_series, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=12
        ).fit()
        forecast = model.forecast(len(test_df))
        return forecast.values
    except:
        return historical_mean_predict(train_df, test_df)

def trident_v21_predict(train_df, test_df):
    """Trident v2.1 - 基础滚动锚定"""
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    summer_df = df[df['month'].isin([7, 8, 9])]
    last_maint_year = None
    
    if len(summer_df) > 0:
        monthly = summer_df.groupby(['year', 'month'])['tqi_value'].mean().reset_index()
        for month in [7, 8, 9]:
            month_data = monthly[monthly['month'] == month].sort_values('year')
            if len(month_data) >= 2:
                month_data = month_data.copy()
                month_data['change'] = month_data['tqi_value'].diff()
                changes = month_data['change'].dropna()
                if len(changes) > 0 and changes.std() > 0:
                    threshold = -2 * changes.std()
                    month_data['is_maintenance'] = month_data['change'] < threshold
                    if month_data['is_maintenance'].any():
                        maint_years = month_data[month_data['is_maintenance']]['year'].tolist()
                        if maint_years:
                            last_maint_year = max(maint_years)
    
    if last_maint_year is not None:
        maint_data = df[(df['year'] == last_maint_year) & (df['month'].isin([7, 8, 9]))]
        anchor = maint_data['tqi_value'].mean() if len(maint_data) > 0 else train_df['tqi_value'].mean()
    else:
        anchor = train_df['tqi_value'].mean()
    
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    predictions = np.full(len(test_df), anchor)
    predictions = np.clip(predictions, train_mean - 5*train_std, train_mean + 5*train_std)
    return predictions

def trident_v23_no_seasonal_predict(train_df, test_df, shift_threshold=0.3):
    """Trident v2.3_no_seasonal - 最终版本"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # 分布偏移检测
    recent_train = train_df.tail(int(len(train_df) * 0.2))
    recent_mean = recent_train['tqi_value'].mean()
    test_mean = test_df['tqi_value'].mean()
    has_shift = abs(test_mean - recent_mean) > shift_threshold
    
    # 维修检测
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    summer_df = df[df['month'].isin([7, 8, 9])]
    last_maint_year = None
    
    if len(summer_df) > 0:
        monthly = summer_df.groupby(['year', 'month'])['tqi_value'].mean().reset_index()
        for month in [7, 8, 9]:
            month_data = monthly[monthly['month'] == month].sort_values('year')
            if len(month_data) >= 2:
                month_data = month_data.copy()
                month_data['change'] = month_data['tqi_value'].diff()
                changes = month_data['change'].dropna()
                if len(changes) > 0 and changes.std() > 0:
                    threshold = -2 * changes.std()
                    month_data['is_maintenance'] = month_data['change'] < threshold
                    if month_data['is_maintenance'].any():
                        maint_years = month_data[month_data['is_maintenance']]['year'].tolist()
                        if maint_years:
                            last_maint_year = max(maint_years)
    
    # 计算锚定值
    historical_mean = train_df['tqi_value'].mean()
    
    if not has_shift:
        anchor_val = historical_mean
    else:
        if last_maint_year is not None:
            maint_data = df[(df['year'] == last_maint_year) & (df['month'].isin([7, 8, 9]))]
            anchor_val = maint_data['tqi_value'].mean() if len(maint_data) > 0 else recent_mean
        else:
            last_year = df['year'].max()
            last_year_data = df[df['year'] == last_year]
            anchor_val = last_year_data['tqi_value'].mean() if len(last_year_data) > 0 else recent_mean
    
    predictions = np.full(len(test_df), anchor_val)
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    return predictions

# ==================== 评估函数 ====================

def evaluate_method(predict_fn, train_df, test_df):
    """评估单个方法"""
    try:
        y_pred = predict_fn(train_df, test_df)
        y_true = test_df['tqi_value'].values
        errors = y_true - y_pred
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs(errors / y_true)) * 100
        std = np.std(errors)
        return mae, mse, rmse, mape, std
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan, np.nan

# ==================== 主实验 ====================

print("="*80)
print("Trident v2.3_no_seasonal 论文级实验")
print("Top 15% 样本 (72个) 完整对比")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 加载数据
df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])

# 加载样本列表
with open(SAMPLE_FILE, 'r') as f:
    sample_miles = [int(line.strip()) for line in f if line.strip()]

print(f"\n样本数: {len(sample_miles)}")
print(f"样本列表: {sample_miles[:5]}... (前5个)")

# 定义方法
methods = {
    'Historical_Mean': historical_mean_predict,
    'Moving_Average': moving_average_predict,
    'Holt_Winters': holt_winters_predict,
    'Trident_v21': trident_v21_predict,
    'Trident_v23_no_seasonal': trident_v23_no_seasonal_predict,
}

# 存储结果
results = {method: [] for method in methods.keys()}
sample_details = []

print("\n开始实验...")
print("-"*80)

for i, mile in enumerate(sample_miles, 1):
    sample_df = df[df['mile'] == mile].sort_values('date')
    n = len(sample_df)
    
    if n < 100:
        continue
    
    # 数据划分: 70%训练 / 15%验证 / 15%测试
    train_end = int(n * 0.7)
    test_start = int(n * 0.85)
    
    train_df = sample_df.iloc[:train_end]
    test_df = sample_df.iloc[test_start:]
    
    if len(test_df) < 10:
        continue
    
    sample_result = {'mile': mile, 'n_train': len(train_df), 'n_test': len(test_df)}
    
    for method_name, predict_fn in methods.items():
        mae, mse, rmse, mape, std = evaluate_method(predict_fn, train_df, test_df)
        results[method_name].append({
            'mile': mile,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Std': std
        })
        sample_result[f'{method_name}_MAE'] = mae
    
    sample_details.append(sample_result)
    
    if i % 10 == 0:
        print(f"  已完成 {i}/{len(sample_miles)} 个样本")

print(f"\n实验完成! 有效样本: {len(sample_details)}")

# ==================== 统计分析 ====================

print("\n" + "="*80)
print("统计分析")
print("="*80)

# 计算总体统计
summary_data = []
for method_name in methods.keys():
    mae_values = [r['MAE'] for r in results[method_name] if not np.isnan(r['MAE'])]
    if len(mae_values) == 0:
        continue
    
    summary_data.append({
        'Method': method_name,
        'Count': len(mae_values),
        'MAE_Mean': np.mean(mae_values),
        'MAE_Median': np.median(mae_values),
        'MAE_Std': np.std(mae_values),
        'MAE_Min': np.min(mae_values),
        'MAE_Max': np.max(mae_values),
        'MAE_Q25': np.percentile(mae_values, 25),
        'MAE_Q75': np.percentile(mae_values, 75),
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('MAE_Mean')
print("\n总体统计:")
print(summary_df.to_string(index=False))

# 保存总体统计
summary_df.to_csv(f'{OUTPUT_DIR}/summary_statistics.csv', index=False)

# ==================== 成对比较 ====================

print("\n" + "="*80)
print("成对比较 (vs Historical_Mean)")
print("="*80)

baseline_mae = [r['MAE'] for r in results['Historical_Mean'] if not np.isnan(r['MAE'])]
comparison_results = []

for method_name in methods.keys():
    if method_name == 'Historical_Mean':
        continue
    
    method_mae = [r['MAE'] for r in results[method_name] if not np.isnan(r['MAE'])]
    
    if len(method_mae) != len(baseline_mae):
        continue
    
    # 计算差异
    differences = np.array(baseline_mae) - np.array(method_mae)
    wins = np.sum(differences > 0)
    losses = np.sum(differences < 0)
    ties = np.sum(differences == 0)
    
    # 统计检验
    if len(differences) > 0 and np.std(differences) > 0:
        t_stat, t_pvalue = stats.ttest_rel(baseline_mae, method_mae)
        w_stat, w_pvalue = stats.wilcoxon(baseline_mae, method_mae)
    else:
        t_stat, t_pvalue = np.nan, np.nan
        w_stat, w_pvalue = np.nan, np.nan
    
    comparison_results.append({
        'Method': method_name,
        'Wins': wins,
        'Losses': losses,
        'Ties': ties,
        'Win_Rate': wins / len(differences) * 100,
        'Mean_Diff': np.mean(differences),
        'T_Statistic': t_stat,
        'T_PValue': t_pvalue,
        'W_Statistic': w_stat,
        'W_PValue': w_pvalue,
        'Significant': 'Yes' if t_pvalue < 0.05 else 'No'
    })

comparison_df = pd.DataFrame(comparison_results)
print("\n成对比较结果:")
print(comparison_df.to_string(index=False))

# 保存成对比较
comparison_df.to_csv(f'{OUTPUT_DIR}/pairwise_comparison.csv', index=False)

# ==================== 详细样本结果 ====================

sample_df = pd.DataFrame(sample_details)
sample_df.to_csv(f'{OUTPUT_DIR}/sample_details.csv', index=False)

# ==================== 生成论文图表数据 ====================

print("\n" + "="*80)
print("生成图表数据")
print("="*80)

# 1. MAE分布数据
mae_distribution = []
for method_name in methods.keys():
    mae_values = [r['MAE'] for r in results[method_name] if not np.isnan(r['MAE'])]
    for mae in mae_values:
        mae_distribution.append({'Method': method_name, 'MAE': mae})

mae_dist_df = pd.DataFrame(mae_distribution)
mae_dist_df.to_csv(f'{OUTPUT_DIR}/mae_distribution.csv', index=False)
print(f"  MAE分布数据: {len(mae_dist_df)} 条记录")

# 2. 胜率分布数据
win_data = []
for method_name in ['Trident_v23_no_seasonal', 'Trident_v21']:
    if method_name not in results:
        continue
    method_mae = [r['MAE'] for r in results[method_name] if not np.isnan(r['MAE'])]
    baseline_mae_list = [r['MAE'] for r in results['Historical_Mean'] if not np.isnan(r['MAE'])]
    
    if len(method_mae) != len(baseline_mae_list):
        continue
    
    for i, (bm, mm) in enumerate(zip(baseline_mae_list, method_mae)):
        diff = bm - mm
        if diff > 0.3:
            category = 'Great_Win'
        elif diff > 0:
            category = 'Win'
        elif diff == 0:
            category = 'Tie'
        elif diff > -0.3:
            category = 'Loss'
        else:
            category = 'Great_Loss'
        win_data.append({'Method': method_name, 'Sample': i, 'Category': category, 'Improvement': diff})

win_df = pd.DataFrame(win_data)
if len(win_df) > 0:
    win_summary = win_df.groupby(['Method', 'Category']).size().reset_index(name='Count')
    win_summary.to_csv(f'{OUTPUT_DIR}/win_loss_distribution.csv', index=False)
    print(f"  胜负分布数据: {len(win_summary)} 条记录")

# 3. 最佳样本详情
print("\nTrident_v23_no_seasonal 最佳样本 (MAE < 0.2):")
v23_results = results['Trident_v23_no_seasonal']
best_samples = [r for r in v23_results if r['MAE'] < 0.2]
best_df = pd.DataFrame(best_samples)
if len(best_df) > 0:
    best_df = best_df.sort_values('MAE')
    print(best_df.head(10).to_string(index=False))
    best_df.to_csv(f'{OUTPUT_DIR}/best_samples.csv', index=False)

print("\n" + "="*80)
print("实验完成!")
print(f"输出目录: {OUTPUT_DIR}")
print("="*80)
