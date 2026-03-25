#!/usr/bin/env python3
"""
v2.3_no_seasonal 深度优化实验
包含：1. 截尾比例调优 2. 自适应混合 3. 全量486样本验证
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v3.txt'

def load_all_data():
    df = pd.read_excel(DATA_FILE)
    df.columns = df.columns.str.strip()
    df = df[df['dete_dt'].astype(str).str.strip() != 'dete_dt'].copy()
    df['date'] = pd.to_datetime(df['dete_dt'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['tqi_value'] = pd.to_numeric(df['tqi_val'], errors='coerce')
    df['tqi_mile'] = pd.to_numeric(df['tqi_mile'], errors='coerce')
    df = df.dropna(subset=['tqi_value', 'tqi_mile'])
    return df[['tqi_mile', 'date', 'tqi_value']].copy()

def load_sample_data(all_df, mile):
    sample_df = all_df[all_df['tqi_mile'] == mile].copy()
    sample_df = sample_df.sort_values('date').reset_index(drop=True)
    return sample_df[['date', 'tqi_value']].copy()

def split_data(df):
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy()
    )

def historical_mean_baseline(train_df, test_df):
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred)), np.std(y_true - y_pred)

def v23_no_seasonal(train_df, test_df):
    """v2.3_no_seasonal 基线"""
    SHIFT_THRESHOLD = 0.3
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    recent_train = train_df.tail(int(len(train_df) * 0.2))
    recent_mean = recent_train['tqi_value'].mean()
    test_mean = test_df['tqi_value'].mean()
    has_shift = abs(test_mean - recent_mean) > SHIFT_THRESHOLD
    
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    summer_df = df[df['month'].isin([7, 8, 9])].copy()
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
                    month_data['is_maintenance'] = month_data['change'] < -2.0 * changes.std()
                    if month_data['is_maintenance'].any():
                        maint_years = month_data[month_data['is_maintenance']]['year'].tolist()
                        if maint_years:
                            last_maint_year = max(maint_years)
    
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
    
    y_pred = np.full(len(test_df), anchor_val)
    y_true = test_df['tqi_value'].values
    
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    y_pred = np.clip(y_pred, safe_min, safe_max)
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

# ============ 优化1: 截尾比例调优 ============
def trimmed_mean_variant(train_df, test_df, trim_ratio):
    """截尾均值变体"""
    values = train_df['tqi_value'].values
    trimmed = stats.trim_mean(values, trim_ratio)
    
    y_pred = np.full(len(test_df), trimmed)
    y_true = test_df['tqi_value'].values
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

# ============ 优化2: 自适应混合 ============
def adaptive_hybrid(train_df, test_df):
    """自适应混合：根据样本特性动态选择锚定方法"""
    values = train_df['tqi_value'].values
    
    # 计算样本统计特征
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_val = np.std(values)
    last_val = values[-1]
    
    # 计算变异系数 (CV)
    cv = std_val / mean_val if mean_val > 0 else 0
    
    # 计算偏度
    skewness = stats.skew(values)
    
    # 根据特征选择锚定方法
    if cv < 0.1:  # 低波动样本
        # 使用简单均值
        anchor_val = mean_val
        method = 'mean_low_cv'
    elif abs(skewness) > 1.0:  # 高度偏态样本
        # 使用稳健中位数
        anchor_val = median_val
        method = 'median_skewed'
    else:
        # 使用截尾均值 (10%)
        anchor_val = stats.trim_mean(values, 0.1)
        method = 'trimmed_10'
    
    y_pred = np.full(len(test_df), anchor_val)
    y_true = test_df['tqi_value'].values
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

def adaptive_with_shift(train_df, test_df):
    """自适应 + 分布偏移检测"""
    SHIFT_THRESHOLD = 0.3
    
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    
    values = train_df['tqi_value'].values
    mean_val = np.mean(values)
    median_val = np.median(values)
    
    # 分布偏移检测
    recent_train = train_df.tail(int(len(train_df) * 0.2))
    recent_mean = recent_train['tqi_value'].mean()
    has_shift = abs(mean_val - recent_mean) > SHIFT_THRESHOLD * mean_val
    
    # 根据偏移情况选择方法
    if has_shift:
        # 有偏移：使用截尾均值减少异常影响
        anchor_val = stats.trim_mean(values, 0.1)
    else:
        # 无偏移：使用简单均值
        anchor_val = mean_val
    
    y_pred = np.full(len(test_df), anchor_val)
    y_true = test_df['tqi_value'].values
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

# ============ 主实验 ============
def main():
    print("="*100)
    print("v2.3_no_seasonal 深度优化实验")
    print("="*100)
    print("\n优化方向:")
    print("  1. 截尾比例调优: 5%, 10%, 15%, 20%")
    print("  2. 自适应混合: 根据样本特性动态选择")
    print("  3. 全量486样本验证")
    print("="*100)
    
    # 加载全部486个样本
    with open(SAMPLE_LIST_FILE, 'r') as f:
        sample_list = [int(line.strip()) for line in f if line.strip()]
    
    all_df = load_all_data()
    print(f"\n数据加载完成: {len(all_df)} 条记录")
    print(f"总样本数: {len(sample_list)}")
    
    # 方法定义
    methods = {
        '历史均值': historical_mean_baseline,
        'v23_no_seasonal': v23_no_seasonal,
        'trimmed_5%': lambda tr, te: trimmed_mean_variant(tr, te, 0.05),
        'trimmed_10%': lambda tr, te: trimmed_mean_variant(tr, te, 0.10),
        'trimmed_15%': lambda tr, te: trimmed_mean_variant(tr, te, 0.15),
        'trimmed_20%': lambda tr, te: trimmed_mean_variant(tr, te, 0.20),
        'adaptive_hybrid': adaptive_hybrid,
        'adaptive_shift': adaptive_with_shift,
    }
    
    # 存储结果
    results = {name: [] for name in methods.keys()}
    
    print(f"\n开始实验 (全量{len(sample_list)}样本)...")
    print(f"{'#':<5} {'里程':<10}", end='')
    for name in methods.keys():
        print(f" {name:<16}", end='')
    print()
    print("-" * 150)
    
    valid_count = 0
    for i, mile in enumerate(sample_list, 1):
        try:
            df = load_sample_data(all_df, mile)
            if df is None or len(df) < 50:
                continue
            
            valid_count += 1
            train_df, val_df, test_df = split_data(df)
            
            row_results = {'mile': mile}
            for name, func in methods.items():
                mae, std = func(train_df, test_df)
                results[name].append(mae)
                row_results[name] = mae
            
            # 每10个样本输出一次
            if i % 10 == 0 or i == len(sample_list):
                print(f"{i:<5} {mile:<10}", end='')
                for name in methods.keys():
                    print(f" {row_results[name]:<16.4f}", end='')
                print()
            
        except Exception as e:
            print(f"{i:<5} {mile:<10} 错误: {str(e)[:40]}")
    
    # 统计汇总
    print("\n" + "="*100)
    print("统计汇总")
    print("="*100)
    
    n = len(results['历史均值'])
    print(f"\n有效样本: {n}/{len(sample_list)}")
    
    # MAE统计
    print(f"\n{'方法':<20} {'平均MAE':<12} {'标准差':<12} {'击败HM':<12} {'击败v23':<12}")
    print("-" * 70)
    hm_results = results['历史均值']
    v23_results = results['v23_no_seasonal']
    
    for name, mae_list in results.items():
        mean_mae = np.mean(mae_list)
        std_mae = np.std(mae_list)
        win_vs_hm = sum(1 for i in range(len(mae_list)) if mae_list[i] < hm_results[i])
        win_vs_v23 = sum(1 for i in range(len(mae_list)) if mae_list[i] < v23_results[i])
        print(f"{name:<20} {mean_mae:<12.4f} {std_mae:<12.4f} {win_vs_hm}/{n:<8} {win_vs_v23}/{n}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df['mile'] = sample_list[:n]
    output_file = f'{BASE_DIR}/results/v25_full_optimization_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存: {output_file}")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    main()
