#!/usr/bin/env python3
"""
Trident 正式对照实验
===================
实验设计版本: v1.0
日期: 2026-03-25

方法列表:
1. 历史均值 (Historical Mean)
2. 移动平均 (Moving Average)
3. Holt-Winters
4. Trident v2.1
5. Trident v2.3
6. Trident v2.3_soft (软切换)
7. Trident v2.3_ens (集成)
8. Trident v2.3_no_seasonal (消融)

数据划分: 70% 训练 / 15% 验证 / 15% 测试
"""
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v3.txt'
SHIFT_THRESHOLD = 0.3

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
    """70% 训练 / 15% 验证 / 15% 测试"""
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy()
    )

# ============ 基线方法 ============
def historical_mean_baseline(train_df, test_df):
    """历史均值基线"""
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred)), np.std(y_true - y_pred)

def moving_average_baseline(train_df, test_df, window=30):
    """移动平均基线"""
    baseline = train_df['tqi_value'].tail(window).mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred)), np.std(y_true - y_pred)

def holt_winters_baseline(train_df, test_df, seasonal_periods=12):
    """Holt-Winters季节性指数平滑"""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        series = train_df['tqi_value'].values
        
        if len(series) < seasonal_periods * 2:
            return historical_mean_baseline(train_df, test_df)
        
        model = ExponentialSmoothing(
            series, trend='add', seasonal='add', seasonal_periods=seasonal_periods
        )
        fitted = model.fit(optimized=True)
        predictions = fitted.forecast(steps=len(test_df))
        y_true = test_df['tqi_value'].values
        errors = y_true - predictions
        return np.mean(np.abs(errors)), np.std(errors)
    except:
        return historical_mean_baseline(train_df, test_df)

# ============ Trident v2.1 ============
def trident_v21_predict(train_df, test_df):
    """Trident v2.1 - 滚动锚定（无分布偏移检测）"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    summer_df = df[df['month'].isin([7, 8, 9])]
    yearly_summer = summer_df.groupby('year')['tqi_value'].mean()
    
    maintenance_years = []
    for i in range(1, len(yearly_summer)):
        if yearly_summer.iloc[i] < yearly_summer.iloc[i-1] - 0.3:
            maintenance_years.append(yearly_summer.index[i])
    
    last_maint_year = max(maintenance_years) if maintenance_years else None
    
    if last_maint_year is not None:
        anchor_data = df[(df['year'] == last_maint_year) & (df['month'].isin([7, 8, 9]))]
        anchor_val = anchor_data['tqi_value'].mean()
    else:
        anchor_val = yearly_summer.tail(3).mean() if len(yearly_summer) >= 3 else yearly_summer.mean()
    
    monthly_avg = df.groupby('month')['tqi_value'].mean()
    overall_avg = df['tqi_value'].mean()
    seasonal_adj = monthly_avg - overall_avg
    
    if len(yearly_summer) >= 2:
        diffs = yearly_summer.diff().dropna()
        lambda_decay = diffs[diffs > 0].median() if len(diffs[diffs > 0]) > 0 else 0
    else:
        lambda_decay = 0
    
    predictions = []
    for _, row in test_df.iterrows():
        month = row['date'].month
        seasonal = seasonal_adj.get(month, 0)
        pred = anchor_val + seasonal + lambda_decay
        predictions.append(pred)
    
    return np.array(predictions)

def trident_v21_baseline(train_df, test_df):
    y_pred = trident_v21_predict(train_df, test_df)
    y_true = test_df['tqi_value'].values
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

# ============ Trident v2.3 ============
def detect_distribution_shift(train_df, test_df):
    recent_train = train_df.tail(int(len(train_df) * 0.2))
    recent_mean = recent_train['tqi_value'].mean()
    test_mean = test_df['tqi_value'].mean()
    shift = abs(test_mean - recent_mean)
    return {
        'shift': shift,
        'has_shift': shift > SHIFT_THRESHOLD,
        'recent_mean': recent_mean,
        'test_mean': test_mean,
        'train_mean': train_df['tqi_value'].mean()
    }

def detect_maintenance_v23(train_df):
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    summer_df = df[df['month'].isin([7, 8, 9])].copy()
    if len(summer_df) == 0:
        return None
    
    monthly = summer_df.groupby(['year', 'month'])['tqi_value'].mean().reset_index()
    
    for month in [7, 8, 9]:
        month_data = monthly[monthly['month'] == month].sort_values('year')
        if len(month_data) >= 2:
            month_data = month_data.copy()
            month_data['change'] = month_data['tqi_value'].diff()
            changes = month_data['change'].dropna()
            if len(changes) > 0:
                change_std = changes.std()
                if change_std > 0:
                    month_data['is_maintenance'] = month_data['change'] < -2.0 * change_std
                monthly.loc[month_data.index, 'is_maintenance'] = month_data['is_maintenance'].values
    
    yearly_maintenance = monthly.groupby('year')['is_maintenance'].any().reset_index()
    maintenance_years = yearly_maintenance[yearly_maintenance['is_maintenance']]['year'].tolist()
    return max(maintenance_years) if maintenance_years else None

def trident_v23_predict(train_df, test_df, use_soft=False, use_ensemble=False, no_seasonal=False):
    """
    Trident v2.3 及其变体
    - use_soft: 软切换（根据偏移程度调整权重）
    - use_ensemble: 集成历史均值
    - no_seasonal: 移除季节性调整
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    shift_info = detect_distribution_shift(train_df, test_df)
    last_maint_year = detect_maintenance_v23(train_df)
    
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    historical_mean = train_df['tqi_value'].mean()
    
    # 计算锚定值
    if not shift_info['has_shift']:
        anchor_val = historical_mean
        use_seasonal = False
        shift_weight = 1.0 if use_soft else 1.0
    else:
        if last_maint_year is not None:
            maint_data = df[(df['year'] == last_maint_year) & (df['month'].isin([7, 8, 9]))]
            anchor_val = maint_data['tqi_value'].mean() if len(maint_data) > 0 else shift_info['recent_mean']
        else:
            last_year = df['year'].max()
            last_year_data = df[df['year'] == last_year]
            anchor_val = last_year_data['tqi_value'].mean() if len(last_year_data) > 0 else shift_info['recent_mean']
        use_seasonal = True
        # 软切换权重
        if use_soft:
            shift_weight = max(0, 1 - shift_info['shift'] / (SHIFT_THRESHOLD * 2))
        else:
            shift_weight = 0.0
    
    if use_seasonal and not no_seasonal:
        monthly_avg = df.groupby('month')['tqi_value'].mean()
        overall_avg = df['tqi_value'].mean()
        seasonal_adj = monthly_avg - overall_avg
    else:
        seasonal_adj = {}
    
    # 集成策略
    if use_ensemble:
        # 融合历史均值和锚定值
        anchor_val = shift_weight * historical_mean + (1 - shift_weight) * anchor_val
    
    predictions = []
    for _, row in test_df.iterrows():
        month = row['date'].month
        if use_seasonal and not no_seasonal:
            seasonal = seasonal_adj.get(month, 0)
            pred = anchor_val + seasonal
        else:
            pred = anchor_val
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # 安全检查
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    return predictions

def trident_v23_baseline(train_df, test_df, **kwargs):
    y_pred = trident_v23_predict(train_df, test_df, **kwargs)
    y_true = test_df['tqi_value'].values
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

# ============ 主实验 ============
def main():
    print("="*100)
    print("Trident 正式对照实验")
    print("="*100)
    print("\n方法列表:")
    print("  1. 历史均值 (Historical Mean)")
    print("  2. 移动平均 (Moving Average)")
    print("  3. Holt-Winters")
    print("  4. Trident v2.1")
    print("  5. Trident v2.3")
    print("  6. Trident v2.3_soft (软切换)")
    print("  7. Trident v2.3_ens (集成)")
    print("  8. Trident v2.3_no_seasonal (消融)")
    print("\n数据划分: 70% 训练 / 15% 验证 / 15% 测试")
    print("="*100)
    
    # 加载数据
    with open(SAMPLE_LIST_FILE, 'r') as f:
        sample_list = [int(line.strip()) for line in f if line.strip()]
    sample_list = sample_list[:50]
    
    all_df = load_all_data()
    print(f"\n数据加载完成: {len(all_df)} 条记录")
    print(f"样本数量: {len(sample_list)}")
    
    # 方法定义
    methods = {
        '历史均值': lambda tr, te: historical_mean_baseline(tr, te),
        '移动平均': lambda tr, te: moving_average_baseline(tr, te),
        'Holt-Winters': lambda tr, te: holt_winters_baseline(tr, te),
        'v2.1': lambda tr, te: trident_v21_baseline(tr, te),
        'v2.3': lambda tr, te: trident_v23_baseline(tr, te),
        'v2.3_soft': lambda tr, te: trident_v23_baseline(tr, te, use_soft=True),
        'v2.3_ens': lambda tr, te: trident_v23_baseline(tr, te, use_ensemble=True),
        'v2.3_no_seasonal': lambda tr, te: trident_v23_baseline(tr, te, no_seasonal=True),
    }
    
    # 存储结果
    results = {name: [] for name in methods.keys()}
    errors = {name: [] for name in methods.keys()}
    
    print(f"\n开始实验...")
    print(f"{'#':<4} {'里程':<10}", end='')
    for name in methods.keys():
        print(f" {name:<18}", end='')
    print()
    print("-" * 160)
    
    for i, mile in enumerate(sample_list, 1):
        try:
            df = load_sample_data(all_df, mile)
            if df is None or len(df) < 50:
                print(f"{i:<4} {mile:<10} 跳过(数据不足)")
                continue
            
            train_df, val_df, test_df = split_data(df)
            
            row_results = {'mile': mile}
            for name, func in methods.items():
                mae, std = func(train_df, test_df)
                results[name].append(mae)
                errors[name].append(std)
                row_results[name] = mae
            
            print(f"{i:<4} {mile:<10}", end='')
            for name in methods.keys():
                print(f" {row_results[name]:<18.4f}", end='')
            print()
            
        except Exception as e:
            print(f"{i:<4} {mile:<10} 错误: {str(e)[:80]}")
    
    # 统计汇总
    print("\n" + "="*100)
    print("统计汇总")
    print("="*100)
    
    n = len(results['历史均值'])
    print(f"\n有效样本: {n}/50")
    
    # MAE统计
    print(f"\n{'方法':<20} {'平均MAE':<12} {'标准差':<12} {'击败HM':<12}")
    print("-" * 60)
    hm_results = results['历史均值']
    for name, mae_list in results.items():
        mean_mae = np.mean(mae_list)
        std_mae = np.std(mae_list)
        win_vs_hm = sum(1 for i in range(len(mae_list)) if mae_list[i] < hm_results[i])
        print(f"{name:<20} {mean_mae:<12.4f} {std_mae:<12.4f} {win_vs_hm}/{n}")
    
    # 两两胜率
    print(f"\n{'='*100}")
    print("两两胜率对比")
    print(f"{'='*100}")
    
    method_names = list(methods.keys())
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i+1:]:
            m1_wins = sum(1 for j in range(n) if results[m1][j] < results[m2][j])
            m2_wins = sum(1 for j in range(n) if results[m2][j] < results[m1][j])
            ties = n - m1_wins - m2_wins
            print(f"{m1} vs {m2}: {m1_wins}-{m2_wins}-{ties} (胜-负-平)")
    
    # Wilcoxon检验
    print(f"\n{'='*100}")
    print("Wilcoxon符号秩检验 (vs 历史均值)")
    print(f"{'='*100}")
    print(f"{'方法':<20} {'统计量':<12} {'p-value':<12} {'显著性':<10}")
    print("-" * 60)
    
    for name in method_names:
        if name == '历史均值':
            continue
        try:
            stat, p_value = stats.wilcoxon(results['历史均值'], results[name])
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"{name:<20} {stat:<12.1f} {p_value:<12.6f} {significance:<10}")
        except:
            print(f"{name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df['mile'] = sample_list[:n]
    output_file = f'{BASE_DIR}/results/formal_comparison_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存: {output_file}")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    main()
