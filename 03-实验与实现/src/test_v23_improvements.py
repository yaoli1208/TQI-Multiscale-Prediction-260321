#!/usr/bin/env python3
"""
v2.3 改进方案对比实验
测试4个改进方向：
1. 平滑策略过渡（加权融合）
2. 更严格的预测裁剪（3σ）
3. 调整分布偏移阈值（0.4, 0.5）
4. 多锚定融合
"""
import pandas as pd
import numpy as np
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
    test_start = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[test_start:].copy()

def historical_mean_baseline(train_df, test_df):
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred))

def detect_maintenance(train_df):
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    summer_df = df[df['month'].isin([7, 8, 9])].copy()
    if len(summer_df) == 0:
        return None
    
    monthly = summer_df.groupby(['year', 'month'])['tqi_value'].mean().reset_index()
    monthly['is_maintenance'] = False
    
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

# ============ v2.3 Baseline ============
def v23_baseline(train_df, test_df, shift_threshold=0.3, sigma_clip=5.0):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # 分布偏移检测
    recent_train = train_df.tail(int(len(train_df) * 0.2))
    recent_mean = recent_train['tqi_value'].mean()
    test_mean = test_df['tqi_value'].mean()
    shift = abs(test_mean - recent_mean)
    has_shift = shift > shift_threshold
    
    # 维修检测
    last_maint_year = detect_maintenance(train_df)
    
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    if not has_shift:
        anchor_val = train_df['tqi_value'].mean()
        use_seasonal = False
    else:
        if last_maint_year is not None:
            maint_data = df[(df['year'] == last_maint_year) & (df['month'].isin([7, 8, 9]))]
            anchor_val = maint_data['tqi_value'].mean() if len(maint_data) > 0 else recent_mean
        else:
            last_year = df['year'].max()
            last_year_data = df[df['year'] == last_year]
            anchor_val = last_year_data['tqi_value'].mean() if len(last_year_data) > 0 else recent_mean
        use_seasonal = True
    
    if use_seasonal:
        monthly_avg = df.groupby('month')['tqi_value'].mean()
        overall_avg = df['tqi_value'].mean()
        seasonal_adj = monthly_avg - overall_avg
    else:
        seasonal_adj = {}
    
    predictions = []
    for _, row in test_df.iterrows():
        month = row['date'].month
        if use_seasonal:
            seasonal = seasonal_adj.get(month, 0)
            pred = anchor_val + seasonal
        else:
            pred = anchor_val
        predictions.append(pred)
    
    predictions = np.array(predictions)
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - sigma_clip * train_std)
    safe_max = train_mean + sigma_clip * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    return np.mean(np.abs(test_df['tqi_value'].values - predictions))

# ============ 改进1: 平滑策略过渡（加权融合） ============
def v23_smooth(train_df, test_df, shift_threshold=0.3, sigma_clip=5.0):
    """平滑过渡：根据偏移程度加权融合历史均值和近期锚定"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    recent_train = train_df.tail(int(len(train_df) * 0.2))
    recent_mean = recent_train['tqi_value'].mean()
    test_mean = test_df['tqi_value'].mean()
    shift = abs(test_mean - recent_mean)
    
    # 平滑权重：偏移越大，历史均值权重越低
    # w = exp(-shift * 2)  # 偏移0.5时 w≈0.37
    w = max(0, 1 - shift / (shift_threshold * 2))  # 线性过渡
    
    last_maint_year = detect_maintenance(train_df)
    
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # 三个锚定值
    historical_mean = train_df['tqi_value'].mean()
    
    if last_maint_year is not None:
        maint_data = df[(df['year'] == last_maint_year) & (df['month'].isin([7, 8, 9]))]
        maint_anchor = maint_data['tqi_value'].mean() if len(maint_data) > 0 else recent_mean
    else:
        maint_anchor = recent_mean
    
    last_year = df['year'].max()
    last_year_data = df[df['year'] == last_year]
    recent_anchor = last_year_data['tqi_value'].mean() if len(last_year_data) > 0 else recent_mean
    
    # 季节性
    monthly_avg = df.groupby('month')['tqi_value'].mean()
    overall_avg = df['tqi_value'].mean()
    seasonal_adj = monthly_avg - overall_avg
    
    # 加权融合锚定
    if last_maint_year is not None:
        anchor_val = w * historical_mean + (1 - w) * maint_anchor
    else:
        anchor_val = w * historical_mean + (1 - w) * recent_anchor
    
    predictions = []
    for _, row in test_df.iterrows():
        month = row['date'].month
        seasonal = seasonal_adj.get(month, 0)
        pred = anchor_val + seasonal
        predictions.append(pred)
    
    predictions = np.array(predictions)
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - sigma_clip * train_std)
    safe_max = train_mean + sigma_clip * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    return np.mean(np.abs(test_df['tqi_value'].values - predictions))

# ============ 改进2: 更严格的裁剪（3σ） ============
def v23_clip3(train_df, test_df, shift_threshold=0.3):
    """使用3σ裁剪"""
    return v23_baseline(train_df, test_df, shift_threshold=shift_threshold, sigma_clip=3.0)

# ============ 改进3: 调整分布偏移阈值 ============
def v23_thresh04(train_df, test_df):
    """阈值0.4"""
    return v23_baseline(train_df, test_df, shift_threshold=0.4)

def v23_thresh05(train_df, test_df):
    """阈值0.5"""
    return v23_baseline(train_df, test_df, shift_threshold=0.5)

# ============ 改进4: 多锚定融合 ============
def v23_multi_anchor(train_df, test_df, shift_threshold=0.3, sigma_clip=5.0):
    """三个锚定加权融合：历史均值 + 近期锚定 + 维修锚定"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    recent_train = train_df.tail(int(len(train_df) * 0.2))
    recent_mean = recent_train['tqi_value'].mean()
    test_mean = test_df['tqi_value'].mean()
    shift = abs(test_mean - recent_mean)
    has_shift = shift > shift_threshold
    
    last_maint_year = detect_maintenance(train_df)
    
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # 三个锚定
    historical_mean = train_df['tqi_value'].mean()
    
    last_year = df['year'].max()
    last_year_data = df[df['year'] == last_year]
    recent_anchor = last_year_data['tqi_value'].mean() if len(last_year_data) > 0 else recent_mean
    
    if last_maint_year is not None:
        maint_data = df[(df['year'] == last_maint_year) & (df['month'].isin([7, 8, 9]))]
        maint_anchor = maint_data['tqi_value'].mean() if len(maint_data) > 0 else recent_mean
    else:
        maint_anchor = recent_anchor
    
    # 动态权重（基于偏移程度和维修检测）
    if not has_shift:
        w_hist, w_recent, w_maint = 0.8, 0.2, 0.0
    elif last_maint_year is not None:
        w_hist, w_recent, w_maint = 0.2, 0.2, 0.6  # 维修后，维修锚定权重高
    else:
        w_hist, w_recent, w_maint = 0.3, 0.7, 0.0  # 无维修，近期锚定权重高
    
    anchor_val = w_hist * historical_mean + w_recent * recent_anchor + w_maint * maint_anchor
    
    # 季节性
    monthly_avg = df.groupby('month')['tqi_value'].mean()
    overall_avg = df['tqi_value'].mean()
    seasonal_adj = monthly_avg - overall_avg
    
    predictions = []
    for _, row in test_df.iterrows():
        month = row['date'].month
        seasonal = seasonal_adj.get(month, 0)
        pred = anchor_val + seasonal
        predictions.append(pred)
    
    predictions = np.array(predictions)
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - sigma_clip * train_std)
    safe_max = train_mean + sigma_clip * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    return np.mean(np.abs(test_df['tqi_value'].values - predictions))

def main():
    print("="*110)
    print("v2.3 改进方案对比实验")
    print("="*110)
    
    with open(SAMPLE_LIST_FILE, 'r') as f:
        sample_list = [int(line.strip()) for line in f if line.strip()]
    sample_list = sample_list[:50]
    
    all_df = load_all_data()
    print(f"数据加载完成: {len(all_df)} 条记录\n")
    
    methods = {
        '历史均值': historical_mean_baseline,
        'v2.3_baseline': v23_baseline,
        'v2.3_smooth': v23_smooth,
        'v2.3_clip3': v23_clip3,
        'v2.3_thresh04': v23_thresh04,
        'v2.3_thresh05': v23_thresh05,
        'v2.3_multi': v23_multi_anchor,
    }
    
    results = {name: [] for name in methods.keys()}
    
    print(f"{'#':<4} {'里程':<10}", end='')
    for name in methods.keys():
        print(f" {name:<15}", end='')
    print()
    print("-" * 110)
    
    for i, mile in enumerate(sample_list, 1):
        try:
            df = load_sample_data(all_df, mile)
            if df is None or len(df) < 50:
                continue
            
            train_df, test_df = split_data(df)
            
            row_results = {'mile': mile}
            for name, func in methods.items():
                mae = func(train_df, test_df)
                results[name].append(mae)
                row_results[name] = mae
            
            print(f"{i:<4} {mile:<10}", end='')
            for name in methods.keys():
                print(f" {row_results[name]:<15.4f}", end='')
            print()
            
        except Exception as e:
            print(f"{i:<4} {mile:<10} 错误: {str(e)[:40]}")
    
    # 统计汇总
    print("\n" + "="*110)
    print("统计汇总")
    print("="*110)
    
    print(f"\n{'方法':<20} {'平均MAE':<12} {'标准差':<12} {'击败HM':<12} {'vs v2.3_base':<15}")
    print("-" * 110)
    
    hm_results = results['历史均值']
    v23_base_results = results['v2.3_baseline']
    
    for name, mae_list in results.items():
        mean_mae = np.mean(mae_list)
        std_mae = np.std(mae_list)
        win_vs_hm = sum(1 for i in range(len(mae_list)) if mae_list[i] < hm_results[i])
        win_vs_v23 = sum(1 for i in range(len(mae_list)) if mae_list[i] < v23_base_results[i]) if name != 'v2.3_baseline' else '-'
        
        win_vs_v23_str = f"{win_vs_v23}/50" if isinstance(win_vs_v23, int) else win_vs_v23
        print(f"{name:<20} {mean_mae:<12.4f} {std_mae:<12.4f} {win_vs_hm}/50{'':<6} {win_vs_v23_str:<15}")
    
    # 找出最佳改进方案
    print(f"\n{'='*110}")
    print("改进效果排名（vs v2.3 baseline）")
    print(f"{'='*110}")
    
    improvements = []
    for name in ['v2.3_smooth', 'v2.3_clip3', 'v2.3_thresh04', 'v2.3_thresh05', 'v2.3_multi']:
        mae_list = results[name]
        mean_mae = np.mean(mae_list)
        std_mae = np.std(mae_list)
        base_mean = np.mean(v23_base_results)
        base_std = np.std(v23_base_results)
        mae_improve = base_mean - mean_mae
        std_improve = base_std - std_mae
        wins = sum(1 for i in range(len(mae_list)) if mae_list[i] < v23_base_results[i])
        improvements.append((name, mean_mae, std_mae, mae_improve, std_improve, wins))
    
    improvements.sort(key=lambda x: (x[3], x[4]), reverse=True)
    
    print(f"{'排名':<6} {'方案':<20} {'MAE':<12} {'标准差':<12} {'MAE改善':<12} {'标准差改善':<12} {'胜场':<10}")
    print("-" * 110)
    for i, (name, mae, std, mae_imp, std_imp, wins) in enumerate(improvements, 1):
        print(f"{i:<6} {name:<20} {mae:<12.4f} {std:<12.4f} {mae_imp:>+12.4f} {std_imp:>+12.4f} {wins}/50{'':<4}")
    
    # 标准差最小方案
    print(f"\n{'='*110}")
    print("标准差最小的方案")
    print(f"{'='*110}")
    
    std_ranking = [(name, np.std(mae_list), np.mean(mae_list)) for name, mae_list in results.items()]
    std_ranking.sort(key=lambda x: x[1])
    
    print(f"{'排名':<6} {'方法':<20} {'标准差':<12} {'平均MAE':<12}")
    print("-" * 50)
    for i, (name, std, mean) in enumerate(std_ranking, 1):
        print(f"{i:<6} {name:<20} {std:<12.4f} {mean:<12.4f}")
    
    print("\n" + "="*110)

if __name__ == '__main__':
    main()
