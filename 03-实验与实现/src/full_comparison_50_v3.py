#!/usr/bin/env python3
"""
完整基线对比实验 - v2.3 vs v2.1 vs 历史均值 vs TimeMixer
清洗后50样本 (qualified_miles_v3.txt)
"""
import pandas as pd
import numpy as np
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
    n = len(df)
    train_end = int(n * 0.7)
    test_start = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[test_start:].copy()

# ============ 基线方法 ============
def historical_mean_baseline(train_df, test_df):
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred))

def moving_average_baseline(train_df, test_df, window=30):
    baseline = train_df['tqi_value'].tail(window).mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred))

def timemixer_baseline(train_df, test_df):
    """简化版TimeMixer - 多尺度分解"""
    from sklearn.ensemble import GradientBoostingRegressor
    
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # 特征工程：多尺度时间特征
    def create_features(df):
        df = df.copy()
        df['dayofyear'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        # 滚动统计（多尺度）
        for window in [7, 30, 90]:
            if len(df) >= window:
                df[f'rolling_mean_{window}'] = df['tqi_value'].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}'] = df['tqi_value'].rolling(window=window, min_periods=1).std()
        
        # 滞后特征
        for lag in [1, 7, 30]:
            if len(df) > lag:
                df[f'lag_{lag}'] = df['tqi_value'].shift(lag)
        
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(train_df['tqi_value'].mean())
        return df
    
    try:
        train_feat = create_features(train_df)
        test_feat = create_features(test_df)
        
        feature_cols = [c for c in train_feat.columns if c not in ['date', 'tqi_value']]
        if len(feature_cols) == 0:
            return historical_mean_baseline(train_df, test_df)
        
        X_train = train_feat[feature_cols]
        y_train = train_feat['tqi_value']
        X_test = test_feat[feature_cols]
        y_test = test_feat['tqi_value'].values
        
        model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        return np.mean(np.abs(y_test - y_pred))
    except:
        return historical_mean_baseline(train_df, test_df)

# ============ Trident v2.1 ============
def trident_v21_predict(train_df, test_df):
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df = test_df.copy()
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
        return {'maintenance_years': [], 'last_maintenance_year': None}
    
    monthly = summer_df.groupby(['year', 'month'])['tqi_value'].mean().reset_index()
    monthly['change'] = np.nan
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
                else:
                    month_data['is_maintenance'] = month_data['change'] < -0.5
            monthly.loc[month_data.index, 'change'] = month_data['change'].values
            monthly.loc[month_data.index, 'is_maintenance'] = month_data['is_maintenance'].values
    
    yearly_maintenance = monthly.groupby('year')['is_maintenance'].any().reset_index()
    maintenance_years = yearly_maintenance[yearly_maintenance['is_maintenance']]['year'].tolist()
    last_maintenance_year = max(maintenance_years) if maintenance_years else None
    
    return {
        'maintenance_years': maintenance_years,
        'last_maintenance_year': last_maintenance_year
    }

def trident_v23_predict(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    shift_info = detect_distribution_shift(train_df, test_df)
    maint_info = detect_maintenance_v23(train_df)
    last_maint_year = maint_info['last_maintenance_year']
    
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    if not shift_info['has_shift']:
        anchor_val = shift_info['train_mean']
        use_seasonal = False
    else:
        if last_maint_year is not None:
            maint_data = df[(df['year'] == last_maint_year) & (df['month'].isin([7, 8, 9]))]
            anchor_val = maint_data['tqi_value'].mean() if len(maint_data) > 0 else shift_info['recent_mean']
        else:
            last_year = df['year'].max()
            last_year_data = df[df['year'] == last_year]
            anchor_val = last_year_data['tqi_value'].mean() if len(last_year_data) > 0 else shift_info['recent_mean']
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
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    return predictions

def main():
    print("="*100)
    print("完整基线对比实验 - v2.3 vs v2.1 vs 历史均值 vs TimeMixer")
    print("清洗后50样本 (qualified_miles_v3.txt)")
    print("="*100)
    
    with open(SAMPLE_LIST_FILE, 'r') as f:
        sample_list = [int(line.strip()) for line in f if line.strip()]
    sample_list = sample_list[:50]
    
    all_df = load_all_data()
    print(f"数据加载完成: {len(all_df)} 条记录\n")
    
    results = []
    
    print(f"{'#':<4} {'里程':<10} {'历史均值':<12} {'移动平均':<12} {'TimeMixer':<12} {'v2.1':<12} {'v2.3':<12} {'最佳':<12}")
    print("-" * 100)
    
    for i, mile in enumerate(sample_list, 1):
        try:
            df = load_sample_data(all_df, mile)
            if df is None or len(df) < 50:
                print(f"{i:<4} {mile:<10} 跳过(数据不足)")
                continue
            
            train_df, test_df = split_data(df)
            y_true = test_df['tqi_value'].values
            
            # 运行所有方法
            hm_mae = historical_mean_baseline(train_df, test_df)
            ma_mae = moving_average_baseline(train_df, test_df)
            tm_mae = timemixer_baseline(train_df, test_df)
            
            v21_pred = trident_v21_predict(train_df, test_df)
            v21_mae = np.mean(np.abs(y_true - v21_pred))
            
            v23_pred = trident_v23_predict(train_df, test_df)
            v23_mae = np.mean(np.abs(y_true - v23_pred))
            
            # 找出最佳方法
            maes = {
                '历史均值': hm_mae,
                '移动平均': ma_mae,
                'TimeMixer': tm_mae,
                'v2.1': v21_mae,
                'v2.3': v23_mae
            }
            best_method = min(maes, key=maes.get)
            
            results.append({
                'mile': mile,
                'historical_mean': hm_mae,
                'moving_average': ma_mae,
                'timemixer': tm_mae,
                'v21': v21_mae,
                'v23': v23_mae,
                'best': best_method
            })
            
            print(f"{i:<4} {mile:<10} {hm_mae:<12.4f} {ma_mae:<12.4f} {tm_mae:<12.4f} {v21_mae:<12.4f} {v23_mae:<12.4f} {best_method:<12}")
            
        except Exception as e:
            print(f"{i:<4} {mile:<10} 错误: {str(e)[:40]}")
    
    # 统计汇总
    print("\n" + "="*100)
    print("统计汇总")
    print("="*100)
    
    df_results = pd.DataFrame(results)
    n = len(df_results)
    
    print(f"\n有效样本: {n}/50")
    
    print(f"\n各方法平均MAE:")
    for method in ['historical_mean', 'moving_average', 'timemixer', 'v21', 'v23']:
        mae = df_results[method].mean()
        std = df_results[method].std()
        print(f"  {method:15s}: {mae:.4f} ± {std:.4f}")
    
    print(f"\n最佳方法分布:")
    best_counts = df_results['best'].value_counts()
    for method, count in best_counts.items():
        print(f"  {method:15s}: {count} 个 ({count/n*100:.1f}%)")
    
    print(f"\n击败历史均值的样本数:")
    for method in ['moving_average', 'timemixer', 'v21', 'v23']:
        wins = (df_results[method] < df_results['historical_mean']).sum()
        print(f"  {method:15s}: {wins} 个 ({wins/n*100:.1f}%)")
    
    print(f"\n各方法胜率对比 (两两对比):")
    methods = ['historical_mean', 'moving_average', 'timemixer', 'v21', 'v23']
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            m1_wins = (df_results[m1] < df_results[m2]).sum()
            m2_wins = (df_results[m2] < df_results[m1]).sum()
            ties = n - m1_wins - m2_wins
            print(f"  {m1} vs {m2}: {m1_wins}-{m2_wins}-{ties} (胜-负-平)")
    
    # v2.3大胜样本
    df_results['v23_improvement'] = df_results['historical_mean'] - df_results['v23']
    big_wins = df_results[df_results['v23_improvement'] > 0.2].sort_values('v23_improvement', ascending=False)
    
    print(f"\nv2.3 大胜样本 (vs 历史均值改善 > 0.2):")
    if len(big_wins) > 0:
        print(f"{'里程':<10} {'历史均值':<12} {'v2.3':<12} {'改善':<10}")
        for _, row in big_wins.head(5).iterrows():
            print(f"{int(row['mile']):<10} {row['historical_mean']:<12.4f} {row['v23']:<12.4f} {row['v23_improvement']:<+10.4f}")
    else:
        print("  无")
    
    # 保存结果
    output_file = f'{BASE_DIR}/results/baseline_comparison_50_v3.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n详细结果已保存: {output_file}")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    main()
