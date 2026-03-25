#!/usr/bin/env python3
"""
v2.2 vs v2.1 vs 历史均值 - 50样本对比 (优化版)
清洗数据版本 (qualified_miles_v3.txt)
只读取一次Excel文件
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v3.txt'

def load_all_data():
    """只读取一次Excel文件"""
    print("正在加载数据文件 (22MB)...")
    df = pd.read_excel(DATA_FILE)
    df.columns = df.columns.str.strip()
    # 过滤掉可能的列名重复行（脏数据）
    df = df[df['dete_dt'].astype(str).str.strip() != 'dete_dt'].copy()
    df['date'] = pd.to_datetime(df['dete_dt'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['tqi_value'] = pd.to_numeric(df['tqi_val'], errors='coerce')
    df['tqi_mile'] = pd.to_numeric(df['tqi_mile'], errors='coerce')
    df = df.dropna(subset=['tqi_value', 'tqi_mile'])
    return df[['tqi_mile', 'date', 'tqi_value']].copy()

def load_sample_data(all_df, mile):
    """从已加载的数据中筛选"""
    sample_df = all_df[all_df['tqi_mile'] == mile].copy()
    sample_df = sample_df.sort_values('date').reset_index(drop=True)
    return sample_df[['date', 'tqi_value']].copy()

def split_data(df):
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[val_end:].copy()

def historical_mean_baseline(train_df, test_df):
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred))

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

def trident_v22_predict(train_df, test_df):
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # 维修检测
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    summer_df = df[df['month'].isin([7, 8, 9])].copy()
    
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
    last_maint_year = max(maintenance_years) if maintenance_years else None
    
    # 劣化趋势
    summer_yearly = summer_df.groupby('year')['tqi_value'].mean().reset_index()
    if last_maint_year is not None:
        summer_yearly = summer_yearly[summer_yearly['year'] >= last_maint_year]
    
    if len(summer_yearly) >= 3:
        summer_yearly['change'] = summer_yearly['tqi_value'].diff()
        changes = summer_yearly['change'].dropna()
        if len(changes) > 0:
            annual_deterioration = changes.median()
            is_reliable = abs(annual_deterioration) > 0.5 * changes.std()
        else:
            annual_deterioration, is_reliable = 0.0, False
    else:
        annual_deterioration, is_reliable = 0.0, False
    
    # 预测
    last_train_year = df['year'].max()
    prev_year_data = df[df['year'] == last_train_year]
    if len(prev_year_data) == 0:
        prev_year_data = df[df['year'] == df['year'].max()]
    monthly_avg_last_year = prev_year_data.groupby('month')['tqi_value'].mean()
    
    test_df['year'] = test_df['date'].dt.year
    test_df['month'] = test_df['date'].dt.month
    
    predictions = []
    for _, row in test_df.iterrows():
        target_month = row['month']
        if is_reliable and abs(annual_deterioration) > 0.01:
            base_value = monthly_avg_last_year.get(target_month, df['tqi_value'].mean())
            prediction = base_value + annual_deterioration * (row['year'] - last_train_year)
        else:
            prediction = monthly_avg_last_year.get(target_month, df['tqi_value'].mean())
        predictions.append(prediction)
    
    predictions = np.array(predictions)
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    predictions = np.clip(predictions, max(0, train_mean - 5 * train_std), train_mean + 5 * train_std)
    
    return predictions

def main():
    print("="*90)
    print("Trident v2.2 vs v2.1 vs 历史均值 - 50样本对比 (v3.1清洗数据)")
    print("="*90)
    
    # 加载样本列表
    with open(SAMPLE_LIST_FILE, 'r') as f:
        sample_list = [int(line.strip()) for line in f if line.strip()]
    sample_list = sample_list[:50]
    print(f"样本数: {len(sample_list)}")
    
    # 只读取一次数据
    all_df = load_all_data()
    print(f"数据加载完成，共 {len(all_df)} 条记录")
    
    results = []
    
    print(f"\n{'#':<4} {'里程':<10} {'历史均值':<12} {'v2.1':<12} {'v2.2':<12} {'v2.2胜HM':<10} {'v2.1胜HM':<10} {'v2.2胜v2.1':<12}")
    print("-" * 90)
    
    for i, mile in enumerate(sample_list, 1):
        try:
            df = load_sample_data(all_df, mile)
            if df is None or len(df) < 50:
                print(f"{i:<4} {mile:<10} 跳过(数据不足)")
                continue
            
            train_df, test_df = split_data(df)
            
            hm_mae = historical_mean_baseline(train_df, test_df)
            
            v21_pred = trident_v21_predict(train_df, test_df)
            v21_mae = np.mean(np.abs(test_df['tqi_value'].values - v21_pred))
            
            v22_pred = trident_v22_predict(train_df, test_df)
            v22_mae = np.mean(np.abs(test_df['tqi_value'].values - v22_pred))
            
            v22_win_hm = v22_mae < hm_mae
            v21_win_hm = v21_mae < hm_mae
            v22_win_v21 = v22_mae < v21_mae
            
            results.append({
                'mile': mile,
                'hm_mae': hm_mae,
                'v21_mae': v21_mae,
                'v22_mae': v22_mae,
                'v22_win_hm': v22_win_hm,
                'v21_win_hm': v21_win_hm,
                'v22_win_v21': v22_win_v21
            })
            
            print(f"{i:<4} {mile:<10} {hm_mae:<12.4f} {v21_mae:<12.4f} {v22_mae:<12.4f} {v22_win_hm!s:<10} {v21_win_hm!s:<10} {v22_win_v21!s:<12}")
            
        except Exception as e:
            print(f"{i:<4} {mile:<10} 错误: {str(e)[:30]}")
    
    # 统计
    print("\n" + "="*90)
    print("统计汇总")
    print("="*90)
    
    df_results = pd.DataFrame(results)
    n = len(df_results)
    
    print(f"\n有效样本: {n}/50")
    
    print(f"\n击败历史均值:")
    print(f"  v2.1: {df_results['v21_win_hm'].sum()} 个 ({df_results['v21_win_hm'].mean()*100:.1f}%)")
    print(f"  v2.2: {df_results['v22_win_hm'].sum()} 个 ({df_results['v22_win_hm'].mean()*100:.1f}%)")
    
    print(f"\n平均MAE:")
    print(f"  历史均值: {df_results['hm_mae'].mean():.4f} ± {df_results['hm_mae'].std():.4f}")
    print(f"  v2.1:     {df_results['v21_mae'].mean():.4f} ± {df_results['v21_mae'].std():.4f}")
    print(f"  v2.2:     {df_results['v22_mae'].mean():.4f} ± {df_results['v22_mae'].std():.4f}")
    
    print(f"\nv2.2 vs v2.1 直接对比:")
    v22_better = df_results['v22_win_v21'].sum()
    v21_better = n - v22_better
    print(f"  v2.2胜: {v22_better} 个 ({v22_better/n*100:.1f}%)")
    print(f"  v2.1胜: {v21_better} 个 ({v21_better/n*100:.1f}%)")
    
    # 大胜样本
    df_results['improvement_vs_v21'] = df_results['v21_mae'] - df_results['v22_mae']
    big_wins = df_results[df_results['improvement_vs_v21'] > 0.1].sort_values('improvement_vs_v21', ascending=False)
    
    print(f"\nv2.2 大胜样本 (vs v2.1 改善 > 0.1):")
    if len(big_wins) > 0:
        print(f"{'里程':<10} {'v2.1':<12} {'v2.2':<12} {'改善':<10}")
        for _, row in big_wins.head(5).iterrows():
            print(f"{int(row['mile']):<10} {row['v21_mae']:<12.4f} {row['v22_mae']:<12.4f} {row['improvement_vs_v21']:<+10.4f}")
    else:
        print("  无")
    
    # 大负样本
    big_losses = df_results[df_results['improvement_vs_v21'] < -0.1].sort_values('improvement_vs_v21')
    print(f"\nv2.2 表现较差样本 (vs v2.1 恶化 > 0.1):")
    if len(big_losses) > 0:
        print(f"{'里程':<10} {'v2.1':<12} {'v2.2':<12} {'恶化':<10}")
        for _, row in big_losses.head(5).iterrows():
            print(f"{int(row['mile']):<10} {row['v21_mae']:<12.4f} {row['v22_mae']:<12.4f} {row['improvement_vs_v21']:<+10.4f}")
    else:
        print("  无")
    
    print("\n" + "="*90)

if __name__ == '__main__':
    main()
