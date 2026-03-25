#!/usr/bin/env python3
"""
简化版 v2.2 vs v2.1 对比测试 - 绕过导入问题
"""
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v3.txt'

# ============ 数据加载 ============
def load_sample_list():
    with open(SAMPLE_LIST_FILE, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

def load_sample_data(mile):
    df = pd.read_excel(DATA_FILE)
    df.columns = df.columns.str.strip()
    sample_df = df[df['tqi_mile'] == mile].copy()
    sample_df = sample_df.sort_values('dete_dt').reset_index(drop=True)
    sample_df['date'] = pd.to_datetime(sample_df['dete_dt'])
    sample_df['tqi_value'] = sample_df['tqi_val']
    return sample_df[['date', 'tqi_value']].copy()

def split_data(df):
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

# ============ 历史均值基线 ============
def historical_mean_baseline(train_df, test_df):
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    mae = np.mean(np.abs(y_true - y_pred))
    return {'mae': mae}

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

def trident_v21_baseline(train_df, test_df):
    predictions = trident_v21_predict(train_df, test_df)
    actuals = test_df['tqi_value'].values
    mae = np.mean(np.abs(actuals - predictions))
    return {'mae': mae}

# ============ Trident v2.2 ============
def detect_monthly_maintenance(train_df, threshold_factor=2.0):
    df = train_df.copy()
    df['date'] = pd.to_datetime(df['date'])
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
                    month_data['is_maintenance'] = month_data['change'] < -threshold_factor * change_std
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

def calculate_yearly_deterioration(train_df, last_maintenance_year=None, min_years=2):
    df = train_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    summer_df = df[df['date'].dt.month.isin([7, 8, 9])]
    yearly_summer = summer_df.groupby('year')['tqi_value'].mean().reset_index()
    
    if last_maintenance_year is not None:
        yearly_summer = yearly_summer[yearly_summer['year'] >= last_maintenance_year]
    
    if len(yearly_summer) < min_years + 1:
        return 0.0, False
    
    yearly_summer['change'] = yearly_summer['tqi_value'].diff()
    changes = yearly_summer['change'].dropna()
    
    if len(changes) == 0:
        return 0.0, False
    
    annual_deterioration = changes.median()
    change_std = changes.std()
    is_reliable = abs(annual_deterioration) > 0.5 * change_std
    
    return annual_deterioration, is_reliable

def trident_v22_predict(train_df, test_df):
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    maint_info = detect_monthly_maintenance(train_df)
    last_maint_year = maint_info['last_maintenance_year']
    
    annual_deterioration, is_reliable = calculate_yearly_deterioration(
        train_df, last_maintenance_year=last_maint_year
    )
    
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
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
        target_year = row['year']
        years_gap = target_year - last_train_year
        
        if is_reliable and abs(annual_deterioration) > 0.01:
            base_value = monthly_avg_last_year.get(target_month, df['tqi_value'].mean())
            prediction = base_value + annual_deterioration * years_gap
        else:
            prediction = monthly_avg_last_year.get(target_month, df['tqi_value'].mean())
        
        predictions.append(prediction)
    
    predictions = np.array(predictions)
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    return predictions

def trident_v22_baseline(train_df, test_df):
    predictions = trident_v22_predict(train_df, test_df)
    actuals = test_df['tqi_value'].values
    mae = np.mean(np.abs(actuals - predictions))
    return {'mae': mae}

# ============ 主程序 ============
if __name__ == '__main__':
    print("="*80)
    print("Trident v2.2 vs v2.1 对比测试 (前50样本)")
    print("="*80)
    
    sample_list = load_sample_list()[:50]
    print(f"样本数: {len(sample_list)}")
    
    results = []
    
    print(f"\n{'#':<4} {'里程':<10} {'历史均值':<12} {'v2.1 MAE':<12} {'v2.2 MAE':<12} {'v2.2胜':<8} {'v2.1胜':<8}")
    print("-" * 80)
    
    for i, mile in enumerate(sample_list, 1):
        try:
            df = load_sample_data(mile)
            if df is None or len(df) < 50:
                print(f"{i:<4} {mile:<10} 跳过(数据不足)")
                continue
            
            train_df, val_df, test_df = split_data(df)
            
            hm_result = historical_mean_baseline(train_df, test_df)
            v21_result = trident_v21_baseline(train_df, test_df)
            v22_result = trident_v22_baseline(train_df, test_df)
            
            hm_mae = hm_result['mae']
            v21_mae = v21_result['mae']
            v22_mae = v22_result['mae']
            
            v22_win = v22_mae < hm_mae
            v21_win = v21_mae < hm_mae
            
            results.append({
                'mile': mile,
                'hm_mae': hm_mae,
                'v21_mae': v21_mae,
                'v22_mae': v22_mae,
                'v22_win': v22_win,
                'v21_win': v21_win
            })
            
            print(f"{i:<4} {mile:<10} {hm_mae:<12.4f} {v21_mae:<12.4f} {v22_mae:<12.4f} {v22_win!s:<8} {v21_win!s:<8}")
            
        except Exception as e:
            print(f"{i:<4} {mile:<10} 错误: {e}")
    
    # 统计
    print("\n" + "="*80)
    print("统计汇总")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    
    print(f"\n有效样本: {len(df_results)}/50")
    print(f"\n击败历史均值:")
    print(f"  v2.1: {df_results['v21_win'].sum()} 个 ({df_results['v21_win'].mean()*100:.1f}%)")
    print(f"  v2.2: {df_results['v22_win'].sum()} 个 ({df_results['v22_win'].mean()*100:.1f}%)")
    
    print(f"\n平均MAE:")
    print(f"  历史均值: {df_results['hm_mae'].mean():.4f} ± {df_results['hm_mae'].std():.4f}")
    print(f"  v2.1:     {df_results['v21_mae'].mean():.4f} ± {df_results['v21_mae'].std():.4f}")
    print(f"  v2.2:     {df_results['v22_mae'].mean():.4f} ± {df_results['v22_mae'].std():.4f}")
    
    # v2.2 vs v2.1 直接对比
    df_results['v22_vs_v21'] = df_results['v22_mae'] < df_results['v21_mae']
    v22_better = df_results['v22_vs_v21'].sum()
    v21_better = (~df_results['v22_vs_v21']).sum()
    
    print(f"\nv2.2 vs v2.1 直接对比:")
    print(f"  v2.2胜: {v22_better} 个 ({v22_better/len(df_results)*100:.1f}%)")
    print(f"  v2.1胜: {v21_better} 个 ({v21_better/len(df_results)*100:.1f}%)")
    
    # v2.2大胜样本
    df_results['improvement'] = df_results['v21_mae'] - df_results['v22_mae']
    big_wins = df_results[df_results['improvement'] > 0.1].sort_values('improvement', ascending=False)
    
    print(f"\nv2.2 大胜样本 (改善 > 0.1):")
    if len(big_wins) > 0:
        print(f"{'里程':<10} {'v2.1':<12} {'v2.2':<12} {'改善':<10}")
        for _, row in big_wins.head(10).iterrows():
            print(f"{int(row['mile']):<10} {row['v21_mae']:<12.4f} {row['v22_mae']:<12.4f} {row['improvement']:<+10.4f}")
    else:
        print("  无")
    
    # v2.2大负样本
    big_losses = df_results[df_results['improvement'] < -0.1].sort_values('improvement')
    print(f"\nv2.2 表现较差样本 (恶化 > 0.1):")
    if len(big_losses) > 0:
        print(f"{'里程':<10} {'v2.1':<12} {'v2.2':<12} {'恶化':<10}")
        for _, row in big_losses.head(10).iterrows():
            print(f"{int(row['mile']):<10} {row['v21_mae']:<12.4f} {row['v22_mae']:<12.4f} {row['improvement']:<+10.4f}")
    else:
        print("  无")
    
    print("\n" + "="*80)
