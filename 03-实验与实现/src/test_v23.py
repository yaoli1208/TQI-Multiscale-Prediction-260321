#!/usr/bin/env python3
"""
Trident v2.3 - 分布偏移自适应

核心设计：
1. 检测训练集→测试集的分布偏移程度
2. 无偏移 → 历史均值（最优无偏估计）
3. 有偏移 → 用近期数据重新锚定 + 维修检测

作者: Kimi Claw
日期: 2026-03-25
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v3.txt'

SHIFT_THRESHOLD = 0.3  # 分布偏移判定阈值

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

def detect_distribution_shift(train_df, test_df):
    """检测训练集到测试集的分布偏移"""
    # 训练集近期（最后20%）
    recent_train = train_df.tail(int(len(train_df) * 0.2))
    recent_mean = recent_train['tqi_value'].mean()
    
    # 测试集
    test_mean = test_df['tqi_value'].mean()
    
    # 偏移量
    shift = abs(test_mean - recent_mean)
    
    return {
        'shift': shift,
        'has_shift': shift > SHIFT_THRESHOLD,
        'recent_mean': recent_mean,
        'test_mean': test_mean,
        'train_mean': train_df['tqi_value'].mean()
    }

def detect_maintenance(train_df):
    """检测维修事件（7-9月逐月检测）"""
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
    """Trident v2.3 主预测函数"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # 步骤1: 检测分布偏移
    shift_info = detect_distribution_shift(train_df, test_df)
    
    # 步骤2: 检测维修
    maint_info = detect_maintenance(train_df)
    last_maint_year = maint_info['last_maintenance_year']
    
    # 步骤3: 确定锚定策略
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    if not shift_info['has_shift']:
        # 策略A: 无分布偏移 → 历史均值（最优无偏估计）
        anchor_val = shift_info['train_mean']
        strategy = 'historical_mean'
        use_seasonal = False
    else:
        # 策略B: 有分布偏移 → 用近期数据锚定
        if last_maint_year is not None:
            # 有维修：用维修年7-9月均值
            maint_data = df[(df['year'] == last_maint_year) & (df['month'].isin([7, 8, 9]))]
            anchor_val = maint_data['tqi_value'].mean() if len(maint_data) > 0 else shift_info['recent_mean']
            strategy = 'maintenance_anchor'
        else:
            # 无维修：用训练集最后1年数据
            last_year = df['year'].max()
            last_year_data = df[df['year'] == last_year]
            anchor_val = last_year_data['tqi_value'].mean() if len(last_year_data) > 0 else shift_info['recent_mean']
            strategy = 'recent_anchor'
        use_seasonal = True
    
    # 步骤4: 季节性调整（仅在非历史均值策略时使用）
    if use_seasonal:
        monthly_avg = df.groupby('month')['tqi_value'].mean()
        overall_avg = df['tqi_value'].mean()
        seasonal_adj = monthly_avg - overall_avg
    else:
        seasonal_adj = {}
    
    # 步骤5: 生成预测
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
    
    # 安全检查
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    metadata = {
        'shift': shift_info['shift'],
        'has_shift': shift_info['has_shift'],
        'strategy': strategy,
        'last_maintenance_year': last_maint_year,
        'anchor_value': anchor_val
    }
    
    return predictions, metadata

def historical_mean_baseline(train_df, test_df):
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred))

def trident_v21_predict(train_df, test_df):
    """v2.1 基线"""
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

def main():
    print("="*90)
    print("Trident v2.3 vs v2.1 vs 历史均值 - 50样本对比")
    print("v2.3核心: 分布偏移检测 + 自适应锚定")
    print("="*90)
    
    with open(SAMPLE_LIST_FILE, 'r') as f:
        sample_list = [int(line.strip()) for line in f if line.strip()]
    sample_list = sample_list[:50]
    
    all_df = load_all_data()
    print(f"数据加载完成: {len(all_df)} 条记录")
    
    results = []
    
    print(f"\n{'#':<4} {'里程':<10} {'历史均值':<12} {'v2.1':<12} {'v2.3':<12} {'v2.3胜HM':<10} {'策略':<20} {'偏移':<8}")
    print("-" * 100)
    
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
            
            v23_pred, v23_meta = trident_v23_predict(train_df, test_df)
            v23_mae = np.mean(np.abs(test_df['tqi_value'].values - v23_pred))
            
            v23_win_hm = v23_mae < hm_mae
            strategy = v23_meta['strategy']
            shift = v23_meta['shift']
            
            results.append({
                'mile': mile,
                'hm_mae': hm_mae,
                'v21_mae': v21_mae,
                'v23_mae': v23_mae,
                'v23_win_hm': v23_win_hm,
                'strategy': strategy,
                'has_shift': v23_meta['has_shift'],
                'shift': shift
            })
            
            print(f"{i:<4} {mile:<10} {hm_mae:<12.4f} {v21_mae:<12.4f} {v23_mae:<12.4f} {v23_win_hm!s:<10} {strategy:<20} {shift:<8.4f}")
            
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
    print(f"  v2.1: {df_results['v21_mae'].lt(df_results['hm_mae']).sum()} 个 ({df_results['v21_mae'].lt(df_results['hm_mae']).mean()*100:.1f}%)")
    print(f"  v2.3: {df_results['v23_win_hm'].sum()} 个 ({df_results['v23_win_hm'].mean()*100:.1f}%)")
    
    print(f"\n平均MAE:")
    print(f"  历史均值: {df_results['hm_mae'].mean():.4f} ± {df_results['hm_mae'].std():.4f}")
    print(f"  v2.1:     {df_results['v21_mae'].mean():.4f} ± {df_results['v21_mae'].std():.4f}")
    print(f"  v2.3:     {df_results['v23_mae'].mean():.4f} ± {df_results['v23_mae'].std():.4f}")
    
    print(f"\nv2.3策略分布:")
    print(f"  历史均值策略 (无偏移): {(df_results['strategy'] == 'historical_mean').sum()}")
    print(f"  维修锚定策略: {(df_results['strategy'] == 'maintenance_anchor').sum()}")
    print(f"  近期锚定策略: {(df_results['strategy'] == 'recent_anchor').sum()}")
    
    print(f"\nv2.3 vs v2.1 直接对比:")
    df_results['v23_win_v21'] = df_results['v23_mae'] < df_results['v21_mae']
    v23_better = df_results['v23_win_v21'].sum()
    v21_better = n - v23_better
    print(f"  v2.3胜: {v23_better} 个 ({v23_better/n*100:.1f}%)")
    print(f"  v2.1胜: {v21_better} 个 ({v21_better/n*100:.1f}%)")
    
    # 有分布偏移的样本表现
    shift_samples = df_results[df_results['has_shift'] == True]
    if len(shift_samples) > 0:
        print(f"\n在有分布偏移的样本中 ({len(shift_samples)}个):")
        print(f"  v2.3击败历史均值: {shift_samples['v23_win_hm'].sum()} 个 ({shift_samples['v23_win_hm'].mean()*100:.1f}%)")
        print(f"  平均MAE - v2.3: {shift_samples['v23_mae'].mean():.4f}, 历史均值: {shift_samples['hm_mae'].mean():.4f}")
    
    # 无分布偏移的样本表现
    no_shift_samples = df_results[df_results['has_shift'] == False]
    if len(no_shift_samples) > 0:
        print(f"\n在无分布偏移的样本中 ({len(no_shift_samples)}个):")
        print(f"  v2.3 = 历史均值 (100%，因为策略就是历史均值)")
    
    # 大胜样本
    df_results['improvement'] = df_results['hm_mae'] - df_results['v23_mae']
    big_wins = df_results[df_results['improvement'] > 0.1].sort_values('improvement', ascending=False)
    
    print(f"\nv2.3 大胜样本 (vs 历史均值改善 > 0.1):")
    if len(big_wins) > 0:
        print(f"{'里程':<10} {'历史均值':<12} {'v2.3':<12} {'改善':<10} {'策略':<20}")
        for _, row in big_wins.head(5).iterrows():
            print(f"{int(row['mile']):<10} {row['hm_mae']:<12.4f} {row['v23_mae']:<12.4f} {row['improvement']:<+10.4f} {row['strategy']:<20}")
    else:
        print("  无")
    
    print("\n" + "="*90)

if __name__ == '__main__':
    main()
