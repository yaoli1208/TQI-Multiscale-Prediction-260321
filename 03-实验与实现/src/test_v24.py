#!/usr/bin/env python3
"""
Trident v2.4 - 自适应多锚定融合

核心设计：
1. 三个锚定：历史均值、近期锚定、维修锚定
2. 权重不是固定的，而是基于各锚定在训练集上的验证表现自适应调整
3. 验证表现好的锚定获得更高权重

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
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()

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

def trident_v24_predict(train_df, val_df, test_df):
    """
    Trident v2.4: 自适应多锚定融合
    
    使用验证集评估各锚定表现，自适应调整权重
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    for df in [train_df, val_df, test_df]:
        df['date'] = pd.to_datetime(df['date'])
    
    df = train_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # ========== 构建三个锚定 ==========
    
    # 锚定1: 历史均值
    historical_mean = train_df['tqi_value'].mean()
    
    # 锚定2: 近期锚定（训练集最后1年）
    last_year = df['year'].max()
    last_year_data = df[df['year'] == last_year]
    recent_anchor = last_year_data['tqi_value'].mean() if len(last_year_data) > 0 else historical_mean
    
    # 锚定3: 维修锚定
    last_maint_year = detect_maintenance(train_df)
    if last_maint_year is not None:
        maint_data = df[(df['year'] == last_maint_year) & (df['month'].isin([7, 8, 9]))]
        maint_anchor = maint_data['tqi_value'].mean() if len(maint_data) > 0 else recent_anchor
    else:
        maint_anchor = recent_anchor
    
    # 季节性调整
    monthly_avg = df.groupby('month')['tqi_value'].mean()
    overall_avg = df['tqi_value'].mean()
    seasonal_adj = monthly_avg - overall_avg
    
    # ========== 验证集评估各锚定 ==========
    
    def evaluate_anchor(anchor_val, eval_df):
        """评估单个锚定在验证集上的表现"""
        predictions = []
        for _, row in eval_df.iterrows():
            month = row['date'].month
            seasonal = seasonal_adj.get(month, 0)
            pred = anchor_val + seasonal
            predictions.append(pred)
        predictions = np.array(predictions)
        actuals = eval_df['tqi_value'].values
        mae = np.mean(np.abs(actuals - predictions))
        return mae
    
    # 在验证集上评估三个锚定
    mae_hist = evaluate_anchor(historical_mean, val_df)
    mae_recent = evaluate_anchor(recent_anchor, val_df)
    mae_maint = evaluate_anchor(maint_anchor, val_df)
    
    # ========== 自适应权重（基于验证表现）==========
    
    # 使用逆MAE作为置信度（表现越好，权重越高）
    # 加小值防止除零
    confidence_hist = 1.0 / (mae_hist + 0.01)
    confidence_recent = 1.0 / (mae_recent + 0.01)
    confidence_maint = 1.0 / (mae_maint + 0.01)
    
    # Softmax归一化得到权重
    confidences = np.array([confidence_hist, confidence_recent, confidence_maint])
    weights = np.exp(confidences) / np.sum(np.exp(confidences))
    
    w_hist, w_recent, w_maint = weights
    
    # ========== 融合锚定 ==========
    
    fused_anchor = w_hist * historical_mean + w_recent * recent_anchor + w_maint * maint_anchor
    
    # ========== 生成预测 ==========
    
    predictions = []
    for _, row in test_df.iterrows():
        month = row['date'].month
        seasonal = seasonal_adj.get(month, 0)
        pred = fused_anchor + seasonal
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # 安全检查
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    metadata = {
        'weights': {'historical': w_hist, 'recent': w_recent, 'maintenance': w_maint},
        'maes': {'historical': mae_hist, 'recent': mae_recent, 'maintenance': mae_maint},
        'anchors': {'historical': historical_mean, 'recent': recent_anchor, 'maintenance': maint_anchor},
        'fused_anchor': fused_anchor,
        'last_maint_year': last_maint_year
    }
    
    return predictions, metadata

def historical_mean_baseline(train_df, test_df):
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred))

def trident_v23_predict(train_df, test_df, shift_threshold=0.3):
    """v2.3 baseline for comparison"""
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
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    predictions = np.clip(predictions, safe_min, safe_max)
    
    return predictions

def main():
    print("="*110)
    print("Trident v2.4 - 自适应多锚定融合 vs v2.3 vs 历史均值")
    print("="*110)
    
    with open(SAMPLE_LIST_FILE, 'r') as f:
        sample_list = [int(line.strip()) for line in f if line.strip()]
    sample_list = sample_list[:50]
    
    all_df = load_all_data()
    print(f"数据加载完成: {len(all_df)} 条记录\n")
    
    results = []
    
    print(f"{'#':<4} {'里程':<10} {'历史均值':<12} {'v2.3':<12} {'v2.4':<12} {'v2.4胜':<10} {'权重(H/R/M)':<25}")
    print("-" * 110)
    
    for i, mile in enumerate(sample_list, 1):
        try:
            df = load_sample_data(all_df, mile)
            if df is None or len(df) < 50:
                print(f"{i:<4} {mile:<10} 跳过(数据不足)")
                continue
            
            train_df, val_df, test_df = split_data(df)
            y_true = test_df['tqi_value'].values
            
            # 历史均值
            hm_mae = historical_mean_baseline(train_df, test_df)
            
            # v2.3
            v23_pred = trident_v23_predict(train_df, test_df)
            v23_mae = np.mean(np.abs(y_true - v23_pred))
            
            # v2.4
            v24_pred, v24_meta = trident_v24_predict(train_df, val_df, test_df)
            v24_mae = np.mean(np.abs(y_true - v24_pred))
            
            v24_win = v24_mae < hm_mae
            weights = v24_meta['weights']
            weight_str = f"{weights['historical']:.2f}/{weights['recent']:.2f}/{weights['maintenance']:.2f}"
            
            results.append({
                'mile': mile,
                'hm_mae': hm_mae,
                'v23_mae': v23_mae,
                'v24_mae': v24_mae,
                'v24_win': v24_win,
                'w_hist': weights['historical'],
                'w_recent': weights['recent'],
                'w_maint': weights['maintenance']
            })
            
            print(f"{i:<4} {mile:<10} {hm_mae:<12.4f} {v23_mae:<12.4f} {v24_mae:<12.4f} {v24_win!s:<10} {weight_str:<25}")
            
        except Exception as e:
            print(f"{i:<4} {mile:<10} 错误: {str(e)[:40]}")
    
    # 统计汇总
    print("\n" + "="*110)
    print("统计汇总")
    print("="*110)
    
    df_results = pd.DataFrame(results)
    n = len(df_results)
    
    print(f"\n有效样本: {n}/50")
    
    print(f"\n各方法平均MAE:")
    print(f"  历史均值: {df_results['hm_mae'].mean():.4f} ± {df_results['hm_mae'].std():.4f}")
    print(f"  v2.3:     {df_results['v23_mae'].mean():.4f} ± {df_results['v23_mae'].std():.4f}")
    print(f"  v2.4:     {df_results['v24_mae'].mean():.4f} ± {df_results['v24_mae'].std():.4f}")
    
    print(f"\n击败历史均值:")
    print(f"  v2.3: {(df_results['v23_mae'] < df_results['hm_mae']).sum()} 个 ({(df_results['v23_mae'] < df_results['hm_mae']).mean()*100:.1f}%)")
    print(f"  v2.4: {df_results['v24_win'].sum()} 个 ({df_results['v24_win'].mean()*100:.1f}%)")
    
    print(f"\nv2.4 vs v2.3 对比:")
    df_results['v24_win_v23'] = df_results['v24_mae'] < df_results['v23_mae']
    v24_better = df_results['v24_win_v23'].sum()
    print(f"  v2.4胜: {v24_better} 个 ({v24_better/n*100:.1f}%)")
    print(f"  v2.3胜: {n - v24_better} 个 ({(n-v24_better)/n*100:.1f}%)")
    
    print(f"\nv2.4 平均权重分配:")
    print(f"  历史均值: {df_results['w_hist'].mean():.3f}")
    print(f"  近期锚定: {df_results['w_recent'].mean():.3f}")
    print(f"  维修锚定: {df_results['w_maint'].mean():.3f}")
    
    # 大胜样本
    df_results['improvement'] = df_results['hm_mae'] - df_results['v24_mae']
    big_wins = df_results[df_results['improvement'] > 0.1].sort_values('improvement', ascending=False)
    
    print(f"\nv2.4 大胜样本 (vs 历史均值改善 > 0.1):")
    if len(big_wins) > 0:
        print(f"{'里程':<10} {'历史均值':<12} {'v2.4':<12} {'改善':<10} {'权重(H/R/M)':<20}")
        for _, row in big_wins.head(5).iterrows():
            w_str = f"{row['w_hist']:.2f}/{row['w_recent']:.2f}/{row['w_maint']:.2f}"
            print(f"{int(row['mile']):<10} {row['hm_mae']:<12.4f} {row['v24_mae']:<12.4f} {row['improvement']:<+10.4f} {w_str:<20}")
    else:
        print("  无")
    
    print("\n" + "="*110)

if __name__ == '__main__':
    main()
