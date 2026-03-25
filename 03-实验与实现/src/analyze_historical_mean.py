#!/usr/bin/env python3
"""
历史均值深度分析 - 找出它的弱点
分析50个样本，找出历史均值表现差的情况
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v3.txt'

def load_all_data():
    print("加载数据...")
    df = pd.read_excel(DATA_FILE)
    df.columns = df.columns.str.strip()
    df = df[df['dete_dt'].astype(str).str.strip() != 'dete_dt'].copy()
    df['date'] = pd.to_datetime(df['dete_dt'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['tqi_value'] = pd.to_numeric(df['tqi_val'], errors='coerce')
    df['tqi_mile'] = pd.to_numeric(df['tqi_mile'], errors='coerce')
    df = df.dropna(subset=['tqi_value', 'tqi_mile'])
    return df[['tqi_mile', 'date', 'tqi_value']].copy()

def analyze_sample(all_df, mile):
    """分析单个样本的特征"""
    sample_df = all_df[all_df['tqi_mile'] == mile].copy()
    sample_df = sample_df.sort_values('date').reset_index(drop=True)
    
    if len(sample_df) < 50:
        return None
    
    # 数据划分
    n = len(sample_df)
    train_end = int(n * 0.7)
    test_start = int(n * 0.85)
    
    train_df = sample_df.iloc[:train_end].copy()
    test_df = sample_df.iloc[test_start:].copy()
    
    # 历史均值预测
    hm_pred = train_df['tqi_value'].mean()
    hm_mae = np.mean(np.abs(test_df['tqi_value'].values - hm_pred))
    
    # 分析特征
    train_df['year'] = train_df['date'].dt.year
    train_df['month'] = train_df['date'].dt.month
    
    # 1. 趋势特征
    yearly_mean = train_df.groupby('year')['tqi_value'].mean()
    yearly_trend = np.polyfit(range(len(yearly_mean)), yearly_mean.values, 1)[0] if len(yearly_mean) >= 3 else 0
    
    # 2. 季节性特征
    monthly_std = train_df.groupby('month')['tqi_value'].mean().std()
    
    # 3. 7-9月稳定性
    summer_df = train_df[train_df['month'].isin([7, 8, 9])]
    summer_yearly = summer_df.groupby('year')['tqi_value'].mean()
    summer_cv = summer_yearly.std() / summer_yearly.mean() if summer_yearly.mean() > 0 else 0
    
    # 4. 近期vs远期差异
    recent_mean = train_df.tail(int(len(train_df)*0.3))['tqi_value'].mean()
    old_mean = train_df.head(int(len(train_df)*0.3))['tqi_value'].mean()
    recency_diff = abs(recent_mean - old_mean)
    
    # 5. 测试集vs训练集差异（历史均值的误差来源）
    test_mean = test_df['tqi_value'].mean()
    train_mean = train_df['tqi_value'].mean()
    distribution_shift = abs(test_mean - train_mean)
    
    return {
        'mile': mile,
        'hm_mae': hm_mae,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'yearly_trend': yearly_trend,
        'seasonal_strength': monthly_std,
        'summer_cv': summer_cv,
        'recency_diff': recency_diff,
        'distribution_shift': distribution_shift,
        'train_mean': train_mean,
        'test_mean': test_mean
    }

def main():
    print("="*90)
    print("历史均值深度分析 - 找出它的弱点")
    print("="*90)
    
    with open(SAMPLE_LIST_FILE, 'r') as f:
        sample_list = [int(line.strip()) for line in f if line.strip()]
    sample_list = sample_list[:50]
    
    all_df = load_all_data()
    print(f"数据加载完成: {len(all_df)} 条记录")
    
    results = []
    for mile in sample_list:
        result = analyze_sample(all_df, mile)
        if result:
            results.append(result)
    
    df = pd.DataFrame(results)
    
    print(f"\n有效样本: {len(df)}/50")
    print(f"\n历史均值MAE统计:")
    print(f"  均值: {df['hm_mae'].mean():.4f}")
    print(f"  中位数: {df['hm_mae'].median():.4f}")
    print(f"  标准差: {df['hm_mae'].std():.4f}")
    print(f"  范围: [{df['hm_mae'].min():.4f}, {df['hm_mae'].max():.4f}]")
    
    # 找出历史均值表现最差和最好的样本
    worst = df.nlargest(10, 'hm_mae')[['mile', 'hm_mae', 'distribution_shift', 'yearly_trend', 'recency_diff']]
    best = df.nsmallest(10, 'hm_mae')[['mile', 'hm_mae', 'distribution_shift', 'yearly_trend', 'recency_diff']]
    
    print(f"\n{'='*90}")
    print("历史均值表现最差的10个样本 (MAE最高)")
    print(f"{'='*90}")
    print(worst.to_string(index=False))
    
    print(f"\n{'='*90}")
    print("历史均值表现最好的10个样本 (MAE最低)")
    print(f"{'='*90}")
    print(best.to_string(index=False))
    
    # 相关性分析
    print(f"\n{'='*90}")
    print("特征与历史均值MAE的相关性")
    print(f"{'='*90}")
    
    features = ['yearly_trend', 'seasonal_strength', 'summer_cv', 'recency_diff', 'distribution_shift']
    for feat in features:
        corr = df['hm_mae'].corr(df[feat])
        print(f"  {feat:20s}: {corr:+.4f}")
    
    # 洞察
    print(f"\n{'='*90}")
    print("关键洞察")
    print(f"{'='*90}")
    
    # 分布偏移大的样本
    high_shift = df[df['distribution_shift'] > df['distribution_shift'].quantile(0.8)]
    print(f"\n1. 分布偏移大的样本 (Top 20%):")
    print(f"   数量: {len(high_shift)}")
    print(f"   平均MAE: {high_shift['hm_mae'].mean():.4f}")
    print(f"   整体平均MAE: {df['hm_mae'].mean():.4f}")
    print(f"   恶化: {(high_shift['hm_mae'].mean() / df['hm_mae'].mean() - 1)*100:.1f}%")
    
    # 趋势强的样本
    high_trend = df[abs(df['yearly_trend']) > abs(df['yearly_trend']).quantile(0.8)]
    print(f"\n2. 趋势强的样本 (Top 20%):")
    print(f"   数量: {len(high_trend)}")
    print(f"   平均MAE: {high_trend['hm_mae'].mean():.4f}")
    
    # 近期变化大的样本
    high_recency = df[df['recency_diff'] > df['recency_diff'].quantile(0.8)]
    print(f"\n3. 近期变化大的样本 (Top 20%):")
    print(f"   数量: {len(high_recency)}")
    print(f"   平均MAE: {high_recency['hm_mae'].mean():.4f}")
    
    print(f"\n{'='*90}")
    print("结论：历史均值在以下情况表现差")
    print(f"{'='*90}")
    print("1. 训练集和测试集分布不一致（TQI水平变化）")
    print("2. 有明显的时间趋势（上升/下降）")
    print("3. 近期数据与早期数据差异大")
    print("\n→ Trident应该重点针对这些情况优化！")
    print(f"{'='*90}")

if __name__ == '__main__':
    main()
