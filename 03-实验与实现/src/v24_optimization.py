#!/usr/bin/env python3
"""
v2.3_no_seasonal 优化实验
============================
基于 v2.3_no_seasonal (当前最优，MAE 0.9496) 进行多方向优化
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
    baseline = train_df['tqi_value'].mean()
    y_pred = np.full(len(test_df), baseline)
    y_true = test_df['tqi_value'].values
    return np.mean(np.abs(y_true - y_pred)), np.std(y_true - y_pred)

def trident_v23_no_seasonal_baseline(train_df, test_df):
    """v2.3_no_seasonal 基线"""
    SHIFT_THRESHOLD = 0.3
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # 分布偏移检测
    recent_train = train_df.tail(int(len(train_df) * 0.2))
    recent_mean = recent_train['tqi_value'].mean()
    test_mean = test_df['tqi_value'].mean()
    has_shift = abs(test_mean - recent_mean) > SHIFT_THRESHOLD
    
    # 维修检测
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
    
    # 计算锚定值 (no_seasonal 版本)
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
    
    # 安全检查
    train_mean = train_df['tqi_value'].mean()
    train_std = train_df['tqi_value'].std()
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    y_pred = np.clip(y_pred, safe_min, safe_max)
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

# ============ 优化版本 ============

def v24_ema_anchor(train_df, test_df, alpha=0.3):
    """优化1: 指数移动平均作为锚定值"""
    SHIFT_THRESHOLD = 0.3
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    
    # EMA锚定值（更重视近期数据）
    values = train_df['tqi_value'].values
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
    
    y_pred = np.full(len(test_df), ema)
    y_true = test_df['tqi_value'].values
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

def v24_recent_weighted(train_df, test_df, recent_ratio=0.3):
    """优化2: 近期加权锚定值"""
    values = train_df['tqi_value'].values
    n = len(values)
    recent_n = int(n * recent_ratio)
    
    # 近期数据权重更高
    recent_mean = np.mean(values[-recent_n:])
    overall_mean = np.mean(values)
    
    # 加权融合
    anchor_val = 0.6 * recent_mean + 0.4 * overall_mean
    
    y_pred = np.full(len(test_df), anchor_val)
    y_true = test_df['tqi_value'].values
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

def v24_robust_stats(train_df, test_df):
    """优化3: 稳健统计量（中位数 + MAD）"""
    values = train_df['tqi_value'].values
    
    # 使用中位数代替均值
    median_val = np.median(values)
    # MAD (Median Absolute Deviation)
    mad = np.median(np.abs(values - median_val))
    
    # 稳健锚定值
    anchor_val = median_val
    
    y_pred = np.full(len(test_df), anchor_val)
    y_true = test_df['tqi_value'].values
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

def v24_trend_adjusted(train_df, test_df):
    """优化4: 趋势调整版本"""
    values = train_df['tqi_value'].values
    n = len(values)
    
    # 计算简单线性趋势
    x = np.arange(n)
    slope, intercept, _, _, _ = stats.linregress(x, values)
    
    # 当前锚定值
    current_anchor = intercept + slope * (n - 1)
    
    # 预测：锚定值 + 趋势延续
    predictions = []
    for i in range(len(test_df)):
        pred = current_anchor + slope * (i + 1)
        predictions.append(pred)
    
    y_pred = np.array(predictions)
    y_true = test_df['tqi_value'].values
    
    # 安全检查
    train_mean = np.mean(values)
    train_std = np.std(values)
    safe_min = max(0, train_mean - 5 * train_std)
    safe_max = train_mean + 5 * train_std
    y_pred = np.clip(y_pred, safe_min, safe_max)
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

def v24_quarterly_anchor(train_df, test_df):
    """优化5: 季度最近锚定值"""
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    
    # 获取最近一个季度的数据作为锚定
    last_date = train_df['date'].max()
    three_months_ago = last_date - pd.Timedelta(days=90)
    recent_quarter = train_df[train_df['date'] >= three_months_ago]
    
    if len(recent_quarter) >= 3:
        anchor_val = recent_quarter['tqi_value'].mean()
    else:
        anchor_val = train_df['tqi_value'].tail(30).mean()
    
    y_pred = np.full(len(test_df), anchor_val)
    y_true = test_df['tqi_value'].values
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

def v24_last_value(train_df, test_df):
    """优化6: 最后一个观测值（朴素预测）"""
    anchor_val = train_df['tqi_value'].iloc[-1]
    
    y_pred = np.full(len(test_df), anchor_val)
    y_true = test_df['tqi_value'].values
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

def v24_trimmed_mean(train_df, test_df, trim_ratio=0.1):
    """优化7: 截尾均值（去除极端值）"""
    values = train_df['tqi_value'].values
    
    # 截尾均值
    trimmed = stats.trim_mean(values, trim_ratio)
    
    y_pred = np.full(len(test_df), trimmed)
    y_true = test_df['tqi_value'].values
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

def v24_ensemble_simple(train_df, test_df):
    """优化8: 简单集成（多种锚定值平均）"""
    values = train_df['tqi_value'].values
    
    # 多种锚定值
    mean_val = np.mean(values)
    median_val = np.median(values)
    last_val = values[-1]
    recent_mean = np.mean(values[-30:]) if len(values) >= 30 else mean_val
    
    # 集成
    anchor_val = (mean_val + median_val + last_val + recent_mean) / 4
    
    y_pred = np.full(len(test_df), anchor_val)
    y_true = test_df['tqi_value'].values
    
    errors = y_true - y_pred
    return np.mean(np.abs(errors)), np.std(errors)

# ============ 主实验 ============
def main():
    print("="*100)
    print("v2.3_no_seasonal 优化实验")
    print("="*100)
    print("\n方法列表:")
    print("  1. 历史均值 (Baseline)")
    print("  2. v2.3_no_seasonal (当前最优)")
    print("  3. v2.4_ema (EMA锚定)")
    print("  4. v2.4_recent (近期加权)")
    print("  5. v2.4_robust (稳健统计)")
    print("  6. v2.4_trend (趋势调整)")
    print("  7. v2.4_quarter (季度锚定)")
    print("  8. v2.4_last (最后值)")
    print("  9. v2.4_trimmed (截尾均值)")
    print(" 10. v2.4_ensemble (简单集成)")
    print("\n数据划分: 70%训练 / 15%验证 / 15%测试")
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
        '历史均值': historical_mean_baseline,
        'v2.3_no_seasonal': trident_v23_no_seasonal_baseline,
        'v2.4_ema': v24_ema_anchor,
        'v2.4_recent': v24_recent_weighted,
        'v2.4_robust': v24_robust_stats,
        'v2.4_trend': v24_trend_adjusted,
        'v2.4_quarter': v24_quarterly_anchor,
        'v2.4_last': v24_last_value,
        'v2.4_trimmed': v24_trimmed_mean,
        'v2.4_ensemble': v24_ensemble_simple,
    }
    
    # 存储结果
    results = {name: [] for name in methods.keys()}
    
    print(f"\n开始实验...")
    print(f"{'#':<4} {'里程':<10}", end='')
    for name in methods.keys():
        print(f" {name:<16}", end='')
    print()
    print("-" * 180)
    
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
                row_results[name] = mae
            
            print(f"{i:<4} {mile:<10}", end='')
            for name in methods.keys():
                print(f" {row_results[name]:<16.4f}", end='')
            print()
            
        except Exception as e:
            print(f"{i:<4} {mile:<10} 错误: {str(e)[:50]}")
    
    # 统计汇总
    print("\n" + "="*100)
    print("统计汇总")
    print("="*100)
    
    n = len(results['历史均值'])
    print(f"\n有效样本: {n}/50")
    
    # MAE统计
    print(f"\n{'方法':<20} {'平均MAE':<12} {'标准差':<12} {'击败HM':<12} {'击败v2.3':<12}")
    print("-" * 70)
    hm_results = results['历史均值']
    v23_results = results['v2.3_no_seasonal']
    
    for name, mae_list in results.items():
        mean_mae = np.mean(mae_list)
        std_mae = np.std(mae_list)
        win_vs_hm = sum(1 for i in range(len(mae_list)) if mae_list[i] < hm_results[i])
        win_vs_v23 = sum(1 for i in range(len(mae_list)) if mae_list[i] < v23_results[i])
        print(f"{name:<20} {mean_mae:<12.4f} {std_mae:<12.4f} {win_vs_hm}/{n:<8} {win_vs_v23}/{n}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df['mile'] = sample_list[:n]
    output_file = f'{BASE_DIR}/results/v24_optimization_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存: {output_file}")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    main()
