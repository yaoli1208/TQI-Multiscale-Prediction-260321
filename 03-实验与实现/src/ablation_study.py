"""
消融实验 - 验证实验F各组件贡献
==============================
对比:
1. 完整模型（锚定 + 季节性 + 劣化趋势）
2. 去掉季节性调整
3. 去掉劣化趋势
4. 只用锚定值（最简化）
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
sys.path.insert(0, '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/src')

from experiment_f_business_aware import BusinessAwarePredictor


def load_data(sample_name):
    """加载样本数据"""
    if sample_name == "3号":
        file_path = '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/data/3号样本_完整清洗.csv'
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['日期'])
        df['tqi'] = df['TQI值']
        df['tqi_lprf'] = df['TQI左高低']
        df['tqi_rprf'] = df['TQI右高低']
        df['tqi_laln'] = df['TQI左轨向']
        df['tqi_raln'] = df['TQI右轨向']
        df['tqi_gage'] = df['TQI轨距']
        df['tqi_warp1'] = df['TQI三角坑']
        df['tqi_xlvl'] = df['TQI水平']
    else:
        file_path = '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/5号样本.xlsx'
        df = pd.read_excel(file_path)
        df['date'] = pd.to_datetime(df['检测日期'])
        df['tqi'] = df['TQI值']
        df['tqi_lprf'] = df['TQI左高低']
        df['tqi_rprf'] = df['TQI右高低']
        df['tqi_laln'] = df['TQI左轨向']
        df['tqi_raln'] = df['TQI右轨向']
        df['tqi_gage'] = df['TQI轨距']
        df['tqi_warp1'] = df['TQI三角坑']
        df['tqi_xlvl'] = df['TQI水平']
    
    df = df.sort_values('date').reset_index(drop=True)
    return df


def ablation_study(sample_name):
    """消融实验"""
    print("="*70)
    print(f"  消融实验 - {sample_name}样本")
    print("="*70)
    
    df = load_data(sample_name)
    print(f"\n数据: {len(df)}条")
    
    # 时序划分
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # 学习业务模式
    predictor = BusinessAwarePredictor()
    all_data = pd.concat([train_df, val_df, test_df]).sort_values('date')
    predictor.analyze_historical_pattern(all_data)
    
    results = []
    
    # 1. 完整模型
    print("\n【1】完整模型（锚定+季节性+劣化趋势）")
    mae_full = evaluate_ablation(predictor, test_df, use_seasonal=True, use_trend=True)
    print(f"  MAE: {mae_full:.4f}")
    results.append({'variant': '完整模型', 'mae': mae_full})
    
    # 2. 去掉季节性
    print("\n【2】去掉季节性调整")
    mae_no_seasonal = evaluate_ablation(predictor, test_df, use_seasonal=False, use_trend=True)
    print(f"  MAE: {mae_no_seasonal:.4f} (变差{(mae_no_seasonal-mae_full)/mae_full*100:+.1f}%)")
    results.append({'variant': '-季节性', 'mae': mae_no_seasonal})
    
    # 3. 去掉劣化趋势
    print("\n【3】去掉劣化趋势")
    mae_no_trend = evaluate_ablation(predictor, test_df, use_seasonal=True, use_trend=False)
    print(f"  MAE: {mae_no_trend:.4f} (变差{(mae_no_trend-mae_full)/mae_full*100:+.1f}%)")
    results.append({'variant': '-劣化趋势', 'mae': mae_no_trend})
    
    # 4. 只用锚定值
    print("\n【4】只用锚定值（最简化）")
    mae_anchor_only = evaluate_ablation(predictor, test_df, use_seasonal=False, use_trend=False)
    print(f"  MAE: {mae_anchor_only:.4f} (变差{(mae_anchor_only-mae_full)/mae_full*100:+.1f}%)")
    results.append({'variant': '仅锚定值', 'mae': mae_anchor_only})
    
    # 汇总
    print("\n" + "="*70)
    print("  消融实验汇总")
    print("="*70)
    print(f"\n{'变体':<20} {'MAE':<10} {'相对完整模型':<15}")
    print("-"*50)
    for r in results:
        diff = (r['mae'] - mae_full) / mae_full * 100
        print(f"{r['variant']:<20} {r['mae']:<10.4f} {diff:+.1f}%")
    
    return results


def evaluate_ablation(predictor, test_df, use_seasonal=True, use_trend=True):
    """评估特定消融配置"""
    predictions = []
    actuals = []
    
    for idx, row in test_df.iterrows():
        target_date = row['date']
        target_month = target_date.month
        
        # 找最近锚定值
        anchor_year = target_date.year - 1
        anchor = None
        while anchor_year >= target_date.year - 3:
            if anchor_year in predictor.maintenance_anchors:
                anchor = predictor.maintenance_anchors[anchor_year]
                break
            anchor_year -= 1
        
        if anchor is None:
            predictions.append(predictor.plane_baseline + predictor.elevation_baseline)
            actuals.append(row['tqi'])
            continue
        
        tqi_base = anchor['tqi']
        
        # 季节性调整
        seasonal_adj = 0
        if use_seasonal and predictor.learned_seasonal_pattern is not None:
            if target_month in predictor.learned_seasonal_pattern.index:
                seasonal_adj = predictor.learned_seasonal_pattern[target_month]
        
        # 劣化趋势调整
        trend_adj = 0
        if use_trend:
            year_diff = target_date.year - anchor_year
            monthly_degradation = -0.01
            if hasattr(predictor, 'degradation_rate'):
                monthly_degradation = predictor.degradation_rate / 12
            trend_adj = monthly_degradation * (year_diff * 12)
        
        pred_tqi = tqi_base + seasonal_adj + trend_adj
        predictions.append(pred_tqi)
        actuals.append(row['tqi'])
    
    y_true = np.array(actuals)
    y_pred = np.array(predictions)
    
    return mean_absolute_error(y_true, y_pred)


if __name__ == "__main__":
    # 跑两个样本
    results_5 = ablation_study("5号")
    print("\n\n")
    results_3 = ablation_study("3号")
    
    # 最终汇总
    print("\n\n" + "="*70)
    print("  消融实验对比：5号 vs 3号样本")
    print("="*70)
    print(f"\n{'组件':<15} {'5号影响':<15} {'3号影响':<15}")
    print("-"*45)
    print(f"{'季节性':<15} {(results_5[1]['mae']-results_5[0]['mae'])/results_5[0]['mae']*100:+.1f}% {(results_3[1]['mae']-results_3[0]['mae'])/results_3[0]['mae']*100:+.1f}%")
    print(f"{'劣化趋势':<15} {(results_5[2]['mae']-results_5[0]['mae'])/results_5[0]['mae']*100:+.1f}% {(results_3[2]['mae']-results_3[0]['mae'])/results_3[0]['mae']*100:+.1f}%")
