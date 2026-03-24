"""
实验F：三种策略对比脚本
=======================
对比:
1. 修后预测 (Post-Maintenance Prediction)
2. 滚动锚定预测 (Rolling Anchor Prediction)
3. 数据驱动预测 (Data-Driven/Trident)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os

sys.path.insert(0, '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/src')

from experiment_f_business_aware import BusinessAwarePredictor, ExperimentF


def load_data(sample_name):
    """加载样本数据"""
    if sample_name == "3号":
        file_path = '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/data/3号样本_完整清洗.csv'
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['日期'])
        df['tqi'] = df['TQI值']
        for col in ['TQI左高低', 'TQI右高低', 'TQI左轨向', 'TQI右轨向', 'TQI轨距', 'TQI三角坑', 'TQI水平']:
            df[col.replace('TQI', 'tqi_').replace('左', 'l').replace('右', 'r').replace('高低', 'prf').replace('轨向', 'aln').replace('轨距', 'gage').replace('三角坑', 'warp1').replace('水平', 'xlvl')] = df[col]
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


def data_driven_prediction(train_df, test_df):
    """数据驱动预测 - 历史均值基线"""
    historical_mean = train_df['tqi'].mean()
    y_pred = np.full(len(test_df), historical_mean)
    y_true = test_df['tqi'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'name': '数据驱动(历史均值)',
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'n_points': len(y_true)
    }


def rolling_anchor_prediction(predictor, train_df, val_df, test_df):
    """滚动锚定预测"""
    all_historical = pd.concat([train_df, val_df], ignore_index=True)
    
    predictions = []
    actuals = []
    
    for idx, row in test_df.iterrows():
        pred = predictor.predict_with_rolling_anchor(row['date'], all_historical)
        if pred:
            predictions.append(pred['tqi'])
            actuals.append(row['tqi'])
        else:
            predictions.append(all_historical['tqi'].mean())
            actuals.append(row['tqi'])
    
    y_true = np.array(actuals)
    y_pred = np.array(predictions)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'name': '滚动锚定预测',
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'n_points': len(y_true)
    }


def post_maintenance_summary(exp, train_df, val_df, test_df):
    """修后预测汇总"""
    # 复用ExperimentF的修后预测逻辑
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_data = all_data.sort_values('date').reset_index(drop=True)
    
    # 重新学习业务模式
    exp.business_predictor.analyze_historical_pattern(all_data)
    
    # 运行修后预测并获取结果
    results = exp._predict_post_maintenance_mode(all_data)
    
    if results['y_true'] is not None:
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'name': '修后预测',
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'n_points': len(y_true),
            'n_windows': len(results.get('window_stats', []))
        }
    
    return {'name': '修后预测', 'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'n_points': 0}


def compare_strategies(sample_name):
    """对比三种策略"""
    print("="*80)
    print(f"  实验F：三种策略对比 - {sample_name}样本")
    print("="*80)
    
    # 加载数据
    df = load_data(sample_name)
    print(f"\n数据概况: {len(df)}条, {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
    
    # 时序划分
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"划分: 训练{len(train_df)} / 验证{len(val_df)} / 测试{len(test_df)}")
    
    # 1. 数据驱动预测
    print("\n" + "-"*80)
    print("【策略1】数据驱动预测 (历史均值基线)")
    print("-"*80)
    result_dd = data_driven_prediction(train_df, test_df)
    print(f"  MAE: {result_dd['mae']:.4f}")
    print(f"  RMSE: {result_dd['rmse']:.4f}")
    print(f"  MAPE: {result_dd['mape']:.2f}%")
    
    # 初始化业务预测器
    predictor = BusinessAwarePredictor()
    predictor.analyze_historical_pattern(pd.concat([train_df, val_df]))
    
    # 2. 滚动锚定预测
    print("\n" + "-"*80)
    print("【策略2】滚动锚定预测")
    print("-"*80)
    print(f"  检测到 {len(predictor.maintenance_anchors)} 年锚定值")
    result_ra = rolling_anchor_prediction(predictor, train_df, val_df, test_df)
    print(f"  MAE: {result_ra['mae']:.4f}")
    print(f"  RMSE: {result_ra['rmse']:.4f}")
    print(f"  MAPE: {result_ra['mape']:.2f}%")
    
    # 3. 修后预测
    print("\n" + "-"*80)
    print("【策略3】修后预测")
    print("-"*80)
    exp = ExperimentF(train_df, val_df, test_df)
    result_pm = post_maintenance_summary(exp, train_df, val_df, test_df)
    print(f"  窗口数: {result_pm.get('n_windows', 0)}")
    print(f"  总预测点数: {result_pm['n_points']}")
    print(f"  MAE: {result_pm['mae']:.4f}")
    print(f"  RMSE: {result_pm['rmse']:.4f}")
    print(f"  MAPE: {result_pm['mape']:.2f}%")
    
    # 汇总对比
    print("\n" + "="*80)
    print("  策略对比汇总")
    print("="*80)
    print(f"\n{'策略':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'相对提升':<12}")
    print("-"*65)
    
    baseline_mae = result_dd['mae']
    
    for result in [result_dd, result_ra, result_pm]:
        improvement = (baseline_mae - result['mae']) / baseline_mae * 100
        improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        if result['name'] == '数据驱动(历史均值)':
            improvement_str = "(基线)"
        print(f"{result['name']:<20} {result['mae']:<10.4f} {result['rmse']:<10.4f} {result['mape']:<10.2f} {improvement_str:<12}")
    
    return {
        'sample': sample_name,
        'data_driven': result_dd,
        'rolling_anchor': result_ra,
        'post_maintenance': result_pm
    }


if __name__ == "__main__":
    # 对比两个样本
    results_5 = compare_strategies("5号")
    print("\n\n")
    results_3 = compare_strategies("3号")
    
    # 最终汇总
    print("\n\n" + "="*80)
    print("  最终对比：5号 vs 3号样本")
    print("="*80)
    print(f"\n{'样本':<8} {'修后MAE':<12} {'滚动MAE':<12} {'数据MAE':<12} {'最佳策略':<15}")
    print("-"*60)
    
    for results in [results_5, results_3]:
        sample = results['sample']
        pm_mae = results['post_maintenance']['mae']
        ra_mae = results['rolling_anchor']['mae']
        dd_mae = results['data_driven']['mae']
        
        best = min([(pm_mae, '修后预测'), (ra_mae, '滚动锚定'), (dd_mae, '数据驱动')])
        
        print(f"{sample:<8} {pm_mae:<12.4f} {ra_mae:<12.4f} {dd_mae:<12.4f} {best[1]:<15}")
