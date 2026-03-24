#!/usr/bin/env python3
"""
491样本基线实验 - 小批量测试 (前50样本)
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 配置路径
BASE_DIR = '/root/.openclaw/workspace/02-research/active/2026-03-21-Trident-TQI预测研究/03-实验与实现'
DATA_FILE = f'{BASE_DIR}/data/raw/iic_tqi_all.xlsx'
SAMPLE_LIST_FILE = f'{BASE_DIR}/data/processed/qualified_miles_v2.txt'
RESULTS_DIR = f'{BASE_DIR}/results/baseline_491_experiment'

os.makedirs(RESULTS_DIR, exist_ok=True)

# 导入函数
sys.path.insert(0, f'{BASE_DIR}/src')
from baseline_491_experiment import (
    load_sample_data, split_data,
    historical_mean_baseline, moving_average_baseline, 
    holt_winters_baseline, mlp_baseline, lstm_baseline,
    timemixer_baseline, trident_rolling_anchor
)
from trident_v2 import trident_v2_baseline
from trident_v21 import trident_v21_baseline

# 加载样本列表
with open(SAMPLE_LIST_FILE, 'r') as f:
    sample_list = [int(float(line.strip())) for line in f if line.strip()]

print(f"总样本数: {len(sample_list)}")
print(f"\n{'='*60}")
print(f"小批量测试模式 (前50个样本)")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

# 只取前50个样本
test_samples = sample_list[:50]
print(f"\n测试样本: 前50个")

# 定义方法
methods = [
    ('historical_mean', historical_mean_baseline, '历史均值'),
    ('ma', moving_average_baseline, '移动平均'),
    ('holt_winters', holt_winters_baseline, 'Holt-Winters'),
    ('mlp', mlp_baseline, 'MLP'),
    ('lstm', lstm_baseline, 'LSTM'),
    ('timemixer', timemixer_baseline, 'TimeMixer'),
    ('trident_v1', trident_rolling_anchor, 'Trident v1'),
    ('trident_v2', trident_v2_baseline, 'Trident v2'),
    ('trident_v21', trident_v21_baseline, 'Trident v2.1')
]

results = []
method_times = {k: [] for k, _, _ in methods}

for i, mile in enumerate(test_samples, 1):
    print(f"\n[{i}/50] 样本: {mile}", flush=True)
    
    df = load_sample_data(mile)
    if df is None or len(df) < 50:
        print(f"  跳过: 数据不足 ({len(df) if df is not None else 0}条)", flush=True)
        continue
    
    train_df, val_df, test_df = split_data(df)
    result = {
        'tqi_mile': mile,
        'record_count': len(df),
        'train_count': len(train_df),
        'test_count': len(test_df)
    }
    
    for key, func, name in methods:
        start = time.time()
        try:
            metrics = func(train_df, test_df)
            elapsed = time.time() - start
            method_times[key].append(elapsed)
            
            result[f'{key}_mae'] = metrics['mae']
            result[f'{key}_rmse'] = metrics['rmse']
            result[f'{key}_mape'] = metrics['mape']
            
            status = f"MAE={metrics['mae']:.4f} ({elapsed:.1f}s)"
        except Exception as e:
            elapsed = time.time() - start
            method_times[key].append(elapsed)
            result[f'{key}_mae'] = float('nan')
            result[f'{key}_rmse'] = float('nan')
            result[f'{key}_mape'] = float('nan')
            status = f"失败: {str(e)[:40]}"
        
        print(f"  {name:15s}: {status}", flush=True)
    
    results.append(result)
    
    # 每10个样本保存一次中间结果
    if i % 10 == 0:
        pd.DataFrame(results).to_csv(f'{RESULTS_DIR}/test_50samples_temp.csv', index=False)
        print(f"  [已保存中间结果 - {i}/50]", flush=True)

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv(f'{RESULTS_DIR}/test_50samples_results.csv', index=False)

print(f"\n{'='*60}")
print(f"测试完成! 有效样本: {len(results)}/50")
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 汇总报告
print(f"\n{'='*60}")
print(f"测试结果汇总")
print(f"{'='*60}")

print("\n【各方法平均MAE (前50样本)】")
method_mae_summary = {}
for key, _, name in methods:
    mae_col = f'{key}_mae'
    if mae_col in results_df.columns:
        valid = results_df[mae_col].dropna()
        if len(valid) > 0:
            method_mae_summary[key] = {
                'name': name,
                'mean': float(valid.mean()),
                'median': float(valid.median()),
                'std': float(valid.std()),
                'valid': int(len(valid))
            }
            print(f"  {name:15s}: 均值={valid.mean():.4f}, 中位数={valid.median():.4f}, 有效={len(valid)}/50")

print("\n【MAE排名 (从低到高)】")
sorted_methods = sorted(method_mae_summary.items(), key=lambda x: x[1]['mean'])
for rank, (key, info) in enumerate(sorted_methods, 1):
    print(f"  {rank}. {info['name']:15s}: {info['mean']:.4f}")

print("\n【各方法平均耗时】")
for key, _, name in methods:
    times = method_times[key]
    if times:
        print(f"  {name:15s}: {np.mean(times):.2f}s/样本 (总: {np.sum(times):.1f}s)")

print("\n【全量491样本时间估算】")
total_time = 0
for key, _, name in methods:
    times = method_times[key]
    if times:
        est_time = np.mean(times) * 491 / 60
        total_time += est_time
        print(f"  {name:15s}: ~{est_time:.1f}分钟")
print(f"  {'总计':15s}: ~{total_time:.1f}分钟 ({total_time/60:.1f}小时)")

# 保存JSON报告
report = {
    'test_info': {
        'sample_count': 50,
        'valid_samples': len(results),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    },
    'method_mae': method_mae_summary,
    'method_time': {k: {'mean': float(np.mean(v)), 'total': float(np.sum(v))} for k, v in method_times.items() if v},
    'full_scale_estimate_hours': round(total_time / 60, 2)
}

with open(f'{RESULTS_DIR}/test_50samples_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n测试报告已保存:")
print(f"  - {RESULTS_DIR}/test_50samples_results.csv")
print(f"  - {RESULTS_DIR}/test_50samples_report.json")
