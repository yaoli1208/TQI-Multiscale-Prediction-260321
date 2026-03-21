#!/usr/bin/env python3
"""
多尺度TQI预测实验 - 主程序
=============================================
运行全部实验并生成报告

Usage:
    python run_experiments.py
"""

import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import TQIDataLoader
from experiment_a_stl import STLDecompositionExperiment
from experiment_b_winter import WinterPhaseExperiment
from experiment_c_maintenance import MaintenanceResponseExperiment
from experiment_d_prediction import ExperimentD

import json
from datetime import datetime


def main():
    """主程序"""
    print("\n" + "="*80)
    print("  多尺度TQI预测实验 - 完整实验流程")
    print("  基于2号样本（2024-01-06 至 2025-12-14）")
    print("="*80)
    
    # ========== 数据路径 ==========
    data_path = '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/19d0f20d-3f32-8628-8000-0000b1563466_2号样本.xlsx'
    
    # ========== Step 1: 数据加载 ==========
    print("\n" + "#"*80)
    print("# Step 1: 数据加载与预处理")
    print("#"*80)
    
    loader = TQIDataLoader(data_path)
    data = loader.run()
    
    # ========== Step 2: 实验A - STL分解 ==========
    print("\n" + "#"*80)
    print("# Step 2: 实验A - 多尺度分解有效性验证")
    print("#"*80)
    
    exp_a = STLDecompositionExperiment(data['train'])
    results_a = exp_a.run()
    
    # ========== Step 3: 实验B - 冬季三阶段 ==========
    print("\n" + "#"*80)
    print("# Step 3: 实验B - 冬季三阶段存在性验证")
    print("#"*80)
    
    exp_b = WinterPhaseExperiment(data['train'])
    results_b = exp_b.run()
    
    # ========== Step 4: 实验C - 维修响应 ==========
    print("\n" + "#"*80)
    print("# Step 4: 实验C - 维修响应曲线建模")
    print("#"*80)
    
    exp_c = MaintenanceResponseExperiment(data)
    results_c = exp_c.run()
    
    # ========== Step 5: 实验D - 多尺度融合预测 ==========
    print("\n" + "#"*80)
    print("# Step 5: 实验D - 多尺度融合预测模型")
    print("#"*80)
    
    exp_d = ExperimentD(data)
    results_d = exp_d.run()
    
    # ========== Step 6: 汇总报告 ==========
    print("\n" + "="*80)
    print("  实验汇总报告")
    print("="*80)
    
    summary = f"""
==============================================
多尺度TQI预测实验 - 汇总报告
==============================================

实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据来源: 2号样本.xlsx
样本信息:
  - 总记录数: {len(data['processed'])}
  - 训练集: {len(data['train'])}
  - 验证集: {len(data['val'])}
  - 测试集: {len(data['test'])}
  - 维修记录: {len(data['maintenance'])}

【实验A】多尺度分解有效性
--------------------------------
{results_a['report']}

【实验B】冬季三阶段存在性
--------------------------------
{results_b['report']}

【实验C】维修响应曲线建模
--------------------------------
{results_c['report']}

==============================================
实验完成！
==============================================
"""
    
    print(summary)
    
    # 保存汇总报告
    output_dir = '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/results'
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'experiment_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n报告已保存至: {report_path}")
    
    # 保存结构化结果
    results_dict = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'data_path': data_path,
            'sample_count': len(data['processed'])
        },
        'experiment_a': {
            'variance': results_a['variance'],
            'summer_test': results_a['summer_test']
        },
        'experiment_b': {
            'phase_stats': results_b['phase_stats'].to_dict() if hasattr(results_b['phase_stats'], 'to_dict') else {},
            'anova': {
                'f_stat': results_b['anova']['f_stat'],
                'p_value': results_b['anova']['p_value'],
                'anova_ok': results_b['anova']['anova_ok']
            } if results_b['anova'] else {}
        },
        'experiment_c': {
            'fit_result': results_c['fit_result'],
            'cv_mae': results_c['cv_result']['mae'] if results_c['cv_result'] else None
        }
    }
    
    json_path = os.path.join(output_dir, 'experiment_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"结构化结果已保存至: {json_path}")
    
    print("\n" + "="*80)
    print("  全部实验完成！")
    print("="*80)


if __name__ == "__main__":
    main()
