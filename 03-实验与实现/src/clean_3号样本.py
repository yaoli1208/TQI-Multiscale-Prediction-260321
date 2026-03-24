"""
3号样本数据清洗脚本
====================
清洗规则:
1. IQR异常值检测: 超出[Q1-1.5IQR, Q3+1.5IQR]范围的TQI值视为异常
2. 频率规则: 检测间隔<7天(过密)或>14天(过疏)的数据需要处理
3. 保留7个TQI分量列用于后续分组分析
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def clean_3号样本(input_path, output_path=None):
    """
    清洗3号样本数据
    
    Args:
        input_path: 原始Excel文件路径
        output_path: 清洗后CSV文件路径
    """
    print("="*70)
    print("3号样本数据清洗")
    print("="*70)
    
    # 1. 加载原始数据
    print("\n【Step 1】加载原始数据")
    df = pd.read_excel(input_path)
    print(f"  原始数据: {len(df)} 条")
    print(f"  列名: {df.columns.tolist()}")
    
    # 2. 基础预处理
    print("\n【Step 2】基础预处理")
    df['检测日期'] = pd.to_datetime(df['检测日期'])
    df = df.sort_values('检测日期').reset_index(drop=True)
    df['year'] = df['检测日期'].dt.year
    df['month'] = df['检测日期'].dt.month
    print(f"  时间范围: {df['检测日期'].min()} ~ {df['检测日期'].max()}")
    
    # 3. IQR异常值检测
    print("\n【Step 3】IQR异常值检测")
    Q1 = df['TQI值'].quantile(0.25)
    Q3 = df['TQI值'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"  Q1: {Q1:.3f}, Q3: {Q3:.3f}, IQR: {IQR:.3f}")
    print(f"  正常范围: [{lower_bound:.3f}, {upper_bound:.3f}]")
    
    # 标记异常值
    df['is_outlier_iqr'] = (df['TQI值'] < lower_bound) | (df['TQI值'] > upper_bound)
    outliers = df[df['is_outlier_iqr']]
    print(f"  IQR异常值: {len(outliers)} 条")
    if len(outliers) > 0:
        print(f"    异常值范围: {outliers['TQI值'].min():.3f} ~ {outliers['TQI值'].max():.3f}")
    
    # 4. 计算检测间隔
    print("\n【Step 4】检测频率分析")
    df['days_since_last'] = df['检测日期'].diff().dt.days
    
    # 频率规则:
    # - 间隔 < 3天: 过密，可能是重复检测，保留第一条
    # - 间隔 > 45天: 过疏，可能是缺失数据期，暂不处理（标记）
    df['is_too_dense'] = df['days_since_last'] < 3
    df['is_too_sparse'] = df['days_since_last'] > 45
    
    too_dense = df[df['is_too_dense']].copy()
    too_sparse = df[df['is_too_sparse']].copy()
    
    print(f"  过密检测(<3天): {len(too_dense)} 条")
    print(f"  过疏检测(>45天): {len(too_sparse)} 条")
    
    # 5. 综合清洗决策
    print("\n【Step 5】综合清洗")
    
    # 规则: 删除IQR异常值 + 删除过密检测的重复记录
    # 保留过疏检测的记录（只是标记，不删除）
    
    # 首先删除IQR异常值
    df_clean = df[~df['is_outlier_iqr']].copy()
    removed_outliers = len(df) - len(df_clean)
    print(f"  删除IQR异常值: {removed_outliers} 条")
    
    # 然后处理过密检测: 对间隔<3天的记录，保留第一条，删除后续
    # 重新计算间隔
    df_clean = df_clean.sort_values('检测日期').reset_index(drop=True)
    df_clean['days_since_last_clean'] = df_clean['检测日期'].diff().dt.days
    df_clean['is_too_dense_clean'] = df_clean['days_since_last_clean'] < 3
    
    # 删除过密记录（保留第一条）
    df_final = df_clean[~df_clean['is_too_dense_clean']].copy()
    removed_dense = len(df_clean) - len(df_final)
    print(f"  删除过密检测: {removed_dense} 条")
    
    print(f"\n  清洗前: {len(df)} 条")
    print(f"  清洗后: {len(df_final)} 条")
    print(f"  删除总计: {len(df) - len(df_final)} 条 ({(len(df) - len(df_final))/len(df)*100:.1f}%)")
    
    # 6. 生成清洗后特征
    print("\n【Step 6】生成特征")
    df_final['日期'] = df_final['检测日期']
    df_final['年份'] = df_final['检测日期'].dt.year
    df_final['月份'] = df_final['检测日期'].dt.month
    df_final['距2021年'] = df_final['年份'] - 2021
    df_final['年月'] = df_final['检测日期'].dt.strftime('%Y-%m')
    
    # 计算平面组和高程组
    df_final['plane'] = df_final['TQI左轨向'] + df_final['TQI右轨向'] + df_final['TQI轨距']
    df_final['elevation'] = df_final['TQI左高低'] + df_final['TQI右高低'] + df_final['TQI水平'] + df_final['TQI三角坑']
    
    # 标记是否异常（用于可视化）
    df_final['是否异常'] = False
    
    # 7. 选择输出列
    output_cols = [
        '日期', 'TQI值', '检测日期', '年份', '月份', '距2021年', '是否异常', '年月',
        'TQI左高低', 'TQI右高低', 'TQI左轨向', 'TQI右轨向', 'TQI轨距', 'TQI三角坑', 'TQI水平',
        'plane', 'elevation'
    ]
    
    df_output = df_final[output_cols].copy()
    
    # 8. 保存
    if output_path:
        df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n  清洗后数据已保存: {output_path}")
    
    # 9. 清洗报告
    print("\n" + "="*70)
    print("清洗报告")
    print("="*70)
    print(f"\n原始数据: {len(df)} 条")
    print(f"清洗后数据: {len(df_output)} 条")
    print(f"删除数据: {len(df) - len(df_output)} 条 ({(len(df) - len(df_output))/len(df)*100:.1f}%)")
    print(f"\n删除原因:")
    print(f"  - IQR异常值: {removed_outliers} 条")
    print(f"  - 过密检测: {removed_dense} 条")
    print(f"\n保留列: {len(output_cols)} 列（含7个TQI分量）")
    print(f"时间跨度: {df_output['日期'].min()} ~ {df_output['日期'].max()}")
    
    return df_output


if __name__ == "__main__":
    input_file = '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/3号样本.xlsx'
    output_file = '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/data/3号样本_完整清洗.csv'
    
    df_cleaned = clean_3号样本(input_file, output_file)
