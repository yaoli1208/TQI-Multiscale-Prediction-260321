"""
实验A：多尺度分解有效性验证
=============================================
验证TQI是否可分解为趋势+季节+残差
验证7-9月稳定性假设
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class STLDecompositionExperiment:
    """STL分解实验类"""
    
    def __init__(self, train_data):
        """
        初始化实验
        
        Args:
            train_data: 训练集DataFrame
        """
        self.train_data = train_data.copy()
        self.tqi_series = train_data['tqi'].values
        self.dates = train_data['date'].values
        self.results = {}
        
    def simple_decomposition(self, period=6):
        """
        简化版时间序列分解（移动平均法）
        
        Args:
            period: 季节周期（这里用6个检测周期≈2个月）
        
        Returns:
            trend, seasonal, residual
        """
        print("\n" + "=" * 60)
        print("【实验A】多尺度分解有效性验证")
        print("=" * 60)
        
        tqi = self.tqi_series
        n = len(tqi)
        
        print(f"\n[Step 1] 时间序列分解")
        print(f"  - 样本数: {n}")
        print(f"  - 假设季节周期: {period} 个检测点")
        
        # ========== 趋势项：中心化移动平均 ==========
        trend = np.zeros(n)
        half_period = period // 2
        
        for i in range(n):
            start = max(0, i - half_period)
            end = min(n, i + half_period + 1)
            trend[i] = np.mean(tqi[start:end])
        
        # 去趋势
        detrended = tqi - trend
        
        # ========== 季节项：周期平均 ==========
        seasonal = np.zeros(n)
        for i in range(period):
            indices = np.arange(i, n, period)
            if len(indices) > 0:
                seasonal_avg = np.mean(detrended[indices])
                seasonal[indices] = seasonal_avg
        
        # 季节项零均值化
        seasonal = seasonal - np.mean(seasonal)
        
        # ========== 残差项 ==========
        residual = tqi - trend - seasonal
        
        self.results['trend'] = trend
        self.results['seasonal'] = seasonal
        self.results['residual'] = residual
        self.results['period'] = period
        
        print(f"  ✓ 趋势项范围: [{trend.min():.3f}, {trend.max():.3f}]")
        print(f"  ✓ 季节项范围: [{seasonal.min():.3f}, {seasonal.max():.3f}]")
        print(f"  ✓ 残差项范围: [{residual.min():.3f}, {residual.max():.3f}]")
        
        return trend, seasonal, residual
    
    def variance_analysis(self):
        """方差分析"""
        print("\n[Step 2] 方差分解分析")
        
        tqi = self.tqi_series
        trend = self.results['trend']
        seasonal = self.results['seasonal']
        residual = self.results['residual']
        
        # 计算各方差
        total_var = np.var(tqi)
        trend_var = np.var(trend)
        seasonal_var = np.var(seasonal)
        residual_var = np.var(residual)
        
        # 方差占比
        trend_ratio = trend_var / total_var * 100
        seasonal_ratio = seasonal_var / total_var * 100
        residual_ratio = residual_var / total_var * 100
        
        print(f"\n  {'成分':<12} {'方差':<12} {'占比':<10} {'阈值判定'}")
        print(f"  {'-'*50}")
        print(f"  {'趋势项':<10} {trend_var:<12.4f} {trend_ratio:<10.1f}% {'✓' if trend_ratio > 30 else '✗'} (>30%)")
        print(f"  {'季节项':<10} {seasonal_var:<12.4f} {seasonal_ratio:<10.1f}% {'✓' if seasonal_ratio > 20 else '✗'} (>20%)")
        print(f"  {'残差项':<10} {residual_var:<12.4f} {residual_ratio:<10.1f}%")
        print(f"  {'-'*50}")
        print(f"  {'总计':<10} {total_var:<12.4f} {'100.0%':<10}")
        
        self.results['variance'] = {
            'total': total_var,
            'trend': trend_var,
            'seasonal': seasonal_var,
            'residual': residual_var,
            'trend_ratio': trend_ratio,
            'seasonal_ratio': seasonal_ratio,
            'residual_ratio': residual_ratio
        }
        
        # 判定结果
        trend_ok = trend_ratio > 30
        seasonal_ok = seasonal_ratio > 20
        
        print(f"\n  判定结果:")
        print(f"    - 趋势项解释力: {'通过' if trend_ok else '未通过'} (目标>30%)")
        print(f"    - 季节项解释力: {'通过' if seasonal_ok else '未通过'} (目标>20%)")
        
        return trend_ok and seasonal_ok
    
    def summer_stability_test(self):
        """7-9月稳定性验证"""
        print("\n[Step 3] 7-9月稳定性验证")
        
        df = self.train_data.copy()
        df['residual'] = self.results['residual']
        
        # 按月份分组计算残差标准差
        monthly_std = df.groupby('month')['residual'].agg(['std', 'mean', 'count'])
        
        print(f"\n  各月残差标准差:")
        print(f"  {'月份':<8} {'标准差':<10} {'均值':<10} {'样本数':<8}")
        print(f"  {'-'*40}")
        for month, row in monthly_std.iterrows():
            marker = " ← 夏季" if month in [7, 8, 9] else ""
            print(f"  {month:<8} {row['std']:<10.4f} {row['mean']:<10.4f} {int(row['count']):<8}{marker}")
        
        # 夏季vs其他月份比较
        summer_months = [7, 8, 9]
        summer_std = monthly_std.loc[summer_months, 'std'].mean()
        other_std = monthly_std.drop(summer_months)['std'].mean()
        
        reduction = (other_std - summer_std) / other_std * 100
        
        print(f"\n  统计结果:")
        print(f"    - 夏季(7-9月)平均残差标准差: {summer_std:.4f}")
        print(f"    - 其他月份平均残差标准差: {other_std:.4f}")
        print(f"    - 夏季降低幅度: {reduction:.1f}%")
        
        # Levene方差齐性检验
        summer_residuals = df[df['month'].isin(summer_months)]['residual'].values
        other_residuals = df[~df['month'].isin(summer_months)]['residual'].values
        
        stat, pvalue = stats.levene(summer_residuals, other_residuals)
        
        print(f"\n  Levene方差齐性检验:")
        print(f"    - 统计量: {stat:.4f}")
        print(f"    - p-value: {pvalue:.4f}")
        print(f"    - 判定: {'夏季更稳定' if pvalue < 0.05 and reduction > 0 else '差异不显著'}")
        
        self.results['summer_test'] = {
            'summer_std': summer_std,
            'other_std': other_std,
            'reduction': reduction,
            'levene_stat': stat,
            'levene_pvalue': pvalue,
            'stable': pvalue < 0.05 and reduction > 30
        }
        
        return reduction > 30
    
    def generate_report(self):
        """生成实验报告"""
        print("\n" + "=" * 60)
        print("【实验A总结报告】")
        print("=" * 60)
        
        var = self.results['variance']
        summer = self.results['summer_test']
        
        report = f"""
实验A：多尺度分解有效性验证
================================

1. 方差分解结果:
   - 趋势项方差: {var['trend']:.4f} ({var['trend_ratio']:.1f}%)
   - 季节项方差: {var['seasonal']:.4f} ({var['seasonal_ratio']:.1f}%)
   - 残差项方差: {var['residual']:.4f} ({var['residual_ratio']:.1f}%)

2. 判定标准:
   - 趋势项解释力 > 30%: {'✓ 通过' if var['trend_ratio'] > 30 else '✗ 未通过'}
   - 季节项解释力 > 20%: {'✓ 通过' if var['seasonal_ratio'] > 20 else '✗ 未通过'}

3. 7-9月稳定性:
   - 夏季残差标准差: {summer['summer_std']:.4f}
   - 其他月份标准差: {summer['other_std']:.4f}
   - 降低幅度: {summer['reduction']:.1f}% {'✓ 通过(>30%)' if summer['reduction'] > 30 else '✗ 未通过'}
   - Levene检验p值: {summer['levene_pvalue']:.4f}

4. 综合判定: {'✓ 多尺度分解有效' if (var['trend_ratio'] > 20 and var['seasonal_ratio'] > 10) else '✗ 需调整分解方法'}
"""
        print(report)
        
        return report
    
    def run(self):
        """运行完整实验"""
        # 1. STL分解
        trend, seasonal, residual = self.simple_decomposition(period=6)
        
        # 2. 方差分析
        var_ok = self.variance_analysis()
        
        # 3. 夏季稳定性检验
        summer_ok = self.summer_stability_test()
        
        # 4. 生成报告
        report = self.generate_report()
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'variance': self.results['variance'],
            'summer_test': self.results['summer_test'],
            'report': report
        }


if __name__ == "__main__":
    # 测试需要加载数据
    from data_loader import TQIDataLoader
    
    loader = TQIDataLoader('/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/2号样本.xlsx')
    data = loader.run()
    
    exp = STLDecompositionExperiment(data['train'])
    results = exp.run()
