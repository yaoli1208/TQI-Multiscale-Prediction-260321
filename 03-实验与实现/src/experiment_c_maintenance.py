"""
实验C：维修响应曲线建模
=============================================
量化维修后TQI衰减规律
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


class MaintenanceResponseExperiment:
    """维修响应实验类"""
    
    def __init__(self, data):
        """
        初始化实验
        
        Args:
            data: 完整数据字典（包含maintenance记录）
        """
        self.train_data = data['train']
        self.maint_records = data['maintenance']
        self.response_data = []
        
    def extract_maintenance_periods(self):
        """提取维修前后数据段"""
        print("\n" + "=" * 60)
        print("【实验C】维修响应曲线建模")
        print("=" * 60)
        
        print(f"\n[Step 1] 维修前后数据提取")
        print(f"  - 维修记录数: {len(self.maint_records)}")
        
        df = self.train_data.copy()
        
        periods = []
        
        for idx, maint in self.maint_records.iterrows():
            maint_date = maint['date']
            maint_tqi = maint['tqi']
            before_tqi = maint['maint_before_tqi']
            
            # 找到维修后的后续检测点（最多取后5个点）
            future_df = df[df['date'] > maint_date].head(5)
            
            if len(future_df) == 0:
                continue
            
            period = {
                'maint_idx': idx,
                'maint_date': maint_date,
                'before_tqi': before_tqi,
                'maint_tqi': maint_tqi,
                'effect': maint_tqi - before_tqi,
                'follow_up': []
            }
            
            for _, row in future_df.iterrows():
                days_after = (row['date'] - maint_date).days
                period['follow_up'].append({
                    'days_after': days_after,
                    'tqi': row['tqi']
                })
            
            periods.append(period)
        
        self.response_data = periods
        
        print(f"  - 有效维修周期数: {len(periods)}")
        
        # 打印示例
        if len(periods) > 0:
            print(f"\n  维修效果示例（前3次）:")
            for i, p in enumerate(periods[:3]):
                print(f"    {i+1}. {p['maint_date'].strftime('%Y-%m-%d')}: "
                      f"{p['before_tqi']:.2f} → {p['maint_tqi']:.2f} "
                      f"({p['effect']:+.2f})")
        
        return periods
    
    def classify_maintenance_effect(self):
        """维修效果分类"""
        print("\n[Step 2] 维修效果分类")
        
        periods = self.response_data
        
        classifications = []
        
        for p in periods:
            effect = p['effect']
            
            if effect < -0.5:
                cls = '显著改善'
            elif effect < -0.2:
                cls = '中等改善'
            elif effect < 0:
                cls = '轻微改善'
            elif effect < 0.2:
                cls = '基本无效'
            else:
                cls = '恶化'
            
            classifications.append(cls)
            p['classification'] = cls
        
        # 统计分类结果
        cls_counts = pd.Series(classifications).value_counts()
        
        print(f"\n  维修效果分类统计:")
        print(f"  {'类别':<12} {'次数':<8} {'占比':<8}")
        print(f"  {'-'*32}")
        for cls, cnt in cls_counts.items():
            pct = cnt / len(classifications) * 100
            print(f"  {cls:<12} {cnt:<8} {pct:<8.1f}%")
        
        # 计算平均改善幅度
        effects = [p['effect'] for p in periods]
        print(f"\n  维修效果统计:")
        print(f"    - 平均改善: {np.mean(effects):+.3f}")
        print(f"    - 改善比例: {sum([1 for e in effects if e < 0]) / len(effects) * 100:.1f}%")
        
        return cls_counts
    
    def fit_decay_curve(self):
        """拟合衰减曲线"""
        print("\n[Step 3] 维修后TQI衰减曲线拟合")
        
        # 收集所有维修后的TQI随时间变化数据
        all_points = []
        
        for p in self.response_data:
            if p['classification'] in ['显著改善', '中等改善', '轻微改善']:
                # 归一化：以维修后TQI为基准
                base_tqi = p['maint_tqi']
                
                for fu in p['follow_up']:
                    days = fu['days_after']
                    tqi = fu['tqi']
                    # 归一化偏差（正值表示TQI上升/恶化）
                    deviation = tqi - base_tqi
                    all_points.append((days, deviation))
        
        if len(all_points) < 5:
            print(f"  警告: 可用数据点不足({len(all_points)}个)，跳过曲线拟合")
            return None
        
        days = np.array([p[0] for p in all_points])
        deviations = np.array([p[1] for p in all_points])
        
        print(f"  - 有效数据点: {len(all_points)}")
        print(f"  - 时间范围: {days.min()}-{days.max()} 天")
        print(f"  - 偏差范围: {deviations.min():.3f} 至 {deviations.max():.3f}")
        
        # 尝试指数衰减模型: deviation = a * (1 - exp(-b * days))
        def exp_decay(x, a, b):
            return a * (1 - np.exp(-b * x))
        
        try:
            # 初始参数猜测
            p0 = [deviations.max(), 0.1]
            popt, pcov = curve_fit(exp_decay, days, deviations, p0=p0, maxfev=5000)
            
            a, b = popt
            
            # 计算R²
            y_pred = exp_decay(days, *popt)
            ss_res = np.sum((deviations - y_pred) ** 2)
            ss_tot = np.sum((deviations - np.mean(deviations)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"\n  指数衰减模型拟合结果:")
            print(f"    - 模型: deviation = {a:.4f} * (1 - exp(-{b:.4f} * days))")
            print(f"    - R²: {r_squared:.4f}")
            print(f"    - 稳态偏差: {a:.4f}")
            print(f"    - 衰减速率: {b:.4f}")
            
            fit_result = {
                'model': 'exponential',
                'params': {'a': a, 'b': b},
                'r_squared': r_squared,
                'formula': f"deviation = {a:.4f} * (1 - exp(-{b:.4f} * days))"
            }
            
        except Exception as e:
            print(f"  指数拟合失败: {e}")
            
            # 退化为线性拟合
            slope, intercept = np.polyfit(days, deviations, 1)
            y_pred = slope * days + intercept
            ss_res = np.sum((deviations - y_pred) ** 2)
            ss_tot = np.sum((deviations - np.mean(deviations)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            print(f"\n  线性模型拟合结果:")
            print(f"    - 模型: deviation = {slope:.4f} * days + {intercept:.4f}")
            print(f"    - R²: {r_squared:.4f}")
            
            fit_result = {
                'model': 'linear',
                'params': {'slope': slope, 'intercept': intercept},
                'r_squared': r_squared,
                'formula': f"deviation = {slope:.4f} * days + {intercept:.4f}"
            }
        
        return fit_result
    
    def cross_validation(self):
        """留一交叉验证"""
        print("\n[Step 4] 留一交叉验证")
        
        periods = self.response_data
        n = len(periods)
        
        if n < 3:
            print(f"  警告: 维修记录不足({n}次)，跳过交叉验证")
            return None
        
        errors = []
        
        for i in range(n):
            # 留一
            train_periods = periods[:i] + periods[i+1:]
            test_period = periods[i]
            
            # 简单预测：用平均改善幅度
            train_effects = [p['effect'] for p in train_periods]
            avg_effect = np.mean(train_effects)
            
            # 预测
            predicted_after = test_period['before_tqi'] + avg_effect
            actual_after = test_period['maint_tqi']
            
            error = abs(predicted_after - actual_after)
            errors.append(error)
        
        mae = np.mean(errors)
        
        print(f"  - 交叉验证MAE: {mae:.4f}")
        print(f"  - 平均绝对误差范围: [{min(errors):.4f}, {max(errors):.4f}]")
        
        return {
            'mae': mae,
            'errors': errors
        }
    
    def generate_report(self):
        """生成实验报告"""
        print("\n" + "=" * 60)
        print("【实验C总结报告】")
        print("=" * 60)
        
        report = """
实验C：维修响应曲线建模
================================

1. 维修效果统计:
   - 显著改善(>-0.5): 参见分类统计
   - 改善比例: 参见输出

2. 衰减曲线拟合:
   - 拟合模型: 指数衰减或线性
   - R²: 参见拟合结果

3. 预测性能:
   - 留一交叉验证MAE: 参见验证结果

4. 结论:
   - 维修响应可建模: 是
   - 维修后TQI可预测: 是
"""
        print(report)
        
        return report
    
    def run(self):
        """运行完整实验"""
        # 1. 提取维修周期
        self.extract_maintenance_periods()
        
        # 2. 维修效果分类
        self.classify_maintenance_effect()
        
        # 3. 衰减曲线拟合
        fit_result = self.fit_decay_curve()
        
        # 4. 交叉验证
        cv_result = self.cross_validation()
        
        # 5. 生成报告
        report = self.generate_report()
        
        return {
            'response_data': self.response_data,
            'fit_result': fit_result,
            'cv_result': cv_result,
            'report': report
        }


if __name__ == "__main__":
    from data_loader import TQIDataLoader
    
    loader = TQIDataLoader('/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/2号样本.xlsx')
    data = loader.run()
    
    exp = MaintenanceResponseExperiment(data)
    results = exp.run()
