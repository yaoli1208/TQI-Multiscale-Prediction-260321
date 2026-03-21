"""
实验E：年际劣化趋势量化
=============================================
利用2012-2025年长期TQI数据，量化年度劣化速率
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class AnnualDegradationExperiment:
    """年际劣化趋势实验类"""
    
    def __init__(self, data):
        """
        初始化实验
        
        Args:
            data: 完整数据字典（包含多年数据）
        """
        self.processed_data = data['processed']
        self.yearly_stats = {}
        self.degradation_rate = None
        
    def extract_yearly_data(self):
        """提取年度统计数据"""
        print("\n" + "=" * 60)
        print("【实验E】年际劣化趋势量化")
        print("=" * 60)
        
        df = self.processed_data.copy()
        
        # 按年份统计
        yearly = df.groupby('year').agg({
            'tqi': ['mean', 'std', 'min', 'max', 'count'],
            'date': ['min', 'max']
        }).reset_index()
        
        yearly.columns = ['year', 'tqi_mean', 'tqi_std', 'tqi_min', 'tqi_max', 
                         'count', 'date_min', 'date_max']
        
        print(f"\n[Step 1] 年度数据统计")
        print(f"  - 数据年份范围: {df['year'].min()} - {df['year'].max()}")
        print(f"  - 共 {len(yearly)} 个年度")
        print(f"  - 总样本数: {len(df)}")
        
        print(f"\n  {'年份':<8} {'样本数':<8} {'TQI均值':<10} {'TQI标准差':<10} {'TQI范围'}")
        print(f"  {'-'*60}")
        for _, row in yearly.iterrows():
            print(f"  {int(row['year']):<8} {int(row['count']):<8} "
                  f"{row['tqi_mean']:<10.3f} {row['tqi_std']:<10.3f} "
                  f"[{row['tqi_min']:.2f}, {row['tqi_max']:.2f}]")
        
        self.yearly_stats = yearly
        
        return yearly
    
    def calculate_degradation_rate(self):
        """计算年际劣化速率"""
        print("\n[Step 2] 年际劣化速率计算")
        
        yearly = self.yearly_stats
        
        # 方法1: 年度均值线性回归
        X = yearly['year'].values.reshape(-1, 1)
        y = yearly['tqi_mean'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(X, y)
        
        # 显著性检验
        n = len(yearly)
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        x_mean = np.mean(yearly['year'].values)
        ss_x = np.sum((yearly['year'].values - x_mean)**2)
        se_slope = np.sqrt(mse / ss_x)
        t_stat = slope / se_slope
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        print(f"\n  线性回归结果:")
        print(f"    - 模型: TQI均值 = {intercept:.4f} + {slope:.4f} × 年份")
        print(f"    - 斜率（年劣化速率）: {slope:+.4f} /年")
        print(f"    - R²: {r_squared:.4f}")
        print(f"    - p-value: {p_value:.4f}")
        print(f"    - 显著性: {'✓ 显著' if p_value < 0.05 else '✗ 不显著'}")
        
        self.degradation_rate = {
            'method': 'linear_regression',
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': '劣化' if slope > 0 else '改善'
        }
        
        # 方法2: 逐年变化率
        yearly['tqi_change'] = yearly['tqi_mean'].diff()
        yearly['tqi_change_rate'] = yearly['tqi_mean'].pct_change() * 100
        
        print(f"\n  逐年变化率:")
        print(f"  {'年份':<8} {'TQI均值':<10} {'较上年变化':<12} {'变化率'}")
        print(f"  {'-'*45}")
        for _, row in yearly.iterrows():
            change = row['tqi_change']
            rate = row['tqi_change_rate']
            if pd.notna(change):
                print(f"  {int(row['year']):<8} {row['tqi_mean']:<10.3f} "
                      f"{change:+10.3f} {rate:+8.2f}%")
        
        avg_change = yearly['tqi_change'].mean()
        print(f"\n  平均年变化: {avg_change:+.4f}")
        
        return self.degradation_rate
    
    def seasonal_degradation_analysis(self):
        """分季节劣化分析"""
        print("\n[Step 3] 分季节劣化分析")
        
        df = self.processed_data.copy()
        
        # 定义季节
        season_map = {
            12: '冬季', 1: '冬季', 2: '冬季',
            3: '春季', 4: '春季', 5: '春季',
            6: '夏季', 7: '夏季', 8: '夏季',
            9: '秋季', 10: '秋季', 11: '秋季'
        }
        df['season'] = df['month'].map(season_map)
        
        # 按年份和季节统计
        yearly_seasonal = df.groupby(['year', 'season'])['tqi'].mean().unstack()
        
        print(f"\n  各季节年度TQI均值:")
        print(yearly_seasonal.round(3).to_string())
        
        # 计算各季节的劣化速率
        print(f"\n  各季节劣化速率:")
        print(f"  {'季节':<10} {'劣化速率(/年)':<15} {'R²':<10} {'趋势'}")
        print(f"  {'-'*50}")
        
        seasonal_slopes = {}
        for season in ['春季', '夏季', '秋季', '冬季']:
            if season in yearly_seasonal.columns:
                y = yearly_seasonal[season].dropna().values
                years = yearly_seasonal[season].dropna().index.values
                
                if len(y) >= 3:
                    X = years.reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X, y)
                    slope = model.coef_[0]
                    r2 = model.score(X, y)
                    trend = '↑劣化' if slope > 0.01 else ('↓改善' if slope < -0.01 else '→稳定')
                    
                    seasonal_slopes[season] = {'slope': slope, 'r2': r2}
                    print(f"  {season:<10} {slope:+12.4f}     {r2:<10.4f} {trend}")
        
        return seasonal_slopes
    
    def cross_year_stability(self):
        """跨年稳定性分析"""
        print("\n[Step 4] 跨年稳定性分析")
        
        df = self.processed_data.copy()
        
        # 计算每年TQI的变异系数（稳定性指标）
        yearly_cv = df.groupby('year')['tqi'].agg(['mean', 'std']).reset_index()
        yearly_cv['cv'] = yearly_cv['std'] / yearly_cv['mean'] * 100  # 变异系数(%)
        
        print(f"\n  年度变异系数（CV = 标准差/均值 × 100%）:")
        print(f"  {'年份':<8} {'TQI均值':<10} {'标准差':<10} {'变异系数(CV)':<15} {'稳定性'}")
        print(f"  {'-'*65}")
        for _, row in yearly_cv.iterrows():
            cv = row['cv']
            stability = '高' if cv < 5 else ('中' if cv < 10 else '低')
            print(f"  {int(row['year']):<8} {row['mean']:<10.3f} {row['std']:<10.3f} "
                  f"{cv:<15.2f}% {stability}")
        
        # 年度间CV趋势
        X = yearly_cv['year'].values.reshape(-1, 1)
        y = yearly_cv['cv'].values
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        
        print(f"\n  变异系数趋势: {slope:+.4f} %/年")
        print(f"  解读: {'稳定性下降' if slope > 0 else '稳定性提升'}")
        
        return yearly_cv
    
    def generate_report(self):
        """生成实验报告"""
        print("\n" + "=" * 60)
        print("【实验E总结报告】")
        print("=" * 60)
        
        rate = self.degradation_rate
        
        report = f"""
实验E：年际劣化趋势量化
================================

1. 数据概况:
   - 分析年份: {self.processed_data['year'].min()}-{self.processed_data['year'].max()}
   - 年数: {len(self.yearly_stats)} 年
   - 总样本: {len(self.processed_data)} 条

2. 年际劣化速率:
   - 线性趋势: {rate['slope']:+.4f} TQI单位/年
   - R²: {rate['r_squared']:.4f}
   - p-value: {rate['p_value']:.4f}
   - 显著性: {'✓ 显著' if rate['significant'] else '✗ 不显著'} (α=0.05)
   - 趋势判定: {rate['interpretation']}

3. 预测应用:
   - 若按当前速率，10年后TQI将变化: {rate['slope'] * 10:+.2f}
   - 建议: {'加强养护' if rate['slope'] > 0.1 else '维持现状' if rate['slope'] > 0 else '可适当延长周期'}

4. 结论:
   - 年际劣化趋势 {'✓ 已量化' if rate['significant'] else '✗ 不显著，无法可靠量化'}
"""
        print(report)
        
        return report
    
    def run(self):
        """运行完整实验"""
        # 1. 提取年度数据
        self.extract_yearly_data()
        
        # 2. 计算劣化速率
        self.calculate_degradation_rate()
        
        # 3. 分季节分析
        self.seasonal_degradation_analysis()
        
        # 4. 跨年稳定性
        self.cross_year_stability()
        
        # 5. 生成报告
        report = self.generate_report()
        
        return {
            'yearly_stats': self.yearly_stats,
            'degradation_rate': self.degradation_rate,
            'report': report
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/src')
    
    from data_loader import TQIDataLoader
    
    # 注意：这里需要用完整的历史数据文件路径
    loader = TQIDataLoader('/path/to/historical_tqi_data.xlsx')
    data = loader.run()
    
    exp = AnnualDegradationExperiment(data)
    results = exp.run()
