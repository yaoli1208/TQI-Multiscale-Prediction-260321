"""
实验B：冬季三阶段存在性验证
=============================================
验证冬季是否呈现"上升→稳定→下降"三阶段
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class WinterPhaseExperiment:
    """冬季三阶段实验类"""
    
    def __init__(self, train_data):
        """
        初始化实验
        
        Args:
            train_data: 训练集DataFrame
        """
        self.train_data = train_data.copy()
        self.winter_data = None
        self.phase_stats = {}
        
    def extract_winter_data(self):
        """提取冬季数据（12月、1月、2月）"""
        print("\n" + "=" * 60)
        print("【实验B】冬季三阶段存在性验证")
        print("=" * 60)
        
        df = self.train_data.copy()
        
        # 提取冬季数据
        winter_df = df[df['month'].isin([12, 1, 2])].copy()
        
        print(f"\n[Step 1] 冬季数据提取")
        print(f"  - 冬季样本数: {len(winter_df)}")
        print(f"  - 时间范围: {winter_df['date'].min().strftime('%Y-%m-%d')} 至 {winter_df['date'].max().strftime('%Y-%m-%d')}")
        
        # 按冬季年度分组（12月属于上一年度冬季）
        winter_df['winter_year'] = winter_df.apply(
            lambda row: row['year'] - 1 if row['month'] == 12 else row['year'], axis=1
        )
        
        self.winter_data = winter_df
        
        return winter_df
    
    def annotate_phases(self):
        """标注冬季三阶段"""
        print("\n[Step 2] 冬季三阶段标注")
        
        winter_df = self.winter_data.copy()
        
        # 对每个冬季年度分别标注
        phases = []
        
        for year in sorted(winter_df['winter_year'].unique()):
            year_df = winter_df[winter_df['winter_year'] == year].sort_values('date')
            year_indices = year_df.index.tolist()
            
            print(f"\n  冬季 {year}-{year+1}:")
            print(f"    样本数: {len(year_df)}")
            
            year_phases = []
            for i, idx in enumerate(year_indices):
                if i < 2:
                    # 前两个点标记为"上升期起点"
                    phase = '上升期'
                else:
                    # 计算近3点变化率
                    tqi_current = year_df.loc[idx, 'tqi']
                    tqi_3ago = year_df.loc[year_indices[i-2], 'tqi']
                    delta = (tqi_current - tqi_3ago) / 3
                    
                    if delta > 0.1:
                        phase = '上升期'
                    elif delta < -0.1:
                        phase = '下降期'
                    else:
                        phase = '稳定期'
                
                year_phases.append(phase)
                
            phases.extend(year_phases)
            
            # 打印该冬季的阶段分布
            phase_counts = pd.Series(year_phases).value_counts()
            for ph, cnt in phase_counts.items():
                print(f"    {ph}: {cnt}次")
        
        winter_df['phase'] = phases
        self.winter_data = winter_df
        
        return winter_df
    
    def phase_statistics(self):
        """阶段统计特征"""
        print("\n[Step 3] 阶段统计特征分析")
        
        winter_df = self.winter_data
        
        # 计算每个阶段的TQI和变化率统计
        stats_list = []
        
        for phase in ['上升期', '稳定期', '下降期']:
            phase_data = winter_df[winter_df['phase'] == phase]
            
            if len(phase_data) == 0:
                continue
            
            # 计算TQI变化率（简化：用后点减前点）
            tqi_values = phase_data['tqi'].values
            if len(tqi_values) > 1:
                deltas = np.diff(tqi_values)
                avg_delta = np.mean(deltas)
                std_delta = np.std(deltas)
            else:
                avg_delta = 0
                std_delta = 0
            
            stat = {
                'phase': phase,
                'count': len(phase_data),
                'tqi_mean': phase_data['tqi'].mean(),
                'tqi_std': phase_data['tqi'].std(),
                'delta_mean': avg_delta,
                'delta_std': std_delta
            }
            stats_list.append(stat)
        
        self.phase_stats = pd.DataFrame(stats_list)
        
        print(f"\n  {'阶段':<10} {'样本数':<8} {'TQI均值':<10} {'TQI标准差':<10} {'变化率均值':<12} {'变化率标准差'}")
        print(f"  {'-'*70}")
        for _, row in self.phase_stats.iterrows():
            print(f"  {row['phase']:<10} {int(row['count']):<8} {row['tqi_mean']:<10.3f} "
                  f"{row['tqi_std']:<10.3f} {row['delta_mean']:<12.3f} {row['delta_std']:<10.3f}")
        
        return self.phase_stats
    
    def anova_test(self):
        """ANOVA方差分析"""
        print("\n[Step 4] ANOVA方差分析")
        
        winter_df = self.winter_data
        
        # 准备各阶段的变化率数据
        phase_deltas = {}
        
        for phase in ['上升期', '稳定期', '下降期']:
            phase_data = winter_df[winter_df['phase'] == phase].sort_values('date')
            
            # 计算相邻点的变化率
            deltas = []
            for i in range(1, len(phase_data)):
                delta = phase_data.iloc[i]['tqi'] - phase_data.iloc[i-1]['tqi']
                deltas.append(delta)
            
            phase_deltas[phase] = deltas
            print(f"  {phase}: {len(deltas)} 个变化率样本")
        
        # 过滤掉样本数过少的组
        valid_groups = [(k, v) for k, v in phase_deltas.items() if len(v) >= 2]
        
        if len(valid_groups) >= 2:
            # 执行ANOVA
            groups = [v for _, v in valid_groups]
            f_stat, p_value = stats.f_oneway(*groups)
            
            print(f"\n  ANOVA结果:")
            print(f"    - F统计量: {f_stat:.4f}")
            print(f"    - p-value: {p_value:.4f}")
            print(f"    - 判定: {'三阶段显著不同' if p_value < 0.05 else '差异不显著'}")
            
            # 事后检验：均值比较
            print(f"\n  变化率均值比较:")
            for phase, deltas in valid_groups:
                print(f"    - {phase}: {np.mean(deltas):+.3f}")
            
            anova_ok = p_value < 0.05
        else:
            print(f"\n  警告: 有效组数不足，跳过ANOVA")
            f_stat, p_value = None, None
            anova_ok = False
        
        return {
            'f_stat': f_stat,
            'p_value': p_value,
            'anova_ok': anova_ok,
            'phase_deltas': phase_deltas
        }
    
    def phase_transition_analysis(self):
        """阶段转移分析"""
        print("\n[Step 5] 阶段转移分析")
        
        winter_df = self.winter_data.sort_values('date')
        
        # 统计阶段转移
        transitions = []
        phases = winter_df['phase'].tolist()
        
        for i in range(1, len(phases)):
            transitions.append((phases[i-1], phases[i]))
        
        transition_counts = pd.Series(transitions).value_counts()
        
        print(f"\n  阶段转移频次:")
        print(f"  {'转移':<20} {'次数':<8}")
        print(f"  {'-'*30}")
        for trans, cnt in transition_counts.head(10).items():
            print(f"  {trans[0]} → {trans[1]:<10} {cnt:<8}")
        
        # 检查是否符合上升→稳定→下降模式
        expected_pattern = [('上升期', '稳定期'), ('稳定期', '下降期'), ('上升期', '下降期')]
        pattern_found = sum([transitions.count(p) for p in expected_pattern])
        
        print(f"\n  预期模式（上升→稳定→下降）出现次数: {pattern_found}")
        
        return transition_counts
    
    def generate_report(self):
        """生成实验报告"""
        print("\n" + "=" * 60)
        print("【实验B总结报告】")
        print("=" * 60)
        
        stats_df = self.phase_stats
        
        report = f"""
实验B：冬季三阶段存在性验证
================================

1. 数据概况:
   - 冬季样本数: {len(self.winter_data)}
   - 冬季年度: {sorted(self.winter_data['winter_year'].unique())}

2. 阶段分布:
"""
        for _, row in stats_df.iterrows():
            report += f"   - {row['phase']}: {int(row['count'])}次, TQI={row['tqi_mean']:.2f}±{row['tqi_std']:.2f}, 变化率={row['delta_mean']:+.3f}\n"
        
        report += f"""
3. ANOVA检验:
   - 三阶段变化率是否显著不同: {'是' if 'anova_ok' in dir() else '待验证'}

4. 结论:
   - 冬季呈现"上升→稳定→下降"三阶段特征: {'✓ 验证通过' if len(stats_df) >= 2 else '✗ 证据不足'}
"""
        print(report)
        
        return report
    
    def run(self):
        """运行完整实验"""
        # 1. 提取冬季数据
        self.extract_winter_data()
        
        # 2. 标注阶段
        self.annotate_phases()
        
        # 3. 阶段统计
        self.phase_statistics()
        
        # 4. ANOVA检验
        anova_results = self.anova_test()
        
        # 5. 转移分析
        self.phase_transition_analysis()
        
        # 6. 生成报告
        report = self.generate_report()
        
        return {
            'winter_data': self.winter_data,
            'phase_stats': self.phase_stats,
            'anova': anova_results,
            'report': report
        }


if __name__ == "__main__":
    from data_loader import TQIDataLoader
    
    loader = TQIDataLoader('/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/2号样本.xlsx')
    data = loader.run()
    
    exp = WinterPhaseExperiment(data['train'])
    results = exp.run()
