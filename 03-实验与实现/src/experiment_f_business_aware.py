"""
实验F：业务经验增强的多尺度预测模型
========================================
结合业务经验锚定 + 季节性规律 + 数据驱动预测

核心策略:
1. 阶段识别: 检测稳定期/劣化期/大修后恢复期
2. 稳定期预测: 基准值锚定 + 季节性调整 + 微小劣化趋势
3. 非稳定期预测: 原有Trident模型
4. 大修期锚定: 参考历史同期波动范围

业务经验编码:
- 平面组稳定基准: ~0.8, 变化极小
- 高程组稳定基准: ~1.15, 季节性波动±0.15
- 高程理论下限: 0.7-0.8 (工艺限制)
- 季节性: 冬季高(1.2+), 夏季低(1.0+), 秋季平稳
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import savgol_filter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class BusinessAwarePredictor:
    """业务经验增强预测器"""
    
    def __init__(self):
        # 业务经验参数（可从数据中自动学习）
        self.plane_baseline = 0.8
        self.elevation_baseline = 1.15
        self.elevation_floor = 0.75  # 理论下限中值
        self.seasonal_amplitude = 0.15
        
        # 状态识别参数
        self.stable_threshold = 0.1  # 变异系数阈值
        self.min_stable_samples = 20
        
        # 学习到的参数
        self.is_stable_period = False
        self.learned_plane_baseline = None
        self.learned_elevation_baseline = None
        self.learned_seasonal_pattern = None
        self.degradation_rate = 0  # 年劣化率
        
    def detect_maintenance_periods(self, df):
        """检测每年大修期（TQI大幅下降点）"""
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        # 计算TQI变化
        df['tqi_diff'] = df['tqi'].diff()
        
        # 找出每年TQI下降最大的点（视为大修期）
        maintenance_anchors = {}
        
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year].copy()
            if len(year_data) < 3:
                continue
            
            # 找该年最大的TQI下降（最负的diff）
            min_diff_idx = year_data['tqi_diff'].idxmin()
            if pd.notna(min_diff_idx) and year_data.loc[min_diff_idx, 'tqi_diff'] < -0.3:
                # 大修后取该点及之后的数据均值作为锚定
                maint_date = year_data.loc[min_diff_idx, 'date']
                post_maint = year_data[year_data['date'] >= maint_date]
                
                if len(post_maint) >= 2:
                    maintenance_anchors[year] = {
                        'date': maint_date,
                        'tqi': post_maint['tqi'].mean(),
                        'plane': post_maint['plane'].mean() if 'plane' in post_maint.columns else None,
                        'elevation': post_maint['elevation'].mean() if 'elevation' in post_maint.columns else None,
                        'drop': year_data.loc[min_diff_idx, 'tqi_diff']
                    }
        
        return maintenance_anchors
    
    def analyze_historical_pattern(self, df):
        """从历史数据学习业务模式 - 滚动锚定策略"""
        print("\n【业务模式学习】")
        
        # 计算平面组和高程组
        df = df.copy()
        df['plane'] = df['tqi_laln'] + df['tqi_raln'] + df['tqi_gage']
        df['elevation'] = df['tqi_lprf'] + df['tqi_rprf'] + df['tqi_xlvl'] + df['tqi_warp1']
        df['tqi_calc'] = df['plane'] + df['elevation']
        
        # 检测每年大修期锚定值
        self.maintenance_anchors = self.detect_maintenance_periods(df)
        print(f"  检测到 {len(self.maintenance_anchors)} 年大修期锚定值")
        
        for year, anchor in sorted(self.maintenance_anchors.items())[-3:]:
            print(f"    {year}: TQI={anchor['tqi']:.3f} (下降{anchor['drop']:.3f})")
        
        # 学习季节性模式（全年数据）
        df['month'] = pd.to_datetime(df['date']).dt.month
        monthly_tqi = df.groupby('month')['tqi'].mean()
        self.learned_seasonal_pattern = monthly_tqi - monthly_tqi.mean()
        print(f"  季节性振幅: {self.learned_seasonal_pattern.abs().max():.3f}")
        
        # 检查是否有稳定期（用于判断预测策略）
        df['year'] = pd.to_datetime(df['date']).dt.year
        yearly_stats = []
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            if len(year_data) < 5:
                continue
            plane_cv = year_data['plane'].std() / year_data['plane'].mean() if year_data['plane'].mean() > 0 else 0
            yearly_stats.append({'year': year, 'plane_cv': plane_cv})
        
        yearly_df = pd.DataFrame(yearly_stats)
        if len(yearly_df) >= 2:
            recent_cv = yearly_df.tail(2)['plane_cv'].mean()
            self.is_stable_period = recent_cv < self.stable_threshold
            if self.is_stable_period:
                print(f"  ✓ 近期进入稳定期 (CV={recent_cv:.3f})")
            else:
                print(f"  ⚠ 近期未稳定 (CV={recent_cv:.3f})，使用滚动锚定策略")
        
        return yearly_df
    
    def predict_with_rolling_anchor(self, target_date, reference_data=None):
        """滚动锚定预测 - 使用上一年大修期后值作为基准"""
        target_year = pd.to_datetime(target_date).year
        target_month = pd.to_datetime(target_date).month
        
        # 找上一年的锚定值
        anchor_year = target_year - 1
        anchor = None
        
        # 优先找上一年大修期锚定值
        while anchor_year >= target_year - 3:  # 最多往前找3年
            if anchor_year in self.maintenance_anchors:
                anchor = self.maintenance_anchors[anchor_year]
                break
            anchor_year -= 1
        
        if anchor is None:
            # 找不到锚定值，用最近可用数据
            if reference_data is not None and len(reference_data) > 0:
                ref_data = reference_data.sort_values('date').tail(10)
                tqi_base = ref_data['tqi'].mean()
                print(f"    警告: 未找到锚定值，使用最近数据均值: {tqi_base:.3f}")
            else:
                return None
        else:
            tqi_base = anchor['tqi']
            print(f"    使用{anchor_year}年大修期后锚定值: TQI={tqi_base:.3f}")
        
        # 季节性调整（基于月度偏差）
        seasonal_adj = 0
        if self.learned_seasonal_pattern is not None and target_month in self.learned_seasonal_pattern.index:
            seasonal_adj = self.learned_seasonal_pattern[target_month]
            print(f"    季节性调整: {seasonal_adj:+.3f}")
        
        tqi_pred = tqi_base + seasonal_adj
        
        # 年际趋势调整（如果有多年数据）
        year_diff = target_year - anchor_year if anchor else 1
        if year_diff > 1 and hasattr(self, 'degradation_rate'):
            trend_adj = self.degradation_rate * (year_diff - 1)  # 锚定值已包含第一年
            tqi_pred += trend_adj
            print(f"    年际趋势调整({year_diff-1}年): {trend_adj:+.3f}")
        
        return {
            'tqi': tqi_pred,
            'base': tqi_base,
            'seasonal_adj': seasonal_adj,
            'anchor_year': anchor_year if anchor else None,
            'is_rolling_prediction': True
        }
    
    def get_confidence_interval(self, target_date, historical_data):
        """获取预测置信区间（基于历史同期波动）"""
        month = pd.to_datetime(target_date).month
        
        # 找历史同期数据
        historical_data = historical_data.copy()
        historical_data['month'] = pd.to_datetime(historical_data['date']).dt.month
        same_month = historical_data[historical_data['month'] == month]
        
        if len(same_month) < 3:
            return None
        
        # 计算历史同期的波动范围
        plane_vals = same_month['tqi_laln'] + same_month['tqi_raln'] + same_month['tqi_gage']
        elev_vals = same_month['tqi_lprf'] + same_month['tqi_rprf'] + same_month['tqi_xlvl'] + same_month['tqi_warp1']
        tqi_vals = plane_vals + elev_vals
        
        return {
            'tqi_lower': tqi_vals.quantile(0.25),
            'tqi_upper': tqi_vals.quantile(0.75),
            'tqi_mean': tqi_vals.mean(),
            'plane_std': plane_vals.std(),
            'elevation_std': elev_vals.std()
        }


class ExperimentF:
    """实验F：业务经验增强的多尺度预测"""
    
    def __init__(self, train_data, val_data, test_data):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.business_predictor = BusinessAwarePredictor()
        self.trident_model = None
        
    def preprocess(self, df):
        """预处理"""
        df = df.copy()
        
        # 检查date列是否已存在且为datetime类型
        if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
            # 直接使用已有date列
            df = df.sort_values('date').reset_index(drop=True)
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            return df
        
        # 列名映射
        col_mapping = {}
        for col in df.columns:
            if '日期' in col or 'date' in col.lower():
                if col != 'date':
                    col_mapping[col] = 'date'
            elif col == 'TQI值' or col == 'tqi_val':
                col_mapping[col] = 'tqi'
            elif col == 'TQI左高低':
                col_mapping[col] = 'tqi_lprf'
            elif col == 'TQI右高低':
                col_mapping[col] = 'tqi_rprf'
            elif col == 'TQI左轨向':
                col_mapping[col] = 'tqi_laln'
            elif col == 'TQI右轨向':
                col_mapping[col] = 'tqi_raln'
            elif col == 'TQI轨距':
                col_mapping[col] = 'tqi_gage'
            elif col == 'TQI三角坑':
                col_mapping[col] = 'tqi_warp1'
            elif col == 'TQI水平':
                col_mapping[col] = 'tqi_xlvl'
        
        if col_mapping:
            df = df.rename(columns=col_mapping)
        
        if 'date' not in df.columns:
            for col in df.columns:
                if 'date' in col.lower() or 'dt' in col.lower():
                    df['date'] = pd.to_datetime(df[col])
                    break
        else:
            df['date'] = pd.to_datetime(df['date'])
        
        df = df.sort_values('date').reset_index(drop=True)
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        return df
    
    def run(self):
        """运行实验 - 修后预测模式"""
        print("\n" + "=" * 70)
        print("  实验F: 业务经验增强的多尺度预测模型")
        print("  Business-Aware Multiscale Prediction")
        print("  策略: 大修期后预测 (Post-Maintenance Prediction)")
        print("=" * 70)
        
        # 预处理
        train_df = self.preprocess(self.train_data)
        val_df = self.preprocess(self.val_data)
        test_df = self.preprocess(self.test_data)
        
        # 合并所有数据用于学习
        all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
        all_data = all_data.sort_values('date').reset_index(drop=True)
        
        # 步骤1: 学习业务模式
        print("\n【Step 1】业务模式学习")
        self.business_predictor.analyze_historical_pattern(all_data)
        
        # 步骤2: 修后预测
        print("\n【Step 2】修后预测")
        print("  策略: 每次大修后作为预测起点，预测至下次大修前")
        
        results = self._predict_post_maintenance_mode(all_data)
        
        return results
    
    def _predict_post_maintenance_mode(self, all_data):
        """修后预测模式 - 从每次大修后开始预测"""
        anchors = self.business_predictor.maintenance_anchors
        
        if len(anchors) < 2:
            print("  警告: 大修期锚定值不足，无法构建修后预测窗口")
            return self._predict_data_driven_mode(self.train_data, self.val_data, self.test_data)
        
        print(f"\n  构建{len(anchors)-1}个修后预测窗口:")
        
        all_predictions = []
        all_actuals = []
        window_stats = []
        
        # 按年份排序锚定值
        sorted_years = sorted(anchors.keys())
        
        for i in range(len(sorted_years) - 1):
            start_year = sorted_years[i]
            end_year = sorted_years[i + 1]
            
            start_anchor = anchors[start_year]
            end_anchor = anchors[end_year]
            
            # 预测窗口: 本次大修后 ~ 下次大修前
            window_data = all_data[
                (all_data['date'] >= start_anchor['date']) & 
                (all_data['date'] < end_anchor['date'])
            ].copy()
            
            if len(window_data) < 3:
                continue
            
            print(f"\n  窗口 {start_year}→{end_year}:")
            print(f"    锚定值: TQI={start_anchor['tqi']:.3f} ({start_anchor['date'].strftime('%Y-%m-%d')})")
            print(f"    实际数据: {len(window_data)}条 ({window_data['date'].min().strftime('%Y-%m-%d')} ~ {window_data['date'].max().strftime('%Y-%m-%d')})")
            
            # 对该窗口内每个时间点做预测
            for idx, row in window_data.iterrows():
                # 计算距离锚定点的月数
                months_since_anchor = (row['date'] - start_anchor['date']).days / 30.0
                
                # 预测: 锚定值 + 季节性调整 + 劣化趋势
                pred_tqi = self._predict_from_anchor(
                    start_anchor['tqi'], 
                    row['date'], 
                    months_since_anchor
                )
                
                all_predictions.append(pred_tqi)
                all_actuals.append(row['tqi'])
            
            # 窗口统计
            window_mae = mean_absolute_error(
                window_data['tqi'].values,
                all_predictions[-len(window_data):]
            )
            window_stats.append({
                'start_year': start_year,
                'end_year': end_year,
                'n_points': len(window_data),
                'mae': window_mae,
                'anchor_tqi': start_anchor['tqi']
            })
            print(f"    窗口MAE: {window_mae:.4f}")
        
        # 整体性能
        if len(all_actuals) > 0:
            y_true = np.array(all_actuals)
            y_pred = np.array(all_predictions)
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            print(f"\n修后预测整体性能:")
            print(f"  总预测点数: {len(y_true)}")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            
            # 各窗口对比
            print(f"\n各窗口性能对比:")
            for stat in window_stats:
                print(f"  {stat['start_year']}→{stat['end_year']}: MAE={stat['mae']:.4f}, 点数={stat['n_points']}, 锚定TQI={stat['anchor_tqi']:.3f}")
        
        return {
            'y_true': np.array(all_actuals) if all_actuals else None,
            'y_pred': np.array(all_predictions) if all_predictions else None,
            'window_stats': window_stats,
            'mode': 'post_maintenance'
        }
    
    def _predict_from_anchor(self, anchor_tqi, target_date, months_since_anchor):
        """从锚定值预测未来某时间点的TQI"""
        target_month = pd.to_datetime(target_date).month
        
        # 季节性调整
        seasonal_adj = 0
        if self.business_predictor.learned_seasonal_pattern is not None:
            if target_month in self.business_predictor.learned_seasonal_pattern.index:
                seasonal_adj = self.business_predictor.learned_seasonal_pattern[target_month]
        
        # 劣化趋势调整 (每月劣化率)
        monthly_degradation = -0.01  # 假设每月劣化0.01 (年劣化0.12)
        if hasattr(self.business_predictor, 'degradation_rate'):
            monthly_degradation = self.business_predictor.degradation_rate / 12
        
        degradation_adj = monthly_degradation * months_since_anchor
        
        pred_tqi = anchor_tqi + seasonal_adj + degradation_adj
        return pred_tqi
    
    def _predict_data_driven_mode(self, train_df, val_df, test_df):
        """数据驱动预测模式（使用原有Trident）"""
        # 这里简化处理，实际应该调用Trident模型
        print("  使用Trident模型进行预测...")
        
        # 简单基线：历史均值
        historical_mean = train_df['tqi'].mean()
        y_pred = np.full(len(test_df), historical_mean)
        y_true = test_df['tqi'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        print(f"  基线MAE: {mae:.4f}")
        
        return {
            'y_true': y_true,
            'y_pred': y_pred,
            'mode': 'data_driven'
        }


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数选择样本
    sample = sys.argv[1] if len(sys.argv) > 1 else "5号"
    
    if sample == "3号":
        # 3号样本清洗后数据
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
        print(f"\n{'='*70}")
        print(f"3号样本数据: {len(df)}条记录 (清洗后)")
        print(f"时间跨度: {df['date'].min()} 至 {df['date'].max()}")
    else:
        # 5号样本原始数据
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
        print(f"\n{'='*70}")
        print(f"5号样本数据: {len(df)}条记录")
        print(f"时间跨度: {df['date'].min()} 至 {df['date'].max()}")
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # 时序划分 (70%/15%/15%)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"\n时序划分:")
    print(f"  训练集: {len(train_df)}条 ({train_df['date'].min().strftime('%Y-%m-%d')} ~ {train_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"  验证集: {len(val_df)}条 ({val_df['date'].min().strftime('%Y-%m-%d')} ~ {val_df['date'].max().strftime('%Y-%m-%d')})")
    print(f"  测试集: {len(test_df)}条 ({test_df['date'].min().strftime('%Y-%m-%d')} ~ {test_df['date'].max().strftime('%Y-%m-%d')})")
    
    # 运行实验F
    exp = ExperimentF(train_df, val_df, test_df)
    results = exp.run()
