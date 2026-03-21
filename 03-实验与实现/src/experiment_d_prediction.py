"""
实验D：多尺度融合预测模型
=============================================
四层架构：年际趋势 + 季节调整 + 冬季阶段 + 维修响应
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class MultiscalePredictionModel:
    """多尺度融合预测模型"""
    
    def __init__(self):
        """初始化模型"""
        self.trend_model = None
        self.seasonal_factors = {}
        self.phase_adjustments = {}
        self.decay_params = {}
        self.is_fitted = False
        
    def fit(self, train_data):
        """
        训练多尺度模型
        
        Args:
            train_data: 训练集DataFrame
        """
        print("\n" + "=" * 60)
        print("【实验D】多尺度融合预测模型")
        print("=" * 60)
        
        df = train_data.copy()
        
        # ========== Layer 1: 年际趋势模型 ==========
        print("\n[Layer 1] 年际趋势模型训练")
        self.trend_model = self._fit_trend_model(df)
        
        # ========== Layer 2: 季节调整因子 ==========
        print("\n[Layer 2] 季节调整因子计算")
        self.seasonal_factors = self._compute_seasonal_factors(df)
        
        # ========== Layer 3: 冬季阶段调整 ==========
        print("\n[Layer 3] 冬季阶段调整量计算")
        self.phase_adjustments = self._compute_phase_adjustments(df)
        
        # ========== Layer 4: 维修响应衰减 ==========
        print("\n[Layer 4] 维修响应衰减参数拟合")
        self.decay_params = self._fit_decay_params(df)
        
        self.is_fitted = True
        print("\n✓ 多尺度模型训练完成")
        
        return self
    
    def _fit_trend_model(self, df):
        """拟合趋势模型"""
        # 使用时间的序数作为特征
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['tqi'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        trend_slope = model.coef_[0]
        trend_intercept = model.intercept_
        
        print(f"  - 趋势模型: TQI = {trend_intercept:.4f} + {trend_slope:.4f} × t")
        print(f"  - 趋势方向: {'上升' if trend_slope > 0 else '下降'}")
        
        return {
            'model': model,
            'slope': trend_slope,
            'intercept': trend_intercept
        }
    
    def _compute_seasonal_factors(self, df):
        """计算季节调整因子"""
        # 按月份计算相对于趋势的平均偏差
        seasonal = {}
        
        # 先去除趋势
        X = np.arange(len(df)).reshape(-1, 1)
        trend_pred = self.trend_model['model'].predict(X)
        detrended = df['tqi'].values - trend_pred
        
        df_temp = df.copy()
        df_temp['detrended'] = detrended
        
        monthly_avg = df_temp.groupby('month')['detrended'].mean()
        
        for month in range(1, 13):
            if month in monthly_avg.index:
                seasonal[month] = monthly_avg[month]
            else:
                seasonal[month] = 0.0
        
        print(f"  - 季节因子范围: [{min(seasonal.values()):.4f}, {max(seasonal.values()):.4f}]")
        
        return seasonal
    
    def _compute_phase_adjustments(self, df):
        """计算冬季阶段调整量"""
        adjustments = {
            '非冬季': 0.0,
            '上升期': 0.0,
            '稳定期': 0.0,
            '下降期': 0.0
        }
        
        # 按阶段分组计算平均偏差
        if 'winter_phase' in df.columns:
            phase_avg = df.groupby('winter_phase')['tqi'].mean()
            overall_avg = df['tqi'].mean()
            
            for phase in adjustments.keys():
                if phase in phase_avg.index:
                    adjustments[phase] = phase_avg[phase] - overall_avg
        
        print(f"  - 阶段调整量:")
        for phase, adj in adjustments.items():
            print(f"    {phase}: {adj:+.4f}")
        
        return adjustments
    
    def _fit_decay_params(self, df):
        """拟合维修衰减参数"""
        # 简单线性衰减模型
        maint_mask = df['is_maintenance'] == 1
        
        if maint_mask.sum() < 2:
            return {'type': 'none'}
        
        # 计算维修后TQI与维修前TQI的关系
        maint_data = df[maint_mask].copy()
        
        if 'maint_before_tqi' in df.columns and df['maint_before_tqi'].notna().sum() > 0:
            valid_maint = df[df['maint_before_tqi'].notna()].copy()
            
            # 维修效果 = 维修后 - 维修前
            effects = valid_maint['tqi'] - valid_maint['maint_before_tqi']
            avg_effect = effects.mean()
            
            print(f"  - 平均维修效果: {avg_effect:+.4f}")
            
            return {
                'type': 'average',
                'avg_effect': avg_effect
            }
        
        return {'type': 'none'}
    
    def predict(self, df):
        """
        预测TQI
        
        Args:
            df: 待预测数据DataFrame
            
        Returns:
            预测结果DataFrame（增加pred_tqi列）
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit()")
        
        result = df.copy()
        predictions = []
        
        for idx, row in result.iterrows():
            # Layer 1: 趋势预测
            t_idx = idx  # 简化为索引作为时间
            trend_pred = self.trend_model['intercept'] + self.trend_model['slope'] * t_idx
            
            # Layer 2: 季节调整
            month = int(row['month'])
            seasonal_adj = self.seasonal_factors.get(month, 0.0)
            
            # Layer 3: 冬季阶段调整
            phase = row.get('winter_phase', '非冬季')
            phase_adj = self.phase_adjustments.get(phase, 0.0)
            
            # Layer 4: 维修响应（如果是维修点）
            maint_adj = 0.0
            if row.get('is_maintenance', 0) == 1 and self.decay_params.get('type') == 'average':
                maint_adj = self.decay_params['avg_effect']
            
            # 融合
            pred = trend_pred + seasonal_adj + phase_adj + maint_adj
            
            predictions.append({
                'trend': trend_pred,
                'seasonal': seasonal_adj,
                'phase': phase_adj,
                'maintenance': maint_adj,
                'pred_tqi': pred
            })
        
        pred_df = pd.DataFrame(predictions)
        result = pd.concat([result.reset_index(drop=True), pred_df], axis=1)
        
        return result


class BaselineModels:
    """基线模型集合"""
    
    @staticmethod
    def naive_predict(train_data, test_data):
        """朴素预测：上一期值"""
        predictions = []
        for i in range(len(test_data)):
            if i == 0:
                # 第一个点用训练集最后一个值
                pred = train_data['tqi'].iloc[-1]
            else:
                pred = test_data['tqi'].iloc[i-1]
            predictions.append(pred)
        return np.array(predictions)
    
    @staticmethod
    def ma_predict(train_data, test_data, window=3):
        """滑动平均预测"""
        # 合并训练集和测试集以便计算滑动平均
        combined = pd.concat([train_data, test_data], ignore_index=True)
        ma = combined['tqi'].rolling(window=window, min_periods=1).mean()
        
        # 取测试集对应部分
        test_start = len(train_data)
        predictions = ma.iloc[test_start:].values
        
        return predictions
    
    @staticmethod
    def linear_trend_predict(train_data, test_data):
        """线性趋势外推"""
        # 用训练集最后3点拟合线性趋势
        n = len(train_data)
        X_train = np.arange(max(0, n-3), n).reshape(-1, 1)
        y_train = train_data['tqi'].iloc[-3:].values
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 预测测试集
        X_test = np.arange(n, n + len(test_data)).reshape(-1, 1)
        predictions = model.predict(X_test)
        
        return predictions


class ExperimentD:
    """实验D主类"""
    
    def __init__(self, data):
        """
        初始化实验
        
        Args:
            data: 完整数据字典
        """
        self.train_data = data['train']
        self.val_data = data['val']
        self.test_data = data['test']
        self.results = {}
        
    def run_all_models(self):
        """运行所有模型对比"""
        print("\n" + "=" * 60)
        print("【实验D】多尺度融合预测模型对比")
        print("=" * 60)
        
        # 合并验证集和测试集用于评估
        eval_data = pd.concat([self.val_data, self.test_data], ignore_index=True)
        
        models_results = {}
        
        # ========== Baseline 1: 朴素预测 ==========
        print("\n[Baseline-1] 朴素预测")
        naive_pred = BaselineModels.naive_predict(self.train_data, eval_data)
        naive_metrics = self._evaluate(eval_data['tqi'].values, naive_pred)
        models_results['naive'] = {'pred': naive_pred, 'metrics': naive_metrics}
        self._print_metrics(naive_metrics)
        
        # ========== Baseline 2: 滑动平均 ==========
        print("\n[Baseline-2] 滑动平均(MA3)")
        ma_pred = BaselineModels.ma_predict(self.train_data, eval_data)
        ma_metrics = self._evaluate(eval_data['tqi'].values, ma_pred)
        models_results['ma3'] = {'pred': ma_pred, 'metrics': ma_metrics}
        self._print_metrics(ma_metrics)
        
        # ========== Baseline 3: 线性趋势 ==========
        print("\n[Baseline-3] 线性趋势外推")
        linear_pred = BaselineModels.linear_trend_predict(self.train_data, eval_data)
        linear_metrics = self._evaluate(eval_data['tqi'].values, linear_pred)
        models_results['linear'] = {'pred': linear_pred, 'metrics': linear_metrics}
        self._print_metrics(linear_metrics)
        
        # ========== Proposed: 多尺度融合 ==========
        print("\n[Proposed] 多尺度融合模型")
        multiscale = MultiscalePredictionModel()
        multiscale.fit(self.train_data)
        multiscale_result = multiscale.predict(eval_data)
        ms_pred = multiscale_result['pred_tqi'].values
        ms_metrics = self._evaluate(eval_data['tqi'].values, ms_pred)
        models_results['multiscale'] = {'pred': ms_pred, 'metrics': ms_metrics, 'detail': multiscale_result}
        self._print_metrics(ms_metrics)
        
        self.results = models_results
        
        return models_results
    
    def _evaluate(self, y_true, y_pred):
        """评估指标计算"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def _print_metrics(self, metrics):
        """打印评估指标"""
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    def ablation_study(self):
        """消融实验"""
        print("\n" + "=" * 60)
        print("【消融实验】")
        print("=" * 60)
        
        eval_data = pd.concat([self.val_data, self.test_data], ignore_index=True)
        
        # 这里简化处理，实际应该重新训练去掉某层的模型
        print("\n消融实验待实现...")
        print("  - 去掉季节因子")
        print("  - 去掉冬季阶段调整")
        print("  - 去掉维修响应")
        
    def generate_report(self):
        """生成实验报告"""
        print("\n" + "=" * 60)
        print("【实验D总结报告】")
        print("=" * 60)
        
        if not self.results:
            print("请先运行run_all_models()")
            return
        
        # 汇总表格
        print("\n模型性能对比:")
        print(f"{'模型':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 45)
        
        for name, result in self.results.items():
            m = result['metrics']
            print(f"{name:<15} {m['MAE']:<10.4f} {m['RMSE']:<10.4f} {m['MAPE']:<10.2f}%")
        
        # 计算提升
        baseline_mae = self.results['naive']['metrics']['MAE']
        proposed_mae = self.results['multiscale']['metrics']['MAE']
        improvement = (baseline_mae - proposed_mae) / baseline_mae * 100
        
        print(f"\n多尺度模型相比朴素预测:")
        print(f"  - MAE降低: {improvement:.1f}%")
        print(f"  - {'通过' if improvement > 15 else '未通过'} (目标>15%)")
        
    def run(self):
        """运行完整实验"""
        # 1. 运行所有模型对比
        self.run_all_models()
        
        # 2. 消融实验
        self.ablation_study()
        
        # 3. 生成报告
        self.generate_report()
        
        return self.results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/src')
    
    from data_loader import TQIDataLoader
    
    loader = TQIDataLoader('/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/19d0f20d-3f32-8628-8000-0000b1563466_2号样本.xlsx')
    data = loader.run()
    
    exp = ExperimentD(data)
    results = exp.run()
