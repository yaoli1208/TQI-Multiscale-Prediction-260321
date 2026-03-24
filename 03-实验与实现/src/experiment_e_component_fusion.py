"""
实验E：分量分组融合预测模型 (Component Group Fusion)
=====================================================
7个分量分两组分别建模，最后融合预测TQI

分组策略:
- 组1(平面位置): 左轨向(tqi_laln) + 右轨向(tqi_raln) + 轨距(tqi_gage)
- 组2(高程平顺): 左高低(tqi_lprf) + 右高低(tqi_rprf) + 水平(tqi_xlvl) + 三角坑(tqi_warp1)

架构: 双塔Trident → 分量级TQI预测 → 加权融合 → 最终TQI
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')


class ComponentGroupTrident:
    """单分组的Trident预测模型"""
    
    def __init__(self, group_name, component_cols, target_col='tqi'):
        """
        初始化分组模型
        
        Args:
            group_name: 分组名称 ('plane'或'elevation')
            component_cols: 该分组包含的分量列名列表
            target_col: 目标列名
        """
        self.group_name = group_name
        self.component_cols = component_cols
        self.target_col = target_col
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.fitted = False
        
    def extract_multiscale_features(self, df):
        """提取多尺度特征"""
        features = pd.DataFrame(index=df.index)
        
        # 对每个分量提取多尺度特征
        for col in self.component_cols:
            # 原始值
            features[f'{col}_raw'] = df[col]
            
            # 7天MA
            features[f'{col}_ma7'] = df[col].rolling(window=3, min_periods=1).mean()
            
            # 14天MA
            features[f'{col}_ma14'] = df[col].rolling(window=5, min_periods=1).mean()
            
            # 21天MA
            features[f'{col}_ma21'] = df[col].rolling(window=7, min_periods=1).mean()
            
            # 一阶差分
            features[f'{col}_diff1'] = df[col].diff(1).fillna(0)
            
            # 趋势（Savitzky-Golay滤波）
            if len(df) >= 5:
                try:
                    trend = savgol_filter(df[col].fillna(df[col].mean()), 5, 2)
                    features[f'{col}_trend'] = trend
                except:
                    features[f'{col}_trend'] = df[col]
            else:
                features[f'{col}_trend'] = df[col]
        
        # 时间特征
        if 'month' in df.columns:
            features['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 维修后特征（如果有）
        if 'years_since_2021' in df.columns:
            features['years_since_2021'] = df['years_since_2021']
        
        return features.fillna(0)
    
    def fit(self, train_df, val_df=None):
        """训练模型"""
        print(f"\n  [{self.group_name}] 训练Trident模型...")
        print(f"    分量: {self.component_cols}")
        
        # 提取特征
        X_train = self.extract_multiscale_features(train_df)
        y_train = train_df[self.target_col].values
        
        # 标准化
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # 训练MLP
        self.model = MLPRegressor(
            hidden_layer_sizes=(48, 24),
            activation='relu',
            solver='adam',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # 训练集评估
        train_pred_scaled = self.model.predict(X_train_scaled)
        train_pred = self.scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).ravel()
        train_mae = mean_absolute_error(y_train, train_pred)
        
        print(f"    训练MAE: {train_mae:.4f}")
        
        self.fitted = True
        return self
    
    def predict(self, df):
        """预测"""
        if not self.fitted:
            raise ValueError("模型未训练")
        
        X = self.extract_multiscale_features(df)
        X_scaled = self.scaler_X.transform(X)
        
        pred_scaled = self.model.predict(X_scaled)
        pred = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        
        return pred


class ComponentFusionModel:
    """分量分组融合模型"""
    
    def __init__(self):
        """初始化融合模型"""
        # 组1: 平面位置 (轨向+轨距)
        self.plane_group = ComponentGroupTrident(
            group_name='平面位置组',
            component_cols=['tqi_laln', 'tqi_raln', 'tqi_gage']
        )
        
        # 组2: 高程平顺 (高低+水平+三角坑)
        self.elevation_group = ComponentGroupTrident(
            group_name='高程平顺组',
            component_cols=['tqi_lprf', 'tqi_rprf', 'tqi_xlvl', 'tqi_warp1']
        )
        
        # 融合权重 (可学习)
        self.plane_weight = 0.5
        self.elevation_weight = 0.5
        self.fitted = False
    
    def preprocess_data(self, df):
        """数据预处理"""
        df = df.copy()
        
        # 列名映射（中文 -> 英文）
        col_mapping = {}
        
        # 检测日期列
        for col in df.columns:
            if '日期' in col or 'date' in col.lower():
                if col != 'date':
                    col_mapping[col] = 'date'
                break
        
        # TQI列映射
        for col in df.columns:
            if col == 'TQI值' or col == 'tqi_val':
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
        
        # 确保日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            # 尝试找日期列
            for col in df.columns:
                if 'date' in col.lower() or 'dt' in col.lower():
                    df['date'] = pd.to_datetime(df[col])
                    break
        
        # 排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 时间特征
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # 维修后特征
        if 'year' in df.columns:
            df['years_since_2021'] = (df['year'] - 2021).clip(lower=0)
        
        # 重命名TQI列
        if 'tqi_val' in df.columns and 'tqi' not in df.columns:
            df['tqi'] = df['tqi_val']
        
        return df
    
    def fit(self, train_df, val_df=None):
        """训练双塔模型"""
        print("\n" + "=" * 60)
        print("【实验E】分量分组融合模型 - 训练阶段")
        print("=" * 60)
        
        # 预处理
        train_df = self.preprocess_data(train_df)
        if val_df is not None:
            val_df = self.preprocess_data(val_df)
        
        # 检查必要的列
        required_cols = ['tqi', 'tqi_laln', 'tqi_raln', 'tqi_gage', 
                        'tqi_lprf', 'tqi_rprf', 'tqi_xlvl', 'tqi_warp1']
        missing = [c for c in required_cols if c not in train_df.columns]
        if missing:
            available = [c for c in train_df.columns if 'tqi' in c]
            print(f"警告: 缺少列 {missing}")
            print(f"可用TQI相关列: {available}")
            # 尝试自动映射
            col_map = {}
            for req in missing:
                for avail in available:
                    if req.replace('tqi_', '') in avail.lower() or avail.lower() in req:
                        col_map[req] = avail
                        break
            if col_map:
                print(f"自动映射: {col_map}")
                train_df = train_df.rename(columns=col_map)
        
        # 训练组1 (平面位置)
        print("\n[Group 1] 平面位置组 (轨向+轨距)")
        self.plane_group.fit(train_df, val_df)
        
        # 训练组2 (高程平顺)
        print("\n[Group 2] 高程平顺组 (高低+水平+三角坑)")
        self.elevation_group.fit(train_df, val_df)
        
        # 优化融合权重
        self._optimize_fusion_weights(train_df, val_df)
        
        self.fitted = True
        print("\n✓ 双塔模型训练完成")
        
        return self
    
    def _optimize_fusion_weights(self, train_df, val_df=None):
        """优化融合权重"""
        print("\n[融合权重优化]")
        
        # 在训练集上获取两组预测
        plane_pred = self.plane_group.predict(train_df)
        elevation_pred = self.elevation_group.predict(train_df)
        y_true = train_df['tqi'].values
        
        # 计算各组与目标的相关性
        plane_corr = np.corrcoef(plane_pred, y_true)[0, 1]
        elevation_corr = np.corrcoef(elevation_pred, y_true)[0, 1]
        
        print(f"  平面组与TQI相关性: {plane_corr:.4f}")
        print(f"  高程组与TQI相关性: {elevation_corr:.4f}")
        
        # 基于相关性初始化权重
        total_corr = abs(plane_corr) + abs(elevation_corr)
        if total_corr > 0:
            self.plane_weight = abs(plane_corr) / total_corr
            self.elevation_weight = abs(elevation_corr) / total_corr
        
        print(f"  初始化权重: 平面={self.plane_weight:.3f}, 高程={self.elevation_weight:.3f}")
        
        # 简单网格搜索优化
        best_mae = float('inf')
        best_weights = (self.plane_weight, self.elevation_weight)
        
        for w1 in np.arange(0.1, 0.9, 0.05):
            w2 = 1 - w1
            fused = w1 * plane_pred + w2 * elevation_pred
            mae = mean_absolute_error(y_true, fused)
            if mae < best_mae:
                best_mae = mae
                best_weights = (w1, w2)
        
        self.plane_weight, self.elevation_weight = best_weights
        print(f"  优化后权重: 平面={self.plane_weight:.3f}, 高程={self.elevation_weight:.3f}")
        print(f"  训练集融合MAE: {best_mae:.4f}")
    
    def predict(self, df, return_components=False):
        """
        预测TQI
        
        Args:
            df: 输入数据
            return_components: 是否返回各组预测结果
        """
        if not self.fitted:
            raise ValueError("模型未训练")
        
        df = self.preprocess_data(df)
        
        # 两组分别预测
        plane_pred = self.plane_group.predict(df)
        elevation_pred = self.elevation_group.predict(df)
        
        # 加权融合
        fused_pred = self.plane_weight * plane_pred + self.elevation_weight * elevation_pred
        
        if return_components:
            return {
                'fused': fused_pred,
                'plane': plane_pred,
                'elevation': elevation_pred
            }
        return fused_pred


class ExperimentE:
    """实验E主类"""
    
    def __init__(self, train_data, val_data, test_data):
        """
        初始化实验
        
        Args:
            train_data: 训练集DataFrame
            val_data: 验证集DataFrame
            test_data: 测试集DataFrame
        """
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.model = None
        self.results = {}
    
    def run(self):
        """运行完整实验"""
        print("\n" + "=" * 70)
        print("  实验E: 分量分组融合预测模型")
        print("  Component Group Fusion for TQI Prediction")
        print("=" * 70)
        
        # 1. 训练模型
        self.model = ComponentFusionModel()
        self.model.fit(self.train_data, self.val_data)
        
        # 2. 验证集评估
        print("\n" + "=" * 60)
        print("【验证集评估】")
        print("=" * 60)
        val_results = self._evaluate(self.val_data, "验证集")
        
        # 3. 测试集评估
        print("\n" + "=" * 60)
        print("【测试集评估】")
        print("=" * 60)
        test_results = self._evaluate(self.test_data, "测试集")
        
        # 4. 对比分析
        print("\n" + "=" * 60)
        print("【分组贡献分析】")
        print("=" * 60)
        self._analyze_contribution()
        
        # 5. 生成报告
        self._generate_report(val_results, test_results)
        
        return {
            'val': val_results,
            'test': test_results,
            'model': self.model
        }
    
    def _evaluate(self, data, split_name):
        """评估模型"""
        # 融合预测
        fused_pred = self.model.predict(data)
        y_true = self.model.preprocess_data(data)['tqi'].values
        
        # 各组单独预测
        components = self.model.predict(data, return_components=True)
        plane_pred = components['plane']
        elevation_pred = components['elevation']
        
        # 计算指标
        def calc_metrics(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
        
        fused_metrics = calc_metrics(y_true, fused_pred)
        plane_metrics = calc_metrics(y_true, plane_pred)
        elevation_metrics = calc_metrics(y_true, elevation_pred)
        
        print(f"\n{split_name}性能:")
        print(f"{'模型':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
        print("-" * 45)
        print(f"{'平面位置组':<15} {plane_metrics['MAE']:<10.4f} {plane_metrics['RMSE']:<10.4f} {plane_metrics['MAPE']:<10.2f}%")
        print(f"{'高程平顺组':<15} {elevation_metrics['MAE']:<10.4f} {elevation_metrics['RMSE']:<10.4f} {elevation_metrics['MAPE']:<10.2f}%")
        print(f"{'融合模型':<15} {fused_metrics['MAE']:<10.4f} {fused_metrics['RMSE']:<10.4f} {fused_metrics['MAPE']:<10.2f}%")
        
        # 计算提升
        best_single = min(plane_metrics['MAE'], elevation_metrics['MAE'])
        improvement = (best_single - fused_metrics['MAE']) / best_single * 100
        print(f"\n融合相比最佳单组: MAE降低 {improvement:.1f}%")
        
        return {
            'fused': fused_metrics,
            'plane': plane_metrics,
            'elevation': elevation_metrics,
            'predictions': {
                'fused': fused_pred,
                'plane': plane_pred,
                'elevation': elevation_pred,
                'true': y_true
            }
        }
    
    def _analyze_contribution(self):
        """分析各组贡献"""
        # 合并所有数据
        all_data = pd.concat([self.train_data, self.val_data, self.test_data], ignore_index=True)
        all_data = self.model.preprocess_data(all_data)
        
        # 计算各分量与TQI的相关系数
        components_group1 = ['tqi_laln', 'tqi_raln', 'tqi_gage']
        components_group2 = ['tqi_lprf', 'tqi_rprf', 'tqi_xlvl', 'tqi_warp1']
        
        print("\n各分量与TQI相关性:")
        print("\n[平面位置组]")
        for col in components_group1:
            if col in all_data.columns:
                corr = np.corrcoef(all_data[col], all_data['tqi'])[0, 1]
                print(f"  {col}: {corr:.4f}")
        
        print("\n[高程平顺组]")
        for col in components_group2:
            if col in all_data.columns:
                corr = np.corrcoef(all_data[col], all_data['tqi'])[0, 1]
                print(f"  {col}: {corr:.4f}")
        
        print(f"\n融合权重: 平面={self.model.plane_weight:.3f}, 高程={self.model.elevation_weight:.3f}")
    
    def _generate_report(self, val_results, test_results):
        """生成实验报告"""
        print("\n" + "=" * 60)
        print("【实验E总结报告】")
        print("=" * 60)
        
        print("\n实验设计:")
        print("  - 分组策略: 7个TQI分量分为2组")
        print("  - 组1(平面位置): 左轨向 + 右轨向 + 轨距")
        print("  - 组2(高程平顺): 左高低 + 右高低 + 水平 + 三角坑")
        print("  - 每组独立Trident模型")
        print("  - 自适应加权融合")
        
        print("\n关键发现:")
        val_fused = val_results['fused']['MAE']
        test_fused = test_results['fused']['MAE']
        print(f"  - 验证集MAE: {val_fused:.4f}")
        print(f"  - 测试集MAE: {test_fused:.4f}")
        
        # 判断哪个组更重要
        plane_val = val_results['plane']['MAE']
        elevation_val = val_results['elevation']['MAE']
        if plane_val < elevation_val:
            print(f"  - 平面位置组预测更准确 (MAE: {plane_val:.4f} < {elevation_val:.4f})")
        else:
            print(f"  - 高程平顺组预测更准确 (MAE: {elevation_val:.4f} < {plane_val:.4f})")
        
        print("\n创新点:")
        print("  ✓ 物理意义明确的分组策略")
        print("  ✓ 双塔架构，可解释性强")
        print("  ✓ 自适应融合权重优化")


if __name__ == "__main__":
    # 加载5号样本数据
    import sys
    sys.path.insert(0, '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/src')
    
    # 读取5号样本数据
    file_path = '/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/5号样本.xlsx'
    df = pd.read_excel(file_path)
    
    print(f"\n5号样本数据: {len(df)}条记录")
    print(f"时间跨度: {df.iloc[:, 0].min()} 至 {df.iloc[:, 0].max()}")
    
    # 按时间排序后划分数据集 (时序划分)
    # 训练集: 2013-2021 (约70%)
    # 验证集: 2022-2023 (约15%) 
    # 测试集: 2024-2026 (约15%)
    df['检测日期'] = pd.to_datetime(df['检测日期'])
    df = df.sort_values('检测日期').reset_index(drop=True)
    
    train_df = df[df['检测日期'] < '2022-01-01'].copy()
    val_df = df[(df['检测日期'] >= '2022-01-01') & (df['检测日期'] < '2024-01-01')].copy()
    test_df = df[df['检测日期'] >= '2024-01-01'].copy()
    
    print(f"\n时序划分:")
    print(f"  训练集: {len(train_df)}条 ({train_df['检测日期'].min().strftime('%Y-%m-%d')} ~ {train_df['检测日期'].max().strftime('%Y-%m-%d')})")
    print(f"  验证集: {len(val_df)}条 ({val_df['检测日期'].min().strftime('%Y-%m-%d')} ~ {val_df['检测日期'].max().strftime('%Y-%m-%d')})")
    print(f"  测试集: {len(test_df)}条 ({test_df['检测日期'].min().strftime('%Y-%m-%d')} ~ {test_df['检测日期'].max().strftime('%Y-%m-%d')})")
    
    # 运行实验E
    exp = ExperimentE(train_df, val_df, test_df)
    results = exp.run()
