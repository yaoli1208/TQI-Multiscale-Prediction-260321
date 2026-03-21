"""
多尺度TQI预测实验 - 数据加载与预处理模块
=============================================
基于2号样本（2024-01-06 至 2025-12-14）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TQIDataLoader:
    """TQI数据加载与预处理类"""
    
    def __init__(self, file_path):
        """
        初始化数据加载器
        
        Args:
            file_path: Excel文件路径
        """
        self.file_path = file_path
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_data(self):
        """加载原始数据"""
        print("=" * 60)
        print("【Step 1】加载原始数据")
        print("=" * 60)
        
        df = pd.read_excel(self.file_path)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        
        # 重命名列
        df.rename(columns={
            '日期': 'date',
            'TQI': 'tqi',
            '维修记录': 'maintenance'
        }, inplace=True)
        
        self.raw_data = df.copy()
        
        print(f"✓ 加载完成: {len(df)} 条记录")
        print(f"✓ 时间范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"✓ 维修记录: {df['maintenance'].notna().sum()} 次")
        
        return df
    
    def create_features(self, df):
        """
        创建特征工程
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            特征工程后的DataFrame
        """
        print("\n" + "=" * 60)
        print("【Step 2】特征工程")
        print("=" * 60)
        
        df = df.copy()
        
        # ========== 时间特征 ==========
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        # 季节编码（用于周期性建模）
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        print("✓ 时间特征: year, month, day_of_year, is_winter, month_sin, month_cos")
        
        # ========== 滞后特征 ==========
        for lag in [1, 2, 3]:
            df[f'tqi_lag{lag}'] = df['tqi'].shift(lag)
        
        print("✓ 滞后特征: tqi_lag1, tqi_lag2, tqi_lag3")
        
        # ========== 统计特征 ==========
        df['tqi_ma3'] = df['tqi'].rolling(window=3, min_periods=1).mean()
        df['tqi_std3'] = df['tqi'].rolling(window=3, min_periods=1).std()
        df['tqi_diff1'] = df['tqi'].diff(1)
        
        print("✓ 统计特征: tqi_ma3, tqi_std3, tqi_diff1")
        
        # ========== 维修特征 ==========
        # 标记维修记录
        df['is_maintenance'] = df['maintenance'].notna().astype(int)
        
        # 距上次维修天数
        df['days_since_maint'] = 999  # 默认值（表示很久未维修）
        last_maint_idx = -1
        
        for idx in range(len(df)):
            if df.loc[idx, 'is_maintenance'] == 1:
                last_maint_idx = idx
                df.loc[idx, 'days_since_maint'] = 0
            elif last_maint_idx >= 0:
                days_diff = (df.loc[idx, 'date'] - df.loc[last_maint_idx, 'date']).days
                df.loc[idx, 'days_since_maint'] = days_diff
        
        # 近6月维修次数
        df['maint_count_6m'] = 0
        for idx in range(len(df)):
            current_date = df.loc[idx, 'date']
            six_months_ago = current_date - pd.DateOffset(months=6)
            count = df[(df['date'] >= six_months_ago) & 
                      (df['date'] <= current_date) & 
                      (df['is_maintenance'] == 1)].shape[0]
            df.loc[idx, 'maint_count_6m'] = count
        
        # 维修前TQI
        df['maint_before_tqi'] = np.nan
        for idx in df[df['is_maintenance'] == 1].index:
            if idx > 0:
                df.loc[idx, 'maint_before_tqi'] = df.loc[idx-1, 'tqi']
        
        print("✓ 维修特征: is_maintenance, days_since_maint, maint_count_6m, maint_before_tqi")
        
        # ========== 趋势特征 ==========
        # 近3月线性趋势（简化版：用3点斜率）
        df['trend_3m'] = 0.0
        for idx in range(2, len(df)):
            y = df.loc[idx-2:idx, 'tqi'].values
            x = np.arange(3)
            slope = np.polyfit(x, y, 1)[0] if len(y) == 3 else 0
            df.loc[idx, 'trend_3m'] = slope
        
        # 近6月线性趋势
        df['trend_6m'] = 0.0
        for idx in range(5, len(df)):
            y = df.loc[idx-5:idx, 'tqi'].values
            x = np.arange(6)
            slope = np.polyfit(x, y, 1)[0] if len(y) == 6 else 0
            df.loc[idx, 'trend_6m'] = slope
        
        print("✓ 趋势特征: trend_3m, trend_6m")
        
        # ========== 冬季三阶段特征 ==========
        df['winter_phase'] = '非冬季'
        
        for idx in range(len(df)):
            if df.loc[idx, 'is_winter'] == 1 and idx >= 2:
                # 计算近3步变化率
                tqi_current = df.loc[idx, 'tqi']
                tqi_3ago = df.loc[idx-2, 'tqi']
                delta = (tqi_current - tqi_3ago) / 3
                
                if delta > 0.1:
                    df.loc[idx, 'winter_phase'] = '上升期'
                elif delta < -0.1:
                    df.loc[idx, 'winter_phase'] = '下降期'
                else:
                    df.loc[idx, 'winter_phase'] = '稳定期'
        
        print("✓ 冬季阶段特征: winter_phase")
        
        # 删除前几行（有NaN的特征）
        df = df.dropna(subset=['tqi_lag3']).reset_index(drop=True)
        
        self.processed_data = df.copy()
        
        print(f"\n✓ 特征工程完成")
        print(f"  - 特征维度: {df.shape[1]} 列")
        print(f"  - 有效样本: {len(df)} 条")
        
        return df
    
    def split_data(self, df):
        """
        划分训练/验证/测试集
        
        划分规则：
        - 训练集：2024-01-06 至 2025-05-31
        - 验证集：2025-06-01 至 2025-08-31（夏季稳定期）
        - 测试集：2025-09-01 至 2025-12-14（秋冬过渡期）
        """
        print("\n" + "=" * 60)
        print("【Step 3】数据集划分")
        print("=" * 60)
        
        train_end = pd.Timestamp('2025-05-31')
        val_end = pd.Timestamp('2025-08-31')
        
        self.train_data = df[df['date'] <= train_end].copy()
        self.val_data = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
        self.test_data = df[df['date'] > val_end].copy()
        
        print(f"✓ 训练集: {len(self.train_data)} 条 ({self.train_data['date'].min().strftime('%Y-%m-%d')} 至 {self.train_data['date'].max().strftime('%Y-%m-%d')})")
        print(f"✓ 验证集: {len(self.val_data)} 条 ({self.val_data['date'].min().strftime('%Y-%m-%d')} 至 {self.val_data['date'].max().strftime('%Y-%m-%d')})")
        print(f"✓ 测试集: {len(self.test_data)} 条 ({self.test_data['date'].min().strftime('%Y-%m-%d')} 至 {self.test_data['date'].max().strftime('%Y-%m-%d')})")
        
        # 打印各集合中的维修记录
        print(f"\n维修记录分布:")
        print(f"  - 训练集: {self.train_data['is_maintenance'].sum()} 次")
        print(f"  - 验证集: {self.val_data['is_maintenance'].sum()} 次")
        print(f"  - 测试集: {self.test_data['is_maintenance'].sum()} 次")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_maintenance_records(self):
        """获取维修记录详情"""
        df = self.processed_data
        maint_records = df[df['is_maintenance'] == 1][[
            'date', 'tqi', 'maint_before_tqi', 'days_since_maint', 
            'maint_count_6m', 'winter_phase'
        ]].copy()
        
        # 计算维修效果
        maint_records['maint_effect'] = maint_records['tqi'] - maint_records['maint_before_tqi']
        
        return maint_records.reset_index(drop=True)
    
    def run(self):
        """运行完整的数据加载流程"""
        print("\n" + "="*70)
        print("  多尺度TQI预测实验 - 数据加载与预处理")
        print("="*70)
        
        # 1. 加载数据
        raw_df = self.load_data()
        
        # 2. 特征工程
        processed_df = self.create_features(raw_df)
        
        # 3. 划分数据集
        self.split_data(processed_df)
        
        # 4. 获取维修记录
        maint_records = self.get_maintenance_records()
        
        print("\n" + "=" * 60)
        print("【维修记录汇总】")
        print("=" * 60)
        print(maint_records.to_string(index=False))
        
        print("\n" + "="*70)
        print("  数据加载完成！")
        print("="*70)
        
        return {
            'raw': raw_df,
            'processed': processed_df,
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data,
            'maintenance': maint_records
        }


if __name__ == "__main__":
    # 测试数据加载
    loader = TQIDataLoader('/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/2号样本.xlsx')
    data = loader.run()
