"""
多尺度TQI预测实验 - 数据加载与预处理模块（增强版）
=============================================
支持长期历史数据（2012年至今）
支持无维修记录的数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TQIDataLoader:
    """TQI数据加载与预处理类（增强版）"""
    
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
        self.has_maintenance = False
        self.min_year = None
        self.max_year = None
        
    def load_data(self):
        """加载原始数据"""
        print("=" * 60)
        print("【Step 1】加载原始数据")
        print("=" * 60)
        
        df = pd.read_excel(self.file_path)
        
        # 自动检测日期列名
        date_col = None
        for col in df.columns:
            if '日期' in col or 'date' in col.lower():
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("未找到日期列，请确保列名包含'日期'或'date'")
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # 重命名列为标准格式
        col_mapping = {date_col: 'date'}
        
        # 自动检测TQI列
        for col in df.columns:
            if 'TQI' in col or 'tqi' in col.lower():
                col_mapping[col] = 'tqi'
                break
        
        # 自动检测维修记录列
        for col in df.columns:
            if '维修' in col or 'maint' in col.lower():
                col_mapping[col] = 'maintenance'
                self.has_maintenance = True
                break
        
        df.rename(columns=col_mapping, inplace=True)
        
        if 'tqi' not in df.columns:
            raise ValueError("未找到TQI列")
        
        self.raw_data = df.copy()
        self.min_year = df['date'].dt.year.min()
        self.max_year = df['date'].dt.year.max()
        
        print(f"✓ 加载完成: {len(df)} 条记录")
        print(f"✓ 时间范围: {df['date'].min().strftime('%Y-%m-%d')} 至 {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"✓ 跨度: {self.max_year - self.min_year + 1} 年 ({self.min_year}-{self.max_year})")
        
        if self.has_maintenance:
            maint_count = df['maintenance'].notna().sum()
            print(f"✓ 维修记录: {maint_count} 次")
        else:
            print(f"⚠ 无维修记录列，将跳过维修相关特征")
        
        return df
    
    def create_features(self, df):
        """创建特征工程"""
        print("\n" + "=" * 60)
        print("【Step 2】特征工程")
        print("=" * 60)
        
        df = df.copy()
        
        # 时间特征
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['year_norm'] = (df['year'] - self.min_year) / max(1, self.max_year - self.min_year)
        
        print("✓ 时间特征: year, month, day_of_year, is_winter, month_sin, month_cos, year_norm")
        
        # 滞后特征
        for lag in [1, 2, 3]:
            df[f'tqi_lag{lag}'] = df['tqi'].shift(lag)
        print("✓ 滞后特征: tqi_lag1, tqi_lag2, tqi_lag3")
        
        # 统计特征
        df['tqi_ma3'] = df['tqi'].rolling(window=3, min_periods=1).mean()
        df['tqi_std3'] = df['tqi'].rolling(window=3, min_periods=1).std()
        df['tqi_diff1'] = df['tqi'].diff(1)
        
        # 年度统计（如果跨多年）
        if self.max_year - self.min_year >= 2:
            yearly_mean = df.groupby('year')['tqi'].transform('mean')
            df['tqi_year_mean'] = yearly_mean
            df['tqi_dev_from_year'] = df['tqi'] - yearly_mean
            print("✓ 统计特征: tqi_ma3, tqi_std3, tqi_diff1, tqi_year_mean, tqi_dev_from_year")
        else:
            print("✓ 统计特征: tqi_ma3, tqi_std3, tqi_diff1")
        
        # 维修特征（如果有）
        if self.has_maintenance:
            df['is_maintenance'] = df['maintenance'].notna().astype(int)
            df['days_since_maint'] = 999
            last_maint_idx = -1
            
            for idx in range(len(df)):
                if df.loc[idx, 'is_maintenance'] == 1:
                    last_maint_idx = idx
                    df.loc[idx, 'days_since_maint'] = 0
                elif last_maint_idx >= 0:
                    days_diff = (df.loc[idx, 'date'] - df.loc[last_maint_idx, 'date']).days
                    df.loc[idx, 'days_since_maint'] = days_diff
            
            df['maint_count_6m'] = 0
            for idx in range(len(df)):
                current_date = df.loc[idx, 'date']
                six_months_ago = current_date - pd.DateOffset(months=6)
                count = df[(df['date'] >= six_months_ago) & 
                          (df['date'] <= current_date) & 
                          (df['is_maintenance'] == 1)].shape[0]
                df.loc[idx, 'maint_count_6m'] = count
            
            df['maint_before_tqi'] = np.nan
            for idx in df[df['is_maintenance'] == 1].index:
                if idx > 0:
                    df.loc[idx, 'maint_before_tqi'] = df.loc[idx-1, 'tqi']
            
            print("✓ 维修特征: is_maintenance, days_since_maint, maint_count_6m, maint_before_tqi")
        else:
            df['is_maintenance'] = 0
            df['days_since_maint'] = 999
            df['maint_count_6m'] = 0
            df['maint_before_tqi'] = np.nan
            print("⚠ 无维修数据，维修特征设为默认值")
        
        # 趋势特征
        df['trend_3m'] = 0.0
        for idx in range(2, len(df)):
            y = df.loc[idx-2:idx, 'tqi'].values
            x = np.arange(3)
            slope = np.polyfit(x, y, 1)[0] if len(y) == 3 else 0
            df.loc[idx, 'trend_3m'] = slope
        
        df['trend_6m'] = 0.0
        for idx in range(5, len(df)):
            y = df.loc[idx-5:idx, 'tqi'].values
            x = np.arange(6)
            slope = np.polyfit(x, y, 1)[0] if len(y) == 6 else 0
            df.loc[idx, 'trend_6m'] = slope
        
        if self.max_year - self.min_year >= 2:
            df['trend_year'] = 0.0
            for idx in range(len(df)):
                year = df.loc[idx, 'year']
                year_data = df[df['year'] == year]['tqi'].values
                if len(year_data) >= 3:
                    x = np.arange(len(year_data))
                    slope = np.polyfit(x, year_data, 1)[0]
                    df.loc[idx, 'trend_year'] = slope
            print("✓ 趋势特征: trend_3m, trend_6m, trend_year")
        else:
            print("✓ 趋势特征: trend_3m, trend_6m")
        
        # 冬季三阶段特征
        df['winter_phase'] = '非冬季'
        for idx in range(len(df)):
            if df.loc[idx, 'is_winter'] == 1 and idx >= 2:
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
        
        # 删除NaN行
        df = df.dropna(subset=['tqi_lag3']).reset_index(drop=True)
        
        self.processed_data = df.copy()
        
        print(f"\n✓ 特征工程完成")
        print(f"  - 特征维度: {df.shape[1]} 列")
        print(f"  - 有效样本: {len(df)} 条")
        
        return df
    
    def split_data(self, df, train_ratio=0.7, val_ratio=0.15):
        """
        划分训练/验证/测试集
        
        Args:
            df: 处理后的数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        print("\n" + "=" * 60)
        print("【Step 3】数据集划分")
        print("=" * 60)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        self.train_data = df.iloc[:train_end].copy()
        self.val_data = df.iloc[train_end:val_end].copy()
        self.test_data = df.iloc[val_end:].copy()
        
        print(f"✓ 训练集: {len(self.train_data)} 条 ({self.train_data['date'].min().strftime('%Y-%m-%d')} 至 {self.train_data['date'].max().strftime('%Y-%m-%d')})")
        print(f"✓ 验证集: {len(self.val_data)} 条 ({self.val_data['date'].min().strftime('%Y-%m-%d')} 至 {self.val_data['date'].max().strftime('%Y-%m-%d')})")
        print(f"✓ 测试集: {len(self.test_data)} 条 ({self.test_data['date'].min().strftime('%Y-%m-%d')} 至 {self.test_data['date'].max().strftime('%Y-%m-%d')})")
        
        if self.has_maintenance:
            print(f"\n维修记录分布:")
            print(f"  - 训练集: {self.train_data['is_maintenance'].sum()} 次")
            print(f"  - 验证集: {self.val_data['is_maintenance'].sum()} 次")
            print(f"  - 测试集: {self.test_data['is_maintenance'].sum()} 次")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_maintenance_records(self):
        """获取维修记录详情"""
        if not self.has_maintenance:
            return pd.DataFrame()
        
        df = self.processed_data
        maint_records = df[df['is_maintenance'] == 1][[
            'date', 'tqi', 'maint_before_tqi', 'days_since_maint', 
            'maint_count_6m', 'winter_phase'
        ]].copy()
        
        maint_records['maint_effect'] = maint_records['tqi'] - maint_records['maint_before_tqi']
        
        return maint_records.reset_index(drop=True)
    
    def run(self):
        """运行完整的数据加载流程"""
        print("\n" + "="*70)
        print("  多尺度TQI预测实验 - 数据加载与预处理（增强版）")
        print("="*70)
        
        # 1. 加载数据
        raw_df = self.load_data()
        
        # 2. 特征工程
        processed_df = self.create_features(raw_df)
        
        # 3. 划分数据集
        self.split_data(processed_df)
        
        # 4. 获取维修记录
        maint_records = self.get_maintenance_records()
        
        if self.has_maintenance and len(maint_records) > 0:
            print("\n" + "=" * 60)
            print("【维修记录汇总】")
            print("=" * 60)
            print(maint_records.to_string(index=False))
        
        print("\n" + "="*70)
        print("  数据加载完成！")
        print("="*70)
        
        return {
            'raw': raw_df,
            'processed': self.processed_data,
            'train': self.train_data,
            'val': self.val_data,
            'test': self.test_data,
            'maintenance': maint_records,
            'has_maintenance': self.has_maintenance,
            'year_range': (self.min_year, self.max_year)
        }


if __name__ == "__main__":
    # 测试数据加载
    loader = TQIDataLoader('/root/.openclaw/workspace/02-research/active/2026-03-21-基于大模型的数据智能分析助手框架研究/03-实验与实现/2号样本.xlsx')
    data = loader.run()
