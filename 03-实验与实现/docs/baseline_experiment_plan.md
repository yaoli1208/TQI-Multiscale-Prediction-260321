# TQI预测基线对比实验方案 (491样本)

**版本**: v1.1  
**更新日期**: 2026-03-24  
**样本数量**: 491个高质量样本  
**更新内容**: 将"数据驱动基线"替换为"历史均值基线"，确保有文献支撑

---

## 一、实验目标

1. 确定491个样本各自的最佳基线方法
2. 量化Trident策略相比最佳基线的改善幅度
3. 识别Trident有效的样本特征
4. 建立样本-方法适配关系

---

## 二、基线方法选择

### 2.1 必测基线（6种）

| 类别 | 方法 | 标准命名 | 原因 |
|:---|:---|:---|:---|
| **统计基线** | 历史均值 | Historical Mean | 下限基线，证明新方法价值 |
| **传统统计** | 移动平均 (30天) | Moving Average (MA) | 工程常用，33%样本最佳 |
| **传统统计** | Holt-Winters | Triple Exponential Smoothing | 捕捉季节性，适合TQI数据 |
| **深度学习** | LSTM | Long Short-Term Memory | 验证深度学习在小数据表现 |
| **深度学习** | **TimeMixer** | TimeMixer (ICLR 2023) | 21%样本最佳，小数据SOTA |
| **我们提出** | **Trident (滚动锚定)** | Trident Framework | 核心对比方法 |

### 2.2 可选基线（3种，抽样验证）

| 方法 | 验证目的 |
|:---|:---|
| XGBoost | 验证梯度提升树效果 |
| PatchTST | 最新Transformer SOTA |
| 维修后锚定 | Trident另一策略 |

---

## 三、实验设计

### 3.1 时间序列划分策略

```
训练集: 70% (历史数据)
验证集: 15% (调参使用)
测试集: 15% (最终评估)
```

**划分方式**: 按时间顺序划分，禁止随机打乱

```python
# 示例: 13年数据划分
data_2012_2026 = 4845天
├── 训练集: 2012-2021 (约9年, 3391天)
├── 验证集: 2022-2023 (约2年, 727天)
└── 测试集: 2024-2026 (约2年, 727天)
```

### 3.2 预测任务定义

**任务类型**: 单步预测 (One-step Ahead Prediction)
- 输入: 历史TQI序列
- 输出: 下一个检测时间点的TQI值

**预测粒度**: 每次预测1个时间步

### 3.3 评估指标

| 指标 | 公式 | 说明 |
|:---|:---|:---|
| **MAE** | mean(\|y_pred - y_true\|) | 主要指标，绝对误差 |
| RMSE | sqrt(mean((y_pred - y_true)²)) | 对大误差敏感 |
| MAPE | mean(\|y_pred - y_true\|/y_true) | 百分比误差 |
| R² | 1 - SSE/SST | 拟合优度 |

**主要决策指标**: MAE

---

## 四、各方法实现细节

### 4.1 历史均值 (Historical Mean)

```python
def historical_mean_predict(train_data):
    """
    历史均值基线 - 最简单的预测方法
    使用训练集整体均值作为未来所有时间点的预测值
    
    文献参考:
    - Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice
    - Makridakis et al. (2020). The M5 Accuracy competition
    """
    return np.mean(train_data)
```

**特点**:
- 预测值为常数（不随时间变化）
- 时间序列预测的"下限基线"
- 如果连历史均值都超不过，新方法无价值

### 4.2 移动平均 (Moving Average)

```python
def ma_predict(train_data, window=30):
    """
    使用最近window个时间步的平均值
    """
    return np.mean(train_data[-window:])
```

### 4.3 Holt-Winters (三重指数平滑)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def holt_winters_predict(train_data, seasonal_periods=12):
    """
    Holt-Winters季节性指数平滑
    - 捕捉趋势和季节性
    """
    model = ExponentialSmoothing(
        train_data,
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods
    )
    fitted = model.fit()
    return fitted.forecast(steps=1)
```

### 4.4 LSTM

```python
def lstm_predict(train_data, window=30):
    """
    LSTM预测
    - 序列长度: window (默认30)
    - 隐藏层: 64
    - 层数: 2
    - 早停: 验证集MAE不下降则停止
    """
    # 构建序列数据
    # 标准化
    # 训练LSTM
    # 预测
    pass
```

### 4.5 TimeMixer

```python
def timemixer_predict(train_data):
    """
    TimeMixer预测
    - 使用官方实现或简化版
    - 多尺度混合架构
    - 适合小数据场景
    """
    pass
```

### 4.6 Trident (滚动锚定)

```python
def trident_rolling_anchor(train_data, dates):
    """
    Trident滚动锚定策略
    1. 识别年度基准期 (每年1-2月)
    2. 学习季节性调整因子
    3. 叠加劣化趋势
    4. 预测 = 基准值 + 季节性 + 趋势
    """
    # 提取年度基准值
    # 计算季节性因子
    # 拟合劣化趋势
    # 组合预测
    pass
```

---

## 五、实验流程

### 5.1 整体流程

```
开始
  │
  ▼
读取491个合格样本列表
  │
  ▼
对于每个样本 (共491个):
  │
  ├── 读取该样本的时间序列数据
  │
  ├── 数据预处理
  │   ├── 按时间排序
  │   ├── 处理缺失值
  │   └── 标准化(如需要)
  │
  ├── 时间序列划分 (70/15/15)
  │
  ├── 对每个基线方法:
  │   ├── 在训练集上训练/拟合
  │   ├── 在验证集上调参(如需要)
  │   └── 在测试集上预测并计算MAE
  │
  ├── 记录各方法MAE
  │
  ├── 确定最佳基线 (最小MAE)
  │
  └── 计算Trident改善幅度
  │
  ▼
汇总所有样本结果
  │
  ▼
统计分析
  ├── 各基线获胜次数统计
  ├── Trident有效率统计
  └── 有效样本特征分析
  │
  ▼
生成实验报告
```

### 5.2 分批执行策略

491个样本分批次执行，避免内存溢出:

```python
BATCH_SIZE = 50  # 每批处理50个样本
NUM_BATCHES = 10  # 共10批

for batch_id in range(NUM_BATCHES):
    batch_samples = samples[batch_id*BATCH_SIZE : (batch_id+1)*BATCH_SIZE]
    results = process_batch(batch_samples)
    save_checkpoint(results, batch_id)
```

---

## 六、输出结果定义

### 6.1 单个样本结果格式

```json
{
  "tqi_mile": 733400,
  "record_count": 484,
  "time_span_days": 4845,
  "tqi_mean": 3.69,
  "results": {
    "historical_mean": {"mae": 0.385, "rmse": 0.452, "mape": 10.5},
    "ma_30": {"mae": 0.206, "rmse": 0.268, "mape": 5.6},
    "holt_winters": {"mae": 0.223, "rmse": 0.291, "mape": 6.1},
    "lstm": {"mae": 0.658, "rmse": 0.824, "mape": 17.8},
    "timemixer": {"mae": 0.255, "rmse": 0.332, "mape": 6.9},
    "trident_rolling": {"mae": 0.161, "rmse": 0.208, "mape": 4.4}
  },
  "best_baseline": "ma_30",
  "best_baseline_mae": 0.206,
  "trident_mae": 0.161,
  "trident_improvement": 0.218,
  "is_trident_best": true,
  "is_trident_effective": true
}
```

### 6.2 汇总统计结果

```json
{
  "total_samples": 491,
  "baseline_wins": {
    "historical_mean": 45,
    "ma_30": 165,
    "holt_winters": 42,
    "lstm": 12,
    "timemixer": 127
  },
  "trident_stats": {
    "is_best_count": 298,
    "is_best_ratio": 0.607,
    "effective_count": 245,
    "effective_ratio": 0.499,
    "mean_improvement": 0.156
  },
  "effective_samples": [733400, 734400, ...],
  "ineffective_samples": [820400, 1023400, ...]
}
```

---

## 七、代码结构

```
experiments/
├── baseline_experiment.py      # 主实验脚本
├── methods/
│   ├── __init__.py
│   ├── historical_mean.py      # 历史均值基线
│   ├── moving_average.py       # 移动平均
│   ├── holt_winters.py         # Holt-Winters
│   ├── lstm_model.py           # LSTM
│   ├── timemixer_model.py      # TimeMixer
│   └── trident.py              # Trident策略
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # 数据加载
│   ├── evaluation.py           # 评估指标
│   └── visualization.py        # 可视化
├── config.yaml                 # 实验配置
└── run_experiment.sh           # 运行脚本
```

---

## 八、时间安排

| 阶段 | 任务 | 预计时间 |
|:---|:---|:---:|
| 第1天 | 基线方法实现 (除TimeMixer) | 4小时 |
| 第2天 | TimeMixer集成 & 调试 | 4小时 |
| 第3天 | 小规模验证 (10个样本) | 2小时 |
| 第4天 | 全量实验执行 (491样本) | 8-12小时 |
| 第5天 | 结果分析与报告生成 | 4小时 |

---

## 九、风险评估

| 风险 | 影响 | 应对策略 |
|:---|:---:|:---|
| TimeMixer实现复杂 | 高 | 使用简化版或官方预训练权重 |
| 实验运行时间过长 | 中 | 分批执行，设置checkpoint |
| 内存溢出 | 中 | 每批处理50个样本，及时释放内存 |
| 某些样本所有方法效果都很差 | 低 | 标记为"困难样本"，单独分析 |

---

**下一步**: 开始实现基线方法代码？
