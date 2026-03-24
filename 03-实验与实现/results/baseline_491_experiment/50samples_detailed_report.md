# 491样本基线实验详细分析报告 (前50样本)

**报告日期**: 2026-03-24  
**实验批次**: 前50个样本 (测试批次)  
**数据来源**: `results/baseline_491_experiment/test_50samples_results.csv`

---

## 一、实验概览

### 1.1 实验配置

| 配置项 | 数值 |
|:---|:---|
| **样本数** | 50个 |
| **方法数** | 7种 |
| **数据划分** | 训练集70% / 验证集15% / 测试集15% |
| **评估指标** | MAE (主要), RMSE, MAPE |
| **运行时间** | 36分钟 |

### 1.2 参与方法

| 编号 | 方法 | 类型 | 特点 |
|:---:|:---|:---|:---|
| 1 | **Historical Mean** | 统计基线 | 常数预测 (训练集均值) |
| 2 | **Moving Average** | 传统统计 | 最近30天均值 |
| 3 | **Holt-Winters** | 传统统计 | 季节性指数平滑 |
| 4 | **MLP** | 深度学习 | 多层感知机 (sklearn) |
| 5 | **LSTM** | 深度学习 | 循环神经网络 (TensorFlow) |
| 6 | **TimeMixer** | 深度学习 | 时序混合模型 (ICLR 2023) |
| 7 | **Trident** | **我们的方法** | 滚动锚定框架 |

---

## 二、核心结果

### 2.1 MAE排名 (从低到高)

| 排名 | 方法 | 平均MAE | 中位数MAE | 标准差 | 获胜次数 |
|:---:|:---|:---:|:---:|:---:|:---:|
| 🥇 | **Historical Mean** | **0.958** | 1.000 | 0.40 | 8 |
| 🥈 | **TimeMixer** | 1.252 | 1.224 | 0.64 | 3 |
| 🥉 | **Trident** | 1.298 | 1.356 | 0.48 | 1 |
| 4 | Moving Average | 1.364 | 1.303 | 0.57 | 2 |
| 5 | Holt-Winters | 1.601 | 0.920 | 1.87 | 11 |
| 6 | MLP | 2.526 | 2.136 | 1.32 | 0 |
| ❌ | **LSTM** | **287.99** | 0.973 | 1691.52 | 25 |

> **注**: LSTM平均MAE异常高是因为5个样本出现灾难性失败(MAE>10,000)，中位数0.973其实表现不错。

### 2.2 关键发现

#### 🔍 发现1: 简单方法反而更好
- **Historical Mean** (最简单的常数预测) 获得最低平均MAE
- 原因分析: TQI数据本身波动较小，复杂模型容易过拟合

#### 🔍 发现2: LSTM表现两极分化
- **25个样本** LSTM获胜 (最高获胜次数!)
- **5个样本** 出现灾难性失败 (MAE > 10)
  - 712400: MAE = 11,926
  - 725400: MAE = 93.7
  - 733400: MAE = 720
  - 735400: MAE = 432.9
  - 748400: MAE = 1,184

#### 🔍 发现3: Trident表现稳健但不够突出
- 平均MAE排名第三，但获胜次数仅1次
- 标准差0.48，稳定性较好
- 相比最佳基线，**49/50样本表现更差**

---

## 三、Trident深度分析

### 3.1 Trident表现统计

| 指标 | 数值 | 解读 |
|:---|:---:|:---|
| 平均MAE | 1.298 | 中等水平 |
| 中位数MAE | 1.356 | 多数样本在此附近 |
| 标准差 | 0.476 | 波动性中等 |
| 最小MAE | 0.216 | 样本712400 (最优表现) |
| 最大MAE | 2.324 | 样本721400 (最差表现) |
| 获胜次数 | 1/50 (2%) | 竞争力不足 |
| 改善样本数 | 1/50 | 仅1个样本优于最佳基线 |

### 3.2 Trident vs 各方法对比

| 对比方法 | Trident更优样本数 | Trident更差样本数 | 平均差距 |
|:---|:---:|:---:|:---:|
| Historical Mean | 1/50 | 49/50 | +35.5% |
| Moving Average | 7/50 | 43/50 | -4.9% |
| Holt-Winters | 14/50 | 36/50 | -18.9% |
| MLP | 47/50 | 3/50 | +48.6% |
| TimeMixer | 5/50 | 45/50 | -3.7% |

### 3.3 样本难度分布 (基于Trident MAE)

| 难度等级 | MAE范围 | 样本数 | 占比 |
|:---|:---:|:---:|:---:|
| **Easy** | < 0.5 | 4 | 8% |
| **Medium** | 0.5 - 1.0 | 10 | 20% |
| **Hard** | 1.0 - 1.5 | 22 | 44% |
| **Very Hard** | 1.5 - 2.0 | 12 | 24% |
| **Extreme** | > 2.0 | 2 | 4% |

> **结论**: 68%的样本属于Hard或Very Hard，Trident在这些样本上表现有提升空间。

---

## 四、Trident优化方向

基于前50样本的详细分析，提出以下**5个优化方向**:

### 优化方向1: 季节性参数自适应 🔥 (优先级: 高)

**问题**: 当前Trident使用固定的季节性模式，没有考虑不同里程的季节性差异。

**证据**: 
- 样本721400 (Trident MAE=2.324, 最差) 的Holt-Winters MAE=1.411
- Holt-Winters能捕捉季节性，而Trident固定模式失效

**优化方案**:
```python
# 当前: 固定季节性系数
seasonal_factor = default_seasonal_pattern[month]

# 优化: 每个样本自适应季节性
sample_seasonal = detect_seasonality(train_data)  # 用FFT或ACF检测
seasonal_factor = sample_seasonal[month]
```

**预期提升**: 在季节性明显的样本上，MAE可降低20-30%

---

### 优化方向2: 劣化趋势建模改进 🔥 (优先级: 高)

**问题**: 当前线性劣化趋势过于简单，无法捕捉非线性劣化。

**证据**:
- Historical Mean (无趋势) MAE=0.958 < Trident MAE=1.298
- 说明Trident的线性趋势反而引入了误差

**优化方案**:
```python
# 当前: 线性趋势
degradation = trend_rate * days

# 优化方案A: 分段线性 (维修后重置)
if detect_maintenance(train_data):
    trend_rate = calculate_trend_after_maintenance()

# 优化方案B: 非线性趋势 (指数/对数)
degradation = a * np.log(days + 1) + b * days  # 对数+线性混合

# 优化方案C: 数据驱动趋势 (平滑样条)
from scipy.interpolate import UnivariateSpline
trend_model = UnivariateSpline(train_days, train_tqi, s=smoothing_factor)
```

**预期提升**: 非线性劣化样本MAE降低15-25%

---

### 优化方向3: 锚定值计算优化 (优先级: 中)

**问题**: 当前使用训练集最后值的移动平均作为锚定值，可能不够稳定。

**优化方案**:
```python
# 当前: 简单移动平均
anchor = train_data['tqi'].tail(anchor_window).mean()

# 优化: 加权移动平均 (近期权重更高)
weights = np.exp(np.linspace(-1, 0, anchor_window))
anchor = np.average(train_data['tqi'].tail(anchor_window), weights=weights)

# 或: 指数平滑
anchor = train_data['tqi'].ewm(span=anchor_window).mean().iloc[-1]
```

**预期提升**: 锚定值稳定性提升，MAE降低5-10%

---

### 优化方向4: 集成策略 (优先级: 中)

**问题**: Trident单独表现不如Historical Mean，但可能互补。

**优化方案**: Trident + Historical Mean 集成
```python
# 简单平均集成
ensemble_pred = 0.5 * trident_pred + 0.5 * historical_mean

# 或: 基于验证集性能加权
if val_mae_trident < val_mae_historical:
    weight = 0.7
else:
    weight = 0.3
ensemble_pred = weight * trident_pred + (1-weight) * historical_mean
```

**验证思路**: 在当前50样本上测试，Historical Mean平均0.958，Trident平均1.298，简单平均可能降到1.1左右。

---

### 优化方向5: 异常样本检测与回退 (优先级: 中)

**问题**: 某些样本可能不适合Trident模型（如频繁维修、数据异常）。

**优化方案**:
```python
# 训练阶段: 检测样本是否适合Trident
def sample_suitability(train_data):
    # 检查1: 是否有明显趋势
    trend_strength = np.abs(np.polyfit(range(len(train_data)), train_data, 1)[0])
    
    # 检查2: 季节性是否明显
    seasonal_variance = calculate_seasonal_variance(train_data)
    
    # 检查3: 数据稳定性
    data_variance = train_data.std()
    
    # 综合评分
    suitability_score = trend_score + seasonal_score - variance_penalty
    return suitability_score

# 预测阶段: 不适合的样本回退到Historical Mean
if suitability_score < threshold:
    return historical_mean_baseline(train_data, test_data)
else:
    return trident_predict(train_data, test_data)
```

---

## 五、下一步实验建议

### 建议A: 快速验证优化方向 (推荐)

**时间**: 1-2小时  
**内容**: 选择优化方向1和2，在前50样本上做A/B测试

**步骤**:
1. 实现自适应季节性检测 (FFT方法)
2. 实现非线性劣化趋势 (对数+线性混合)
3. 对比优化前后Trident的MAE
4. 如效果显著，应用到全量491样本

### 建议B: 全量491样本实验

**时间**: 2-3小时 (取决于LSTM是否保留)  
**内容**: 用当前版本跑全量数据，获得完整统计

**风险**: Trident在当前版本下可能整体表现不如Historical Mean

### 建议C: 移除LSTM，简化实验

**理由**: LSTM 50 epoch在小数据上不稳定  
**替代**: 用10-20 epoch，或完全移除LSTM (保留6种方法)

---

## 六、附录

### 6.1 可视化图表

- `50samples_detailed_analysis.png`: 9子图综合分析
  - MAE分布直方图
  - MAE箱线图
  - 平均MAE对比
  - Trident vs Baselines散点图
  - 获胜次数统计
  - Trident改善率分布
  - 样本难度分布
  - MAE热力图
  - 方法相关性矩阵

### 6.2 原始数据文件

- `test_50samples_results.csv`: 完整实验结果
- `test_50samples_report.json`: 统计摘要

### 6.3 LSTM异常样本详情

| 样本里程 | LSTM MAE | 可能原因 |
|:---:|:---:|:---|
| 712400 | 11,926 | 梯度爆炸 |
| 725400 | 93.7 | 过拟合 |
| 733400 | 720.0 | 训练不稳定 |
| 735400 | 432.9 | 训练不稳定 |
| 748400 | 1,184 | 梯度爆炸 |

---

**报告生成**: Kimi Claw  
**审核状态**: 待用户审阅优化方案
