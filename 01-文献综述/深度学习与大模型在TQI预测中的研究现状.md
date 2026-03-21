# 深度学习与大模型在TQI预测中的研究现状

**调研时间**: 2026-03-21  
**调研范围**: 深度学习、时间序列预测大模型、TQI预测开源代码

---

## 一、深度学习在TQI预测中的应用

### 1.1 主流深度学习方法

| 方法 | 特点 | 典型研究 | 开源代码 |
|------|------|----------|----------|
| **LSTM/GRU** | 捕捉长期时序依赖 | 北京交通大学地铁TQI预测 | 可用PyTorch/TensorFlow实现 |
| **BiLSTM-CNN** | 双向+卷积特征提取 | AQI预测(类似任务) | 参考AQI-LSTM-prediction |
| **GRNN** | 径向基网络，收敛快 | Nedevska et al. (2025) | Neural Tools平台 |
| **3D CNN** | 空间-时间联合建模 | Rail3D点云分割 | github.com/Rail3D |

### 1.2 最新研究成果

**论文1: Predictive Modeling of Track Quality Index with Neural Networks (2025)**
- **作者**: Ivona Nedevska Trajkova, Zlatko Zafirovski
- **方法**: General Regression Neural Network (GRNN)
- **数据集**: 821个观测样本（多周期轨道检测数据）
- **结果**: R²=0.699, RMSE=4.70, MAE=3.55
- **关键发现**: 历史TQI值和捣固次数是最重要预测因子
- **链接**: https://www.engineeringscience.rs/articles/predictive-modeling-of-track-quality-index-with-neural-networks

**论文2: 基于机器学习的地铁轨道几何劣化规律个性化建模研究 (2020)**
- **作者**: 王志鸥、刘仍奎、邱荣华、韩蕊
- **机构**: 北京交通大学交通运输学院
- **数据**: 北京地铁1号线，2016-2019年，17次TQI检测数据
- **方法**: 前馈神经网络
- **结果**: R²=0.938, MAPE=4.80%
- **DOI**: 10.3969/j.jssn.1672-6073.2020.04.010

**论文3: Predicting Track Geometry Using Machine Learning Methods**
- **机构**: UNLV (University of Nevada, Las Vegas)
- **方法**: 机器学习预测轨道几何形位
- **发现**: 预测误差随时间增加，6个月后MSE增加53%
- **应用**: 预测性维护规划
- **链接**: https://www.unlv.edu/sites/default/files/media/document/2024-11/railteam-final_report-predicting_track_geometry_using_machine_learning_methods.pdf

---

## 二、时间序列大模型（LLM for Time Series）

### 2.1 开源时间序列大模型

| 模型 | 机构 | 特点 | 代码地址 |
|------|------|------|----------|
| **Lag-Llama** | Morgan Stanley, ServiceNow, Mila | 首个开源时序大模型，单变量概率预测 | github.com/time-series-foundation-models/lag-llama |
| **Time-LLM** | 清华等 (ICLR 2024) | 重新编程LLM进行时序预测 | github.com/KimMeen/Time-LLM |
| **Timer** | 清华大学 | 生成式Transformer，单序列格式 | github.com/thuml/Timer |
| **AutoTimes** | THUML | 自回归时序预测，兼容任意LLM | github.com/thuml/AutoTimes |
| **MOMENT** | 多机构 | 通用时序分析基础模型家族 | github.com/moment-timeseries-foundation-model |

### 2.2 大模型详细对比

#### Lag-Llama
- **发布时间**: 2024年2月
- **论文**: *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting*
- **核心特点**:
  - 基于Transformer的纯解码器架构
  - 使用滞后特征构建时间序列标记
  - 支持多频率数据（季度、月、周、天、小时、秒）
  - 零样本(zero-shot)预测能力
  - 概率预测（提供置信区间）
- **训练数据**: 27个时间序列数据集，多领域
- **对比实验**: 与TFT、DeepAR对比，5 epoch训练
- **适用场景**: 单变量时间序列预测，可扩展到TQI预测

#### Time-LLM (ICLR 2024)
- **核心思想**: 将时间序列预测任务重新编程(reprogramming)为LLM可处理的形式
- **支持任务**:
  - 长期预测(long_term_forecast)
  - 短期预测(short_term_forecast)
  - 插值(imputation)
  - 分类(classification)
  - 异常检测(anomaly_detection)
- **支持LLM**: LLaMA, GPT2, BERT
- **优势**: 利用预训练LLM的语义理解能力

#### Timer (清华大学)
- **核心创新**: 
  - 单序列(S3)格式：将异构时序统一为一维序列
  - 仅解码器架构（类似GPT）
  - 下一词预测(Next Token Prediction)预训练
- **架构**: GPT风格的Decoder-only Transformer
- **优势**: 自回归生成，可预测任意长度

### 2.4 TimeMixer: 多尺度混合架构 (ICLR 2024)

**论文**: *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*  
**机构**: 蚂蚁集团 + 清华大学  
**会议**: ICLR 2024  
**GitHub**: https://github.com/kwuking/TimeMixer

#### 核心思想
TimeMixer 提出**多尺度混合（Multiscale Mixing）**新视角：
- 不同采样尺度呈现不同模式（细尺度→微观信息，粗尺度→宏观信息）
- 纯MLP架构，但性能超越Transformer类模型
- 计算效率极高，适合工业部署

#### 架构设计

```
输入序列
    ↓
多尺度生成（平均池化降采样）→ x0, x1, x2, ... xM
    ↓
┌─────────────────────────────────────┐
│ Past-Decomposable-Mixing (PDM)      │
│  • 对每个尺度做分解（季节+趋势）     │
│  • 季节性：自底向上混合（细→粗）     │
│  • 趋势：自顶向下混合（粗→细）       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Future-Multipredictor-Mixing (FMM)  │
│  • 多尺度预测器集成                  │
│  • 利用各尺度的互补预测能力          │
└─────────────────────────────────────┘
    ↓
预测输出
```

#### 关键创新

| 组件 | 功能 | 效果 |
|------|------|------|
| **PDM** | 分解多尺度的季节/趋势成分，双向混合 | 微观季节性+宏观趋势信息聚合 |
| **FMM** | 多预测器集成 | 互补预测能力融合 |
| **纯MLP** | 无Attention，纯多层感知机 | 计算效率极高，SOTA性能 |

#### 性能表现
- **长期预测** + **短期预测**：双SOTA
- **8类时间序列任务**：预测、插补、分类、异常检测等全面领先
- **运行效率**：优于Transformer类模型

#### TimeMixer++ (ICLR 2025)
**升级版**新增能力：
- 多分辨率时间成像（MRTI）
- 时间图像分解（TID）
- 支持更通用的预测分析任务

#### 与我们的多尺度模型对比

| 特性 | TimeMixer | DeepMultiscaleLLM (我们的设计) |
|------|-----------|-------------------------------|
| 分解层 | PDM多尺度分解 | STL分解 |
| 特征提取 | MLP混合 | LSTM/Transformer/CNN |
| 大模型融合 | ❌ 无 | ✅ Lag-Llama/Time-LLM |
| 可解释性 | ⚠️ 中等 | ✅ 显式分层输出 |
| 零样本能力 | ❌ 无 | ✅ 继承大模型能力 |
| 计算效率 | ✅ 极高 | ⚠️ 中等（大模型开销） |

**结论**: TimeMixer是非常强的基线模型，建议纳入对比实验。

---

### 2.5 大模型应用建议

**对于TQI预测任务**:

1. **短期预测(1-3个月)**
   - 推荐: Lag-Llama（零样本，快速部署）
   - 备选: Time-LLM + LLaMA微调

2. **长期趋势分析(年度劣化)**
   - 推荐: Timer（自回归生成，适合长期预测）
   - 备选: 传统LSTM/GRU（可解释性强）

3. **多变量预测（加入环境因素）**
   - 推荐: Time-LLM（利用LLM的多模态能力）
   - 备选: TFT (Temporal Fusion Transformer)

---

## 三、相关开源数据集与代码资源

### 3.1 TQI相关开源项目

| 项目名称 | 内容 | 链接 |
|----------|------|------|
| **Rail3D** | 轨道点云分割、TQI相关 | github.com/Rail3D |
| **UFATD** | 轨道异物入侵检测 | github.com/UFATD |
| **PMx-Data** | 预测性维护数据集汇总 | github.com/PMx-Data |

### 3.2 通用时间序列预测库

| 库名 | 功能 | 安装 |
|------|------|------|
| **GluonTS** | 概率时序预测（DeepAR, TFT等） | `pip install gluonts` |
| **Darts** | 多模型集成（Prophet, TCN, N-BEATS等） | `pip install u8darts` |
| **NeuralForecast** | 神经网络时序预测 | `pip install neuralforecast` |
| **AutoTS** | 自动时序预测 | `pip install autots` |

---

## 四、实验建议与对比方案

### 4.1 基于14年TQI数据的实验设计

| 实验组 | 方法 | 目的 |
|--------|------|------|
| **Baseline** | ARIMA, Prophet | 传统方法基线 |
| **DL-1** | LSTM/GRU | 验证深度学习有效性 |
| **DL-2** | Transformer | 验证注意力机制 |
| **LLM-1** | Lag-Llama (零样本) | 验证大模型零样本能力 |
| **LLM-2** | Time-LLM (微调) | 验证大模型微调效果 |
| **Ours** | 多尺度融合模型 | 对比验证本研究方法的优越性 |

### 4.2 评估指标

- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数
- **sMAPE**: 对称平均绝对百分比误差
- **ND**: 归一化偏差

### 4.3 关键问题与注意事项

1. **数据频率**: TQI通常为14天周期，属于低频时序数据
2. **维修干预**: 14年数据中维修记录可能缺失，需考虑干预效应
3. **季节性**: TQI有明显的季节性模式，大模型需适配
4. **计算资源**: Lag-Llama和Time-LLM需要GPU资源

---

## 五、参考文献

1. Nedevska Trajkova, I., & Zafirovski, Z. (2025). Predictive modeling of track quality index with neural networks. *Journal of Applied Engineering Science*, 23(1308), 735-741.

2. 王志鸥, 刘仍奎, 邱荣华, 韩蕊. (2020). 基于机器学习的地铁轨道几何劣化规律个性化建模研究. *都市快轨交通*, 33(4).

3. Rasul, K., et al. (2024). Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting. *arXiv preprint*.

4. Jin, M., et al. (2024). Time-LLM: Time Series Forecasting by Reprogramming Large Language Models. *ICLR 2024*.

5. Liu, Y., et al. (2024). Timer: Transformers for Time Series Analysis at Scale. *arXiv preprint*.

6. Wang, S., et al. (2024). TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. *ICLR 2024*.

7. Wang, S., et al. (2025). TimeMixer++: General Multiscale Pattern Modeling for Time Series Analysis. *ICLR 2025*.

---

**备注**: 本调研结果可指导后续14年TQI数据的深度学习/大模型实验设计。建议优先尝试Lag-Llama零样本预测作为强基线，再与多尺度融合模型对比。
