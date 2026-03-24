# Trident - 基于业务经验增强的多尺度TQI预测研究

**项目全称**: 基于业务经验增强的多尺度轨道质量指数预测方法  
**项目代号**: Trident  
**研究方向**: 轨道质量预测 / 智能铁路维护  
**创建时间**: 2026-03-21

---

## 项目概述

本项目提出Trident框架，一种业务经验增强的多尺度轨道质量指数(TQI)预测方法。通过显式编码大修期锚定值、季节性规律和劣化趋势，解决传统数据驱动方法难以处理维修干预后预测的问题。

### 核心创新

1. **平面/高程分量分组策略** - 针对不同劣化机制差异化建模
2. **大修期锚定机制** - 利用维修后实测值作为预测基准
3. **修后预测策略** - 针对"已知大修时间点，预测下次大修前变化"的实际维护场景优化

### 主要成果

- 在514个有效样本上验证，平均MAE改善30.5%
- 最佳样本MAE改善达83.8%
- 51.8%的样本MAE<0.3，预测精度满足工程要求
- 仅需数百条记录即可有效工作，数据需求低、可解释性强

---

## 目录结构

```
2026-03-21-Trident-TQI预测研究/
├── 00-开题与立项/              # 项目立项与规划文档
│   └── TQI论文规划_20260306.md
│
├── 01-文献综述/                # 相关研究调研
│   ├── TQI预测研究现状调研报告_20260321.md
│   ├── TQI预测与养护维修决策研究现状综述.md
│   ├── 深度学习与大模型在TQI预测中的研究现状.md
│   └── 开源代码与数据集资源清单.md
│
├── 02-方法论与框架设计/        # 方法论与架构设计
│   ├── 深度多尺度时序大模型架构设计.md
│   └── (其他框架设计文档)
│
├── 03-实验与实现/              # 实验代码与数据
│   ├── src/                    # 源代码
│   │   ├── data_loader.py      # 数据加载
│   │   ├── experiment_*.py     # 实验A-F代码
│   │   ├── *_experiment.py     # 批量实验脚本
│   │   ├── baseline_comparison.py
│   │   ├── ablation_study.py
│   │   └── generate_figures.py
│   │
│   ├── data/                   # 实验数据
│   │   ├── raw/                # 原始数据(iic_tqi_all.xlsx等)
│   │   ├── processed/          # 处理后数据
│   │   ├── samples/            # 样本数据(2-6号样本)
│   │   └── experiment_results/ # 实验结果CSV
│   │
│   ├── results/                # 实验结果
│   │   ├── experiment_reports/ # 实验报告
│   │   ├── batch_experiments/  # 批量实验结果(514样本)
│   │   ├── full_experiments_514/
│   │   ├── full_experiments_top100/
│   │   └── figures/            # 图表
│   │
│   ├── scripts/                # 辅助脚本与配置
│   └── figures/                # 生成的图表
│
├── 04-数据分析/                # 数据分析相关
│
├── 05-论文撰写/                # 论文与投稿材料
│   ├── manuscript/             # 论文主文件
│   │   ├── manuscript.md       # 论文正文
│   │   ├── manuscript.docx     # Word版本
│   │   ├── cover_letter.md     # 投稿信
│   │   └── README.md
│   │
│   ├── figures/                # 论文图表
│   │   ├── fig1_timeseries.png
│   │   ├── fig2_baseline.png
│   │   ├── fig3_ablation.png
│   │   └── fig4_prediction.png
│   │
│   └── submissions/            # 投稿相关材料
│
├── 06-参考文献/                # 参考文献
│
└── README.md                   # 本文件
```

---

## 核心实验

### 实验A: 时序分解 (STL)
分析TQI的年度趋势、季节性周期和残差分量。

### 实验B: 冬季效应
研究冬季低温对轨道平顺性的影响，识别冻融循环导致的TQI波动。

### 实验C: 大修期识别
自动检测大修时间点(TQI单月下降>0.3的局部极值点)。

### 实验D: 基线对比
对比移动平均、指数平滑、历史均值、LSTM、TimeMixer等方法。

### 实验E: 分量分组
平面位置组 vs 高程平顺组的差异化建模。

### 实验F: 业务经验增强 (核心)
- **数据驱动**: 历史均值基线
- **滚动锚定**: 上一年大修后值+季节性调整
- **修后预测**: 本次大修后→下次大修前

### 批量实验 (514样本)
对iic_tqi_all.xlsx中的1,203个样本进行批量测试，识别514个Trident有效样本。

---

## 关键结果

| 实验 | 样本数 | 最佳MAE | 改善幅度 |
|:---|:---:|:---:|:---:|
| 论文实验-5号 | 250条 | 0.087 | +4% |
| 论文实验-3号 | 477条 | 0.294 | +54% |
| 批量实验 | 514个 | 0.047(Top) | 平均30.5% |

---

## 快速开始

### 环境要求
- Python 3.8+
- pandas, numpy, matplotlib, scikit-learn
- (可选) PyTorch (for LSTM/TimeMixer对比)

### 运行实验

```bash
cd 03-实验与实现

# 运行单个实验
python src/experiment_f_business_aware.py

# 运行批量实验
python src/batch_experiment.py

# 生成图表
python src/generate_figures.py
```

### 数据准备

原始数据放在 `data/raw/`:
- `iic_tqi_all.xlsx` - 全量TQI数据(435,753条)
- `3号样本.xlsx`, `5号样本.xlsx` 等 - 论文实验样本

---

## 主要文件说明

### 论文相关
- `05-论文撰写/manuscript/manuscript.md` - 论文正文(Markdown)
- `05-论文撰写/manuscript/manuscript.docx` - 论文正文(Word)
- `05-论文撰写/manuscript/cover_letter.md` - 投稿信

### 实验报告
- `03-实验与实现/results/experiment_reports/` - 各实验详细报告
- `03-实验与实现/results/batch_experiments/Trident_514样本实验报告.md` - 批量实验报告

### 核心代码
- `src/experiment_f_business_aware.py` - 业务经验增强实验(核心)
- `src/batch_experiment.py` - 批量实验脚本
- `src/baseline_comparison.py` - 基线方法对比

---

## 贡献者

- 项目负责人: [待填写]
- 技术实现: Kimi Claw (AI助手)
- 数据支持: XX铁路局工务段

---

## 更新日志

- **2026-03-21** - 项目启动，文献调研
- **2026-03-22** - 实验A-F完成，论文初稿
- **2026-03-23** - 批量实验(514样本)，文件整理

---

## 许可

[待填写]
