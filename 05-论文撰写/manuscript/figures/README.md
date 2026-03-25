# 论文图表清单

## 图表概览

本文档列出论文中的所有图表及其说明。

---

## 图1: 全量样本方法性能对比
**文件名**: `fig1_full_dataset_comparison.png`

**内容**: 486个样本上6种方法的MAE对比
- Trident_v23 (no seasonal): 0.868 (最优，红色高亮)
- Historical Mean: 0.908
- Trident_v23 (full): 0.893
- Trident_v21: 0.950
- Holt-Winters: 0.952
- Moving Average: 1.142

**放置位置**: 第4节 (Results) - 4.1 Full Dataset Results

---

## 图2: 72优质样本方法性能对比
**文件名**: `fig2_72samples_comparison.png`

**内容**: 72个高优质样本上6种方法的MAE对比，带显著性标记
- Trident_v21: 0.376 (最优，显著性*** p<0.001)
- Trident_v23 (full): 0.432
- Trident_v23 (no seasonal): 0.448
- Historical Mean: 0.518
- Holt-Winters: 0.518
- Moving Average: 0.590

**放置位置**: 第4节 (Results) - 4.2 High-Quality Subset Results

---

## 图3: 自适应方法选择策略
**文件名**: `fig3_adaptive_strategy.png`

**内容**: 方法选择流程图
1. 输入新样本
2. 计算TQI波动率（训练集标准差）
3. 判断: σ < 0.4?
   - Yes → 使用Trident_v21 (MAE: 0.376)
   - No → 使用Trident_v23 (MAE: 0.868)
4. 输出最优预测

**关键洞察**: 简单模型用于稳定数据，复杂模型用于波动数据

**放置位置**: 第4节 (Results) - 4.4 Adaptive Strategy Performance

---

## 图4: 不同波动区间的性能对比
**文件名**: `fig4_volatility_comparison.png`

**内容**: v21和v23在不同波动区间的MAE对比
- All Samples: v21=0.95, v23=0.87
- Low Volatility (σ<0.4): v21=0.38, v23=0.45 (v21更好)
- Medium (0.4≤σ<0.6): v21=0.65, v23=0.62 (v23更好)
- High (σ≥0.6): v21=1.25, v23=1.15 (v23更好)

**放置位置**: 第4节 (Results) - 4.3 Why Does Simpler Work Better?

---

## 图5: v21 vs v23 胜率分布（72样本）
**文件名**: `fig5_win_rate_distribution.png`

**内容**: 饼图展示v21和v23在72样本上的直接对比
- v21 Wins: 48.6% (35样本)
- Ties: 41.7% (30样本)
- v23 Wins: 9.7% (7样本)

**放置位置**: 第4节 (Results) - 4.2 High-Quality Subset Results

---

## 表格清单

### 表1: 全量样本性能对比 (486 samples)
**位置**: 第4.1节
**列**: Method, MAE, RMSE, MAPE, Win Rate

### 表2: 72优质样本性能对比
**位置**: 第4.2节
**列**: Method, MAE, RMSE, MAPE, Win Rate

### 表3: 统计显著性检验
**位置**: 第4.2节
**列**: Method, Mean Diff, t-statistic, p-value, Significant

### 表4: 自适应策略性能
**位置**: 第4.4节
**列**: Strategy, MAE (Low σ), MAE (High σ), Overall MAE

### 附录表B1: 486样本完整结果
**位置**: Appendix B
**说明**: 所有486个样本的详细MAE数据

### 附录表B2: 72样本完整结果
**位置**: Appendix B
**说明**: 所有72个优质样本的详细MAE数据

---

## 图表使用建议

1. **彩色印刷**: 图1-5使用不同颜色区分方法，建议使用彩色印刷
2. **分辨率**: 所有图表以300 DPI保存，适合论文印刷
3. **尺寸**: 建议单栏宽度8cm或双栏宽度17cm
4. **格式**: PNG格式，如需矢量图可重新生成SVG版本

---

## 文件位置

所有图表位于:
```
05-论文撰写/manuscript/figures/
├── fig1_full_dataset_comparison.png
├── fig2_72samples_comparison.png
├── fig3_adaptive_strategy.png
├── fig4_volatility_comparison.png
└── fig5_win_rate_distribution.png
```

生成代码位于:
```
03-实验与实现/src/generate_paper_figures.py
```
