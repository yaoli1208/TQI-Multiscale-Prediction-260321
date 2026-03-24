# T-ITS 投稿材料

**期刊：** IEEE Transactions on Intelligent Transportation Systems (T-ITS)  
**论文标题：** Trident: Business Knowledge-Enhanced Multi-Scale Prediction for Track Quality Index  
**投稿日期：** 2026-03-22

---

## 文件清单

| 文件 | 说明 |
|:---|:---|
| `manuscript.md` | 论文正文（约8000字） |
| `cover_letter.md` | 投稿信（Cover Letter） |
| `figures/` | 4张图表（PNG+PDF格式） |

---

## 论文核心贡献

1. **分量分组融合机制**：平面/高程分组，差异化建模
2. **业务经验锚定策略**：大修期锚定值+季节性调整+劣化趋势
3. **修后预测场景优化**：符合工务段实际维护workflow
4. **轻量级实用方案**：数百条数据即可工作，可解释性强

---

## 关键实验结果

| 样本 | 移动平均MAE | 指数平滑MAE | Trident MAE | 提升 |
|:---:|:---:|:---:|:---:|:---:|
| 5号（稳定型） | 0.091 | 0.134 | **0.087** | +4% |
| 3号（波动型） | 0.642 | 1.355 | **0.294** | **+78%** |

---

## 待办事项

- [ ] 填写作者信息（manuscript.md第3-4行）
- [ ] 填写通讯作者邮箱
- [ ] 准备推荐审稿人列表（3-5位）
- [ ] 创建GitHub代码仓库并获取DOI
- [ ] 英文翻译（如需国际版投稿）

---

## 相关链接

- T-ITS期刊主页：https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6979
- 投稿系统：https://mc.manuscriptcentral.com/t-its
