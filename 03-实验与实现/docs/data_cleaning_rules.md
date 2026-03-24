# TQI数据清洗规则文档

**文档版本**: v2.0  
**创建时间**: 2026-03-24  
**适用数据**: iic_tqi_all.xlsx (原始轨检数据)

---

## 一、数据说明

### 原始数据字段
| 字段名 | 说明 | 类型 |
|:---|:---|:---|
| dete_dt | 检测日期 | datetime |
| tqi_mile | 里程 (轨道位置标识) | numeric |
| tqi_val | TQI总值 (轨道质量指数) | numeric |
| tqi_lprf | 左高低 | numeric |
| tqi_rprf | 右高低 | numeric |
| tqi_laln | 左轨向 | numeric |
| tqi_raln | 右轨向 | numeric |
| tqi_gage | 轨距 | numeric |
| tqi_warp1 | 三角坑 | numeric |
| tqi_xlvl | 水平 | numeric |

### 原始数据统计
- **总记录数**: 435,753 条
- **时间跨度**: 2012-12-12 至 2026-03-21
- **唯一里程数**: 1,245 个

---

## 二、清洗流程 (8步)

### Step 1: 数据加载与列名标准化

**操作**:
```python
df = pd.read_excel('data/raw/iic_tqi_all.xlsx')
df.columns = df.columns.str.strip()  # 去除列名空格
```

**说明**: 
- 原始Excel列名包含多余空格，需先清理
- 保留全部10个字段

**结果**: 435,753 条记录

---

### Step 2: 数据类型转换与基础清洗

**操作**:
```python
df['tqi_mile'] = pd.to_numeric(df['tqi_mile'], errors='coerce')
df['tqi_val'] = pd.to_numeric(df['tqi_val'], errors='coerce')
df['dete_dt'] = pd.to_datetime(df['dete_dt'], errors='coerce')
df = df.dropna(subset=['tqi_mile', 'tqi_val', 'dete_dt'])
```

**规则**:
| 检查项 | 处理方式 |
|:---|:---|
| tqi_mile 非数字 | 转为NaN后删除 |
| tqi_val 非数字 | 转为NaN后删除 |
| dete_dt 无效日期 | 转为NaT后删除 |
| 关键字段缺失 | 删除整行 |

**结果**: 423,063 条记录 (-12,690)

---

### Step 3: 重复数据去重 (同一里程+日期)

**问题发现**:
- 完全重复行: 12,006 条
- 同一里程+日期重复: 12,714 条
- 受影响的(里程,日期)组合: 4,114 个

**操作**:
```python
df = df.groupby(['tqi_mile', 'dete_dt'])['tqi_val'].mean().reset_index()
```

**规则**:
- 按 `tqi_mile` + `dete_dt` 分组
- 同一位置同一天的多次检测取 **TQI平均值**
- 保留第一条记录也是可选方案，但平均值更稳健

**原因**:
- 同一位置同一天的多次检测视为重复测量
- 取平均可消除测量噪声
- 避免数据泄露和过拟合

**结果**: 410,349 条记录 (-12,714)

---

### Step 4: 月度频率限制 (每月≤4条)

**操作**:
```python
df['year_month'] = df['dete_dt'].dt.to_period('M')
df['month_rank'] = df.groupby(['tqi_mile', 'year_month'])['dete_dt'].rank(method='first')
df = df[df['month_rank'] <= 4].copy()
```

**规则**:
- 提取年月信息 (year_month)
- 同一里程在同一个月内最多保留 **4条记录**
- 保留规则: 按日期排序，取前4条（最早的4次检测）

**原因**:
- 防止某个月份数据过密影响季节性建模
- 保持月度采样的均匀性
- 4次/月的频率足够捕捉月内变化

**结果**: 384,779 条记录 (-25,570)

---

### Step 5: 按里程分组统计

**操作**:
```python
mile_stats = df.groupby('tqi_mile').agg({
    'tqi_val': ['count', 'mean', 'std', 'min', 'max'],
    'dete_dt': ['min', 'max'],
    'year_month': 'nunique'
}).reset_index()
```

**计算指标**:
| 指标 | 说明 |
|:---|:---|
| record_count | 记录数 |
| tqi_mean | TQI平均值 |
| tqi_std | TQI标准差 |
| tqi_min/tqi_max | TQI极值 |
| date_min/date_max | 时间范围 |
| time_span_days | 时间跨度(天) |
| unique_months | 有数据的月份数 |
| expected_months | 理论月份数 |
| monthly_coverage | 月度覆盖度 |

**结果**: 1,245 个里程样本待筛选

---

### Step 6: 记录数筛选 (≥400条)

**规则**:
```python
qualified = mile_stats[mile_stats['record_count'] >= 400]
```

**原因**:
- 保证训练集/验证集/测试集的合理划分
- 400条记录可支持时间序列交叉验证
- 深度学习方法(LSTM等)需要足够样本

**阈值设定依据**:
- 按时间划分: 训练(70%) / 验证(15%) / 测试(15%)
- 测试集至少60条记录才能保证统计显著性
- 400 × 15% = 60

---

### Step 7: TQI范围筛选 (≤6)

**规则**:
```python
qualified = mile_stats[mile_stats['tqi_mean'] <= 6]
```

**原因**:
- TQI>6表示轨道质量很差的路段
- 这类路段通常处于维修期或故障期，规律性差
- 过滤极端异常值，保证数据质量

---

### Step 8: 时间跨度筛选 (≥10年)

**规则**:
```python
qualified = mile_stats[mile_stats['time_span_days'] >= 3650]
```

**原因**:
- 保证至少覆盖10个季节性周期
- 捕捉长期劣化趋势
- 经历多次维修周期，规律更丰富

---

### Step 9: 月度连续性检查 (覆盖度≥95%)

**操作**:
```python
mile_stats['expected_months'] = ((mile_stats['date_max'] - mile_stats['date_min']).dt.days / 30).astype(int) + 1
mile_stats['monthly_coverage'] = mile_stats['unique_months'] / mile_stats['expected_months']
qualified = mile_stats[mile_stats['monthly_coverage'] >= 0.95]
```

**规则**:
- 计算理论月份数: `(天数/30) + 1`
- 计算实际有数据的月份数
- 月度覆盖度 = 实际月份数 / 理论月份数
- 要求覆盖度 **≥ 95%**

**原因**:
- 保证时间序列的连续性
- 避免大块缺失影响趋势建模
- 95%允许少量月份缺失（如检测设备故障）

---

## 三、清洗结果汇总

### 各阶段数据量变化

| 步骤 | 操作 | 记录数/样本数 | 变化 |
|:---|:---|:---:|:---:|
| 0 | 原始数据 | 435,753 条 | - |
| 1 | 列名标准化 | 435,753 条 | - |
| 2 | 数据类型转换 | 423,063 条 | -12,690 |
| 3 | 同日去重 | 410,349 条 | -12,714 |
| 4 | 月度频率限制 | 384,779 条 | -25,570 |
| 5 | 按里程分组 | 1,245 个样本 | - |
| 6-9 | 综合筛选 | **491 个合格样本** | **-754** |

### 合格样本统计

| 指标 | 平均值 | 最小值 | 最大值 |
|:---|:---:|:---:|:---:|
| 样本数 | **491** | - | - |
| 记录数 | 448 条 | 427 条 | 454 条 |
| TQI均值 | 3.04 | 2.22 | 5.94 |
| 时间跨度 | 4845 天 (13.3年) | 4800 天 | 4845 天 |
| 月度覆盖度 | 97.20% | 95.06% | 98.15% |

---

## 四、输出文件

### 清洗后数据文件
| 文件名 | 说明 |
|:---|:---|
| `qualified_samples_v2.csv` | 合格样本完整统计信息 |
| `qualified_miles_v2.txt` | 合格里程列表 (491个) |

### CSV文件字段
- tqi_mile: 里程
- record_count: 记录数
- tqi_mean: TQI平均值
- tqi_std: TQI标准差
- tqi_min/tqi_max: TQI极值
- date_min/date_max: 时间范围
- time_span_days: 时间跨度(天)
- unique_months: 实际月份数
- expected_months: 理论月份数
- monthly_coverage: 月度覆盖度

---

## 五、注意事项

1. **数据敏感性**: TQI数据涉及铁路运营，注意保密
2. **重复检测**: 同一里程同一天的多次检测可能来自不同检测设备或复核检测
3. **月度限制**: 4次/月的限制基于实际检测频率，可调整
4. **覆盖度阈值**: 95%允许少量缺失，如需严格连续性可调至100%
5. **版本控制**: 清洗规则变更需更新此文档版本号

---

**文档维护**: 数据清洗规则如有调整，同步更新本文档
