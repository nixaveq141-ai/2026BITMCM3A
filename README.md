# 2026 世界杯区域转播权定价模型

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

> **2026 年北京理工大学数学建模竞赛（第三级）A 题**
>
> 基于 AHP-熵权组合赋权、分层 TOPSIS、灰色关联度与动态修正因子的世界杯区域转播权定价模型。

---

## 📋 目录

- [赛题背景](#赛题背景)
- [项目结构](#项目结构)
- [建模方法](#建模方法)
- [核心结论](#核心结论)
- [环境与运行](#环境与运行)
- [数据说明](#数据说明)
- [参考资料](#参考资料)

---

## 赛题背景

2026 年美加墨世界杯于 6 月 11 日开幕，决赛于 7 月 19 日举行，三国 16 个城市共同承办 104 场比赛（比上届 64 场增加 62.5%）。FIFA 以赛事扩军、内容增量为由，提高了全球转播权报价。然而由于比赛时间对亚洲地区（凌晨/午夜）极不友好，多个市场与 FIFA 长期未能达成协议——其中中国内地尤为突出：FIFA 初始报价超过 2.5 亿美元，最终以约 6000 万美元成交。

赛题要求建立数学模型解决以下四个问题：

| 小问 | 任务 | 核心目标 |
|:---:|------|----------|
| **1** | 选择一个已成交国家/地区，分析成交价估算方法 | 以日本为对象，建立复盘解释模型 |
| **2** | 作为 FIFA 定价负责人，给出通用定价模型 | 构建可推广的多层市场定价框架 |
| **3** | 计算 2026 年中国内地合理报价，解释与 FIFA 报价、最终成交价的差异 | 中国内地报价 + 四阶段谈判差异分解 |
| **4** | 给 FIFA 写一份不超过 2 页的定价策略建议书 | 将模型结论转化为可执行商业策略 |

---

## 项目结构

```
2026BITMCM3A/
├── README.md                          # 项目说明
├── competition_questions.txt          # 赛题原文
├── overall_analysis.md                # 赛题精读与整体分析
│
├── datas/                             # 数据集（7 个 CSV）
│   ├── README.md                      # 数据集详细说明文档
│   ├── 01_broadcasting_rights_2026.csv
│   ├── 02_historical_broadcasting_rights.csv
│   ├── 03_economic_indicators.csv
│   ├── 04_football_market_indicators.csv
│   ├── 05_worldcup_viewership.csv
│   ├── 06_timezone_viewer_convenience.csv
│   └── 07_market_structure.csv
│
├── original_documents/                # 竞赛原始文档
│   ├── 0-2026校赛第三轮赛题.docx
│   ├── 1-2026年北京理工大学数学建模竞赛论文规范.docx
│   ├── 2-2026年北京理工大学数学建模竞赛论文模板.docx
│   └── 校赛重要提示.txt
│
├── q1/                                # 第一问：日本成交价复盘
│   ├── overall_model_q1.md            # 模型说明文档
│   ├── analysis_q1.md                 # 分析过程
│   ├── run_q1_analysis.py             # 主分析脚本
│   └── outputs/                       # 输出结果（CSV/PNG/MD）
│
├── q2/                                # 第二问：通用定价模型
│   ├── overall_model_layered_market.md # 模型说明文档
│   ├── analysis_q2_layered_market.md  # 分析过程
│   ├── run_q2_analysis_layered_market.py  # 主分析脚本
│   └── outputs_layered_market/        # 输出结果（CSV/PNG/MD）
│
├── q3/                                # 第三问：中国内地报价
│   ├── overall_model_q3.md            # 模型说明文档
│   ├── analysis_q3.md                 # 分析过程
│   ├── run_q3_analysis.py             # 主分析脚本
│   └── outputs/                       # 输出结果（CSV/PNG/MD）
│
└── q4/                                # 第四问：FIFA 建议书
    └── recommendations_for_reference.md
```

---

## 建模方法

### 整体技术路线

```text
数据准备 → 市场分层 → 结构 TOPSIS 综合评估 → 灰色关联度选取参考市场
       → 体量平滑校准 → 动态修正（竞购/时区/权利包） → 层级溢价 → 最终报价区间
```

### 各问模型概览

#### 第一问：日本成交价复盘 — "分层动态参考市场成交价复盘模型"

- 选择 **日本** 作为研究对象（亚洲发达市场 + 球队参赛 + 数据完整）
- **AHP-熵权法组合赋权**：主观专家判断与客观数据变异度相结合
- **TOPSIS 综合评估**：计算各市场的结构和市场价值指数
- **灰色关联度分析**：筛选与日本最相似的参考市场
- **五因子动态修正**：剥离/回乘动态因子（扩军溢价、竞购溢价、时区折扣、权利包折扣、市场层级因子）
- 模型复盘值与日本实际成交价（约 2 亿美元）高度吻合

#### 第二问：通用定价模型 — "市场分层—结构 TOPSIS—邻层灰色关联—动态修正报价模型"

在第一问基础上进行三项关键改进：

1. **市场分层机制**：按综合市场价值指数将市场分为 T1（超大规模）→ T4（新兴市场）四层，限制跨层参考
2. **分层参数寻优**：每层独立优化 α（弹性系数）和 k（体量平滑参数），通过网格搜索 + 留一交叉验证确定最优参数
3. **层级溢价引入**：T3/T4 层市场相对 T1/T2 层添加基础溢价，反映 FIFA 对不同层级市场的最低价格预期

留一交叉验证平均相对误差控制在 **15% 以内**。

#### 第三问：中国内地报价 — "参考市场迁移 + 中国市场修正 + 谈判差异分解"

- 继承第二问参数，对中国市场进行四项专属修正：
  - **结构价值弹性下调**：买方集中度高（央视垄断），压低议价能力
  - **层级溢价下调**：中国作为 T2 层但具有买方市场特征
  - **竞购系数修正**：有效竞购方少，缺乏竞价压力
  - **权利结构系数修正**：免费播出法律要求，限制广告变现
- 给出三种报价方案：保守 $45M / 基准 $62M / 进取 $85M
- **四阶段谈判差异分解**：将 FIFA 初始报价 → 最终成交价的过程分解为 7 个分量

#### 第四问：FIFA 策略建议书

从模型结论出发，为 FIFA 提出以下建议：
- 分市场层级采用差异化定价策略
- 引入动态定价机制（赛事时间友好度折扣）
- 打包销售与单届灵活组合
- 提前建立长期合作关系，降低谈判成本

---

## 核心结论

### 中国内地 2026 年世界杯转播权定价

| 指标 | 数值 |
|------|:----:|
| 模型基准合理报价 | **~$62M** |
| 保守报价下限 | ~$45M |
| 进取报价上限 | ~$85M |
| FIFA 初始报价 | $250M–300M |
| 最终成交价 | ~$60M |
| 模型与最终成交价偏差 | **+3.3%** |

### 中国谈判价格差异分解

| 差异来源 | 金额贡献 | 占比 |
|----------|:--------:|:----:|
| 扩军溢价（FIFA 过度定价） | +$110M | 46% |
| 买方垄断压价（央视） | -$80M | 33% |
| 时区折扣 | -$45M | 19% |
| 心理锚定效应（2022 年高价参考） | +$30M | 13% |
| 其他因素（赞助商压力等） | -$25M | 10% |

---

## 环境与运行

### 依赖

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

### 运行

```bash
# 第一问：日本成交价复盘
python q1/run_q1_analysis.py

# 第二问：通用定价模型
python q2/run_q2_analysis_layered_market.py

# 第三问：中国内地报价
python q3/run_q3_analysis.py
```

各脚本会读取 `datas/` 中的数据文件，并将结果输出到对应 `outputs/` 目录。

### Python 版本

Python 3.10+

---

## 数据说明

本项目使用的 7 个数据集涵盖以下维度：

| 数据集 | 维度 | 关键字段 |
|--------|------|----------|
| `01_broadcasting_rights_2026.csv` | 2026 转播权成交价及报价 | 国家、转播商、价格、交易状态 |
| `02_historical_broadcasting_rights.csv` | 2010–2022 历史价格 | 国家、年份、价格 |
| `03_economic_indicators.csv` | 经济与数字化指标 | GDP、人口、互联网渗透率 |
| `04_football_market_indicators.csv` | 足球市场指标 | FIFA 排名、球迷比、联赛水平 |
| `05_worldcup_viewership.csv` | 收视数据 | 决赛观众、小组赛观众 |
| `06_timezone_viewer_convenience.csv` | 时区观赛便利度 | 时差、友好度评分 |
| `07_market_structure.csv` | 市场结构 | 竞购方数量、媒体市场成熟度 |

所有数据均来自公开来源（World Bank、FIFA、Statista、Nielsen、各新闻媒体报道），已在 `datas/README.md` 中逐项标注出处。

---

## 参考资料

- FIFA World Cup 2022 Commercial Report — [fifa.com](https://quality.fifa.com/tournament-organisation/world-cup-2022-in-numbers/fifa-world-cup-qatar-2022-commercial)
- World Bank Open Data — [worldbank.org](https://data.worldbank.org/)
- SportBusiness Global Media Report 2024–2025
- Nielsen 2025 Global Sports Report
- 央视新闻、澎湃新闻、东方财富、搜狐等公开报道（详见 `datas/README.md`）

---

## 许可

本项目仅用于学术竞赛与学习交流目的。数据版权归原始来源所有。
