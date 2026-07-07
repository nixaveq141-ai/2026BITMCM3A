# 2026 世界杯区域转播权定价 — 数据集说明文档

## 概述

本文件夹包含 2026 年美加墨世界杯区域转播权定价数学建模所需的全部数据集，共 7 个 CSV 文件，涵盖广播权价格、经济指标、足球市场、收视数据、时区便利度和市场结构等维度。每个数据集均附有明确的数据来源。

---

## 数据文件清单

| 编号 | 文件名 | 内容说明 | 关键字段 |
|------|--------|----------|----------|
| 01 | `01_broadcasting_rights_2026.csv` | 2026 世界杯各国/地区转播权成交价及FIFA报价 | 国家、转播商、价格(百万美元)、交易状态、协议年份 |
| 02 | `02_historical_broadcasting_rights.csv` | 2010-2022 历届世界杯转播权历史价格 | 国家、世界杯年份、价格(百万美元)、转播商 |
| 03 | `03_economic_indicators.csv` | 各国/地区经济与数字化指标 | GDP人均、人口、互联网渗透率、体育媒体市场规模 |
| 04 | `04_football_market_indicators.csv` | 各国/地区足球市场指标 | FIFA排名、世界杯资格、球迷比例、球迷数量、联赛水平 |
| 05 | `05_worldcup_viewership.csv` | 2018/2022 世界杯各国收视数据 | 决赛观众、小组赛观众、累计触达、收视份额 |
| 06 | `06_timezone_viewer_convenience.csv` | 2026世界杯各时区观赛便利度 | 时区、时差、典型比赛时间、友好度评分 |
| 07 | `07_market_structure.csv` | 各国/地区市场规模与竞争结构 | GDP总量、媒体市场成熟度、竞购方数量、世界杯参赛历史 |

---

## 数据集详细说明

### 01. 2026 世界杯转播权价格 (`01_broadcasting_rights_2026.csv`)

**为什么需要这个数据？**
这是建模的**核心目标变量（因变量）**。已成交的价格是训练定价模型的标签，未成交/FIFA报价数据可用于验证模型预测的合理性。价格从美国的 $945M 到越南的 $15M 不等，跨度达 63 倍，反映了市场规模、竞争程度、时区便利度和足球文化差异的多重影响。

**数据来源：**
- 网易 (163.com)：《世界杯版权费曝光！美国9.45亿第一》[链接](https://m.163.com/dy/article/KT03QOFQ0536NZ17.html)
- Yahoo Sports: "How much FIFA are making from 2026 World Cup broadcasting deals" [链接](https://sports.yahoo.com/articles/much-fifa-making-2026-world-160000920.html)
- 东方财富：《世界杯转播权仍未谈拢，国际足联高管计划访华》[链接](https://finance.eastmoney.com/a/202605073729917685.html)
- 澎湃新闻：《世界杯，中国如何说不？》[链接](https://m.thepaper.cn/newsDetail_forward_33119810)
- 央视新闻/中国日报：《CCTV secures 2026 FIFA World Cup broadcasting rights》[链接](https://www.chinadaily.com.cn/a/202605/15/WS6a06dfeaa310d6866eb48e1e.html)
- NDTV Sports: "FIFA Strikes CCTV Deal to End China's 2026 World Cup Blackout" [链接](https://sports.ndtv.com/us/fifa/fifa-strikes-cctv-deal-to-end-chinas-2026-world-cup-blackout-11502116)
- 搜狐：《14亿人口大国竟把2026美加墨世界杯转播费砍到200万美金》[链接](https://www.sohu.com/a/1018621620_121144486)
- 香港01：《央视硬刚国际足联》[链接](https://www.hk01.com/即時中國/60350531)

**特殊说明：**
- 中国最终成交价存在多个来源口径：$60M（2026单届）或 $110M（含2030男足+女足世界杯）
- 中国央视还价区间：$60M-80M
- FIFA对中国初始报价为 $250M-300M，后降至 $120M-150M
- 印度报价存在争议：部分来源称最终以极低价（约$2M）成交，待官方确认

---

### 02. 历届世界杯转播权历史价格 (`02_historical_broadcasting_rights.csv`)

**为什么需要这个数据？**
历史价格是定价模型的重要**基准（baseline）**。转播权定价具有明显的**锚定效应**——FIFA和新买家都会参考上一届的价格进行谈判。通过比较2022→2026的价格涨幅，可以识别FIFA的提价策略和各国的议价能力。例如，中国2022年支付$150M/届，FIFA对2026年初始报价$250M+，涨幅约67%；而美国Fox从$212.5M涨至$480M，涨幅达126%，反映东道主溢价效应。

**数据来源：**
- CBC Sports: "Fox wins World Cup TV rights for U.S." [链接](https://www.cbc.ca/sports/soccer/fox-wins-world-cup-tv-rights-for-u-s-report-1.1021929)
- Seattle Times: "Fox, Telemundo get U.S. television rights for 2018, 2022 World Cups" [链接](https://www.seattletimes.com/sports/other-sports/fox-telemundo-get-us-television-rights-for-2018-2022-world-cups-soccer/)
- 腾讯新闻：《央视拿下两届世界杯转播权，今年版权费据悉为6000万美元》[链接](https://news.qq.com/rain/a/20260515Q09NP400)
- NHK Broadcasting Studies: 日本世界杯转播权分析 [链接](https://www.nhk.or.jp/bunken/d/_data/en/book/studies/articles/BUNA0000130000020008/files/03_no2_08.pdf)
- FIFA Official: 2022 World Cup Commercial Report [链接](https://quality.fifa.com/tournament-organisation/world-cup-2022-in-numbers/fifa-world-cup-qatar-2022-commercial)

**特殊说明：**
- 部分国家（如越南、马来西亚）的历史价格来自行业估算而非官方披露
- 打包交易（如2018+2022）的价格已按届均分配
- 巴西的2010/2014数据以欧元计价，已标注

---

### 03. 经济与数字化指标 (`03_economic_indicators.csv`)

**为什么需要这个数据？**
GDP人均是衡量一国**支付能力**的核心指标——高GDP国家通常能承担更高的转播权费用。人口规模决定了**潜在观众基数**的上限。互联网渗透率影响**流媒体观看便利度**和数字广告变现能力。体育媒体市场规模反映该国体育产业的整体**商业化程度**。

在建模中，这些指标是**关键自变量（特征）**：
- `gdp_per_capita_usd_2024`: 代表购买力
- `population_million_2024`: 代表潜在市场规模
- `internet_penetration_pct_2024`: 代表流媒体基础设施
- `sports_media_market_billion_usd_est`: 代表体育媒体商业生态

**数据来源：**
- WIFO/OECD Economic Outlook (Spring 2025): GDP per capita projections [链接](https://www.wifo.ac.at/wp-content/uploads/wtabellen/e/1.3_e_out.pdf)
- World Bank Open Data: GDP per capita, population [链接](https://data.worldbank.org/)
- CEIC Data: Country economic indicators [链接](https://www.ceicdata.com/en/indicator/gdp-per-capita)
- ITU World Telecommunication/ICT Indicators Database: Internet penetration [链接](https://www.itu.int/en/ITU-D/Statistics/Pages/stat/default.aspx)
- DataReportal Digital 2025 Reports: Internet users [链接](https://datareportal.com/)
- SportBusiness Global Media Report 2024-2025: Sports media market size [链接](https://www.sportbusiness.com/global-media-report-2025/)
- MarketIntelo: Sports Media Rights Market Research Report 2034 [链接](https://marketintelo.com/report/sports-media-rights-market)

---

### 04. 足球市场指标 (`04_football_market_indicators.csv`)

**为什么需要这个数据？**
这些是衡量一国"**足球文化深度**"的核心变量。一个国家即使GDP不高，如果足球文化深厚（如巴西、阿根廷），其转播权价值也会显著高于同等GDP但足球文化淡薄的国家。具体来说：

- `fifa_ranking_june_2026`: 国家队实力越强，国民关注度越高→转播价值越高
- `qualified_2026_wc`: **是否有本国球队参赛是最重要的定价因子之一**。有国家队参赛的国家，收视率通常有50%-200%的提升
- `football_fan_pct_of_population`: 直接衡量足球在该国的受欢迎程度
- `football_fans_million`: 球迷绝对数量，对于人口大国（中国、印度）格外重要
- `domestic_league_strength_tier`: 反映该国足球产业的系统深度

**数据来源：**
- FIFA/Coca-Cola World Ranking (June 2026): [链接](https://www.fifa.com/fifa-world-ranking/men)
- USA Today: "All remaining 2026 World Cup teams ranked, according to FIFA" [链接](https://www.usatoday.com/story/sports/soccer/worldcup/2026/07/01/world-cup-fifa-rankings-remaining-teams/90765491007/)
- Statista Consumer Insights (Apr 2024–Mar 2025): Football following by country [链接](https://www.statista.com/)
- Nielsen 2025 Global Sports Report: "Football Dominates Global Sports Landscape with 51% Fandom" [链接](https://www.medianews4u.com/football-dominates-global-sports-landscape-with-51-fandom-nielsen-finds/)
- 俄罗斯卫星通讯社：《调查：中国43%的体育迷对足球感兴趣》[链接](https://cdn.sputniknews.cn/20250604/1065820427.html)
- The Daily Star: "Full Round of 32 fixtures and more (2026 WC)" [链接](https://www.thedailystar.net/sports/sports-special/fifa-world-cup-2026/news/full-round-32-fixtures-and-more-4210111)
- Sporting News: "World Cup teams to qualify for Round of 32" [链接](https://www.sportingnews.com/us/soccer/news/world-cup-2026-bracket-schedule-round-32/30ebc00f809cfaafa0098b48)

**特殊说明：**
- 中国香港、越南、马来西亚、泰国、印度均未获得2026世界杯参赛资格
- FIFA排名数据截止2026年6月世界杯开幕前最新一期
- 球迷比例来自不同机构的调查，口径可能存在差异

---

### 05. 世界杯收视数据 (`05_worldcup_viewership.csv`)

**为什么需要这个数据？**
收视数据是**转播权商业价值的直接体现**。转播商的广告收入与收视人数成正比，因此历史收视数据是预测未来转播权价值的关键输入。数据分2018和2022两届，可分析增长趋势和球队参赛的影响。

关键模式：
- **东道主效应**：美国2022年收视大幅增长（决赛+47%），部分原因是2026年将成为东道主
- **球队参赛效应**：德国2022年因小组出局，收视暴跌64%；日本因晋级16强，收视暴涨74%
- **人口基数效应**：巴西决赛36.9M观众（81%人口覆盖率）
- **中国市场特殊性**：2022年中国贡献全球49.8%的数字/社交媒体观看时长，但传统电视收视因时差而受限

**数据来源：**
- FIFA Official: "FIFA World Cup Qatar 2022 in Numbers — Commercial" [链接](https://quality.fifa.com/tournament-organisation/world-cup-2022-in-numbers/fifa-world-cup-qatar-2022-commercial)
- FIFA Inside: "Early figures suggest the FIFA World Cup is as popular as ever" [链接](https://inside.fifa.com/tournaments/mens/worldcup/qatar2022/news/early-figures-suggest-the-fifa-world-cup-is-as-popular-as-ever)
- SportsPro: "World Cup final scores record 29.4m TV audience in France" [链接](https://www.sportspro.com/news/fifa-world-cup-qatar-2022-final-tv-viewership-audience-ratings-tf1-bbc-fox/)
- Broadcast Now: "FIFA releases international World Cup viewing figures" [链接](https://www.broadcastnow.co.uk/broadcasting/fifa-releases-international-world-cup-viewing-figures/5177107.article)
- TV Technology: "FIFA World Cup 2022 Ratings on Telemundo and Peacock Up 72% vs 2018" [链接](https://www.tvtechnology.com/news/fifa-world-cup-2022-ratings-on-telemundo-and-peacock-up-72-vs-2018)
- Streaming Media Blog: "FIFA World Cup TV and Streaming Viewership Numbers" [链接](http://www.streamingmediablog.com/2022/12/worldcup-viewership.html)
- BroadcastPro ME: "beIN Sports records 5.4bn cumulative views during FIFA WC" [链接](https://www.broadcastprome.com/news/bein-sports-records-5-4bn-cumulative-views-during-fifa-wc/)

---

### 06. 时区观赛便利度 (`06_timezone_viewer_convenience.csv`)

**为什么需要这个数据？**
2026世界杯在北美举办，时区对亚洲观众**极其不友好**，这是影响亚洲市场价格的关键因素。中国、马来西亚、香港的比赛时间在**午夜至上午10点**之间，几乎没有黄金时段比赛。这直接导致：
- 广告收入大幅缩水（央视估计仅为正常黄金时段的12-15%）
- 观众总数下降
- 转播权支付意愿降低

`viewer_friendliness_score_1_to_5` 是一个综合评分：5=大部分比赛在黄金时间，1=几乎全部在深夜/凌晨。

**数据来源：**
- Yahoo Sports UK: "World Cup 2026 calendar — Download for China Standard Time" [链接](https://uk.sports.yahoo.com/news/world-cup-2026-calendar-download-103743128.html)
- LiveSoccerTV: "What Time Are the World Cup Games? Full Match Schedule & Kickoff Times" [链接](https://www.livesoccertv.com/tr/news/537918/what-time-are-the-world-cup-games-full-match-schedule-kickoff-times/)
- Middle East Eye: "World Cup 2026: Match dates, kick-off times and how to watch" [链接](https://www.middleeasteye.net/discover/world-cup-2026-match-dates-kick-times-and-how-watch)
- 东方财富：《2026世界杯转播权陷天价僵局》——含时差分析 [链接](https://finance.eastmoney.com/a/202605073729183777.html)
- 新浪香港：《FIFA误判高估 有多少中国人会熬夜看世界杯？》[链接](https://portal.sina.com.hk/sports/sina/2026/05/08/1758999/fifa误判高估-有多少中国人会熬夜看世界杯？)

---

### 07. 市场结构 (`07_market_structure.csv`)

**为什么需要这个数据？**
市场结构影响**议价权力分配**：
- `competition_level` 和 `num_major_bidders`: 多家转播商竞购会推高价格（如美国、英国），垄断市场则会压低价格
- `media_broadcast_market_maturity`: 成熟市场的转播权估值更透明，新兴市场波动更大
- `world_cup_participation_frequency_last5`: 反映一国足球文化和国家队实力的长期趋势
- `historical_wc_viewing_growth_trend`: 反映市场增长潜力

**数据来源：**
- World Bank: Total GDP 2024 [链接](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD)
- SportBusiness Global Media Report 2024-2025 [链接](https://www.sportbusiness.com/global-media-report-2025/)
- 行业分析：各国世界杯参赛历史 [链接](https://www.fifa.com/)
- 各国广播市场分析（综合媒体报道）

---

## 数据使用建议

### 建模关系

```
转播权价格 = f(
    经济因素: GDP人均、GDP总量、体育媒体市场规模
    人口与数字化: 人口规模、互联网渗透率
    足球因素: FIFA排名、是否有球队参赛、球迷比例、球迷绝对数、联赛水平
    历史因素: 往届转播权价格、往届收视数据
    时区因素: 观赛友好度评分、黄金时段比例
    市场因素: 竞购方数量、媒体市场成熟度
    赛事因素: 是否东道主、世界杯参赛频率
)
```

### 建模注意事项

1. **多重共线性**：GDP与体育媒体市场规模高度相关；球迷比例与世界杯参赛频率相关。建议使用PCA降维或VIF检验。

2. **样本量有限**：已确认成交的国家/地区约16个，小样本下线性回归可能过拟合，建议使用正则化方法（Ridge/Lasso）或考虑贝叶斯方法。

3. **异质性**：发达国家与发展中国家遵循不同的定价逻辑，可考虑分层建模或引入交互项。

4. **时区折扣因子**：亚洲市场的"深夜折扣"是一个可量化的调节变量，建议在模型中以乘法交互形式引入。

5. **中国特殊性**：中国市场的"政策因素"（央视垄断、免费播出法律要求、赞助商压力）使得纯市场模型可能高估。建议在模型中加入"政府管制"哑变量。

---

## 数据更新记录

| 日期 | 更新内容 |
|------|----------|
| 2026-07-06 | 初始数据集创建，包含截至2026年7月的最新数据 |

---

## 附录：关键数据摘要

### 2026已成交国家/地区价格排名

| 排名 | 国家/地区 | 价格($M) | 人均GDP($) | 人口(M) | 世界杯参赛 |
|:----:|-----------|:--------:|:----------:|:-------:|:----------:|
| 1 | 美国(Fox) | 480 | 85,784 | 340 | ✅ (东道主) |
| 2 | 美国(Telemundo) | 465 | 85,784 | 340 | ✅ (东道主) |
| 3 | 英国 | 350/2届 | 52,580 | 69.1 | ✅ (英格兰) |
| 4 | 日本 | 200 | 32,513 | 124 | ✅ |
| 5 | 法国 | 150 | 46,293 | 66.5 | ✅ |
| 6 | 韩国 | 125 | 36,239 | 51.7 | ✅ |
| 7 | 德国(ARD/ZDF) | 120/2届 | 54,986 | 84.6 | ✅ |
| 8 | 西班牙(Mediapro) | 120 | 33,980 | 48.6 | ✅ |
| 9 | 意大利 | 120 | 38,574 | 58.9 | ❌ |
| 10 | 巴西 | 110 | 10,280 | 212 | ✅ |
| 11 | 德国(MagentaTV) | 80 | 54,986 | 84.6 | ✅ |
| 12 | 中国(最终) | 60 | 13,303 | 1,419 | ❌ |
| 13 | 香港 | 25 | 54,082 | 7.5 | ❌ |
| 14 | 澳大利亚 | 15 | 65,890 | 27.0 | ✅ |
| 15 | 越南 | 15 | 4,700 | 101 | ❌ |

### 中国谈判关键数据

| 谈判阶段 | 金额($M) | 日期 |
|----------|:--------:|------|
| FIFA初始报价 | 250-300 | 2025年初 |
| FIFA第一次降价 | ~150 | 2025年底 |
| FIFA第二次降价 | 120-150 | 2026年初 |
| FIFA秘书长访华后报价 | ~120 | 2026年5月 |
| 央视还价区间 | 60-80 | 全程 |
| **最终成交价** | **~60** | **2026年5月15日** |
| 对比2022年价格 | 150/届 | ↓60% |
| 对比FIFA初始报价 | 250-300 | ↓75-80% |
