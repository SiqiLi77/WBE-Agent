# 02 系统架构

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        WBE-Agent System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │  异常检测模块  │───▶│           Agent 核心引擎              │  │
│  │  (Upstream)   │    │                                      │  │
│  │              │    │  ┌─────────┐   ┌─────────────────┐  │  │
│  │ - Z-score    │    │  │ System  │   │   ReAct Loop    │  │  │
│  │ - STL残差    │    │  │ Prompt  │──▶│                 │  │  │
│  │ - 变点检测   │    │  │(Domain  │   │ Thought→Action  │  │  │
│  │ - Ensemble   │    │  │Knowledge│   │ →Observation    │  │  │
│  └──────────────┘    │  └─────────┘   │ →Thought→...   │  │  │
│                      │                │                 │  │  │
│                      │                └────────┬────────┘  │  │
│                      │                         │           │  │
│                      │                    Tool Calls       │  │
│                      │                         │           │  │
│                      │  ┌──────────────────────▼────────┐  │  │
│                      │  │          工具集 (Tools)         │  │  │
│                      │  │                                │  │  │
│                      │  │ ┌────────┐ ┌────────┐        │  │  │
│                      │  │ │气象查询│ │水文查询│        │  │  │
│                      │  │ └────────┘ └────────┘        │  │  │
│                      │  │ ┌────────┐ ┌────────┐        │  │  │
│                      │  │ │临床查询│ │变异株  │        │  │  │
│                      │  │ └────────┘ │查询    │        │  │  │
│                      │  │ ┌────────┐ └────────┘        │  │  │
│                      │  │ │周边站点│ ┌────────┐        │  │  │
│                      │  │ │查询    │ │站点元数│        │  │  │
│                      │  │ └────────┘ │据查询  │        │  │  │
│                      │  │            └────────┘        │  │  │
│                      │  └────────────────────────────────┘  │  │
│                      └──────────────────────────────────────┘  │
│                                         │                      │
│                                         ▼                      │
│                      ┌──────────────────────────────────────┐  │
│                      │         结构化输出 (Report)           │  │
│                      │                                      │  │
│                      │ { classification, confidence,        │  │
│                      │   evidence, reasoning_chain }        │  │
│                      └──────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    多源对齐数据库                          │  │
│  │  NWSS污水 | HHS住院 | NOAA气象 | USGS水文 | CDC变异株    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Agent 核心设计

### 2.1 推理模式：ReAct (Reasoning + Acting)

选择 ReAct 而非 Plan-then-Execute 的原因：
- 调查路径依赖中间发现，无法预先规划完整计划
- 每一步的 thought 提供可解释的推理过程
- 自然支持早停（简单案例 2-3 步即可判定）

```
循环流程：
1. Thought: "站点X在时间T出现浓度骤升，先检查是否有降雨事件"
2. Action: query_weather(site_id, time_range)
3. Observation: "T-1日降雨量 45mm，超过稀释阈值"
4. Thought: "有显著降雨，但降雨通常导致信号下降而非上升。检查临床数据..."
5. Action: query_hospitalization(state, time_range)
6. Observation: "住院数在T+7日开始上升"
7. Thought: "临床数据跟随上升，说明这是真实疫情信号，降雨可能部分掩盖了真实幅度"
8. Action: output_report(...)
```

### 2.2 约束条件

| 参数 | 设定 | 理由 |
|------|------|------|
| 最大 tool call 次数 | 10 | 防止无限循环，控制成本 |
| LLM temperature | 0 | 确保可复现性 |
| 模型版本 | 锁定（如 gpt-4-turbo-2024-04-09） | 确保实验一致性 |
| 输出格式 | 结构化 JSON | 便于自动化评估 |

### 2.3 输出 Schema

```json
{
  "event_id": "string",
  "site_id": "string",
  "anomaly_date": "YYYY-MM-DD",
  "classification": "epidemic | environmental | sampling | mixed | uncertain",
  "confidence": 0.0-1.0,
  "primary_factors": [
    {
      "factor": "string (e.g., 'heavy_rainfall', 'variant_emergence')",
      "contribution": "high | medium | low",
      "evidence": "string (具体数据引用)"
    }
  ],
  "reasoning_chain": [
    {
      "step": 1,
      "thought": "string",
      "action": "string",
      "observation": "string"
    }
  ],
  "recommendation": "string (对公共卫生决策的建议)",
  "data_gaps": ["string (调查中发现的数据缺失)"],
  "tool_calls_count": "int",
  "total_tokens": "int"
}
```

## 3. 工具集设计

### 3.1 query_weather — 气象数据查询
```
输入: site_id, start_date, end_date
输出: {
  daily_precipitation_mm: [],
  daily_temp_max_c: [],
  daily_temp_min_c: [],
  daily_temp_mean_c: [],
  significant_events: [  // 预计算的显著事件标记
    {date, event_type: "heavy_rain"|"freeze"|"heatwave", severity}
  ]
}
数据源: NOAA GHCN-Daily（通过空间匹配的最近气象站）
```

### 3.2 query_hydrology — 水文数据查询
```
输入: site_id, start_date, end_date
输出: {
  daily_discharge_cfs: [],
  discharge_percentile: [],  // 相对于该站历史分布的百分位
  significant_events: [
    {date, event_type: "high_flow"|"low_flow"|"cso_risk", severity}
  ]
}
数据源: USGS NWIS（通过空间匹配的最近水文站）
备注: CSO 风险基于流量超过历史 90th percentile 的规则推断
```

### 3.3 query_hospitalization — 临床数据查询
```
输入: state, start_date, end_date
输出: {
  daily_admissions: [],
  7day_avg_admissions: [],
  admission_trend: "increasing"|"stable"|"decreasing",
  trend_change_dates: [],  // 趋势转折点
  icu_occupancy_pct: []
}
数据源: HHS COVID-19 住院数据
粒度限制: 州级（需在 system prompt 中告知 agent 这一局限）
```

### 3.4 query_nearby_sites — 周边站点查询
```
输入: site_id, radius_km (default=100), start_date, end_date
输出: {
  nearby_sites: [
    {
      site_id, distance_km, state,
      concentration_trend: "increasing"|"stable"|"decreasing",
      has_concurrent_anomaly: bool,
      anomaly_correlation: float  // 与目标站点的信号相关性
    }
  ]
}
数据源: NWSS 数据库内部查询
用途: 判断异常是局部事件还是区域性趋势
```

### 3.5 query_variants — 变异株数据查询
```
输入: hhs_region, start_date, end_date
输出: {
  weekly_proportions: [
    {week, dominant_variant, proportion,
     emerging_variants: [{name, proportion, growth_rate}]}
  ],
  notable_shifts: [
    {date_range, from_variant, to_variant, description}
  ]
}
数据源: CDC Variant Surveillance
粒度: HHS Region 级，周频
```

### 3.6 query_site_metadata — 站点元数据查询
```
输入: site_id
输出: {
  state, county, lat, lon,
  population_served, sample_type,
  normalization_method, collection_frequency,
  climate_zone, nearest_noaa_station_id,
  nearest_usgs_station_id,
  historical_stats: {mean, std, median, p25, p75}
}
数据源: 站点元数据表
```

## 4. System Prompt 中的领域知识

Agent 的 system prompt 需要编码以下领域知识（阈值来自 WBE 文献，需要文献支撑）：

### 4.1 稀释效应
- 日降雨量 > 10mm：可能产生可检测的稀释效应
- 日降雨量 > 25mm：显著稀释，信号可能下降 30-50%
- 合流制管网（CSO）：当流量超过处理能力时，未处理污水溢流，信号可能骤升或骤降

### 4.2 RNA 降解
- 管网温度 > 25°C：RNA 半衰期显著缩短（< 24h）
- 管网温度 < 10°C：RNA 相对稳定（半衰期 > 72h）
- 夏季高温地区的信号系统性偏低

### 4.3 信号-临床时滞
- 污水信号通常领先住院数据 4-14 天（中位数约 7 天）
- 领先时间因社区规模和医疗系统响应速度而异
- 如果污水信号上升后 14 天内临床数据无响应，大概率不是真实疫情变化

### 4.4 变异株影响
- 新变异株出现可能改变排毒动力学（如 Omicron 的粪便排毒量高于 Delta）
- 变异株替代期间，即使感染人数不变，污水信号也可能变化
- 需要结合变异株比例变化速率判断

### 4.5 采样异常特征
- 孤立的单点 spike（前后数据点正常）→ 高度疑似采样/检测异常
- 周末/节假日采样可能有系统性偏差
- 同一站点不同 normalization 方法的结果不一致 → 可能是流量数据问题

## 5. 异常检测模块（Agent 上游）

异常检测模块独立于 Agent，负责从连续时间序列中识别候选异常事件。

### 5.1 检测方法（Ensemble）

| 方法 | 实现 | 检测目标 |
|------|------|----------|
| Rolling Z-score | `(x - rolling_median_7d) / rolling_std_7d > 2` | 短期突变 |
| STL 残差 | STL 分解后残差超过 2σ | 去季节性后的异常 |
| 周环比变化 | `abs(pct_change(7)) > threshold` | 趋势突变 |
| PELT 变点检测 | `ruptures.Pelt` | 均值/方差的结构性变化 |

### 5.2 Ensemble 策略
- 任意 ≥2 种方法同时检测到 → 标记为候选异常事件
- 记录每种方法的检测结果，作为 Agent 输入的一部分
- 宁可多检测（高召回），让 Agent 负责过滤假阳性

### 5.3 异常事件定义
- 一个"事件"是连续异常天数的聚合（gap ≤ 3 天的异常合并为同一事件）
- 每个事件记录：event_id, site_id, start_date, end_date, peak_date, peak_zscore, detection_methods
