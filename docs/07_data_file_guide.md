# 07 数据文件说明

本文档列出项目中所有数据文件的含义、用途和来源。

---

## 一、原始数据 (`data/raw/`)

这些是从外部数据源下载的原始文件，未经任何处理。

### 1. `NWSS_Public_SARS-CoV-2_Concentration_in_Wastewater_Data.csv`

- **是什么**：CDC 国家污水监测系统（NWSS）的公开数据，记录了全美各污水处理厂采集的污水样本中 SARS-CoV-2 病毒的浓度。
- **用来干啥**：这是整个项目的核心数据。我们从中筛选出 16 个站点的污水病毒浓度时间序列，用于检测异常信号。
- **怎么得来的**：从 CDC NWSS 公开数据集直接下载。
- **关键列**：`key_plot_id`（站点 ID）、`date`（采样日期）、`pcr_conc_lin`（病毒浓度，PCR 检测值）、`normalization`（标准化方式，flow-population）。

### 2. `COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State.csv`

- **是什么**：美国 HHS（卫生与公众服务部）发布的各州 COVID-19 住院数据，包含每日新增住院人数、ICU 占用率、床位使用率等。
- **用来干啥**：当污水信号出现异常时，Agent 会查询该州的住院数据来判断是否是真实的疫情爆发——如果住院人数也在上升，说明污水信号可能反映了真实疫情。
- **怎么得来的**：从 HHS 公开数据集下载，由 `src/data_pipeline/step2_hhs.py` 筛选出 16 个相关州的数据。
- **关键列**：`state`、`date`、`previous_day_admission_adult_covid_confirmed`（前日成人确诊住院数）等 100+ 列。

### 3. `cdc_variants.csv`

- **是什么**：CDC 发布的 SARS-CoV-2 变异株比例数据，按 HHS Region（美国 10 个卫生区域）和周统计各变异株的流行占比。
- **用来干啥**：Agent 用它来判断异常信号是否与新变异株出现有关——比如 JN.1 取代 HV.1 成为主流毒株时，可能导致污水浓度突然变化。
- **怎么得来的**：通过 CDC SODA API 下载（`src/data_pipeline/step5_variants.py`），筛选了 share ≥ 5% 的变异株，共 209,439 条记录。
- **关键列**：`usa_or_hhsregion`（HHS 区域编号）、`week_ending`（周末日期）、`variant`（变异株名称，如 JN.1、HV.1）、`share`（占比）。

### 4. `ghcn_daily/` 目录

- **`ghcnd-stations.txt`**：全球 GHCN 气象站的元数据（站点 ID、经纬度、海拔、名称）。用于找到离每个污水站点最近的气象站。
- **`ghcnd-inventory.txt`**：各气象站有哪些观测要素、覆盖哪些年份。用于过滤出有近期数据的活跃站点。
- **`station_data/*.parquet`**：11 个匹配气象站的日降水量和气温数据。文件名格式为 `站点ID_起始日期_结束日期.parquet`。
- **怎么得来的**：由 `src/data_pipeline/step3_noaa.py` 从 NOAA GHCN-Daily 数据集下载。先用 inventory 找到活跃站点，再按距离匹配最近的气象站，最后下载该站的日数据。

### 5. `usgs_cache/` 目录

- **`site_queries/sites_XX_00060.csv`**：各州 USGS 水文站的查询结果（站点列表）。`00060` 是 USGS 的流量参数代码。
- **`daily_data/*.parquet`**：11 个匹配水文站的日流量数据。文件名格式为 `站点编号_起始日期_结束日期.parquet`。
- **怎么得来的**：由 `src/data_pipeline/step4_usgs.py` 通过 USGS NWIS API 下载。先按州查询有流量数据的水文站，再按距离匹配最近的站，最后下载日流量。

---

## 二、处理后数据 (`data/processed/`)

这些是经过清洗、对齐、合并后的中间产物，供后续分析使用。

### 1. `nwss_cleaned.parquet`

- **是什么**：清洗后的 NWSS 污水数据，16 个站点、18,810 行。
- **用来干啥**：异常检测的直接输入。包含原始浓度、log 变换值、滚动中位数、滚动标准差、z-score 等派生特征。
- **怎么得来的**：由 `src/data_pipeline/step1_nwss.py` 从原始 NWSS CSV 中筛选 16 个站点，做日期对齐、去重、log1p 变换、滚动统计等处理。

### 2. `hhs_cleaned.parquet`

- **是什么**：清洗后的 HHS 住院数据，按州-日期组织。
- **用来干啥**：合并到主数据集中，供 Agent 的 `query_hospitalization` 工具查询。
- **怎么得来的**：由 `src/data_pipeline/step2_hhs.py` 从原始 HHS CSV 中筛选 16 个州，做日期解析和格式统一。

### 3. `noaa_matched.parquet`

- **是什么**：与污水站点空间匹配后的 NOAA 气象数据（5,824 行，9 个站点有数据）。
- **用来干啥**：合并到主数据集中，供 Agent 的 `query_weather` 工具查询降水和气温。
- **怎么得来的**：由 `src/data_pipeline/step3_noaa.py` 将气象站数据按最近距离匹配到污水站点，并对齐到站点的时间网格上。

### 4. `usgs_matched.parquet`

- **是什么**：与污水站点空间匹配后的 USGS 水文数据（13,721 行，11 个站点全部匹配）。
- **用来干啥**：合并到主数据集中，供 Agent 的 `query_hydrology` 工具查询河流流量。暴雨后流量激增可能稀释污水，导致假异常。
- **怎么得来的**：由 `src/data_pipeline/step4_usgs.py` 将水文站数据按最近距离匹配到污水站点。

### 5. `variants_aligned.parquet`

- **是什么**：按州对齐后的变异株比例数据（20,496 行，16 个州）。
- **用来干啥**：合并到主数据集中，供 Agent 的 `query_variants` 工具查询。原始数据是周频的，这里已插值为日频。
- **怎么得来的**：由 `src/data_pipeline/step5_variants.py` 将 CDC 变异株数据从 HHS Region 级映射到州级，并从周频对齐到日频。

### 6. `merged_multisource.parquet`

- **是什么**：所有数据源合并后的主数据集（18,810 行 × 166 列）。每行是一个"站点-日期"记录。
- **用来干啥**：这是 Agent 所有工具的底层数据源。Agent 调查某个事件时，工具函数从这个文件中查询对应站点和日期范围的数据。
- **怎么得来的**：由 `src/data_pipeline/step6_merge.py` 以 NWSS 数据为主表，左连接 HHS、NOAA、USGS、变异株数据。

### 7. `nwss_sampling_patterns.csv`

- **是什么**：各站点的采样频率分析（中位采样间隔、平均间隔、是否规律采样等）。
- **用来干啥**：了解各站点的数据密度。比如 NV 站点几乎每天采样，而 FL 站点平均每 3-4 天采一次。
- **怎么得来的**：由 `src/data_pipeline/step1_nwss.py` 在清洗过程中统计生成。

---

## 三、标注数据 (`data/labeled/`)

### 1. `auto_labeled_events.csv`

- **是什么**：对 152 个异常事件的规则化自动标注结果。
- **用来干啥**：作为 Agent 调查结果的对比基准。规则很简单（比如"有降雨就标 environmental"、"住院上升就标 epidemic"），所以 55% 的事件被标为 uncertain。Agent 的价值就在于能解决这些规则解决不了的事件。
- **怎么得来的**：由 `src/labeling/auto_label.py` 基于简单规则生成。规则逻辑：查降水量是否 > 阈值、住院是否上升、是否为孤立单点 spike 等。
- **关键列**：`event_id`、`auto_label`（5 类分类）、`auto_confidence`（置信度）、`auto_reasoning`（判断理由，中文）。

---

## 四、输出文件 (`outputs/`)

### 1. `investigation_results.csv` ⭐ 核心结果

- **是什么**：Agent 对 152 个异常事件的调查结论，每行一个事件。
- **用来干啥**：这是整个项目最重要的产出。记录了 Agent 对每个异常信号的分类判断（5 类）、置信度、消耗的资源（token 数、工具调用次数）和一句话摘要。
- **怎么得来的**：由 `scripts/run_agent.py` 调用 `src/agent/runner.py`，对每个事件启动一个 LLM Agent（gpt-4o-mini via OpenRouter），Agent 依次调用 6 个工具收集证据，最终给出分类。
- **5 类分类含义**：
  - `epidemic`：真实疫情信号（住院上升 + 变异株切换等证据支持）
  - `environmental`：环境因素导致的假信号（暴雨、洪水等）
  - `sampling`：采样或实验室误差（孤立 spike，前后正常）
  - `mixed`：多因素叠加，无法归因到单一原因
  - `uncertain`：证据不足，Agent 也无法判断

### 2. `anomaly_event_catalog.csv`

- **是什么**：异常检测阶段产出的事件目录，152 个事件。
- **用来干啥**：Agent 调查的输入——"嫌疑人名单"。每个事件是一个被多种统计方法同时标记为异常的污水浓度波动。
- **怎么得来的**：由 `scripts/run_detection.py` 调用 `src/anomaly_detection/` 模块，用 4 种方法（Rolling Z-score、STL 分解、周环比、PELT 变点检测）做 Ensemble 投票，≥2 票的标记为异常，再将连续异常天合并为事件。
- **关键列**：`event_id`、`site_id`、`start_date`/`end_date`（异常持续时间）、`peak_zscore`（异常程度）、`detection_methods`（哪些方法投票）、`vote_count_max`（最高票数）。

### 3. `logs/traces/EVT-*.json`（152 个文件）

- **是什么**：每个事件的 Agent 推理过程完整记录。
- **用来干啥**：可以回溯 Agent 是怎么一步步调查的——调用了哪些工具、传了什么参数、拿到了什么数据、最终怎么判断的。对于调试 Agent 行为和写论文的 case study 非常有用。
- **怎么得来的**：Agent 运行时自动记录，由 `src/agent/runner.py` 在每次工具调用后写入。
- **结构**：JSON 格式，包含 `event_id` 和 `tool_calls` 数组。每个 tool call 有 `step`（第几步）、`tool`（工具名）、`arguments`（参数）、`result_preview`（返回数据预览）。
- **典型调查流程**（以 EVT-00001 为例）：
  1. `query_site_metadata` → 了解站点基本信息（FL 州，亚热带气候）
  2. `query_weather` → 查异常日前后天气（12/26 有 5.3mm 小雨）
  3. `query_hydrology` → 查河流流量（正常，无洪水）
  4. `query_nearby_sites` → 查附近站点是否也异常（100km 内无其他站点）
  5. `query_hospitalization` → 查住院数据（FL 住院从 180 升到 277）
  6. `query_variants` → 查变异株（HV.1 → JN.1 切换中）

### 4. `quality_report/` 目录（4 个文件）

数据质量报告，帮助了解数据的完整性和可靠性。

- **`coverage.csv`**：每个站点各数据源的覆盖率。比如 NV 站点 NWSS 覆盖 96.3%（几乎每天都有采样），而 AZ 站点 NOAA 覆盖 0%（附近没有匹配到气象站）。
- **`missing.csv`**：各列的缺失率排名。温度数据缺失 85%（只有 MT 和 PA 两个站有），`geocoded_state` 100% 缺失（未使用的字段）。
- **`outliers.csv`**：各列的异常值统计。降水有 921 个极端高值（暴雨天），原始病毒浓度有 442 个极端高值。
- **`spatial_match.csv`**：NOAA 和 USGS 站点的空间匹配距离。NOAA 平均 1.6km（很近，数据可靠），USGS 平均 4.1km（可接受）。
- **怎么得来的**：由 `src/data_pipeline/step7_quality.py` 在数据合并后自动生成。

### 5. `tier1_core_final_sites.csv` / `tier2_extension_sites.csv` / `tier3_challenge_sites.csv`

- **是什么**：三层站点选择清单。
  - Tier 1（11 站）：数据质量最好的核心站点，用于开发和主要评估
  - Tier 2（4 站）：扩展站点，增加地理和气候覆盖
  - Tier 3（1 站）：挑战站点（NM），数据质量差，测试 Agent 鲁棒性
- **用来干啥**：定义了实验的站点范围和角色分配（train/test/extension_test/phase2_only）。
- **怎么得来的**：由 `scripts/prepare_data.py` 基于数据质量评分、气候带覆盖、地理分布等标准筛选生成。
- **关键列**：`state`、`key_plot_id`、`quality_score`（质量评分）、`climate_bucket`（气候带）、`modeling_role`（角色：train/test 等）、`latitude`/`longitude`。

### 6. `hhs_16states.csv`

- **是什么**：16 个州的 HHS 住院原始数据提取（24,568 行，100+ 列）。
- **用来干啥**：`src/data_pipeline/step2_hhs.py` 的中间产出，后续被清洗为 `data/processed/hhs_cleaned.parquet`。
- **怎么得来的**：从原始 HHS CSV 中按 16 个州筛选。

---

## 五、文件之间的关系（数据流）

```
原始数据 (data/raw/)
  │
  ├─ NWSS CSV ──→ step1_nwss.py ──→ nwss_cleaned.parquet ─┐
  ├─ HHS CSV  ──→ step2_hhs.py  ──→ hhs_cleaned.parquet  ─┤
  ├─ GHCN 气象 ─→ step3_noaa.py ──→ noaa_matched.parquet ─┤
  ├─ USGS 水文 ─→ step4_usgs.py ──→ usgs_matched.parquet ─┼──→ step6_merge.py ──→ merged_multisource.parquet
  └─ CDC 变异株 → step5_variants ─→ variants_aligned.parquet┘           │
                                                                         ↓
                                                              step7_quality.py ──→ quality_report/
                                                                         │
                                                                         ↓
                                                              anomaly detection ──→ anomaly_event_catalog.csv
                                                                         │
                                                                         ↓
                                                              auto_label.py ──→ auto_labeled_events.csv
                                                                         │
                                                                         ↓
                                                              Agent 调查 ──→ investigation_results.csv
                                                                         └──→ logs/traces/EVT-*.json
```
