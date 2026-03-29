# 03 数据管线

## 1. 数据源总览

| 数据源 | URL / 获取方式 | 粒度 | 频率 | 关键字段 | 预估大小 |
|--------|---------------|------|------|----------|----------|
| NWSS 污水 | `data.cdc.gov/api/views/g653-rqe2/rows.csv` | 站点 | 1-7天 | key_plot_id, date, pcr_conc_lin, normalization | ~200MB |
| HHS 住院 | `healthdata.gov/api/views/g62h-syeh/rows.csv` | 州 | 日 | state, date, previous_day_admission_adult_covid_confirmed | ~500MB |
| NOAA GHCN-Daily | `ncei.noaa.gov/pub/data/ghcn/daily/` 批量下载 | 气象站 | 日 | STATION, DATE, PRCP, TMAX, TMIN | ~30GB(全量) |
| USGS NWIS | `waterdata.usgs.gov/nwis/dv` REST API | 水文站 | 日 | site_no, datetime, discharge_cfs | 按站点查询 |
| CDC Variants | `covid.cdc.gov/covid-data-tracker` | HHS Region | 周 | week, variant, proportion | ~10MB |

## 2. 站点选择（已完成）

### 2.1 三层站点体系

**Tier 1 — Core（11 站，深度分析）**

训练主力（>700天 NWSS-HHS 重叠）：
| 站点 | 州 | 气候带 | 角色 |
|------|----|--------|------|
| MT | Montana | continental_north | train |
| OR | Oregon | marine_west | train |
| NV | Nevada | arid_semiarid | train |
| WI | Wisconsin | continental_north | train |
| IL | Illinois | continental_central | train |
| MN | Minnesota | continental_north | train |
| GA | Georgia | subtropical_south | train |
| NY | New York | continental_central | train |

评估站（250-700天）：
| 站点 | 州 | 气候带 | 角色 |
|------|----|--------|------|
| PA | Pennsylvania | continental_central | test |
| NC | North Carolina | subtropical_south | test |
| FL | Florida | subtropical_south | test |

**Tier 2 — Extension（8 站，定量评估补充）**
| 站点 | 州 | 气候带 | 角色 |
|------|----|--------|------|
| TX | Texas | subtropical_south | test |
| WY | Wyoming | continental_north | test |
| AZ | Arizona | arid_semiarid | test |
| WA | Washington | marine_west | test |
| + 4 站待定 | | | |

**Tier 3 — Challenge（4 站 / 全量部署）**
| 站点 | 州 | 备注 |
|------|----|------|
| NM | New Mexico | 仅污水阶段，NWSS-HHS 重叠不足 |
| + 全量 1188 站点部署（方案B） | | |

### 2.2 气候带覆盖

| 气候带 | Core 站点 | Extension 站点 | 覆盖状态 |
|--------|-----------|---------------|----------|
| continental_north | MT, WI, MN | WY | ✅ 充足 |
| continental_central | IL, NY | PA | ✅ 充足 |
| subtropical_south | GA | NC, FL, TX | ✅ 充足 |
| marine_west | OR | WA | ⚠️ 可接受 |
| arid_semiarid | NV | AZ, NM | ⚠️ 数据薄弱，需 acknowledge |

### 2.3 已知数据质量问题
- **AZ:** 仅 188 obs，median_gap=4天，置信区间会较宽
- **TX:** cv_raw=5.98，信号噪声大，如果影响整体结果可降级到 extension
- **NM:** 146 obs, 549天，NWSS-HHS 重叠不足，仅用于 Phase 2 部署模拟

## 3. 数据预处理管线

### Step 1: 加载与过滤

```python
# 伪代码
nwss_raw = pd.read_csv("NWSS_*.csv", usecols=["key_plot_id", "date", "pcr_conc_lin", "normalization"])
nwss = nwss_raw[
    (nwss_raw["key_plot_id"].isin(selected_sites)) &
    (nwss_raw["normalization"] == "flow-population")
]
```

**注意：** 原始 CSV 可能很大，用 `usecols` 限制列，或分块读取。

### Step 2: 日期解析与去重

```python
nwss["date"] = pd.to_datetime(nwss["date"])
nwss = nwss.sort_values(["key_plot_id", "date"])

# 检查重复
dupes = nwss.groupby(["key_plot_id", "date"]).size()
dupes_found = dupes[dupes > 1]
# 处理策略：取均值，记录处理日志
```

### Step 3: 数据完整性校验

对每个站点验证：
- [ ] n_obs 与选定清单一致
- [ ] start_date / end_date 吻合
- [ ] pcr_conc_lin 无负值
- [ ] NaN 比例在可接受范围内
- [ ] 零值处理：检测下限以下 → 替换为 LOD/2（需确认各站点 LOD）

### Step 4: 时间网格对齐

```python
# 对每个站点，构建完整日历网格
for site_id in selected_sites:
    site_data = nwss[nwss["key_plot_id"] == site_id]
    full_index = pd.date_range(site_data["date"].min(), site_data["date"].max(), freq="D")
    site_data = site_data.set_index("date").reindex(full_index)
    # 缺失日标记为 NaN，不做插值
```

各站点采样频率参考：
| 站点 | median_gap | 模式 |
|------|-----------|------|
| WI, IL, OR | 1-2 天 | 近似每日 |
| GA, NC | 5 天 | 约每周 |
| MT | 7 天 | 每周 |
| AZ | 4 天 | 不规则 |

### Step 5: 信号变换（派生列）

| 派生列 | 计算方式 | 用途 |
|--------|---------|------|
| `log1p_conc` | `np.log1p(pcr_conc_lin)` | 压缩量级差异 |
| `rolling_median_7d` | `rolling(7, min_periods=4).median()` | 平滑噪声 |
| `rolling_median_14d` | `rolling(14, min_periods=7).median()` | 趋势提取 |
| `pct_change_7d` | `pct_change(periods=7)` | 周变化率 |
| `rolling_zscore` | `(x - rolling_median) / rolling_std` | 异常检测输入 |
| `site_zscore` | `(log1p - site_mean) / site_std` | 跨站点可比性 |

**原则：** 所有变换保留为独立列，不覆盖原始值。

### Step 6: 跨站点标准化

- 站点内建模：保留原始 `log1p_conc`
- 跨站点比较/聚合：使用 `site_zscore`（站点内 z-score 标准化）
- 可视化：根据需要选择 min-max 或 percentile rank

## 4. 空间匹配

### 4.1 NWSS → NOAA 气象站匹配

```
对每个 NWSS 站点（lat, lon）：
1. 从 GHCN-Daily 站点目录中找最近的气象站（Haversine 距离）
2. 约束：距离 < 50km，且该气象站在 NWSS 时间窗口内有 >80% 的日降水数据
3. 如果最近站不满足数据完整性要求，尝试次近站
4. 记录匹配距离，作为数据质量指标
```

### 4.2 NWSS → USGS 水文站匹配

```
对每个 NWSS 站点：
1. 查询 USGS 站点目录，找同一 HUC8 流域内的活跃水文站
2. 优先选择同一河流/溪流上的站点
3. 约束：距离 < 30km，有连续的日流量数据
4. Fallback：如果无合适水文站，用 NOAA 降水量作为稀释代理变量
```

### 4.3 NWSS → HHS 住院数据匹配

```
直接按州匹配（HHS 数据为州级）
已知局限：同一州内不同站点共享同一住院数据
论文中需 acknowledge 这一 ecological fallacy
```

### 4.4 NWSS → CDC 变异株匹配

```
按 HHS Region 匹配（CDC 变异株数据为 HHS Region 级）
10 个 HHS Region 覆盖全美
时间分辨率：周级 → 需要对齐到日级（同一周内填充相同值）
```

## 5. 已有产出文件

| 文件 | 路径 | 内容 |
|------|------|------|
| tier1_core_final_sites.csv | D:\agent\outputs\ | 11 个核心站点清单 |
| tier2_extension_sites.csv | D:\agent\outputs\ | 8 个扩展站点清单 |
| tier3_challenge_sites.csv | D:\agent\outputs\ | 4 个挑战站点清单 |
| hhs_16states.csv | D:\agent\outputs\ | 16 州全量 HHS 数据（24568 行） |
| hhs_16states_overlap_only.csv | D:\agent\outputs\ | NWSS-HHS 重叠窗口数据（10269 行） |
| hhs_nwss_overlap.csv | D:\agent\outputs\ | 重叠窗口元数据 + modeling_role |

## 6. 待完成的数据工作

- [ ] NWSS 原始数据预处理（Step 1-6）
- [ ] NOAA 气象站空间匹配 + 数据下载
- [ ] USGS 水文站空间匹配 + 数据下载
- [ ] CDC 变异株数据获取与对齐
- [ ] 多源数据合并为统一数据库
- [ ] 数据质量报告生成
