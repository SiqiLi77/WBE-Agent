# 04 实验设计

## 1. 三层评估体系

```
Tier 1 (深度)          Tier 2 (广度)           Tier 3 (泛化)
8-11 站点              ~50 异常事件             全量 1188 站点
专家标注               半自动标注               无标注/proxy
Case study + 推理链    定量指标 + Ablation      汇总统计
Motivation figure      Main results table       Core finding
```

### 1.1 Tier 1 — 深度案例分析

**目的：** 展示 Agent 的完整推理过程，提供 motivation figure

**站点：** Core 8 站（MT, OR, NV, WI, IL, MN, GA, NY）

**流程：**
1. 对每个站点运行异常检测，获取所有候选异常事件
2. 人工挑选 3-4 个典型案例（覆盖所有归因类别）
3. 运行 Agent，记录完整推理链
4. 领域专家独立标注（不看 Agent 输出）
5. 对比 Agent 判断 vs 专家判断

**产出：**
- 3-4 个 case study figure（污水浓度 + 降雨 + 住院 + Agent 推理链）
- 定性分析：Agent 在哪些情况下推理正确/错误/部分正确

### 1.2 Tier 2 — 定量评估

**目的：** 提供可量化的性能指标，支撑 baseline 对比和 ablation

**事件集：** ~50 个异常事件（从 Tier 1 + Extension 站点中收集）

**标注方案：**
1. 自动预标注：基于规则生成初始标签
   - 异常后 14 天内住院数上升 >20% → 预标注为 epidemic
   - 异常前 48h 内降雨 >25mm 且住院无变化 → 预标注为 environmental
   - 孤立单点 spike → 预标注为 sampling
   - 其他 → 预标注为 uncertain
2. 专家审核：2-3 位 WBE 领域专家对预标注结果进行审核和修正
3. 计算 inter-rater reliability（Cohen's kappa / Fleiss' kappa）
4. 不一致的标签通过讨论达成共识，或标记为 uncertain

**事件平衡目标：**
| 类别 | 目标数量 | 备注 |
|------|---------|------|
| Epidemic | 15-20 | 确保覆盖不同幅度的疫情变化 |
| Environmental | 15-20 | 覆盖降雨、高温、融雪等不同机制 |
| Sampling | 5-10 | 相对少见但重要 |
| Mixed | 5-10 | 最难判断的类别 |
| Uncertain | 0-5 | 允许但尽量减少 |

**评估指标：**
| 指标 | 计算方式 | 用途 |
|------|---------|------|
| Overall Accuracy | 正确分类 / 总事件数 | 整体性能 |
| Macro F1 | 各类别 F1 的均值 | 类别平衡性能 |
| Cohen's Kappa (Agent vs Expert) | κ 统计量 | 与专家的一致性 |
| Per-class Precision/Recall | 分类别计算 | 各类别的强弱项 |
| Reasoning Quality Score | 专家盲评 1-5 分 | 推理链的合理性 |

### 1.3 Tier 3 — 大规模泛化验证

**目的：** 产出论文核心发现，证明框架的泛化性

**推荐方案B：** 全量部署 + 汇总统计

**流程：**
1. 对全部 1188 个 NWSS 站点运行异常检测
2. 对所有检测到的异常事件运行 Agent
3. 不做逐个标注，只报汇总统计

**核心发现（预期）：**
- 全美 NWSS 网络中，X% 的信号异常被 Agent 归因为环境干扰
- 该比例在不同气候区的分布：arid > continental > subtropical > marine
- 季节性模式：夏季（高温降解）和春季（融雪稀释）的环境干扰比例最高
- 不同归因类别的 Agent 置信度分布

**成本估算：**
```
假设：
- 1188 站点 × 平均每站 20 个异常事件 = ~24000 次 Agent 调用
- 每次调用平均 5 个 tool call，每次 ~2000 tokens
- 总 token 消耗：~240M tokens
- GPT-4-turbo 价格：~$10/1M input tokens
- 预估成本：$2400-4800（取决于实际 token 消耗）
```

## 2. Baseline 对比

| 方法 | 描述 | 对比目的 |
|------|------|----------|
| **Rule-based** | 简单规则：下雨→环境干扰，临床跟随→疫情，其他→不确定 | 下界：最简单的方法能做到什么程度 |
| **Logistic Regression** | 用气象+水文+临床特征训练分类器 | 传统 ML baseline |
| **Random Forest / XGBoost** | 同上，更强的 ML 方法 | ML 上界 |
| **Zero-shot LLM** | 一次性给所有数据，直接分类，不用 tool | 隔离"动态调查"的价值 |
| **Few-shot LLM** | 同上 + 3-5 个示例 | 隔离"示例"vs"工具"的价值 |
| **Human Expert** | 领域专家的判断 | 上界参考 |

**最关键的对比：Agent (with tools) vs Zero-shot LLM (without tools)**
- 如果 Agent 显著优于 Zero-shot → 证明动态调查的价值
- 如果差异不大 → 说明静态分类已经足够，Agent 的复杂度不值得

## 3. 消融实验

逐个移除 Agent 的工具/信息源，观察性能变化：

| 消融条件 | 移除内容 | 预期影响 |
|----------|---------|----------|
| -Weather | 移除气象数据查询工具 | Environmental 类别的识别率大幅下降 |
| -Hydrology | 移除水文数据查询工具 | 稀释相关的 Environmental 识别受影响 |
| -Clinical | 移除临床数据查询工具 | Epidemic 类别的确认能力下降 |
| -Nearby | 移除周边站点查询工具 | 区域性 vs 局部性的判断能力下降 |
| -Variants | 移除变异株查询工具 | 对变异株驱动的疫情变化识别受影响 |
| -Domain Knowledge | 移除 system prompt 中的领域知识 | 整体推理质量下降 |
| -All Tools | 只保留 system prompt，不给任何工具 | 退化为 zero-shot（验证一致性） |

## 4. 异常检测方法评估

异常检测模块本身也需要评估（作为补充实验）：

| 方法 | 参数 | 评估指标 |
|------|------|----------|
| Rolling Z-score | threshold = {1.5, 2.0, 2.5, 3.0} | Precision, Recall vs 专家标注 |
| STL 残差 | period = {7, 14}, threshold = {2.0, 2.5} | 同上 |
| 周环比变化 | threshold = {50%, 100%, 200%} | 同上 |
| PELT 变点 | penalty = {auto, BIC, AIC} | 同上 |
| Ensemble (≥2) | 上述组合 | 同上 |

**目标：** 选择一个高召回率的配置（宁可多检测），让 Agent 负责过滤假阳性。

## 5. 统计分析计划

### 5.1 主要分析
- Agent accuracy vs baselines：McNemar's test（配对分类比较）
- Agent vs Expert agreement：Cohen's kappa + 95% CI
- 消融实验：每个条件的 accuracy drop + 显著性检验

### 5.2 次要分析
- 按气候带分层的 Agent 性能差异：Kruskal-Wallis test
- 按季节分层的归因分布差异：Chi-square test
- Agent 置信度与实际准确率的校准曲线
- Tool call 次数与事件复杂度的关系

### 5.3 可视化计划
| 图表 | 内容 | 用于 |
|------|------|------|
| Fig 1 | Motivation: 3 个典型案例的时间序列 + 归因 | Introduction |
| Fig 2 | 系统架构图 | Methods |
| Fig 3 | Agent 推理链示例（完整 case study） | Results |
| Fig 4 | Accuracy comparison: Agent vs all baselines | Results |
| Fig 5 | Ablation results: 各工具的贡献 | Results |
| Fig 6 | 全美归因分布地图（Tier 3） | Results |
| Fig 7 | 气候带 × 季节的归因热力图 | Results |
| Table 1 | 数据源和站点概览 | Methods |
| Table 2 | Tier 2 定量评估结果 | Results |
| Table 3 | 消融实验结果 | Results |
