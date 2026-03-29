# 01 研究设计

## 1. 研究问题

**核心问题：** 当污水 SARS-CoV-2 监测信号出现异常波动时，能否通过 LLM Agent 自动整合多源数据，准确判定异常的根本原因？

**子问题：**
- Q1: Agent 的归因准确率是否显著优于静态分类方法（规则、ML、zero-shot LLM）？
- Q2: 动态多步调查（tool use）相比一次性判断（zero-shot）带来多大提升？
- Q3: 全美 NWSS 网络中，多大比例的信号异常可归因于环境因素？该比例在不同气候区/季节如何变化？

## 2. 异常归因分类体系

Agent 对每个异常事件输出以下分类之一：

| 类别 | 定义 | 典型特征 |
|------|------|----------|
| **真实疫情变化** (Epidemic) | 社区感染率确实发生了变化 | 临床数据 4-14 天后跟随上升；周边站点同步变化；可能伴随新变异株 |
| **环境干扰** (Environmental) | 气象/水文因素导致信号失真 | 降雨事件导致稀释；高温加速 RNA 降解；CSO 溢流；临床数据无对应变化 |
| **采样异常** (Sampling) | 采样或检测过程中的问题 | 孤立 spike/dip；周边站点无同步；无气象事件；可能是实验室误差或采样时间异常 |
| **混合因素** (Mixed) | 多种因素叠加 | 真实疫情变化 + 降雨稀释同时发生；需要分解各因素贡献 |
| **不确定** (Uncertain) | 证据不足以判定 | 数据缺失严重；多种解释同等合理 |

## 3. 为什么是 Agent 而非传统 ML

**不是说 ML "做不了"，而是 Agent 方法在这个场景下有结构性优势：**

### 3.1 传统方法的局限
- **规则系统：** 需要预先穷举所有因果路径。但现实中异常原因的组合是开放的（施工扰动管网、节假日采样异常、处理厂工艺调整……），规则无法覆盖长尾场景。
- **ML 分类器：** 需要大量标注数据训练，且输出是黑盒分类，无法提供可解释的推理链。公共卫生决策者需要知道"为什么"，不只是"是什么"。
- **静态 LLM 分类：** 一次性给所有数据让 LLM 判断，但不同异常需要不同深度的调查——有些看一眼降雨数据就够了，有些需要逐层排查。静态方法无法做这种自适应。

### 3.2 Agent 方法的优势
- **动态证据收集：** 根据中间发现决定下一步查什么，模拟人类流行病学家的调查过程
- **可解释推理链：** 每一步 thought → action → observation 都有记录，决策过程完全透明
- **开放式发现：** 能处理预设规则之外的异常原因
- **成本效率：** 简单案例 2-3 步就能判定，复杂案例才需要深入调查，避免对所有事件做全量分析

### 3.3 论文中的定位
> "We propose an LLM agent framework that mirrors the investigative workflow of human epidemiologists — dynamically gathering and synthesizing multi-source evidence to attribute anomalous wastewater signals to their root causes. Unlike static classifiers that require pre-specified feature sets, the agent adaptively determines which evidence sources to consult based on intermediate findings."

## 4. 创新点

1. **方法创新：** 首次将 LLM Agent 的动态推理能力应用于 WBE 信号解读，提出"异常信号调查"这一新范式
2. **实证贡献：** 基于全美 NWSS 网络的大规模分析，量化环境干扰对污水监测信号的影响程度及其时空分布
3. **实用价值：** 为公共卫生部门提供可部署的自动化工具，减少人工研判负担，提高响应速度

## 5. 目标期刊

| 期刊 | 定位角度 | 优先级 |
|------|----------|--------|
| Nature Water | WBE 方法学突破 | 高（如果结果强） |
| Environmental Science & Technology | 环境监测 + AI | 高 |
| PNAS | 跨学科（AI + 公共卫生） | 中 |
| The Lancet Digital Health | 数字健康工具 | 中 |
| AAAI/NeurIPS Workshop (AI4Science) | AI 方法论 | 备选 |

## 6. 相关工作定位

需要覆盖的文献领域：
- **WBE 信号解读：** 现有的降雨校正方法、流量归一化方法、信号-临床关联分析
- **环境因素对 WBE 的影响：** RNA 降解动力学、稀释效应、CSO 溢流研究
- **LLM Agent 在科学领域的应用：** ChemCrow、BioPlanner 等先例
- **异常检测方法：** 时间序列异常检测的经典方法和在公共卫生监测中的应用
- **多源数据融合：** 环境健康领域的数据整合方法
