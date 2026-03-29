<div align="center">

# 🧪 WBE-Agent

**National-Scale Agentic Interpretation of Wastewater SARS-CoV-2 Anomalies**

**污水 SARS-CoV-2 异常信号的国家尺度 Agent 解读框架**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Next.js](https://img.shields.io/badge/Web_UI-Next.js_15-black.svg)](web/)

---

[English](#overview) · [中文](#概述)

</div>

## Overview

Wastewater surveillance detects community pathogen transmission before clinical indicators, but anomalous signals are difficult to interpret — the same concentration spike can reflect a real epidemic, rainfall dilution, sampling error, or a combination of causes.

**WBE-Agent** is an LLM agent framework that automates root-cause investigation of wastewater anomalies by iteratively querying aligned multi-source evidence (weather, hydrology, hospitalization, variant surveillance, nearby sites). Instead of one-shot classification, the agent conducts a structured investigation — mimicking how an epidemiologist would review each event — and produces auditable, decision-support explanations.

### Key Results

| Metric | WBE-Agent | Best ML Baseline | Zero-shot LLM |
|--------|-----------|-----------------|---------------|
| Accuracy | **78.9%** | 65.3% | 13.1% |
| Macro-F1 | **0.767** | 0.507 | 0.139 |
| Cohen's κ | **0.619** | 0.358 | 0.049 |
| Epidemic Recall | **81.8%** | 72.7% | 22.7% |
| Environmental Recall | **94.7%** | 63.2% | 10.5% |

*Evaluated on 199 expert-investigated events across 16 U.S. states and 5 climate zones (2020–2025).*

### Highlights

- 🔬 **Multi-source evidence coordination** — 5 public data sources aligned into 18,810 site-days × 166 variables
- 🤖 **ReAct investigation loop** — 6 domain tools, structured reasoning chains, explicit uncertainty
- 📊 **Comprehensive evaluation** — 8 baselines, 7 ablation conditions, 11 foundation-model substitutions
- 🌐 **Interactive web dashboard** — Next.js UI for browsing investigations and triggering workflows
- ✅ **85.2% uncertain-event resolution** — converts ambiguous alerts into actionable interpretations

---

## 概述

污水监测可以在临床指标之前检测到社区病原体传播，但异常信号难以解读——同一个浓度突变可能反映真实疫情、降雨稀释、采样误差或多种原因的叠加。

**WBE-Agent** 是一个基于大语言模型的 Agent 框架，通过迭代查询多源对齐证据（气象、水文、住院、变异株、周边站点），自动化执行污水异常信号的根因调查。Agent 不是一次性分类，而是模拟流行病学家的调查流程，产生可审计的决策支持解释。

### 核心结果

| 指标 | WBE-Agent | 最强 ML 基线 | 零样本 LLM |
|------|-----------|-------------|-----------|
| 准确率 | **78.9%** | 65.3% | 13.1% |
| 宏 F1 | **0.767** | 0.507 | 0.139 |
| Cohen's κ | **0.619** | 0.358 | 0.049 |
| 疫情召回率 | **81.8%** | 72.7% | 22.7% |
| 环境召回率 | **94.7%** | 63.2% | 10.5% |

*在美国 16 个州、5 个气候带的 199 个专家调查事件上评估（2020–2025）。*

### 亮点

- 🔬 **多源证据协调** — 5 个公开数据源对齐为 18,810 站点-日 × 166 变量
- 🤖 **ReAct 调查循环** — 6 个领域工具、结构化推理链、显式不确定性
- 📊 **全面评估** — 8 种基线、7 种消融条件、11 种基础模型替换
- 🌐 **交互式 Web 仪表盘** — Next.js 界面浏览调查结果和触发工作流
- ✅ **85.2% 不确定事件解析** — 将模糊告警转化为可操作的解读

---

## Architecture / 架构

```
┌─────────────────────────────────────────────────────────┐
│                     WBE-Agent System                     │
├──────────────┬──────────────────────────────────────────┤
│  5 Data      │         ReAct Investigation Loop          │
│  Sources     │  ┌─────────┐  ┌──────────────────────┐  │
│              │  │ System   │  │ Thought → Action     │  │
│  • NWSS      │  │ Prompt   │→ │ → Observation        │  │
│  • HHS       │  │ (Domain  │  │ → Thought → ...      │  │
│  • NOAA      │  │  Knowledge)│ └──────────┬───────────┘  │
│  • USGS      │  └─────────┘              │              │
│  • CDC       │                      Tool Calls           │
│  Variants    │         ┌─────────────────┤              │
│              │         ▼                 ▼              │
│              │  ┌────────────┐  ┌──────────────┐       │
│              │  │ Weather    │  │ Hospitalization│      │
│              │  │ Hydrology  │  │ Variants      │       │
│              │  │ Nearby Sites│ │ Site Metadata │       │
│              │  └────────────┘  └──────────────┘       │
├──────────────┴──────────────────────────────────────────┤
│  Structured Output: classification, confidence,          │
│  reasoning_chain, primary_factors, data_gaps             │
└─────────────────────────────────────────────────────────┘
```


## Quick Start / 快速开始

### Prerequisites / 前置条件

- Python 3.11+
- Node.js 18+ (for web dashboard)
- An OpenRouter or OpenAI API key

### Installation / 安装

```bash
git clone https://github.com/YOUR_USERNAME/WBE-Agent.git
cd WBE-Agent

# Python environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration / 配置

Edit `config/settings.yaml` and set your API key:

```yaml
agent:
  api_key: "your-openrouter-api-key-here"
  api_base: "https://openrouter.ai/api/v1"
  model: "openai/gpt-4o-mini"
```

### Run the Pipeline / 运行流程

```bash
# Step 1: Prepare raw data (downloads from public APIs)
# 步骤 1：准备原始数据（从公开 API 下载）
python scripts/prepare_data.py --all

# Step 2: Run data pipeline (clean, align, merge 5 sources)
# 步骤 2：运行数据管线（清洗、对齐、合并 5 个数据源）
python scripts/run_pipeline.py --step all

# Step 3: Detect anomaly events (5-method ensemble, ≥2 votes)
# 步骤 3：检测异常事件（5 方法集成，≥2 票）
python scripts/run_detection.py --min-votes 2

# Step 4: Run agent investigations
# 步骤 4：运行 Agent 调查
python scripts/run_agent.py --max-events 10

# Step 5: Evaluate against baselines
# 步骤 5：与基线方法对比评估
python scripts/run_evaluation.py --methods all
```

### Web Dashboard / Web 仪表盘

```bash
# Terminal 1: Start backend API
# 终端 1：启动后端 API
python scripts/run_api.py

# Terminal 2: Start frontend
# 终端 2：启动前端
cd web
npm install
npm run dev
# Open http://localhost:3000
```

---

## Project Structure / 项目结构

```
WBE-Agent/
├── src/                          # Production code / 生产代码
│   ├── data_pipeline/            #   Steps 1-7: ingest, clean, merge
│   │   ├── step1_nwss.py         #     Wastewater concentration data
│   │   ├── step2_hhs.py          #     Hospitalization data
│   │   ├── step3_noaa.py         #     Weather data (GHCN-Daily)
│   │   ├── step4_usgs.py         #     Hydrology data (NWIS)
│   │   ├── step5_variants.py     #     Variant surveillance
│   │   ├── step6_merge.py        #     Multi-source merge
│   │   └── step7_quality.py      #     Quality control reports
│   ├── anomaly_detection/        #   Ensemble anomaly detection
│   │   ├── detectors.py          #     5 detection methods
│   │   ├── ensemble.py           #     Voting mechanism
│   │   └── events.py             #     Event catalog builder
│   ├── agent/                    #   LLM investigation agent
│   │   ├── runner.py             #     ReAct loop orchestration
│   │   ├── prompts.py            #     System & event prompts
│   │   ├── schema.py             #     Output schema (Pydantic)
│   │   └── tools/                #     6 domain-specific tools
│   ├── evaluation/               #   Benchmarking
│   │   ├── baselines.py          #     7 baseline methods
│   │   └── metrics.py            #     Scoring functions
│   ├── labeling/                 #   Label generation
│   │   ├── auto_label.py         #     Rule-based auto-labels
│   │   └── llm_silver.py         #     Multi-judge silver labels
│   ├── webapi/                   #   FastAPI backend
│   └── utils/                    #   Shared helpers
├── scripts/                      # CLI entrypoints / 命令行入口
├── config/settings.yaml          # Thresholds and paths / 阈值与路径配置
├── web/                          # Next.js dashboard / Web 仪表盘
├── data/labeled/                 # Expert reference labels / 专家参考标签
├── outputs/evaluation/           # Benchmark results / 评估结果
└── docs/                         # Design documentation / 设计文档
```

---

## Data Sources / 数据来源

All data are from U.S. federal public sources:

| Source | Agency | Granularity | Key Variables |
|--------|--------|-------------|---------------|
| [NWSS](https://www.cdc.gov/nwss/) | CDC | Site-day | SARS-CoV-2 concentration |
| [HHS Hospital](https://healthdata.gov/) | HHS | State-day | COVID-19 admissions |
| [GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) | NOAA | Station-day | Precipitation, temperature |
| [NWIS](https://waterservices.usgs.gov/) | USGS | Gauge-day | Streamflow discharge |
| [Variant Tracker](https://covid.cdc.gov/covid-data-tracker/) | CDC | Region-week | Lineage proportions |

Raw data files are not included in this repository due to size. Run `python scripts/prepare_data.py --all` to download them automatically.

所有数据均来自美国联邦公开数据源。原始数据文件因体积过大未包含在仓库中，运行 `python scripts/prepare_data.py --all` 可自动下载。

---

## Root-Cause Taxonomy / 根因分类体系

| Class | Description | 描述 |
|-------|-------------|------|
| **Epidemic** | True infection change with clinical corroboration | 有临床佐证的真实感染变化 |
| **Environmental** | Hydrometeorological distortion (rain, flow, temperature) | 水文气象干扰（降雨、流量、温度） |
| **Sampling** | Isolated measurement artefact | 孤立的测量伪影 |
| **Mixed** | Multiple concurrent mechanisms | 多种机制同时作用 |
| **Uncertain** | Insufficient or conflicting evidence | 证据不足或相互矛盾 |

---

## Ablation Study / 消融实验

Removing any single evidence source degrades performance, confirming complementary roles:

| Ablation | Macro-F1 | Δ vs Full |
|----------|----------|-----------|
| Full Agent | 0.767 | — |
| − Clinical | 0.115 | −0.652 |
| − All Tools | 0.256 | −0.511 |
| − Variants | 0.395 | −0.372 |
| − Weather | 0.410 | −0.357 |
| − Nearby Sites | 0.413 | −0.354 |
| − Hydrology | 0.418 | −0.349 |
| − Domain Knowledge | 0.427 | −0.340 |

---

## Citation / 引用

If you use this code or data in your research, please cite:

```bibtex
@article{wbeagent2026,
  title={National-scale agentic interpretation of wastewater SARS-CoV-2 anomalies},
  year={2026},
  note={Manuscript under review}
}
```

---

## License / 许可证

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE)。
