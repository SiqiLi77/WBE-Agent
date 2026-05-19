<div align="center">

# WBE-Agent

**National-scale interpretation of wastewater SARS-CoV-2 anomalies with a language model agent**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Next.js](https://img.shields.io/badge/Web_UI-Next.js_15-black.svg)](web/)

</div>

---

## Overview

Wastewater-based epidemiology can provide population-level early warning of infectious disease dynamics, but anomalous wastewater measurements are often difficult to interpret. Similar wastewater excursions can arise from epidemic change, hydrometeorological perturbation, sampling variability, or conflicting evidence across data streams.

**WBE-Agent** is a tool-augmented language model workflow for interpreting anomalous SARS-CoV-2 signals in wastewater surveillance. The workflow investigates each candidate anomaly by sequentially retrieving and integrating wastewater, weather, hydrology, hospitalization, variant-surveillance, nearby-site, and site-metadata evidence. It returns a structured classification, confidence score, reasoning trace, and data-gap inventory.

This repository supports the manuscript:

**National-scale interpretation of wastewater SARS-CoV-2 anomalies with a language model agent**

---

## Key Results

| Metric | WBE-Agent | Best machine-learning baseline | Zero-shot language model |
|--------|-----------|--------------------------------|---------------------------|
| Accuracy | **78.9%** | 65.3% | 13.1% |
| Macro-F1 | **0.767** | 0.507 | 0.139 |
| Cohen's kappa | **0.619** | 0.358 | 0.049 |
| Epidemic recall | **81.8%** | 72.7% | 22.7% |
| Environmental recall | **94.7%** | 63.2% | 10.5% |

The benchmark contains 199 candidate anomaly events with reference labels derived from independent human review across 16 U.S. states and five climate zones.

---

## Manuscript Data

The manuscript-ready event-level reference benchmark is provided at:

```text
data/human_review_benchmark_199.csv
```

Field definitions are provided at:

```text
data/README_BENCHMARK.md
```

Legacy exploratory silver-label files are not used as the final manuscript reference benchmark.

Source data underlying the figures are provided separately with the manuscript through the journal submission system.

---

## Highlights

- **Multi-source evidence integration**: wastewater, weather, hydrology, hospitalization, variant surveillance, nearby-site, and site-metadata evidence.
- **Traceable decision support**: each output includes a classification, confidence score, reasoning trace, and data-gap inventory.
- **Benchmark evaluation**: 199 candidate anomaly events with independent human-review reference labels.
- **Model portability**: 11 foundation-model configurations were evaluated to characterize the quality-cost frontier.
- **Operational triage**: the workflow is designed to help prioritize events for expert review, not to replace public-health judgment.

---

## Data Sources

All source datasets are publicly available from U.S. federal data systems:

| Source | Agency | Granularity | Key variables |
|--------|--------|-------------|---------------|
| [National Wastewater Surveillance System](https://www.cdc.gov/nwss/) | CDC | Site-day | SARS-CoV-2 wastewater concentration |
| [COVID-19 patient impact dataset](https://healthdata.gov/) | HHS | State-day | Adult confirmed COVID-19 admissions |
| [Global Historical Climatology Network Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) | NOAA | Station-day | Precipitation and temperature |
| [National Water Information System](https://waterservices.usgs.gov/) | USGS | Gauge-day | Streamflow discharge |
| [CDC regional variant surveillance summaries](https://www.cdc.gov/nwss/rv/COVID19-nationaltrend.html) | CDC | Region-week | SARS-CoV-2 lineage proportions |

Raw third-party data files are not redistributed in this repository because they are publicly available from the original sources and may be updated by the data providers.

---

## Taxonomy

WBE-Agent assigns each candidate event to one of five interpretive classes:

| Class | Description |
|-------|-------------|
| **Epidemic** | Wastewater anomaly with supporting epidemiological evidence |
| **Environmental** | Hydrometeorological perturbation or dilution is the primary explanation |
| **Sampling** | Isolated measurement or sampling artefact is the primary explanation |
| **Mixed** | Multiple mechanisms plausibly contribute |
| **Uncertain** | Evidence is insufficient or conflicting |

---

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                     WBE-Agent System                    │
├──────────────┬──────────────────────────────────────────┤
│  Public      │       Sequential investigation loop       │
│  data        │  ┌──────────┐  ┌──────────────────────┐  │
│  sources     │  │ System   │  │ Thought → Action     │  │
│              │  │ prompt   │→ │ → Observation        │  │
│  • NWSS      │  │ and      │  │ → Thought → ...      │  │
│  • HHS       │  │ taxonomy │  └──────────┬───────────┘  │
│  • NOAA      │  └──────────┘             │              │
│  • USGS      │                      Tool calls           │
│  • CDC       │         ┌─────────────────┤              │
│  variants    │         ▼                 ▼              │
│              │  ┌────────────┐  ┌──────────────┐       │
│              │  │ Weather    │  │ Hospitalization│      │
│              │  │ Hydrology  │  │ Variants       │      │
│              │  │ Nearby     │  │ Site metadata  │      │
│              │  │ sites      │  │                │      │
│              │  └────────────┘  └──────────────┘       │
├──────────────┴──────────────────────────────────────────┤
│  Structured output: classification, confidence,          │
│  reasoning trace, primary factors, and data gaps         │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```text
WBE-Agent/
├── config/                         # Configuration files
├── data/
│   ├── human_review_benchmark_199.csv
│   └── README_BENCHMARK.md
├── docs/                           # Method and workflow notes
├── outputs/                        # Processed outputs and evaluation summaries
├── scripts/                        # Command-line entry points
├── src/
│   ├── data_pipeline/              # Data ingestion, cleaning, and alignment
│   ├── anomaly_detection/          # Ensemble anomaly detection
│   ├── agent/                      # Tool-augmented investigation workflow
│   ├── evaluation/                 # Baselines and benchmark metrics
│   └── webapi/                     # API utilities
├── web/                            # Optional web dashboard
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/SiqiLi77/WBE-Agent.git
cd WBE-Agent

python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux or macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## Configuration

Set the model provider and API key in `config/settings.yaml`:

```yaml
agent:
  api_key: "your-api-key"
  api_base: "https://openrouter.ai/api/v1"
  model: "openai/gpt-4o-mini"
```

---

## Reproducing the Workflow

```bash
# Prepare public input data
python scripts/prepare_data.py --all

# Clean and align multi-source data
python scripts/run_pipeline.py --step all

# Detect anomaly events
python scripts/run_detection.py --min-votes 2

# Run agent investigations
python scripts/run_agent.py --max-events 10

# Evaluate benchmark performance
python scripts/run_evaluation.py --methods all
```

---

## Web Dashboard

```bash
# Terminal 1: backend API
python scripts/run_api.py

# Terminal 2: frontend
cd web
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{wbeagent2026,
  title={National-scale interpretation of wastewater SARS-CoV-2 anomalies with a language model agent},
  author={Li, Siqi and Li, Zhicheng and Li, Xuan and Wang, Qilin},
  year={2026},
  note={Manuscript under review}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
