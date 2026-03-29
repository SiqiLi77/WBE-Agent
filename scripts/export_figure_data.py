"""
Export CSV data + README for every panel in build_paper_figures.py.

Usage:
    python scripts/export_figure_data.py

Output:
    paper/generated/csv/fig{N}{panel}_{name}.csv
    paper/generated/csv/fig{N}{panel}_{name}_README.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "paper" / "generated" / "csv"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants (mirrored from build_paper_figures.py)
# ---------------------------------------------------------------------------
CLASS_ORDER = ["sampling", "mixed", "environmental", "epidemic", "uncertain"]
METHOD_ORDER = [
    "majority_class", "rule_based", "logistic_regression",
    "random_forest", "gradient_boosting",
    "zero_shot_llm", "few_shot_llm", "agent",
]
METHOD_NAMES = {
    "majority_class": "Majority class",
    "rule_based": "Rule-based",
    "logistic_regression": "Logistic reg.",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boost.",
    "zero_shot_llm": "Zero-shot LLM",
    "few_shot_llm": "Few-shot LLM",
    "agent": "WBE-Agent",
}
MODEL_DISPLAY_NAMES = {
    "default": "default / current",
    "v6_gpt4omini": "gpt-4o-mini (v6)",
    "qwen3_5_27b": "Qwen3.5-27B",
    "grok41_fast": "Grok 4.1 Fast",
    "haiku45": "Claude Haiku 4.5",
    "gpt54_nano": "GPT-5.4 Nano",
    "minimax47": "MiniMax 2.7",
    "kimi_k25": "Kimi K2.5",
    "deepseek32": "DeepSeek 3.2",
    "glm5": "GLM-5",
    "gemini31_flash": "Gemini 3.1 Flash",
    "gpt5mini": "GPT-5 Mini",
}
ABLATION_ORDER = [
    "no_tools", "no_variants", "no_clinical",
    "no_nearby_sites", "no_domain_knowledge", "no_weather", "no_hydrology",
]
ABLATION_NAMES = {
    "no_tools": "No tools",
    "no_variants": "- Variants",
    "no_clinical": "- Clinical",
    "no_nearby_sites": "- Nearby sites",
    "no_domain_knowledge": "- Domain know.",
    "no_weather": "- Weather",
    "no_hydrology": "- Hydrology",
}
CURRENT_ABLATION_DIR = PROJECT_ROOT / "outputs" / "ablation_199"
CURRENT_ABLATION_SUMMARY = CURRENT_ABLATION_DIR / "ablation_summary_canonical.csv"
CASE_EVENTS = [
    {"event_id": "EVT-00001", "state": "FL", "date": "2023-12-27"},
    {"event_id": "EVT-00002", "state": "FL", "date": "2024-03-12"},
    {"event_id": "EVT-00004", "state": "FL", "date": "2024-08-08"},
    {"event_id": "EVT-00078", "state": "NV", "date": "2022-06-20"},
]


def get_best_baseline_method(summary: pd.DataFrame) -> str:
    """Return the strongest non-agent baseline under the current benchmark."""
    baseline_rows = summary[summary["method"] != "agent"].copy()
    ranked = baseline_rows.sort_values(["macro_f1", "accuracy"], ascending=False)
    return str(ranked.iloc[0]["method"])


def save(df: pd.DataFrame, name: str, readme: str):
    df.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / f"{name}_README.txt").write_text(readme, encoding="utf-8")
    print(f"  {name}.csv  ({len(df)} rows)")


def save_text(name: str, text: str):
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"  {name}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all():
    inv = pd.read_csv(PROJECT_ROOT / "outputs" / "investigation_results.csv")
    auto = pd.read_csv(PROJECT_ROOT / "data" / "labeled" / "auto_labeled_events.csv")
    silver = pd.read_csv(PROJECT_ROOT / "data" / "labeled" / "labeled_events.csv")
    catalog = pd.read_csv(PROJECT_ROOT / "outputs" / "anomaly_event_catalog.csv")
    bench = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_summary.csv")
    ablation = pd.read_csv(PROJECT_ROOT / "outputs" / "ablation" / "ablation_summary.csv")
    return inv, auto, silver, catalog, bench, ablation


def load_sites():
    frames = []
    for fname, tier in [
        ("tier1_core_final_sites.csv", "core"),
        ("tier2_extension_sites.csv", "extension"),
        ("tier3_challenge_sites.csv", "challenge"),
    ]:
        p = PROJECT_ROOT / "outputs" / fname
        if p.exists():
            d = pd.read_csv(p)
            d["tier"] = tier
            if "date_span_days" in d.columns:
                d["site_days"] = d["date_span_days"] + 1
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_current_ablation_summary() -> pd.DataFrame:
    if CURRENT_ABLATION_SUMMARY.exists():
        return pd.read_csv(CURRENT_ABLATION_SUMMARY)
    return pd.read_csv(CURRENT_ABLATION_DIR / "ablation_summary.csv")


def load_current_ablation_prediction(inv: pd.DataFrame, ablation_name: str) -> pd.DataFrame:
    if ablation_name == "full":
        return inv[["event_id", "classification", "total_tokens", "tool_calls_count"]].copy()
    return pd.read_csv(CURRENT_ABLATION_DIR / f"{ablation_name}_investigation_results.csv")


def load_model_workload() -> pd.DataFrame:
    breakdown_path = PROJECT_ROOT / "outputs" / "evaluation" / "model_workload_breakdown.csv"
    summary_path = PROJECT_ROOT / "outputs" / "evaluation" / "model_comparison_summary.csv"
    if breakdown_path.exists():
        return pd.read_csv(breakdown_path)
    if not summary_path.exists():
        raise FileNotFoundError(
            "Missing outputs/evaluation/model_workload_breakdown.csv and "
            "outputs/evaluation/model_comparison_summary.csv. "
            "Run scripts/build_model_workload_figure.py first."
        )

    df = pd.read_csv(summary_path).copy()
    df["display_name"] = df["method"].map(MODEL_DISPLAY_NAMES).fillna(df["method"])
    df["total_tokens"] = df["avg_tokens"].fillna(0) * df["n_samples"]
    df["total_tool_calls"] = df["avg_tool_calls"].fillna(0) * df["n_samples"]
    df["completed_event_runs"] = df["prediction_rows"].fillna(df["n_samples"]).astype(int)
    df["missing_predictions"] = df["missing_predictions"].fillna(0).astype(int)
    df["coverage_pct"] = df["coverage"].fillna(0) * 100
    df["macro_f1_pct"] = df["macro_f1"] * 100
    df["accuracy_pct"] = df["accuracy"] * 100
    df["tokens_k"] = df["total_tokens"] / 1000
    df["tokens_m"] = df["total_tokens"] / 1_000_000
    df["tool_calls_k"] = df["total_tool_calls"] / 1000
    correct_events = (df["accuracy"].fillna(0) * df["n_samples"].replace(0, np.nan)).replace(0, np.nan)
    df["tokens_per_correct"] = df["total_tokens"] / correct_events
    df["tokens_per_correct_k"] = df["tokens_per_correct"] / 1000
    return df



# ===================================================================
# Fig 1 — Site map + data coverage
# ===================================================================
def export_fig1(sites: pd.DataFrame):
    # (a) Site locations
    cols = ["site_id", "state", "latitude", "longitude", "climate_bucket",
            "tier", "population_served"]
    available = [c for c in cols if c in sites.columns]
    save(sites[available], "fig1a_site_locations", """fig1a_site_locations.csv
=====================================
对应图表: Figure 1a — 美国研究站点地图
图表类型: 散点地图 (scatter on map)
数据来源: outputs/tier{1,2,3}_*.csv (站点元数据)

列说明:
- site_id: NWSS 站点唯一标识符 (CDC 编码)
- state: 美国州缩写 (如 FL=佛罗里达, IL=伊利诺伊)
- latitude: 纬度 (WGS84)
- longitude: 经度 (WGS84, 负值表示西经)
- climate_bucket: 气候分区
    continental_north = 大陆性北部 (MN, WI, NY)
    continental_central = 大陆性中部 (IL, PA)
    subtropical_south = 亚热带南部 (FL, GA, NC)
    marine_west = 海洋性西部 (OR)
    arid_semiarid = 干旱/半干旱 (NV, MT)
- tier: 站点层级
    core = 核心站点 (数据最完整, 用于主要分析)
    extension = 扩展站点 (补充地理覆盖)
    challenge = 挑战站点 (数据稀疏, 测试鲁棒性)
- population_served: 该污水处理厂服务人口数

Origin 建议: 用 XY Scatter 绘制, X=longitude, Y=latitude, 按 climate_bucket 分组着色, 按 tier 分组设置不同标记形状。叠加美国地图底图。
""")

    # (b) Data coverage
    cov_cols = ["state", "tier", "climate_bucket", "date_min", "date_max", "n_obs", "site_days"]
    if "site_days" not in sites.columns and {"date_min", "date_max"}.issubset(sites.columns):
        date_min = pd.to_datetime(sites["date_min"])
        date_max = pd.to_datetime(sites["date_max"])
        sites["site_days"] = (date_max - date_min).dt.days + 1
    available = [c for c in cov_cols if c in sites.columns]
    save(sites[available].sort_values("date_min"), "fig1b_data_coverage", """fig1b_data_coverage.csv
=====================================
Corresponding figure: Figure 1b - Coverage timeline
Chart type: horizontal bar / Gantt chart
Data source: outputs/tier{1,2,3}_*.csv

Columns
- state: two-letter state code
- tier: site tier (core/extension/challenge)
- climate_bucket: climate category
- date_min: first wastewater observation date
- date_max: last wastewater observation date
- n_obs: direct wastewater observations
- site_days: aligned daily panel rows from date_min to date_max, inclusive

Origin suggestion: use a horizontal bar chart with sites on Y, date on X, bars spanning date_min to date_max, colour by climate_bucket, and right-side annotations for obs and site_days.
""")


# ===================================================================
# Fig 3 — Investigation outcomes
# ===================================================================
def export_fig3(inv: pd.DataFrame, auto: pd.DataFrame):
    dist_order = ["sampling", "environmental", "epidemic", "uncertain", "mixed"]
    stack_order = ["sampling", "environmental", "epidemic", "mixed", "uncertain"]

    # (a) Final classification distribution
    counts = inv["classification"].value_counts().reindex(dist_order, fill_value=0)
    df_a = pd.DataFrame({
        "classification": dist_order,
        "count": counts.values,
        "percentage": (counts.values / counts.sum() * 100).round(1),
    })
    save(df_a, "fig3a_classification_dist", """fig3a_classification_dist.csv
=====================================
Corresponding figure: Figure 3a - Final agent classifications
Chart type: horizontal bar chart
Data source: outputs/investigation_results.csv

Columns
- classification: final agent class
- count: number of events in that class
- percentage: share of the 199 investigated events

Note: this panel summarizes the final decision-support labels after agent investigation.
""")

    # (b) Confidence by class
    df_b = inv[["event_id", "classification", "confidence"]].copy()
    save(df_b, "fig3b_confidence_by_class", """fig3b_confidence_by_class.csv
=====================================
Corresponding figure: Figure 3b - Confidence by predicted class
Chart type: violin plot with jittered points
Data source: outputs/investigation_results.csv

Columns
- event_id: event identifier
- classification: final agent class
- confidence: model-reported confidence score from 0 to 1

Note: this panel shows how the confidence distribution differs across final classes.
""")

    # (c) Weak auto-label redistribution
    merged = auto[["event_id", "auto_label"]].merge(
        inv[["event_id", "classification"]], on="event_id", how="inner"
    )
    preferred_auto_order = ["uncertain", "sampling", "epidemic", "environmental", "mixed"]
    auto_present = set(merged["auto_label"].dropna().unique())
    auto_order = [label for label in preferred_auto_order if label in auto_present]
    trans = pd.crosstab(merged["auto_label"], merged["classification"]).reindex(
        index=auto_order, columns=stack_order, fill_value=0
    )
    row_totals = trans.sum(axis=1)
    rows = []
    for auto_label in auto_order:
        total = int(row_totals.loc[auto_label])
        for final_label in stack_order:
            count = int(trans.loc[auto_label, final_label])
            rows.append({
                "auto_label": auto_label,
                "final_label": final_label,
                "count": count,
                "row_total": total,
                "row_pct": round((count / total * 100) if total else 0.0, 1),
            })
    df_c = pd.DataFrame(rows)
    save(df_c, "fig3c_transition_matrix", """fig3c_transition_matrix.csv
=====================================
Corresponding figure: Figure 3c - Redistribution of weak auto-labels
Chart type: row-normalized stacked horizontal bar chart
Data source: data/labeled/auto_labeled_events.csv + outputs/investigation_results.csv

Columns
- auto_label: weak initial label produced by the exploratory rule-based auto-labeler
- final_label: final agent class after investigation
- count: number of events moving from auto_label to final_label
- row_total: total events in that auto_label row
- row_pct: within-row percentage for that transition

Note: this panel is designed to show how evidence-seeking investigation changed the composition of each weak auto-label group.
""")

    # (d) Focused breakdown of uncertain auto-labels
    uncertain_total = int(row_totals.loc["uncertain"]) if "uncertain" in row_totals.index else 0
    rows = []
    if uncertain_total:
        for final_label in stack_order:
            count = int(trans.loc["uncertain", final_label])
            if count <= 0:
                continue
            rows.append({
                "final_label": final_label,
                "count": count,
                "share_pct": round(count / uncertain_total * 100, 1),
                "resolved": final_label != "uncertain",
                "total_uncertain": uncertain_total,
            })
    df_d = pd.DataFrame(rows)
    save(df_d, "fig3d_uncertain_resolution", """fig3d_uncertain_resolution.csv
=====================================
Corresponding figure: Figure 3d - Focused breakdown of weak uncertain auto-labels
Chart type: stacked horizontal bar chart
Data source: same as Figure 3c

Columns
- final_label: final agent class assigned to events that started as weak uncertain auto-labels
- count: number of events in that final class
- share_pct: percentage within the uncertain auto-label subset
- resolved: True when the event was reassigned to a specific explanation, False when it remained uncertain
- total_uncertain: total number of weak uncertain auto-label events

Note: this panel isolates the main ambiguity-resolution claim of Figure 3.
""")


# ===================================================================
# Fig 4 — Case studies (4 events × 3 time series)
# ===================================================================
def export_fig4():
    merged = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "merged_multisource.parquet")
    merged["date"] = pd.to_datetime(merged["date"])

    for i, case in enumerate(CASE_EVENTS):
        eid = case["event_id"]
        state = case["state"]
        event_date = pd.Timestamp(case["date"])
        window_start = event_date - pd.Timedelta(days=30)
        window_end = event_date + pd.Timedelta(days=30)

        mask = (
            (merged["state"] == state)
            & (merged["date"] >= window_start)
            & (merged["date"] <= window_end)
        )
        window = merged[mask].sort_values("date").copy()

        # Wastewater
        ww_cols = ["date", "pcr_conc_lin", "pcr_conc_lin_log1p"]
        available_ww = [c for c in ww_cols if c in window.columns]
        df_ww = window[available_ww].dropna(subset=["pcr_conc_lin"] if "pcr_conc_lin" in window.columns else []).copy()
        df_ww["event_date"] = case["date"]
        save(df_ww, f"fig4_{eid}_wastewater", f"""fig4_{eid}_wastewater.csv
=====================================
对应图表: Figure 4 第{i+1}行第1列 — {eid} 污水浓度时序
图表类型: 折线图 + 面积填充 (Line + Area)
数据来源: data/processed/merged_multisource.parquet
事件: {eid}, 州={state}, 异常日期={case['date']}
时间窗口: 异常日期前后各30天

列说明:
- date: 日期
- pcr_conc_lin: SARS-CoV-2 PCR 浓度 (gene copies/L, 线性尺度)
- pcr_conc_lin_log1p: log1p 变换后的浓度 (用于异常检测)
- event_date: 异常事件发生日期 (用于绘制垂直标记线)

Origin 建议: X=date, Y=pcr_conc_lin (或除以1e6显示为百万), 折线图, 在 event_date 处画垂直虚线标记异常点。
""")

        # Environmental
        env_cols = ["date", "precipitation_mm", "temp_avg_c", "temp_max_c", "temp_min_c",
                    "discharge_cfs", "discharge_percentile"]
        available_env = [c for c in env_cols if c in window.columns]
        df_env = window[available_env].copy()
        df_env["event_date"] = case["date"]
        save(df_env, f"fig4_{eid}_environment", f"""fig4_{eid}_environment.csv
=====================================
对应图表: Figure 4 第{i+1}行第2列 — {eid} 环境数据
图表类型: 双Y轴图 (条形图 + 折线图)
数据来源: data/processed/merged_multisource.parquet
事件: {eid}, 州={state}

列说明:
- date: 日期
- precipitation_mm: 日降水量 (mm), 来自 NOAA GHCN-Daily 最近气象站
- temp_avg_c: 日均气温 (°C)
- temp_max_c: 日最高气温 (°C)
- temp_min_c: 日最低气温 (°C)
- discharge_cfs: 日均河流流量 (立方英尺/秒), 来自 USGS 最近水文站
- discharge_percentile: 流量百分位 (0~1), 相对于该站历史分布
    >0.9 表示高流量, 有 CSO (合流制溢流) 风险
- event_date: 异常日期标记

Origin 建议: 左Y轴=precipitation_mm (条形图), 右Y轴=discharge_percentile×100 (折线图),
在 percentile=90 处画水平参考线, event_date 处画垂直虚线。
""")

        # Hospitalization
        hosp_col = "previous_day_admission_adult_covid_confirmed"
        hosp_cols = ["date", hosp_col, "admission_7d_avg"] if "admission_7d_avg" in window.columns else ["date", hosp_col]
        available_hosp = [c for c in hosp_cols if c in window.columns]
        df_hosp = window[available_hosp].drop_duplicates(subset=["date"]).copy()
        df_hosp["event_date"] = case["date"]
        save(df_hosp, f"fig4_{eid}_hospitalization", f"""fig4_{eid}_hospitalization.csv
=====================================
对应图表: Figure 4 第{i+1}行第3列 — {eid} 住院数据
图表类型: 折线图 + 面积填充
数据来源: data/processed/merged_multisource.parquet (原始来自 HHS)
事件: {eid}, 州={state}
注意: 住院数据为州级粒度, 非站点级

列说明:
- date: 日期
- previous_day_admission_adult_covid_confirmed: 前一日成人 COVID-19 确诊入院数
- admission_7d_avg: 7天滚动平均入院数 (如果可用)
- event_date: 异常日期标记

Origin 建议: X=date, Y=admissions, 折线图 + 浅色面积填充, event_date 处画垂直虚线。
趋势上升支持 epidemic 分类, 平稳/下降则不支持。
""")

    inv = pd.read_csv(PROJECT_ROOT / "outputs" / "investigation_results.csv")
    case_lookup = inv.set_index("event_id")
    rows = []
    for idx, case in enumerate(CASE_EVENTS, start=1):
        eid = case["event_id"]
        final_class = case_lookup.loc[eid, "classification"] if eid in case_lookup.index else ""
        rows.append({
            "panel_row": idx,
            "event_id": eid,
            "state": case["state"],
            "event_date": case["date"],
            "final_class": final_class,
            "wastewater_csv": f"fig4_{eid}_wastewater.csv",
            "environment_csv": f"fig4_{eid}_environment.csv",
            "hospitalization_csv": f"fig4_{eid}_hospitalization.csv",
        })
    df_index = pd.DataFrame(rows)
    save(df_index, "fig4_case_index", """fig4_case_index.csv
=====================================
Corresponding figure: Figure 4 - Representative investigations illustrating context-dependent interpretation
Chart type: Figure-level case index
Source: scripts/export_figure_data.py + outputs/investigation_results.csv

Columns
- panel_row: Row order in Figure 4
- event_id: Case event identifier
- state: State associated with the case event
- event_date: Anomaly date used as the panel centre
- final_class: Final WBE-Agent interpretation used in the manuscript
- wastewater_csv / environment_csv / hospitalization_csv: Panel-level source files for the three traces shown in that row

Note: use this index first, then open the three row-specific CSV files for the actual time series used in the plot.
""")
    save_text("fig4_DATA_GUIDE.txt", """Figure 4 data guide
=====================================
Figure 4 is a four-row case-study figure. Each row is one event, and each row is built from three CSV files:

- wastewater trace
- environmental trace
- hospitalization trace

Current manuscript cases:
- EVT-00001 (uncertain)
- EVT-00002 (sampling)
- EVT-00004 (environmental)
- EVT-00078 (epidemic)

Start with fig4_case_index.csv to see the current row order and file mapping.
Then open the row-level CSVs listed in the index.

Important:
- Some older fig4_* CSV files may remain in this folder from previous case selections.
- The current manuscript Figure 4 uses only the four events listed above.
""")



# ===================================================================
# Fig 5 — Benchmark comparison
# ===================================================================
def export_fig5(bench: pd.DataFrame, silver: pd.DataFrame):
    bench_sorted = bench.set_index("method").loc[METHOD_ORDER].reset_index()
    best_baseline = get_best_baseline_method(bench_sorted)
    best_baseline_display = METHOD_NAMES.get(best_baseline, best_baseline)
    eval_dir = PROJECT_ROOT / "outputs" / "evaluation"
    agent_pred = pd.read_csv(eval_dir / "agent_predictions.csv")
    baseline_pred = pd.read_csv(eval_dir / f"{best_baseline}_predictions.csv")
    merged = silver[["event_id", "ground_truth_label", "consensus_status", "agreement_ratio"]].merge(
        agent_pred[["event_id", "y_pred"]].rename(columns={"y_pred": "agent"}),
        on="event_id",
        how="left",
    ).merge(
        baseline_pred[["event_id", "y_pred"]].rename(columns={"y_pred": "baseline"}),
        on="event_id",
        how="left",
    )

    ranking = bench_sorted.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    ranking["method_display"] = ranking["method"].map(METHOD_NAMES)
    ranking["is_agent"] = ranking["method"] == "agent"
    ranking["is_best_baseline"] = ranking["method"] == best_baseline
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["delta_vs_best_baseline"] = ranking["macro_f1"] - float(
        ranking.loc[ranking["method"] == best_baseline, "macro_f1"].iloc[0]
    )
    save(ranking, "fig5a_benchmark_ranking", f"""fig5a_benchmark_ranking.csv
=====================================
Panel: Figure 5a benchmark ranking
Plot type: Horizontal ranking bars
Source: outputs/evaluation/evaluation_summary.csv

Columns:
- rank: Macro-F1 rank across all methods
- method / method_display: Method identifier and display name
- accuracy, macro_f1, weighted_f1, cohen_kappa: Headline benchmark metrics
- is_agent: True for WBE-Agent
- is_best_baseline: True for the strongest non-agent baseline
- delta_vs_best_baseline: Macro-F1 gap relative to {best_baseline_display}

Interpretation:
This panel is the hero benchmark ranking. WBE-Agent is highlighted against all
baselines, with {best_baseline_display} acting as the strongest non-agent
comparator.
""")

    recall_classes = ["epidemic", "environmental", "sampling", "uncertain", "mixed"]
    support = merged["ground_truth_label"].value_counts()
    agent_report = classification_report(
        merged["ground_truth_label"],
        merged["agent"],
        output_dict=True,
        zero_division=0,
    )
    baseline_report = classification_report(
        merged["ground_truth_label"],
        merged["baseline"],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls in recall_classes:
        agent_recall = float(agent_report.get(cls, {}).get("recall", 0.0))
        baseline_recall = float(baseline_report.get(cls, {}).get("recall", 0.0))
        rows.append({
            "class": cls,
            "support": int(support.get(cls, 0)),
            "agent_recall": round(agent_recall, 4),
            "baseline_recall": round(baseline_recall, 4),
            "delta_recall": round(agent_recall - baseline_recall, 4),
            "baseline_method": best_baseline,
            "baseline_display": best_baseline_display,
        })
    df_b = pd.DataFrame(rows)
    save(df_b, "fig5b_per_class_recall", f"""fig5b_per_class_recall.csv
=====================================
Panel: Figure 5b per-class recall
Plot type: Dumbbell comparison
Source: outputs/evaluation/agent_predictions.csv, outputs/evaluation/{best_baseline}_predictions.csv, data/labeled/labeled_events.csv

Columns:
- class: Reference class
- support: Number of benchmark events in that class
- agent_recall: WBE-Agent recall
- baseline_recall: Recall for {best_baseline_display}
- delta_recall: Agent minus baseline recall
- baseline_method / baseline_display: Strongest baseline metadata

Interpretation:
Positive delta_recall values mark the classes where WBE-Agent retrieves more
reference events than the strongest baseline.
""")

    subset_frames = [
        ("All", merged),
        ("Accepted sampling", merged[(merged["consensus_status"] == "accepted") & (merged["ground_truth_label"] == "sampling")]),
        ("Epidemic + environmental", merged[merged["ground_truth_label"].isin(["epidemic", "environmental"])]),
        ("Disagreement", merged[merged["consensus_status"] == "disagreement"]),
    ]
    rows = []
    for label, df in subset_frames:
        n = len(df)
        agent_acc = float((df["agent"] == df["ground_truth_label"]).mean()) if n > 0 else 0.0
        baseline_acc = float((df["baseline"] == df["ground_truth_label"]).mean()) if n > 0 else 0.0
        rows.append({
            "subset": label,
            "n_events": n,
            "agent_accuracy": round(agent_acc, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "delta_accuracy": round(agent_acc - baseline_acc, 4),
            "baseline_method": best_baseline,
            "baseline_display": best_baseline_display,
        })
    df_c = pd.DataFrame(rows)
    save(df_c, "fig5c_subset_accuracy", f"""fig5c_subset_accuracy.csv
=====================================
Panel: Figure 5c operational subsets
Plot type: Dumbbell comparison
Source: outputs/evaluation/agent_predictions.csv, outputs/evaluation/{best_baseline}_predictions.csv, data/labeled/labeled_events.csv

Columns:
- subset: Operational subset shown in the panel
- n_events: Number of benchmark events in that subset
- agent_accuracy: Accuracy of WBE-Agent on the subset
- baseline_accuracy: Accuracy of {best_baseline_display} on the subset
- delta_accuracy: Agent minus baseline accuracy
- baseline_method / baseline_display: Strongest baseline metadata

Interpretation:
This panel shows whether the agent's advantage is concentrated in the subsets
that matter most operationally, rather than only in the overall average.
""")

    status_order = ["accepted", "majority", "disagreement"]
    status_labels = {
        "accepted": "Full agreement",
        "majority": "Majority agreement",
        "disagreement": "Persistent disagreement",
    }
    status_counts = silver["consensus_status"].value_counts().reindex(status_order, fill_value=0)
    total_events = int(status_counts.sum())
    mean_agreement = float(silver["agreement_ratio"].dropna().mean())
    df_d = pd.DataFrame({
        "consensus_status": status_order,
        "status_display": [status_labels[s] for s in status_order],
        "count": [int(status_counts[s]) for s in status_order],
        "fraction": [round(float(status_counts[s] / total_events), 4) for s in status_order],
        "mean_agreement_ratio": [round(mean_agreement, 4)] * len(status_order),
        "total_events": [total_events] * len(status_order),
    })
    save(df_d, "fig5d_expert_agreement_tiers", """fig5d_expert_agreement_tiers.csv
=====================================
Panel: Figure 5d expert agreement tiers
Plot type: Stacked benchmark-reliability bar
Source: data/labeled/labeled_events.csv

Columns:
- consensus_status: Agreement tier recorded for each benchmark event
- status_display: Display label used in the figure
- count: Number of events in that tier
- fraction: Share of the benchmark in that tier
- mean_agreement_ratio: Mean agreement ratio across all benchmark events
- total_events: Total number of benchmark events

Interpretation:
This panel summarizes benchmark reliability. The agreement tiers are shown as
counts and fractions, with the mean inter-expert agreement ratio reported as a
single benchmark-level summary.
""")
    panel_index = pd.DataFrame([
        {
            "panel": "a",
            "csv_file": "fig5a_benchmark_ranking.csv",
            "purpose": "Hero benchmark ranking across all eight methods",
        },
        {
            "panel": "b",
            "csv_file": "fig5b_per_class_recall.csv",
            "purpose": "Per-class recall comparison between WBE-Agent and the strongest baseline",
        },
        {
            "panel": "c",
            "csv_file": "fig5c_subset_accuracy.csv",
            "purpose": "Accuracy comparison on operationally important subsets",
        },
        {
            "panel": "d",
            "csv_file": "fig5d_expert_agreement_tiers.csv",
            "purpose": "Agreement tiers for the expert reference benchmark",
        },
    ])
    save(panel_index, "fig5_panel_index", """fig5_panel_index.csv
=====================================
Corresponding figure: Figure 5 - Performance comparison on the expert reference benchmark
Chart type: Figure-level panel index

Columns
- panel: Panel letter in the manuscript
- csv_file: CSV file containing the plotted data for that panel
- purpose: One-line description of the analytical role of the panel

Note: use this file to identify which CSV belongs to each panel in the current Figure 5 layout.
""")
    save_text("fig5_DATA_GUIDE.txt", """Figure 5 data guide
=====================================
Figure 5 is the main benchmark figure in the manuscript.
The current panel-to-data mapping is:

- a -> fig5a_benchmark_ranking.csv
- b -> fig5b_per_class_recall.csv
- c -> fig5c_subset_accuracy.csv
- d -> fig5d_expert_agreement_tiers.csv

Legacy CSVs from earlier Figure 5 designs may still exist in this folder.
The current manuscript uses the four files listed above.
""")


def export_fig6(bench: pd.DataFrame, ablation: pd.DataFrame, silver: pd.DataFrame):
    bench_idx = bench.set_index("method")
    full_macro = float(bench_idx.loc["agent", "macro_f1"])
    abl = ablation.set_index("ablation").loc[ABLATION_ORDER].reset_index()
    abl["delta_f1"] = abl["macro_f1"] - full_macro

    # (a) Waterfall
    rows = [{"variant": "Full Agent", "macro_f1": full_macro, "delta_f1": 0.0}]
    for _, r in abl.iterrows():
        rows.append({
            "variant": ABLATION_NAMES.get(r["ablation"], r["ablation"]),
            "macro_f1": r["macro_f1"],
            "delta_f1": r["delta_f1"],
        })
    df_a = pd.DataFrame(rows)
    save(df_a, "fig6a_ablation_waterfall", """fig6a_ablation_waterfall.csv
=====================================
对应图表: Figure 6a — 消融实验瀑布图
图表类型: 瀑布图 / 条形图 (Waterfall / Bar chart)
数据来源: outputs/ablation/ablation_summary.csv + outputs/evaluation/evaluation_summary.csv

列说明:
- variant: 消融变体名称
    Full Agent = 完整 Agent (基线, 所有工具+领域知识)
    No tools = 移除所有工具 (Agent 只能靠推理)
    - Variants = 移除变异株查询工具
    - Clinical = 移除住院数据查询工具
    - Nearby sites = 移除周边站点查询工具
    - Domain know. = 移除领域知识 (prompt 中的 WBE 专家规则)
    - Weather = 移除气象查询工具
    - Hydrology = 移除水文查询工具
- macro_f1: 该变体的 Macro-F1 分数
- delta_f1: 相对于 Full Agent 的 F1 变化量 (负值=性能下降)

含义: delta_f1 绝对值越大, 说明该组件对 Agent 性能越重要。

Origin 建议: 条形图, X轴=variant, Y轴=macro_f1, 在每个条形上方标注 delta_f1。
Full Agent 用深色, 其他按 delta 严重程度着色 (红>黄>灰)。画 Full Agent 水平参考线。
""")

    # (b) Per-class recall delta heatmap
    recall_classes = ["environmental", "mixed", "sampling", "uncertain"]
    agent_pred = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "agent_predictions.csv")
    m_full = silver[["event_id", "ground_truth_label"]].merge(
        agent_pred[["event_id", "y_pred"]], on="event_id", how="left"
    )
    full_report = classification_report(m_full["ground_truth_label"], m_full["y_pred"],
                                         output_dict=True, zero_division=0)
    rows = []
    for abl_name in ABLATION_ORDER:
        pred_path = PROJECT_ROOT / "outputs" / "ablation" / f"{abl_name}_investigation_results.csv"
        if not pred_path.exists():
            continue
        pred = pd.read_csv(pred_path)
        m_abl = silver[["event_id", "ground_truth_label"]].merge(
            pred[["event_id", "classification"]], on="event_id", how="inner"
        )
        rep = classification_report(m_abl["ground_truth_label"], m_abl["classification"],
                                     output_dict=True, zero_division=0)
        row = {"ablation": abl_name, "ablation_display": ABLATION_NAMES.get(abl_name, abl_name)}
        for cls in recall_classes:
            full_recall = full_report.get(cls, {}).get("recall", 0.0)
            abl_recall = rep.get(cls, {}).get("recall", 0.0)
            row[f"{cls}_recall_delta"] = round(abl_recall - full_recall, 4)
            row[f"{cls}_recall_absolute"] = round(abl_recall, 4)
        rows.append(row)
    df_b = pd.DataFrame(rows)
    save(df_b, "fig6b_recall_delta_heatmap", """fig6b_recall_delta_heatmap.csv
=====================================
对应图表: Figure 6b — 分类别 Recall 变化热力图
图表类型: 热力图 (Heatmap), 发散色标 (红=下降, 蓝=上升)
数据来源: outputs/ablation/*_investigation_results.csv + outputs/evaluation/agent_predictions.csv

列说明:
- ablation: 消融变体内部名
- ablation_display: 显示名
- {class}_recall_delta: 该消融变体相对于 Full Agent 的 Recall 变化
    正值 = 该变体在此类别上 recall 反而更高 (移除该工具后偶然改善)
    负值 = recall 下降 (该工具对识别此类异常有帮助)
- {class}_recall_absolute: 该消融变体的绝对 Recall 值

Recall = 真正例/(真正例+假反例), 衡量"该类别的事件有多少被正确识别出来"。

Origin 建议: 矩阵热力图, 行=消融变体, 列=类别, 值=recall_delta, 用红白蓝发散色标。
""")

    # (c) Token & tool-call cost
    full_tokens = float(bench_idx.loc["agent", "avg_tokens"])
    full_tools = float(bench_idx.loc["agent", "avg_tool_calls"])
    rows = [{"variant": "Full Agent", "avg_tokens": full_tokens, "avg_tool_calls": full_tools}]
    for _, r in abl.iterrows():
        rows.append({
            "variant": ABLATION_NAMES.get(r["ablation"], r["ablation"]),
            "avg_tokens": r["avg_tokens"],
            "avg_tool_calls": r["avg_tool_calls"],
        })
    df_c = pd.DataFrame(rows)
    save(df_c, "fig6c_resource_usage", """fig6c_resource_usage.csv
=====================================
对应图表: Figure 6c — 资源消耗对比 (Token + 工具调用)
图表类型: 水平条形图 + 散点 (双指标)
数据来源: outputs/ablation/ablation_summary.csv

列说明:
- variant: 消融变体名称
- avg_tokens: 平均每个事件消耗的 Token 数 (LLM 输入+输出)
    Token 数直接关系到 API 调用成本
- avg_tool_calls: 平均每个事件的工具调用次数
    工具调用次数反映 Agent 的调查深度

Origin 建议: 水平双指标图, 条形=tokens (浅色), 散点=tool_calls (深色), Y轴=variant。
""")


def export_fig6_current_ablation(inv: pd.DataFrame, auto: pd.DataFrame, silver: pd.DataFrame):
    summary = load_current_ablation_summary().copy()
    order = ["full"] + (
        summary.loc[summary["ablation"] != "full"]
        .sort_values(["macro_f1", "accuracy"], ascending=False)["ablation"]
        .tolist()
    )
    summary = summary.set_index("ablation").loc[order].reset_index()
    display_map = {"full": "Full agent", **ABLATION_NAMES}
    summary["display_name"] = summary["ablation"].map(display_map)
    full_macro = float(summary.loc[summary["ablation"] == "full", "macro_f1"].iloc[0])
    summary["delta_macro_f1"] = summary["macro_f1"] - full_macro

    df_a = summary[[
        "ablation", "display_name", "n_samples", "accuracy", "macro_f1",
        "weighted_f1", "cohen_kappa", "avg_tokens", "avg_tool_calls", "delta_macro_f1",
    ]].copy()
    save(df_a, "fig6a_overall_degradation", """fig6a_overall_degradation.csv
=====================================
Corresponding figure: Figure 6a - Overall ablation degradation
Chart type: Horizontal lollipop ranking
Data source: outputs/ablation_199/ablation_summary_canonical.csv

Columns
- ablation: Internal ablation name
- display_name: Label shown in the figure
- n_samples: Evaluated events
- accuracy / macro_f1 / weighted_f1 / cohen_kappa: Summary metrics for that ablation
- avg_tokens / avg_tool_calls: Mean investigation cost per event
- delta_macro_f1: Macro-F1 change relative to the canonical full agent

Note: the full row is anchored to the canonical 199-event benchmark used in the manuscript.
""")

    class_order = ["sampling", "environmental", "epidemic", "uncertain", "mixed"]
    full_pred = load_current_ablation_prediction(inv, "full")
    full_merge = silver[["event_id", "ground_truth_label"]].merge(
        full_pred[["event_id", "classification"]], on="event_id", how="inner"
    )
    full_report = classification_report(
        full_merge["ground_truth_label"], full_merge["classification"],
        output_dict=True, zero_division=0,
    )
    rows = []
    for abl_name in order[1:]:
        pred = load_current_ablation_prediction(inv, abl_name)
        merged = silver[["event_id", "ground_truth_label"]].merge(
            pred[["event_id", "classification"]], on="event_id", how="inner"
        )
        rep = classification_report(
            merged["ground_truth_label"], merged["classification"],
            output_dict=True, zero_division=0,
        )
        row = {"ablation": abl_name, "display_name": display_map.get(abl_name, abl_name)}
        for cls in class_order:
            full_recall = full_report.get(cls, {}).get("recall", 0.0)
            abl_recall = rep.get(cls, {}).get("recall", 0.0)
            row[f"{cls}_recall"] = round(abl_recall, 4)
            row[f"{cls}_delta"] = round(abl_recall - full_recall, 4)
        rows.append(row)
    df_b = pd.DataFrame(rows)
    save(df_b, "fig6b_recall_loss_matrix", """fig6b_recall_loss_matrix.csv
=====================================
Corresponding figure: Figure 6b - Class-specific recall loss
Chart type: Heatmap
Data source: outputs/ablation_199/*_investigation_results.csv + outputs/investigation_results.csv + data/labeled/labeled_events.csv

Columns
- ablation / display_name: Ablation identifier and display label
- {class}_recall: Absolute recall under that ablation
- {class}_delta: Recall change relative to the canonical full workflow

Note: negative deltas indicate recall loss after removing that evidence stream.
""")

    uncertain_ids = auto.loc[auto["auto_label"] == "uncertain", "event_id"].drop_duplicates()
    stack_order = ["sampling", "environmental", "epidemic", "mixed", "uncertain"]
    rows = []
    for abl_name in order:
        pred = load_current_ablation_prediction(inv, abl_name)
        merged = uncertain_ids.to_frame().merge(
            pred[["event_id", "classification"]], on="event_id", how="left"
        )
        counts = merged["classification"].value_counts()
        row = {
            "ablation": abl_name,
            "display_name": display_map.get(abl_name, abl_name),
            "n_weak_uncertain": int(len(uncertain_ids)),
            "resolved_count": int(len(uncertain_ids) - counts.get("uncertain", 0)),
        }
        for cls in stack_order:
            row[f"{cls}_count"] = int(counts.get(cls, 0))
            row[f"{cls}_pct"] = round(counts.get(cls, 0) / len(uncertain_ids) * 100, 2) if len(uncertain_ids) else 0.0
        rows.append(row)
    df_c = pd.DataFrame(rows)
    save(df_c, "fig6c_weak_uncertain_resolution", """fig6c_weak_uncertain_resolution.csv
=====================================
Corresponding figure: Figure 6c - Redistribution of weak uncertain alerts
Chart type: Stacked horizontal bars
Data source: data/labeled/auto_labeled_events.csv + outputs/ablation_199/*_investigation_results.csv + outputs/investigation_results.csv

Columns
- ablation / display_name: Ablation identifier and display label
- n_weak_uncertain: Number of events that started with weak auto-label = uncertain
- resolved_count: Number of those events reassigned to a more specific class
- {class}_count: Count ending in each final class
- {class}_pct: Percentage ending in each final class

Note: this panel shows how each evidence stream affects ambiguity resolution rather than only headline accuracy.
""")

    df_d = summary[[
        "ablation", "display_name", "avg_tokens", "avg_tool_calls", "accuracy", "macro_f1"
    ]].copy()
    df_d["tokens_k"] = (df_d["avg_tokens"] / 1000).round(2)
    save(df_d, "fig6d_resource_profile", """fig6d_resource_profile.csv
=====================================
Corresponding figure: Figure 6d - Investigation cost profile under ablation
Chart type: Horizontal bars with overlaid tool-call points
Data source: outputs/ablation_199/ablation_summary_canonical.csv

Columns
- ablation / display_name: Ablation identifier and display label
- avg_tokens: Average tokens per event
- avg_tool_calls: Average tool calls per event
- tokens_k: Average tokens per event in thousands
- accuracy / macro_f1: Included for joint reading with cost

Note: use this panel to interpret whether lower performance coincides with shallower or more expensive investigations.
""")

    panel_index = pd.DataFrame([
        {"panel": "a", "csv_file": "fig6a_overall_degradation.csv", "purpose": "Overall macro-F1 degradation relative to the canonical full workflow"},
        {"panel": "b", "csv_file": "fig6b_recall_loss_matrix.csv", "purpose": "Class-specific recall losses under each ablation"},
        {"panel": "c", "csv_file": "fig6c_weak_uncertain_resolution.csv", "purpose": "How weak uncertain alerts are redistributed under each ablation"},
        {"panel": "d", "csv_file": "fig6d_resource_profile.csv", "purpose": "Average token and tool-call burden for each ablation"},
    ])
    save(panel_index, "fig6_panel_index", """fig6_panel_index.csv
=====================================
Corresponding figure: Figure 6 - Ablation analysis of the agentic investigation workflow
Chart type: Figure-level panel index

Columns
- panel: Panel letter in the manuscript
- csv_file: CSV file containing the exported data
- purpose: One-line description of the panel

Note: use this index to navigate the current Figure 6 data files.
""")
    save_text("fig6_DATA_GUIDE.txt", """Figure 6 data guide
=====================================
Figure 6 is the current ablation figure in the manuscript.

Current panel mapping:
- a -> fig6a_overall_degradation.csv
- b -> fig6b_recall_loss_matrix.csv
- c -> fig6c_weak_uncertain_resolution.csv
- d -> fig6d_resource_profile.csv

These files correspond to the current Figure 6 in paper/main.tex.
Older Figure 6 exports from previous manuscript structures may still exist in the folder or under legacy/.
""")


def export_fig6_model_workload():
    workload = load_model_workload().copy()
    workload["display_name"] = workload["display_name"].fillna(workload["method"])

    summary_rows = pd.DataFrame([
        {
            "metric": "models_compared",
            "value": int(len(workload)),
            "display_value": f"{len(workload)}",
            "note": f"{int((workload['coverage'] >= 0.999).sum())} with full 199-event coverage",
        },
        {
            "metric": "completed_event_runs",
            "value": int(workload["completed_event_runs"].sum()),
            "display_value": f"{int(workload['completed_event_runs'].sum()):,}",
            "note": f"{int(workload['n_samples'].max())} labeled events evaluated per model",
        },
        {
            "metric": "total_tokens",
            "value": float(workload["total_tokens"].sum()),
            "display_value": f"{workload['total_tokens'].sum() / 1_000_000:.2f}M",
            "note": "Across all saved multi-model investigations",
        },
        {
            "metric": "total_tool_calls",
            "value": float(workload["total_tool_calls"].sum()),
            "display_value": f"{workload['total_tool_calls'].sum():,.0f}",
            "note": "Agent tool invocations across all model runs",
        },
    ])
    save(summary_rows, "fig7_summary_cards", """fig7_summary_cards.csv
=====================================
Corresponding figure: Figure 7 summary cards
Chart type: Figure-level summary cards
Source: outputs/evaluation/model_workload_breakdown.csv

Columns
- metric: Identifier for the summary card
- value: Raw numeric value
- display_value: Formatted value shown in the figure
- note: Subtitle shown below the main card value

Note: these rows correspond to the four cards shown above panels a-d in Figure 7.
""")

    df_a = workload[[
        "method", "display_name", "n_samples", "prediction_rows", "coverage", "coverage_pct",
        "accuracy", "accuracy_pct", "macro_f1", "macro_f1_pct",
        "total_tokens", "tokens_m", "total_tool_calls",
    ]].sort_values(["macro_f1", "accuracy"], ascending=[False, False])
    save(df_a, "fig7a_performance_vs_total_workload", """fig7a_performance_vs_total_workload.csv
=====================================
Corresponding figure: Figure 7a - Performance versus total workload
Chart type: Bubble scatter plot
Source: outputs/evaluation/model_workload_breakdown.csv

Columns
- method / display_name: Model identifier and display label
- n_samples / prediction_rows: Benchmark size and completed predictions
- coverage / coverage_pct: Completed prediction coverage
- accuracy / accuracy_pct: Accuracy on the benchmark
- macro_f1 / macro_f1_pct: Macro-F1 on the benchmark
- total_tokens / tokens_m: Total token consumption across the run
- total_tool_calls: Total tool invocations across the run

Plot mapping
- x-axis: tokens_m
- y-axis: macro_f1_pct
- bubble size: total_tool_calls

Note: this is the main quality-versus-workload frontier for the current Figure 7.
""")

    df_b = workload[[
        "method", "display_name", "total_tokens", "tokens_k",
        "total_tool_calls", "tool_calls_k", "coverage", "coverage_pct",
    ]].sort_values("total_tokens", ascending=True)
    save(df_b, "fig7b_token_budget_by_model", """fig7b_token_budget_by_model.csv
=====================================
Corresponding figure: Figure 7b - Absolute token budget by model
Chart type: Horizontal bar chart
Source: outputs/evaluation/model_workload_breakdown.csv

Columns
- method / display_name: Model identifier and display label
- total_tokens / tokens_k: Total token budget for the full benchmark run
- total_tool_calls / tool_calls_k: Total tool invocations for the run
- coverage / coverage_pct: Completed prediction coverage

Note: labels drawn after each bar report tool calls and benchmark coverage.
""")

    df_c = workload[[
        "method", "display_name", "tokens_per_correct", "tokens_per_correct_k",
        "accuracy", "accuracy_pct", "coverage", "coverage_pct", "missing_predictions",
    ]].sort_values("tokens_per_correct", ascending=True)
    save(df_c, "fig7c_tokens_per_correct", """fig7c_tokens_per_correct.csv
=====================================
Corresponding figure: Figure 7c - Tokens per correct benchmark decision
Chart type: Horizontal bar chart
Source: outputs/evaluation/model_workload_breakdown.csv

Columns
- method / display_name: Model identifier and display label
- tokens_per_correct / tokens_per_correct_k: Token burden per correct labeled event
- accuracy / accuracy_pct: Accuracy on the benchmark
- coverage / coverage_pct: Prediction coverage
- missing_predictions: Number of missing benchmark predictions

Note: this panel folds both quality and efficiency into one metric. Lower values are better.
""")

    rank_cols = [
        "method", "display_name", "accuracy", "macro_f1", "cohen_kappa",
        "coverage", "coverage_pct", "avg_tokens", "avg_tool_calls",
    ]
    df_d = workload[rank_cols].sort_values(["macro_f1", "accuracy"], ascending=[False, False]).head(5)
    save(df_d, "fig7d_top_models_table", """fig7d_top_models_table.csv
=====================================
Corresponding figure: Figure 7d - Top-performing models table
Chart type: Tabular summary embedded in the figure
Source: outputs/evaluation/model_workload_breakdown.csv

Columns
- method / display_name: Model identifier and display label
- accuracy / macro_f1 / cohen_kappa: Headline benchmark metrics
- coverage / coverage_pct: Prediction coverage on the 199-event benchmark
- avg_tokens / avg_tool_calls: Mean investigation cost per event

Note: this table captures the top-performing region of the quality-cost frontier.
""")

    panel_index = pd.DataFrame([
        {"panel": "summary cards", "csv_file": "fig7_summary_cards.csv", "purpose": "Figure-level workload summary cards"},
        {"panel": "a", "csv_file": "fig7a_performance_vs_total_workload.csv", "purpose": "Performance versus total token workload scatter"},
        {"panel": "b", "csv_file": "fig7b_token_budget_by_model.csv", "purpose": "Absolute total token budget by model"},
        {"panel": "c", "csv_file": "fig7c_tokens_per_correct.csv", "purpose": "Efficiency after accounting for correctness"},
        {"panel": "d", "csv_file": "fig7d_top_models_table.csv", "purpose": "Top-performing models table used in the right-side panel"},
    ])
    save(panel_index, "fig7_panel_index", """fig7_panel_index.csv
=====================================
Corresponding figure: Figure 7 - Robustness and workload across foundation-model substitutions
Chart type: Figure-level panel index

Columns
- panel: Panel identifier in the current manuscript layout
- csv_file: CSV file containing the plotted or tabulated data
- purpose: One-line description of the panel

Note: Figure 7 also includes four workload summary cards. Those are exported in fig7_summary_cards.csv.
""")
    save_text("fig7_DATA_GUIDE.txt", """Figure 7 data guide
=====================================
Figure 7 is the current multi-model robustness and workload figure in the manuscript.

Current panel mapping:
- summary cards -> fig7_summary_cards.csv
- a -> fig7a_performance_vs_total_workload.csv
- b -> fig7b_token_budget_by_model.csv
- c -> fig7c_tokens_per_correct.csv
- d -> fig7d_top_models_table.csv

These files correspond to the current Figure 7 in paper/main.tex.
Older Figure 7 exports from previous manuscript structures may still exist in this folder or under legacy/.
""")


# ===================================================================
# Fig 7 ??Agent behavioral analysis
# ===================================================================
def export_fig7(inv: pd.DataFrame, catalog: pd.DataFrame):
    factor_map = {
        "isolated_spike": "Isolated spike",
        "heavy_rainfall": "Heavy rainfall",
        "variant_emergence": "Variant emergence",
        "high_flow_cso_risk": "High flow / CSO",
        "hospitalization_increase": "Hosp. increase",
        "stable_hospitalization": "Stable hosp.",
        "stable_hospitalizations": "Stable hosp.",
        "stable_hospitalization_trend": "Stable hosp.",
        "hospitalization_trend": "Hosp. trend",
        "increasing_hospitalizations": "Hosp. increase",
        "decreasing_hospitalizations": "Hosp. decrease",
        "isolated_signal": "Isolated spike",
        "low_precipitation": "Low precipitation",
        "discharge_percentile": "Discharge pctl.",
        "epidemic_signal": "Epidemic signal",
    }

    def parse_factors(summary):
        if pd.isna(summary):
            return []
        m = re.search(r"factors=([^)]*)\)", str(summary))
        if not m:
            return []
        raw = m.group(1).strip()
        return [f.strip() for f in raw.split(",") if f.strip()]

    # (a) Factor frequency
    all_factors = {}
    for _, row in inv.iterrows():
        for f in parse_factors(row.get("summary", "")):
            canonical = factor_map.get(f, f.replace("_", " ").title())
            all_factors[canonical] = all_factors.get(canonical, 0) + 1

    sorted_factors = sorted(all_factors.items(), key=lambda x: x[1], reverse=True)
    df_a = pd.DataFrame(sorted_factors, columns=["factor", "frequency"])
    save(df_a, "fig7a_factor_frequency", """fig7a_factor_frequency.csv
=====================================
对应图表: Figure 7a — Agent 识别的主要因素频率 (水平条形图)
图表类型: 水平条形图 (Horizontal bar chart)
数据来源: outputs/investigation_results.csv (从 summary 字段解析)

列说明:
- factor: Agent 在调查报告中识别的主要因素名称
    Isolated spike = 孤立尖峰 (单点异常, 前后正常 → 采样问题)
    Heavy rainfall = 强降雨 (稀释污水浓度 → 环境因素)
    Variant emergence = 变异株出现 (新变异株替代 → 可能影响信号)
    High flow / CSO = 高流量/合流制溢流 (暴雨导致未处理污水溢出)
    Hosp. increase = 住院数上升 (支持疫情信号)
    Stable hosp. = 住院数稳定 (不支持疫情信号)
    Hosp. decrease = 住院数下降
    Epidemic signal = 疫情信号
    Low precipitation = 低降水 (排除降雨干扰)
    Discharge pctl. = 流量百分位异常
- frequency: ???? 199 ????????????

Origin 建议: 水平条形图, 按频率降序排列, 按因素类别着色 (蓝=采样, 绿=环境, 红=临床, 黄=变异株)。
""")

    # (b) Classification by state
    def extract_state(site_id):
        parts = str(site_id).split("_")
        for p in parts:
            if len(p) == 2 and p.isalpha():
                return p.upper()
        return "??"

    inv_copy = inv.copy()
    inv_copy["state"] = inv_copy["site_id"].apply(extract_state)
    state_class = pd.crosstab(inv_copy["state"], inv_copy["classification"]).reindex(
        columns=CLASS_ORDER, fill_value=0
    )
    state_class["total"] = state_class.sum(axis=1)
    df_b = state_class.sort_values("total", ascending=False).reset_index()
    save(df_b, "fig7b_classification_by_state", """fig7b_classification_by_state.csv
=====================================
对应图表: Figure 7b — 按州分类分布 (堆叠水平条形图)
图表类型: 堆叠水平条形图 (Stacked horizontal bar)
数据来源: outputs/investigation_results.csv

列说明:
- state: 美国州缩写
- sampling/mixed/environmental/epidemic/uncertain: 该州各分类的事件数
- total: 该州总事件数

Origin 建议: 水平堆叠条形图, Y轴=state (按 total 降序), 每段按分类着色。
""")

    # (c) Temporal distribution
    inv_copy["peak_dt"] = pd.to_datetime(inv_copy["anomaly_date"])
    inv_copy["quarter"] = inv_copy["peak_dt"].dt.to_period("Q").astype(str)
    temporal = pd.crosstab(inv_copy["quarter"], inv_copy["classification"]).reindex(
        columns=CLASS_ORDER, fill_value=0
    )
    temporal["total"] = temporal.sum(axis=1)
    df_c = temporal.reset_index()
    save(df_c, "fig7c_temporal_distribution", """fig7c_temporal_distribution.csv
=====================================
对应图表: Figure 7c — 事件时间分布 (按季度堆叠条形图)
图表类型: 堆叠条形图 (Stacked bar chart)
数据来源: outputs/investigation_results.csv

列说明:
- quarter: 季度 (如 2022Q1, 2023Q4)
- sampling/mixed/environmental/epidemic/uncertain: 该季度各分类的事件数
- total: 该季度总事件数

Origin 建议: X轴=quarter, Y轴=事件数, 堆叠条形图按分类着色。可观察异常事件的季节性模式。
""")



# ===================================================================
# Fig 8 — Label agreement & agent strengths
# ===================================================================
def export_fig8(inv: pd.DataFrame, silver: pd.DataFrame):
    best_baseline = get_best_baseline_method(
        pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_summary.csv")
    )

    # (a) Expert reference -> Agent transition
    merged = silver[["event_id", "ground_truth_label"]].merge(
        inv[["event_id", "classification"]], on="event_id", how="inner"
    )
    reference_classes = CLASS_ORDER
    trans = pd.crosstab(merged["ground_truth_label"], merged["classification"]).reindex(
        index=reference_classes, columns=CLASS_ORDER, fill_value=0
    )
    df_a = trans.reset_index().rename(columns={"ground_truth_label": "reference_label"})
    save(df_a, "fig8a_reference_agent_transition", """fig8a_reference_agent_transition.csv
=====================================
????: Figure 8a ? ?????? -> Agent ??????
????: ??? (Heatmap)
????: data/labeled/labeled_events.csv + outputs/investigation_results.csv

???:
- reference_label: ? = ?????? (3?????????????)
- sampling~uncertain: ? = Agent ????
- ???? = ???

?????? vs Agent ???:
- ??????: 3????????????????, ????????????
- Agent: ?? ReAct ???????????????????
??????, ?????????? Agent ?????

Origin ??: ?????, ????, ????????
""")

    # (b) Inter-expert agreement distribution
    if "agreement_ratio" in silver.columns:
        bins = [0, 0.35, 0.5, 0.7, 0.85, 1.01]
        bin_labels = ["<0.35", "0.35-0.5", "0.5-0.7", "0.7-0.85", "0.85-1.0"]
        silver_copy = silver.copy()
        silver_copy["agreement_bin"] = pd.cut(silver_copy["agreement_ratio"], bins=bins, labels=bin_labels)
        counts = silver_copy["agreement_bin"].value_counts().reindex(bin_labels, fill_value=0)
        df_b = pd.DataFrame({"agreement_bin": bin_labels, "count": counts.values})
        if "consensus_status" in silver.columns:
            status_counts = silver["consensus_status"].value_counts()
            for status, cnt in status_counts.items():
                df_b[f"status_{status}"] = cnt
        save(df_b, "fig8b_expert_agreement", """fig8b_expert_agreement.csv
=====================================
????: Figure 8b ? ??????? (???)
????: ??? / ??? (Bar chart / Histogram)
????: data/labeled/labeled_events.csv

???:
- agreement_bin: ???????
    agreement_ratio = 3????????/3
    ?? 3/3=1.0 (????), 2/3=0.67 (????), 1/3=0.33 (????)
- count: ?????????
- status_*: ???????? (accepted=????, majority=????, disagreement=????)

Origin ??: ???, X?=agreement_bin, Y?=count, ???? (?->?) ??????????
""")

    # (c) Agent vs strongest baseline accuracy on subsets
    agent_pred = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "agent_predictions.csv")
    baseline_pred = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / f"{best_baseline}_predictions.csv")
    m = silver[["event_id", "ground_truth_label"]].copy()
    if "consensus_status" in silver.columns:
        m["consensus_status"] = silver["consensus_status"]
    m = m.merge(
        agent_pred[["event_id", "y_pred"]].rename(columns={"y_pred": "agent_pred"}),
        on="event_id"
    ).merge(
        baseline_pred[["event_id", "y_pred"]].rename(columns={"y_pred": "baseline_pred"}),
        on="event_id"
    )

    subsets = {
        "All": m,
        "Disagreement": m[m["consensus_status"] == "disagreement"] if "consensus_status" in m.columns else m.head(0),
        "Accepted_sampling": m[(m.get("consensus_status", "") == "accepted") & (m["ground_truth_label"] == "sampling")] if "consensus_status" in m.columns else m.head(0),
        "Epidemic_environmental": m[m["ground_truth_label"].isin(["epidemic", "environmental"])],
    }
    rows = []
    for name, df in subsets.items():
        n = len(df)
        agent_acc = float((df["agent_pred"] == df["ground_truth_label"]).mean()) if n > 0 else 0
        baseline_acc = float((df["baseline_pred"] == df["ground_truth_label"]).mean()) if n > 0 else 0
        rows.append({
            "subset": name,
            "n_events": n,
            "agent_accuracy": round(agent_acc, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "agent_advantage": round(agent_acc - baseline_acc, 4),
            "baseline_method": best_baseline,
            "baseline_display": METHOD_NAMES.get(best_baseline, best_baseline),
        })
    df_c = pd.DataFrame(rows)
    save(df_c, "fig8c_subset_accuracy", f"""fig8c_subset_accuracy.csv
=====================================
????: Figure 8c ? Agent vs ?????????????? (?????)
????: ????? (Grouped bar chart)
????: outputs/evaluation/{{agent,{best_baseline}}}_predictions.csv

???:
- subset: ??????
    All = ?? 199 ???
    Disagreement = ??????????????
    Accepted_sampling = ?????? sampling ???
    Epidemic_environmental = epidemic ? environmental ??????
- n_events: ???????
- agent_accuracy: WBE-Agent ?????????
- baseline_accuracy: ?????????????
- agent_advantage: Agent ????????????? (??=Agent??)
- baseline_method / baseline_display: ?????????? Agent ??

Origin ??: ?????, X?=subset, ????=agent/baseline, ?????
""")

    # (d) Head-to-head wins/losses by class against strongest baseline
    m["agent_correct"] = m["agent_pred"] == m["ground_truth_label"]
    m["baseline_correct"] = m["baseline_pred"] == m["ground_truth_label"]
    wins = m[m["agent_correct"] & ~m["baseline_correct"]]
    losses = m[~m["agent_correct"] & m["baseline_correct"]]
    rows = []
    for cls in CLASS_ORDER:
        win_count = int((wins["ground_truth_label"] == cls).sum())
        loss_count = int((losses["ground_truth_label"] == cls).sum())
        rows.append({
            "class": cls,
            "agent_wins": win_count,
            "baseline_wins": loss_count,
            "net_advantage": win_count - loss_count,
            "baseline_method": best_baseline,
            "baseline_display": METHOD_NAMES.get(best_baseline, best_baseline),
        })
    df_d = pd.DataFrame(rows)
    save(df_d, "fig8d_head_to_head", f"""fig8d_head_to_head.csv
=====================================
????: Figure 8d ? Agent ???????????
????: ??????? (Diverging horizontal bar chart)
????: outputs/evaluation/{{agent,{best_baseline}}}_predictions.csv + data/labeled/labeled_events.csv

???:
- class: ????
- agent_wins: Agent ???????????????
- baseline_wins: ????????? Agent ??????
- net_advantage: Agent ??????????
- baseline_method / baseline_display: ?????????? Agent ??

Origin ??: ???????, 0 ????, ?????? baseline / Agent ???
""")


def export_fig7_reliability(inv: pd.DataFrame, silver: pd.DataFrame):
    m = silver[["event_id", "ground_truth_label"]].merge(
        inv[["event_id", "classification", "confidence", "tool_calls_count", "total_tokens"]],
        on="event_id"
    )
    m["correct"] = (m["classification"] == m["ground_truth_label"]).astype(int)

    bins = [0.65, 0.80, 0.90, 0.97, 1.01]
    rows = []
    for i in range(len(bins) - 1):
        mask = (m["confidence"] >= bins[i]) & (m["confidence"] < bins[i + 1])
        subset = m[mask]
        if len(subset) > 0:
            rows.append({
                "bin_low": bins[i],
                "bin_high": bins[i + 1],
                "mean_confidence": round(subset["confidence"].mean(), 4),
                "observed_accuracy": round(subset["correct"].mean(), 4),
                "n_events": len(subset),
            })
    df_a = pd.DataFrame(rows)
    save(df_a, "fig8a_calibration", """fig8a_calibration.csv
=====================================
Corresponding figure: Figure 8a - Confidence calibration
Chart type: Calibration line plot
Data source: outputs/investigation_results.csv + data/labeled/labeled_events.csv

Columns
- bin_low / bin_high: Confidence interval bounds
- mean_confidence: Average predicted confidence within the bin
- observed_accuracy: Fraction of events correctly classified within the bin
- n_events: Number of events in the bin

Note: this panel checks whether higher predicted confidence corresponds to higher observed accuracy.
""")

    df_b = m[["event_id", "confidence", "correct"]].copy()
    df_b["correct_label"] = df_b["correct"].map({0: "incorrect", 1: "correct"})
    save(df_b, "fig8b_confidence_correctness", """fig8b_confidence_correctness.csv
=====================================
Corresponding figure: Figure 8b - Confidence by correctness
Chart type: Box-and-strip plot
Data source: outputs/investigation_results.csv + data/labeled/labeled_events.csv

Columns
- event_id: Event identifier
- confidence: Predicted confidence score
- correct: 1 if the final class matches the reference label, else 0
- correct_label: Textual version of correct

Note: this panel shows whether correct predictions tend to receive higher confidence than incorrect ones.
""")

    df_c = inv[["event_id", "classification", "tool_calls_count"]].copy()
    save(df_c, "fig8c_tool_calls_by_class", """fig8c_tool_calls_by_class.csv
=====================================
Corresponding figure: Figure 8c - Tool calls by predicted class
Chart type: Box-and-strip plot
Data source: outputs/investigation_results.csv

Columns
- event_id: Event identifier
- classification: Final predicted class
- tool_calls_count: Number of tool invocations used for that event

Note: this panel shows how investigation depth varied across final classes.
""")

    df_d = inv[["event_id", "classification", "total_tokens"]].copy()
    df_d["tokens_k"] = (df_d["total_tokens"] / 1000).round(2)
    save(df_d, "fig8d_token_usage_by_class", """fig8d_token_usage_by_class.csv
=====================================
Corresponding figure: Figure 8d - Token usage by predicted class
Chart type: Box-and-strip plot
Data source: outputs/investigation_results.csv

Columns
- event_id: Event identifier
- classification: Final predicted class
- total_tokens: Total token usage for the event
- tokens_k: Total token usage in thousands

Note: this panel shows how token budgets varied across final classes.
""")

    panel_index = pd.DataFrame([
        {"panel": "a", "csv_file": "fig8a_calibration.csv", "purpose": "Confidence calibration across binned predictions"},
        {"panel": "b", "csv_file": "fig8b_confidence_correctness.csv", "purpose": "Confidence distributions for correct versus incorrect calls"},
        {"panel": "c", "csv_file": "fig8c_tool_calls_by_class.csv", "purpose": "Tool-call usage by predicted class"},
        {"panel": "d", "csv_file": "fig8d_token_usage_by_class.csv", "purpose": "Token usage by predicted class"},
    ])
    save(panel_index, "fig8_panel_index", """fig8_panel_index.csv
=====================================
Corresponding figure: Figure 8 - Reliability and resource profile of agentic investigation
Chart type: Figure-level panel index

Columns
- panel: Panel letter in the manuscript
- csv_file: CSV file containing the plotted data
- purpose: One-line description of the panel

Note: use this file to navigate the current Figure 8 panel data.
""")
    save_text("fig8_DATA_GUIDE.txt", """Figure 8 data guide
=====================================
Figure 8 is the current reliability and resource-use figure in the manuscript.

Current panel mapping:
- a -> fig8a_calibration.csv
- b -> fig8b_confidence_correctness.csv
- c -> fig8c_tool_calls_by_class.csv
- d -> fig8d_token_usage_by_class.csv

These four files correspond to the current Figure 8 in paper/main.tex.
Older Figure 8 exports from previous manuscript structures may still exist in this folder or under legacy/.
""")


def export_current_main_figure_index():
    rows = [
        {"figure": "Figure 4", "panel": "row index", "csv_file": "fig4_case_index.csv", "purpose": "Current case-study row order and source-file mapping"},
        {"figure": "Figure 5", "panel": "index", "csv_file": "fig5_panel_index.csv", "purpose": "Current benchmark figure panel mapping"},
        {"figure": "Figure 6", "panel": "index", "csv_file": "fig6_panel_index.csv", "purpose": "Current ablation figure panel mapping"},
        {"figure": "Figure 7", "panel": "index", "csv_file": "fig7_panel_index.csv", "purpose": "Current multi-model workload figure panel mapping"},
        {"figure": "Figure 7", "panel": "summary", "csv_file": "fig7_summary_cards.csv", "purpose": "Figure-level workload summary cards"},
        {"figure": "Figure 8", "panel": "index", "csv_file": "fig8_panel_index.csv", "purpose": "Current reliability figure panel mapping"},
    ]
    save(pd.DataFrame(rows), "current_main_figure_data_index", """current_main_figure_data_index.csv
=====================================
Scope: Current main-text Figures 4-8 in paper/main.tex
Type: High-level index

Columns
- figure: Figure number in the manuscript
- panel: Panel or index type
- csv_file: File to open first
- purpose: Why this file matters

Note: this is the best entry point if you want to inspect the current data backing Figures 4-8 without scanning the whole csv directory.
""")
    save_text("current_main_figure_data_index_README.txt", """Current main-text figure data index
=====================================
This folder contains historical exports from several manuscript iterations.

If you only want the data behind the current Figures 4-8 in paper/main.tex,
start here:

- current_main_figure_data_index.csv

Then follow the figure-specific guides:
- fig4_DATA_GUIDE.txt
- fig5_DATA_GUIDE.txt
- fig6_DATA_GUIDE.txt
- fig7_DATA_GUIDE.txt
- fig8_DATA_GUIDE.txt
""")


def main():
    print("Loading data...")
    inv, auto, silver, catalog, bench, ablation = load_all()
    sites = load_sites()
    print(f"  {len(inv)} investigations, {len(silver)} reference labels, "
          f"{len(catalog)} catalog events, {len(sites)} sites")

    print("\nExporting Fig 1 - Site map...")
    export_fig1(sites)

    print("\nExporting Fig 3 - Investigation outcomes...")
    export_fig3(inv, auto)

    print("\nExporting Fig 4 - Case studies...")
    export_fig4()

    print("\nExporting Fig 5 - Benchmark comparison...")
    export_fig5(bench, silver)

    print("\nExporting current Fig 6 - Ablation analysis...")
    export_fig6_current_ablation(inv, auto, silver)

    print("\nExporting legacy Fig 7 - Agent behavior...")
    export_fig7(inv, catalog)

    print("\nExporting Fig 8 - Label agreement...")
    export_fig8(inv, silver)

    print("\nExporting current Fig 7 - Multi-model workload...")
    export_fig6_model_workload()

    print("\nExporting current Fig 8 - Reliability and resource profile...")
    export_fig7_reliability(inv, silver)

    print("\nExporting current main-text figure index...")
    export_current_main_figure_index()

    print(f"\nDone. All CSVs + READMEs saved to {OUT}")
    print("(Fig 2 is a pure architecture diagram - no data to export.)")

if __name__ == "__main__":
    main()
