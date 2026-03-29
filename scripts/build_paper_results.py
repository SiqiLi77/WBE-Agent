"""Generate paper-ready figures and LaTeX snippets for completed experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PAPER_DIR = PROJECT_ROOT / "paper"
GENERATED_DIR = PAPER_DIR / "generated"
ALL_LABELS = ["epidemic", "environmental", "sampling", "mixed", "uncertain"]

METHOD_NAMES = {
    "majority_class": "Majority class",
    "rule_based": "Rule-based",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "zero_shot_llm": "Zero-shot language model",
    "few_shot_llm": "Few-shot language model",
    "agent": "WBE-Agent",
}

METHOD_ORDER = [
    "majority_class",
    "rule_based",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "zero_shot_llm",
    "few_shot_llm",
    "agent",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generated = GENERATED_DIR
    generated.mkdir(parents=True, exist_ok=True)

    investigation = pd.read_csv(PROJECT_ROOT / "outputs" / "investigation_results.csv")
    auto_labels = pd.read_csv(PROJECT_ROOT / "data" / "labeled" / "auto_labeled_events.csv")
    silver_labels = pd.read_csv(PROJECT_ROOT / "data" / "labeled" / "labeled_events.csv")
    benchmark = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_summary.csv")
    return investigation, auto_labels, silver_labels, benchmark


def save_benchmark_rows(benchmark: pd.DataFrame) -> None:
    benchmark = benchmark.set_index("method").loc[METHOD_ORDER].reset_index()
    lines: list[str] = []
    for _, row in benchmark.iterrows():
        lines.append(
            f"{METHOD_NAMES[row['method']]} & "
            f"{row['accuracy']:.3f} & "
            f"{row['macro_f1']:.3f} & "
            f"{row['weighted_f1']:.3f} & "
            f"{row['cohen_kappa']:.3f} \\\\"
        )
    (GENERATED_DIR / "benchmark_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outcomes_figure(investigation: pd.DataFrame, auto_labels: pd.DataFrame) -> None:
    class_order = ["sampling", "uncertain", "mixed", "environmental", "epidemic"]
    counts = investigation["classification"].value_counts().reindex(class_order, fill_value=0)
    confidences = [investigation.loc[investigation["classification"] == label, "confidence"].to_numpy() for label in class_order]

    merged = auto_labels[["event_id", "auto_label"]].merge(
        investigation[["event_id", "classification"]], on="event_id", how="inner"
    )
    auto_order = ["uncertain", "sampling", "epidemic", "environmental"]
    trans = pd.crosstab(merged["auto_label"], merged["classification"]).reindex(
        index=auto_order, columns=class_order, fill_value=0
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)

    ax = axes[0]
    y = np.arange(len(class_order))
    bars = ax.barh(y, counts.values, color=["#1f4e79", "#8f6f3a", "#4f7c5d", "#a65e2e", "#7a3e8e"])
    ax.set_yticks(y, class_order)
    ax.invert_yaxis()
    ax.set_xlabel("Events")
    ax.set_title("Agent output distribution")
    total = len(investigation)
    for bar, value in zip(bars, counts.values):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value} ({value / total * 100:.1f}%)", va="center", fontsize=8)

    ax = axes[1]
    bp = ax.boxplot(confidences, tick_labels=class_order, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#1f4e79", "#8f6f3a", "#4f7c5d", "#a65e2e", "#7a3e8e"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence by predicted class")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[2]
    im = ax.imshow(trans.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(class_order)), class_order, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(auto_order)), auto_order)
    ax.set_title("Auto-label to agent transition")
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            value = int(trans.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(GENERATED_DIR / "fig_outcomes.pdf", bbox_inches="tight")
    plt.close(fig)


def build_benchmark_figure(benchmark: pd.DataFrame, silver_labels: pd.DataFrame) -> None:
    benchmark = benchmark.set_index("method").loc[METHOD_ORDER].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)

    ax = axes[0]
    colors = ["#b0b0b0", "#b0b0b0", "#b0b0b0", "#3366aa", "#5f8dd3", "#c4a24d", "#d9bf77", "#c0392b"]
    x = np.arange(len(benchmark))
    bars = ax.bar(x, benchmark["macro_f1"], color=colors)
    ax.set_xticks(x, [METHOD_NAMES[m] for m in benchmark["method"]], rotation=30, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 0.52)
    ax.set_title("Benchmark comparison")
    for bar, value in zip(bars, benchmark["macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    agent_pred = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "agent_predictions.csv")
    merged = silver_labels[["event_id", "ground_truth_label"]].merge(
        agent_pred[["event_id", "y_pred"]], on="event_id", how="left"
    )
    cm = confusion_matrix(merged["ground_truth_label"], merged["y_pred"], labels=ALL_LABELS)
    im = ax.imshow(cm, cmap="Reds")
    ax.set_xticks(np.arange(len(ALL_LABELS)), ALL_LABELS, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ALL_LABELS)), ALL_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Silver label")
    ax.set_title("Agent confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(GENERATED_DIR / "fig_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)


def load_ablation_summary(silver_labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    labels = silver_labels[["event_id", "ground_truth_label"]].rename(columns={"ground_truth_label": "y_true"})
    for path in sorted((PROJECT_ROOT / "outputs" / "ablation").glob("*_investigation_results.csv")):
        ablation = path.name.replace("_investigation_results.csv", "")
        pred = pd.read_csv(path)
        merged = labels.merge(pred[["event_id", "classification", "tool_calls_count", "total_tokens"]], on="event_id", how="inner")
        y_true = merged["y_true"].tolist()
        y_pred = merged["classification"].tolist()
        from src.evaluation.metrics import compute_all_metrics
        from src.evaluation.baselines import normalize_label

        metrics = compute_all_metrics(y_true, [normalize_label(x) for x in y_pred])
        rows.append(
            {
                "ablation": ablation,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "cohen_kappa": metrics["cohen_kappa"],
                "avg_tokens": float(pd.to_numeric(merged["total_tokens"], errors="coerce").fillna(0).mean()),
                "avg_tool_calls": float(pd.to_numeric(merged["tool_calls_count"], errors="coerce").fillna(0).mean()),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("macro_f1", ascending=False)
        df.to_csv(PROJECT_ROOT / "outputs" / "ablation" / "ablation_summary.csv", index=False)
    return df


def build_ablation_figure(benchmark: pd.DataFrame, silver_labels: pd.DataFrame) -> pd.DataFrame:
    ablation_df = load_ablation_summary(silver_labels)
    if ablation_df.empty:
        return ablation_df

    full_macro_f1 = float(benchmark.set_index("method").loc["agent", "macro_f1"])
    order = [
        "no_tools",
        "no_variants",
        "no_clinical",
        "no_nearby_sites",
        "no_domain_knowledge",
        "no_weather",
        "no_hydrology",
    ]
    display_names = {
        "no_tools": "No tools",
        "no_variants": "No variants",
        "no_clinical": "No clinical",
        "no_nearby_sites": "No nearby sites",
        "no_domain_knowledge": "No domain knowledge",
        "no_weather": "No weather",
        "no_hydrology": "No hydrology",
    }

    ablation_df = ablation_df.set_index("ablation").loc[order].reset_index()
    ablation_df["macro_f1_delta"] = ablation_df["macro_f1"] - full_macro_f1

    full_pred = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "agent_predictions.csv")[["event_id", "y_pred"]].rename(columns={"y_pred": "full"})
    merged = silver_labels[["event_id", "ground_truth_label"]].rename(columns={"ground_truth_label": "y_true"}).merge(full_pred, on="event_id")
    classes = ["environmental", "mixed", "sampling", "uncertain"]
    from sklearn.metrics import classification_report

    full_report = classification_report(merged["y_true"], merged["full"], output_dict=True, zero_division=0)
    recall_delta = pd.DataFrame(index=order, columns=classes, dtype=float)
    for ablation in order:
        pred = pd.read_csv(PROJECT_ROOT / "outputs" / "ablation" / f"{ablation}_investigation_results.csv")[["event_id", "classification"]]
        rep = classification_report(
            merged["y_true"],
            merged.merge(pred, on="event_id")["classification"],
            output_dict=True,
            zero_division=0,
        )
        for cls in classes:
            recall_delta.loc[ablation, cls] = rep.get(cls, {}).get("recall", 0.0) - full_report.get(cls, {}).get("recall", 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(ablation_df))
    colors = ["#c0392b" if value < 0 else "#4f7c5d" for value in ablation_df["macro_f1_delta"]]
    bars = ax.bar(x, ablation_df["macro_f1_delta"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [display_names[name] for name in ablation_df["ablation"]], rotation=30, ha="right")
    ax.set_ylabel("Macro-F1 delta vs full agent")
    ax.set_title("Ablation impact on macro-F1")
    for bar, value in zip(bars, ablation_df["macro_f1_delta"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.006 if value >= 0 else -0.018), f"{value:+.3f}", ha="center", fontsize=8)

    ax = axes[1]
    heatmap = recall_delta.loc[order, classes].to_numpy(dtype=float)
    im = ax.imshow(heatmap, cmap="coolwarm", vmin=-0.35, vmax=0.35, aspect="auto")
    ax.set_xticks(np.arange(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(order)), [display_names[name] for name in order])
    ax.set_title("Recall delta vs full agent")
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            ax.text(j, i, f"{heatmap[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(GENERATED_DIR / "fig_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    return ablation_df


def build_strengths_figure(silver_labels: pd.DataFrame) -> None:
    agent = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "agent_predictions.csv")[["event_id", "y_pred"]].rename(columns={"y_pred": "agent"})
    rf = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "random_forest_predictions.csv")[["event_id", "y_pred"]].rename(columns={"y_pred": "random_forest"})
    inv = pd.read_csv(PROJECT_ROOT / "outputs" / "investigation_results.csv")[["event_id", "summary"]]
    merged = silver_labels[["event_id", "ground_truth_label", "consensus_status"]].rename(columns={"ground_truth_label": "y_true"})
    merged = merged.merge(agent, on="event_id").merge(rf, on="event_id").merge(inv, on="event_id")

    subsets = {
        "All events": merged,
        "Judge disagreement": merged[merged["consensus_status"] == "disagreement"],
        "Accepted sampling": merged[(merged["consensus_status"] == "accepted") & (merged["y_true"] == "sampling")],
        "High-flow trace flag": merged[merged["summary"].str.contains("high_flow_cso_risk", regex=False)],
    }
    subset_names = list(subsets.keys())
    agent_scores = [float((df["agent"] == df["y_true"]).mean()) if len(df) else 0.0 for df in subsets.values()]
    rf_scores = [float((df["random_forest"] == df["y_true"]).mean()) if len(df) else 0.0 for df in subsets.values()]
    subset_sizes = [len(df) for df in subsets.values()]

    wins = merged[(merged["agent"] == merged["y_true"]) & (merged["random_forest"] != merged["y_true"])]
    win_counts = wins["y_true"].value_counts().reindex(["sampling", "mixed", "uncertain", "environmental"], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(subset_names))
    width = 0.36
    ax.bar(x - width / 2, agent_scores, width, label="WBE-Agent", color="#c0392b")
    ax.bar(x + width / 2, rf_scores, width, label="Random forest", color="#3366aa")
    ax.set_xticks(x, [f"{name}\n(n={n})" for name, n in zip(subset_names, subset_sizes)], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Slices where the agent adds value")
    ax.legend(frameon=False, loc="upper left")
    for xpos, score in zip(x - width / 2, agent_scores):
        ax.text(xpos, score + 0.03, f"{score:.2f}", ha="center", fontsize=8)
    for xpos, score in zip(x + width / 2, rf_scores):
        ax.text(xpos, score + 0.03, f"{score:.2f}", ha="center", fontsize=8)

    ax = axes[1]
    y = np.arange(len(win_counts))
    bars = ax.barh(y, win_counts.values, color="#c0392b")
    ax.set_yticks(y, win_counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Events")
    ax.set_title("Agent-correct / RF-wrong events")
    for bar, value in zip(bars, win_counts.values):
        ax.text(value + 0.15, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=8)

    fig.savefig(GENERATED_DIR / "fig_strengths.pdf", bbox_inches="tight")
    plt.close(fig)


def save_progress_summary(
    benchmark: pd.DataFrame,
    silver_labels: pd.DataFrame,
    investigation: pd.DataFrame,
    ablation_df: pd.DataFrame,
) -> None:
    benchmark = benchmark.set_index("method")
    agent = benchmark.loc["agent"]
    rf = benchmark.loc["random_forest"]
    zero = benchmark.loc["zero_shot_llm"]
    few = benchmark.loc["few_shot_llm"]
    ablation_summary = ""
    if not ablation_df.empty:
        worst = ablation_df.sort_values("macro_f1").iloc[0]
        best = ablation_df.sort_values("macro_f1", ascending=False).iloc[0]
        ablation_summary = (
            f"- Largest ablation drop: {worst['ablation']} ({worst['macro_f1'] - agent['macro_f1']:+.3f} macro-F1)\n"
            f"- Best single-source ablation under current silver labels: {best['ablation']} ({best['macro_f1']:.3f} macro-F1)\n"
        )
    summary = (
        "Completed experiments\n"
        f"- Three-judge silver-label benchmark: {len(silver_labels)} events\n"
        f"- Agent: accuracy {agent['accuracy']:.3f}, macro-F1 {agent['macro_f1']:.3f}, kappa {agent['cohen_kappa']:.3f}\n"
        f"- Strongest baseline: random forest, macro-F1 {rf['macro_f1']:.3f}\n"
        f"- Tool-use gain over zero-shot LLM: +{agent['macro_f1'] - zero['macro_f1']:.3f} macro-F1\n"
        f"- Tool-use gain over few-shot LLM: +{agent['macro_f1'] - few['macro_f1']:.3f} macro-F1\n"
        f"- Mean tool calls: {investigation['tool_calls_count'].mean():.2f}\n"
        f"- Mean tokens: {investigation['total_tokens'].mean():.2f}\n"
        f"{ablation_summary}"
        "Pending placeholders retained in the manuscript\n"
        "- Map/workflow/case-panel final artwork\n"
    )
    (GENERATED_DIR / "experiment_progress.txt").write_text(summary, encoding="utf-8")


def main() -> None:
    investigation, auto_labels, silver_labels, benchmark = load_data()
    save_benchmark_rows(benchmark)
    build_outcomes_figure(investigation, auto_labels)
    build_benchmark_figure(benchmark, silver_labels)
    ablation_df = build_ablation_figure(benchmark, silver_labels)
    build_strengths_figure(silver_labels)
    save_progress_summary(benchmark, silver_labels, investigation, ablation_df)


if __name__ == "__main__":
    main()
