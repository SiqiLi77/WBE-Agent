"""
Build a workload-focused visualization for multi-model WBE-Agent experiments.

Outputs:
    outputs/evaluation/model_workload_figure.png
    outputs/evaluation/model_workload_figure.pdf
    outputs/evaluation/model_workload_breakdown.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
PAPER_GENERATED_DIR = PROJECT_ROOT / "paper" / "generated"
SUMMARY_PATH = OUTPUT_DIR / "model_comparison_summary.csv"
BREAKDOWN_PATH = OUTPUT_DIR / "model_workload_breakdown.csv"
PNG_PATH = OUTPUT_DIR / "model_workload_figure.png"
PDF_PATH = OUTPUT_DIR / "model_workload_figure.pdf"
PAPER_PNG_PATH = PAPER_GENERATED_DIR / "fig_model_workload.png"
PAPER_PDF_PATH = PAPER_GENERATED_DIR / "fig_model_workload.pdf"

MODEL_COLORS = {
    "default": "#c85050",
    "v6_gpt4omini": "#d96b5f",
    "qwen3_5_27b": "#4878a8",
    "grok41_fast": "#5ba05b",
    "haiku45": "#c89838",
    "gpt54_nano": "#8c8c8c",
    "minimax47": "#7a6ff0",
    "kimi_k25": "#ef8b2c",
    "deepseek32": "#2f9d8f",
    "glm5": "#5e81ac",
    "gemini31_flash": "#e8a838",
    "gpt5mini": "#6c757d",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 220,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
    })


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", ha="left",
    )


def load_summary() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Batch model summary not found: {SUMMARY_PATH}. "
            "Run scripts/run_evaluation.py --agent-batch-glob=outputs/investigation_results*.csv first."
        )
    return pd.read_csv(SUMMARY_PATH)


def enrich_workload(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    df["display_name"] = df["method"].replace({
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
    })
    df["color"] = df["method"].map(MODEL_COLORS).fillna("#94a3b8")
    df["total_tokens"] = df["avg_tokens"].fillna(0) * df["prediction_rows"].fillna(df["n_samples"])
    df["total_tool_calls"] = df["avg_tool_calls"].fillna(0) * df["prediction_rows"].fillna(df["n_samples"])
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
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def draw_metric_card(ax, title: str, value: str, subtitle: str, color: str) -> None:
    ax.axis("off")
    card = mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.018,rounding_size=0.04",
        facecolor="#f8fafc",
        edgecolor="#d7dee7",
        linewidth=0.9,
    )
    ax.add_patch(card)
    ax.add_patch(mpatches.Rectangle((0.0, 0.0), 0.025, 1.0, color=color, transform=ax.transAxes))
    ax.text(0.08, 0.78, title, transform=ax.transAxes, fontsize=8, color="#5b6470", va="top")
    ax.text(0.08, 0.48, value, transform=ax.transAxes, fontsize=20, fontweight="bold", color="#0f172a", va="center")
    ax.text(0.08, 0.16, subtitle, transform=ax.transAxes, fontsize=8, color="#6b7280", va="bottom")


def build_figure(df: pd.DataFrame) -> plt.Figure:
    total_models = len(df)
    total_events = int(df["n_samples"].max()) if not df.empty else 0
    total_completed_runs = int(df["completed_event_runs"].sum())
    total_tokens = float(df["total_tokens"].sum())
    total_tool_calls = float(df["total_tool_calls"].sum())
    full_coverage_models = int((df["coverage"] >= 0.999).sum())

    fig = plt.figure(figsize=(13.5, 10.2))
    outer = gridspec.GridSpec(
        3, 2,
        height_ratios=[0.88, 1.50, 1.52],
        width_ratios=[1.2, 1.0],
        hspace=0.74, wspace=0.26,
    )

    cards = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0, :], wspace=0.18)
    card_axes = [fig.add_subplot(cards[0, i]) for i in range(4)]
    draw_metric_card(card_axes[0], "Models Compared", f"{total_models}", f"{full_coverage_models} with full 199-event coverage", "#c85050")
    draw_metric_card(card_axes[1], "Completed Event Runs", f"{total_completed_runs:,}", f"{total_events} labeled events evaluated per model", "#4878a8")
    draw_metric_card(card_axes[2], "Total LLM Tokens", f"{total_tokens / 1_000_000:.2f}M", "Across all saved multi-model investigations", "#5ba05b")
    draw_metric_card(card_axes[3], "Total Tool Calls", f"{total_tool_calls:,.0f}", "Agent tool invocations across all model runs", "#e8a838")

    ax_scatter = fig.add_subplot(outer[1, 0])
    panel_label(ax_scatter, "a")
    size_scale = 18.0
    bubble_sizes = np.maximum(df["total_tool_calls"].to_numpy() / size_scale, 90)
    ax_scatter.scatter(
        df["tokens_m"], df["macro_f1_pct"],
        s=bubble_sizes,
        c=df["color"],
        alpha=0.82,
        linewidth=0.9,
        edgecolors="white",
    )
    label_df = df.copy()
    label_df["x_key"] = label_df["tokens_m"].round(3)
    label_df["y_key"] = label_df["macro_f1_pct"].round(3)
    for _, group in label_df.groupby(["x_key", "y_key"], sort=False):
        label = " / ".join(group["display_name"].tolist())
        row = group.iloc[0]
        ax_scatter.text(
            row["tokens_m"] + 0.03, row["macro_f1_pct"] + 0.15,
            label,
            fontsize=7.3, color="#334155",
        )
    ax_scatter.set_xlabel("Total tokens consumed (millions)")
    ax_scatter.set_ylabel("Macro-F1 (%)")
    ax_scatter.set_title("Performance versus total workload", loc="left", pad=10)
    ax_scatter.grid(alpha=0.18, linewidth=0.6)
    for cov in [95, 100]:
        mask = (df["coverage_pct"].round(0) == cov)
        if mask.any():
            ax_scatter.scatter([], [], s=110, c="#cbd5e1", alpha=0.0, label=f"{cov:.0f}% coverage")

    tool_ref = [500, 700, 900]
    legend_handles = [
        plt.scatter([], [], s=max(v / size_scale, 90), color="#94a3b8", alpha=0.45, edgecolors="white")
        for v in tool_ref
    ]
    ax_scatter.legend(
        legend_handles, [f"{v:.0f} tool calls" for v in tool_ref],
        title="Bubble size", frameon=False, loc="lower right",
    )

    ax_bars = fig.add_subplot(outer[1, 1])
    panel_label(ax_bars, "b")
    plot_df = df.sort_values("total_tokens", ascending=True).copy()
    y = np.arange(len(plot_df))
    bars = ax_bars.barh(
        y, plot_df["tokens_k"],
        color=plot_df["color"],
        alpha=0.9,
        height=0.62,
    )
    ax_bars.set_yticks(y)
    ax_bars.set_yticklabels(plot_df["display_name"])
    ax_bars.set_xlabel("Total tokens (thousands)")
    ax_bars.set_title("Absolute token budget by model", loc="left", pad=18)
    ax_bars.grid(axis="x", alpha=0.18, linewidth=0.6)
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax_bars.text(
            bar.get_width() + 18,
            bar.get_y() + bar.get_height() / 2,
            f"{row['tool_calls_k']:.2f}k calls | {row['coverage_pct']:.1f}%",
            va="center",
            fontsize=7,
            color="#475569",
        )
    ax_bars.text(
        0.99, 0.03,
        "Labels show tool calls and coverage.",
        transform=ax_bars.transAxes, fontsize=7.2, color="#64748b", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
    )

    bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2, :], width_ratios=[1.15, 0.85], wspace=0.18)

    ax_runs = fig.add_subplot(bottom[0, 0])
    panel_label(ax_runs, "c")
    runs_df = df.sort_values(["tokens_per_correct", "macro_f1"], ascending=[True, False]).copy()
    y = np.arange(len(runs_df))
    bars = ax_runs.barh(
        y, runs_df["tokens_per_correct_k"],
        color=runs_df["color"], edgecolor="white", linewidth=0.6, height=0.62, alpha=0.92
    )
    ax_runs.set_yticks(y)
    ax_runs.set_yticklabels(runs_df["display_name"])
    ax_runs.set_xlabel("Tokens per correct benchmark decision (thousands)")
    ax_runs.set_title("Efficiency after accounting for correctness", loc="left", pad=18)
    ax_runs.grid(axis="x", alpha=0.18, linewidth=0.6)
    # ax_runs.text(
    #     0.99, 0.03,
    #     "Lower is better; labels show accuracy and coverage.",
    #     transform=ax_runs.transAxes, fontsize=7.2, color="#64748b", ha="right", va="bottom",
    #     bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82),
    # )
    for bar, (_, row) in zip(bars, runs_df.iterrows()):
        ax_runs.text(
            bar.get_width() + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{row['accuracy_pct']:.1f}% acc | {row['coverage_pct']:.1f}% cov",
            va="center",
            fontsize=7,
            color="#475569",
        )

    ax_table = fig.add_subplot(bottom[0, 1])
    panel_label(ax_table, "d")
    ax_table.axis("off")
    top_df = df.sort_values(["macro_f1", "accuracy"], ascending=False).head(6).copy()
    table_rows = [
        ["Model", "M-F1", "Acc", "Tokens", "Calls"]
    ]
    for _, row in top_df.iterrows():
        table_rows.append([
            row["display_name"],
            f"{row['macro_f1_pct']:.1f}",
            f"{row['accuracy_pct']:.1f}",
            f"{row['tokens_m']:.2f}M",
            f"{row['total_tool_calls']:.0f}",
        ])

    table = ax_table.table(
        cellText=table_rows,
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        bbox=[0.0, 0.28, 1.0, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#d7dee7")
        if r == 0:
            cell.set_facecolor("#eff6ff")
            cell.set_text_props(weight="bold", color="#1e293b")
        else:
            cell.set_facecolor("#ffffff" if r % 2 == 1 else "#f8fafc")

    workload_note = (
        "Workload framing:\n"
        f"- {total_models} base models evaluated\n"
        f"- {total_completed_runs:,} saved model-event investigations\n"
        f"- {total_tokens / 1_000_000:.2f}M total tokens consumed\n"
        f"- {total_tool_calls:,.0f} total tool calls recorded\n\n"
        "This figure emphasizes that the model-swap experiment is not a single benchmark pass.\n"
        "It is a repeated, tool-using agent evaluation over the same 199-event benchmark."
    )
    ax_table.text(
        0.0, 0.17, workload_note,
        transform=ax_table.transAxes,
        fontsize=8.5, color="#334155", va="top",
        linespacing=1.45,
    )

    fig.suptitle(
        "Multi-model WBE-Agent evaluation: performance and experimental workload",
        x=0.055, y=0.99, ha="left", fontsize=14, fontweight="bold",
    )
    fig.text(
        0.055, 0.955,
        "Each point or bar corresponds to one full agent run over the labeled anomaly benchmark. "
        "Workload is quantified from saved investigation outputs, tokens, and tool usage.",
        fontsize=9.2, color="#475569",
    )
    return fig


def main() -> None:
    setup_style()
    PAPER_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = load_summary()
    workload_df = enrich_workload(summary_df)
    workload_df.to_csv(BREAKDOWN_PATH, index=False)
    fig = build_figure(workload_df)
    fig.savefig(PNG_PATH)
    fig.savefig(PDF_PATH)
    fig.savefig(PAPER_PNG_PATH)
    fig.savefig(PAPER_PDF_PATH)
    plt.close(fig)
    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")
    print(f"Saved {PAPER_PNG_PATH}")
    print(f"Saved {PAPER_PDF_PATH}")
    print(f"Saved {BREAKDOWN_PATH}")


if __name__ == "__main__":
    main()
