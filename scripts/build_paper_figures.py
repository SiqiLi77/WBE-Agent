"""
Generate publication-quality figures for Nature Water submission.

Produces 6 data-driven composite figures (Figs 3, 5, 6, 7, 8, S-strengths)
with consistent styling, multi-panel layouts, and advanced chart types.

Usage:
    python scripts/build_paper_figures.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PAPER_DIR = PROJECT_ROOT / "paper"
GENERATED_DIR = PAPER_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Nature Water style configuration
# ---------------------------------------------------------------------------
# Palette inspired by Nature journals: muted, accessible, professional
PALETTE = {
    "sampling":      "#4878A8",   # steel blue
    "uncertain":     "#8C8C8C",   # warm grey
    "mixed":         "#E8A838",   # amber
    "environmental": "#5BA05B",   # sage green
    "epidemic":      "#C85050",   # muted red
}
CLASS_ORDER = ["sampling", "mixed", "environmental", "epidemic", "uncertain"]
CLASS_LABELS = {
    "sampling": "Sampling",
    "mixed": "Mixed",
    "environmental": "Environmental",
    "epidemic": "Epidemic",
    "uncertain": "Uncertain",
}

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
METHOD_COLORS = {
    "majority_class": "#BFBFBF",
    "rule_based": "#A0A0A0",
    "logistic_regression": "#7BA7CC",
    "random_forest": "#4878A8",
    "gradient_boosting": "#3A6088",
    "zero_shot_llm": "#D4A84D",
    "few_shot_llm": "#C89838",
    "agent": "#C85050",
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


def setup_style():
    """Apply Nature Water-compatible matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "pdf.fonttype": 42,  # TrueType for editability
        "ps.fonttype": 42,
    })


def panel_label(ax, label, x=-0.12, y=1.08):
    """Add bold panel label (a, b, c, ...) to an axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


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


def get_best_baseline_method(bench: pd.DataFrame) -> str:
    """Return the strongest non-agent baseline under the current benchmark."""
    baseline_rows = bench[bench["method"] != "agent"].copy()
    ranked = baseline_rows.sort_values(["macro_f1", "accuracy"], ascending=False)
    return str(ranked.iloc[0]["method"])


def get_ablation_subset_context(inv: pd.DataFrame, silver: pd.DataFrame) -> dict[str, object]:
    """Compute full-agent metrics on the 152-event subset used by cached ablation runs."""
    subset_path = PROJECT_ROOT / "outputs" / "ablation" / "no_tools_investigation_results.csv"
    subset_pred = pd.read_csv(subset_path)
    subset_ids = subset_pred["event_id"].drop_duplicates().tolist()

    full_subset = (
        silver[["event_id", "ground_truth_label"]]
        .merge(inv[["event_id", "classification", "total_tokens", "tool_calls_count"]], on="event_id", how="inner")
    )
    full_subset = full_subset[full_subset["event_id"].isin(subset_ids)].copy()

    full_report = classification_report(
        full_subset["ground_truth_label"],
        full_subset["classification"],
        output_dict=True,
        zero_division=0,
    )
    accuracy = float((full_subset["ground_truth_label"] == full_subset["classification"]).mean())
    macro_f1 = float(full_report["macro avg"]["f1-score"])
    return {
        "subset_ids": subset_ids,
        "subset_n": len(full_subset),
        "full_subset": full_subset,
        "full_report": full_report,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "avg_tokens": float(full_subset["total_tokens"].mean()),
        "avg_tool_calls": float(full_subset["tool_calls_count"].mean()),
    }


def load_current_ablation_summary() -> pd.DataFrame:
    """Load the current 199-event ablation summary, preferring canonicalized full-agent metrics."""
    if CURRENT_ABLATION_SUMMARY.exists():
        return pd.read_csv(CURRENT_ABLATION_SUMMARY)
    return pd.read_csv(CURRENT_ABLATION_DIR / "ablation_summary.csv")


def get_ablation_predictions(inv: pd.DataFrame, ablation_name: str) -> pd.DataFrame:
    """Return predictions for one ablation setting, using canonical full-agent outputs for `full`."""
    if ablation_name == "full":
        return inv[["event_id", "classification", "total_tokens", "tool_calls_count"]].copy()
    return pd.read_csv(CURRENT_ABLATION_DIR / f"{ablation_name}_investigation_results.csv")



# ===================================================================
# FIGURE 3 - Investigation outcomes composite (4 panels)
# ===================================================================
def build_fig3_outcomes(inv: pd.DataFrame, auto: pd.DataFrame):
    """
    a) Horizontal bar chart of final agent classifications
    b) Violin + strip plot of confidence by class
    c) Row-normalized stacked bars from weak auto-labels to final agent outputs
    d) Focused breakdown of weak uncertain auto-labels
    """
    fig = plt.figure(figsize=(7.25, 6.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                           left=0.09, right=0.97, top=0.94, bottom=0.08)

    dist_order = ["sampling", "environmental", "epidemic", "uncertain", "mixed"]
    stack_order = ["sampling", "environmental", "epidemic", "mixed", "uncertain"]
    merged = auto[["event_id", "auto_label"]].merge(
        inv[["event_id", "classification", "confidence"]], on="event_id", how="inner"
    )

    # --- (a) Final classification distribution ---
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")
    counts = inv["classification"].value_counts().reindex(dist_order, fill_value=0)
    total = int(counts.sum())
    y_pos = np.arange(len(dist_order))
    bars = ax_a.barh(
        y_pos,
        counts.values,
        color=[PALETTE[c] for c in dist_order],
        edgecolor="white",
        linewidth=0.7,
        height=0.62,
    )
    for y, cls, val in zip(y_pos, dist_order, counts.values):
        pct = val / total * 100 if total else 0
        ax_a.text(val + 2.0, y, f"{int(val)} ({pct:.1f}%)",
                  va="center", ha="left", fontsize=6.8, color="#333333")
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([CLASS_LABELS[c] for c in dist_order])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, counts.max() + 28)
    ax_a.set_xlabel("Events")
    ax_a.set_title("Final agent classifications", pad=8)
    ax_a.grid(axis="x", color="#e8edf2", linewidth=0.6)
    ax_a.set_axisbelow(True)
    ax_a.text(0.98, 0.04, f"n={total}", transform=ax_a.transAxes,
              ha="right", va="bottom", fontsize=7, color="#5b6470")

    # --- (b) Confidence by predicted class ---
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    conf_data = []
    positions = []
    for i, cls in enumerate(dist_order):
        vals = inv.loc[inv["classification"] == cls, "confidence"].dropna().values
        if len(vals) > 0:
            conf_data.append(vals)
            positions.append(i)
    vp = ax_b.violinplot(conf_data, positions=positions, showmeans=False,
                         showmedians=False, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        cls = dist_order[positions[i]]
        body.set_facecolor(PALETTE[cls])
        body.set_alpha(0.32)
        body.set_edgecolor(PALETTE[cls])
        body.set_linewidth(0.8)
    rng = np.random.default_rng(42)
    for i, cls in enumerate(dist_order):
        vals = inv.loc[inv["classification"] == cls, "confidence"].dropna().values
        if len(vals) > 0:
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ax_b.scatter(i + jitter, vals, s=8, alpha=0.52,
                         color=PALETTE[cls], edgecolors="none", zorder=3)
            med = np.median(vals)
            ax_b.hlines(med, i - 0.24, i + 0.24, colors=PALETTE[cls],
                        linewidth=1.9, zorder=4)
    ax_b.set_xticks(range(len(dist_order)))
    ax_b.set_xticklabels([CLASS_LABELS[c] for c in dist_order], rotation=25, ha="right")
    ax_b.set_ylabel("Confidence")
    ax_b.set_ylim(0.3, 1.0)
    ax_b.set_title("Confidence by predicted class", pad=8)
    ax_b.grid(axis="y", color="#e8edf2", linewidth=0.6)
    ax_b.set_axisbelow(True)

    # --- (c) Redistribution of weak auto-labels ---
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "c")
    preferred_auto_order = ["uncertain", "sampling", "epidemic", "environmental", "mixed"]
    auto_present = set(merged["auto_label"].dropna().unique())
    auto_order = [label for label in preferred_auto_order if label in auto_present]
    trans_counts = pd.crosstab(merged["auto_label"], merged["classification"]).reindex(
        index=auto_order, columns=stack_order, fill_value=0
    )
    row_totals = trans_counts.sum(axis=1)
    trans_share = trans_counts.div(row_totals, axis=0).fillna(0) * 100
    y_pos = np.arange(len(auto_order))
    left = np.zeros(len(auto_order))
    for cls in stack_order:
        widths = trans_share[cls].values
        counts_cls = trans_counts[cls].values
        ax_c.barh(y_pos, widths, left=left, color=PALETTE[cls],
                  edgecolor="white", linewidth=0.6, height=0.62)
        for y, start, width, cnt in zip(y_pos, left, widths, counts_cls):
            if cnt <= 0:
                continue
            if width >= 8:
                ax_c.text(start + width / 2, y, f"{int(cnt)}",
                          ha="center", va="center", fontsize=6.6,
                          color="white", fontweight="medium",
                          path_effects=[pe.withStroke(linewidth=1.2, foreground="black", alpha=0.35)])
        left += widths
    for y, total_row in zip(y_pos, row_totals.values):
        ax_c.text(103.0, y, f"n={int(total_row)}", va="center", ha="left",
                  fontsize=6.5, color="#5b6470")
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([CLASS_LABELS[a] for a in auto_order])
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 111)
    ax_c.set_xticks([0, 25, 50, 75, 100])
    ax_c.set_xlabel("Share of final agent classifications (%)")
    ax_c.set_ylabel("Weak auto-label")
    ax_c.set_title("How weak auto-labels changed after investigation", pad=8)
    ax_c.grid(axis="x", color="#e8edf2", linewidth=0.6)
    ax_c.set_axisbelow(True)

    # --- (d) Focused breakdown of weak uncertain auto-labels ---
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")
    uncertain_breakdown = trans_counts.loc["uncertain"] if "uncertain" in trans_counts.index else pd.Series(0, index=stack_order)
    uncertain_total = int(uncertain_breakdown.sum())
    resolved_total = int(uncertain_total - uncertain_breakdown.get("uncertain", 0))
    left = 0.0
    for cls in stack_order:
        cnt = int(uncertain_breakdown.get(cls, 0))
        if cnt <= 0:
            continue
        ax_d.barh(0, cnt, left=left, color=PALETTE[cls],
                  edgecolor="white", linewidth=0.7, height=0.52)
        label = f"{int(cnt)}"
        if cnt >= 6:
            ax_d.text(left + cnt / 2, 0, label, ha="center", va="center",
                      fontsize=6.8, color="white", fontweight="medium",
                      path_effects=[pe.withStroke(linewidth=1.2, foreground="black", alpha=0.35)])
        left += cnt
    ax_d.set_xlim(0, max(uncertain_total, 1))
    ax_d.set_ylim(-0.7, 0.7)
    ax_d.set_yticks([])
    xticks = [0, 25, 50, 75, 100]
    if uncertain_total not in xticks:
        xticks.append(uncertain_total)
    ax_d.set_xticks(sorted(x for x in xticks if x <= uncertain_total))
    ax_d.set_xlabel("Events from weak uncertain auto-labels")
    ax_d.set_title(
        (f"Most weak uncertain alerts were resolved\n({resolved_total}/{uncertain_total} = {resolved_total/uncertain_total*100:.1f}% reassigned)"
         if uncertain_total else "Most weak uncertain alerts were resolved"),
        pad=8,
        fontsize=8,
    )
    ax_d.grid(axis="x", color="#e8edf2", linewidth=0.6)
    ax_d.set_axisbelow(True)
    legend_order = [cls for cls in stack_order if int(uncertain_breakdown.get(cls, 0)) > 0]
    leg_handles = [mpatches.Patch(facecolor=PALETTE[c], edgecolor="white", label=CLASS_LABELS[c])
                   for c in legend_order]
    ax_d.legend(handles=leg_handles, loc="lower center", bbox_to_anchor=(0.5, -0.34),
                ncol=3, frameon=False, fontsize=6.2, handlelength=1.1, handletextpad=0.35)

    fig.savefig(GENERATED_DIR / "fig_outcomes.pdf")
    plt.close(fig)
    print("  - fig_outcomes.pdf")




# ===================================================================
# FIGURE 5 - Benchmark comparison composite (4 panels)
# ===================================================================
def build_fig5_benchmark(bench: pd.DataFrame, silver: pd.DataFrame):
    """
    a) Hero ranking of macro-F1 across all methods
    b) Per-class recall dumbbell plot (Agent vs strongest baseline)
    c) Operational subset accuracy dumbbell plot
    d) Reference-set agreement tiers
    """
    bench = bench.set_index("method").loc[METHOD_ORDER].reset_index()
    best_baseline = get_best_baseline_method(bench)
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

    fig5_colors = {
        "agent": "#B24A3A",
        "baseline": "#2F6773",
        "muted_bar": "#D8D3CC",
        "grid": "#ECE7E1",
        "agent_fill": "#F7E7E2",
        "baseline_fill": "#E3EFF2",
        "text": "#2E2A26",
    }

    fig = plt.figure(figsize=(6.75, 8.85))
    gs = gridspec.GridSpec(
        4,
        1,
        height_ratios=[1.55, 1.15, 1.2, 0.85],
        hspace=0.46,
        left=0.12,
        right=0.96,
        top=0.965,
        bottom=0.06,
    )

    # --- (a) Macro-F1 hero ranking ---
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a", x=-0.09, y=1.04)
    ranked = bench.sort_values(["macro_f1", "accuracy"], ascending=[True, True]).reset_index(drop=True)
    y_pos = np.arange(len(ranked))
    bar_colors = []
    for method in ranked["method"]:
        if method == "agent":
            bar_colors.append(fig5_colors["agent"])
        elif method == best_baseline:
            bar_colors.append(fig5_colors["baseline"])
        else:
            bar_colors.append(fig5_colors["muted_bar"])
    agent_idx = int(ranked.index[ranked["method"] == "agent"][0])
    baseline_idx = int(ranked.index[ranked["method"] == best_baseline][0])
    ax_a.axhspan(agent_idx - 0.45, agent_idx + 0.45, color=fig5_colors["agent_fill"], zorder=0)
    ax_a.axhspan(baseline_idx - 0.45, baseline_idx + 0.45, color=fig5_colors["baseline_fill"], zorder=0)
    bars = ax_a.barh(
        y_pos,
        ranked["macro_f1"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.68,
        zorder=2,
    )
    for bar, (_, row) in zip(bars, ranked.iterrows()):
        value = float(row["macro_f1"])
        weight = "bold" if row["method"] == "agent" else "normal"
        ax_a.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=6.6,
            fontweight=weight,
            color=fig5_colors["text"],
        )
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([METHOD_NAMES[m] for m in ranked["method"]], fontsize=6.8)
    ax_a.set_xlabel("Macro-F1")
    ax_a.set_xlim(0, 0.84)
    ax_a.grid(axis="x", color=fig5_colors["grid"], linewidth=0.7, zorder=0)
    ax_a.set_title("WBE-Agent leads the benchmark", pad=8)
    agent_macro = float(bench.loc[bench["method"] == "agent", "macro_f1"].iloc[0])
    baseline_macro = float(bench.loc[bench["method"] == best_baseline, "macro_f1"].iloc[0])
    ax_a.axvline(baseline_macro, color=fig5_colors["baseline"], linestyle="--", linewidth=0.8, alpha=0.85)
    ax_a.text(
        0.985,
        0.08,
        f"WBE-Agent\n{agent_macro:.3f}\n+{agent_macro - baseline_macro:.3f} vs {best_baseline_display}",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.1,
        fontweight="bold",
        color=fig5_colors["agent"],
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#FFF7F4", edgecolor="#E2B8AE", linewidth=0.7),
    )
    ax_a.text(
        0.0,
        1.10,
        "Macro-F1 across all eight evaluated methods",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        color="#5A5550",
    )

    # --- (b) Per-class recall vs strongest baseline ---
    ax_b = fig.add_subplot(gs[1, 0])
    panel_label(ax_b, "b", x=-0.09, y=1.04)
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
    y_pos = np.arange(len(recall_classes))[::-1]
    for yp, cls in zip(y_pos, recall_classes):
        baseline_val = float(baseline_report.get(cls, {}).get("recall", 0.0))
        agent_val = float(agent_report.get(cls, {}).get("recall", 0.0))
        line_color = PALETTE[cls]
        ax_b.hlines(yp, min(baseline_val, agent_val), max(baseline_val, agent_val), color=line_color, linewidth=2.2, alpha=0.55, zorder=1)
        ax_b.scatter(
            baseline_val,
            yp,
            s=42,
            facecolors="white",
            edgecolors=fig5_colors["baseline"],
            linewidth=1.1,
            zorder=3,
        )
        ax_b.scatter(
            agent_val,
            yp,
            s=58,
            marker="D",
            color=line_color,
            edgecolors="white",
            linewidth=0.6,
            zorder=4,
        )
        ax_b.text(
            1.03,
            yp,
            f"{agent_val - baseline_val:+.2f}",
            va="center",
            ha="right",
            fontsize=6.2,
            color=line_color,
        )
    ax_b.set_xlim(0, 1.05)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(
        [f"{CLASS_LABELS[c]} (n={int(support.get(c, 0))})" for c in recall_classes],
        fontsize=6.5,
    )
    ax_b.set_xlabel("Recall")
    ax_b.grid(axis="x", color=fig5_colors["grid"], linewidth=0.7, zorder=0)
    ax_b.set_title(f"The largest gains appear in key classes", pad=8)
    ax_b.legend(
        handles=[
            plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=fig5_colors["agent"],
                       markeredgecolor="white", markeredgewidth=0.5, markersize=6, label="WBE-Agent"),
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                       markeredgecolor=fig5_colors["baseline"], markeredgewidth=1.1, markersize=5.8, label=best_baseline_display),
        ],
        loc="lower right",
        frameon=False,
        fontsize=6.1,
        handletextpad=0.35,
    )

    # --- (c) Accuracy on operational subsets ---
    ax_c = fig.add_subplot(gs[2, 0])
    panel_label(ax_c, "c", x=-0.09, y=1.04)
    subset_frames = [
        ("All", merged),
        ("Accepted sampling", merged[(merged["consensus_status"] == "accepted") & (merged["ground_truth_label"] == "sampling")]),
        ("Epidemic + environmental", merged[merged["ground_truth_label"].isin(["epidemic", "environmental"])]),
        ("Disagreement", merged[merged["consensus_status"] == "disagreement"]),
    ]
    subset_rows = []
    for label, df in subset_frames:
        n = len(df)
        agent_acc = float((df["agent"] == df["ground_truth_label"]).mean()) if n > 0 else 0.0
        baseline_acc = float((df["baseline"] == df["ground_truth_label"]).mean()) if n > 0 else 0.0
        subset_rows.append((label, n, agent_acc, baseline_acc))
    y_pos = np.arange(len(subset_rows))[::-1]
    subset_colors = {
        "All": "#6F6A63",
        "Accepted sampling": PALETTE["sampling"],
        "Epidemic + environmental": "#6E8052",
        "Disagreement": PALETTE["uncertain"],
    }
    for yp, (label, n, agent_acc, baseline_acc) in zip(y_pos, subset_rows):
        line_color = subset_colors.get(label, "#888888")
        ax_c.hlines(yp, min(baseline_acc, agent_acc), max(baseline_acc, agent_acc), color=line_color, linewidth=2.2, alpha=0.55, zorder=1)
        ax_c.scatter(
            baseline_acc,
            yp,
            s=42,
            facecolors="white",
            edgecolors=fig5_colors["baseline"],
            linewidth=1.1,
            zorder=3,
        )
        ax_c.scatter(
            agent_acc,
            yp,
            s=58,
            marker="D",
            color=line_color,
            edgecolors="white",
            linewidth=0.6,
            zorder=4,
        )
        ax_c.text(
            1.03,
            yp,
            f"{agent_acc - baseline_acc:+.2f}",
            va="center",
            ha="right",
            fontsize=6.2,
            color=line_color,
        )
    ax_c.set_xlim(0, 1.05)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([f"{label} (n={n})" for label, n, _, _ in subset_rows], fontsize=6.5)
    ax_c.set_xlabel("Accuracy")
    ax_c.grid(axis="x", color=fig5_colors["grid"], linewidth=0.7, zorder=0)
    ax_c.set_title("Operational subsets reveal the practical margin", pad=8)

    # --- (d) Agreement tiers in the expert reference set ---
    ax_d = fig.add_subplot(gs[3, 0])
    panel_label(ax_d, "d", x=-0.09, y=1.03)
    status_order = ["accepted", "majority", "disagreement"]
    status_labels = {
        "accepted": "Full agreement",
        "majority": "Majority agreement",
        "disagreement": "Persistent disagreement",
    }
    status_colors = {
        "accepted": "#4878A8",
        "majority": "#8C8C8C",
        "disagreement": "#C85050",
    }
    status_counts = silver["consensus_status"].value_counts().reindex(status_order, fill_value=0)
    total_events = int(status_counts.sum())
    ax_d.set_xlim(0, total_events)
    ax_d.set_ylim(0, 1)
    left = 0
    for status in status_order:
        width = int(status_counts[status])
        color = status_colors[status]
        ax_d.barh(
            [0.58],
            [width],
            left=left,
            height=0.34,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        x_mid = left + width / 2
        pct = width / total_events if total_events else 0.0
        ax_d.text(
            x_mid,
            0.16,
            status_labels[status],
            ha="center",
            va="top",
            fontsize=6.3,
            color="#4A4A4A",
        )
        ax_d.text(
            x_mid,
            0.58,
            f"{width}",
            ha="center",
            va="center",
            fontsize=9.2,
            color="white",
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=0.8, foreground=color)],
        )
        ax_d.text(
            x_mid,
            0.79,
            f"{pct:.0%}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=color,
            fontweight="bold",
        )
        left += width
    mean_agreement = float(silver["agreement_ratio"].dropna().mean())
    ax_d.text(
        0.5,
        0.97,
        f"Mean agreement ratio = {mean_agreement:.2f}",
        transform=ax_d.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F7F5F2", edgecolor="#DDD6CF", linewidth=0.6),
    )
    ax_d.set_yticks([])
    ax_d.set_xticks([0, 50, 100, 150, total_events])
    ax_d.set_xlabel("Benchmark events")
    ax_d.grid(axis="x", color=fig5_colors["grid"], linewidth=0.7, zorder=0)
    ax_d.set_title("The reference set retains real ambiguity", pad=8)
    ax_d.spines["left"].set_visible(False)

    fig.savefig(GENERATED_DIR / "fig_benchmark.pdf")
    plt.close(fig)
    print("  - fig_benchmark.pdf")



# ===================================================================
# FIGURE 6 - Ablation composite (3 panels)
# ===================================================================
def build_fig6_ablation(bench: pd.DataFrame, ablation: pd.DataFrame, silver: pd.DataFrame):
    """
    a) Waterfall chart: macro-F1 delta for each ablation
    b) Heatmap: per-class recall delta vs full agent
    c) Token / tool-call cost comparison (dot + bar)
    """
    bench_idx = bench.set_index("method")
    full_macro = float(bench_idx.loc["agent", "macro_f1"])
    abl = ablation.set_index("ablation").loc[ABLATION_ORDER].reset_index()
    abl["delta"] = abl["macro_f1"] - full_macro

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35,
                           left=0.10, right=0.96, top=0.95, bottom=0.07,
                           height_ratios=[1, 1.1])

    # --- (a) Waterfall chart ---
    ax_a = fig.add_subplot(gs[0, :])
    panel_label(ax_a, "a", x=-0.06)
    n = len(abl)
    x = np.arange(n + 1)
    # First bar is full agent
    bar_vals = [full_macro] + abl["macro_f1"].tolist()
    bar_labels = ["Full Agent"] + [ABLATION_NAMES[a] for a in abl["ablation"]]
    bar_colors = ["#333333"] + ["#C85050" if d < -0.05 else "#E8A838" if d < -0.02
                                 else "#8C8C8C" for d in abl["delta"]]
    bars = ax_a.bar(x, bar_vals, color=bar_colors, width=0.62,
                    edgecolor="white", linewidth=0.5, zorder=2)
    # Reference line at full agent
    ax_a.axhline(full_macro, color="#333333", linestyle="--", linewidth=0.6,
                 alpha=0.5, zorder=1)
    # Delta annotations
    for i, (bar, val) in enumerate(zip(bars, bar_vals)):
        y_off = 0.012
        if i == 0:
            ax_a.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                      f"{val:.3f}", ha="center", va="bottom", fontsize=6.5,
                      fontweight="bold")
        else:
            delta = abl.iloc[i - 1]["delta"]
            ax_a.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                      f"{val:.3f}\n({delta:+.3f})", ha="center", va="bottom",
                      fontsize=5.8, color="#555555")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(bar_labels, rotation=25, ha="right", fontsize=7)
    ax_a.set_ylabel("Macro-F1")
    ax_a.set_ylim(0, 0.48)
    ax_a.set_title("Ablation impact on macro-F1", pad=10)

    # --- (b) Per-class recall delta heatmap ---
    ax_b = fig.add_subplot(gs[1, 0])
    panel_label(ax_b, "b")
    recall_classes = ["environmental", "mixed", "sampling", "uncertain"]
    # Compute full agent per-class recall
    agent_pred = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "agent_predictions.csv")
    m_full = silver[["event_id", "ground_truth_label"]].merge(
        agent_pred[["event_id", "y_pred"]], on="event_id", how="left"
    )
    full_report = classification_report(m_full["ground_truth_label"], m_full["y_pred"],
                                         output_dict=True, zero_division=0)
    recall_delta = pd.DataFrame(index=ABLATION_ORDER, columns=recall_classes, dtype=float)
    for abl_name in ABLATION_ORDER:
        pred = pd.read_csv(
            PROJECT_ROOT / "outputs" / "ablation" / f"{abl_name}_investigation_results.csv"
        )
        m_abl = silver[["event_id", "ground_truth_label"]].merge(
            pred[["event_id", "classification"]], on="event_id", how="inner"
        )
        rep = classification_report(m_abl["ground_truth_label"], m_abl["classification"],
                                     output_dict=True, zero_division=0)
        for cls in recall_classes:
            recall_delta.loc[abl_name, cls] = (
                rep.get(cls, {}).get("recall", 0.0)
                - full_report.get(cls, {}).get("recall", 0.0)
            )
    heatmap = recall_delta.to_numpy(dtype=float)
    cmap_div = LinearSegmentedColormap.from_list(
        "rdbu", ["#C85050", "#F5E6E6", "#FFFFFF", "#E0EEF0", "#4878A8"]
    )
    vabs = max(abs(heatmap.min()), abs(heatmap.max()), 0.35)
    im = ax_b.imshow(heatmap, cmap=cmap_div, aspect="auto", vmin=-vabs, vmax=vabs)
    ax_b.set_xticks(np.arange(len(recall_classes)))
    ax_b.set_xticklabels([CLASS_LABELS[c] for c in recall_classes], rotation=30, ha="right")
    ax_b.set_yticks(np.arange(len(ABLATION_ORDER)))
    ax_b.set_yticklabels([ABLATION_NAMES[a] for a in ABLATION_ORDER], fontsize=6.5)
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            v = heatmap[i, j]
            color = "white" if abs(v) > vabs * 0.6 else "#333333"
            ax_b.text(j, i, f"{v:+.2f}", ha="center", va="center",
                      fontsize=6, fontweight="medium", color=color)
    ax_b.set_title("Recall delta vs full agent", pad=8)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.06, shrink=0.85)
    cbar.ax.tick_params(labelsize=5.5)

    # --- (c) Token & tool-call comparison ---
    ax_c = fig.add_subplot(gs[1, 1])
    panel_label(ax_c, "c")
    # Add full agent row
    full_tokens = float(bench_idx.loc["agent", "avg_tokens"])
    full_tools = float(bench_idx.loc["agent", "avg_tool_calls"])
    plot_names = ["Full Agent"] + [ABLATION_NAMES[a] for a in ABLATION_ORDER]
    tokens = [full_tokens] + abl["avg_tokens"].tolist()
    tools = [full_tools] + abl["avg_tool_calls"].tolist()
    y_pos = np.arange(len(plot_names))[::-1]
    # Bars for tokens (normalized to max)
    max_tok = max(tokens)
    bar_widths = [t / max_tok * 0.85 for t in tokens]
    tok_colors = ["#333333"] + ["#4878A8"] * len(ABLATION_ORDER)
    for i, (yp, bw, tc) in enumerate(zip(y_pos, bar_widths, tok_colors)):
        ax_c.barh(yp, bw, height=0.55, color=tc, alpha=0.3, zorder=1)
    # Dots for tool calls
    tool_norm = [t / max(tools) * 0.85 for t in tools]
    dot_colors = ["#333333"] + ["#C85050"] * len(ABLATION_ORDER)
    for i, (yp, tn, dc) in enumerate(zip(y_pos, tool_norm, dot_colors)):
        ax_c.scatter(tn, yp, color=dc, s=35, zorder=3, edgecolors="white", linewidth=0.4)
    # Annotations
    for i, (yp, tok, tool) in enumerate(zip(y_pos, tokens, tools)):
        ax_c.text(0.88, yp + 0.22, f"{tok/1000:.1f}k tok", fontsize=5.5,
                  color="#4878A8", ha="right", va="bottom")
        ax_c.text(0.88, yp - 0.22, f"{tool:.1f} calls", fontsize=5.5,
                  color="#C85050", ha="right", va="top")
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(plot_names, fontsize=6.5)
    ax_c.set_xlim(0, 0.95)
    ax_c.set_xticks([])
    ax_c.set_title("Resource usage", pad=8)
    # Legend
    leg = [
        mpatches.Patch(facecolor="#4878A8", alpha=0.3, label="Tokens"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C85050",
                    markersize=5, label="Tool calls"),
    ]
    ax_c.legend(handles=leg, loc="lower right", frameon=False, fontsize=6)

    fig.savefig(GENERATED_DIR / "fig_ablation.pdf")
    plt.close(fig)
    print("  - fig_ablation.pdf")


def build_fig_ablation_current(inv: pd.DataFrame, auto: pd.DataFrame, silver: pd.DataFrame):
    """
    Current 199-event ablation figure (4 x 1 layout).

    a) Overall macro-F1 degradation relative to the canonical full agent
    b) Per-class recall delta vs full agent
    c) Resolution of weak uncertain alerts under each ablation
    d) Resource profile (tokens and tool calls)
    """
    summary = load_current_ablation_summary().copy()
    order = ["full"] + (
        summary.loc[summary["ablation"] != "full"]
        .sort_values(["macro_f1", "accuracy"], ascending=False)["ablation"]
        .tolist()
    )
    summary = summary.set_index("ablation").loc[order].reset_index()

    display_names = {"full": "Full agent", **ABLATION_NAMES}
    summary["display_name"] = summary["ablation"].map(display_names)
    full_macro = float(summary.loc[summary["ablation"] == "full", "macro_f1"].iloc[0])
    summary["delta_macro"] = summary["macro_f1"] - full_macro

    class_order = ["sampling", "environmental", "epidemic", "uncertain", "mixed"]

    fig = plt.figure(figsize=(7.85, 8.6))
    gs = gridspec.GridSpec(
        4, 1,
        hspace=0.34,
        left=0.13, right=0.96, top=0.975, bottom=0.055,
        height_ratios=[0.92, 1.00, 1.08, 0.92],
    )

    # --- (a) Overall performance drop ---
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a", x=-0.08)
    y = np.arange(len(summary))
    perf_colors = []
    for abl, delta in zip(summary["ablation"], summary["delta_macro"]):
        if abl == "full":
            perf_colors.append("#7F2438")
        elif delta <= -0.45:
            perf_colors.append("#A83246")
        elif delta <= -0.30:
            perf_colors.append("#CF5B45")
        elif delta <= -0.15:
            perf_colors.append("#E39A42")
        else:
            perf_colors.append("#9A9A9A")
    ax_a.hlines(y, xmin=0, xmax=summary["macro_f1"], color="#D9E0E6", linewidth=1.0, zorder=1)
    ax_a.scatter(summary["macro_f1"], y, s=70, color=perf_colors,
                 edgecolors="white", linewidth=0.8, zorder=3)
    ax_a.axvline(full_macro, color="#7F2438", linestyle="--", linewidth=0.9, alpha=0.65)
    for yi, row in zip(y, summary.itertuples(index=False)):
        if row.ablation == "full":
            label = f"{row.macro_f1:.3f}"
        else:
            label = f"{row.macro_f1:.3f} ({row.delta_macro:+.3f})"
        ax_a.text(row.macro_f1 + 0.013, yi, label, va="center", ha="left",
                  fontsize=6.8, color="#333333")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(summary["display_name"])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, max(0.82, summary["macro_f1"].max() + 0.10))
    ax_a.set_xlabel("Macro-F1")
    ax_a.set_title("Overall degradation when evidence streams are removed", pad=5)
    ax_a.grid(axis="x", color="#EDF1F4", linewidth=0.6)
    ax_a.set_axisbelow(True)
    ax_a.text(
        0.99, 0.04,
        f"Canonical full agent: {full_macro:.3f} on 199 events",
        transform=ax_a.transAxes, ha="right", va="bottom",
        fontsize=6.6, color="#6B7580",
    )

    # --- (b) Class-specific recall delta ---
    ax_b = fig.add_subplot(gs[1, 0])
    panel_label(ax_b, "b", x=-0.08)
    full_pred = get_ablation_predictions(inv, "full")
    full_merge = silver[["event_id", "ground_truth_label"]].merge(
        full_pred[["event_id", "classification"]], on="event_id", how="inner"
    )
    full_report = classification_report(
        full_merge["ground_truth_label"], full_merge["classification"],
        output_dict=True, zero_division=0
    )
    recall_delta = pd.DataFrame(index=order[1:], columns=class_order, dtype=float)
    for abl_name in order[1:]:
        pred = get_ablation_predictions(inv, abl_name)
        merged = silver[["event_id", "ground_truth_label"]].merge(
            pred[["event_id", "classification"]], on="event_id", how="inner"
        )
        rep = classification_report(
            merged["ground_truth_label"], merged["classification"],
            output_dict=True, zero_division=0
        )
        for cls in class_order:
            recall_delta.loc[abl_name, cls] = (
                rep.get(cls, {}).get("recall", 0.0)
                - full_report.get(cls, {}).get("recall", 0.0)
            )
    heat = recall_delta.to_numpy(dtype=float)
    cmap_div = LinearSegmentedColormap.from_list(
        "ablation_div",
        ["#B8444A", "#F3D7D5", "#FAFAFA", "#D8E8EE", "#2F6B8E"],
    )
    vmax = max(0.55, float(np.nanmax(np.abs(heat)))) if heat.size else 0.55
    im = ax_b.imshow(heat, cmap=cmap_div, aspect="auto", vmin=-vmax, vmax=vmax)
    ax_b.set_xticks(np.arange(len(class_order)))
    ax_b.set_xticklabels([CLASS_LABELS[c] for c in class_order], rotation=25, ha="right")
    ax_b.set_yticks(np.arange(len(order[1:])))
    ax_b.set_yticklabels([display_names[a] for a in order[1:]], fontsize=6.7)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            color = "white" if abs(v) > vmax * 0.55 else "#333333"
            ax_b.text(j, i, f"{v:+.2f}", ha="center", va="center",
                      fontsize=6.1, color=color)
    ax_b.set_title("Class-specific recall loss reveals complementary evidence roles", pad=5)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.030, pad=0.03)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Recall delta vs full", fontsize=6.5)

    # --- (c) Resolution of weak uncertain alerts ---
    ax_c = fig.add_subplot(gs[2, 0])
    panel_label(ax_c, "c", x=-0.08)
    uncertain_ids = auto.loc[auto["auto_label"] == "uncertain", "event_id"].drop_duplicates()
    stack_order = ["sampling", "environmental", "epidemic", "mixed", "uncertain"]
    stack_colors = [PALETTE[c] for c in stack_order]
    uncertain_total = int(len(uncertain_ids))
    y_c = np.arange(len(order))
    left = np.zeros(len(order), dtype=float)
    resolution_counts: dict[str, dict[str, int]] = {}
    for cls, color in zip(stack_order, stack_colors):
        vals = []
        for abl_name in order:
            pred = get_ablation_predictions(inv, abl_name)
            merged = uncertain_ids.to_frame().merge(
                pred[["event_id", "classification"]], on="event_id", how="left"
            )
            counts = merged["classification"].value_counts()
            resolution_counts.setdefault(abl_name, {})
            resolution_counts[abl_name][cls] = int(counts.get(cls, 0))
            vals.append(counts.get(cls, 0) / uncertain_total * 100 if uncertain_total else 0.0)
        ax_c.barh(y_c, vals, left=left, color=color, edgecolor="white",
                  linewidth=0.4, height=0.66, label=CLASS_LABELS[cls])
        left += np.array(vals)
    for yi, abl_name in zip(y_c, order):
        resolved = uncertain_total - resolution_counts[abl_name].get("uncertain", 0)
        ax_c.text(101.5, yi, f"{resolved}/{uncertain_total} resolved",
                  va="center", ha="left", fontsize=6.4, color="#4E5965")
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels([display_names[a] for a in order])
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 114)
    ax_c.set_xlabel("Weak uncertain alerts redistributed (%)")
    ax_c.set_title("Evidence removal changes how ambiguous alerts are resolved", pad=5)
    ax_c.grid(axis="x", color="#EDF1F4", linewidth=0.6)
    ax_c.set_axisbelow(True)
    ax_c.legend(
        loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.00, 1.04),
        handlelength=1.0, columnspacing=0.8, fontsize=5.4
    )

    # --- (d) Resource profile ---
    ax_d = fig.add_subplot(gs[3, 0])
    panel_label(ax_d, "d", x=-0.08)
    tokens_k = summary["avg_tokens"] / 1000.0
    y_d = np.arange(len(summary))
    token_colors = ["#E3C8C1" if abl != "full" else "#D88E84" for abl in summary["ablation"]]
    ax_d.barh(y_d, tokens_k, color=token_colors, edgecolor="white",
              linewidth=0.6, height=0.60, zorder=1)
    ax_d2 = ax_d.twiny()
    ax_d2.scatter(summary["avg_tool_calls"], y_d, color="#2F6B8E", s=36,
                  edgecolors="white", linewidth=0.6, zorder=3)
    for yi, row in zip(y_d, summary.itertuples(index=False)):
        ax_d.text(row.avg_tokens / 1000.0 + 0.25, yi + 0.16, f"{row.avg_tokens/1000:.1f}k tok",
                  fontsize=6.1, ha="left", va="center", color="#7A3E32")
        ax_d2.text(row.avg_tool_calls + 0.22, yi - 0.16, f"{row.avg_tool_calls:.1f} calls",
                   fontsize=6.1, ha="left", va="center", color="#2F6B8E")
    ax_d.set_yticks(y_d)
    ax_d.set_yticklabels(summary["display_name"])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Average tokens per event (thousands)")
    ax_d2.set_xlabel("Average tool calls per event")
    ax_d.set_title("Performance differences coincide with distinct investigation-cost profiles", pad=5)
    ax_d.grid(axis="x", color="#EDF1F4", linewidth=0.6)
    ax_d.set_axisbelow(True)
    ax_d.set_xlim(0, float(tokens_k.max()) + 4.5)
    ax_d2.set_xlim(0, float(summary["avg_tool_calls"].max()) + 3.5)
    ax_d2.spines["top"].set_visible(False)
    ax_d2.tick_params(axis="x", labelsize=6.4)
    ax_d.text(
        0.99, 0.04,
        "Bars show token burden; points show investigation depth.",
        transform=ax_d.transAxes, ha="right", va="bottom",
        fontsize=6.4, color="#6B7580",
    )

    fig.savefig(GENERATED_DIR / "fig_ablation.pdf", bbox_inches=None, pad_inches=0.04)
    fig.savefig(GENERATED_DIR / "fig_ablation.png", bbox_inches=None, pad_inches=0.04)
    plt.close(fig)
    print("  - fig_ablation.pdf")



# ===================================================================
# FIGURE 7 - Agent behavioral analysis (3 panels)
# ===================================================================
def build_fig7_behavior(inv: pd.DataFrame, catalog: pd.DataFrame):
    """
    a) Factor frequency horizontal bar chart (parsed from summary)
    b) Classification by state  - stacked horizontal bars
    c) Temporal distribution  - events over time colored by class
    """
    inv = inv.copy()

    fig = plt.figure(figsize=(7.2, 6.8))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35,
                           left=0.12, right=0.96, top=0.94, bottom=0.08)

    # --- Parse factors from summary ---
    factor_map = {
        "isolated_spike": "Signal isolation",
        "isolated_signal": "Signal isolation",
        "signal isolation": "Signal isolation",
        "precipitation": "Precipitation",
        "heavy_rainfall": "Precipitation",
        "weather": "Weather context",
        "weather conditions": "Weather context",
        "weather data": "Weather context",
        "discharge": "Discharge / flow",
        "discharge flow": "Discharge / flow",
        "discharge percentile": "Discharge / flow",
        "high flow risk": "High flow / CSO",
        "high flow and cso risk": "High flow / CSO",
        "hospitalization trend": "Hospitalization trend",
        "hospitalization_trend": "Hospitalization trend",
        "hospitalization data": "Hospitalization trend",
        "stable hospitalization trend": "Hospitalization trend",
        "variant dynamics": "Variant dynamics",
        "variant emergence": "Variant dynamics",
        "variant shift": "Variant dynamics",
        "nearby sites": "Nearby-site context",
        "nearby site corroboration": "Nearby-site context",
        "nearby_sites": "Nearby-site context",
        "wastewater signal": "Wastewater trend",
        "wastewater signal trend": "Wastewater trend",
        "silver_label_passthrough": None,
    }

    def parse_factors(summary):
        if pd.isna(summary):
            return []
        m = re.search(r"factors=([^)]*)\)", str(summary))
        if not m:
            return []
        raw = m.group(1).strip()
        if not raw:
            return []
        return [f.strip() for f in raw.split(",") if f.strip()]

    inv["factors_list"] = inv["summary"].apply(parse_factors)
    all_factors = {}
    for _, row in inv.iterrows():
        for f in row["factors_list"]:
            canonical = factor_map.get(f.lower(), factor_map.get(f, f.replace("_", " ").title()))
            if canonical is None:
                continue
            all_factors[canonical] = all_factors.get(canonical, 0) + 1

    # Merge similar
    merged_factors = {}
    for k, v in all_factors.items():
        if k in merged_factors:
            merged_factors[k] += v
        else:
            merged_factors[k] = v

    # --- (a) Factor frequency ---
    ax_a = fig.add_subplot(gs[0, :])
    panel_label(ax_a, "a", x=-0.06)
    sorted_factors = sorted(merged_factors.items(), key=lambda x: x[1], reverse=True)[:12]
    factor_names = [f[0] for f in sorted_factors][::-1]
    factor_counts = [f[1] for f in sorted_factors][::-1]
    y_pos = np.arange(len(factor_names))
    # Color by category
    factor_colors = []
    for fn in factor_names:
        fn_lower = fn.lower()
        if "signal isolation" in fn_lower or "isolated" in fn_lower:
            factor_colors.append(PALETTE["sampling"])
        elif "rain" in fn_lower or "flow" in fn_lower or "cso" in fn_lower or "precip" in fn_lower or "discharge" in fn_lower or "weather" in fn_lower:
            factor_colors.append(PALETTE["environmental"])
        elif "hosp" in fn_lower or "epidemic" in fn_lower or "wastewater trend" in fn_lower:
            factor_colors.append(PALETTE["epidemic"])
        elif "variant" in fn_lower:
            factor_colors.append(PALETTE["mixed"])
        else:
            factor_colors.append(PALETTE["uncertain"])
    bars = ax_a.barh(y_pos, factor_counts, height=0.6, color=factor_colors,
                     edgecolor="white", linewidth=0.5, zorder=2)
    for bar, val in zip(bars, factor_counts):
        ax_a.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                  str(val), va="center", fontsize=6.5)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(factor_names, fontsize=6.5)
    ax_a.set_xlabel(f"Frequency across {len(inv)} events")
    ax_a.set_title("Primary factors identified by the agent", pad=10)

    # --- (b) Classification by state  - stacked horizontal bars ---
    ax_b = fig.add_subplot(gs[1, 0])
    panel_label(ax_b, "b")
    # Extract state from site_id
    def extract_state(site_id):
        parts = str(site_id).split("_")
        for p in parts:
            if len(p) == 2 and p.isalpha():
                return p.upper()
        return "??"
    inv["state"] = inv["site_id"].apply(extract_state)
    state_class = pd.crosstab(inv["state"], inv["classification"]).reindex(
        columns=CLASS_ORDER, fill_value=0
    )
    # Sort by total events
    state_class["total"] = state_class.sum(axis=1)
    state_class = state_class.sort_values("total", ascending=True)
    states = state_class.index.tolist()
    y_pos = np.arange(len(states))
    left = np.zeros(len(states))
    for cls in CLASS_ORDER:
        vals = state_class[cls].values.astype(float)
        ax_b.barh(y_pos, vals, left=left, height=0.65, color=PALETTE[cls],
                  edgecolor="white", linewidth=0.3, label=CLASS_LABELS[cls])
        left += vals
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(states, fontsize=6.5)
    ax_b.set_xlabel("Events")
    ax_b.set_title("Classification by state", pad=8)
    ax_b.legend(loc="lower right", frameon=False, fontsize=5.5,
                ncol=2, handlelength=1.0, handletextpad=0.3)

    # --- (c) Temporal distribution ---
    ax_c = fig.add_subplot(gs[1, 1])
    panel_label(ax_c, "c")
    inv["peak_dt"] = pd.to_datetime(inv["anomaly_date"])
    inv["year_month"] = inv["peak_dt"].dt.to_period("Q")
    # Group by quarter and class
    temporal = inv.groupby(["year_month", "classification"]).size().unstack(fill_value=0)
    temporal = temporal.reindex(columns=CLASS_ORDER, fill_value=0)
    quarters = temporal.index.tolist()
    x_pos = np.arange(len(quarters))
    bottom = np.zeros(len(quarters))
    for cls in CLASS_ORDER:
        vals = temporal[cls].values.astype(float)
        ax_c.bar(x_pos, vals, bottom=bottom, width=0.7, color=PALETTE[cls],
                 edgecolor="white", linewidth=0.3)
        bottom += vals
    # Show every 4th label
    tick_labels = [str(q) if i % 2 == 0 else "" for i, q in enumerate(quarters)]
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=5.5)
    ax_c.set_ylabel("Events")
    ax_c.set_title("Temporal distribution (quarterly)", pad=8)

    fig.savefig(GENERATED_DIR / "fig_behavior.pdf")
    plt.close(fig)
    print("  - fig_behavior.pdf")



# ===================================================================
# FIGURE 8 - Inter-expert agreement and agent strengths (4 panels)
# ===================================================================
def build_fig8_agreement(inv: pd.DataFrame, silver: pd.DataFrame):
    """
    a) Expert reference label to agent transition heatmap
    b) Inter-expert agreement distribution (histogram of agreement_ratio)
    c) Agent vs strongest baseline accuracy on subsets (grouped bar)
    d) Agent-correct / strongest-baseline-wrong events by class
    """
    best_baseline = get_best_baseline_method(
        pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "evaluation_summary.csv")
    )

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.38,
                           left=0.10, right=0.96, top=0.94, bottom=0.08)

    # --- (a) Expert reference label to Agent transition heatmap ---
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")
    merged = silver[["event_id", "ground_truth_label"]].merge(
        inv[["event_id", "classification"]], on="event_id", how="inner"
    )
    silver_classes = CLASS_ORDER
    trans = pd.crosstab(merged["ground_truth_label"], merged["classification"]).reindex(
        index=silver_classes, columns=CLASS_ORDER, fill_value=0
    )
    cmap_sa = LinearSegmentedColormap.from_list("wgreen", ["#FFFFFF", "#5BA05B"])
    im = ax_a.imshow(trans.to_numpy(), cmap=cmap_sa, aspect="auto", vmin=0)
    ax_a.set_xticks(np.arange(len(CLASS_ORDER)))
    ax_a.set_xticklabels([CLASS_LABELS[c] for c in CLASS_ORDER], rotation=30, ha="right")
    ax_a.set_yticks(np.arange(len(silver_classes)))
    ax_a.set_yticklabels([CLASS_LABELS[c] for c in silver_classes])
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            v = int(trans.iloc[i, j])
            if v > 0:
                color = "white" if v > trans.to_numpy().max() * 0.55 else "#333333"
                ax_a.text(j, i, str(v), ha="center", va="center",
                          fontsize=7, fontweight="medium", color=color)
    ax_a.set_xlabel("Agent classification")
    ax_a.set_ylabel("Expert reference label")
    ax_a.set_title("Reference label to agent transition", pad=8)
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.06, shrink=0.85)
    cbar.ax.tick_params(labelsize=5.5)

    # --- (b) Inter-expert agreement distribution ---
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    agreement = silver["agreement_ratio"].dropna()
    bins = [0, 0.35, 0.5, 0.7, 0.85, 1.01]
    bin_labels = ["<0.35", "0.35-0.5", "0.5-0.7", "0.7-0.85", "0.85-1.0"]
    counts, _ = np.histogram(agreement, bins=bins)
    x_pos = np.arange(len(bin_labels))
    gradient_colors = ["#C85050", "#E8A838", "#8C8C8C", "#7BA7CC", "#4878A8"]
    bars = ax_b.bar(x_pos, counts, color=gradient_colors, width=0.65,
                    edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, counts):
        if val > 0:
            ax_b.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                      str(val), ha="center", va="bottom", fontsize=6.5)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(bin_labels, fontsize=6.5)
    ax_b.set_xlabel("Agreement ratio")
    ax_b.set_ylabel("Events")
    ax_b.set_title("Inter-expert agreement distribution", pad=8)
    # Annotate consensus status counts
    status_counts = silver["consensus_status"].value_counts()
    status_text = " | ".join([f"{k}: {v}" for k, v in status_counts.items()])
    ax_b.text(0.5, 0.95, status_text, transform=ax_b.transAxes,
              fontsize=5.5, ha="center", va="top", color="#666666",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                        edgecolor="#DDDDDD", linewidth=0.5))

    # --- (c) Agent vs strongest baseline accuracy on subsets ---
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "c")
    agent_pred = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / "agent_predictions.csv")
    rf_pred = pd.read_csv(PROJECT_ROOT / "outputs" / "evaluation" / f"{best_baseline}_predictions.csv")
    m = silver[["event_id", "ground_truth_label", "consensus_status"]].merge(
        agent_pred[["event_id", "y_pred"]].rename(columns={"y_pred": "agent"}),
        on="event_id"
    ).merge(
        rf_pred[["event_id", "y_pred"]].rename(columns={"y_pred": "rf"}),
        on="event_id"
    ).merge(
        inv[["event_id", "summary"]], on="event_id"
    )
    subsets = {
        "All\n(n={n})": m,
        "Disagreement\n(n={n})": m[m["consensus_status"] == "disagreement"],
        "Accepted\nsampling\n(n={n})": m[(m["consensus_status"] == "accepted") & (m["ground_truth_label"] == "sampling")],
        "Epidemic +\nenvironmental\n(n={n})": m[m["ground_truth_label"].isin(["epidemic", "environmental"])],
    }
    subset_names = []
    agent_acc = []
    rf_acc = []
    for name_tmpl, df in subsets.items():
        n = len(df)
        name = name_tmpl.format(n=n)
        subset_names.append(name)
        agent_acc.append(float((df["agent"] == df["ground_truth_label"]).mean()) if n > 0 else 0)
        rf_acc.append(float((df["rf"] == df["ground_truth_label"]).mean()) if n > 0 else 0)

    x_pos = np.arange(len(subset_names))
    w = 0.32
    bars_agent = ax_c.bar(x_pos - w / 2, agent_acc, w, color="#C85050",
                           edgecolor="white", linewidth=0.5, label="WBE-Agent", zorder=2)
    bars_rf = ax_c.bar(x_pos + w / 2, rf_acc, w, color=METHOD_COLORS[best_baseline],
                        edgecolor="white", linewidth=0.5, label=METHOD_NAMES[best_baseline], zorder=2)
    for bar, val in zip(bars_agent, agent_acc):
        ax_c.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                  f"{val:.2f}", ha="center", va="bottom", fontsize=5.5, color="#C85050")
    for bar, val in zip(bars_rf, rf_acc):
        ax_c.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                  f"{val:.2f}", ha="center", va="bottom", fontsize=5.5, color=METHOD_COLORS[best_baseline])
    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(subset_names, fontsize=5.5)
    ax_c.set_ylabel("Accuracy")
    ax_c.set_ylim(0, 1.05)
    ax_c.legend(frameon=False, fontsize=6, loc="upper right")
    ax_c.set_title("Agent advantage on operational subsets", pad=8)

    # --- (d) Agent-correct / strongest-baseline-wrong by class ---
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")
    wins = m[(m["agent"] == m["ground_truth_label"]) & (m["rf"] != m["ground_truth_label"])]
    losses = m[(m["agent"] != m["ground_truth_label"]) & (m["rf"] == m["ground_truth_label"])]
    win_classes = CLASS_ORDER
    win_counts = wins["ground_truth_label"].value_counts().reindex(win_classes, fill_value=0)
    loss_counts = losses["ground_truth_label"].value_counts().reindex(win_classes, fill_value=0)
    y_pos = np.arange(len(win_classes))
    h = 0.32
    ax_d.barh(y_pos - h / 2, win_counts.values, height=h, color="#C85050",
              edgecolor="white", linewidth=0.5, label="Agent correct", zorder=2)
    ax_d.barh(y_pos + h / 2, -loss_counts.values, height=h, color=METHOD_COLORS[best_baseline],
              edgecolor="white", linewidth=0.5, label=f"{METHOD_NAMES[best_baseline]} correct", zorder=2)
    # Annotations
    for i, (wv, lv) in enumerate(zip(win_counts.values, loss_counts.values)):
        if wv > 0:
            ax_d.text(wv + 0.3, i - h / 2, str(wv), va="center", fontsize=6, color="#C85050")
        if lv > 0:
            ax_d.text(-lv - 0.3, i + h / 2, str(lv), va="center", fontsize=6,
                      color=METHOD_COLORS[best_baseline], ha="right")
    ax_d.axvline(0, color="#333333", linewidth=0.6)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels([CLASS_LABELS[c] for c in win_classes], fontsize=7)
    ax_d.set_xlabel("Events")
    ax_d.legend(frameon=False, fontsize=5.5, loc="lower right")
    ax_d.set_title(f"Head-to-head: Agent vs {METHOD_NAMES[best_baseline]}", pad=8)

    fig.savefig(GENERATED_DIR / "fig_strengths.pdf")
    plt.close(fig)
    print("  - fig_strengths.pdf")



# ===================================================================
# FIGURE 7 - Reliability and resource profile of agentic investigation
# ===================================================================
def build_fig7_reliability(inv: pd.DataFrame, silver: pd.DataFrame):
    """
    a) Confidence calibration: binned confidence vs observed accuracy
    b) Confidence by correctness
    c) Tool calls by predicted class
    d) Token usage by predicted class
    """
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35,
                           left=0.10, right=0.96, top=0.94, bottom=0.08)

    class_order = ["sampling", "environmental", "epidemic", "uncertain", "mixed"]
    m = silver[["event_id", "ground_truth_label"]].merge(
        inv[["event_id", "classification", "confidence", "tool_calls_count", "total_tokens"]],
        on="event_id"
    )
    m["correct"] = (m["classification"] == m["ground_truth_label"]).astype(int)

    # --- (a) Confidence calibration ---
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")
    bins = [0.65, 0.80, 0.90, 0.97, 1.01]
    bin_centers = []
    bin_acc = []
    bin_sizes = []
    for i in range(len(bins) - 1):
        mask = (m["confidence"] >= bins[i]) & (m["confidence"] < bins[i + 1])
        subset = m[mask]
        if len(subset) > 0:
            bin_centers.append(subset["confidence"].mean())
            bin_acc.append(subset["correct"].mean())
            bin_sizes.append(len(subset))
    ax_a.plot([0.65, 1.0], [0.65, 1.0], "--", color="#BBBBBB", linewidth=0.8,
              label="Perfect calibration", zorder=1)
    ax_a.plot(bin_centers, bin_acc, "o-", color="#C85050", markersize=6,
              linewidth=1.2, zorder=3, label="WBE-Agent")
    for bc, ba, bs in zip(bin_centers, bin_acc, bin_sizes):
        ax_a.annotate(f"n={bs}", (bc, ba), textcoords="offset points",
                      xytext=(0, 8), fontsize=5.5, ha="center", color="#666666")
    ax_a.set_xlabel("Mean predicted confidence")
    ax_a.set_ylabel("Observed accuracy")
    ax_a.set_xlim(0.65, 1.0)
    ax_a.set_ylim(0.45, 1.0)
    ax_a.legend(frameon=False, fontsize=6, loc="lower right")
    ax_a.set_title("Confidence calibration", pad=8)
    ax_a.set_aspect("equal")

    # --- (b) Confidence by correctness ---
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")
    conf_groups = [
        m.loc[m["correct"] == 0, "confidence"].dropna().values,
        m.loc[m["correct"] == 1, "confidence"].dropna().values,
    ]
    group_labels = ["Incorrect", "Correct"]
    group_colors = ["#8C8C8C", "#C85050"]
    bp = ax_b.boxplot(conf_groups, positions=[0, 1], patch_artist=True, showfliers=False,
                      widths=0.5, medianprops=dict(color="white", linewidth=1.2))
    rng = np.random.default_rng(42)
    for i, (patch, vals, color) in enumerate(zip(bp["boxes"], conf_groups, group_colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
        if len(vals) > 0:
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax_b.scatter(i + jitter, vals, s=7, alpha=0.35, color=color,
                         edgecolors="none", zorder=3)
            ax_b.text(i, min(vals.mean() + 0.03, 0.99), f"mean={vals.mean():.2f}",
                      ha="center", va="bottom", fontsize=6, color=color)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(group_labels)
    ax_b.set_ylabel("Confidence")
    ax_b.set_ylim(0.65, 1.01)
    ax_b.set_title("Correct outputs carried higher confidence", pad=8)
    ax_b.grid(axis="y", color="#e8edf2", linewidth=0.6)
    ax_b.set_axisbelow(True)

    # --- (c) Tool calls by class ---
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "c")
    tool_data = [inv.loc[inv["classification"] == cls, "tool_calls_count"].dropna().values for cls in class_order]
    bp = ax_c.boxplot(tool_data, positions=range(len(class_order)), patch_artist=True,
                      showfliers=False, widths=0.5,
                      medianprops=dict(color="white", linewidth=1.2))
    for i, (patch, cls) in enumerate(zip(bp["boxes"], class_order)):
        patch.set_facecolor(PALETTE[cls])
        patch.set_alpha(0.5)
        patch.set_edgecolor(PALETTE[cls])
        vals = tool_data[i]
        if len(vals) > 0:
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax_c.scatter(i + jitter, vals, s=6, alpha=0.4,
                         color=PALETTE[cls], edgecolors="none", zorder=3)
    ax_c.set_xticks(range(len(class_order)))
    ax_c.set_xticklabels([CLASS_LABELS[c] for c in class_order], rotation=25, ha="right")
    ax_c.set_ylabel("Tool calls per event")
    ax_c.set_title("Investigation depth varied by class", pad=8)
    ax_c.grid(axis="y", color="#e8edf2", linewidth=0.6)
    ax_c.set_axisbelow(True)

    # --- (d) Token usage by class ---
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")
    tok_data = [inv.loc[inv["classification"] == cls, "total_tokens"].dropna().values / 1000 for cls in class_order]
    bp2 = ax_d.boxplot(tok_data, positions=range(len(class_order)), patch_artist=True,
                       showfliers=False, widths=0.5,
                       medianprops=dict(color="white", linewidth=1.2))
    for i, (patch, cls) in enumerate(zip(bp2["boxes"], class_order)):
        patch.set_facecolor(PALETTE[cls])
        patch.set_alpha(0.5)
        patch.set_edgecolor(PALETTE[cls])
        vals = tok_data[i]
        if len(vals) > 0:
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax_d.scatter(i + jitter, vals, s=6, alpha=0.4,
                         color=PALETTE[cls], edgecolors="none", zorder=3)
    ax_d.set_xticks(range(len(class_order)))
    ax_d.set_xticklabels([CLASS_LABELS[c] for c in class_order], rotation=25, ha="right")
    ax_d.set_ylabel("Tokens (x1,000)")
    ax_d.set_title("Token budgets rose on harder cases", pad=8)
    ax_d.grid(axis="y", color="#e8edf2", linewidth=0.6)
    ax_d.set_axisbelow(True)

    fig.savefig(GENERATED_DIR / "fig_reliability.pdf")
    plt.close(fig)
    print("  - fig_reliability.pdf")


# ===================================================================
# FIGURE 1 - Study site map with climate zones and data coverage
# ===================================================================

# Simplified US continental boundary (lon, lat) for matplotlib polygon
_US_OUTLINE = np.array([
    (-124.7, 48.4), (-123.1, 48.4), (-123.1, 46.2), (-124.5, 43.0),
    (-124.4, 40.3), (-120.0, 34.5), (-117.2, 32.5), (-114.6, 32.7),
    (-111.1, 31.3), (-108.2, 31.3), (-106.5, 31.8), (-104.0, 30.0),
    (-103.0, 29.0), (-101.4, 29.8), (-99.1, 26.4), (-97.2, 25.9),
    (-97.2, 27.8), (-94.0, 29.7), (-90.0, 29.0), (-89.0, 29.2),
    (-88.8, 30.4), (-85.0, 29.7), (-83.0, 29.0), (-81.0, 25.1),
    (-80.0, 25.2), (-80.1, 27.0), (-80.8, 30.7), (-81.2, 31.5),
    (-80.8, 32.1), (-79.0, 33.5), (-75.5, 35.2), (-75.5, 37.0),
    (-76.0, 38.0), (-75.0, 38.5), (-74.0, 39.5), (-73.9, 40.5),
    (-72.0, 41.0), (-71.0, 41.5), (-70.0, 41.5), (-69.9, 43.0),
    (-67.0, 44.8), (-67.0, 47.3), (-69.0, 47.4), (-70.7, 46.2),
    (-71.1, 45.3), (-74.7, 45.0), (-76.8, 43.6), (-79.0, 43.2),
    (-79.0, 42.5), (-82.5, 41.7), (-83.5, 41.7), (-84.8, 41.7),
    (-84.8, 46.5), (-88.4, 48.3), (-89.5, 48.0), (-94.7, 48.7),
    (-95.2, 49.0), (-104.0, 49.0), (-116.0, 49.0), (-123.3, 49.0),
    (-124.7, 48.4),
])

CLIMATE_COLORS = {
    "continental_north": "#4878A8",
    "continental_central": "#7BA7CC",
    "subtropical_south": "#E8A838",
    "marine_west": "#5BA05B",
    "arid_semiarid": "#C85050",
}
CLIMATE_LABELS_MAP = {
    "continental_north": "Continental north",
    "continental_central": "Continental central",
    "subtropical_south": "Subtropical south",
    "marine_west": "Marine west",
    "arid_semiarid": "Arid / semi-arid",
}
TIER_MARKERS = {"core": "o", "extension": "s", "challenge": "D"}
TIER_SIZES = {"core": 70, "extension": 55, "challenge": 55}


def _load_sites():
    """Load and concatenate all site tier CSVs."""
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
    return pd.concat(frames, ignore_index=True)


def build_fig1_map():
    """
    US map with 16 study sites colored by climate zone, shaped by tier,
    plus compact data-coverage timeline bars.
    """
    sites = _load_sites()

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.05,
                           left=0.02, right=0.98, top=0.94, bottom=0.06)

    # --- (a) Map ---
    ax_map = fig.add_subplot(gs[0, 0])
    panel_label(ax_map, "a", x=-0.02, y=1.04)

    # Draw US outline
    from matplotlib.patches import Polygon as MplPolygon
    us_poly = MplPolygon(_US_OUTLINE, closed=True, facecolor="#F0F0F0",
                         edgecolor="#AAAAAA", linewidth=0.8, zorder=1)
    ax_map.add_patch(us_poly)

    # Plot sites
    for _, row in sites.iterrows():
        tier = row["tier"]
        climate = row["climate_bucket"]
        ax_map.scatter(
            row["longitude"], row["latitude"],
            c=CLIMATE_COLORS.get(climate, "#888888"),
            marker=TIER_MARKERS.get(tier, "o"),
            s=TIER_SIZES.get(tier, 50),
            edgecolors="white", linewidth=0.8, zorder=3,
        )
        # State label with offset
        offset_x, offset_y = 0.6, 0.5
        ax_map.annotate(
            row["state"], (row["longitude"], row["latitude"]),
            xytext=(offset_x, offset_y), textcoords="offset points",
            fontsize=6, fontweight="bold", color="#333333",
            ha="left", va="bottom",
        )

    ax_map.set_xlim(-128, -64)
    ax_map.set_ylim(24, 50.5)
    ax_map.set_aspect(1.3)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.set_title("Study sites across five climate zones", pad=8, fontsize=9)

    # Climate legend
    climate_handles = [
        mpatches.Patch(facecolor=CLIMATE_COLORS[k], edgecolor="white",
                       label=CLIMATE_LABELS_MAP[k])
        for k in CLIMATE_COLORS
    ]
    # Tier legend
    import matplotlib.lines as mlines
    tier_handles = [
        mlines.Line2D([], [], marker=TIER_MARKERS[t], color="w",
                      markerfacecolor="#888888", markersize=6,
                      markeredgecolor="white", markeredgewidth=0.6,
                      label=f"{t.capitalize()} site")
        for t in ["core", "extension", "challenge"]
    ]
    leg1 = ax_map.legend(handles=climate_handles, loc="lower left",
                         bbox_to_anchor=(0.0, 0.0), frameon=False,
                         fontsize=5.5, handlelength=1.2, handletextpad=0.4,
                         title="Climate zone", title_fontsize=6)
    ax_map.add_artist(leg1)
    ax_map.legend(handles=tier_handles, loc="lower left",
                  bbox_to_anchor=(0.0, 0.22), frameon=False,
                  fontsize=5.5, handlelength=1.2, handletextpad=0.4,
                  title="Site tier", title_fontsize=6)

    # --- (b) Data coverage timeline ---
    ax_cov = fig.add_subplot(gs[0, 1])
    panel_label(ax_cov, "b", x=-0.08, y=1.04)

    # Sort sites by date_min
    sites["date_min_dt"] = pd.to_datetime(sites["date_min"])
    sites["date_max_dt"] = pd.to_datetime(sites["date_max"])
    if "site_days" not in sites.columns:
        sites["site_days"] = (sites["date_max_dt"] - sites["date_min_dt"]).dt.days + 1
    sites_sorted = sites.sort_values("date_min_dt", ascending=False).reset_index(drop=True)

    global_min = sites_sorted["date_min_dt"].min()
    global_max = sites_sorted["date_max_dt"].max()
    total_days = (global_max - global_min).days

    y_pos = np.arange(len(sites_sorted))
    bar_height = 0.6

    for i, (_, row) in enumerate(sites_sorted.iterrows()):
        x_start = (row["date_min_dt"] - global_min).days / total_days
        x_width = (row["date_max_dt"] - row["date_min_dt"]).days / total_days
        climate = row["climate_bucket"]
        color = CLIMATE_COLORS.get(climate, "#888888")

        # Main wastewater bar
        ax_cov.barh(i, x_width, left=x_start, height=bar_height,
                    color=color, edgecolor="white", linewidth=0.3,
                    alpha=0.85, zorder=2)

    ax_cov.set_yticks(y_pos)
    ax_cov.set_yticklabels(
        [f"{r['state']} ({r['tier'][0].upper()})" for _, r in sites_sorted.iterrows()],
        fontsize=5.5,
    )
    # X-axis as years
    year_ticks = []
    year_labels = []
    for yr in range(2020, 2026):
        frac = (pd.Timestamp(f"{yr}-01-01") - global_min).days / total_days
        if 0 <= frac <= 1:
            year_ticks.append(frac)
            year_labels.append(str(yr))
    ax_cov.set_xticks(year_ticks)
    ax_cov.set_xticklabels(year_labels, fontsize=6)
    ax_cov.set_xlim(-0.02, 1.12)
    ax_cov.set_ylim(-0.5, len(sites_sorted) - 0.5)
    ax_cov.set_title("Coverage timeline", pad=8, fontsize=9)
    ax_cov.spines["left"].set_visible(False)
    ax_cov.tick_params(axis="y", length=0)

    # Add wastewater-observation and site-day annotations
    for i, (_, row) in enumerate(sites_sorted.iterrows()):
        x_end = ((row["date_max_dt"] - global_min).days / total_days)
        ax_cov.text(x_end + 0.015, i, f"obs={int(row['n_obs'])}\nsd={int(row['site_days'])}",
                    va="center", fontsize=4.3, color="#666666", linespacing=0.9)

    fig.savefig(GENERATED_DIR / "fig_map.pdf")
    plt.close(fig)
    print("  - fig_map.pdf")


# ===================================================================
# FIGURE 2 - Agent workflow / architecture diagram
# ===================================================================
def build_fig2_workflow():
    """
    Architecture diagram: data sources  to anomaly detection  to ReAct loop
    with 6 tools  to structured output.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    from matplotlib.patches import FancyBboxPatch

    def draw_box(x, y, w, h, text, color="#4878A8", alpha=0.15,
                 fontsize=7, fontweight="normal", text_color="#333333",
                 rounded=True):
        style = "round,pad=0.08" if rounded else "square,pad=0.02"
        box = FancyBboxPatch((x, y), w, h, boxstyle=style,
                             facecolor=color, alpha=alpha,
                             edgecolor=color, linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color,
                linespacing=1.4)

    def draw_arrow(x1, y1, x2, y2, color="#666666"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.0, shrinkA=2, shrinkB=2))

    # ---- Column 1: Data Sources ----
    ax.text(0.65, 5.2, "Data Sources", ha="center", fontsize=8,
            fontweight="bold", color="#333333")
    sources = [
        ("Wastewater\n(NWSS)", "#4878A8"),
        ("Hospitalization\n(HHS)", "#C85050"),
        ("Weather\n(NOAA)", "#5BA05B"),
        ("Hydrology\n(USGS)", "#E8A838"),
        ("Variants\n(CDC)", "#8C8C8C"),
    ]
    for i, (label, color) in enumerate(sources):
        y = 4.6 - i * 0.85
        draw_box(0.05, y, 1.2, 0.65, label, color=color, alpha=0.2,
                 fontsize=5.5)

    # ---- Column 2: Unified Database + Anomaly Detection ----
    draw_arrow(1.35, 2.85, 1.7, 2.85)

    draw_box(1.7, 2.2, 1.5, 1.3, "Unified\nDatabase\n(18,810 rows\n166 variables)",
             color="#4878A8", alpha=0.12, fontsize=6)

    draw_arrow(2.45, 2.2, 2.45, 1.55)

    draw_box(1.7, 0.85, 1.5, 0.7, "Anomaly Detection\n(5-method ensemble)",
             color="#E8A838", alpha=0.2, fontsize=6)

    draw_arrow(3.3, 1.2, 3.65, 1.2)

    draw_box(3.65, 0.85, 1.1, 0.7, "Event Catalog\n(199 events)",
             color="#C85050", alpha=0.15, fontsize=6)

    # ---- Column 3: ReAct Loop ----
    draw_arrow(4.2, 1.55, 4.2, 2.1)

    # ReAct container
    react_box = FancyBboxPatch((3.5, 2.1), 3.2, 3.0,
                                boxstyle="round,pad=0.12",
                                facecolor="#F8F4F0", alpha=0.5,
                                edgecolor="#C85050", linewidth=1.2,
                                linestyle="--")
    ax.add_patch(react_box)
    ax.text(5.1, 4.95, "ReAct Investigation Loop", ha="center",
            fontsize=8, fontweight="bold", color="#C85050")

    # Thought  to Action  to Observation cycle
    draw_box(3.75, 4.0, 1.1, 0.6, "Thought\n(reasoning)", color="#4878A8",
             alpha=0.2, fontsize=6)
    draw_arrow(4.85, 4.3, 5.25, 4.3)
    draw_box(5.25, 4.0, 1.1, 0.6, "Action\n(tool call)", color="#E8A838",
             alpha=0.2, fontsize=6)
    draw_arrow(5.8, 4.0, 5.8, 3.45)
    draw_box(4.35, 2.85, 1.5, 0.55, "Observation\n(tool result)", color="#5BA05B",
             alpha=0.2, fontsize=6)
    # Loop arrow back
    ax.annotate("", xy=(4.3, 4.0), xytext=(4.6, 3.4),
                arrowprops=dict(arrowstyle="-|>", color="#888888",
                                lw=0.8, connectionstyle="arc3,rad=0.3"))

    # Iteration label
    ax.text(3.85, 3.2, "up to 10\niterations", fontsize=5, color="#888888",
            ha="center", va="center", style="italic")

    # ---- Column 3.5: Tools ----
    ax.text(8.3, 5.2, "Domain Tools", ha="center", fontsize=8,
            fontweight="bold", color="#333333")
    tools = [
        "Site metadata",
        "Weather lookup",
        "Hydrology query",
        "Hospitalization",
        "Nearby sites",
        "Variant surveillance",
    ]
    tool_colors = ["#8C8C8C", "#5BA05B", "#E8A838", "#C85050",
                   "#4878A8", "#8C8C8C"]
    for i, (tool, tc) in enumerate(zip(tools, tool_colors)):
        y = 4.55 - i * 0.62
        draw_box(7.4, y, 1.8, 0.45, tool, color=tc, alpha=0.18,
                 fontsize=5.5)

    # Arrow from Action to Tools
    draw_arrow(6.35, 4.3, 7.35, 4.3)
    # Arrow from Tools back to Observation
    draw_arrow(7.35, 2.5, 5.85, 3.1)

    # ---- Column 4: Structured Output ----
    draw_arrow(5.1, 2.1, 5.1, 0.95)

    draw_box(3.65, 0.05, 3.0, 0.8,
             "Structured Output\nclassification | confidence | factors\n"
             "reasoning chain | recommendation | data gaps",
             color="#5BA05B", alpha=0.15, fontsize=5.5)

    fig.savefig(GENERATED_DIR / "fig_workflow.pdf")
    plt.close(fig)
    print("  - fig_workflow.pdf")


# ===================================================================
# FIGURE 4  - Case study panels (4 events x 3 data streams)
# ===================================================================

# Case study event definitions
CASE_EVENTS = [
    {"event_id": "EVT-00001", "state": "FL", "date": "2023-12-27",
     "label": "Uncertain", "color": PALETTE["uncertain"],
     "note": "Rain + JN.1 shift + admissions rise"},
    {"event_id": "EVT-00002", "state": "FL", "date": "2024-03-12",
     "label": "Sampling", "color": PALETTE["sampling"],
     "note": "Hosp. decline, no rain, no nearby support"},
    {"event_id": "EVT-00004", "state": "FL", "date": "2024-08-08",
     "label": "Environmental", "color": PALETTE["environmental"],
     "note": "44.4 mm rain + rising discharge"},
    {"event_id": "EVT-00078", "state": "NV", "date": "2022-06-20",
     "label": "Epidemic", "color": PALETTE["epidemic"],
     "note": "No rain, flat flow, admissions 43\u219281"},
]


def build_fig4_cases():
    """
    4 rows (one per case event) x 3 columns (wastewater, environment,
    hospitalization) with aligned time series and event markers.
    """
    merged = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "merged_multisource.parquet")
    merged["date"] = pd.to_datetime(merged["date"])

    fig = plt.figure(figsize=(7.2, 8.5))
    gs = gridspec.GridSpec(4, 3, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.97, top=0.95, bottom=0.04)

    col_titles = ["Wastewater concentration", "Environmental context",
                  "Hospitalization"]

    for row_idx, case in enumerate(CASE_EVENTS):
        event_date = pd.Timestamp(case["date"])
        state = case["state"]
        window_start = event_date - pd.Timedelta(days=30)
        window_end = event_date + pd.Timedelta(days=30)

        # Filter data for this state and window
        mask = (
            (merged["state"] == state)
            & (merged["date"] >= window_start)
            & (merged["date"] <= window_end)
        )
        window = merged[mask].sort_values("date").copy()

        # Row label
        row_label = f"{case['event_id']}\n{state}, {case['date']}\n{case['label']}"

        # --- Column 0: Wastewater ---
        ax_ww = fig.add_subplot(gs[row_idx, 0])
        if row_idx == 0:
            ax_ww.set_title(col_titles[0], fontsize=8, fontweight="bold", pad=6)
        panel_label(ax_ww, chr(ord("a") + row_idx * 3), x=-0.18, y=1.12)

        if len(window) > 0 and "pcr_conc_lin" in window.columns:
            vals = window["pcr_conc_lin"].dropna()
            dates = window.loc[vals.index, "date"]
            ax_ww.plot(dates, vals / 1e6, color="#4878A8", linewidth=1.0,
                       marker="o", markersize=2.5, zorder=2)
            ax_ww.fill_between(dates, 0, vals / 1e6, alpha=0.08,
                               color="#4878A8")
        ax_ww.axvline(event_date, color=case["color"], linewidth=1.5,
                      linestyle="--", alpha=0.8, zorder=3)
        ax_ww.set_ylabel("Conc. (\u00d710\u2076)", fontsize=6)
        ax_ww.tick_params(axis="x", rotation=30, labelsize=5)
        ax_ww.tick_params(axis="y", labelsize=5.5)

        # Add case label in top-left
        ax_ww.text(0.02, 0.95, row_label, transform=ax_ww.transAxes,
                   fontsize=5, va="top", ha="left", color=case["color"],
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                             edgecolor=case["color"], alpha=0.85, linewidth=0.5))

        # --- Column 1: Environmental (precip + discharge) ---
        ax_env = fig.add_subplot(gs[row_idx, 1])
        if row_idx == 0:
            ax_env.set_title(col_titles[1], fontsize=8, fontweight="bold", pad=6)
        panel_label(ax_env, chr(ord("a") + row_idx * 3 + 1), x=-0.18, y=1.12)

        has_precip = len(window) > 0 and window["precipitation_mm"].notna().any()
        has_discharge = len(window) > 0 and window["discharge_cfs"].notna().any()

        if has_precip:
            precip = window[["date", "precipitation_mm"]].dropna(subset=["precipitation_mm"])
            ax_env.bar(precip["date"], precip["precipitation_mm"],
                       width=0.8, color="#5BA05B", alpha=0.6, zorder=2,
                       label="Precip. (mm)")
            ax_env.set_ylabel("Precip. (mm)", fontsize=6, color="#5BA05B")

        if has_discharge:
            ax_q = ax_env.twinx()
            q_data = window[["date", "discharge_percentile"]].dropna(
                subset=["discharge_percentile"])
            ax_q.plot(q_data["date"], q_data["discharge_percentile"] * 100,
                      color="#E8A838", linewidth=1.0, marker="s",
                      markersize=2, zorder=2, label="Discharge pctl.")
            ax_q.axhline(90, color="#E8A838", linewidth=0.5, linestyle=":",
                         alpha=0.5)
            ax_q.set_ylabel("Discharge pctl.", fontsize=6, color="#E8A838")
            ax_q.set_ylim(0, 105)
            ax_q.tick_params(axis="y", labelsize=5.5, colors="#E8A838")
            ax_q.spines["right"].set_visible(True)
            ax_q.spines["right"].set_color("#E8A838")
            ax_q.spines["right"].set_linewidth(0.5)

        if not has_precip and not has_discharge:
            ax_env.text(0.5, 0.5, "No weather/hydrology\ndata available",
                        transform=ax_env.transAxes, ha="center", va="center",
                        fontsize=7, color="#AAAAAA", style="italic")

        ax_env.axvline(event_date, color=case["color"], linewidth=1.5,
                       linestyle="--", alpha=0.8, zorder=3)
        ax_env.tick_params(axis="x", rotation=30, labelsize=5)
        ax_env.tick_params(axis="y", labelsize=5.5)

        # --- Column 2: Hospitalization ---
        ax_hosp = fig.add_subplot(gs[row_idx, 2])
        if row_idx == 0:
            ax_hosp.set_title(col_titles[2], fontsize=8, fontweight="bold", pad=6)
        panel_label(ax_hosp, chr(ord("a") + row_idx * 3 + 2), x=-0.18, y=1.12)

        hosp_col = "previous_day_admission_adult_covid_confirmed"
        has_hosp = len(window) > 0 and window[hosp_col].notna().any()

        if has_hosp:
            h_data = window[["date", hosp_col]].dropna(subset=[hosp_col])
            ax_hosp.plot(h_data["date"], h_data[hosp_col],
                         color="#C85050", linewidth=1.0, marker="^",
                         markersize=2.5, zorder=2)
            ax_hosp.fill_between(h_data["date"], 0, h_data[hosp_col],
                                 alpha=0.08, color="#C85050")
            ax_hosp.set_ylabel("Admissions", fontsize=6)
        else:
            ax_hosp.text(0.5, 0.5, "No hospitalization\ndata available",
                         transform=ax_hosp.transAxes, ha="center", va="center",
                         fontsize=7, color="#AAAAAA", style="italic")

        ax_hosp.axvline(event_date, color=case["color"], linewidth=1.5,
                        linestyle="--", alpha=0.8, zorder=3)
        ax_hosp.tick_params(axis="x", rotation=30, labelsize=5)
        ax_hosp.tick_params(axis="y", labelsize=5.5)

        # Add short note at bottom-right of hosp panel
        ax_hosp.text(0.98, 0.05, case["note"], transform=ax_hosp.transAxes,
                     fontsize=4.5, va="bottom", ha="right", color="#888888",
                     style="italic")

    fig.savefig(GENERATED_DIR / "fig_cases.pdf")
    plt.close(fig)
    print("  - fig_cases.pdf")


# ===================================================================
# Main
# ===================================================================
def main():
    setup_style()
    print("Loading data...")
    inv, auto, silver, catalog, bench, ablation = load_all()
    print(f"  {len(inv)} investigation results, {len(silver)} reference labels, "
          f"{len(catalog)} catalog events")

    print("\nGenerating figures...")
    build_fig1_map()
    build_fig2_workflow()
    build_fig3_outcomes(inv, auto)
    build_fig4_cases()
    build_fig5_benchmark(bench, silver)
    build_fig7_behavior(inv, catalog)
    build_fig8_agreement(inv, silver)
    build_fig7_reliability(inv, silver)
    build_fig_ablation_current(inv, auto, silver)

    print(f"\nAll figures saved to {GENERATED_DIR}")


if __name__ == "__main__":
    main()


