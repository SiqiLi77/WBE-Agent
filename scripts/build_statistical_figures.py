"""
Generate enhanced Figure 5 (benchmark with bootstrap CI + McNemar significance)
and Figure 8 (reliability with calibration CI + class/correctness significance).

Usage:
    python scripts/build_statistical_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GENERATED = PROJECT_ROOT / "paper" / "generated"
GENERATED.mkdir(parents=True, exist_ok=True)
EVAL_DIR = PROJECT_ROOT / "outputs" / "evaluation"
INV_PATH = PROJECT_ROOT / "outputs" / "investigation_results.csv"

# ── Style ──
AGENT_COLOR = "#C85050"
LR_COLOR = "#4878A8"
GREY = "#8C8C8C"

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

CLASS_ORDER = ["sampling", "mixed", "environmental", "epidemic", "uncertain"]
CLASS_LABELS = {
    "sampling": "Sampling", "mixed": "Mixed", "environmental": "Environmental",
    "epidemic": "Epidemic", "uncertain": "Uncertain",
}
PALETTE = {
    "sampling": "#4878A8", "uncertain": "#8C8C8C", "mixed": "#E8A838",
    "environmental": "#5BA05B", "epidemic": "#C85050",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


def load_predictions():
    """Load all method predictions aligned by event_id."""
    frames = {}
    for m in METHOD_ORDER:
        p = EVAL_DIR / f"{m}_predictions.csv"
        if p.exists():
            df = pd.read_csv(p)
            frames[m] = df.set_index("event_id")["y_pred"]
    return frames


def bootstrap_metric(y_true, y_pred, metric_fn, n_boot=2000, seed=42):
    """Bootstrap a metric, return (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(metric_fn(y_true[idx], y_pred[idx]))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test: returns chi2, p-value."""
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    # b correct & a wrong
    b01 = int(((~correct_a) & correct_b).sum())
    # a correct & b wrong
    b10 = int((correct_a & (~correct_b)).sum())
    n = b01 + b10
    if n == 0:
        return 0.0, 1.0
    # McNemar with continuity correction
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    p = float(sp_stats.chi2.sf(chi2, df=1))
    return chi2, p


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# ===================================================================
# FIGURE 5 — Enhanced benchmark with bootstrap CI + McNemar
# ===================================================================
def build_fig5_enhanced():
    preds = load_predictions()
    if "agent" not in preds:
        print("  ✗ agent predictions not found, skipping fig5")
        return

    # Align all predictions to agent's event set
    agent_events = preds["agent"].index
    y_true_series = pd.read_csv(EVAL_DIR / "agent_predictions.csv").set_index("event_id")["y_true"]
    y_true = y_true_series.loc[agent_events].values

    fig = plt.figure(figsize=(7.2, 7.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38,
                           left=0.12, right=0.96, top=0.95, bottom=0.07)

    # ── (a) Macro-F1 with bootstrap CI ──
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")

    macro_fn = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
    results = []
    for m in METHOD_ORDER:
        if m not in preds:
            continue
        yp = preds[m].reindex(agent_events, fill_value="uncertain").values
        pt, lo, hi = bootstrap_metric(y_true, yp, macro_fn)
        results.append((m, pt, lo, hi))

    y_pos = np.arange(len(results))[::-1]
    for i, (m, pt, lo, hi) in enumerate(results):
        yp_i = y_pos[i]
        col = METHOD_COLORS[m]
        marker = "D" if m == "agent" else "o"
        size = 55 if m == "agent" else 40
        # CI bar
        ax_a.plot([lo, hi], [yp_i, yp_i], color=col, linewidth=2.0, solid_capstyle="round", zorder=2)
        # Point
        ax_a.scatter(pt, yp_i, color=col, s=size, zorder=3, marker=marker,
                     edgecolors="white", linewidth=0.5)
        ax_a.text(hi + 0.015, yp_i, f"{pt:.3f}", va="center", fontsize=6.5,
                  fontweight="bold" if m == "agent" else "normal")

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([METHOD_NAMES[r[0]] for r in results], fontsize=7)
    ax_a.set_xlabel("Macro-F1 (95% bootstrap CI)")
    ax_a.set_xlim(-0.02, 0.95)
    ax_a.set_title("Benchmark: Macro-F1 with uncertainty", pad=8)

    # ── (b) McNemar significance: Agent vs each baseline ──
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")

    agent_pred = preds["agent"].reindex(agent_events, fill_value="uncertain").values
    comparisons = []
    for m in METHOD_ORDER:
        if m == "agent" or m not in preds:
            continue
        yp = preds[m].reindex(agent_events, fill_value="uncertain").values
        chi2, p = mcnemar_test(y_true, agent_pred, yp)
        comparisons.append((m, chi2, p))

    comp_names = [METHOD_NAMES[c[0]] for c in comparisons]
    p_vals = [c[2] for c in comparisons]
    chi2_vals = [c[1] for c in comparisons]
    y_comp = np.arange(len(comparisons))[::-1]

    bars = ax_b.barh(y_comp, [-np.log10(max(p, 1e-20)) for p in p_vals],
                     height=0.55, color=[METHOD_COLORS[c[0]] for c in comparisons],
                     edgecolor="white", linewidth=0.5, zorder=2)
    # Significance thresholds
    for thresh, label, ls in [(1.301, "p=0.05", "--"), (2.0, "p=0.01", ":"), (3.0, "p=0.001", "-.")]:
        ax_b.axvline(thresh, color="#AAAAAA", linestyle=ls, linewidth=0.6, zorder=1)
        ax_b.text(thresh + 0.05, len(comparisons) - 0.3, label, fontsize=5.5, color="#888888", va="top")

    for i, (m, chi2, p) in enumerate(comparisons):
        yp_i = y_comp[i]
        stars = sig_stars(p)
        x_val = -np.log10(max(p, 1e-20))
        ax_b.text(x_val + 0.15, yp_i, f"{stars} (p={p:.3f})" if p >= 0.001 else f"{stars} (p<0.001)",
                  va="center", fontsize=6, color="#333333")

    ax_b.set_yticks(y_comp)
    ax_b.set_yticklabels(comp_names, fontsize=7)
    ax_b.set_xlabel("$-\\log_{10}(p)$ (McNemar's test vs WBE-Agent)")
    ax_b.set_title("Pairwise significance", pad=8)

    # ── (c) Per-class recall: Agent vs LR with bootstrap CI ──
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "c")

    classes_present = [c for c in CLASS_ORDER if c in set(y_true)]
    x_pos = np.arange(len(classes_present))
    width = 0.35

    for offset, m, col, label in [(-width/2, "agent", AGENT_COLOR, "WBE-Agent"),
                                    (width/2, "logistic_regression", LR_COLOR, "Logistic reg.")]:
        if m not in preds:
            continue
        yp = preds[m].reindex(agent_events, fill_value="uncertain").values
        recalls, lo_errs, hi_errs = [], [], []
        for cls in classes_present:
            mask = (y_true == cls)
            if mask.sum() == 0:
                recalls.append(0); lo_errs.append(0); hi_errs.append(0)
                continue
            recall_fn = lambda yt, yp_: (yp_[yt == cls] == cls).mean() if (yt == cls).sum() > 0 else 0.0
            pt, lo, hi = bootstrap_metric(y_true, yp, recall_fn)
            recalls.append(pt)
            lo_errs.append(pt - lo)
            hi_errs.append(hi - pt)

        ax_c.bar(x_pos + offset, recalls, width, yerr=[lo_errs, hi_errs],
                 color=col, alpha=0.85, edgecolor="white", linewidth=0.5,
                 capsize=3, error_kw={"linewidth": 0.8}, label=label, zorder=2)

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels([CLASS_LABELS[c] for c in classes_present], rotation=25, ha="right")
    ax_c.set_ylabel("Recall (95% CI)")
    ax_c.set_ylim(0, 1.15)
    ax_c.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax_c.set_title("Per-class recall comparison", pad=8)

    # ── (d) Accuracy on subsets with bootstrap CI ──
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")

    # Load agreement metadata from labeled_events
    labels_path = PROJECT_ROOT / "data" / "labeled" / "labeled_events.csv"
    labels_df = pd.read_csv(labels_path)
    labels_df["event_id"] = labels_df["event_id"].astype(str)

    # Define subsets
    all_events = set(agent_events)
    sampling_events = set(y_true_series[y_true_series == "sampling"].index) & all_events
    epi_env_events = set(y_true_series[y_true_series.isin(["epidemic", "environmental"])].index) & all_events

    subsets = [
        ("All events", list(all_events)),
        ("Sampling", list(sampling_events)),
        ("Epi + Env", list(epi_env_events)),
    ]

    x_sub = np.arange(len(subsets))
    width_sub = 0.35
    for offset, m, col, label in [(-width_sub/2, "agent", AGENT_COLOR, "WBE-Agent"),
                                    (width_sub/2, "logistic_regression", LR_COLOR, "Logistic reg.")]:
        if m not in preds:
            continue
        accs, lo_errs, hi_errs = [], [], []
        for sname, sevents in subsets:
            mask = np.isin(agent_events, sevents)
            if mask.sum() == 0:
                accs.append(0); lo_errs.append(0); hi_errs.append(0)
                continue
            yt_sub = y_true[mask]
            yp_sub = preds[m].reindex(agent_events, fill_value="uncertain").values[mask]
            acc_fn = lambda yt, yp: (yt == yp).mean()
            pt, lo, hi = bootstrap_metric(yt_sub, yp_sub, acc_fn)
            accs.append(pt)
            lo_errs.append(pt - lo)
            hi_errs.append(hi - pt)

        ax_d.bar(x_sub + offset, accs, width_sub, yerr=[lo_errs, hi_errs],
                 color=col, alpha=0.85, edgecolor="white", linewidth=0.5,
                 capsize=3, error_kw={"linewidth": 0.8}, label=label, zorder=2)

    ax_d.set_xticks(x_sub)
    ax_d.set_xticklabels([s[0] for s in subsets], fontsize=7)
    ax_d.set_ylabel("Accuracy (95% CI)")
    ax_d.set_ylim(0, 1.15)
    ax_d.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax_d.set_title("Subset accuracy comparison", pad=8)

    fig.savefig(GENERATED / "fig_benchmark_v2.pdf")
    fig.savefig(GENERATED / "fig_benchmark_v2.png")
    plt.close(fig)
    print("  [ok] fig_benchmark_v2.pdf")


# ===================================================================
# FIGURE 8 — Enhanced reliability with calibration CI + significance
# ===================================================================
def build_fig8_enhanced():
    inv = pd.read_csv(INV_PATH)
    agent_pred = pd.read_csv(EVAL_DIR / "agent_predictions.csv")

    # Merge confidence with y_true
    merged = agent_pred.merge(inv[["event_id", "confidence", "tool_calls_count", "total_tokens"]],
                               on="event_id", how="left")
    merged["correct"] = (merged["y_pred"] == merged["y_true"]).astype(int)
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce")
    merged = merged.dropna(subset=["confidence"])

    fig = plt.figure(figsize=(7.2, 7.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.38,
                           left=0.10, right=0.96, top=0.95, bottom=0.07)

    # ── (a) Calibration curve with Wilson CI ──
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "a")

    bins = [0.0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
    bin_labels = []
    bin_accs = []
    bin_los = []
    bin_his = []
    bin_counts = []
    bin_centers = []

    for i in range(len(bins) - 1):
        mask = (merged["confidence"] >= bins[i]) & (merged["confidence"] < bins[i + 1])
        subset = merged[mask]
        n = len(subset)
        if n == 0:
            continue
        k = int(subset["correct"].sum())
        p_hat = k / n
        # Wilson score interval
        z = 1.96
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
        lo = max(0, center - margin)
        hi = min(1, center + margin)

        bin_center = (bins[i] + bins[i + 1]) / 2
        bin_centers.append(bin_center)
        bin_accs.append(p_hat)
        bin_los.append(p_hat - lo)
        bin_his.append(hi - p_hat)
        bin_counts.append(n)
        bin_labels.append(f"[{bins[i]:.2f},{bins[i+1]:.2f})")

    # Perfect calibration line
    ax_a.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.4, label="Perfect calibration")
    # Calibration with CI
    ax_a.errorbar(bin_centers, bin_accs, yerr=[bin_los, bin_his],
                  fmt="o-", color=AGENT_COLOR, markersize=6, linewidth=1.5,
                  capsize=4, capthick=1.0, elinewidth=1.0, label="WBE-Agent", zorder=3)
    # Annotate counts
    for x, y, n in zip(bin_centers, bin_accs, bin_counts):
        ax_a.annotate(f"n={n}", (x, y), textcoords="offset points",
                      xytext=(0, 10), fontsize=5.5, ha="center", color="#666666")

    ax_a.set_xlabel("Predicted confidence")
    ax_a.set_ylabel("Observed accuracy")
    ax_a.set_xlim(0.45, 1.02)
    ax_a.set_ylim(0.3, 1.05)
    ax_a.legend(loc="lower right", frameon=False, fontsize=6.5)
    ax_a.set_title("Confidence calibration (Wilson 95% CI)", pad=8)

    # ── (b) Confidence by class with significance ──
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "b")

    classes_present = [c for c in CLASS_ORDER if c in merged["y_pred"].values]
    conf_data = {}
    for cls in classes_present:
        vals = merged.loc[merged["y_pred"] == cls, "confidence"].dropna().values
        if len(vals) > 0:
            conf_data[cls] = vals

    positions = list(range(len(classes_present)))
    vp_data = [conf_data.get(c, np.array([])) for c in classes_present]

    # Violin
    parts = ax_b.violinplot([d for d in vp_data if len(d) > 0],
                             positions=[p for p, d in zip(positions, vp_data) if len(d) > 0],
                             showmeans=False, showmedians=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        cls = [c for c, d in zip(classes_present, vp_data) if len(d) > 0][i]
        body.set_facecolor(PALETTE[cls])
        body.set_alpha(0.3)
        body.set_edgecolor(PALETTE[cls])

    # Strip + median
    rng = np.random.default_rng(42)
    for i, cls in enumerate(classes_present):
        vals = conf_data.get(cls, np.array([]))
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax_b.scatter(i + jitter, vals, s=8, alpha=0.5, color=PALETTE[cls], edgecolors="none", zorder=3)
        med = np.median(vals)
        ax_b.hlines(med, i - 0.25, i + 0.25, colors=PALETTE[cls], linewidth=1.8, zorder=4)

    # Kruskal-Wallis test across classes
    groups = [conf_data[c] for c in classes_present if c in conf_data and len(conf_data[c]) > 1]
    if len(groups) >= 2:
        kw_stat, kw_p = sp_stats.kruskal(*groups)
        ax_b.text(0.98, 0.02, f"Kruskal-Wallis: H={kw_stat:.1f}, p={kw_p:.1e}",
                  transform=ax_b.transAxes, fontsize=5.5, ha="right", va="bottom",
                  color="#555555", style="italic")

    # Pairwise: uncertain vs sampling (the key comparison)
    if "uncertain" in conf_data and "sampling" in conf_data:
        u_stat, u_p = sp_stats.mannwhitneyu(conf_data["uncertain"], conf_data["sampling"], alternative="two-sided")
        idx_s = classes_present.index("sampling")
        idx_u = classes_present.index("uncertain")
        y_bracket = 1.02
        ax_b.plot([idx_s, idx_s, idx_u, idx_u], [y_bracket - 0.01, y_bracket, y_bracket, y_bracket - 0.01],
                  color="#333333", linewidth=0.7)
        n1, n2 = len(conf_data["uncertain"]), len(conf_data["sampling"])
        p_str = f"p={u_p:.1e}" if u_p >= 0.001 else f"p<0.001"
        ax_b.text((idx_s + idx_u) / 2, y_bracket + 0.005,
                  f"Mann-Whitney {p_str} {sig_stars(u_p)}",
                  ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    # Pairwise: uncertain vs epidemic
    if "uncertain" in conf_data and "epidemic" in conf_data:
        u_stat2, u_p2 = sp_stats.mannwhitneyu(conf_data["uncertain"], conf_data["epidemic"], alternative="two-sided")
        idx_e = classes_present.index("epidemic")
        idx_u = classes_present.index("uncertain")
        y_bracket2 = 1.09
        ax_b.plot([idx_e, idx_e, idx_u, idx_u], [y_bracket2 - 0.01, y_bracket2, y_bracket2, y_bracket2 - 0.01],
                  color="#333333", linewidth=0.7)
        p_str2 = f"p={u_p2:.1e}" if u_p2 >= 0.001 else f"p<0.001"
        ax_b.text((idx_e + idx_u) / 2, y_bracket2 + 0.005,
                  f"Mann-Whitney {p_str2} {sig_stars(u_p2)}",
                  ha="center", va="bottom", fontsize=5.5, fontweight="bold")

    ax_b.set_xticks(positions)
    ax_b.set_xticklabels([CLASS_LABELS[c] for c in classes_present], rotation=25, ha="right")
    ax_b.set_ylabel("Confidence")
    ax_b.set_ylim(0.3, 1.18)
    ax_b.set_title("Confidence by predicted class", pad=8)

    # ── (c) Correct vs incorrect confidence with significance ──
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "c")

    correct_conf = merged.loc[merged["correct"] == 1, "confidence"].values
    incorrect_conf = merged.loc[merged["correct"] == 0, "confidence"].values

    bp = ax_c.boxplot([correct_conf, incorrect_conf], positions=[0, 1], widths=0.5,
                       patch_artist=True, showfliers=False,
                       medianprops=dict(color="white", linewidth=1.5))
    bp["boxes"][0].set_facecolor("#5BA05B")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#C85050")
    bp["boxes"][1].set_alpha(0.7)

    # Strip
    for pos, vals, col in [(0, correct_conf, "#5BA05B"), (1, incorrect_conf, "#C85050")]:
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax_c.scatter(pos + jitter, vals, s=6, alpha=0.4, color=col, edgecolors="none", zorder=3)

    # Mann-Whitney U test
    if len(correct_conf) > 0 and len(incorrect_conf) > 0:
        u_stat, u_p = sp_stats.mannwhitneyu(correct_conf, incorrect_conf, alternative="two-sided")
        y_bracket = max(correct_conf.max(), incorrect_conf.max()) + 0.02
        ax_c.plot([0, 0, 1, 1], [y_bracket - 0.01, y_bracket, y_bracket, y_bracket - 0.01],
                  color="#333333", linewidth=0.7)
        p_str = f"p={u_p:.1e}" if u_p >= 0.001 else f"p<0.001"
        ax_c.text(0.5, y_bracket + 0.005,
                  f"Mann-Whitney {p_str} {sig_stars(u_p)}",
                  ha="center", va="bottom", fontsize=6, fontweight="bold")

    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels([f"Correct\n(n={len(correct_conf)})", f"Incorrect\n(n={len(incorrect_conf)})"], fontsize=7)
    ax_c.set_ylabel("Confidence")
    ax_c.set_title("Confidence: correct vs incorrect", pad=8)

    # ── (d) Resource use by class with significance ──
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")

    tok_data = {}
    for cls in classes_present:
        vals = merged.loc[merged["y_pred"] == cls, "total_tokens"].dropna().values
        if len(vals) > 0:
            tok_data[cls] = vals / 1000  # convert to k

    bp2 = ax_d.boxplot([tok_data.get(c, []) for c in classes_present],
                        positions=list(range(len(classes_present))),
                        widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", linewidth=1.5))
    for i, cls in enumerate(classes_present):
        if i < len(bp2["boxes"]):
            bp2["boxes"][i].set_facecolor(PALETTE[cls])
            bp2["boxes"][i].set_alpha(0.6)

    # Strip
    for i, cls in enumerate(classes_present):
        vals = tok_data.get(cls, np.array([]))
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax_d.scatter(i + jitter, vals, s=6, alpha=0.4, color=PALETTE[cls], edgecolors="none", zorder=3)

    # Kruskal-Wallis
    tok_groups = [tok_data[c] for c in classes_present if c in tok_data and len(tok_data[c]) > 1]
    if len(tok_groups) >= 2:
        kw_stat, kw_p = sp_stats.kruskal(*tok_groups)
        ax_d.text(0.98, 0.98, f"Kruskal-Wallis: H={kw_stat:.1f}, p={kw_p:.1e}",
                  transform=ax_d.transAxes, fontsize=5.5, ha="right", va="top",
                  color="#555555", style="italic")

    ax_d.set_xticks(list(range(len(classes_present))))
    ax_d.set_xticklabels([CLASS_LABELS[c] for c in classes_present], rotation=25, ha="right")
    ax_d.set_ylabel("Tokens (×1000)")
    ax_d.set_title("Resource use by class", pad=8)

    fig.savefig(GENERATED / "fig_reliability_v2.pdf")
    fig.savefig(GENERATED / "fig_reliability_v2.png")
    plt.close(fig)
    print("  [ok] fig_reliability_v2.pdf")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    setup_style()
    print("Building enhanced statistical figures...")
    build_fig5_enhanced()
    build_fig8_enhanced()
    print("Done.")
