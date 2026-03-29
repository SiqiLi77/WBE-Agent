"""
评估指标计算。

提供 Agent 性能评估所需的所有指标。
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from loguru import logger


CLASSIFICATION_LABELS = ["epidemic", "environmental", "sampling", "mixed", "uncertain"]


def compute_all_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict:
    """
    计算完整的评估指标集。

    Parameters
    ----------
    y_true : ground truth 标签列表
    y_pred : 预测标签列表
    labels : 标签列表，默认使用 CLASSIFICATION_LABELS

    Returns
    -------
    dict : 包含所有指标的字典
    """
    if labels is None:
        labels = CLASSIFICATION_LABELS

    # 过滤掉 uncertain（如果需要）
    valid_mask = [(t in labels and p in labels) for t, p in zip(y_true, y_pred)]
    y_true_f = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_f = [p for p, v in zip(y_pred, valid_mask) if v]

    if not y_true_f:
        return {"error": "无有效样本"}

    metrics = {
        "n_samples": len(y_true_f),
        "accuracy": accuracy_score(y_true_f, y_pred_f),
        "macro_f1": f1_score(y_true_f, y_pred_f, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_f, y_pred_f, labels=labels, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true_f, y_pred_f),
    }

    # 分类别指标
    for label in labels:
        y_true_bin = [1 if t == label else 0 for t in y_true_f]
        y_pred_bin = [1 if p == label else 0 for p in y_pred_f]
        if sum(y_true_bin) > 0 or sum(y_pred_bin) > 0:
            metrics[f"{label}_precision"] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            metrics[f"{label}_recall"] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            metrics[f"{label}_f1"] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            metrics[f"{label}_support"] = sum(y_true_bin)

    # 混淆矩阵
    present_labels = sorted(set(y_true_f + y_pred_f))
    cm = confusion_matrix(y_true_f, y_pred_f, labels=present_labels)
    metrics["confusion_matrix"] = {
        "labels": present_labels,
        "matrix": cm.tolist(),
    }

    return metrics


def compute_inter_rater_reliability(
    rater1: list[str],
    rater2: list[str],
) -> dict:
    """
    计算两个标注者之间的一致性。

    Returns
    -------
    dict : {agreement_rate, cohen_kappa, n_samples, n_disagreements}
    """
    assert len(rater1) == len(rater2), "两个标注者的样本数必须相同"

    n = len(rater1)
    n_agree = sum(1 for a, b in zip(rater1, rater2) if a == b)

    return {
        "n_samples": n,
        "agreement_rate": n_agree / n if n > 0 else 0,
        "cohen_kappa": cohen_kappa_score(rater1, rater2),
        "n_disagreements": n - n_agree,
        "disagreement_pairs": [
            {"index": i, "rater1": a, "rater2": b}
            for i, (a, b) in enumerate(zip(rater1, rater2))
            if a != b
        ],
    }


def print_evaluation_report(metrics: dict) -> str:
    """格式化打印评估报告。"""
    lines = [
        "=" * 50,
        "评估报告",
        "=" * 50,
        f"样本数: {metrics.get('n_samples', 'N/A')}",
        f"Overall Accuracy: {metrics.get('accuracy', 0):.3f}",
        f"Macro F1: {metrics.get('macro_f1', 0):.3f}",
        f"Weighted F1: {metrics.get('weighted_f1', 0):.3f}",
        f"Cohen's Kappa: {metrics.get('cohen_kappa', 0):.3f}",
        "",
        "分类别指标:",
    ]

    for label in CLASSIFICATION_LABELS:
        p = metrics.get(f"{label}_precision", 0)
        r = metrics.get(f"{label}_recall", 0)
        f1 = metrics.get(f"{label}_f1", 0)
        s = metrics.get(f"{label}_support", 0)
        if s > 0:
            lines.append(f"  {label:15s}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  support={s}")

    report = "\n".join(lines)
    logger.info(report)
    return report
