"""
Evaluation CLI for agent-vs-baseline comparison.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
from loguru import logger

from src.config import PROJECT_ROOT, ensure_dirs, settings
from src.evaluation.baselines import (
    FewShotLLMBaseline,
    MLBaseline,
    MajorityClassBaseline,
    RuleBasedBaseline,
    ZeroShotLLMBaseline,
    normalize_label,
)
from src.evaluation.metrics import compute_all_metrics


ALL_METHODS = [
    "agent",
    "majority_class",
    "rule_based",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "zero_shot_llm",
    "few_shot_llm",
]

BASELINE_METHODS = [m for m in ALL_METHODS if m != "agent"]


def resolve_labels_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    candidates = [
        PROJECT_ROOT / settings.paths.labeled_data_dir / "labeled_events.csv",
        PROJECT_ROOT / settings.paths.labeled_data_dir / "auto_labeled_events.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def detect_label_column(df: pd.DataFrame) -> str:
    for col in ["ground_truth_label", "label", "auto_label"]:
        if col in df.columns:
            return col
    raise ValueError(
        "No label column found. Expected one of: ground_truth_label, label, auto_label."
    )


def load_eval_events(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")
    df = pd.read_csv(labels_path)
    if "event_id" not in df.columns:
        raise ValueError(f"Label file missing 'event_id': {labels_path}")

    label_col = detect_label_column(df)
    df = df.copy()
    df["event_id"] = df["event_id"].astype(str)
    df["y_true"] = df[label_col].map(normalize_label)

    required_event_cols = [
        "event_id",
        "site_id",
        "peak_date",
        "peak_zscore",
        "duration_days",
        "vote_count_max",
        "y_true",
    ]
    missing_meta_cols = [c for c in required_event_cols if c not in {"y_true"} and c not in df.columns]
    if missing_meta_cols:
        catalog_path = PROJECT_ROOT / settings.paths.outputs_dir / "anomaly_event_catalog.csv"
        if catalog_path.exists():
            catalog_df = pd.read_csv(catalog_path)
            if "event_id" in catalog_df.columns:
                catalog_df["event_id"] = catalog_df["event_id"].astype(str)
                df = df.merge(catalog_df, on="event_id", how="left", suffixes=("", "_catalog"))
                for col in missing_meta_cols:
                    alt_col = f"{col}_catalog"
                    if col not in df.columns and alt_col in df.columns:
                        df[col] = df[alt_col]

    for col in required_event_cols:
        if col not in df.columns:
            if col == "vote_count_max":
                df[col] = 0
            elif col in {"peak_zscore", "duration_days"}:
                df[col] = 0
            else:
                raise ValueError(f"Label file missing required column: {col}")

    eval_df = df[required_event_cols].copy()
    eval_df["peak_date"] = pd.to_datetime(eval_df["peak_date"]).dt.date.astype(str)
    eval_df["peak_zscore"] = pd.to_numeric(eval_df["peak_zscore"], errors="coerce").fillna(0.0)
    eval_df["duration_days"] = pd.to_numeric(eval_df["duration_days"], errors="coerce").fillna(1).astype(int)
    eval_df["vote_count_max"] = pd.to_numeric(eval_df["vote_count_max"], errors="coerce").fillna(0).astype(int)
    return eval_df


def load_merged_database() -> pd.DataFrame:
    path = PROJECT_ROOT / settings.paths.merged_database
    if not path.exists():
        logger.warning(f"Merged database not found: {path}")
        return pd.DataFrame()
    db = pd.read_parquet(path)
    if "site_id" not in db.columns and "key_plot_id" in db.columns:
        db["site_id"] = db["key_plot_id"]
    if "date" in db.columns:
        db["date"] = pd.to_datetime(db["date"])
    return db


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict:
    y_pred_norm = [normalize_label(x) for x in y_pred]
    return compute_all_metrics(y_true, y_pred_norm)


def save_prediction_table(
    method: str,
    eval_df: pd.DataFrame,
    y_pred: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = eval_df[["event_id", "site_id", "peak_date", "y_true"]].copy()
    out["y_pred"] = [normalize_label(x) for x in y_pred]
    out.to_csv(output_dir / f"{method}_predictions.csv", index=False)


def load_cached_predictions(method: str, eval_df: pd.DataFrame, output_dir: Path) -> list[str] | None:
    path = output_dir / f"{method}_predictions.csv"
    if not path.exists():
        return None

    cached_df = pd.read_csv(path)
    if "event_id" not in cached_df.columns or "y_pred" not in cached_df.columns:
        logger.warning(f"Cached predictions missing required columns: {path}")
        return None

    cached_df["event_id"] = cached_df["event_id"].astype(str)
    merged = eval_df[["event_id"]].merge(cached_df[["event_id", "y_pred"]], on="event_id", how="left")
    missing = int(merged["y_pred"].isna().sum())
    if missing:
        logger.warning(f"Cached predictions incomplete for {method}: missing {missing} event(s); ignoring cache.")
        return None

    logger.info(f"Reusing cached predictions for {method}: {path}")
    return merged["y_pred"].map(normalize_label).tolist()


def run_agent_method(eval_df: pd.DataFrame, predictions_path: Path) -> tuple[list[str], dict]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Agent prediction file not found: {predictions_path}")

    pred_df = pd.read_csv(predictions_path)
    if "event_id" not in pred_df.columns or "classification" not in pred_df.columns:
        raise ValueError("Agent prediction file must include event_id and classification columns.")
    pred_df["event_id"] = pred_df["event_id"].astype(str)

    merge_cols = ["event_id", "classification"]
    for optional_col in ["total_tokens", "tool_calls_count"]:
        if optional_col in pred_df.columns:
            merge_cols.append(optional_col)

    merged = eval_df.merge(pred_df[merge_cols], on="event_id", how="left")
    missing_predictions = int(merged["classification"].isna().sum())
    y_pred = merged["classification"].fillna("uncertain").tolist()
    extras = {
        "prediction_rows": int(len(pred_df)),
        "missing_predictions": missing_predictions,
        "coverage": float((len(eval_df) - missing_predictions) / len(eval_df)) if len(eval_df) else 0.0,
    }
    if "total_tokens" in merged.columns:
        extras["avg_tokens"] = float(pd.to_numeric(merged["total_tokens"], errors="coerce").fillna(0).mean())
    if "tool_calls_count" in merged.columns:
        extras["avg_tool_calls"] = float(
            pd.to_numeric(merged["tool_calls_count"], errors="coerce").fillna(0).mean()
        )
    return y_pred, extras


def infer_agent_model_name(predictions_path: Path) -> str:
    stem = predictions_path.stem
    prefix = "investigation_results"
    if stem == prefix:
        return "default"
    if stem.startswith(f"{prefix}_"):
        return stem[len(prefix) + 1 :]
    return stem


def resolve_prediction_batch(batch_glob: str) -> list[Path]:
    pattern_path = Path(batch_glob)
    if pattern_path.is_absolute():
        pattern = str(pattern_path)
    else:
        pattern = str(PROJECT_ROOT / batch_glob)
    return sorted(Path(p) for p in glob.glob(pattern))


def build_summary_row(method: str, metrics: dict, extras: dict[str, float] | None = None) -> dict:
    row = {
        "method": method,
        "n_samples": metrics.get("n_samples", 0),
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "weighted_f1": metrics.get("weighted_f1", 0.0),
        "cohen_kappa": metrics.get("cohen_kappa", 0.0),
    }
    for label in ["epidemic", "environmental", "sampling", "mixed", "uncertain"]:
        row[f"{label}_precision"] = metrics.get(f"{label}_precision", 0.0)
        row[f"{label}_recall"] = metrics.get(f"{label}_recall", 0.0)
        row[f"{label}_f1"] = metrics.get(f"{label}_f1", 0.0)
        row[f"{label}_support"] = metrics.get(f"{label}_support", 0)
    if extras:
        row.update(extras)
    return row


def run_agent_batch_evaluation(
    eval_df: pd.DataFrame,
    batch_glob: str,
    summary_file: Path,
) -> pd.DataFrame:
    y_true = eval_df["y_true"].tolist()
    prediction_files = resolve_prediction_batch(batch_glob)
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files matched: {batch_glob}")

    rows: list[dict] = []
    logger.info(f"Batch-evaluating {len(prediction_files)} agent prediction files")
    for pred_path in prediction_files:
        model_name = infer_agent_model_name(pred_path)
        y_pred, extras = run_agent_method(eval_df, pred_path)
        extras["source_file"] = pred_path.name
        metrics = evaluate_predictions(y_true, y_pred)
        row = build_summary_row(model_name, metrics, extras)
        rows.append(row)
        logger.info(
            f"{model_name}: accuracy={row['accuracy']:.3f}, "
            f"macro_f1={row['macro_f1']:.3f}, kappa={row['cohen_kappa']:.3f}, "
            f"coverage={row.get('coverage', 0.0):.3f}"
        )

    summary_df = pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy", "cohen_kappa"],
        ascending=[False, False, False],
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved batch summary: {summary_file}")
    logger.info(f"\n{summary_df.to_string(index=False)}")
    return summary_df


def parse_methods(methods_raw: str, baselines_only: bool) -> list[str]:
    if methods_raw.strip().lower() == "all":
        return BASELINE_METHODS if baselines_only else ALL_METHODS
    methods = [m.strip() for m in methods_raw.split(",") if m.strip()]
    if not methods:
        raise ValueError("No methods specified.")
    invalid = [m for m in methods if m not in ALL_METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Supported: {ALL_METHODS}")
    if baselines_only:
        methods = [m for m in methods if m != "agent"]
    return methods


@click.command()
@click.option("--baselines-only", is_flag=True, help="Run only baseline methods.")
@click.option("--ablation", is_flag=True, help="Run standard evaluation flow on ablation prediction files.")
@click.option("--methods", default="all", help="Comma-separated methods or 'all'.")
@click.option("--labels-file", default=None, help="Path to labels CSV.")
@click.option("--predictions-file", default=None, help="Path to agent predictions CSV.")
@click.option("--output-dir", default=None, help="Directory for evaluation outputs.")
@click.option(
    "--agent-batch-glob",
    default=None,
    help="Glob for batch agent prediction files, e.g. 'outputs/investigation_results*.csv'.",
)
@click.option(
    "--summary-file",
    default=None,
    help="Optional CSV path for the summary file. In batch mode this defaults to outputs/evaluation/model_comparison_summary.csv.",
)
@click.option("--save-predictions/--no-save-predictions", default=True, help="Save per-method prediction tables.")
@click.option(
    "--reuse-cached-predictions/--no-reuse-cached-predictions",
    default=True,
    help="Reuse existing per-method prediction CSVs when available.",
)
def main(
    baselines_only: bool,
    ablation: bool,
    methods: str,
    labels_file: str | None,
    predictions_file: str | None,
    output_dir: str | None,
    agent_batch_glob: str | None,
    summary_file: str | None,
    save_predictions: bool,
    reuse_cached_predictions: bool,
):
    ensure_dirs()

    if ablation:
        logger.info("Running evaluation in ablation mode.")

    labels_path = resolve_labels_path(labels_file)
    eval_df = load_eval_events(labels_path)
    y_true = eval_df["y_true"].tolist()
    logger.info(f"Loaded labels: {labels_path} ({len(eval_df)} events)")

    if agent_batch_glob:
        if summary_file:
            batch_summary_path = Path(summary_file)
        else:
            batch_summary_path = PROJECT_ROOT / settings.paths.outputs_dir / "evaluation" / "model_comparison_summary.csv"
        run_agent_batch_evaluation(eval_df, agent_batch_glob, batch_summary_path)
        return

    merged_db = load_merged_database()
    methods_to_run = parse_methods(methods, baselines_only)
    logger.info(f"Methods: {methods_to_run}")

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = PROJECT_ROOT / settings.paths.outputs_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for method in methods_to_run:
        logger.info(f"=== Running method: {method} ===")
        extras: dict[str, float] = {}

        if method == "agent":
            pred_path = (
                Path(predictions_file)
                if predictions_file
                else PROJECT_ROOT / settings.paths.outputs_dir / "investigation_results.csv"
            )
            y_pred, extras = run_agent_method(eval_df, pred_path)
        elif method == "majority_class":
            majority_label = pd.Series(y_true).value_counts().idxmax()
            baseline = MajorityClassBaseline(majority_label=majority_label)
            y_pred = baseline.predict(eval_df, merged_db)
        elif method == "rule_based":
            baseline = RuleBasedBaseline()
            y_pred = baseline.predict(eval_df, merged_db)
        elif method in {"logistic_regression", "random_forest", "gradient_boosting"}:
            baseline = MLBaseline(classifier_type=method)
            y_pred = baseline.predict_oof(eval_df, y_true, merged_db)
        elif method == "zero_shot_llm":
            y_pred = load_cached_predictions(method, eval_df, out_dir) if reuse_cached_predictions else None
            if y_pred is None:
                baseline = ZeroShotLLMBaseline()
                y_pred = baseline.predict(eval_df, merged_db)
        elif method == "few_shot_llm":
            y_pred = load_cached_predictions(method, eval_df, out_dir) if reuse_cached_predictions else None
            if y_pred is None:
                baseline = FewShotLLMBaseline()
                y_pred = baseline.predict_oof(eval_df, y_true, merged_db)
        else:
            raise ValueError(f"Unsupported method: {method}")

        metrics = evaluate_predictions(y_true, y_pred)
        row = build_summary_row(method, metrics, extras)
        rows.append(row)

        logger.info(
            f"{method}: "
            f"accuracy={row['accuracy']:.3f}, "
            f"macro_f1={row['macro_f1']:.3f}, "
            f"kappa={row['cohen_kappa']:.3f}"
        )

        if save_predictions:
            save_prediction_table(method, eval_df, y_pred, out_dir)

    summary_path = out_dir / "evaluation_summary.csv"
    new_summary_df = pd.DataFrame(rows)
    if summary_path.exists():
        existing_summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([existing_summary_df, new_summary_df], ignore_index=True)
        summary_df = summary_df.drop_duplicates(subset=["method"], keep="last")
    else:
        summary_df = new_summary_df
    summary_df = summary_df.sort_values("macro_f1", ascending=False)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary: {summary_path}")
    logger.info(f"\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
