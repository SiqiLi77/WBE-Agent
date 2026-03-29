"""Run agent ablations and evaluate them against the current label set."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
from loguru import logger

from src.agent.runner import InvestigationAgent
from src.config import PROJECT_ROOT, ensure_dirs, settings
from src.evaluation.baselines import normalize_label
from src.evaluation.metrics import compute_all_metrics


ABLATIONS = {
    "full": {"disabled_tools": set(), "include_domain_knowledge": True},
    "no_weather": {"disabled_tools": {"query_weather"}, "include_domain_knowledge": True},
    "no_hydrology": {"disabled_tools": {"query_hydrology"}, "include_domain_knowledge": True},
    "no_clinical": {"disabled_tools": {"query_hospitalization"}, "include_domain_knowledge": True},
    "no_nearby_sites": {"disabled_tools": {"query_nearby_sites"}, "include_domain_knowledge": True},
    "no_variants": {"disabled_tools": {"query_variants"}, "include_domain_knowledge": True},
    "no_domain_knowledge": {"disabled_tools": set(), "include_domain_knowledge": False},
    "no_tools": {
        "disabled_tools": {
            "query_weather",
            "query_hydrology",
            "query_hospitalization",
            "query_nearby_sites",
            "query_variants",
            "query_site_metadata",
        },
        "include_domain_knowledge": True,
    },
}


def resolve_labels_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    candidates = [
        PROJECT_ROOT / settings.paths.labeled_data_dir / "labeled_events.csv",
        PROJECT_ROOT / settings.paths.labeled_data_dir / "auto_labeled_events.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_eval_events(labels_path: Path, events_df: pd.DataFrame) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path)
    label_col = "ground_truth_label" if "ground_truth_label" in labels_df.columns else "auto_label"
    labels_df["event_id"] = labels_df["event_id"].astype(str)
    labels_df["y_true"] = labels_df[label_col].map(normalize_label)
    merged = events_df.copy()
    merged["event_id"] = merged["event_id"].astype(str)
    merged = merged.merge(labels_df[["event_id", "y_true"]], on="event_id", how="left")
    merged["y_true"] = merged["y_true"].fillna("uncertain")
    return merged


def load_database() -> tuple[pd.DataFrame, pd.DataFrame]:
    db_path = PROJECT_ROOT / settings.paths.merged_database
    database = pd.read_parquet(db_path)
    database["date"] = pd.to_datetime(database["date"])

    tier_frames = []
    for tier_key in ["tier1_sites", "tier2_sites", "tier3_sites"]:
        tier_path = PROJECT_ROOT / getattr(settings.paths, tier_key)
        if tier_path.exists():
            tier_frames.append(pd.read_csv(tier_path))
    site_meta = pd.concat(tier_frames, ignore_index=True) if tier_frames else pd.DataFrame()

    if "key_plot_id" in site_meta.columns and "site_id" not in site_meta.columns:
        site_meta["site_id"] = site_meta["key_plot_id"]
    if "key_plot_id" in database.columns and "site_id" not in database.columns:
        database["site_id"] = database["key_plot_id"]
    return database, site_meta


def load_events(events_file: str | None, event_ids: set[str] | None, max_events: int | None) -> pd.DataFrame:
    events_path = Path(events_file) if events_file else PROJECT_ROOT / settings.paths.outputs_dir / "anomaly_event_catalog.csv"
    events_df = pd.read_csv(events_path)
    events_df["event_id"] = events_df["event_id"].astype(str)
    if event_ids:
        events_df = events_df[events_df["event_id"].isin(event_ids)].copy()
    if max_events:
        events_df = events_df.head(max_events).copy()
    return events_df.reset_index(drop=True)


def score_predictions(events_df: pd.DataFrame, predictions_path: Path) -> dict[str, float]:
    pred_df = pd.read_csv(predictions_path)
    pred_df["event_id"] = pred_df["event_id"].astype(str)
    merged = events_df[["event_id", "y_true"]].merge(
        pred_df[["event_id", "classification"]], on="event_id", how="left"
    )
    y_true = merged["y_true"].tolist()
    y_pred = merged["classification"].fillna("uncertain").map(normalize_label).tolist()
    metrics = compute_all_metrics(y_true, y_pred)
    avg_tokens = float(pd.to_numeric(pred_df.get("total_tokens"), errors="coerce").fillna(0).mean())
    avg_tool_calls = float(pd.to_numeric(pred_df.get("tool_calls_count"), errors="coerce").fillna(0).mean())
    return {
        "n_samples": metrics.get("n_samples", 0),
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_f1": metrics.get("macro_f1", 0.0),
        "weighted_f1": metrics.get("weighted_f1", 0.0),
        "cohen_kappa": metrics.get("cohen_kappa", 0.0),
        "avg_tokens": avg_tokens,
        "avg_tool_calls": avg_tool_calls,
    }


@click.command()
@click.option("--ablations", default="all", help="Comma-separated ablation names or 'all'.")
@click.option("--labels-file", default=None, help="Optional labels CSV.")
@click.option("--events-file", default=None, help="Optional event catalog CSV.")
@click.option("--event-id", "event_ids", multiple=True, help="Optional event_id filter.")
@click.option("--max-events", default=None, type=int, help="Optional event cap for debugging.")
@click.option("--model", default=None, help="Optional LLM model override.")
@click.option("--output-dir", default=None, help="Directory for ablation outputs.")
@click.option("--sleep-seconds", default=1.0, type=float, show_default=True, help="Pause between events.")
@click.option("--reuse-existing/--no-reuse-existing", default=True, help="Reuse saved prediction files when present.")
def main(
    ablations: str,
    labels_file: str | None,
    events_file: str | None,
    event_ids: tuple[str, ...],
    max_events: int | None,
    model: str | None,
    output_dir: str | None,
    sleep_seconds: float,
    reuse_existing: bool,
) -> None:
    ensure_dirs()
    output_root = Path(output_dir) if output_dir else (PROJECT_ROOT / settings.paths.outputs_dir / "ablation")
    output_root.mkdir(parents=True, exist_ok=True)

    labels_path = resolve_labels_path(labels_file)
    database, site_meta = load_database()
    events_df = load_events(events_file, set(event_ids), max_events)
    eval_events = load_eval_events(labels_path, events_df)

    selected = list(ABLATIONS.keys()) if ablations.strip().lower() == "all" else [a.strip() for a in ablations.split(",") if a.strip()]
    invalid = [name for name in selected if name not in ABLATIONS]
    if invalid:
        raise ValueError(f"Unsupported ablations: {invalid}. Supported: {sorted(ABLATIONS)}")

    rows: list[dict[str, float | str]] = []
    for name in selected:
        config = ABLATIONS[name]
        prediction_path = output_root / f"{name}_investigation_results.csv"
        if reuse_existing and prediction_path.exists():
            logger.info(f"Reusing ablation predictions: {prediction_path}")
        else:
            logger.info(
                f"Running ablation {name}: disabled_tools={sorted(config['disabled_tools'])}, "
                f"include_domain_knowledge={config['include_domain_knowledge']}"
            )
            agent = InvestigationAgent(
                database=database,
                site_metadata=site_meta,
                model=model,
                disabled_tools=config["disabled_tools"],
                include_domain_knowledge=config["include_domain_knowledge"],
                trace_subdir=f"ablation/{name}_traces",
            )
            agent.investigate_batch(
                events_df,
                output_path=prediction_path,
                sleep_seconds=sleep_seconds,
            )

        metrics = score_predictions(eval_events, prediction_path)
        row = {"ablation": name}
        row.update(metrics)
        rows.append(row)
        logger.info(
            f"{name}: accuracy={metrics['accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}, "
            f"kappa={metrics['cohen_kappa']:.3f}"
        )

    new_summary_df = pd.DataFrame(rows)
    summary_path = output_root / "ablation_summary.csv"
    if summary_path.exists():
        existing_summary_df = pd.read_csv(summary_path)
        summary_df = pd.concat([existing_summary_df, new_summary_df], ignore_index=True)
        summary_df = summary_df.drop_duplicates(subset=["ablation"], keep="last")
    else:
        summary_df = new_summary_df
    summary_df = summary_df.sort_values("macro_f1", ascending=False)
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved ablation summary: {summary_path}")
    logger.info(f"\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
