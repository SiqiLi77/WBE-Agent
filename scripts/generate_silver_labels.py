"""CLI to generate LLM-adjudicated silver labels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
from loguru import logger

from src.config import PROJECT_ROOT
from src.labeling.llm_silver import run_silver_labeling


@click.command()
@click.option("--events-file", default=None, help="Optional event catalog CSV path.")
@click.option(
    "--output-file",
    default=str(PROJECT_ROOT / "data" / "labeled" / "labeled_events.csv"),
    show_default=True,
    help="Final consensus label CSV.",
)
@click.option(
    "--raw-output-dir",
    default=str(PROJECT_ROOT / "outputs" / "labeling"),
    show_default=True,
    help="Directory for raw per-judge outputs.",
)
@click.option(
    "--judge-model",
    "judge_models",
    multiple=True,
    help="Judge model name. Repeat this flag to use multiple judges.",
)
@click.option(
    "--confidence-threshold",
    default=0.7,
    show_default=True,
    type=float,
    help="Minimum mean confidence required for a non-uncertain silver label.",
)
@click.option(
    "--agreement-threshold",
    default=0.66,
    show_default=True,
    type=float,
    help="Required agreement ratio across judges for accepting a non-uncertain label.",
)
@click.option("--max-events", default=None, type=int, help="Optional cap for debugging or cost control.")
def main(
    events_file: str | None,
    output_file: str,
    raw_output_dir: str,
    judge_models: tuple[str, ...],
    confidence_threshold: float,
    agreement_threshold: float,
    max_events: int | None,
) -> None:
    models = list(judge_models) if judge_models else []
    if not models:
        from src.config import settings

        models = [settings.agent.model]

    logger.info(
        "Running silver-label generation: "
        f"models={models}, confidence_threshold={confidence_threshold}, "
        f"agreement_threshold={agreement_threshold}, max_events={max_events}"
    )

    final_df, raw_df = run_silver_labeling(
        events_file=events_file,
        output_file=output_file,
        raw_output_dir=raw_output_dir,
        judge_models=models,
        confidence_threshold=confidence_threshold,
        agreement_threshold=agreement_threshold,
        max_events=max_events,
    )

    if "ground_truth_label" in final_df.columns:
        logger.info(f"Final label distribution:\n{final_df['ground_truth_label'].value_counts().to_string()}")
    logger.info(f"Saved {len(final_df)} consensus labels and {len(raw_df)} raw judge rows.")


if __name__ == "__main__":
    main()
