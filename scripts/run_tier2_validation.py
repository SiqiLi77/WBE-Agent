"""Tier-2 独立验证：用高能力模型对银标签进行独立裁定。

不参与银标签投票，而是作为独立验证层，评估银标签质量。
默认使用 Claude Opus 4.6 (anthropic/claude-opus-4)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np
import pandas as pd
from loguru import logger

from src.config import PROJECT_ROOT, ensure_dirs
from src.labeling.llm_silver import (
    EvidencePacketBuilder,
    load_events_catalog,
    load_merged_database,
    load_site_metadata,
)
from src.labeling.prompts import format_evidence_prompt, get_judge_system_prompt
from src.evaluation.baselines import normalize_label

from openai import OpenAI
import os


TIER2_SYSTEM_PROMPT_ADDENDUM = """

ADDITIONAL CONTEXT FOR TIER-2 VALIDATION:
You are serving as an independent high-capability validator. Your judgement will be compared
against a panel of three smaller models to assess label quality and identify systematic biases.

Be especially careful about:
1. Events where clinical hospitalization data rises meaningfully (>20%) within 14 days — these
   may genuinely be epidemic signals even if the evidence packet also contains minor environmental noise.
2. Do not default to "mixed" simply because multiple data sources are present. "Mixed" requires
   genuinely competing explanations of comparable magnitude.
3. If the derived_assessment shows epidemic_candidate=true, seriously consider "epidemic" as the label.
"""


def run_tier2(
    events_file: str | None,
    output_dir: str,
    model: str,
    max_events: int | None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> None:
    ensure_dirs()
    database = load_merged_database()
    site_meta = load_site_metadata()
    events = load_events_catalog(events_file)
    if max_events:
        events = events.head(max_events).copy()

    # Resolve API credentials: CLI args > env vars > settings.yaml
    from src.config import settings
    resolved_key = api_key or os.environ.get("TIER2_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "") or settings.agent.api_key
    resolved_base = api_base or os.environ.get("TIER2_API_BASE") or settings.agent.api_base
    if not resolved_key:
        raise ValueError("No API key found.")

    logger.info(f"Using API base: {resolved_base}")
    logger.info(f"Using model: {model}")

    client = OpenAI(
        base_url=resolved_base,
        api_key=resolved_key,
        timeout=180.0,
        max_retries=2,
    )
    packet_builder = EvidencePacketBuilder(database=database, site_metadata=site_meta)
    system_prompt = get_judge_system_prompt() + TIER2_SYSTEM_PROMPT_ADDENDUM

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, (_, event) in enumerate(events.iterrows(), start=1):
        eid = str(event["event_id"])
        logger.info(f"[{idx}/{len(events)}] Tier-2 validation: {eid}")

        try:
            packet = packet_builder.build(event)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": format_evidence_prompt(packet)},
            ]

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )

            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0

            # Parse JSON
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                payload = json.loads(content[json_start:json_end])
            else:
                payload = {"label": "uncertain", "confidence": 0.0, "rationale": "No JSON in response"}

            results.append({
                "event_id": eid,
                "tier2_model": model,
                "tier2_label": normalize_label(payload.get("label", "uncertain")),
                "tier2_confidence": round(float(payload.get("confidence", 0.0)), 4),
                "tier2_rationale": payload.get("rationale", ""),
                "tier2_key_evidence": json.dumps(payload.get("key_evidence", []), ensure_ascii=True),
                "tier2_data_gaps": json.dumps(payload.get("data_gaps", []), ensure_ascii=True),
                "total_tokens": tokens,
                "error": "",
            })

        except Exception as exc:
            logger.error(f"Tier-2 failed for {eid}: {exc}")
            results.append({
                "event_id": eid,
                "tier2_model": model,
                "tier2_label": "uncertain",
                "tier2_confidence": 0.0,
                "tier2_rationale": f"Error: {exc}",
                "tier2_key_evidence": "[]",
                "tier2_data_gaps": "[]",
                "total_tokens": 0,
                "error": str(exc),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path / "tier2_validation.csv", index=False)

    # Compute agreement with silver labels
    silver_path = PROJECT_ROOT / "data/labeled/labeled_events.csv"
    if silver_path.exists():
        silver = pd.read_csv(silver_path)
        merged = results_df.merge(silver[["event_id", "ground_truth_label"]], on="event_id", how="left")
        agree = (merged["tier2_label"] == merged["ground_truth_label"]).sum()
        total = len(merged)
        logger.info(f"Tier-2 vs Silver agreement: {agree}/{total} ({agree/total*100:.1f}%)")

        # Cohen's kappa
        from sklearn.metrics import cohen_kappa_score
        valid = merged.dropna(subset=["ground_truth_label"])
        if len(valid) > 0:
            kappa = cohen_kappa_score(valid["ground_truth_label"], valid["tier2_label"])
            logger.info(f"Cohen's kappa (Tier-2 vs Silver): {kappa:.3f}")

        # Disagreement analysis
        disagree = merged[merged["tier2_label"] != merged["ground_truth_label"]]
        if len(disagree) > 0:
            logger.info(f"\nDisagreements ({len(disagree)} events):")
            for _, row in disagree.iterrows():
                logger.info(f"  {row['event_id']}: silver={row['ground_truth_label']} tier2={row['tier2_label']}")

        merged.to_csv(out_path / "tier2_vs_silver.csv", index=False)

    # Summary
    dist = results_df["tier2_label"].value_counts()
    summary = {
        "model": model,
        "n_events": len(results_df),
        "label_distribution": dist.to_dict(),
        "avg_confidence": round(float(results_df["tier2_confidence"].mean()), 4),
        "total_tokens": int(results_df["total_tokens"].sum()),
        "errors": int((results_df["error"] != "").sum()),
    }
    (out_path / "tier2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    logger.info(f"\nTier-2 label distribution:\n{dist.to_string()}")
    logger.info(f"Results saved to {out_path}")


@click.command()
@click.option("--events-file", default=None, help="Event catalog CSV path.")
@click.option(
    "--output-dir",
    default=str(PROJECT_ROOT / "outputs" / "tier2_validation"),
    show_default=True,
)
@click.option(
    "--model",
    default="Claude-Opus-4.6",
    show_default=True,
    help="High-capability model for tier-2 validation.",
)
@click.option("--max-events", default=None, type=int, help="Cap for cost control.")
@click.option(
    "--api-base",
    default="https://aiping.cn/api/v1",
    show_default=True,
    help="API base URL for tier-2 model.",
)
@click.option(
    "--api-key",
    default=None,
    help="API key for tier-2 model (overrides env/config).",
)
def main(
    events_file: str | None,
    output_dir: str,
    model: str,
    max_events: int | None,
    api_base: str,
    api_key: str | None,
) -> None:
    logger.info(f"Tier-2 validation with model={model}, max_events={max_events}")
    run_tier2(events_file, output_dir, model, max_events, api_base=api_base, api_key=api_key)


if __name__ == "__main__":
    main()
