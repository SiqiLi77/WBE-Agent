"""LLM-adjudicated silver-label pipeline."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

from src.agent.tools import (
    HospitalizationTool,
    HydrologyTool,
    NearbySitesTool,
    SiteMetadataTool,
    VariantsTool,
    WeatherTool,
)
from src.config import PROJECT_ROOT, ensure_dirs, settings
from src.evaluation.baselines import normalize_label
from src.labeling.prompts import format_evidence_prompt, get_judge_system_prompt


class SilverLabelJudgement(BaseModel):
    """Structured judge output."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    key_evidence: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


@dataclass
class JudgeRunResult:
    """One model's adjudication for one event."""

    event_id: str
    judge_model: str
    label: str
    confidence: float
    rationale: str
    key_evidence: list[str]
    data_gaps: list[str]
    raw_response: str
    total_tokens: int = 0
    error: str = ""


def load_site_metadata() -> pd.DataFrame:
    """Load site metadata from all configured tiers."""
    frames: list[pd.DataFrame] = []
    for tier_key in ["tier1_sites", "tier2_sites", "tier3_sites"]:
        tier_path = PROJECT_ROOT / getattr(settings.paths, tier_key)
        if tier_path.exists():
            frames.append(pd.read_csv(tier_path))

    if not frames:
        return pd.DataFrame()

    site_meta = pd.concat(frames, ignore_index=True)
    if "key_plot_id" in site_meta.columns and "site_id" not in site_meta.columns:
        site_meta["site_id"] = site_meta["key_plot_id"]
    return site_meta.drop_duplicates(subset=["site_id"]).reset_index(drop=True)


def load_events_catalog(events_file: str | None = None) -> pd.DataFrame:
    """Load the anomaly event catalog."""
    path = Path(events_file) if events_file else PROJECT_ROOT / settings.paths.outputs_dir / "anomaly_event_catalog.csv"
    if not path.exists():
        raise FileNotFoundError(f"Event catalog not found: {path}")
    events = pd.read_csv(path)
    events["event_id"] = events["event_id"].astype(str)
    return events


def load_merged_database() -> pd.DataFrame:
    """Load the merged site-day database."""
    path = PROJECT_ROOT / settings.paths.merged_database
    if not path.exists():
        raise FileNotFoundError(f"Merged database not found: {path}")
    db = pd.read_parquet(path)
    if "site_id" not in db.columns and "key_plot_id" in db.columns:
        db["site_id"] = db["key_plot_id"]
    db["date"] = pd.to_datetime(db["date"])
    return db


class EvidencePacketBuilder:
    """Build compact evidence packets from raw aligned data."""

    def __init__(self, database: pd.DataFrame, site_metadata: pd.DataFrame):
        self.db = database
        self.site_meta = site_metadata
        self.site_tool = SiteMetadataTool(database=database, site_metadata=site_metadata)
        self.weather_tool = WeatherTool(database=database)
        self.hydrology_tool = HydrologyTool(database=database)
        self.hosp_tool = HospitalizationTool(database=database)
        self.nearby_tool = NearbySitesTool(database=database, site_metadata=site_metadata)
        self.variants_tool = VariantsTool(database=database)

    def _resolve_state(self, event: pd.Series) -> str:
        if "state" in event and pd.notna(event["state"]):
            return str(event["state"]).upper()

        site_id = str(event["site_id"])
        site_rows = self.db[self.db["site_id"] == site_id]
        if "state" in site_rows.columns:
            state_values = site_rows["state"].dropna().astype(str).str.upper()
            if not state_values.empty:
                return state_values.iloc[0]

        parts = site_id.split("_")
        if len(parts) >= 3:
            return parts[1].upper()
        return ""

    def _round_float(self, value: Any) -> Any:
        if isinstance(value, (float, np.floating)):
            if math.isnan(float(value)) or math.isinf(float(value)):
                return None
            return round(float(value), 4)
        if isinstance(value, (np.integer,)):
            return int(value)
        return value

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize(v) for v in value]
        return self._round_float(value)

    def _signal_context(self, event: pd.Series) -> dict[str, Any]:
        site_id = str(event["site_id"])
        peak_date = pd.Timestamp(event["peak_date"])
        site_data = self.db[self.db["site_id"] == site_id].sort_values("date")

        conc_col = "pcr_conc_lin_log1p" if "pcr_conc_lin_log1p" in site_data.columns else "pcr_conc_lin"
        if conc_col not in site_data.columns:
            return {"error": "No concentration column available"}

        window = site_data[
            (site_data["date"] >= peak_date - pd.Timedelta(days=7))
            & (site_data["date"] <= peak_date + pd.Timedelta(days=7))
        ][["date", conc_col]].dropna()

        pre = window[window["date"] < peak_date][conc_col]
        post = window[window["date"] > peak_date][conc_col]
        peak = window[window["date"] == peak_date][conc_col]
        context = pd.concat([pre.tail(3), post.head(3)], axis=0)

        isolated_spike_candidate = False
        local_change_pct = None
        if not peak.empty and not context.empty:
            context_mean = float(context.mean())
            context_std = float(context.std()) if len(context) > 1 else 0.0
            isolated_spike_candidate = bool(
                event.get("duration_days", 1) == 1
                and context_std > 0
                and abs(float(peak.iloc[0]) - context_mean) > 3 * context_std
            )
            if context_mean != 0:
                local_change_pct = (float(peak.iloc[0]) - context_mean) / abs(context_mean) * 100

        return self._sanitize(
            {
                "concentration_column": conc_col,
                "window_dates": window["date"].astype(str).tolist(),
                "window_values": window[conc_col].tolist(),
                "peak_value": None if peak.empty else float(peak.iloc[0]),
                "pre_mean": None if pre.empty else float(pre.mean()),
                "post_mean": None if post.empty else float(post.mean()),
                "local_change_pct_vs_context": local_change_pct,
                "isolated_spike_candidate": isolated_spike_candidate,
                "n_points_in_window": int(len(window)),
            }
        )

    def build(self, event: pd.Series) -> dict[str, Any]:
        """Build the evidence packet for one event."""
        site_id = str(event["site_id"])
        peak_date = pd.Timestamp(event["peak_date"])
        state = self._resolve_state(event)

        short_start = (peak_date - pd.Timedelta(days=2)).date().isoformat()
        peak_day = peak_date.date().isoformat()
        nearby_start = (peak_date - pd.Timedelta(days=3)).date().isoformat()
        nearby_end = (peak_date + pd.Timedelta(days=3)).date().isoformat()
        hosp_start = (peak_date - pd.Timedelta(days=7)).date().isoformat()
        hosp_end = (peak_date + pd.Timedelta(days=settings.agent.clinical_lag_days)).date().isoformat()
        variant_start = (peak_date - pd.Timedelta(days=28)).date().isoformat()

        packet = {
            "event": {
                "event_id": str(event["event_id"]),
                "site_id": site_id,
                "state": state,
                "peak_date": peak_day,
                "duration_days": int(event.get("duration_days", 1) or 1),
                "peak_zscore": self._round_float(event.get("peak_zscore", 0.0) or 0.0),
                "mean_zscore": self._round_float(event.get("mean_zscore", 0.0) or 0.0),
                "vote_count_max": int(event.get("vote_count_max", 0) or 0),
                "detection_methods": ""
                if pd.isna(event.get("detection_methods", ""))
                else str(event.get("detection_methods", "") or ""),
            },
            "signal_context": self._signal_context(event),
            "site_metadata": self._sanitize(self.site_tool.execute(site_id)),
            "weather": self._sanitize(self.weather_tool.execute(site_id, short_start, peak_day)),
            "hydrology": self._sanitize(self.hydrology_tool.execute(site_id, short_start, peak_day)),
            "hospitalization": self._sanitize(
                self.hosp_tool.execute(state=state, start_date=hosp_start, end_date=hosp_end)
                if state
                else {"error": "State unavailable"}
            ),
            "nearby_sites": self._sanitize(self.nearby_tool.execute(site_id, nearby_start, nearby_end)),
            "variants": self._sanitize(
                self.variants_tool.execute(state=state, start_date=variant_start, end_date=peak_day)
                if state
                else {"error": "State unavailable"}
            ),
        }
        packet["derived_assessment"] = self._sanitize(self._derived_assessment(packet))
        return packet

    def _derived_assessment(self, packet: dict[str, Any]) -> dict[str, Any]:
        peak_date = pd.Timestamp(packet["event"]["peak_date"])
        state = packet["event"]["state"]
        site_id = packet["event"]["site_id"]
        dk = settings.agent.domain_knowledge

        weather = packet.get("weather", {})
        precip_block = weather.get("daily_precipitation_mm", {})
        rain_sum = precip_block.get("sum")
        rain_max = precip_block.get("max")

        hydro = packet.get("hydrology", {})
        flow_block = hydro.get("discharge_percentile", {})
        flow_max = flow_block.get("max")

        hosp_target = "previous_day_admission_adult_covid_confirmed"
        pre_mask = (
            (self.db["state"] == state)
            & (self.db["date"] >= peak_date - pd.Timedelta(days=7))
            & (self.db["date"] <= peak_date)
        )
        post_mask = (
            (self.db["state"] == state)
            & (self.db["date"] > peak_date)
            & (self.db["date"] <= peak_date + pd.Timedelta(days=settings.agent.clinical_lag_days))
        )
        pre_clin = self.db.loc[pre_mask, hosp_target].dropna() if hosp_target in self.db.columns else pd.Series(dtype=float)
        post_clin = self.db.loc[post_mask, hosp_target].dropna() if hosp_target in self.db.columns else pd.Series(dtype=float)

        clinical_change_pct = None
        has_clinical_rise = False
        if not pre_clin.empty and not post_clin.empty and float(pre_clin.mean()) > 0:
            clinical_change_pct = (float(post_clin.mean()) - float(pre_clin.mean())) / float(pre_clin.mean()) * 100
            has_clinical_rise = clinical_change_pct > dk.clinical_followup_threshold_pct

        variants = packet.get("variants", {})
        variant_shifts = variants.get("notable_shifts", []) or []
        has_variant_shift = len(variant_shifts) > 0

        nearby = packet.get("nearby_sites", {})
        nearby_regional = bool(nearby.get("is_regional", False))

        signal = packet.get("signal_context", {})
        isolated_spike = bool(signal.get("isolated_spike_candidate", False))

        return {
            "site_id": site_id,
            "rain_sum_mm_last_3d": rain_sum,
            "rain_max_mm_last_3d": rain_max,
            "has_moderate_rain": bool((rain_sum or 0) >= dk.rainfall_dilution_threshold_mm or (rain_max or 0) >= dk.rainfall_dilution_threshold_mm),
            "has_significant_rain": bool((rain_sum or 0) >= dk.rainfall_significant_mm or (rain_max or 0) >= dk.rainfall_significant_mm),
            "flow_percentile_max_last_3d": flow_max,
            "has_high_flow": bool((flow_max or 0) >= dk.cso_flow_percentile / 100),
            "clinical_pre_mean": None if pre_clin.empty else float(pre_clin.mean()),
            "clinical_post_mean": None if post_clin.empty else float(post_clin.mean()),
            "clinical_change_pct": clinical_change_pct,
            "has_clinical_rise": has_clinical_rise,
            "has_variant_shift": has_variant_shift,
            "variant_shift_count": len(variant_shifts),
            "is_regional_pattern": nearby_regional,
            "isolated_spike_candidate": isolated_spike,
            "epidemic_candidate": bool(
                has_clinical_rise
                and not (
                    (rain_sum or 0) >= dk.rainfall_significant_mm
                    or (rain_max or 0) >= dk.rainfall_significant_mm
                    or (flow_max or 0) >= dk.cso_flow_percentile / 100
                )
            ),
            "epidemic_with_regional_support": bool(
                has_clinical_rise and nearby_regional
            ),
            "epidemic_with_variant_shift": bool(
                has_clinical_rise and has_variant_shift
            ),
            "mixed_candidate": bool(has_clinical_rise and (has_variant_shift or (rain_sum or 0) >= dk.rainfall_dilution_threshold_mm or (flow_max or 0) >= dk.cso_flow_percentile / 100)),
        }


class LLMSilverLabeler:
    """Independent LLM adjudicator over evidence packets."""

    def __init__(
        self,
        database: pd.DataFrame,
        site_metadata: pd.DataFrame,
        judge_models: list[str],
        confidence_threshold: float = 0.6,
        agreement_threshold: float = 0.66,
    ):
        api_key = (
            settings.agent.api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not api_key:
            raise ValueError("No API key configured for LLM silver-label generation.")

        self.client = OpenAI(
            base_url=settings.agent.api_base,
            api_key=api_key,
            timeout=120.0,
            max_retries=2,
        )
        self.packet_builder = EvidencePacketBuilder(database=database, site_metadata=site_metadata)
        self.judge_models = judge_models
        self.confidence_threshold = confidence_threshold
        self.agreement_threshold = agreement_threshold

    def _parse_judgement(self, content: str) -> SilverLabelJudgement:
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            raise ValueError("No JSON object found in model response.")

        payload = json.loads(content[json_start:json_end])
        payload["label"] = normalize_label(payload.get("label"))
        return SilverLabelJudgement(**payload)

    def _judge_once(self, event_packet: dict[str, Any], judge_model: str) -> tuple[SilverLabelJudgement, str, int]:
        messages = [
            {"role": "system", "content": get_judge_system_prompt()},
            {"role": "user", "content": format_evidence_prompt(event_packet)},
        ]

        try:
            response = self.client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self.client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=0,
            )

        content = response.choices[0].message.content or ""
        total_tokens = response.usage.total_tokens if response.usage else 0
        return self._parse_judgement(content), content, int(total_tokens)

    def judge_event(self, event: pd.Series) -> tuple[list[JudgeRunResult], dict[str, Any]]:
        event_packet = self.packet_builder.build(event)
        raw_results: list[JudgeRunResult] = []

        for judge_model in self.judge_models:
            try:
                verdict, raw_response, total_tokens = self._judge_once(event_packet, judge_model)
                raw_results.append(
                    JudgeRunResult(
                        event_id=str(event["event_id"]),
                        judge_model=judge_model,
                        label=normalize_label(verdict.label),
                        confidence=verdict.confidence,
                        rationale=verdict.rationale,
                        key_evidence=verdict.key_evidence,
                        data_gaps=verdict.data_gaps,
                        raw_response=raw_response,
                        total_tokens=total_tokens,
                    )
                )
            except Exception as exc:
                logger.error(f"Judge failed for {event['event_id']} with {judge_model}: {exc}")
                raw_results.append(
                    JudgeRunResult(
                        event_id=str(event["event_id"]),
                        judge_model=judge_model,
                        label="uncertain",
                        confidence=0.0,
                        rationale="Judge call failed",
                        key_evidence=[],
                        data_gaps=[str(exc)],
                        raw_response="",
                        error=str(exc),
                    )
                )

        consensus = self._aggregate_results(event, raw_results)
        return raw_results, consensus

    def _aggregate_results(self, event: pd.Series, raw_results: list[JudgeRunResult]) -> dict[str, Any]:
        valid = [r for r in raw_results if not r.error]
        if not valid:
            return {
                "ground_truth_label": "uncertain",
                "ground_truth_confidence": 0.0,
                "label_source": "llm_silver_failed",
                "label_kind": "silver",
                "consensus_status": "all_failed",
                "agreement_ratio": 0.0,
                "judge_models": "|".join(self.judge_models),
                "n_judges": len(self.judge_models),
                "n_valid_judges": 0,
                "rationale": "All judge calls failed.",
                "key_evidence": json.dumps([], ensure_ascii=True),
                "data_gaps": json.dumps([r.error for r in raw_results if r.error], ensure_ascii=True),
                "raw_labels": json.dumps([], ensure_ascii=True),
                "raw_confidences": json.dumps([], ensure_ascii=True),
            }

        label_counts = Counter(r.label for r in valid)
        top_label, top_count = label_counts.most_common(1)[0]
        agreement_ratio = top_count / len(valid)
        supporters = [r for r in valid if r.label == top_label]
        mean_conf = float(np.mean([r.confidence for r in supporters])) if supporters else 0.0
        consensus_status = "accepted"
        final_label = top_label

        if agreement_ratio < self.agreement_threshold:
            final_label = "uncertain"
            consensus_status = "disagreement"
        elif mean_conf < self.confidence_threshold and top_label != "uncertain":
            final_label = "uncertain"
            consensus_status = "low_confidence"
        elif len(label_counts) > 1:
            consensus_status = "majority"

        key_evidence = []
        data_gaps = []
        rationales = []
        for result in supporters:
            key_evidence.extend(result.key_evidence)
            data_gaps.extend(result.data_gaps)
            rationales.append(result.rationale)

        unique_evidence = list(dict.fromkeys([x for x in key_evidence if x]))
        unique_gaps = list(dict.fromkeys([x for x in data_gaps if x]))

        return {
            "ground_truth_label": normalize_label(final_label),
            "ground_truth_confidence": round(mean_conf, 4),
            "label_source": "llm_silver_consensus",
            "label_kind": "silver",
            "consensus_status": consensus_status,
            "agreement_ratio": round(agreement_ratio, 4),
            "judge_models": "|".join(self.judge_models),
            "n_judges": len(self.judge_models),
            "n_valid_judges": len(valid),
            "rationale": " | ".join(rationales[:2]) if rationales else "",
            "key_evidence": json.dumps(unique_evidence[:8], ensure_ascii=True),
            "data_gaps": json.dumps(unique_gaps[:8], ensure_ascii=True),
            "raw_labels": json.dumps([r.label for r in raw_results], ensure_ascii=True),
            "raw_confidences": json.dumps([round(r.confidence, 4) for r in raw_results], ensure_ascii=True),
        }

    def adjudicate_events(self, events_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Adjudicate all provided events."""
        final_rows: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] = []

        for idx, (_, event) in enumerate(events_df.iterrows(), start=1):
            logger.info(f"[{idx}/{len(events_df)}] Silver-label adjudication: {event['event_id']}")
            raw_results, final_result = self.judge_event(event)

            final_rows.append({**event.to_dict(), **final_result})
            for raw in raw_results:
                raw_rows.append(
                    {
                        "event_id": raw.event_id,
                        "judge_model": raw.judge_model,
                        "label": raw.label,
                        "confidence": round(raw.confidence, 4),
                        "rationale": raw.rationale,
                        "key_evidence": json.dumps(raw.key_evidence, ensure_ascii=True),
                        "data_gaps": json.dumps(raw.data_gaps, ensure_ascii=True),
                        "total_tokens": raw.total_tokens,
                        "error": raw.error,
                        "raw_response": raw.raw_response,
                    }
                )

        final_df = pd.DataFrame(final_rows)
        raw_df = pd.DataFrame(raw_rows)
        return final_df, raw_df


def write_label_outputs(
    final_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    output_file: Path,
    raw_output_dir: Path,
) -> None:
    """Persist silver labels and raw judge outputs."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(output_file, index=False)
    raw_df.to_csv(raw_output_dir / "llm_judge_raw.csv", index=False)

    summary = {
        "n_events": int(len(final_df)),
        "label_distribution": final_df["ground_truth_label"].value_counts().to_dict()
        if "ground_truth_label" in final_df.columns
        else {},
        "consensus_status": final_df["consensus_status"].value_counts().to_dict()
        if "consensus_status" in final_df.columns
        else {},
        "avg_confidence": round(float(pd.to_numeric(final_df["ground_truth_confidence"], errors="coerce").fillna(0).mean()), 4)
        if "ground_truth_confidence" in final_df.columns
        else 0.0,
    }
    (raw_output_dir / "llm_judge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    logger.info(f"Silver labels saved to: {output_file}")
    logger.info(f"Raw judge outputs saved to: {raw_output_dir}")


def run_silver_labeling(
    events_file: str | None,
    output_file: str,
    raw_output_dir: str,
    judge_models: list[str],
    confidence_threshold: float,
    agreement_threshold: float,
    max_events: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end silver-label run."""
    ensure_dirs()
    database = load_merged_database()
    site_meta = load_site_metadata()
    events = load_events_catalog(events_file)
    if max_events:
        events = events.head(max_events).copy()

    labeler = LLMSilverLabeler(
        database=database,
        site_metadata=site_meta,
        judge_models=judge_models,
        confidence_threshold=confidence_threshold,
        agreement_threshold=agreement_threshold,
    )
    final_df, raw_df = labeler.adjudicate_events(events)
    write_label_outputs(
        final_df=final_df,
        raw_df=raw_df,
        output_file=Path(output_file),
        raw_output_dir=Path(raw_output_dir),
    )
    return final_df, raw_df
