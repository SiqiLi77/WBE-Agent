from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


JobType = Literal["prepare_data", "pipeline", "detection", "agent", "evaluation"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class EventRecord(BaseModel):
    event_id: str
    site_id: str
    start_date: str
    end_date: str
    peak_date: str
    duration_days: int
    peak_zscore: float
    mean_zscore: float
    detection_methods: str = ""
    vote_count_max: int = 0
    classification: str | None = None
    confidence: float | None = None
    summary: str | None = None


class EventListResponse(BaseModel):
    total: int
    items: list[EventRecord]


class SummaryResponse(BaseModel):
    event_count: int
    classified_count: int
    unclassified_count: int
    by_classification: dict[str, int]


class JobCreateRequest(BaseModel):
    job_type: JobType
    args: dict[str, Any] = Field(default_factory=dict)


class JobRecord(BaseModel):
    id: str
    job_type: JobType
    status: JobStatus
    command: list[str]
    args: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    return_code: int | None = None
    error: str | None = None
    log_file: str | None = None
    log_tail: list[str] = Field(default_factory=list)


class JobListResponse(BaseModel):
    items: list[JobRecord]


# ── Time-series & analytics schemas ──────────────────────


class TimeSeriesPoint(BaseModel):
    date: str
    concentration: float | None = None
    rolling_median_7d: float | None = None
    rolling_zscore_7d: float | None = None
    precipitation_mm: float | None = None
    temp_avg_c: float | None = None
    discharge_cfs: float | None = None
    admission_7d_avg: float | None = None


class TimeSeriesResponse(BaseModel):
    site_id: str
    points: list[TimeSeriesPoint]


class SiteRecord(BaseModel):
    site_id: str
    state: str = ""
    date_min: str = ""
    date_max: str = ""
    n_days: int = 0
    event_count: int = 0
    classified_count: int = 0


class SiteListResponse(BaseModel):
    total: int
    items: list[SiteRecord]


class StatsResponse(BaseModel):
    by_classification: dict[str, int]
    by_site: dict[str, int]
    by_month: dict[str, int]
    detection_method_counts: dict[str, int]
    duration_histogram: list[int]
    zscore_histogram: list[int]


# ── Experiment schemas ───────────────────────────────────


class BenchmarkRow(BaseModel):
    method: str
    n_samples: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    cohen_kappa: float = 0.0
    avg_tokens: float | None = None
    avg_tool_calls: float | None = None


class EvaluationResponse(BaseModel):
    rows: list[BenchmarkRow]


class AblationRow(BaseModel):
    ablation: str
    n_samples: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    cohen_kappa: float = 0.0
    avg_tool_calls: float = 0.0
    avg_tokens: float = 0.0


class AblationResponse(BaseModel):
    full_agent_f1: float
    rows: list[AblationRow]


class LabelStatsResponse(BaseModel):
    n_events: int
    avg_confidence: float
    label_distribution: dict[str, int]
    consensus_status: dict[str, int]
