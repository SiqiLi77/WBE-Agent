from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.webapi.jobs import job_manager
from src.webapi.repository import (
    read_ablation,
    read_evaluation,
    read_event,
    read_events,
    read_label_stats,
    read_sites,
    read_stats,
    read_summary,
    read_timeseries,
    read_trace,
)
from src.webapi.schemas import (
    AblationResponse,
    EvaluationResponse,
    EventListResponse,
    EventRecord,
    JobCreateRequest,
    JobListResponse,
    JobRecord,
    LabelStatsResponse,
    SiteListResponse,
    StatsResponse,
    SummaryResponse,
    TimeSeriesResponse,
)


app = FastAPI(
    title="WBE-Agent Web API",
    version="0.2.0",
    description="API layer for the WBE-Agent web dashboard.",
)

origins_env = os.environ.get("WBE_API_CORS_ORIGINS", "")
if origins_env.strip():
    allowed_origins = [x.strip() for x in origins_env.split(",") if x.strip()]
else:
    allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Summary & Stats ────────────────────────────────────────

@app.get("/api/summary", response_model=SummaryResponse)
def get_summary() -> SummaryResponse:
    return read_summary()


@app.get("/api/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    return read_stats()


# ── Events ─────────────────────────────────────────────────

@app.get("/api/events", response_model=EventListResponse)
def list_events(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    classification: str | None = Query(default=None),
    site_id: str | None = Query(default=None),
) -> EventListResponse:
    items = read_events()
    if classification:
        items = [x for x in items if x.classification == classification]
    if site_id:
        items = [x for x in items if x.site_id == site_id]
    total = len(items)
    return EventListResponse(total=total, items=items[offset: offset + limit])


@app.get("/api/events/{event_id}", response_model=EventRecord)
def get_event(event_id: str) -> EventRecord:
    event = read_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event not found: {event_id}")
    return event


@app.get("/api/events/{event_id}/trace")
def get_event_trace(event_id: str) -> dict:
    trace = read_trace(event_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace not found: {event_id}")
    return trace


# ── Sites ──────────────────────────────────────────────────

# ── Experiments ─────────────────────────────────────────────

@app.get("/api/evaluation", response_model=EvaluationResponse)
def get_evaluation() -> EvaluationResponse:
    return read_evaluation()


@app.get("/api/ablation", response_model=AblationResponse)
def get_ablation() -> AblationResponse:
    return read_ablation()


@app.get("/api/labels", response_model=LabelStatsResponse)
def get_label_stats() -> LabelStatsResponse:
    return read_label_stats()


# ── Sites ──────────────────────────────────────────────────

@app.get("/api/sites", response_model=SiteListResponse)
def list_sites() -> SiteListResponse:
    return read_sites()


@app.get("/api/sites/{site_id}/timeseries", response_model=TimeSeriesResponse)
def get_site_timeseries(site_id: str) -> TimeSeriesResponse:
    ts = read_timeseries(site_id)
    if not ts.points:
        raise HTTPException(status_code=404, detail=f"No data for site: {site_id}")
    return ts


# ── Jobs ───────────────────────────────────────────────────

@app.get("/api/jobs", response_model=JobListResponse)
def list_jobs() -> JobListResponse:
    return JobListResponse(items=job_manager.list_jobs())


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@app.post("/api/jobs", response_model=JobRecord)
def create_job(req: JobCreateRequest) -> JobRecord:
    try:
        return job_manager.create_job(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
