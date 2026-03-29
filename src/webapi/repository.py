from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import PROJECT_ROOT, settings
from src.webapi.schemas import (
    AblationResponse,
    AblationRow,
    BenchmarkRow,
    EvaluationResponse,
    EventRecord,
    LabelStatsResponse,
    SiteListResponse,
    SiteRecord,
    StatsResponse,
    SummaryResponse,
    TimeSeriesPoint,
    TimeSeriesResponse,
)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _event_catalog_path() -> Path:
    return PROJECT_ROOT / settings.paths.outputs_dir / "anomaly_event_catalog.csv"


def _investigation_results_path() -> Path:
    return PROJECT_ROOT / settings.paths.outputs_dir / "investigation_results.csv"


def _trace_path(event_id: str) -> Path:
    return PROJECT_ROOT / settings.logging.log_dir / "traces" / f"{event_id}_trace.json"


# ── Events ─────────────────────────────────────────────────

def read_events() -> list[EventRecord]:
    events = _safe_read_csv(_event_catalog_path())
    reports = _safe_read_csv(_investigation_results_path())

    if events.empty:
        return []

    if not reports.empty and "event_id" in reports.columns:
        merge_cols = [
            c for c in ["event_id", "classification", "confidence", "summary"]
            if c in reports.columns
        ]
        events = events.merge(reports[merge_cols], on="event_id", how="left")

    for col, default in [
        ("detection_methods", ""),
        ("vote_count_max", 0),
        ("classification", None),
        ("confidence", None),
        ("summary", None),
    ]:
        if col not in events.columns:
            events[col] = default

    if "peak_date" in events.columns:
        events = events.sort_values("peak_date", ascending=False)

    records: list[EventRecord] = []
    for _, row in events.iterrows():
        records.append(
            EventRecord(
                event_id=str(row.get("event_id", "")),
                site_id=str(row.get("site_id", "")),
                start_date=str(row.get("start_date", "")),
                end_date=str(row.get("end_date", "")),
                peak_date=str(row.get("peak_date", "")),
                duration_days=int(row.get("duration_days", 0) or 0),
                peak_zscore=float(row.get("peak_zscore", 0.0) or 0.0),
                mean_zscore=float(row.get("mean_zscore", 0.0) or 0.0),
                detection_methods=(
                    "" if pd.isna(row.get("detection_methods"))
                    else str(row.get("detection_methods", ""))
                ),
                vote_count_max=int(row.get("vote_count_max", 0) or 0),
                classification=(
                    None if pd.isna(row.get("classification"))
                    else str(row.get("classification"))
                ),
                confidence=(
                    None if pd.isna(row.get("confidence"))
                    else float(row.get("confidence"))
                ),
                summary=(
                    None if pd.isna(row.get("summary"))
                    else str(row.get("summary"))
                ),
            )
        )
    return records


def read_event(event_id: str) -> EventRecord | None:
    for item in read_events():
        if item.event_id == event_id:
            return item
    return None


def read_summary() -> SummaryResponse:
    events = read_events()
    by_classification: dict[str, int] = {}
    classified_count = 0
    for event in events:
        if event.classification:
            classified_count += 1
            by_classification[event.classification] = (
                by_classification.get(event.classification, 0) + 1
            )
    return SummaryResponse(
        event_count=len(events),
        classified_count=classified_count,
        unclassified_count=len(events) - classified_count,
        by_classification=by_classification,
    )


def read_trace(event_id: str) -> dict | None:
    path = _trace_path(event_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── Time-series data ───────────────────────────────────────

_merged_cache: pd.DataFrame | None = None


def _load_merged() -> pd.DataFrame:
    global _merged_cache
    if _merged_cache is not None:
        return _merged_cache
    path = PROJECT_ROOT / settings.paths.merged_database
    if not path.exists():
        logger.warning(f"Merged database not found: {path}")
        return pd.DataFrame()
    _merged_cache = pd.read_parquet(path)
    if "date" in _merged_cache.columns:
        _merged_cache["date"] = pd.to_datetime(_merged_cache["date"])
    return _merged_cache


def read_timeseries(site_id: str) -> TimeSeriesResponse:
    df = _load_merged()
    if df.empty:
        return TimeSeriesResponse(site_id=site_id, points=[])
    site_df = df[df["site_id"] == site_id].sort_values("date")
    points: list[TimeSeriesPoint] = []
    for _, row in site_df.iterrows():
        dt = row["date"]
        points.append(TimeSeriesPoint(
            date=str(dt.date()) if hasattr(dt, "date") else str(dt),
            concentration=(
                None if pd.isna(row.get("pcr_conc_lin_log1p"))
                else round(float(row["pcr_conc_lin_log1p"]), 4)
            ),
            rolling_median_7d=(
                None if pd.isna(row.get("rolling_median_7d"))
                else round(float(row["rolling_median_7d"]), 4)
            ),
            rolling_zscore_7d=(
                None if pd.isna(row.get("rolling_zscore_7d"))
                else round(float(row["rolling_zscore_7d"]), 2)
            ),
            precipitation_mm=(
                None if pd.isna(row.get("precipitation_mm"))
                else round(float(row["precipitation_mm"]), 1)
            ),
            temp_avg_c=(
                None if pd.isna(row.get("temp_avg_c"))
                else round(float(row["temp_avg_c"]), 1)
            ),
            discharge_cfs=(
                None if pd.isna(row.get("discharge_cfs"))
                else round(float(row["discharge_cfs"]), 1)
            ),
            admission_7d_avg=(
                None if pd.isna(row.get("admission_7d_avg"))
                else round(float(row["admission_7d_avg"]), 1)
            ),
        ))
    return TimeSeriesResponse(site_id=site_id, points=points)


# ── Site list ──────────────────────────────────────────────

def read_sites() -> SiteListResponse:
    df = _load_merged()
    events = read_events()
    event_counts: dict[str, int] = {}
    classified_counts: dict[str, int] = {}
    for e in events:
        event_counts[e.site_id] = event_counts.get(e.site_id, 0) + 1
        if e.classification:
            classified_counts[e.site_id] = classified_counts.get(e.site_id, 0) + 1

    if df.empty:
        return SiteListResponse(total=0, items=[])

    items: list[SiteRecord] = []
    for site_id, grp in df.groupby("site_id"):
        sid = str(site_id)
        state = str(grp["state"].iloc[0]) if "state" in grp.columns else ""
        items.append(SiteRecord(
            site_id=sid,
            state=state,
            date_min=str(grp["date"].min().date()),
            date_max=str(grp["date"].max().date()),
            n_days=len(grp),
            event_count=event_counts.get(sid, 0),
            classified_count=classified_counts.get(sid, 0),
        ))
    items.sort(key=lambda x: x.event_count, reverse=True)
    return SiteListResponse(total=len(items), items=items)


# ── Stats / analytics ─────────────────────────────────────

def read_stats() -> StatsResponse:
    events = read_events()
    catalog = _safe_read_csv(_event_catalog_path())

    by_classification: dict[str, int] = {}
    by_site: dict[str, int] = {}
    for e in events:
        label = e.classification or "pending"
        by_classification[label] = by_classification.get(label, 0) + 1
        by_site[e.site_id] = by_site.get(e.site_id, 0) + 1

    by_month: dict[str, int] = {}
    for e in events:
        if e.peak_date:
            month_key = e.peak_date[:7]
            by_month[month_key] = by_month.get(month_key, 0) + 1

    detection_method_counts: dict[str, int] = {}
    if not catalog.empty and "detection_methods" in catalog.columns:
        for methods_str in catalog["detection_methods"].dropna():
            for m in str(methods_str).split(","):
                m = m.strip()
                if m:
                    detection_method_counts[m] = detection_method_counts.get(m, 0) + 1

    # Duration histogram: [1d, 2d, 3d, 4d, 5d, 6-10d, 11-20d, 21+d]
    duration_bins = [0] * 8
    for e in events:
        d = e.duration_days
        if d <= 0:
            continue
        if d <= 5:
            duration_bins[d - 1] += 1
        elif d <= 10:
            duration_bins[5] += 1
        elif d <= 20:
            duration_bins[6] += 1
        else:
            duration_bins[7] += 1

    # Z-score histogram: [0-1, 1-2, 2-3, 3-4, 4-5, 5+]
    zscore_bins = [0] * 6
    for e in events:
        z = e.peak_zscore
        idx = min(max(int(z), 0), 5)
        zscore_bins[idx] += 1

    return StatsResponse(
        by_classification=by_classification,
        by_site=by_site,
        by_month=dict(sorted(by_month.items())),
        detection_method_counts=detection_method_counts,
        duration_histogram=duration_bins,
        zscore_histogram=zscore_bins,
    )


# ── Evaluation / Ablation / Labels ─────────────────────────

def _evaluation_path() -> Path:
    return PROJECT_ROOT / settings.paths.outputs_dir / "evaluation" / "evaluation_summary.csv"


def _ablation_path() -> Path:
    return PROJECT_ROOT / settings.paths.outputs_dir / "ablation" / "ablation_summary_current.csv"


def _label_summary_path() -> Path:
    return PROJECT_ROOT / settings.paths.outputs_dir / "labeling" / "llm_judge_summary.json"


def read_evaluation() -> EvaluationResponse:
    df = _safe_read_csv(_evaluation_path())
    if df.empty:
        return EvaluationResponse(rows=[])
    rows: list[BenchmarkRow] = []
    for _, r in df.iterrows():
        rows.append(BenchmarkRow(
            method=str(r.get("method", "")),
            n_samples=int(r.get("n_samples", 0) or 0),
            accuracy=float(r.get("accuracy", 0) or 0),
            macro_f1=float(r.get("macro_f1", 0) or 0),
            weighted_f1=float(r.get("weighted_f1", 0) or 0),
            cohen_kappa=float(r.get("cohen_kappa", 0) or 0),
            avg_tokens=(
                None if pd.isna(r.get("avg_tokens"))
                else float(r["avg_tokens"])
            ),
            avg_tool_calls=(
                None if pd.isna(r.get("avg_tool_calls"))
                else float(r["avg_tool_calls"])
            ),
        ))
    return EvaluationResponse(rows=rows)


def read_ablation() -> AblationResponse:
    df = _safe_read_csv(_ablation_path())
    if df.empty:
        return AblationResponse(full_agent_f1=0, rows=[])

    # Get full agent F1 from evaluation
    eval_resp = read_evaluation()
    full_f1 = 0.0
    for row in eval_resp.rows:
        if row.method == "agent":
            full_f1 = row.macro_f1
            break

    rows: list[AblationRow] = []
    for _, r in df.iterrows():
        rows.append(AblationRow(
            ablation=str(r.get("ablation", "")),
            n_samples=int(r.get("n_samples", 0) or 0),
            accuracy=float(r.get("accuracy", 0) or 0),
            macro_f1=float(r.get("macro_f1", 0) or 0),
            weighted_f1=float(r.get("weighted_f1", 0) or 0),
            cohen_kappa=float(r.get("cohen_kappa", 0) or 0),
            avg_tool_calls=float(r.get("avg_tool_calls", 0) or 0),
            avg_tokens=float(r.get("avg_tokens", 0) or 0),
        ))
    return AblationResponse(full_agent_f1=full_f1, rows=rows)


def read_label_stats() -> LabelStatsResponse:
    path = _label_summary_path()
    if not path.exists():
        return LabelStatsResponse(
            n_events=0, avg_confidence=0, label_distribution={}, consensus_status={},
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LabelStatsResponse(
            n_events=data.get("n_events", 0),
            avg_confidence=data.get("avg_confidence", 0),
            label_distribution=data.get("label_distribution", {}),
            consensus_status=data.get("consensus_status", {}),
        )
    except Exception:
        logger.warning(f"Failed to read label summary: {path}")
        return LabelStatsResponse(
            n_events=0, avg_confidence=0, label_distribution={}, consensus_status={},
        )
