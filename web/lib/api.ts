import type {
  AblationResponse,
  EvaluationResponse,
  EventListResponse,
  EventRecord,
  JobListResponse,
  JobRecord,
  LabelStatsResponse,
  SiteListResponse,
  StatsResponse,
  SummaryResponse,
  TimeSeriesResponse,
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function parseJson<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `${resp.status} ${resp.statusText}`);
  }
  return (await resp.json()) as T;
}

export async function fetchSummary(): Promise<SummaryResponse> {
  const resp = await fetch(`${API_BASE}/api/summary`, { cache: "no-store" });
  return parseJson<SummaryResponse>(resp);
}

export async function fetchStats(): Promise<StatsResponse> {
  const resp = await fetch(`${API_BASE}/api/stats`, { cache: "no-store" });
  return parseJson<StatsResponse>(resp);
}

export async function fetchEvents(limit = 100): Promise<EventListResponse> {
  const resp = await fetch(`${API_BASE}/api/events?limit=${limit}&offset=0`, { cache: "no-store" });
  return parseJson<EventListResponse>(resp);
}

export async function fetchEventsByFilter(params: {
  limit?: number;
  classification?: string;
  site_id?: string;
}): Promise<EventListResponse> {
  const qs = new URLSearchParams();
  qs.set("limit", String(params.limit ?? 200));
  qs.set("offset", "0");
  if (params.classification) qs.set("classification", params.classification);
  if (params.site_id) qs.set("site_id", params.site_id);
  const resp = await fetch(`${API_BASE}/api/events?${qs}`, { cache: "no-store" });
  return parseJson<EventListResponse>(resp);
}

export async function fetchEvent(eventId: string): Promise<EventRecord> {
  const resp = await fetch(`${API_BASE}/api/events/${eventId}`, { cache: "no-store" });
  return parseJson<EventRecord>(resp);
}

export async function fetchEventTrace(eventId: string): Promise<Record<string, unknown>> {
  const resp = await fetch(`${API_BASE}/api/events/${eventId}/trace`, { cache: "no-store" });
  return parseJson<Record<string, unknown>>(resp);
}

export async function fetchSites(): Promise<SiteListResponse> {
  const resp = await fetch(`${API_BASE}/api/sites`, { cache: "no-store" });
  return parseJson<SiteListResponse>(resp);
}

export async function fetchSiteTimeseries(siteId: string): Promise<TimeSeriesResponse> {
  const resp = await fetch(`${API_BASE}/api/sites/${encodeURIComponent(siteId)}/timeseries`, {
    cache: "no-store",
  });
  return parseJson<TimeSeriesResponse>(resp);
}

export async function fetchJobs(): Promise<JobListResponse> {
  const resp = await fetch(`${API_BASE}/api/jobs`, { cache: "no-store" });
  return parseJson<JobListResponse>(resp);
}

export async function createJob(payload: {
  job_type: JobRecord["job_type"];
  args?: Record<string, unknown>;
}): Promise<JobRecord> {
  const resp = await fetch(`${API_BASE}/api/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJson<JobRecord>(resp);
}


// ── Experiment endpoints ──

export async function fetchEvaluation(): Promise<EvaluationResponse> {
  const resp = await fetch(`${API_BASE}/api/evaluation`, { cache: "no-store" });
  return parseJson<EvaluationResponse>(resp);
}

export async function fetchAblation(): Promise<AblationResponse> {
  const resp = await fetch(`${API_BASE}/api/ablation`, { cache: "no-store" });
  return parseJson<AblationResponse>(resp);
}

export async function fetchLabelStats(): Promise<LabelStatsResponse> {
  const resp = await fetch(`${API_BASE}/api/labels`, { cache: "no-store" });
  return parseJson<LabelStatsResponse>(resp);
}
