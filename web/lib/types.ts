export type EventRecord = {
  event_id: string;
  site_id: string;
  start_date: string;
  end_date: string;
  peak_date: string;
  duration_days: number;
  peak_zscore: number;
  mean_zscore: number;
  detection_methods: string;
  vote_count_max: number;
  classification: string | null;
  confidence: number | null;
  summary: string | null;
};

export type EventListResponse = {
  total: number;
  items: EventRecord[];
};

export type SummaryResponse = {
  event_count: number;
  classified_count: number;
  unclassified_count: number;
  by_classification: Record<string, number>;
};

export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export type JobRecord = {
  id: string;
  job_type: "prepare_data" | "pipeline" | "detection" | "agent" | "evaluation";
  status: JobStatus;
  command: string[];
  args: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  return_code: number | null;
  error: string | null;
  log_file: string | null;
  log_tail: string[];
};

export type JobListResponse = {
  items: JobRecord[];
};

// ── New types ──

export type TimeSeriesPoint = {
  date: string;
  concentration: number | null;
  rolling_median_7d: number | null;
  rolling_zscore_7d: number | null;
  precipitation_mm: number | null;
  temp_avg_c: number | null;
  discharge_cfs: number | null;
  admission_7d_avg: number | null;
};

export type TimeSeriesResponse = {
  site_id: string;
  points: TimeSeriesPoint[];
};

export type SiteRecord = {
  site_id: string;
  state: string;
  date_min: string;
  date_max: string;
  n_days: number;
  event_count: number;
  classified_count: number;
};

export type SiteListResponse = {
  total: number;
  items: SiteRecord[];
};

export type StatsResponse = {
  by_classification: Record<string, number>;
  by_site: Record<string, number>;
  by_month: Record<string, number>;
  detection_method_counts: Record<string, number>;
  duration_histogram: number[];
  zscore_histogram: number[];
};

// ── Experiment types ──

export type BenchmarkRow = {
  method: string;
  n_samples: number;
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  cohen_kappa: number;
  avg_tokens: number | null;
  avg_tool_calls: number | null;
};

export type EvaluationResponse = {
  rows: BenchmarkRow[];
};

export type AblationRow = {
  ablation: string;
  n_samples: number;
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  cohen_kappa: number;
  avg_tool_calls: number;
  avg_tokens: number;
};

export type AblationResponse = {
  full_agent_f1: number;
  rows: AblationRow[];
};

export type LabelStatsResponse = {
  n_events: number;
  avg_confidence: number;
  label_distribution: Record<string, number>;
  consensus_status: Record<string, number>;
};
