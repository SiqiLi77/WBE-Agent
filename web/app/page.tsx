"use client";

import { useQuery } from "@tanstack/react-query";
import { useLocale } from "@/lib/i18n-context";

import {
  ClassificationChart,
  TimelineChart,
  DetectionMethodChart,
  DurationHistogram,
} from "@/components/dashboard-charts";
import { EventsTable } from "@/components/events-table";
import { JobControls } from "@/components/job-controls";
import { JobsPanel } from "@/components/jobs-panel";
import { SummaryCards } from "@/components/summary-cards";
import { fetchEvents, fetchJobs, fetchStats, fetchSummary } from "@/lib/api";

export default function HomePage() {
  const { t } = useLocale();

  const summaryQ = useQuery({
    queryKey: ["summary"],
    queryFn: fetchSummary,
    refetchInterval: 5000,
  });

  const statsQ = useQuery({
    queryKey: ["stats"],
    queryFn: fetchStats,
    refetchInterval: 10000,
  });

  const eventsQ = useQuery({
    queryKey: ["events"],
    queryFn: () => fetchEvents(200),
    refetchInterval: 7000,
  });

  const jobsQ = useQuery({
    queryKey: ["jobs"],
    queryFn: fetchJobs,
    refetchInterval: 5000,
  });

  const hasError = summaryQ.isError || eventsQ.isError || jobsQ.isError;

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-5 px-4 py-6 md:px-8">
      <header className="fade-up">
        <div className="mono mb-2 text-xs uppercase tracking-[0.18em] text-teal-700">
          {t("dash.subtitle")}
        </div>
        <h1 className="text-3xl font-semibold md:text-4xl">{t("dash.title")}</h1>
        <p className="mt-2 text-sm text-slate-600">{t("dash.desc")}</p>
      </header>

      {hasError && (
        <section className="panel border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {t("dash.api_error")}{" "}
          <span className="mono">
            {process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"}
          </span>
        </section>
      )}

      {summaryQ.data && <SummaryCards summary={summaryQ.data} />}

      {statsQ.data && (
        <section className="grid grid-cols-1 gap-4 md:grid-cols-2 fade-up [animation-delay:120ms]">
          <div className="panel p-4">
            <div className="mb-2 text-sm font-semibold text-slate-700">
              {t("dash.classification_dist")}
            </div>
            <ClassificationChart data={statsQ.data.by_classification} />
          </div>
          <div className="panel p-4">
            <div className="mb-2 text-sm font-semibold text-slate-700">
              {t("dash.events_by_month")}
            </div>
            <TimelineChart data={statsQ.data.by_month} />
          </div>
          <div className="panel p-4">
            <div className="mb-2 text-sm font-semibold text-slate-700">
              {t("dash.detection_methods")}
            </div>
            <DetectionMethodChart data={statsQ.data.detection_method_counts} />
          </div>
          <div className="panel p-4">
            <div className="mb-2 text-sm font-semibold text-slate-700">
              {t("dash.event_duration")}
            </div>
            <DurationHistogram data={statsQ.data.duration_histogram} />
          </div>
        </section>
      )}

      <JobControls />
      {eventsQ.data && <EventsTable events={eventsQ.data.items} />}
      {jobsQ.data && <JobsPanel jobs={jobsQ.data.items} />}
    </main>
  );
}
