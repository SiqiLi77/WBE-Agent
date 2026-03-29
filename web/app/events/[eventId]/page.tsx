"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useLocale } from "@/lib/i18n-context";

import { ClassificationPill } from "@/components/classification-pill";
import {
  ConcentrationChart,
  EnvironmentChart,
  HospitalizationChart,
} from "@/components/timeseries-chart";
import { fetchEvent, fetchEventTrace, fetchSiteTimeseries } from "@/lib/api";
import type { TimeSeriesPoint } from "@/lib/types";

function trimAround(
  points: TimeSeriesPoint[],
  start: string,
  end: string,
  padDays = 30,
): TimeSeriesPoint[] {
  const s = new Date(start);
  const e = new Date(end);
  s.setDate(s.getDate() - padDays);
  e.setDate(e.getDate() + padDays);
  return points.filter((p) => {
    const d = new Date(p.date);
    return d >= s && d <= e;
  });
}

export default function EventDetailPage() {
  const { t } = useLocale();
  const params = useParams<{ eventId: string }>();
  const eventId = params.eventId;

  const eventQ = useQuery({
    queryKey: ["event", eventId],
    queryFn: () => fetchEvent(eventId),
  });

  const traceQ = useQuery({
    queryKey: ["trace", eventId],
    queryFn: () => fetchEventTrace(eventId),
    retry: false,
  });

  const siteId = eventQ.data?.site_id;
  const tsQ = useQuery({
    queryKey: ["timeseries", siteId],
    queryFn: () => fetchSiteTimeseries(siteId!),
    enabled: !!siteId,
    retry: false,
  });

  if (eventQ.isLoading) {
    return <main className="mx-auto max-w-6xl px-4 py-6">Loading...</main>;
  }

  if (eventQ.isError || !eventQ.data) {
    return (
      <main className="mx-auto max-w-6xl px-4 py-6">
        <p className="text-red-700">Failed to load event.</p>
        <Link href="/" className="text-teal-700 underline">{t("event.back")}</Link>
      </main>
    );
  }

  const event = eventQ.data;
  const trimmed = tsQ.data
    ? trimAround(tsQ.data.points, event.start_date, event.end_date)
    : [];

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-5 px-4 py-6">
      <header className="panel p-5 fade-up">
        <Link href="/" className="mono text-xs text-teal-700 underline">
          {t("event.back")}
        </Link>
        <h1 className="mt-3 text-2xl font-semibold">{event.event_id}</h1>
        <div className="mt-3 flex flex-wrap gap-2">
          <ClassificationPill value={event.classification} />
          <span className="label-pill">Site: {event.site_id}</span>
          <span className="label-pill">{event.start_date} → {event.end_date}</span>
          <span className="label-pill">{t("events.duration")}: {event.duration_days}d</span>
          <span className="label-pill">Peak Z: {event.peak_zscore.toFixed(2)}</span>
          <span className="label-pill">Votes: {event.vote_count_max}/4</span>
          {event.detection_methods && (
            <span className="label-pill">Methods: {event.detection_methods}</span>
          )}
        </div>
        {event.summary && (
          <div className="mt-4">
            <div className="text-xs font-semibold text-slate-500 mb-1">{t("event.summary")}</div>
            <p className="text-sm leading-relaxed text-slate-700">{event.summary}</p>
          </div>
        )}
      </header>

      {tsQ.isLoading && (
        <div className="panel p-5 text-sm text-slate-500">Loading...</div>
      )}
      {trimmed.length > 0 && (
        <section className="flex flex-col gap-4 fade-up [animation-delay:100ms]">
          <div className="panel p-5">
            <ConcentrationChart
              points={trimmed}
              eventStart={event.start_date}
              eventEnd={event.end_date}
            />
          </div>
          <div className="panel p-5">
            <EnvironmentChart points={trimmed} />
          </div>
          <div className="panel p-5">
            <HospitalizationChart points={trimmed} />
          </div>
        </section>
      )}

      <section className="panel p-5 fade-up [animation-delay:200ms]">
        <h2 className="mb-3 text-lg font-semibold">{t("event.trace")}</h2>
        {traceQ.isError ? (
          <p className="text-sm text-slate-500">Trace file not found.</p>
        ) : traceQ.isLoading ? (
          <p className="text-sm text-slate-500">Loading...</p>
        ) : (
          <pre className="mono max-h-[560px] overflow-auto rounded-xl border border-slate-200 bg-slate-900 p-4 text-[11px] text-slate-100">
            {JSON.stringify(traceQ.data, null, 2)}
          </pre>
        )}
      </section>
    </main>
  );
}
