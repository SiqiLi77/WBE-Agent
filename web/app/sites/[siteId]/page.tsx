"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useLocale } from "@/lib/i18n-context";

import {
  ConcentrationChart,
  EnvironmentChart,
  HospitalizationChart,
} from "@/components/timeseries-chart";
import { ClassificationPill } from "@/components/classification-pill";
import { fetchSiteTimeseries, fetchEventsByFilter } from "@/lib/api";

export default function SiteDetailPage() {
  const { t } = useLocale();
  const params = useParams<{ siteId: string }>();
  const siteId = decodeURIComponent(params.siteId);

  const tsQ = useQuery({
    queryKey: ["timeseries", siteId],
    queryFn: () => fetchSiteTimeseries(siteId),
    retry: false,
  });

  const eventsQ = useQuery({
    queryKey: ["events-site", siteId],
    queryFn: () => fetchEventsByFilter({ site_id: siteId, limit: 200 }),
  });

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-5 px-4 py-6">
      <header className="panel p-5 fade-up">
        <Link href="/sites" className="mono text-xs text-teal-700 underline">
          {t("site.back")}
        </Link>
        <h1 className="mt-3 text-xl font-semibold break-all">{siteId}</h1>
        {tsQ.data && (
          <span className="mono text-xs text-slate-500">
            {tsQ.data.points.length} data points
          </span>
        )}
      </header>

      {tsQ.isLoading && <div className="text-sm text-slate-500">Loading...</div>}

      {tsQ.isError && (
        <section className="panel border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {t("dash.api_error")}
        </section>
      )}

      {tsQ.data && tsQ.data.points.length > 0 && (
        <section className="flex flex-col gap-4 fade-up [animation-delay:100ms]">
          <div className="panel p-5">
            <ConcentrationChart points={tsQ.data.points} />
          </div>
          <div className="panel p-5">
            <EnvironmentChart points={tsQ.data.points} />
          </div>
          <div className="panel p-5">
            <HospitalizationChart points={tsQ.data.points} />
          </div>
        </section>
      )}

      {eventsQ.data && eventsQ.data.items.length > 0 && (
        <section className="panel overflow-hidden fade-up [animation-delay:200ms]">
          <div className="border-b border-slate-200 px-5 py-4">
            <span className="text-lg font-semibold">{t("site.events_at")}</span>
            <span className="ml-2 mono text-xs text-slate-500">{eventsQ.data.total}</span>
          </div>
          <div className="overflow-auto">
            <table className="w-full min-w-[700px] border-collapse">
              <thead>
                <tr className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="px-4 py-3">{t("events.event_id")}</th>
                  <th className="px-4 py-3">{t("events.peak_date")}</th>
                  <th className="px-4 py-3">{t("events.duration")}</th>
                  <th className="px-4 py-3">{t("events.zscore")}</th>
                  <th className="px-4 py-3">{t("events.classification")}</th>
                </tr>
              </thead>
              <tbody>
                {eventsQ.data.items.map((ev) => (
                  <tr key={ev.event_id} className="border-t border-slate-100 hover:bg-slate-50">
                    <td className="px-4 py-3 mono text-xs">
                      <Link href={`/events/${ev.event_id}`} className="text-teal-700 hover:underline">
                        {ev.event_id}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm">{ev.peak_date}</td>
                    <td className="px-4 py-3 text-sm">{ev.duration_days}d</td>
                    <td className="px-4 py-3 mono text-xs">{ev.peak_zscore.toFixed(2)}</td>
                    <td className="px-4 py-3"><ClassificationPill value={ev.classification} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </main>
  );
}
