"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { useLocale } from "@/lib/i18n-context";
import { fetchSites } from "@/lib/api";

export default function SitesPage() {
  const { t } = useLocale();
  const sitesQ = useQuery({
    queryKey: ["sites"],
    queryFn: fetchSites,
    refetchInterval: 15000,
  });

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-5 px-4 py-6 md:px-8">
      <header className="fade-up">
        <h1 className="text-3xl font-semibold">{t("sites.title")}</h1>
        <p className="mt-2 text-sm text-slate-600">{t("sites.desc")}</p>
      </header>

      {sitesQ.isError && (
        <section className="panel border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {t("dash.api_error")}
        </section>
      )}

      {sitesQ.isLoading && (
        <div className="text-sm text-slate-500">Loading...</div>
      )}

      {sitesQ.data && (
        <section className="panel overflow-hidden fade-up [animation-delay:100ms]">
          <div className="border-b border-slate-200 px-5 py-4">
            <span className="text-lg font-semibold">{t("nav.sites")}</span>
            <span className="ml-2 mono text-xs text-slate-500">
              {sitesQ.data.total} total
            </span>
          </div>
          <div className="overflow-auto">
            <table className="w-full min-w-[800px] border-collapse">
              <thead>
                <tr className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="px-4 py-3">Site ID</th>
                  <th className="px-4 py-3">{t("sites.state")}</th>
                  <th className="px-4 py-3">{t("sites.date_range")}</th>
                  <th className="px-4 py-3">{t("sites.observations")}</th>
                  <th className="px-4 py-3">{t("sites.events")}</th>
                  <th className="px-4 py-3">{t("card.classified")}</th>
                </tr>
              </thead>
              <tbody>
                {sitesQ.data.items.map((site) => (
                  <tr key={site.site_id} className="border-t border-slate-100 hover:bg-slate-50">
                    <td className="px-4 py-3 text-sm">
                      <Link
                        href={`/sites/${encodeURIComponent(site.site_id)}`}
                        className="text-teal-700 hover:underline"
                      >
                        {site.site_id.length > 50 ? site.site_id.slice(0, 50) + "…" : site.site_id}
                      </Link>
                    </td>
                    <td className="px-4 py-3 text-sm">{site.state}</td>
                    <td className="px-4 py-3 mono text-xs">{site.date_min} → {site.date_max}</td>
                    <td className="px-4 py-3 mono text-xs">{site.n_days}</td>
                    <td className="px-4 py-3 mono text-xs">{site.event_count}</td>
                    <td className="px-4 py-3 mono text-xs">{site.classified_count}</td>
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
