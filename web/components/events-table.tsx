"use client";

import { useState, useMemo } from "react";
import Link from "next/link";

import { useLocale } from "@/lib/i18n-context";
import { ClassificationPill } from "@/components/classification-pill";
import type { EventRecord } from "@/lib/types";

type SortKey = "peak_date" | "duration_days" | "peak_zscore" | "site_id";

export function EventsTable({ events }: { events: EventRecord[] }) {
  const { t } = useLocale();
  const [filterClass, setFilterClass] = useState<string>("");
  const [filterSite, setFilterSite] = useState<string>("");
  const [sortKey, setSortKey] = useState<SortKey>("peak_date");
  const [sortAsc, setSortAsc] = useState(false);

  const classifications = useMemo(() => {
    const set = new Set<string>();
    events.forEach((e) => { if (e.classification) set.add(e.classification); });
    return Array.from(set).sort();
  }, [events]);

  const sites = useMemo(() => {
    const set = new Set<string>();
    events.forEach((e) => set.add(e.site_id));
    return Array.from(set).sort();
  }, [events]);

  const filtered = useMemo(() => {
    let list = events;
    if (filterClass) list = list.filter((e) => e.classification === filterClass);
    if (filterSite) list = list.filter((e) => e.site_id === filterSite);
    return [...list].sort((a, b) => {
      const av = a[sortKey] ?? "";
      const bv = b[sortKey] ?? "";
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortAsc ? cmp : -cmp;
    });
  }, [events, filterClass, filterSite, sortKey, sortAsc]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(false); }
  };

  const sortIcon = (key: SortKey) =>
    sortKey === key ? (sortAsc ? " ↑" : " ↓") : "";

  return (
    <section className="panel overflow-hidden fade-up [animation-delay:180ms]">
      <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 px-5 py-4">
        <h2 className="text-lg font-semibold">{t("events.title")}</h2>
        <span className="mono text-xs text-slate-500">{filtered.length} / {events.length}</span>
        <div className="ml-auto flex flex-wrap gap-2">
          <select
            value={filterClass}
            onChange={(e) => setFilterClass(e.target.value)}
            className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs"
            aria-label="Filter by classification"
          >
            <option value="">{t("events.all")}</option>
            {classifications.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <select
            value={filterSite}
            onChange={(e) => setFilterSite(e.target.value)}
            className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs"
            aria-label="Filter by site"
          >
            <option value="">{t("events.all")}</option>
            {sites.map((s) => (
              <option key={s} value={s}>{s.length > 40 ? s.slice(0, 40) + "…" : s}</option>
            ))}
          </select>
        </div>
      </div>
      <div className="overflow-auto">
        <table className="w-full min-w-[900px] border-collapse">
          <thead>
            <tr className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="px-4 py-3">{t("events.event_id")}</th>
              <th className="px-4 py-3 cursor-pointer select-none" onClick={() => toggleSort("site_id")}>
                {t("events.site")}{sortIcon("site_id")}
              </th>
              <th className="px-4 py-3 cursor-pointer select-none" onClick={() => toggleSort("peak_date")}>
                {t("events.peak_date")}{sortIcon("peak_date")}
              </th>
              <th className="px-4 py-3 cursor-pointer select-none" onClick={() => toggleSort("duration_days")}>
                {t("events.duration")}{sortIcon("duration_days")}
              </th>
              <th className="px-4 py-3 cursor-pointer select-none" onClick={() => toggleSort("peak_zscore")}>
                {t("events.zscore")}{sortIcon("peak_zscore")}
              </th>
              <th className="px-4 py-3">{t("events.classification")}</th>
              <th className="px-4 py-3">{t("events.confidence")}</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((item) => (
              <tr key={item.event_id} className="border-t border-slate-100 hover:bg-slate-50">
                <td className="px-4 py-3 mono text-xs">
                  <Link href={`/events/${item.event_id}`} className="text-teal-700 hover:underline">
                    {item.event_id}
                  </Link>
                </td>
                <td className="max-w-[320px] truncate px-4 py-3 text-sm">{item.site_id}</td>
                <td className="px-4 py-3 text-sm">{item.peak_date}</td>
                <td className="px-4 py-3 text-sm">{item.duration_days}d</td>
                <td className="px-4 py-3 mono text-xs">{item.peak_zscore.toFixed(2)}</td>
                <td className="px-4 py-3">
                  <ClassificationPill value={item.classification} />
                </td>
                <td className="px-4 py-3 mono text-xs">
                  {item.confidence === null ? "-" : item.confidence.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
