"use client";

import { useLocale } from "@/lib/i18n-context";
import type { SummaryResponse } from "@/lib/types";

export function SummaryCards({ summary }: { summary: SummaryResponse }) {
  const { t } = useLocale();

  return (
    <section className="grid grid-cols-1 gap-4 md:grid-cols-4">
      <div className="panel p-4 fade-up">
        <div className="text-sm text-slate-500">{t("card.total_events")}</div>
        <div className="mt-2 text-3xl font-semibold">{summary.event_count}</div>
      </div>
      <div className="panel p-4 fade-up [animation-delay:90ms]">
        <div className="text-sm text-slate-500">{t("card.classified")}</div>
        <div className="mt-2 text-3xl font-semibold">{summary.classified_count}</div>
      </div>
      <div className="panel p-4 fade-up [animation-delay:150ms]">
        <div className="text-sm text-slate-500">{t("card.unclassified")}</div>
        <div className="mt-2 text-3xl font-semibold">{summary.unclassified_count}</div>
      </div>
      <div className="panel p-4 fade-up [animation-delay:220ms]">
        <div className="text-sm text-slate-500">Labels</div>
        <div className="mt-2 flex flex-wrap gap-2">
          {Object.entries(summary.by_classification).length === 0 ? (
            <span className="label-pill">none</span>
          ) : (
            Object.entries(summary.by_classification).map(([k, v]) => (
              <span key={k} className="label-pill">
                {k}: {v}
              </span>
            ))
          )}
        </div>
      </div>
    </section>
  );
}
