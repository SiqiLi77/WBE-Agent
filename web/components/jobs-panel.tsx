"use client";

import type { JobRecord } from "@/lib/types";


function statusClass(status: JobRecord["status"]): string {
  if (status === "running") return "bg-sky-50 text-sky-800 border-sky-200";
  if (status === "succeeded") return "bg-emerald-50 text-emerald-800 border-emerald-200";
  if (status === "failed") return "bg-red-50 text-red-800 border-red-200";
  return "bg-slate-50 text-slate-700 border-slate-200";
}

export function JobsPanel({ jobs }: { jobs: JobRecord[] }) {
  return (
    <section className="panel p-5 fade-up [animation-delay:300ms]">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Job Queue</h2>
        <span className="mono text-xs text-slate-500">polling every 5s</span>
      </div>
      {jobs.length === 0 ? (
        <p className="text-sm text-slate-500">No jobs yet.</p>
      ) : (
        <div className="space-y-4">
          {jobs.slice(0, 5).map((job) => (
            <div key={job.id} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className={`label-pill border ${statusClass(job.status)}`}>{job.status}</span>
                <span className="mono text-xs text-slate-600">{job.job_type}</span>
                <span className="mono text-xs text-slate-500">{job.id}</span>
              </div>
              <div className="mono mb-2 overflow-auto rounded bg-slate-900 p-2 text-[11px] text-slate-100">
                {job.command.join(" ")}
              </div>
              <div className="mono max-h-32 overflow-auto rounded border border-slate-200 bg-white p-2 text-[11px] text-slate-700">
                {job.log_tail.length === 0 ? "No logs yet." : job.log_tail.slice(-14).join("\n")}
              </div>
              {job.error ? (
                <div className="mt-2 text-sm text-red-700">{job.error}</div>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

