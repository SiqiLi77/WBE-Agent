"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";

import { createJob } from "@/lib/api";


type JobPayload = {
  job_type: "prepare_data" | "pipeline" | "detection" | "agent" | "evaluation";
  args?: Record<string, unknown>;
};

const QUICK_ACTIONS: Array<{ label: string; payload: JobPayload }> = [
  { label: "Prepare Data", payload: { job_type: "prepare_data", args: { all: true } } },
  { label: "Run Pipeline", payload: { job_type: "pipeline", args: { step: "all" } } },
  { label: "Run Detection", payload: { job_type: "detection", args: { min_votes: 2 } } },
  { label: "Run Agent", payload: { job_type: "agent", args: { max_events: 20 } } },
];


export function JobControls() {
  const client = useQueryClient();
  const mutation = useMutation({
    mutationFn: (payload: JobPayload) => createJob(payload),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ["jobs"] }),
        client.invalidateQueries({ queryKey: ["events"] }),
        client.invalidateQueries({ queryKey: ["summary"] }),
      ]);
    },
  });

  return (
    <section className="panel p-5 fade-up [animation-delay:220ms]">
      <div className="mb-3 text-lg font-semibold">Quick Tasks</div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {QUICK_ACTIONS.map((action) => (
          <button
            key={action.label}
            type="button"
            disabled={mutation.isPending}
            onClick={() => mutation.mutate(action.payload)}
            className="rounded-xl border border-teal-200 bg-teal-50 px-3 py-2 text-sm font-semibold text-teal-900 transition hover:bg-teal-100 disabled:opacity-60"
          >
            {action.label}
          </button>
        ))}
      </div>
      {mutation.isError ? (
        <p className="mt-3 text-sm text-red-700">
          Failed to submit task: {(mutation.error as Error).message}
        </p>
      ) : null}
    </section>
  );
}

