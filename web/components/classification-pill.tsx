type Props = {
  value: string | null | undefined;
};

const COLOR_MAP: Record<string, string> = {
  epidemic: "bg-red-50 text-red-800 border-red-200",
  environmental: "bg-cyan-50 text-cyan-800 border-cyan-200",
  sampling: "bg-amber-50 text-amber-800 border-amber-200",
  mixed: "bg-emerald-50 text-emerald-800 border-emerald-200",
  uncertain: "bg-slate-50 text-slate-700 border-slate-200",
};

export function ClassificationPill({ value }: Props) {
  const label = value ?? "pending";
  const classes = COLOR_MAP[label] ?? "bg-slate-50 text-slate-700 border-slate-200";
  return (
    <span className={`label-pill border ${classes}`}>
      {label}
    </span>
  );
}

