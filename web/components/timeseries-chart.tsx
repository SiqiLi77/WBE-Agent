"use client";

import {
  ComposedChart,
  Line,
  Bar,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
  Legend,
} from "recharts";
import type { TimeSeriesPoint } from "@/lib/types";

type Props = {
  points: TimeSeriesPoint[];
  eventStart?: string;
  eventEnd?: string;
};

export function ConcentrationChart({ points, eventStart, eventEnd }: Props) {
  if (points.length === 0) return <p className="text-sm text-slate-500">No data</p>;

  return (
    <div>
      <div className="mb-2 text-xs font-semibold text-slate-500 uppercase tracking-wide">
        Wastewater Concentration (log1p)
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={points} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 9 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 10 }} />
          {eventStart && eventEnd && (
            <ReferenceArea
              x1={eventStart}
              x2={eventEnd}
              fill="#ef4444"
              fillOpacity={0.1}
              stroke="#ef4444"
              strokeOpacity={0.3}
            />
          )}
          <Tooltip
            contentStyle={{ fontSize: 11, borderRadius: 8 }}
            labelStyle={{ fontWeight: 600 }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line
            type="monotone"
            dataKey="concentration"
            stroke="#0d9488"
            strokeWidth={1.5}
            dot={false}
            name="Concentration"
            connectNulls
          />
          <Line
            type="monotone"
            dataKey="rolling_median_7d"
            stroke="#6366f1"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            dot={false}
            name="7d Median"
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}


export function EnvironmentChart({ points }: { points: TimeSeriesPoint[] }) {
  if (points.length === 0) return null;

  return (
    <div>
      <div className="mb-2 text-xs font-semibold text-slate-500 uppercase tracking-wide">
        Environmental Context
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={points} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 9 }} interval="preserveStartEnd" />
          <YAxis yAxisId="left" tick={{ fontSize: 10 }} />
          <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} />
          <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar
            yAxisId="left"
            dataKey="precipitation_mm"
            fill="#60a5fa"
            opacity={0.6}
            name="Precip (mm)"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="temp_avg_c"
            stroke="#f59e0b"
            strokeWidth={1.5}
            dot={false}
            name="Temp (°C)"
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function HospitalizationChart({ points }: { points: TimeSeriesPoint[] }) {
  const hasData = points.some((p) => p.admission_7d_avg !== null);
  if (!hasData) return null;

  return (
    <div>
      <div className="mb-2 text-xs font-semibold text-slate-500 uppercase tracking-wide">
        Hospitalization (7-day avg admissions)
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <ComposedChart data={points} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="date" tick={{ fontSize: 9 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 10 }} />
          <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
          <Area
            type="monotone"
            dataKey="admission_7d_avg"
            fill="#fca5a5"
            fillOpacity={0.3}
            stroke="#ef4444"
            strokeWidth={1.5}
            name="Admissions 7d avg"
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
