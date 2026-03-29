"use client";

import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { StatsResponse } from "@/lib/types";

const CLASS_COLORS: Record<string, string> = {
  epidemic: "#ef4444",
  environmental: "#06b6d4",
  sampling: "#f59e0b",
  mixed: "#10b981",
  uncertain: "#94a3b8",
  pending: "#cbd5e1",
};

export function ClassificationChart({ data }: { data: Record<string, number> }) {
  const items = Object.entries(data).map(([name, value]) => ({ name, value }));
  if (items.length === 0) return <p className="text-sm text-slate-500">No data</p>;

  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie
          data={items}
          cx="50%"
          cy="50%"
          innerRadius={50}
          outerRadius={80}
          paddingAngle={3}
          dataKey="value"
          label={({ name, percent }) =>
            `${name} ${((percent ?? 0) * 100).toFixed(0)}%`
          }
          labelLine={false}
          fontSize={11}
        >
          {items.map((entry) => (
            <Cell
              key={entry.name}
              fill={CLASS_COLORS[entry.name] ?? "#94a3b8"}
            />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    </ResponsiveContainer>
  );
}


export function TimelineChart({ data }: { data: Record<string, number> }) {
  const items = Object.entries(data).map(([month, count]) => ({ month, count }));
  if (items.length === 0) return <p className="text-sm text-slate-500">No data</p>;

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={items} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="month" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
        <Tooltip />
        <Bar dataKey="count" fill="#0d9488" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DetectionMethodChart({ data }: { data: Record<string, number> }) {
  const items = Object.entries(data).map(([method, count]) => ({
    method: method.replace(/_/g, " "),
    count,
  }));
  if (items.length === 0) return <p className="text-sm text-slate-500">No data</p>;

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={items} layout="vertical" margin={{ top: 5, right: 10, left: 60, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" tick={{ fontSize: 10 }} allowDecimals={false} />
        <YAxis type="category" dataKey="method" tick={{ fontSize: 10 }} width={100} />
        <Tooltip />
        <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DurationHistogram({ data }: { data: number[] }) {
  const labels = ["1d", "2d", "3d", "4d", "5d", "6-10d", "11-20d", "21+d"];
  const items = labels.map((label, i) => ({ label, count: data[i] ?? 0 }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={items} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="label" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
        <Tooltip />
        <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ZscoreHistogram({ data }: { data: number[] }) {
  const labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"];
  const items = labels.map((label, i) => ({ label, count: data[i] ?? 0 }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={items} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="label" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
        <Tooltip />
        <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
