"use client";

import { useQuery } from "@tanstack/react-query";
import { useLocale } from "@/lib/i18n-context";
import { fetchEvaluation, fetchAblation, fetchLabelStats } from "@/lib/api";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, Legend,
} from "recharts";
import type { BenchmarkRow, AblationRow } from "@/lib/types";

const METHOD_COLORS: Record<string, string> = {
  majority_class: "#BFBFBF",
  rule_based: "#A0A0A0",
  logistic_regression: "#7BA7CC",
  random_forest: "#4878A8",
  gradient_boosting: "#3A6088",
  zero_shot_llm: "#D4A84D",
  few_shot_llm: "#C89838",
  agent: "#C85050",
};

const CLASS_COLORS: Record<string, string> = {
  sampling: "#f59e0b",
  mixed: "#10b981",
  environmental: "#06b6d4",
  epidemic: "#ef4444",
  uncertain: "#94a3b8",
};

const CONSENSUS_COLORS: Record<string, string> = {
  majority: "#4878A8",
  accepted: "#5BA05B",
  disagreement: "#E8A838",
  low_confidence: "#C85050",
};

export default function ExperimentsPage() {
  const { locale, t } = useLocale();
  const zh = locale === "zh";

  const evalQ = useQuery({ queryKey: ["evaluation"], queryFn: fetchEvaluation });
  const ablQ = useQuery({ queryKey: ["ablation"], queryFn: fetchAblation });
  const labelQ = useQuery({ queryKey: ["labels"], queryFn: fetchLabelStats });

  const hasError = evalQ.isError || ablQ.isError || labelQ.isError;

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-8 md:px-8">
      <header className="fade-up">
        <div className="mono mb-2 text-xs uppercase tracking-[0.18em] text-teal-700">
          {t("exp.subtitle")}
        </div>
        <h1 className="text-3xl font-semibold md:text-4xl">{t("exp.title")}</h1>
        <p className="mt-2 text-sm text-slate-600 max-w-2xl">{t("exp.desc")}</p>
      </header>

      {hasError && (
        <div className="panel border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {t("dash.api_error")}{" "}
          <span className="mono">{process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"}</span>
        </div>
      )}

      {/* ── Benchmark Section ── */}
      {evalQ.data && (
        <section className="fade-up space-y-4">
          <div>
            <h2 className="text-xl font-semibold text-slate-800">{t("exp.benchmark")}</h2>
            <p className="text-sm text-slate-500">{t("exp.benchmark_desc")}</p>
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="panel p-4">
              <div className="mb-2 text-sm font-semibold text-slate-700">
                Macro F1 {zh ? "对比" : "Comparison"}
              </div>
              <BenchmarkBarChart rows={evalQ.data.rows} zh={zh} />
            </div>
            <div className="panel p-4">
              <div className="mb-2 text-sm font-semibold text-slate-700">
                {zh ? "多指标对比" : "Multi-metric Comparison"}
              </div>
              <MultiMetricTable rows={evalQ.data.rows} zh={zh} />
            </div>
          </div>
        </section>
      )}

      {/* ── Ablation Section ── */}
      {ablQ.data && (
        <section className="fade-up space-y-4">
          <div>
            <h2 className="text-xl font-semibold text-slate-800">{t("exp.ablation")}</h2>
            <p className="text-sm text-slate-500">{t("exp.ablation_desc")}</p>
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="panel p-4">
              <div className="mb-2 text-sm font-semibold text-slate-700">
                {zh ? "消融瀑布图 (ΔMacro F1)" : "Ablation Waterfall (ΔMacro F1)"}
              </div>
              <AblationWaterfall rows={ablQ.data.rows} fullF1={ablQ.data.full_agent_f1} zh={zh} />
            </div>
            <div className="panel p-4">
              <div className="mb-2 text-sm font-semibold text-slate-700">
                {zh ? "Token 与工具调用消耗" : "Token & Tool Call Cost"}
              </div>
              <AblationCostChart rows={ablQ.data.rows} zh={zh} />
            </div>
          </div>
          <div className="panel p-4">
            <div className="mb-2 text-sm font-semibold text-slate-700">
              {zh ? "消融实验详细数据" : "Ablation Detailed Data"}
            </div>
            <AblationTable rows={ablQ.data.rows} fullF1={ablQ.data.full_agent_f1} zh={zh} />
          </div>
        </section>
      )}

      {/* ── Label Quality Section ── */}
      {labelQ.data && (
        <section className="fade-up space-y-4">
          <div>
            <h2 className="text-xl font-semibold text-slate-800">{t("exp.labels")}</h2>
            <p className="text-sm text-slate-500">{t("exp.labels_desc")}</p>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="panel p-5 text-center">
              <div className="text-3xl font-bold text-teal-700">{labelQ.data.n_events}</div>
              <div className="text-xs text-slate-500 mt-1">{t("exp.n_events")}</div>
            </div>
            <div className="panel p-5 text-center">
              <div className="text-3xl font-bold text-indigo-600">
                {(labelQ.data.avg_confidence * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-slate-500 mt-1">{t("exp.avg_confidence")}</div>
            </div>
            <div className="panel p-5 text-center">
              <div className="text-3xl font-bold text-amber-600">3</div>
              <div className="text-xs text-slate-500 mt-1">LLM Judges</div>
            </div>
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="panel p-4">
              <div className="mb-2 text-sm font-semibold text-slate-700">{t("exp.label_dist")}</div>
              <LabelDistChart data={labelQ.data.label_distribution} />
            </div>
            <div className="panel p-4">
              <div className="mb-2 text-sm font-semibold text-slate-700">{t("exp.consensus")}</div>
              <ConsensusChart data={labelQ.data.consensus_status} zh={zh} />
            </div>
          </div>
        </section>
      )}
    </main>
  );
}

/* ── Benchmark bar chart (horizontal lollipop style) ── */
function BenchmarkBarChart({ rows, zh }: { rows: BenchmarkRow[]; zh: boolean }) {
  const METHOD_NAMES: Record<string, string> = zh
    ? {
        majority_class: "多数类", rule_based: "规则方法", logistic_regression: "逻辑回归",
        random_forest: "随机森林", gradient_boosting: "梯度提升",
        zero_shot_llm: "零样本 LLM", few_shot_llm: "少样本 LLM", agent: "WBE-Agent",
      }
    : {
        majority_class: "Majority Class", rule_based: "Rule-based", logistic_regression: "Logistic Reg.",
        random_forest: "Random Forest", gradient_boosting: "Gradient Boost.",
        zero_shot_llm: "Zero-shot LLM", few_shot_llm: "Few-shot LLM", agent: "WBE-Agent",
      };

  const sorted = [...rows].sort((a, b) => a.macro_f1 - b.macro_f1);
  const data = sorted.map((r) => ({
    name: METHOD_NAMES[r.method] ?? r.method,
    macro_f1: +(r.macro_f1 * 100).toFixed(1),
    fill: METHOD_COLORS[r.method] ?? "#94a3b8",
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" tick={{ fontSize: 10 }} domain={[0, 50]} unit="%" />
        <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={90} />
        <Tooltip formatter={(v) => `${v}%`} contentStyle={{ fontSize: 11, borderRadius: 8 }} />
        <Bar dataKey="macro_f1" radius={[0, 4, 4, 0]} barSize={16}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ── Multi-metric table ── */
function MultiMetricTable({ rows, zh }: { rows: BenchmarkRow[]; zh: boolean }) {
  const METHOD_NAMES: Record<string, string> = zh
    ? {
        majority_class: "多数类", rule_based: "规则方法", logistic_regression: "逻辑回归",
        random_forest: "随机森林", gradient_boosting: "梯度提升",
        zero_shot_llm: "零样本 LLM", few_shot_llm: "少样本 LLM", agent: "WBE-Agent",
      }
    : {
        majority_class: "Majority Class", rule_based: "Rule-based", logistic_regression: "Logistic Reg.",
        random_forest: "Random Forest", gradient_boosting: "Gradient Boost.",
        zero_shot_llm: "Zero-shot LLM", few_shot_llm: "Few-shot LLM", agent: "WBE-Agent",
      };

  const order = [
    "random_forest", "gradient_boosting", "agent", "logistic_regression",
    "rule_based", "few_shot_llm", "zero_shot_llm", "majority_class",
  ];
  const sorted = order
    .map((m) => rows.find((r) => r.method === m))
    .filter((r): r is BenchmarkRow => !!r);

  const best = {
    accuracy: Math.max(...rows.map((r) => r.accuracy)),
    macro_f1: Math.max(...rows.map((r) => r.macro_f1)),
    weighted_f1: Math.max(...rows.map((r) => r.weighted_f1)),
    cohen_kappa: Math.max(...rows.map((r) => r.cohen_kappa)),
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-slate-200 text-left text-slate-500">
            <th className="py-2 pr-3">{zh ? "方法" : "Method"}</th>
            <th className="py-2 px-2 text-right">Acc</th>
            <th className="py-2 px-2 text-right">M-F1</th>
            <th className="py-2 px-2 text-right">W-F1</th>
            <th className="py-2 px-2 text-right">κ</th>
            <th className="py-2 px-2 text-right">Tokens</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.method} className={`border-b border-slate-100 ${r.method === "agent" ? "bg-red-50/50" : ""}`}>
              <td className="py-1.5 pr-3 font-medium text-slate-700">
                <span className="inline-block w-2 h-2 rounded-full mr-1.5" style={{ backgroundColor: METHOD_COLORS[r.method] }} />
                {METHOD_NAMES[r.method] ?? r.method}
              </td>
              <td className={`py-1.5 px-2 text-right ${r.accuracy === best.accuracy ? "font-bold text-teal-700" : ""}`}>
                {(r.accuracy * 100).toFixed(1)}
              </td>
              <td className={`py-1.5 px-2 text-right ${r.macro_f1 === best.macro_f1 ? "font-bold text-teal-700" : ""}`}>
                {(r.macro_f1 * 100).toFixed(1)}
              </td>
              <td className={`py-1.5 px-2 text-right ${r.weighted_f1 === best.weighted_f1 ? "font-bold text-teal-700" : ""}`}>
                {(r.weighted_f1 * 100).toFixed(1)}
              </td>
              <td className={`py-1.5 px-2 text-right ${r.cohen_kappa === best.cohen_kappa ? "font-bold text-teal-700" : ""}`}>
                {r.cohen_kappa.toFixed(3)}
              </td>
              <td className="py-1.5 px-2 text-right text-slate-400">
                {r.avg_tokens ? `${(r.avg_tokens / 1000).toFixed(1)}K` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── Ablation waterfall chart ── */
function AblationWaterfall({ rows, fullF1, zh }: { rows: AblationRow[]; fullF1: number; zh: boolean }) {
  const ABL_NAMES: Record<string, string> = zh
    ? {
        no_tools: "无工具", no_variants: "– 变异株", no_clinical: "– 临床",
        no_nearby_sites: "– 周边站点", no_domain_knowledge: "– 领域知识",
        no_weather: "– 气象", no_hydrology: "– 水文",
      }
    : {
        no_tools: "No Tools", no_variants: "– Variants", no_clinical: "– Clinical",
        no_nearby_sites: "– Nearby", no_domain_knowledge: "– Domain Know.",
        no_weather: "– Weather", no_hydrology: "– Hydrology",
      };

  const sorted = [...rows].sort((a, b) => (a.macro_f1 - fullF1) - (b.macro_f1 - fullF1));
  const data = sorted.map((r) => ({
    name: ABL_NAMES[r.ablation] ?? r.ablation,
    delta: +((r.macro_f1 - fullF1) * 100).toFixed(1),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" tick={{ fontSize: 10 }} unit="%" />
        <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={90} />
        <Tooltip formatter={(v) => `${Number(v) > 0 ? "+" : ""}${v}%`} contentStyle={{ fontSize: 11, borderRadius: 8 }} />
        <Bar dataKey="delta" radius={[0, 4, 4, 0]} barSize={14}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.delta < -5 ? "#C85050" : d.delta < -2 ? "#E8A838" : "#5BA05B"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ── Ablation cost chart ── */
function AblationCostChart({ rows, zh }: { rows: AblationRow[]; zh: boolean }) {
  const ABL_NAMES: Record<string, string> = zh
    ? {
        no_tools: "无工具", no_variants: "– 变异株", no_clinical: "– 临床",
        no_nearby_sites: "– 周边站点", no_domain_knowledge: "– 领域知识",
        no_weather: "– 气象", no_hydrology: "– 水文",
      }
    : {
        no_tools: "No Tools", no_variants: "– Variants", no_clinical: "– Clinical",
        no_nearby_sites: "– Nearby", no_domain_knowledge: "– Domain Know.",
        no_weather: "– Weather", no_hydrology: "– Hydrology",
      };

  const data = rows.map((r) => ({
    name: ABL_NAMES[r.ablation] ?? r.ablation,
    tokens: +(r.avg_tokens / 1000).toFixed(1),
    tools: +r.avg_tool_calls.toFixed(1),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="name" tick={{ fontSize: 8 }} angle={-30} textAnchor="end" height={60} />
        <YAxis yAxisId="left" tick={{ fontSize: 10 }} />
        <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} />
        <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
        <Legend wrapperStyle={{ fontSize: 10 }} />
        <Bar yAxisId="left" dataKey="tokens" fill="#6366f1" name={zh ? "Token (K)" : "Tokens (K)"} radius={[4, 4, 0, 0]} barSize={14} />
        <Bar yAxisId="right" dataKey="tools" fill="#f59e0b" name={zh ? "工具调用" : "Tool Calls"} radius={[4, 4, 0, 0]} barSize={14} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ── Ablation detail table ── */
function AblationTable({ rows, fullF1, zh }: { rows: AblationRow[]; fullF1: number; zh: boolean }) {
  const ABL_NAMES: Record<string, string> = zh
    ? {
        no_tools: "无工具", no_variants: "– 变异株", no_clinical: "– 临床",
        no_nearby_sites: "– 周边站点", no_domain_knowledge: "– 领域知识",
        no_weather: "– 气象", no_hydrology: "– 水文",
      }
    : {
        no_tools: "No Tools", no_variants: "– Variants", no_clinical: "– Clinical",
        no_nearby_sites: "– Nearby Sites", no_domain_knowledge: "– Domain Knowledge",
        no_weather: "– Weather", no_hydrology: "– Hydrology",
      };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-slate-200 text-left text-slate-500">
            <th className="py-2 pr-3">{zh ? "消融条件" : "Ablation"}</th>
            <th className="py-2 px-2 text-right">Acc</th>
            <th className="py-2 px-2 text-right">M-F1</th>
            <th className="py-2 px-2 text-right">ΔF1</th>
            <th className="py-2 px-2 text-right">κ</th>
            <th className="py-2 px-2 text-right">{zh ? "工具调用" : "Tools"}</th>
            <th className="py-2 px-2 text-right">Tokens</th>
          </tr>
        </thead>
        <tbody>
          {/* Full agent baseline row */}
          <tr className="border-b border-slate-200 bg-teal-50/50">
            <td className="py-1.5 pr-3 font-semibold text-teal-700">
              {zh ? "完整 Agent（基线）" : "Full Agent (baseline)"}
            </td>
            <td className="py-1.5 px-2 text-right font-semibold text-teal-700">61.8</td>
            <td className="py-1.5 px-2 text-right font-semibold text-teal-700">
              {(fullF1 * 100).toFixed(1)}
            </td>
            <td className="py-1.5 px-2 text-right text-slate-400">—</td>
            <td className="py-1.5 px-2 text-right font-semibold text-teal-700">0.306</td>
            <td className="py-1.5 px-2 text-right">5.9</td>
            <td className="py-1.5 px-2 text-right">18.9K</td>
          </tr>
          {rows.map((r) => {
            const delta = (r.macro_f1 - fullF1) * 100;
            return (
              <tr key={r.ablation} className="border-b border-slate-100">
                <td className="py-1.5 pr-3 font-medium text-slate-700">
                  {ABL_NAMES[r.ablation] ?? r.ablation}
                </td>
                <td className="py-1.5 px-2 text-right">{(r.accuracy * 100).toFixed(1)}</td>
                <td className="py-1.5 px-2 text-right">{(r.macro_f1 * 100).toFixed(1)}</td>
                <td className={`py-1.5 px-2 text-right font-medium ${delta < -3 ? "text-red-600" : delta < 0 ? "text-amber-600" : "text-green-600"}`}>
                  {delta > 0 ? "+" : ""}{delta.toFixed(1)}
                </td>
                <td className="py-1.5 px-2 text-right">{r.cohen_kappa.toFixed(3)}</td>
                <td className="py-1.5 px-2 text-right">{r.avg_tool_calls.toFixed(1)}</td>
                <td className="py-1.5 px-2 text-right">{(r.avg_tokens / 1000).toFixed(1)}K</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── Label distribution pie chart ── */
function LabelDistChart({ data }: { data: Record<string, number> }) {
  const items = Object.entries(data).map(([name, value]) => ({ name, value }));
  if (items.length === 0) return <p className="text-sm text-slate-500">No data</p>;

  return (
    <ResponsiveContainer width="100%" height={240}>
      <PieChart>
        <Pie
          data={items}
          cx="50%"
          cy="50%"
          innerRadius={45}
          outerRadius={80}
          paddingAngle={3}
          dataKey="value"
          label={({ name, value }) => `${name} (${value})`}
          labelLine={false}
          fontSize={10}
        >
          {items.map((entry) => (
            <Cell key={entry.name} fill={CLASS_COLORS[entry.name] ?? "#94a3b8"} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    </ResponsiveContainer>
  );
}

/* ── Consensus status chart ── */
function ConsensusChart({ data, zh }: { data: Record<string, number>; zh: boolean }) {
  const NAMES: Record<string, string> = zh
    ? { majority: "多数一致", accepted: "接受", disagreement: "分歧", low_confidence: "低置信" }
    : { majority: "Majority", accepted: "Accepted", disagreement: "Disagreement", low_confidence: "Low Conf." };

  const items = Object.entries(data).map(([key, value]) => ({
    name: NAMES[key] ?? key,
    value,
    fill: CONSENSUS_COLORS[key] ?? "#94a3b8",
  }));

  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={items} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="name" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
        <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} />
        <Bar dataKey="value" radius={[4, 4, 0, 0]} barSize={32}>
          {items.map((d, i) => (
            <Cell key={i} fill={d.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
