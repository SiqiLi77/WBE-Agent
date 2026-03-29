"use client";

import { useState } from "react";
import { useLocale } from "@/lib/i18n-context";

/* ── Color tokens ── */
const C = {
  teal: "#0d9488",
  blue: "#3b82f6",
  indigo: "#6366f1",
  amber: "#f59e0b",
  red: "#ef4444",
  green: "#22c55e",
  slate: "#64748b",
  purple: "#8b5cf6",
  cyan: "#06b6d4",
};

/* ── Animated connector arrow (CSS-only pulse) ── */
function Arrow({ vertical = false }: { vertical?: boolean }) {
  return vertical ? (
    <div className="flex justify-center py-1">
      <div className="h-8 w-0.5 animate-pulse bg-gradient-to-b from-teal-400 to-teal-600 rounded-full" />
    </div>
  ) : (
    <div className="flex items-center px-1">
      <div className="h-0.5 w-8 animate-pulse bg-gradient-to-r from-teal-400 to-teal-600 rounded-full" />
      <div className="h-0 w-0 border-y-4 border-y-transparent border-l-4 border-l-teal-600" />
    </div>
  );
}

/* ── Expandable card ── */
function PipeCard({
  color,
  icon,
  title,
  subtitle,
  children,
  defaultOpen = false,
}: {
  color: string;
  icon: string;
  title: string;
  subtitle: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div
      className="group rounded-xl border border-slate-200 bg-white shadow-sm transition-all hover:shadow-md"
      style={{ borderLeftColor: color, borderLeftWidth: 4 }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-3 px-5 py-4 text-left"
      >
        <span className="text-2xl">{icon}</span>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold text-slate-800">{title}</div>
          <div className="text-xs text-slate-500 truncate">{subtitle}</div>
        </div>
        <span
          className={`text-slate-400 transition-transform ${open ? "rotate-180" : ""}`}
        >
          ▾
        </span>
      </button>
      {open && (
        <div className="border-t border-slate-100 px-5 py-4 text-sm text-slate-600 animate-in fade-in slide-in-from-top-2 duration-200">
          {children}
        </div>
      )}
    </div>
  );
}

/* ── Tool badge ── */
function ToolBadge({ icon, name, desc }: { icon: string; name: string; desc: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-slate-100 bg-slate-50 p-3 transition hover:bg-white hover:shadow-sm">
      <span className="text-lg">{icon}</span>
      <div>
        <div className="text-xs font-semibold text-slate-700">{name}</div>
        <div className="text-xs text-slate-500">{desc}</div>
      </div>
    </div>
  );
}

/* ── Classification pill ── */
function ClassPill({ label, color, desc }: { label: string; color: string; desc: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-slate-100 bg-white p-2.5">
      <div className="h-3 w-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
      <div>
        <span className="text-xs font-semibold text-slate-700">{label}</span>
        <span className="text-xs text-slate-500 ml-1.5">{desc}</span>
      </div>
    </div>
  );
}

/* ── Animated step number ── */
function StepNum({ n, color }: { n: number; color: string }) {
  return (
    <div
      className="flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold text-white shadow-sm flex-shrink-0"
      style={{ backgroundColor: color }}
    >
      {n}
    </div>
  );
}

/* ── ReAct animation ── */
function ReactLoop() {
  const { locale } = useLocale();
  const steps = locale === "zh"
    ? [
        { icon: "💭", label: "思考 (Thought)", desc: "分析当前证据，决定下一步调查方向", color: C.indigo },
        { icon: "🔧", label: "行动 (Action)", desc: "调用工具获取气象/水文/临床等数据", color: C.blue },
        { icon: "👁", label: "观察 (Observation)", desc: "解读工具返回的数据，提取关键信息", color: C.green },
        { icon: "📋", label: "输出 (Output)", desc: "综合所有证据，给出分类、置信度和推理链", color: C.red },
      ]
    : [
        { icon: "💭", label: "Thought", desc: "Analyze current evidence, decide next investigation step", color: C.indigo },
        { icon: "🔧", label: "Action", desc: "Call tools to fetch weather/hydrology/clinical data", color: C.blue },
        { icon: "👁", label: "Observation", desc: "Interpret returned data, extract key findings", color: C.green },
        { icon: "📋", label: "Output", desc: "Synthesize all evidence into classification + reasoning chain", color: C.red },
      ];

  return (
    <div className="relative flex flex-col items-center gap-2 py-2">
      {steps.map((s, i) => (
        <div key={i} className="flex items-center gap-3 w-full max-w-md">
          <div
            className="flex h-10 w-10 items-center justify-center rounded-full text-lg shadow-sm flex-shrink-0"
            style={{ backgroundColor: s.color + "18", border: `2px solid ${s.color}` }}
          >
            {s.icon}
          </div>
          <div className="flex-1">
            <div className="text-xs font-semibold" style={{ color: s.color }}>{s.label}</div>
            <div className="text-xs text-slate-500">{s.desc}</div>
          </div>
          {i < steps.length - 1 && (
            <div className="absolute left-5 ml-0" style={{ top: `${(i + 1) * 56 - 8}px` }}>
            </div>
          )}
        </div>
      ))}
      {/* Loop arrow */}
      <div className="mt-1 flex items-center gap-2 text-xs text-slate-400">
        <span>↻</span>
        <span>{locale === "zh" ? "重复直到证据充分（最多 10 轮）" : "Repeat until sufficient evidence (max 10 rounds)"}</span>
      </div>
    </div>
  );
}

/* ── Voting visualization ── */
function VotingViz() {
  const { locale } = useLocale();
  const methods = locale === "zh"
    ? [
        { name: "Rolling Z-Score", desc: "7天滑动窗口，阈值 z>2.0", icon: "📈" },
        { name: "STL 残差", desc: "季节分解，残差超过 2σ", icon: "🔄" },
        { name: "周环比变化", desc: "|周变化率| > 100%", icon: "📊" },
        { name: "PELT 变点", desc: "BIC 惩罚项，检测结构性变化", icon: "📐" },
      ]
    : [
        { name: "Rolling Z-Score", desc: "7-day window, threshold z>2.0", icon: "📈" },
        { name: "STL Residual", desc: "Seasonal decomposition, residual > 2σ", icon: "🔄" },
        { name: "Weekly Change", desc: "|week-over-week change| > 100%", icon: "📊" },
        { name: "PELT Changepoint", desc: "BIC penalty, structural change detection", icon: "📐" },
      ];

  return (
    <div>
      <div className="grid grid-cols-2 gap-2 mb-3">
        {methods.map((m, i) => (
          <div key={i} className="flex items-center gap-2 rounded-lg border border-slate-100 bg-slate-50 p-2.5">
            <span className="text-lg">{m.icon}</span>
            <div>
              <div className="text-xs font-semibold text-slate-700">{m.name}</div>
              <div className="text-xs text-slate-500">{m.desc}</div>
            </div>
          </div>
        ))}
      </div>
      <div className="flex items-center justify-center gap-2 rounded-lg bg-amber-50 border border-amber-200 p-3">
        <span className="text-lg">🗳️</span>
        <div className="text-xs">
          <span className="font-semibold text-amber-800">
            {locale === "zh" ? "投票机制" : "Voting Mechanism"}:
          </span>{" "}
          <span className="text-amber-700">
            {locale === "zh"
              ? "≥2 种方法同时检测 → 标记为异常事件（间隔 ≤3 天合并）"
              : "≥2 methods agree → flagged as anomaly event (merge if gap ≤3 days)"}
          </span>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   Main Pipeline Page
   ══════════════════════════════════════════════════════════ */
export default function PipelinePage() {
  const { locale, t } = useLocale();
  const zh = locale === "zh";

  return (
    <main className="mx-auto flex min-h-screen max-w-5xl flex-col gap-6 px-4 py-8 md:px-8">
      {/* Header */}
      <header className="fade-up">
        <div className="mono mb-2 text-xs uppercase tracking-[0.18em] text-teal-700">
          {t("pipe.subtitle")}
        </div>
        <h1 className="text-3xl font-semibold md:text-4xl">{t("pipe.title")}</h1>
        <p className="mt-2 text-sm text-slate-600 max-w-2xl">{t("pipe.desc")}</p>
      </header>

      {/* ── Phase 1: Data Sources ── */}
      <section className="fade-up">
        <SectionHeader
          num={1}
          color={C.blue}
          title={t("pipe.data_sources")}
          sub={zh ? "5 个公开数据源，覆盖 16 个州" : "5 public data sources across 16 states"}
        />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 mt-3">
          <SourceCard icon="🧪" name="CDC NWSS" stat="18,810"
            desc={zh ? "污水 SARS-CoV-2 浓度（站点-日记录）" : "Wastewater SARS-CoV-2 concentration (site-day records)"} />
          <SourceCard icon="🏥" name="HHS Hospitalization" stat="16 states"
            desc={zh ? "州级 COVID-19 住院数据" : "State-level COVID-19 hospital admissions"} />
          <SourceCard icon="🌧️" name="NOAA GHCN-Daily" stat="~1.6 km"
            desc={zh ? "日降水量、气温（平均匹配距离）" : "Daily precipitation & temperature (avg match distance)"} />
          <SourceCard icon="🌊" name="USGS NWIS" stat="~4.1 km"
            desc={zh ? "日流量数据（平均匹配距离）" : "Daily streamflow discharge (avg match distance)"} />
          <SourceCard icon="🧬" name="CDC Variants" stat="10 regions"
            desc={zh ? "HHS Region 级变异株比例" : "HHS Region-level variant proportions"} />
        </div>
      </section>

      <Arrow vertical />

      {/* ── Phase 2: Data Pipeline ── */}
      <section className="fade-up">
        <SectionHeader
          num={2}
          color={C.teal}
          title={t("pipe.data_pipeline")}
          sub={zh ? "7 步标准化处理流程" : "7-step standardized processing workflow"}
        />
        <div className="mt-3 space-y-2">
          {pipelineSteps(zh).map((s, i) => (
            <PipeCard
              key={i}
              color={s.color}
              icon={s.icon}
              title={`${zh ? "步骤" : "Step"} ${i + 1}: ${s.title}`}
              subtitle={s.sub}
            >
              <div className="text-xs leading-relaxed">{s.detail}</div>
              {s.output && (
                <div className="mt-2 rounded bg-slate-50 px-3 py-1.5 text-xs font-mono text-slate-600">
                  {zh ? "输出" : "Output"}: {s.output}
                </div>
              )}
            </PipeCard>
          ))}
        </div>
      </section>

      <Arrow vertical />

      {/* ── Phase 3: Anomaly Detection ── */}
      <section className="fade-up">
        <SectionHeader
          num={3}
          color={C.amber}
          title={t("pipe.anomaly_detection")}
          sub={zh ? "4 种方法集成投票，≥2 票标记异常" : "4-method ensemble voting, ≥2 votes flags anomaly"}
        />
        <div className="mt-3">
          <VotingViz />
          <div className="mt-3 grid grid-cols-3 gap-3 text-center">
            <StatBox label={zh ? "异常事件" : "Events"} value="152" />
            <StatBox label={zh ? "覆盖站点" : "Sites"} value="15" />
            <StatBox label={zh ? "异常率" : "Anomaly Rate"} value="~2-3%" />
          </div>
        </div>
      </section>

      <Arrow vertical />

      {/* ── Phase 4: Agent Investigation ── */}
      <section className="fade-up">
        <SectionHeader
          num={4}
          color={C.indigo}
          title={t("pipe.agent_investigation")}
          sub={zh ? "基于 ReAct 范式的 LLM Agent 自动调查" : "ReAct-based LLM Agent automated investigation"}
        />

        <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Left: ReAct loop */}
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="text-xs font-semibold text-slate-700 mb-3">
              {t("pipe.react_loop")}
            </div>
            <ReactLoop />
          </div>

          {/* Right: Tools */}
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="text-xs font-semibold text-slate-700 mb-3">
              {t("pipe.tools")}
            </div>
            <div className="grid grid-cols-1 gap-2">
              {agentTools(zh).map((tool, i) => (
                <ToolBadge key={i} icon={tool.icon} name={tool.name} desc={tool.desc} />
              ))}
            </div>
          </div>
        </div>

        {/* Agent stats */}
        <div className="mt-3 grid grid-cols-4 gap-3 text-center">
          <StatBox label={zh ? "平均工具调用" : "Avg Tool Calls"} value="~6" />
          <StatBox label={zh ? "平均 Token" : "Avg Tokens"} value="~19K" />
          <StatBox label={zh ? "调查耗时" : "Investigation Time"} value="~30s" />
          <StatBox label={zh ? "每次成本" : "Cost per Event"} value="~$0.03" />
        </div>
      </section>

      <Arrow vertical />

      {/* ── Phase 5: Classification Output ── */}
      <section className="fade-up">
        <SectionHeader
          num={5}
          color={C.red}
          title={t("pipe.classification")}
          sub={zh ? "5 类分类 + 置信度 + 可解释推理链" : "5-class classification + confidence + explainable reasoning chain"}
        />
        <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {classCategories(zh).map((c, i) => (
            <ClassPill key={i} label={c.label} color={c.color} desc={c.desc} />
          ))}
        </div>
        <div className="mt-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
          <div className="text-xs font-semibold text-slate-700 mb-2">
            {zh ? "结构化输出格式" : "Structured Output Format"}
          </div>
          <pre className="text-xs text-slate-600 font-mono leading-relaxed overflow-x-auto">{`{
  "classification": "environmental",
  "confidence": 0.85,
  "primary_factors": [
    { "factor": "heavy_rainfall", "contribution": "high", "evidence": "32mm precip" }
  ],
  "reasoning_chain": ["Checked weather → heavy rain...", "Checked hospitalization → stable..."],
  "recommendation": "Flag as weather-related dilution event",
  "data_gaps": ["No sub-daily flow data available"]
}`}</pre>
        </div>
      </section>

      {/* ── Domain Knowledge ── */}
      <section className="fade-up mt-4">
        <div className="rounded-xl border border-indigo-200 bg-indigo-50 p-5">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-lg">🧠</span>
            <span className="text-sm font-semibold text-indigo-800">
              {zh ? "编码的领域知识" : "Encoded Domain Knowledge"}
            </span>
          </div>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 text-xs text-indigo-700">
            {domainKnowledge(zh).map((dk, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="text-indigo-400 mt-0.5">•</span>
                <span>{dk}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}

/* ── Section header ── */
function SectionHeader({ num, color, title, sub }: { num: number; color: string; title: string; sub: string }) {
  return (
    <div className="flex items-center gap-3">
      <StepNum n={num} color={color} />
      <div>
        <div className="text-lg font-semibold text-slate-800">{title}</div>
        <div className="text-xs text-slate-500">{sub}</div>
      </div>
    </div>
  );
}

/* ── Source card ── */
function SourceCard({ icon, name, stat, desc }: { icon: string; name: string; stat: string; desc: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 transition hover:shadow-md">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xl">{icon}</span>
        <span className="text-sm font-semibold text-slate-800">{name}</span>
      </div>
      <div className="text-xl font-bold text-teal-700 mb-1">{stat}</div>
      <div className="text-xs text-slate-500">{desc}</div>
    </div>
  );
}

/* ── Stat box ── */
function StatBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="text-lg font-bold text-slate-800">{value}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  );
}

/* ── Data: pipeline steps ── */
function pipelineSteps(zh: boolean) {
  return [
    {
      icon: "🧪", color: C.teal,
      title: zh ? "NWSS 污水数据清洗" : "NWSS Wastewater Cleaning",
      sub: zh ? "去重、时间对齐、log1p 变换、滚动统计" : "Dedup, time alignment, log1p transform, rolling stats",
      detail: zh
        ? "从 CDC NWSS 公开数据中筛选 flow-population 归一化站点，进行日期解析、同站同日去重取均值、daily freq reindex、log1p 变换，计算 7/14 天滚动中位数和 z-score。"
        : "Filter flow-population normalized sites from CDC NWSS, parse dates, deduplicate same-site same-day records, reindex to daily frequency, apply log1p transform, compute 7/14-day rolling median and z-score.",
      output: "nwss_cleaned.parquet (18,810 rows)",
    },
    {
      icon: "🏥", color: C.red,
      title: zh ? "HHS 住院数据处理" : "HHS Hospitalization Processing",
      sub: zh ? "州级 COVID-19 住院数据提取" : "State-level COVID-19 admission extraction",
      detail: zh
        ? "提取 previous_day_admission_adult_covid_confirmed 字段，按州和日期聚合，计算 7 天滑动均值和趋势方向。"
        : "Extract confirmed adult COVID admissions, aggregate by state and date, compute 7-day moving average and trend direction.",
      output: "hhs_cleaned.parquet",
    },
    {
      icon: "🌧️", color: C.blue,
      title: zh ? "NOAA 气象数据匹配" : "NOAA Weather Matching",
      sub: zh ? "Haversine 距离匹配最近气象站" : "Haversine distance matching to nearest weather station",
      detail: zh
        ? "从 GHCN inventory 筛选活跃站点（last_year ≥ 2020），按 Haversine 距离匹配（<50km，覆盖率 >80%），提取日降水量 (PRCP) 和气温 (TMAX/TMIN)。"
        : "Filter active GHCN stations (last_year ≥ 2020), match by Haversine distance (<50km, >80% coverage), extract daily precipitation and temperature.",
      output: "noaa_matched.parquet (9/11 sites matched)",
    },
    {
      icon: "🌊", color: C.cyan,
      title: zh ? "USGS 水文数据匹配" : "USGS Hydrology Matching",
      sub: zh ? "按距离匹配最近水文站" : "Distance-based matching to nearest streamflow gauge",
      detail: zh
        ? "按州查询有流量数据（参数 00060）的活跃站，按距离匹配（<30km），提取日流量和历史百分位。"
        : "Query active USGS sites with discharge data (param 00060) by state, match by distance (<30km), extract daily flow and historical percentiles.",
      output: "usgs_matched.parquet (11/11 sites matched)",
    },
    {
      icon: "🧬", color: C.purple,
      title: zh ? "CDC 变异株数据对齐" : "CDC Variant Alignment",
      sub: zh ? "州→HHS Region 映射，周频→日频" : "State→HHS Region mapping, weekly→daily alignment",
      detail: zh
        ? "通过 SODA API 获取变异株监测数据，筛选 share ≥ 5% 的变异株，按 HHS Region 匹配，周频数据填充为日频。"
        : "Fetch variant surveillance via SODA API, filter variants with share ≥ 5%, match by HHS Region, forward-fill weekly data to daily.",
      output: "variants_aligned.parquet",
    },
    {
      icon: "🔗", color: C.green,
      title: zh ? "多源数据合并" : "Multi-source Merge",
      sub: zh ? "以 NWSS 为主表左连接所有数据源" : "Left-join all sources on NWSS as primary table",
      detail: zh
        ? "以 site_id + date 为键，左连接 HHS（state+date）、NOAA（site_id+date）、USGS（site_id+date）、变异株（state+date）。"
        : "Join on site_id + date as key: left-join HHS (state+date), NOAA (site_id+date), USGS (site_id+date), variants (state+date).",
      output: "merged_multisource.parquet (18,810 × 166 cols)",
    },
    {
      icon: "✅", color: C.slate,
      title: zh ? "数据质量检查" : "Data Quality Check",
      sub: zh ? "覆盖率、缺失值、异常值、空间匹配报告" : "Coverage, missing values, outliers, spatial match reports",
      detail: zh
        ? "生成 4 份质量报告：coverage（各数据源覆盖率）、missing（缺失率统计）、outliers（异常值检测）、spatial_match（空间匹配距离验证）。"
        : "Generate 4 quality reports: coverage (source coverage rates), missing (missing value stats), outliers (outlier detection), spatial_match (distance validation).",
      output: "outputs/quality_report/*.csv",
    },
  ];
}

/* ── Data: agent tools ── */
function agentTools(zh: boolean) {
  return [
    { icon: "📍", name: "query_site_metadata", desc: zh ? "站点基本信息（州、经纬度、气候带、历史统计量）" : "Site info (state, coords, climate zone, historical stats)" },
    { icon: "🌧️", name: "query_weather", desc: zh ? "日降水量、气温（匹配的 NOAA 站点）" : "Daily precipitation & temperature (matched NOAA station)" },
    { icon: "🌊", name: "query_hydrology", desc: zh ? "日流量、流量百分位（匹配的 USGS 站点）" : "Daily discharge & flow percentile (matched USGS gauge)" },
    { icon: "🏥", name: "query_hospitalization", desc: zh ? "州级日住院数、7 天均值、趋势方向" : "State-level daily admissions, 7-day avg, trend direction" },
    { icon: "📡", name: "query_nearby_sites", desc: zh ? "100km 内其他站点的浓度趋势和异常同步性" : "Concentration trends & anomaly sync within 100km radius" },
    { icon: "🧬", name: "query_variants", desc: zh ? "HHS Region 级变异株比例、主导株、新兴株" : "Region-level variant proportions, dominant & emerging strains" },
  ];
}

/* ── Data: classification categories ── */
function classCategories(zh: boolean) {
  return [
    { label: zh ? "疫情信号 (Epidemic)" : "Epidemic", color: "#ef4444", desc: zh ? "社区感染率真实变化" : "True community infection change" },
    { label: zh ? "环境干扰 (Environmental)" : "Environmental", color: "#06b6d4", desc: zh ? "降雨稀释、高温降解、CSO" : "Rain dilution, heat degradation, CSO" },
    { label: zh ? "采样异常 (Sampling)" : "Sampling", color: "#f59e0b", desc: zh ? "采样/检测过程问题" : "Sampling/lab process issues" },
    { label: zh ? "混合因素 (Mixed)" : "Mixed", color: "#10b981", desc: zh ? "多因素叠加" : "Multiple factors combined" },
    { label: zh ? "不确定 (Uncertain)" : "Uncertain", color: "#94a3b8", desc: zh ? "证据不足" : "Insufficient evidence" },
  ];
}

/* ── Data: domain knowledge ── */
function domainKnowledge(zh: boolean) {
  return zh ? [
    "稀释效应：日降雨 >10mm 可检测稀释，>25mm 显著稀释",
    "RNA 降解：>25°C 半衰期 <24h，<10°C 半衰期 >72h",
    "信号-临床时滞：4-14 天（中位数 ~7 天）",
    "CSO 风险：流量超过历史 90th percentile",
    "采样异常特征：孤立单点 spike、周末/节假日偏差",
    "所有阈值均有文献支撑",
  ] : [
    "Dilution: >10mm daily rain detectable, >25mm significant",
    "RNA degradation: >25°C half-life <24h, <10°C half-life >72h",
    "Signal-clinical lag: 4-14 days (median ~7 days)",
    "CSO risk: discharge exceeds historical 90th percentile",
    "Sampling anomaly: isolated single-point spike, weekend/holiday bias",
    "All thresholds are literature-supported",
  ];
}
