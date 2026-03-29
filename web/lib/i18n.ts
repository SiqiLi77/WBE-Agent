// ── i18n dictionary ─────────────────────────────────────
// Lightweight i18n: no external deps, just a typed dictionary.

export type Locale = "en" | "zh";

const dict = {
  // ── Nav ──
  "nav.dashboard": { en: "Dashboard", zh: "仪表盘" },
  "nav.sites": { en: "Sites", zh: "监测站点" },
  "nav.pipeline": { en: "Pipeline", zh: "系统流程" },
  "nav.experiments": { en: "Experiments", zh: "实验结果" },

  // ── Dashboard ──
  "dash.subtitle": { en: "Wastewater Intelligence", zh: "污水监测智能平台" },
  "dash.title": { en: "Dashboard", zh: "仪表盘" },
  "dash.desc": {
    en: "Monitor anomaly events, run workflows, and review agent investigations.",
    zh: "监控异常事件、运行工作流、查看 Agent 调查结果。",
  },
  "dash.api_error": {
    en: "API request failed. Ensure backend is running at",
    zh: "API 请求失败，请确认后端服务运行在",
  },
  "dash.classification_dist": { en: "Classification Distribution", zh: "分类分布" },
  "dash.events_by_month": { en: "Events by Month", zh: "按月事件数" },
  "dash.detection_methods": { en: "Detection Methods", zh: "检测方法" },
  "dash.event_duration": { en: "Event Duration Distribution", zh: "事件持续时间分布" },

  // ── Summary cards ──
  "card.total_events": { en: "Total Events", zh: "总事件数" },
  "card.classified": { en: "Classified", zh: "已分类" },
  "card.unclassified": { en: "Unclassified", zh: "未分类" },

  // ── Events table ──
  "events.title": { en: "Anomaly Events", zh: "异常事件" },
  "events.filter_class": { en: "Filter by class", zh: "按分类筛选" },
  "events.filter_site": { en: "Filter by site", zh: "按站点筛选" },
  "events.all": { en: "All", zh: "全部" },
  "events.event_id": { en: "Event ID", zh: "事件 ID" },
  "events.site": { en: "Site", zh: "站点" },
  "events.peak_date": { en: "Peak Date", zh: "峰值日期" },
  "events.duration": { en: "Duration", zh: "持续天数" },
  "events.zscore": { en: "Z-Score", zh: "Z 分数" },
  "events.classification": { en: "Classification", zh: "分类" },
  "events.confidence": { en: "Confidence", zh: "置信度" },
  "events.no_data": { en: "No events found.", zh: "未找到事件。" },

  // ── Sites ──
  "sites.title": { en: "Monitoring Sites", zh: "监测站点" },
  "sites.desc": {
    en: "All wastewater monitoring sites with event counts.",
    zh: "所有污水监测站点及其事件统计。",
  },
  "sites.state": { en: "State", zh: "州" },
  "sites.date_range": { en: "Date Range", zh: "日期范围" },
  "sites.observations": { en: "Observations", zh: "观测数" },
  "sites.events": { en: "Events", zh: "事件数" },

  // ── Event detail ──
  "event.back": { en: "← Back to Dashboard", zh: "← 返回仪表盘" },
  "event.detail_title": { en: "Event Detail", zh: "事件详情" },
  "event.summary": { en: "Agent Summary", zh: "Agent 调查摘要" },
  "event.trace": { en: "Investigation Trace", zh: "调查轨迹" },
  "event.concentration": { en: "Wastewater Concentration (log1p)", zh: "污水浓度 (log1p)" },
  "event.environment": { en: "Environmental Context", zh: "环境背景" },
  "event.hospitalization": { en: "Hospitalization (7-day avg)", zh: "住院数据 (7日均值)" },

  // ── Site detail ──
  "site.back": { en: "← Back to Sites", zh: "← 返回站点列表" },
  "site.timeseries": { en: "Time Series", zh: "时间序列" },
  "site.events_at": { en: "Events at this site", zh: "该站点的事件" },

  // ── Pipeline page ──
  "pipe.subtitle": { en: "System Architecture", zh: "系统架构" },
  "pipe.title": { en: "WBE-Agent Pipeline", zh: "WBE-Agent 系统流程" },
  "pipe.desc": {
    en: "End-to-end workflow from raw data ingestion to automated anomaly investigation.",
    zh: "从原始数据采集到自动化异常调查的端到端工作流。",
  },
  "pipe.data_sources": { en: "Data Sources", zh: "数据来源" },
  "pipe.data_pipeline": { en: "Data Pipeline", zh: "数据管线" },
  "pipe.anomaly_detection": { en: "Anomaly Detection", zh: "异常检测" },
  "pipe.agent_investigation": { en: "Agent Investigation", zh: "Agent 调查" },
  "pipe.classification": { en: "Classification Output", zh: "分类输出" },
  "pipe.tools": { en: "Agent Tools", zh: "Agent 工具集" },
  "pipe.react_loop": { en: "ReAct Loop", zh: "ReAct 推理循环" },
  "pipe.voting": { en: "Ensemble Voting", zh: "集成投票" },
  "pipe.step": { en: "Step", zh: "步骤" },

  // ── Experiments page ──
  "exp.subtitle": { en: "Evaluation & Analysis", zh: "评估与分析" },
  "exp.title": { en: "Experiment Results", zh: "实验结果" },
  "exp.desc": {
    en: "Benchmark comparison, ablation analysis, and labeling quality assessment.",
    zh: "基准对比、消融实验和标注质量评估。",
  },
  "exp.benchmark": { en: "Benchmark Comparison", zh: "基准方法对比" },
  "exp.benchmark_desc": {
    en: "8 methods evaluated on 152 anomaly events with silver labels.",
    zh: "8 种方法在 152 个异常事件（银标签）上的评估结果。",
  },
  "exp.ablation": { en: "Ablation Study", zh: "消融实验" },
  "exp.ablation_desc": {
    en: "Impact of removing individual tools/knowledge from the agent.",
    zh: "逐个移除 Agent 工具/知识后的性能影响。",
  },
  "exp.labels": { en: "Label Quality", zh: "标注质量" },
  "exp.labels_desc": {
    en: "3-judge LLM consensus labeling statistics.",
    zh: "3 位 LLM 评审的共识标注统计。",
  },
  "exp.method": { en: "Method", zh: "方法" },
  "exp.accuracy": { en: "Accuracy", zh: "准确率" },
  "exp.macro_f1": { en: "Macro F1", zh: "宏 F1" },
  "exp.weighted_f1": { en: "Weighted F1", zh: "加权 F1" },
  "exp.kappa": { en: "Cohen's κ", zh: "Cohen's κ" },
  "exp.tokens": { en: "Avg Tokens", zh: "平均 Token" },
  "exp.tool_calls": { en: "Avg Tool Calls", zh: "平均工具调用" },
  "exp.full_agent": { en: "Full Agent (baseline)", zh: "完整 Agent（基线）" },
  "exp.delta_f1": { en: "ΔF1 vs Full Agent", zh: "ΔF1 vs 完整 Agent" },
  "exp.consensus": { en: "Consensus Status", zh: "共识状态" },
  "exp.n_events": { en: "Total Events", zh: "总事件数" },
  "exp.avg_confidence": { en: "Avg Confidence", zh: "平均置信度" },
  "exp.label_dist": { en: "Label Distribution", zh: "标签分布" },

  // ── Classification names ──
  "class.sampling": { en: "Sampling", zh: "采样异常" },
  "class.epidemic": { en: "Epidemic", zh: "疫情信号" },
  "class.environmental": { en: "Environmental", zh: "环境干扰" },
  "class.mixed": { en: "Mixed", zh: "混合因素" },
  "class.uncertain": { en: "Uncertain", zh: "不确定" },
  "class.pending": { en: "Pending", zh: "待分类" },

  // ── Method names ──
  "method.majority_class": { en: "Majority Class", zh: "多数类" },
  "method.rule_based": { en: "Rule-based", zh: "规则方法" },
  "method.logistic_regression": { en: "Logistic Regression", zh: "逻辑回归" },
  "method.random_forest": { en: "Random Forest", zh: "随机森林" },
  "method.gradient_boosting": { en: "Gradient Boosting", zh: "梯度提升" },
  "method.zero_shot_llm": { en: "Zero-shot LLM", zh: "零样本 LLM" },
  "method.few_shot_llm": { en: "Few-shot LLM", zh: "少样本 LLM" },
  "method.agent": { en: "WBE-Agent", zh: "WBE-Agent" },

  // ── Ablation names ──
  "abl.no_tools": { en: "No Tools", zh: "无工具" },
  "abl.no_variants": { en: "– Variants", zh: "– 变异株" },
  "abl.no_clinical": { en: "– Clinical", zh: "– 临床数据" },
  "abl.no_nearby_sites": { en: "– Nearby Sites", zh: "– 周边站点" },
  "abl.no_domain_knowledge": { en: "– Domain Knowledge", zh: "– 领域知识" },
  "abl.no_weather": { en: "– Weather", zh: "– 气象数据" },
  "abl.no_hydrology": { en: "– Hydrology", zh: "– 水文数据" },

  // ── Jobs ──
  "jobs.title": { en: "Job Queue", zh: "任务队列" },
  "jobs.run": { en: "Run Workflow", zh: "运行工作流" },
} as const;

export type DictKey = keyof typeof dict;

export function t(key: DictKey, locale: Locale): string {
  const entry = dict[key];
  return entry?.[locale] ?? key;
}

export default dict;
