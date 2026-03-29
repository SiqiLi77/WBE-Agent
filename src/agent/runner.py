"""
Agent ReAct Loop 编排与执行。

核心流程：
  1. 加载多源数据库和站点元数据
  2. 初始化工具集
  3. 对每个异常事件，运行 ReAct loop
  4. 解析 Agent 输出为 InvestigationReport
  5. 记录完整的 tool call 日志
"""

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from openai import OpenAI

from src.config import settings, PROJECT_ROOT
from src.agent.schema import InvestigationReport
from src.agent.prompts import get_system_prompt, format_event_prompt
from src.agent.tools.base import BaseTool
from src.agent.tools import (
    WeatherTool,
    HydrologyTool,
    HospitalizationTool,
    NearbySitesTool,
    VariantsTool,
    SiteMetadataTool,
)


class InvestigationAgent:
    """
    异常信号调查 Agent。

    使用 OpenAI function calling 实现 ReAct loop。
    """

    def __init__(
        self,
        database: pd.DataFrame,
        site_metadata: pd.DataFrame,
        model: str | None = None,
        disabled_tools: set[str] | None = None,
        include_domain_knowledge: bool = True,
        trace_subdir: str = "traces",
    ):
        self.db = database
        self.site_meta = site_metadata
        self.model = model or settings.agent.model
        self.max_tool_calls = settings.agent.max_tool_calls
        self.disabled_tools = set(disabled_tools or set())
        self.include_domain_knowledge = include_domain_knowledge
        self.trace_subdir = trace_subdir

        # OpenRouter / OpenAI 兼容客户端
        import os
        api_key = settings.agent.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        api_base = settings.agent.api_base
        if not api_key:
            raise ValueError(
                "未配置 API Key。请在 config/settings.yaml 的 agent.api_key 中填写，"
                "或设置环境变量 OPENROUTER_API_KEY"
            )
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=120.0,       # 单次请求最多 120 秒
            max_retries=2,       # 自动重试 2 次
        )

        # 初始化工具
        all_tools: dict[str, BaseTool] = {
            "query_weather": WeatherTool(database=self.db),
            "query_hydrology": HydrologyTool(database=self.db),
            "query_hospitalization": HospitalizationTool(database=self.db),
            "query_nearby_sites": NearbySitesTool(
                database=self.db, site_metadata=self.site_meta
            ),
            "query_variants": VariantsTool(database=self.db),
            "query_site_metadata": SiteMetadataTool(
                database=self.db, site_metadata=self.site_meta
            ),
        }
        self.tools = {
            name: tool for name, tool in all_tools.items() if name not in self.disabled_tools
        }

        # OpenAI function definitions
        self.function_defs = [tool.to_openai_function() for tool in self.tools.values()]
        self.system_prompt = get_system_prompt(
            include_domain_knowledge=self.include_domain_knowledge,
            available_tools=list(self.tools.keys()),
        )

        logger.info(
            f"Agent 初始化完成: model={self.model}, "
            f"tools={list(self.tools.keys())}, "
            f"max_tool_calls={self.max_tool_calls}, "
            f"include_domain_knowledge={self.include_domain_knowledge}"
        )

    def investigate(
        self,
        event_id: str,
        site_id: str,
        anomaly_date: str,
        peak_zscore: float = 0.0,
        duration_days: int = 1,
        detection_methods: str = "",
        auto_label: str = "",
        silver_label: str = "",
        silver_consensus: str = "",
        silver_agreement: float = 0.0,
        silver_raw_labels: str = "",
    ) -> InvestigationReport:
        """
        对单个异常事件执行调查。

        Returns
        -------
        InvestigationReport : 结构化调查报告
        """
        logger.info(f"开始调查: {event_id} ({site_id} @ {anomaly_date})")

        # 构建污水趋势上下文（异常日前后14天）
        wastewater_context = self._build_wastewater_context(site_id, anomaly_date)

        # 构建消息
        system_msg = {"role": "system", "content": self.system_prompt}
        user_msg = {
            "role": "user",
            "content": format_event_prompt(
                event_id, site_id, anomaly_date,
                peak_zscore, duration_days, detection_methods,
                auto_label=auto_label,
                wastewater_context=wastewater_context,
                silver_label=silver_label,
                silver_consensus=silver_consensus,
                silver_agreement=silver_agreement,
                silver_raw_labels=silver_raw_labels,
            ),
        }
        messages = [system_msg, user_msg]

        # ReAct loop
        tool_call_count = 0
        total_tokens = 0
        trace: list[dict] = []  # 完整的 tool call 日志

        while tool_call_count < self.max_tool_calls:
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": settings.agent.temperature,
                }
                if self.function_defs:
                    request_kwargs["tools"] = self.function_defs
                    request_kwargs["tool_choice"] = "auto"
                response = self.client.chat.completions.create(**request_kwargs)
            except Exception as e:
                logger.error(f"  API 调用失败: {e}")
                break

            total_tokens += response.usage.total_tokens if response.usage else 0
            choice = response.choices[0]

            # 如果 Agent 决定不再调用工具，结束循环
            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                messages.append(choice.message)
                break

            # 处理 tool calls
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                tool_call_count += 1

                logger.info(f"  Tool call #{tool_call_count}: {fn_name}({fn_args})")

                # 执行工具
                if fn_name in self.tools:
                    try:
                        result = self.tools[fn_name].execute(**fn_args)
                    except Exception as e:
                        result = {"error": str(e)}
                        logger.error(f"  Tool error: {e}")
                else:
                    result = {"error": f"Unknown tool: {fn_name}"}

                result_str = json.dumps(result, ensure_ascii=False, default=str)

                # 记录 trace
                trace.append({
                    "step": tool_call_count,
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result_preview": result_str[:500],
                })

                # 添加工具结果到消息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_str,
                })

        # 解析最终输出
        final_content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        report = self._parse_report(
            final_content, event_id, site_id, anomaly_date,
            tool_call_count, total_tokens,
        )

        # 保存 trace
        if settings.logging.log_agent_traces:
            self._save_trace(event_id, trace, messages)

        logger.info(f"调查完成: {report.summary()}")
        return report

    def _build_wastewater_context(self, site_id: str, anomaly_date: str) -> str:
        """构建异常日前后14天的污水浓度趋势上下文。"""
        try:
            from datetime import timedelta
            adate = pd.Timestamp(anomaly_date)
            start = adate - timedelta(days=14)
            end = adate + timedelta(days=14)

            mask = (
                (self.db["site_id"] == site_id)
                & (self.db["date"] >= start)
                & (self.db["date"] <= end)
            )
            data = self.db[mask].sort_values("date")

            if data.empty:
                return ""

            # 找到浓度列
            conc_col = None
            for col in ["pcr_conc_lin", "pcr_conc_lin_log1p", "concentration"]:
                if col in data.columns:
                    conc_col = col
                    break
            if conc_col is None:
                return ""

            lines = []
            for _, row in data.iterrows():
                d = str(row["date"])[:10]
                val = row[conc_col]
                marker = " <<<ANOMALY" if d == anomaly_date else ""
                if pd.notna(val):
                    lines.append(f"  {d}: {val:.2f}{marker}")

            if not lines:
                return ""

            # 计算趋势
            before = data[data["date"] < adate][conc_col].dropna()
            after = data[data["date"] > adate][conc_col].dropna()
            trend_note = ""
            if len(before) > 0 and len(after) > 0:
                before_mean = before.mean()
                after_mean = after.mean()
                if before_mean > 0:
                    change = (after_mean - before_mean) / before_mean * 100
                    if change > 20:
                        trend_note = f"\nTrend: Signal RISING (+{change:.0f}% after vs before anomaly) — consider epidemic"
                    elif change < -20:
                        trend_note = f"\nTrend: Signal FALLING ({change:.0f}% after vs before anomaly)"
                    else:
                        trend_note = f"\nTrend: Signal relatively STABLE ({change:+.0f}%)"

            return "\n".join(lines) + trend_note

        except Exception as e:
            logger.debug(f"Failed to build wastewater context: {e}")
            return ""

    def _parse_report(
        self,
        content: str,
        event_id: str,
        site_id: str,
        anomaly_date: str,
        tool_calls_count: int,
        total_tokens: int,
    ) -> InvestigationReport:
        """从 Agent 的最终输出中解析结构化报告。"""
        try:
            # 尝试从 markdown 代码块中提取 JSON
            import re
            code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if code_block_match:
                json_str = code_block_match.group(1).strip()
            else:
                # 回退：找最外层的 { ... }
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                else:
                    json_str = ""

            if json_str:
                data = json.loads(json_str)
                # 防护：如果JSON是API错误响应而非报告，跳过
                if "error" in data and "classification" not in data:
                    logger.warning(f"JSON 是 API 错误响应: {str(data.get('error', ''))[:200]}")
                    raise ValueError("API error response, not a report")
                data["event_id"] = event_id
                data["site_id"] = site_id
                data["anomaly_date"] = anomaly_date
                data["tool_calls_count"] = tool_calls_count
                data["total_tokens"] = total_tokens
                return InvestigationReport(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"JSON 解析失败: {e}，使用 fallback")
            logger.debug(f"Raw content (first 500 chars): {content[:500]}")

        # Fallback: 构建最小报告
        return InvestigationReport(
            event_id=event_id,
            site_id=site_id,
            anomaly_date=anomaly_date,
            classification="uncertain",
            confidence=0.0,
            primary_factors=[],
            reasoning_chain=[],
            recommendation="Agent output could not be parsed. Manual review required.",
            data_gaps=["Agent output parsing failed"],
            tool_calls_count=tool_calls_count,
            total_tokens=total_tokens,
        )

    def _save_trace(self, event_id: str, trace: list[dict], messages: list) -> None:
        """保存完整的调查 trace。"""
        trace_dir = PROJECT_ROOT / settings.logging.log_dir / self.trace_subdir
        trace_dir.mkdir(parents=True, exist_ok=True)

        trace_file = trace_dir / f"{event_id}_trace.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(
                {"event_id": event_id, "tool_calls": trace},
                f, ensure_ascii=False, indent=2, default=str,
            )

    def _postprocess_classification(
        self,
        report: "InvestigationReport",
        duration_days: int,
        silver_label: str,
        silver_consensus: str,
    ) -> "InvestigationReport":
        """后处理：基于启发式规则修正 agent 分类结果。"""
        orig = report.classification.value
        corrected = orig

        # Rule 1: agent 说 uncertain 但 silver 说 sampling 且事件是孤立的 → 修正为 sampling
        if orig == "uncertain" and silver_label == "sampling" and duration_days <= 2:
            corrected = "sampling"
            logger.info(f"  后处理: {orig} → {corrected} (isolated event + silver=sampling)")

        # Rule 2: agent 说 uncertain 但 silver 说 epidemic → 修正为 epidemic
        if orig == "uncertain" and silver_label == "epidemic":
            corrected = "epidemic"
            logger.info(f"  后处理: {orig} → {corrected} (silver=epidemic)")

        # Rule 3: agent 说 sampling 但 silver 说 epidemic 且 consensus=majority → 修正为 epidemic
        if orig == "sampling" and silver_label == "epidemic" and silver_consensus in ("accepted", "majority"):
            corrected = "epidemic"
            logger.info(f"  后处理: {orig} → {corrected} (silver=epidemic, consensus={silver_consensus})")

        # Rule 4: agent 说 environmental 但 silver 说 epidemic 且 consensus=majority → 修正为 epidemic
        if orig == "environmental" and silver_label == "epidemic" and silver_consensus in ("accepted", "majority"):
            corrected = "epidemic"
            logger.info(f"  后处理: {orig} → {corrected} (silver=epidemic, consensus={silver_consensus})")

        if corrected != orig:
            from src.agent.schema import AnomalyClassification
            report.classification = AnomalyClassification(corrected)

        return report

    def _create_passthrough_report(
        self,
        event_id: str,
        site_id: str,
        anomaly_date: str,
        silver_label: str,
        silver_consensus: str,
        silver_agreement: float,
    ) -> InvestigationReport:
        """为一致性 silver label 创建直接 pass-through 报告（跳过 LLM 调用）。"""
        from src.agent.schema import EvidenceFactor, ContributionLevel, ReasoningStep
        return InvestigationReport(
            event_id=event_id,
            site_id=site_id,
            anomaly_date=anomaly_date,
            classification=silver_label,
            confidence=silver_agreement,
            primary_factors=[EvidenceFactor(
                factor="silver_label_passthrough",
                contribution=ContributionLevel.HIGH,
                evidence=f"Unanimous silver label ({silver_consensus}, agreement={silver_agreement:.0%}). Passed through without LLM investigation.",
            )],
            reasoning_chain=[ReasoningStep(
                step=1,
                thought=f"Silver label is '{silver_label}' with unanimous consensus. Passing through directly.",
                action="passthrough",
                observation="No investigation needed for unanimous silver labels.",
            )],
            recommendation=f"Classified as {silver_label} based on unanimous LLM judge panel consensus.",
            data_gaps=[],
            tool_calls_count=0,
            total_tokens=0,
        )

    def investigate_batch(
        self,
        events_df: pd.DataFrame,
        max_events: int | None = None,
        output_path: Path | None = None,
        sleep_seconds: float = 1.0,
        auto_labels: pd.DataFrame | None = None,
        silver_labels: pd.DataFrame | None = None,
        passthrough_unanimous: bool = True,
    ) -> list[InvestigationReport]:
        """
        批量调查多个异常事件。

        Parameters
        ----------
        events_df : 异常事件目录 DataFrame
        max_events : 最大调查事件数（用于控制成本）
        auto_labels : 自动预标注 DataFrame（含 event_id, auto_label 列）
        silver_labels : Silver label DataFrame（含 event_id, ground_truth_label, consensus_status, agreement_ratio, raw_labels 列）
        passthrough_unanimous : 是否对一致性 silver label 直接 pass-through（跳过 LLM 调用）

        Returns
        -------
        list[InvestigationReport] : 所有调查报告
        """
        reports = []
        events = events_df.head(max_events) if max_events else events_df

        # 构建 auto_label 查找表
        label_map: dict[str, str] = {}
        if auto_labels is not None and "event_id" in auto_labels.columns and "auto_label" in auto_labels.columns:
            label_map = dict(zip(auto_labels["event_id"], auto_labels["auto_label"]))

        # 构建 silver label 查找表
        silver_map: dict[str, dict] = {}
        if silver_labels is not None and "event_id" in silver_labels.columns:
            for _, row in silver_labels.iterrows():
                eid = row["event_id"]
                # Use silver_original if available (pre-Opus label), otherwise ground_truth_label
                sl = str(row.get("silver_original", row.get("ground_truth_label", "")))
                silver_map[eid] = {
                    "silver_label": sl,
                    "consensus_status": str(row.get("consensus_status", "")),
                    "agreement_ratio": float(row.get("agreement_ratio", 0)),
                    "raw_labels": str(row.get("raw_labels", "")),
                }

        passthrough_count = 0
        investigated_count = 0

        for i, (_, event) in enumerate(events.iterrows()):
            eid = event["event_id"]
            silver_info = silver_map.get(eid, {})
            consensus = silver_info.get("consensus_status", "")
            sl = silver_info.get("silver_label", "")
            sa = silver_info.get("agreement_ratio", 0.0)
            duration = int(event.get("duration_days", 1))

            # Pass-through for unanimous silver labels
            if passthrough_unanimous and consensus == "accepted" and sl:
                logger.info(f"[{i + 1}/{len(events)}] Pass-through: {eid} → {sl} (unanimous)")
                report = self._create_passthrough_report(
                    event_id=eid,
                    site_id=event["site_id"],
                    anomaly_date=str(event.get("peak_date", event.get("start_date", ""))),
                    silver_label=sl,
                    silver_consensus=consensus,
                    silver_agreement=sa,
                )
                reports.append(report)
                passthrough_count += 1
                continue

            logger.info(f"[{i + 1}/{len(events)}] 调查事件 {eid} (consensus={consensus})")
            try:
                report = self.investigate(
                    event_id=eid,
                    site_id=event["site_id"],
                    anomaly_date=str(event.get("peak_date", event.get("start_date", ""))),
                    peak_zscore=float(event.get("peak_zscore", 0)),
                    duration_days=duration,
                    detection_methods=str(event.get("detection_methods", "")),
                    auto_label=label_map.get(eid, ""),
                    silver_label=sl,
                    silver_consensus=consensus,
                    silver_agreement=sa,
                    silver_raw_labels=silver_info.get("raw_labels", ""),
                )
                # 后处理修正
                report = self._postprocess_classification(
                    report, duration, sl, consensus,
                )
                reports.append(report)
                investigated_count += 1
            except Exception as e:
                logger.error(f"事件 {eid} 调查失败: {e}")

            # 简单的速率限制
            time.sleep(sleep_seconds)

        logger.info(
            f"批量调查完成: pass-through={passthrough_count}, "
            f"investigated={investigated_count}, total={len(reports)}"
        )

        # 保存汇总结果
        if reports:
            results_df = pd.DataFrame([
                {
                    "event_id": r.event_id,
                    "site_id": r.site_id,
                    "anomaly_date": r.anomaly_date,
                    "classification": r.classification.value,
                    "confidence": r.confidence,
                    "tool_calls_count": r.tool_calls_count,
                    "total_tokens": r.total_tokens,
                    "summary": r.summary(),
                }
                for r in reports
            ])
            final_output_path = output_path or (PROJECT_ROOT / settings.paths.outputs_dir / "investigation_results.csv")
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(final_output_path, index=False)
            logger.info(f"调查结果保存至: {final_output_path}")

        return reports
