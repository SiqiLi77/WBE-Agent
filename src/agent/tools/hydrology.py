"""水文数据查询工具。"""

from typing import Any

import numpy as np
import pandas as pd

from src.agent.tools.base import BaseTool
from src.config import settings


class HydrologyTool(BaseTool):
    name = "query_hydrology"
    description = (
        "查询指定站点在指定时间范围内的水文数据，包括日均流量(cfs)、"
        "流量百分位（相对于历史分布），以及高流量/CSO风险事件标记。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "site_id": {"type": "string", "description": "NWSS 站点 ID"},
            "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
        },
        "required": ["site_id", "start_date", "end_date"],
    }

    def execute(self, site_id: str, start_date: str, end_date: str) -> dict[str, Any]:
        if self.db is None:
            return {"error": "数据库未加载"}

        mask = (
            (self.db["site_id"] == site_id)
            & (self.db["date"] >= start_date)
            & (self.db["date"] <= end_date)
        )
        data = self.db[mask].copy()

        result: dict[str, Any] = {
            "site_id": site_id,
            "period": f"{start_date} to {end_date}",
        }

        if "discharge_cfs" not in data.columns or data["discharge_cfs"].isna().all():
            result["error"] = "该站点无水文数据（可能无匹配的 USGS 站点）"
            result["fallback"] = "请参考降水数据推断稀释效应"
            return result

        discharge = data["discharge_cfs"].dropna()
        percentile = data["discharge_percentile"].dropna() if "discharge_percentile" in data.columns else pd.Series()

        result["daily_discharge_cfs"] = {
            "values": discharge.tolist(),
            "dates": data.loc[discharge.index, "date"].astype(str).tolist(),
            "mean": round(float(discharge.mean()), 1),
            "max": round(float(discharge.max()), 1),
            "min": round(float(discharge.min()), 1),
        }

        if len(percentile) > 0:
            result["discharge_percentile"] = {
                "values": percentile.tolist(),
                "mean": round(float(percentile.mean()), 3),
                "max": round(float(percentile.max()), 3),
            }

        # 标记高流量 / CSO 风险事件
        cso_threshold = settings.agent.domain_knowledge.cso_flow_percentile / 100
        events = []
        if len(percentile) > 0:
            high_flow = data[data["discharge_percentile"] > cso_threshold]
            for _, row in high_flow.iterrows():
                events.append({
                    "date": str(row["date"]),
                    "event_type": "high_flow_cso_risk",
                    "severity": "significant",
                    "value": f"percentile={row['discharge_percentile']:.2f}",
                })

        result["significant_events"] = events
        return result
