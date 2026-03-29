"""气象数据查询工具。"""

from typing import Any

import numpy as np
import pandas as pd

from src.agent.tools.base import BaseTool
from src.config import settings


class WeatherTool(BaseTool):
    name = "query_weather"
    description = (
        "查询指定站点在指定时间范围内的气象数据，包括日降水量(mm)、"
        "最高/最低/平均气温(°C)，以及显著气象事件标记。"
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

        if data.empty:
            return {"error": f"站点 {site_id} 在 {start_date}~{end_date} 无气象数据"}

        # 提取气象列
        result: dict[str, Any] = {
            "site_id": site_id,
            "period": f"{start_date} to {end_date}",
            "n_days": len(data),
        }

        dk = settings.agent.domain_knowledge

        for col, key in [
            ("precipitation_mm", "daily_precipitation_mm"),
            ("temp_max_c", "daily_temp_max_c"),
            ("temp_min_c", "daily_temp_min_c"),
            ("temp_avg_c", "daily_temp_avg_c"),
        ]:
            if col in data.columns:
                series = data[col].dropna()
                result[key] = {
                    "values": series.tolist(),
                    "dates": data.loc[series.index, "date"].astype(str).tolist(),
                    "mean": round(float(series.mean()), 2),
                    "max": round(float(series.max()), 2),
                    "sum": round(float(series.sum()), 2) if "precip" in col else None,
                }

        # 标记显著气象事件
        events = []
        if "precipitation_mm" in data.columns:
            heavy_rain = data[data["precipitation_mm"] > dk.rainfall_dilution_threshold_mm]
            for _, row in heavy_rain.iterrows():
                severity = "significant" if row["precipitation_mm"] > dk.rainfall_significant_mm else "moderate"
                events.append({
                    "date": str(row["date"]),
                    "event_type": "heavy_rain",
                    "severity": severity,
                    "value": f"{row['precipitation_mm']:.1f}mm",
                })

        if "temp_avg_c" in data.columns:
            hot_days = data[data["temp_avg_c"] > dk.temp_high_degradation_c]
            for _, row in hot_days.iterrows():
                events.append({
                    "date": str(row["date"]),
                    "event_type": "high_temperature",
                    "severity": "moderate",
                    "value": f"{row['temp_avg_c']:.1f}°C",
                })

        result["significant_events"] = events
        result["summary"] = (
            f"期间内 {len(events)} 个显著气象事件。"
            f"总降水 {result.get('daily_precipitation_mm', {}).get('sum', 'N/A')}mm。"
        )

        return result
