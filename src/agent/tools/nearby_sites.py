"""周边站点查询工具。"""

from typing import Any

import numpy as np
import pandas as pd

from src.agent.tools.base import BaseTool
from src.config import settings
from src.utils.geo import haversine_km


class NearbySitesTool(BaseTool):
    name = "query_nearby_sites"
    description = (
        "查询目标站点周边的其他 NWSS 站点在指定时间范围内的信号趋势，"
        "用于判断异常是局部事件还是区域性趋势。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "site_id": {"type": "string", "description": "目标 NWSS 站点 ID"},
            "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
            "radius_km": {"type": "number", "description": "搜索半径(km)，默认100"},
        },
        "required": ["site_id", "start_date", "end_date"],
    }

    def __init__(self, database: pd.DataFrame | None = None, site_metadata: pd.DataFrame | None = None):
        super().__init__(database)
        self.site_meta = site_metadata  # 需要包含经纬度

    def execute(
        self,
        site_id: str,
        start_date: str,
        end_date: str,
        radius_km: float | None = None,
    ) -> dict[str, Any]:
        if self.db is None or self.site_meta is None:
            return {"error": "数据库或站点元数据未加载"}

        if radius_km is None:
            radius_km = settings.agent.nearby_sites_radius_km

        # 获取目标站点坐标
        target = self.site_meta[self.site_meta["site_id"] == site_id]
        if target.empty:
            return {"error": f"站点 {site_id} 元数据不存在"}

        target_lat = target.iloc[0]["latitude"]
        target_lon = target.iloc[0]["longitude"]

        # 找周边站点
        nearby = []
        for _, meta in self.site_meta.iterrows():
            if meta["site_id"] == site_id:
                continue
            dist = haversine_km(target_lat, target_lon, meta["latitude"], meta["longitude"])
            if dist <= radius_km:
                nearby.append({"site_id": meta["site_id"], "distance_km": dist})

        if not nearby:
            return {
                "site_id": site_id,
                "radius_km": radius_km,
                "nearby_sites": [],
                "summary": f"半径 {radius_km}km 内无其他 NWSS 站点",
            }

        # 查询每个周边站点的信号趋势
        nearby_results = []
        for nb in nearby:
            nb_id = nb["site_id"]
            mask = (
                (self.db["site_id"] == nb_id)
                & (self.db["date"] >= start_date)
                & (self.db["date"] <= end_date)
            )
            nb_data = self.db[mask]

            if nb_data.empty or "pcr_conc_lin_log1p" not in nb_data.columns:
                continue

            series = nb_data["pcr_conc_lin_log1p"].dropna()
            if len(series) < 2:
                continue

            # 趋势判断
            first_half = series.iloc[: len(series) // 2].mean()
            second_half = series.iloc[len(series) // 2 :].mean()
            if first_half > 0:
                change = (second_half - first_half) / first_half
            else:
                change = 0

            if change > 0.2:
                trend = "increasing"
            elif change < -0.2:
                trend = "decreasing"
            else:
                trend = "stable"

            # 检查是否也有异常
            has_anomaly = False
            if "is_anomaly" in nb_data.columns:
                has_anomaly = bool(nb_data["is_anomaly"].any())

            nearby_results.append({
                "site_id": nb_id,
                "distance_km": round(nb["distance_km"], 1),
                "concentration_trend": trend,
                "trend_change_pct": round(change * 100, 1),
                "has_concurrent_anomaly": has_anomaly,
                "n_observations": len(series),
            })

        # 汇总
        n_increasing = sum(1 for r in nearby_results if r["concentration_trend"] == "increasing")
        n_with_anomaly = sum(1 for r in nearby_results if r["has_concurrent_anomaly"])

        return {
            "site_id": site_id,
            "radius_km": radius_km,
            "n_nearby_sites": len(nearby_results),
            "nearby_sites": nearby_results,
            "summary": (
                f"半径 {radius_km}km 内有 {len(nearby_results)} 个站点。"
                f"其中 {n_increasing} 个呈上升趋势，"
                f"{n_with_anomaly} 个同期也有异常。"
            ),
            "is_regional": n_increasing >= len(nearby_results) * 0.5 if nearby_results else False,
        }
