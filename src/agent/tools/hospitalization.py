"""临床住院数据查询工具。"""

from typing import Any

import numpy as np
import pandas as pd

from src.agent.tools.base import BaseTool
from src.config import settings


class HospitalizationTool(BaseTool):
    name = "query_hospitalization"
    description = (
        "查询指定州在指定时间范围内的 COVID-19 住院数据，包括每日新增入院数、"
        "7天滚动均值、趋势方向，以及趋势转折点。"
        "注意：数据为州级粒度，同一州内不同站点共享同一住院数据。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "state": {"type": "string", "description": "州缩写，如 NY, CA"},
            "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
        },
        "required": ["state", "start_date", "end_date"],
    }

    def execute(self, state: str, start_date: str, end_date: str) -> dict[str, Any]:
        if self.db is None:
            return {"error": "数据库未加载"}

        target_col = "previous_day_admission_adult_covid_confirmed"

        mask = (
            (self.db["state"] == state.upper())
            & (self.db["date"] >= start_date)
            & (self.db["date"] <= end_date)
        )
        data = self.db[mask].drop_duplicates(subset=["state", "date"]).copy()

        if data.empty or target_col not in data.columns:
            return {"error": f"州 {state} 在 {start_date}~{end_date} 无住院数据"}

        admissions = data[target_col].dropna()
        avg_7d = data["admission_7d_avg"].dropna() if "admission_7d_avg" in data.columns else pd.Series()

        result: dict[str, Any] = {
            "state": state.upper(),
            "period": f"{start_date} to {end_date}",
            "data_granularity": "state-level (注意：非站点级)",
            "daily_admissions": {
                "values": admissions.tolist(),
                "dates": data.loc[admissions.index, "date"].astype(str).tolist(),
                "mean": round(float(admissions.mean()), 1),
                "max": round(float(admissions.max()), 1),
                "min": round(float(admissions.min()), 1),
            },
        }

        if len(avg_7d) > 0:
            result["7day_avg_admissions"] = {
                "values": avg_7d.tolist(),
                "start": round(float(avg_7d.iloc[0]), 1) if len(avg_7d) > 0 else None,
                "end": round(float(avg_7d.iloc[-1]), 1) if len(avg_7d) > 0 else None,
            }

        # 趋势判断
        if len(avg_7d) >= 7:
            first_week = avg_7d.iloc[:7].mean()
            last_week = avg_7d.iloc[-7:].mean()
            if first_week > 0:
                change_pct = (last_week - first_week) / first_week * 100
            else:
                change_pct = 0

            threshold = settings.agent.domain_knowledge.clinical_followup_threshold_pct
            if change_pct > threshold:
                trend = "increasing"
            elif change_pct < -threshold:
                trend = "decreasing"
            else:
                trend = "stable"

            result["admission_trend"] = trend
            result["trend_change_pct"] = round(change_pct, 1)

        # 检测趋势转折点（7天均值的二阶差分变号）
        if len(avg_7d) >= 14:
            diff1 = avg_7d.diff()
            sign_changes = diff1.apply(np.sign).diff().abs() > 1
            change_dates = data.loc[sign_changes[sign_changes].index, "date"]
            result["trend_change_dates"] = change_dates.astype(str).tolist()

        return result
