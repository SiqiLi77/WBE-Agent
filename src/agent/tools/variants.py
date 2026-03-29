"""变异株数据查询工具。"""

from typing import Any

import pandas as pd

from src.agent.tools.base import BaseTool


class VariantsTool(BaseTool):
    name = "query_variants"
    description = (
        "查询指定州（对应 HHS Region）在指定时间范围内的 SARS-CoV-2 变异株数据，"
        "包括主导变异株、占比、新兴变异株及其增长率。"
        "数据粒度为 HHS Region 级，周频。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "state": {"type": "string", "description": "州缩写"},
            "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
            "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
        },
        "required": ["state", "start_date", "end_date"],
    }

    def execute(self, state: str, start_date: str, end_date: str) -> dict[str, Any]:
        if self.db is None:
            return {"error": "数据库未加载"}

        mask = (
            (self.db["state"] == state.upper())
            & (self.db["date"] >= start_date)
            & (self.db["date"] <= end_date)
        )
        data = self.db[mask].copy()

        result: dict[str, Any] = {
            "state": state.upper(),
            "period": f"{start_date} to {end_date}",
            "data_granularity": "HHS Region level, weekly",
        }

        if "dominant_variant" not in data.columns or data["dominant_variant"].isna().all():
            result["error"] = "该时间范围内无变异株数据"
            return result

        # 周级摘要（去重到周）
        weekly = data.drop_duplicates(subset=["state", "dominant_variant", "date"]).sort_values("date")

        # 主导株变化
        dominant_changes = weekly[["date", "dominant_variant", "dominant_proportion"]].dropna()
        if len(dominant_changes) > 0:
            result["dominant_variant_timeline"] = [
                {
                    "date": str(row["date"]),
                    "variant": row["dominant_variant"],
                    "proportion": round(float(row["dominant_proportion"]), 3)
                    if pd.notna(row["dominant_proportion"]) else None,
                }
                for _, row in dominant_changes.iterrows()
            ]

        # 检测变异株替代事件
        shifts = []
        variants_seen = dominant_changes["dominant_variant"].tolist()
        for i in range(1, len(variants_seen)):
            if variants_seen[i] != variants_seen[i - 1]:
                shifts.append({
                    "date": str(dominant_changes.iloc[i]["date"]),
                    "from_variant": variants_seen[i - 1],
                    "to_variant": variants_seen[i],
                })

        result["notable_shifts"] = shifts

        # 新兴株
        if "n_emerging_variants" in data.columns:
            emerging_days = data[data["n_emerging_variants"] > 0]
            if len(emerging_days) > 0:
                result["emerging_variants"] = [
                    {
                        "date": str(row["date"]),
                        "variants": row.get("emerging_variants", ""),
                        "count": int(row["n_emerging_variants"]),
                    }
                    for _, row in emerging_days.drop_duplicates(subset=["date"]).iterrows()
                ]

        result["summary"] = (
            f"期间内有 {len(shifts)} 次主导株替代事件，"
            f"当前主导株: {variants_seen[-1] if variants_seen else 'unknown'}。"
        )

        return result
