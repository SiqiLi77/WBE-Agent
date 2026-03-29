"""站点元数据查询工具。"""

from typing import Any

import numpy as np
import pandas as pd

from src.agent.tools.base import BaseTool


class SiteMetadataTool(BaseTool):
    name = "query_site_metadata"
    description = (
        "查询指定 NWSS 站点的元数据，包括地理位置、服务人口、采样类型、"
        "归一化方法、气候带、匹配的气象站/水文站信息，以及历史浓度统计。"
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "site_id": {"type": "string", "description": "NWSS 站点 ID"},
        },
        "required": ["site_id"],
    }

    def __init__(self, database: pd.DataFrame | None = None, site_metadata: pd.DataFrame | None = None):
        super().__init__(database)
        self.site_meta = site_metadata

    def execute(self, site_id: str) -> dict[str, Any]:
        result: dict[str, Any] = {"site_id": site_id}

        # 从元数据表获取基本信息
        if self.site_meta is not None:
            meta = self.site_meta[self.site_meta["site_id"] == site_id]
            if not meta.empty:
                row = meta.iloc[0]
                for col in meta.columns:
                    val = row[col]
                    if pd.notna(val):
                        result[col] = val if not isinstance(val, (np.integer, np.floating)) else float(val)

        # 从数据库计算历史统计
        if self.db is not None:
            site_data = self.db[self.db["site_id"] == site_id]
            if not site_data.empty and "pcr_conc_lin" in site_data.columns:
                conc = site_data["pcr_conc_lin"].dropna()
                result["historical_stats"] = {
                    "n_observations": len(conc),
                    "date_range": f"{site_data['date'].min()} to {site_data['date'].max()}",
                    "mean": round(float(conc.mean()), 2),
                    "std": round(float(conc.std()), 2),
                    "median": round(float(conc.median()), 2),
                    "p25": round(float(conc.quantile(0.25)), 2),
                    "p75": round(float(conc.quantile(0.75)), 2),
                    "min": round(float(conc.min()), 2),
                    "max": round(float(conc.max()), 2),
                }

        return result
