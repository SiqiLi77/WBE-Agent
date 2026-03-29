"""
自动预标注逻辑。

基于规则对异常事件生成初始标签，供专家审核。
这些标签不是 ground truth，只是加速标注流程的起点。
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings


def auto_label_event(
    event: pd.Series,
    merged_db: pd.DataFrame,
) -> dict:
    """
    对单个异常事件生成自动预标签。

    规则（按优先级）：
    1. 孤立单点 spike (duration=1, 前后正常) → sampling
    2. 异常前 48h 降雨 >25mm 且 14天内住院无变化 → environmental
    3. 14天内住院数上升 >20% → epidemic
    4. 降雨 >25mm 且住院也上升 → mixed
    5. 其他 → uncertain

    Returns
    -------
    dict : {label, confidence, reasoning}
    """
    site_id = event["site_id"]
    peak_date = pd.Timestamp(event["peak_date"])
    duration = event.get("duration_days", 1)

    dk = settings.agent.domain_knowledge
    lag_days = settings.agent.clinical_lag_days

    # 获取事件周围的数据
    window_start = peak_date - pd.Timedelta(days=3)
    window_end = peak_date + pd.Timedelta(days=lag_days)

    site_data = merged_db[
        (merged_db["site_id"] == site_id)
        & (merged_db["date"] >= window_start)
        & (merged_db["date"] <= window_end)
    ].sort_values("date")

    if site_data.empty:
        return {"label": "uncertain", "confidence": 0.0, "reasoning": "无可用数据"}

    # ── 规则 1: 孤立单点 spike ──
    if duration == 1:
        before = site_data[site_data["date"] < peak_date]["pcr_conc_lin_log1p"].dropna()
        after = site_data[site_data["date"] > peak_date]["pcr_conc_lin_log1p"].dropna()
        peak_val = site_data[site_data["date"] == peak_date]["pcr_conc_lin_log1p"].dropna()

        if len(before) > 0 and len(after) > 0 and len(peak_val) > 0:
            context_mean = pd.concat([before.tail(3), after.head(3)]).mean()
            context_std = pd.concat([before.tail(3), after.head(3)]).std()
            if context_std > 0 and abs(peak_val.iloc[0] - context_mean) > 3 * context_std:
                return {
                    "label": "sampling",
                    "confidence": 0.7,
                    "reasoning": "孤立单点 spike，前后数据正常",
                }

    # ── 规则 2 & 4: 检查降雨 ──
    has_rain = False
    rain_before = site_data[
        (site_data["date"] >= peak_date - pd.Timedelta(days=2))
        & (site_data["date"] <= peak_date)
    ]
    if "precipitation_mm" in rain_before.columns:
        total_rain = rain_before["precipitation_mm"].sum()
        has_rain = total_rain > dk.rainfall_significant_mm

    # ── 规则 3 & 4: 检查临床跟随 ──
    has_clinical_rise = False
    target_col = "previous_day_admission_adult_covid_confirmed"
    if target_col in site_data.columns:
        pre_clinical = site_data[
            (site_data["date"] >= peak_date - pd.Timedelta(days=7))
            & (site_data["date"] <= peak_date)
        ][target_col].dropna()

        post_clinical = site_data[
            (site_data["date"] > peak_date)
            & (site_data["date"] <= peak_date + pd.Timedelta(days=lag_days))
        ][target_col].dropna()

        if len(pre_clinical) > 0 and len(post_clinical) > 0:
            pre_mean = pre_clinical.mean()
            post_mean = post_clinical.mean()
            if pre_mean > 0:
                change_pct = (post_mean - pre_mean) / pre_mean * 100
                has_clinical_rise = change_pct > dk.clinical_followup_threshold_pct

    # ── 综合判断 ──
    if has_rain and has_clinical_rise:
        return {
            "label": "mixed",
            "confidence": 0.5,
            "reasoning": f"降雨 {total_rain:.0f}mm + 临床数据上升",
        }
    elif has_rain and not has_clinical_rise:
        return {
            "label": "environmental",
            "confidence": 0.6,
            "reasoning": f"降雨 {total_rain:.0f}mm，临床数据无对应变化",
        }
    elif has_clinical_rise and not has_rain:
        return {
            "label": "epidemic",
            "confidence": 0.6,
            "reasoning": "临床数据在 14 天内跟随上升，无显著降雨",
        }
    else:
        return {
            "label": "uncertain",
            "confidence": 0.3,
            "reasoning": "无明确的降雨事件或临床跟随",
        }


def auto_label_batch(
    events_df: pd.DataFrame,
    merged_db: pd.DataFrame,
) -> pd.DataFrame:
    """
    批量自动预标注。

    Returns
    -------
    pd.DataFrame : events_df 附加 auto_label, auto_confidence, auto_reasoning 列
    """
    labels = []
    for _, event in events_df.iterrows():
        result = auto_label_event(event, merged_db)
        labels.append(result)

    labels_df = pd.DataFrame(labels)
    result = pd.concat([events_df.reset_index(drop=True), labels_df], axis=1)
    result = result.rename(columns={
        "label": "auto_label",
        "confidence": "auto_confidence",
        "reasoning": "auto_reasoning",
    })

    # 统计
    dist = result["auto_label"].value_counts()
    logger.info(f"自动预标注分布:\n{dist}")

    return result
