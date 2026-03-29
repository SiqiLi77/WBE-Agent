"""
异常事件聚合与目录生成。

将逐日的异常标记聚合为"事件"（连续异常天数的合并），
生成结构化的事件目录供 Agent 调查。
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings


@dataclass
class AnomalyEvent:
    """单个异常事件。"""

    event_id: str
    site_id: str
    start_date: date
    end_date: date
    peak_date: date
    duration_days: int
    peak_zscore: float
    mean_zscore: float
    detection_methods: list[str]
    vote_count_max: int

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "site_id": self.site_id,
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "peak_date": str(self.peak_date),
            "duration_days": self.duration_days,
            "peak_zscore": self.peak_zscore,
            "mean_zscore": self.mean_zscore,
            "detection_methods": ",".join(self.detection_methods),
            "vote_count_max": self.vote_count_max,
        }


def aggregate_events(
    detection_df: pd.DataFrame,
    site_col: str = "site_id",
    date_col: str = "date",
    anomaly_col: str = "is_anomaly",
    zscore_col: str = "rolling_zscore",
    merge_gap_days: int | None = None,
) -> list[AnomalyEvent]:
    """
    将逐日异常标记聚合为事件。

    规则：间隔 ≤ merge_gap_days 天的连续异常合并为同一事件。

    Parameters
    ----------
    detection_df : Ensemble 检测结果 DataFrame
    merge_gap_days : 合并间隔天数，默认从配置读取

    Returns
    -------
    list[AnomalyEvent] : 异常事件列表
    """
    if merge_gap_days is None:
        merge_gap_days = settings.anomaly_detection.event_merge_gap_days

    all_events: list[AnomalyEvent] = []
    event_counter = 0

    # 获取检测器列名（join 后 bool 可能变为 object，需要宽松匹配）
    _exclude = {site_col, date_col, anomaly_col, "vote_count",
                zscore_col, "pct_change_7d", "site_zscore"}
    detector_names = [
        c for c in detection_df.columns
        if c not in _exclude
        and (detection_df[c].dtype == bool
             or set(detection_df[c].dropna().unique()).issubset({True, False, 0, 1}))
    ]
    # 进一步过滤：只保留已知检测器名称模式
    _known_detectors = {"rolling_zscore", "stl_residual", "weekly_pct_change",
                        "pelt_changepoint", "cusum_trend", "iqr"}
    if not detector_names:
        detector_names = [c for c in detection_df.columns if c in _known_detectors]

    for site_id, site_df in detection_df.groupby(site_col):
        site_df = site_df.sort_values(date_col).copy()
        anomaly_dates = site_df[site_df[anomaly_col] == True][date_col].tolist()

        if not anomaly_dates:
            continue

        # 合并相邻异常日为事件
        events_raw: list[list] = [[anomaly_dates[0]]]
        for dt in anomaly_dates[1:]:
            gap = (dt - events_raw[-1][-1]).days
            if gap <= merge_gap_days:
                events_raw[-1].append(dt)
            else:
                events_raw.append([dt])

        # 构建 AnomalyEvent 对象
        for event_dates in events_raw:
            event_counter += 1
            start = min(event_dates)
            end = max(event_dates)

            # 获取事件窗口内的数据
            mask = (site_df[date_col] >= start) & (site_df[date_col] <= end)
            window = site_df[mask]

            # Z-score 统计
            if zscore_col in window.columns:
                zscores = window[zscore_col].dropna()
                peak_idx = zscores.abs().idxmax() if len(zscores) > 0 else None
                peak_zscore = zscores.abs().max() if len(zscores) > 0 else 0.0
                mean_zscore = zscores.abs().mean() if len(zscores) > 0 else 0.0
                peak_date = window.loc[peak_idx, date_col] if peak_idx is not None else start
            else:
                peak_zscore = 0.0
                mean_zscore = 0.0
                peak_date = start

            # 哪些检测器触发了
            methods = []
            for det_name in detector_names:
                if det_name in window.columns and window[det_name].any():
                    methods.append(det_name)

            vote_max = int(window["vote_count"].max()) if "vote_count" in window.columns else 0

            event = AnomalyEvent(
                event_id=f"EVT-{event_counter:05d}",
                site_id=str(site_id),
                start_date=start.date() if hasattr(start, "date") else start,
                end_date=end.date() if hasattr(end, "date") else end,
                peak_date=peak_date.date() if hasattr(peak_date, "date") else peak_date,
                duration_days=(end - start).days + 1,
                peak_zscore=float(peak_zscore),
                mean_zscore=float(mean_zscore),
                detection_methods=methods,
                vote_count_max=vote_max,
            )
            all_events.append(event)

    logger.info(
        f"事件聚合完成: {len(all_events)} 个事件, "
        f"来自 {len(set(e.site_id for e in all_events))} 个站点"
    )
    return all_events


def events_to_dataframe(events: list[AnomalyEvent]) -> pd.DataFrame:
    """将事件列表转为 DataFrame。"""
    return pd.DataFrame([e.to_dict() for e in events])


def generate_event_catalog(
    detection_df: pd.DataFrame,
    output_path: str | None = None,
    zscore_col: str = "rolling_zscore",
) -> pd.DataFrame:
    """
    生成异常事件目录并保存。

    Parameters
    ----------
    detection_df : Ensemble 检测结果
    output_path : 保存路径，默认 outputs/anomaly_event_catalog.csv
    zscore_col : z-score 列名

    Returns
    -------
    pd.DataFrame : 事件目录
    """
    events = aggregate_events(detection_df, zscore_col=zscore_col)
    catalog = events_to_dataframe(events)

    if output_path:
        catalog.to_csv(output_path, index=False)
        logger.info(f"事件目录保存至: {output_path}")

    # 统计摘要
    if len(catalog) > 0:
        logger.info(f"\n=== 事件目录摘要 ===")
        logger.info(f"总事件数: {len(catalog)}")
        logger.info(f"站点数: {catalog['site_id'].nunique()}")
        logger.info(f"平均持续天数: {catalog['duration_days'].mean():.1f}")
        logger.info(f"平均 peak z-score: {catalog['peak_zscore'].mean():.2f}")
        logger.info(f"每站点事件数:\n{catalog.groupby('site_id').size().describe()}")

    return catalog
