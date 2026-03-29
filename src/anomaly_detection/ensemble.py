"""
Ensemble 异常检测聚合。

运行多个检测器，按投票策略聚合结果。
"""

import pandas as pd
from loguru import logger

from src.config import settings
from src.anomaly_detection.detectors import BaseDetector, create_detector


class EnsembleDetector:
    """
    Ensemble 异常检测器。

    策略：任意 ≥ min_votes 种方法同时检测到 → 标记为异常。
    """

    def __init__(
        self,
        detectors: list[BaseDetector] | None = None,
        min_votes: int = 2,
    ):
        if detectors is not None:
            self.detectors = detectors
        else:
            self.detectors = self._build_from_config()
        self.min_votes = min_votes

    def _build_from_config(self) -> list[BaseDetector]:
        """从配置文件构建检测器列表。"""
        detectors = []
        for name, method_cfg in settings.anomaly_detection.methods.items():
            if not method_cfg.enabled:
                continue
            kwargs = {}
            if method_cfg.window is not None:
                kwargs["window"] = method_cfg.window
            if method_cfg.period is not None:
                kwargs["period"] = method_cfg.period
            if method_cfg.threshold is not None:
                kwargs["threshold"] = method_cfg.threshold
            if method_cfg.penalty is not None:
                kwargs["penalty"] = method_cfg.penalty
            if method_cfg.drift is not None:
                kwargs["drift"] = method_cfg.drift
            try:
                det = create_detector(name, **kwargs)
                detectors.append(det)
                logger.info(f"  启用检测器: {det}")
            except ValueError as e:
                logger.warning(f"  跳过检测器 {name}: {e}")
        return detectors

    def detect(self, series: pd.Series) -> pd.DataFrame:
        """
        对单站点时间序列运行所有检测器。

        Parameters
        ----------
        series : pd.Series, index=DatetimeIndex

        Returns
        -------
        pd.DataFrame : 列包含每个检测器的结果 + vote_count + is_anomaly
        """
        results = pd.DataFrame(index=series.index)

        for det in self.detectors:
            try:
                result = det.detect(series)
                results[det.name] = result.reindex(series.index, fill_value=False)
            except Exception as e:
                logger.error(f"检测器 {det.name} 失败: {e}")
                results[det.name] = False

        # 投票计数
        detector_cols = [det.name for det in self.detectors if det.name in results.columns]
        results["vote_count"] = results[detector_cols].sum(axis=1)
        results["is_anomaly"] = results["vote_count"] >= self.min_votes

        n_anomaly = results["is_anomaly"].sum()
        n_total = len(results)
        logger.info(
            f"Ensemble 检测完成: {n_anomaly}/{n_total} 天标记为异常 "
            f"({n_anomaly / n_total * 100:.1f}%)"
        )

        return results

    def detect_all_sites(
        self,
        df: pd.DataFrame,
        value_col: str = "pcr_conc_lin_log1p",
        site_col: str = "site_id",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        对所有站点运行 Ensemble 检测。

        Returns
        -------
        pd.DataFrame : 原始 df 附加检测结果列
        """
        all_results = []

        for site_id, site_df in df.groupby(site_col):
            logger.info(f"检测站点: {site_id}")
            site_df = site_df.set_index(date_col).sort_index()

            if value_col not in site_df.columns:
                logger.warning(f"站点 {site_id} 缺少列 {value_col}，跳过")
                continue

            series = site_df[value_col].dropna()
            if len(series) < 14:
                logger.warning(f"站点 {site_id} 数据不足 ({len(series)} 天)，跳过")
                continue

            detection = self.detect(series)
            # 合并回原始数据
            site_result = site_df.join(detection, how="left")
            site_result[site_col] = site_id
            all_results.append(site_result.reset_index())

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()
