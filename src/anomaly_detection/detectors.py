"""
异常检测方法实现。

每个检测器接收单站点的时间序列，返回逐日的异常标记。
所有检测器遵循统一接口：
  输入: pd.Series (index=date, values=concentration)
  输出: pd.Series (index=date, values=bool, True=异常)
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger


class BaseDetector(ABC):
    """异常检测器基类。"""

    name: str = "base"

    @abstractmethod
    def detect(self, series: pd.Series) -> pd.Series:
        """
        对时间序列执行异常检测。

        Parameters
        ----------
        series : pd.Series, index=DatetimeIndex, values=float (log1p concentration)

        Returns
        -------
        pd.Series : index=DatetimeIndex, values=bool (True=异常)
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class RollingZScoreDetector(BaseDetector):
    """
    滚动 Z-score 异常检测。

    计算 (x - rolling_median) / rolling_std，超过阈值标记为异常。
    适合检测短期突变（spike / dip）。
    """

    name = "rolling_zscore"

    def __init__(self, window: int = 7, threshold: float = 2.0, min_periods: int = 4):
        self.window = window
        self.threshold = threshold
        self.min_periods = min_periods

    def detect(self, series: pd.Series) -> pd.Series:
        rolling = series.rolling(window=self.window, min_periods=self.min_periods)
        median = rolling.median()
        std = rolling.std()
        zscore = (series - median) / std.replace(0, np.nan)
        return zscore.abs() > self.threshold


class STLResidualDetector(BaseDetector):
    """
    STL 季节分解残差异常检测。

    对时间序列做 STL 分解，残差超过阈值标记为异常。
    适合检测去除季节性后的真实异常。
    """

    name = "stl_residual"

    def __init__(self, period: int = 7, threshold: float = 2.0):
        self.period = period
        self.threshold = threshold

    def detect(self, series: pd.Series) -> pd.Series:
        from statsmodels.tsa.seasonal import STL

        # STL 需要无缺失值，先插值
        filled = series.interpolate(method="linear", limit=3)
        filled = filled.ffill().bfill()

        if len(filled.dropna()) < 2 * self.period:
            logger.warning(f"STL: 数据不足 ({len(filled.dropna())} < {2 * self.period})，跳过")
            return pd.Series(False, index=series.index)

        try:
            stl = STL(filled, period=self.period, robust=True)
            result = stl.fit()
            residual = result.resid
            resid_std = residual.std()
            if resid_std == 0:
                return pd.Series(False, index=series.index)
            return (residual.abs() / resid_std) > self.threshold
        except Exception as e:
            logger.warning(f"STL 分解失败: {e}")
            return pd.Series(False, index=series.index)


class WeeklyChangeDetector(BaseDetector):
    """
    周环比变化率异常检测。

    计算 7 天变化率，超过阈值标记为异常。
    适合检测趋势突变。
    """

    name = "weekly_pct_change"

    def __init__(self, threshold: float = 1.0):
        """threshold=1.0 表示 100% 的周变化率。"""
        self.threshold = threshold

    def detect(self, series: pd.Series) -> pd.Series:
        pct_change = series.pct_change(periods=7)
        return pct_change.abs() > self.threshold


class CUSUMTrendDetector(BaseDetector):
    """
    CUSUM 趋势检测器。

    检测持续性上升趋势（如疫情波次），而非孤立尖峰。
    使用累积和（CUSUM）算法检测均值偏移。
    """

    name = "cusum_trend"

    def __init__(self, drift: float = 0.5, threshold: float = 5.0, window: int = 14):
        self.drift = drift
        self.threshold = threshold
        self.window = window

    def detect(self, series: pd.Series) -> pd.Series:
        filled = series.interpolate(method="linear", limit=3).ffill().bfill()
        if len(filled.dropna()) < self.window:
            return pd.Series(False, index=series.index)

        values = filled.values
        rolling_mean = pd.Series(values, index=filled.index).rolling(
            window=self.window, min_periods=self.window // 2
        ).mean()
        rolling_std = pd.Series(values, index=filled.index).rolling(
            window=self.window, min_periods=self.window // 2
        ).std().replace(0, np.nan)

        # Standardize relative to rolling baseline
        z = ((filled - rolling_mean) / rolling_std).fillna(0).values

        # One-sided upper CUSUM for sustained increases
        cusum_pos = np.zeros(len(z))
        for i in range(1, len(z)):
            cusum_pos[i] = max(0, cusum_pos[i - 1] + z[i] - self.drift)

        anomaly = pd.Series(cusum_pos > self.threshold, index=series.index)

        n_flagged = anomaly.sum()
        if n_flagged > 0:
            logger.debug(f"CUSUM trend: {n_flagged} days flagged as sustained rise")

        return anomaly


class PELTChangepointDetector(BaseDetector):
    """
    PELT 变点检测。

    使用 ruptures 库的 PELT 算法检测均值/方差的结构性变化。
    变点附近的数据点标记为异常。
    """

    name = "pelt_changepoint"

    def __init__(self, penalty: str = "bic", margin_days: int = 3):
        self.penalty = penalty
        self.margin_days = margin_days

    def detect(self, series: pd.Series) -> pd.Series:
        import ruptures as rpt

        # 需要无缺失值
        filled = series.interpolate(method="linear", limit=3).ffill().bfill()
        values = filled.dropna().values

        if len(values) < 10:
            return pd.Series(False, index=series.index)

        try:
            algo = rpt.Pelt(model="rbf", min_size=7).fit(values)

            if self.penalty == "bic":
                pen = np.log(len(values)) * values.var()
            elif self.penalty == "aic":
                pen = 2 * values.var()
            else:
                pen = float(self.penalty) if self.penalty.replace(".", "").isdigit() else 1.0

            changepoints = algo.predict(pen=pen)
            # changepoints 是索引列表（不含最后一个点）
            changepoints = [cp for cp in changepoints if cp < len(values)]

            # 在变点附近标记异常
            anomaly_mask = pd.Series(False, index=series.index)
            valid_dates = filled.dropna().index
            for cp in changepoints:
                if cp >= len(valid_dates):
                    continue
                cp_date = valid_dates[cp - 1] if cp > 0 else valid_dates[0]
                # 标记变点前后 margin_days 天
                start = cp_date - pd.Timedelta(days=self.margin_days)
                end = cp_date + pd.Timedelta(days=self.margin_days)
                anomaly_mask.loc[start:end] = True

            return anomaly_mask

        except Exception as e:
            logger.warning(f"PELT 检测失败: {e}")
            return pd.Series(False, index=series.index)


# ── 检测器注册表 ──
DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "rolling_zscore": RollingZScoreDetector,
    "stl_residual": STLResidualDetector,
    "weekly_pct_change": WeeklyChangeDetector,
    "cusum_trend": CUSUMTrendDetector,
    "pelt_changepoint": PELTChangepointDetector,
}


def create_detector(name: str, **kwargs) -> BaseDetector:
    """根据名称创建检测器实例。"""
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"未知检测器: {name}，可选: {list(DETECTOR_REGISTRY.keys())}")
    return DETECTOR_REGISTRY[name](**kwargs)
