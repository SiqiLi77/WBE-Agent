"""
时间序列对齐与特征工程工具。

将不规则采样的时间序列映射到日历日网格，
并计算滚动统计量、变化率等派生特征。
"""

import numpy as np
import pandas as pd


def align_to_daily_grid(
    df: pd.DataFrame,
    date_col: str = "date",
    group_col: str = "site_id",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    将不规则时间序列 reindex 到每日网格上。

    缺失日填充为 NaN（不做插值）。
    每个 group 独立处理，使用各自的 min/max date 作为范围。

    Parameters
    ----------
    df : 输入 DataFrame，需包含日期列和分组列
    date_col : 日期列名
    group_col : 分组列名（如 site_id）
    start_date, end_date : 可选的全局日期范围覆盖

    Returns
    -------
    pd.DataFrame : 对齐后的 DataFrame，每个 group 每天一行
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    aligned_parts = []
    for group_id, group_df in df.groupby(group_col):
        group_df = group_df.set_index(date_col).sort_index()

        # 确定日期范围
        d_start = pd.Timestamp(start_date) if start_date else group_df.index.min()
        d_end = pd.Timestamp(end_date) if end_date else group_df.index.max()
        full_idx = pd.date_range(d_start, d_end, freq="D", name=date_col)

        # reindex，缺失日自动填 NaN
        group_aligned = group_df.reindex(full_idx)
        group_aligned[group_col] = group_id
        aligned_parts.append(group_aligned.reset_index())

    return pd.concat(aligned_parts, ignore_index=True)


def compute_rolling_features(
    df: pd.DataFrame,
    value_col: str = "pcr_conc_lin",
    group_col: str = "site_id",
    windows: list[int] | None = None,
    min_periods_ratio: float = 0.57,
) -> pd.DataFrame:
    """
    对每个站点计算滚动统计特征。

    生成的列：
    - {value_col}_log1p : log(1+x) 变换
    - rolling_median_{w}d : w 天滚动中位数
    - rolling_std_{w}d : w 天滚动标准差
    - rolling_zscore_{w}d : (x - rolling_median) / rolling_std
    - pct_change_7d : 7 天变化率

    Parameters
    ----------
    df : 已对齐到日网格的 DataFrame
    value_col : 浓度值列名
    group_col : 分组列名
    windows : 滚动窗口列表，默认 [7, 14]
    min_periods_ratio : min_periods = window * ratio

    Returns
    -------
    pd.DataFrame : 附加了派生列的 DataFrame
    """
    if windows is None:
        windows = [7, 14]

    df = df.copy()

    # log1p 变换
    log_col = f"{value_col}_log1p"
    df[log_col] = np.log1p(df[value_col].clip(lower=0))

    result_parts = []
    for _, group_df in df.groupby(group_col):
        group_df = group_df.sort_values("date").copy()
        series = group_df[log_col]

        for w in windows:
            min_p = max(1, int(w * min_periods_ratio))
            rolling = series.rolling(window=w, min_periods=min_p)

            median_col = f"rolling_median_{w}d"
            std_col = f"rolling_std_{w}d"
            zscore_col = f"rolling_zscore_{w}d"

            group_df[median_col] = rolling.median()
            group_df[std_col] = rolling.std()
            group_df[zscore_col] = (
                (series - group_df[median_col]) / group_df[std_col].replace(0, np.nan)
            )

        # 7 天变化率
        group_df["pct_change_7d"] = series.pct_change(periods=7, fill_method=None)

        # 站点内 z-score 标准化（用于跨站点比较）
        site_mean = series.mean()
        site_std = series.std()
        group_df["site_zscore"] = (
            (series - site_mean) / site_std if site_std > 0 else 0.0
        )

        result_parts.append(group_df)

    return pd.concat(result_parts, ignore_index=True)


def compute_sampling_pattern(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "pcr_conc_lin",
    group_col: str = "site_id",
) -> pd.DataFrame:
    """
    分析每个站点的采样模式（规律性周采样 vs 随机缺失）。

    Returns
    -------
    pd.DataFrame : 每个站点一行，包含 median_gap, modal_gap, gap_cv,
                   is_regular (bool), dominant_frequency
    """
    records = []
    for site_id, group_df in df.groupby(group_col):
        observed = group_df.dropna(subset=[value_col])[date_col].sort_values()
        if len(observed) < 2:
            records.append({"site_id": site_id, "n_obs": len(observed)})
            continue

        gaps = observed.diff().dt.days.dropna()
        modal_gap = gaps.mode().iloc[0] if len(gaps.mode()) > 0 else gaps.median()

        records.append(
            {
                "site_id": site_id,
                "n_obs": len(observed),
                "median_gap": gaps.median(),
                "mean_gap": gaps.mean(),
                "modal_gap": modal_gap,
                "gap_cv": gaps.std() / gaps.mean() if gaps.mean() > 0 else 0,
                "is_regular": gaps.std() / gaps.mean() < 0.3 if gaps.mean() > 0 else False,
                "dominant_frequency": f"~{int(modal_gap)}d",
            }
        )
    return pd.DataFrame(records)
