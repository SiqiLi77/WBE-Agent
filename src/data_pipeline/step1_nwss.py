"""
Step 1: NWSS 污水数据加载、过滤与预处理。

输入：原始 NWSS CSV + 站点清单
输出：清洗后的 parquet 文件，包含原始值和所有派生特征

处理流程：
  1. 加载原始 CSV（仅读取需要的列）
  2. 按 key_plot_id 过滤选定站点
  3. 按 normalization == "flow-population" 过滤
  4. 日期解析、去重
  5. 数据完整性校验
  6. 时间网格对齐
  7. 信号变换（log1p, rolling median, z-score 等）
  8. 保存为 parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings, PROJECT_ROOT
from src.utils.time_align import (
    align_to_daily_grid,
    compute_rolling_features,
    compute_sampling_pattern,
)


def load_site_list() -> set[str]:
    """加载所有层级的选定站点 key_plot_id 集合。"""
    site_ids: set[str] = set()
    for tier_key in ["tier1_sites", "tier2_sites", "tier3_sites"]:
        path = PROJECT_ROOT / getattr(settings.paths, tier_key)
        if path.exists():
            df = pd.read_csv(path)
            # 尝试常见列名
            for col in ["key_plot_id", "site_id", "Key Plot ID"]:
                if col in df.columns:
                    site_ids.update(df[col].dropna().astype(str).tolist())
                    break
            logger.info(f"从 {path.name} 加载了 {len(df)} 个站点")
        else:
            logger.warning(f"站点文件不存在: {path}")
    logger.info(f"共加载 {len(site_ids)} 个唯一站点")
    return site_ids


def load_and_filter_nwss(site_ids: set[str]) -> pd.DataFrame:
    """
    加载原始 NWSS CSV 并过滤。

    只读取需要的列以节省内存。
    过滤条件：key_plot_id ∈ site_ids AND normalization == flow-population
    """
    csv_path = PROJECT_ROOT / settings.paths.nwss_raw_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"NWSS 原始数据不存在: {csv_path}")

    logger.info(f"加载 NWSS 数据: {csv_path}")

    # 尝试只读需要的列
    usecols = ["key_plot_id", "date", "pcr_conc_lin", "normalization"]
    try:
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except ValueError:
        # 列名可能不同，先读取全部列名
        sample = pd.read_csv(csv_path, nrows=0)
        logger.warning(f"列名不匹配，实际列: {list(sample.columns)}")
        df = pd.read_csv(csv_path, low_memory=False)

    logger.info(f"原始数据: {len(df)} 行")

    # 过滤站点
    df = df[df["key_plot_id"].astype(str).isin(site_ids)]
    logger.info(f"站点过滤后: {len(df)} 行")

    # 过滤 normalization
    norm_filter = settings.nwss.normalization_filter
    df = df[df["normalization"] == norm_filter]
    logger.info(f"normalization={norm_filter} 过滤后: {len(df)} 行")

    # 提取州信息
    def _extract_state(kpid: str) -> str:
        parts = str(kpid).split("_")
        if parts[0] == "CDC" and len(parts) > 2:
            return parts[2].upper()
        elif parts[0] == "NWSS" and len(parts) > 1:
            return parts[1].upper()
        return ""
    df["state"] = df["key_plot_id"].apply(_extract_state)

    return df


def parse_dates_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """日期解析、排序、去重。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["key_plot_id", "date"])

    # 检查重复
    dupes = df.groupby(["key_plot_id", "date"]).size()
    n_dupes = (dupes > 1).sum()
    if n_dupes > 0:
        logger.warning(f"发现 {n_dupes} 个 (站点, 日期) 重复组合")
        strategy = settings.nwss.duplicate_strategy
        if strategy == "mean":
            df = df.groupby(["key_plot_id", "date"], as_index=False).agg(
                {"pcr_conc_lin": "mean", "normalization": "first"}
            )
        elif strategy == "first":
            df = df.drop_duplicates(subset=["key_plot_id", "date"], keep="first")
        else:
            df = df.drop_duplicates(subset=["key_plot_id", "date"], keep="last")
        logger.info(f"去重策略: {strategy}，去重后: {len(df)} 行")

    return df


def validate_site_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    逐站点数据完整性校验。

    检查并处理：
    - NaN 值：保留标记
    - 零值：根据配置替换为 LOD/2
    - 负值：标记剔除
    """
    df = df.copy()

    # 负值处理
    n_negative = (df["pcr_conc_lin"] < 0).sum()
    if n_negative > 0:
        logger.warning(f"发现 {n_negative} 个负值，标记为 NaN")
        df.loc[df["pcr_conc_lin"] < 0, "pcr_conc_lin"] = np.nan

    # 零值处理
    n_zero = (df["pcr_conc_lin"] == 0).sum()
    if n_zero > 0:
        handling = settings.nwss.zero_handling
        if handling == "lod_half":
            lod_half = settings.nwss.default_lod / 2
            logger.info(f"发现 {n_zero} 个零值，替换为 LOD/2 = {lod_half}")
            df.loc[df["pcr_conc_lin"] == 0, "pcr_conc_lin"] = lod_half
        elif handling == "drop":
            df.loc[df["pcr_conc_lin"] == 0, "pcr_conc_lin"] = np.nan
        # "keep" 则不处理

    # 逐站点统计
    site_stats = df.groupby("key_plot_id").agg(
        n_obs=("pcr_conc_lin", "count"),
        n_missing=("pcr_conc_lin", lambda x: x.isna().sum()),
        date_min=("date", "min"),
        date_max=("date", "max"),
        conc_mean=("pcr_conc_lin", "mean"),
        conc_std=("pcr_conc_lin", "std"),
        conc_median=("pcr_conc_lin", "median"),
    )
    logger.info(f"站点数据校验完成:\n{site_stats.to_string()}")

    return df


def run_step1() -> pd.DataFrame:
    """
    执行 Step 1 完整流程。

    Returns
    -------
    pd.DataFrame : 处理完成的 NWSS 数据，包含原始值和所有派生特征
    """
    logger.info("=" * 60)
    logger.info("Step 1: NWSS 污水数据预处理")
    logger.info("=" * 60)

    # 1. 加载站点清单
    site_ids = load_site_list()
    if not site_ids:
        raise ValueError("未找到任何选定站点")

    # 2. 加载并过滤原始数据
    df = load_and_filter_nwss(site_ids)

    # 3. 日期解析与去重
    df = parse_dates_and_deduplicate(df)

    # 4. 数据完整性校验
    df = validate_site_data(df)

    # 5. 采样模式分析
    sampling = compute_sampling_pattern(
        df, date_col="date", value_col="pcr_conc_lin", group_col="key_plot_id"
    )
    logger.info(f"采样模式:\n{sampling.to_string()}")

    # 6. 时间网格对齐
    df = align_to_daily_grid(df, date_col="date", group_col="key_plot_id")

    # 7. 信号变换
    df = compute_rolling_features(
        df,
        value_col="pcr_conc_lin",
        group_col="key_plot_id",
        windows=settings.nwss.rolling_windows,
        min_periods_ratio=settings.nwss.rolling_min_periods_ratio,
    )

    # 8. 保存
    output_path = PROJECT_ROOT / settings.paths.nwss_processed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"NWSS 处理完成，保存至: {output_path} ({len(df)} 行)")

    # 保存采样模式报告
    sampling_path = output_path.parent / "nwss_sampling_patterns.csv"
    sampling.to_csv(sampling_path, index=False)

    return df


if __name__ == "__main__":
    run_step1()
