"""
Step 2: HHS 住院数据处理。

输入：原始 HHS CSV 或已处理的 overlap 文件
输出：清洗后的 parquet，按州-日期索引

处理流程：
  1. 加载已有的 HHS overlap 数据（如果存在）或原始 HHS CSV
  2. 选择建模所需字段
  3. 日期解析与排序
  4. 计算派生指标（7天均值、趋势方向等）
  5. 保存为 parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings, PROJECT_ROOT


# 建模所需的 HHS 字段
HHS_KEY_COLUMNS = [
    "state",
    "date",
    "previous_day_admission_adult_covid_confirmed",
    "previous_day_admission_pediatric_covid_confirmed",
    "total_adult_patients_hospitalized_confirmed_covid",
    "staffed_adult_icu_bed_occupancy",
    "deaths_covid",
    "inpatient_beds",
    "inpatient_bed_covid_utilization_numerator",
]


def load_hhs_data() -> pd.DataFrame:
    """
    加载 HHS 数据。

    优先使用已处理的 overlap 文件（更小、已对齐），
    否则从原始 CSV 加载并过滤到选定的 16 州。
    """
    # 优先尝试已处理的 overlap 文件
    overlap_path = PROJECT_ROOT / "outputs" / "hhs_16states_overlap_only.csv"
    if overlap_path.exists():
        logger.info(f"使用已处理的 HHS overlap 数据: {overlap_path}")
        df = pd.read_csv(overlap_path, low_memory=False)
        logger.info(f"加载 {len(df)} 行")
        return df

    # 尝试 16 州全量文件
    states_path = PROJECT_ROOT / "outputs" / "hhs_16states.csv"
    if states_path.exists():
        logger.info(f"使用 16 州 HHS 数据: {states_path}")
        df = pd.read_csv(states_path, low_memory=False)
        logger.info(f"加载 {len(df)} 行")
        return df

    # 从原始 CSV 加载
    raw_path = PROJECT_ROOT / settings.paths.hhs_raw_csv
    if not raw_path.exists():
        raise FileNotFoundError(
            f"HHS 数据不存在: {raw_path}\n"
            "请下载: https://healthdata.gov/api/views/g62h-syeh/rows.csv"
        )

    logger.info(f"从原始 HHS CSV 加载: {raw_path}")
    # 文件很大，只读需要的列
    available_cols = pd.read_csv(raw_path, nrows=0).columns.tolist()
    use_cols = [c for c in HHS_KEY_COLUMNS if c in available_cols]
    df = pd.read_csv(raw_path, usecols=use_cols, low_memory=False)
    logger.info(f"原始 HHS 数据: {len(df)} 行")

    return df


def process_hhs(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 HHS 数据并计算派生指标。"""
    df = df.copy()

    # 日期解析
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["state", "date"])

    # 主目标列：成人 COVID 确诊入院
    target_col = "previous_day_admission_adult_covid_confirmed"
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        # 逐州计算派生指标
        parts = []
        for state, state_df in df.groupby("state"):
            state_df = state_df.copy()
            series = state_df[target_col]

            # 7 天滚动均值
            state_df["admission_7d_avg"] = series.rolling(7, min_periods=4).mean()

            # 14 天滚动均值
            state_df["admission_14d_avg"] = series.rolling(14, min_periods=7).mean()

            # 周环比变化率
            state_df["admission_pct_change_7d"] = series.pct_change(periods=7, fill_method=None)

            # 趋势方向（基于 7 天均值的 7 天变化）
            avg_change = state_df["admission_7d_avg"].diff(periods=7)
            state_df["admission_trend"] = pd.cut(
                avg_change,
                bins=[-np.inf, -0.1, 0.1, np.inf],
                labels=["decreasing", "stable", "increasing"],
            )

            parts.append(state_df)

        df = pd.concat(parts, ignore_index=True)

    logger.info(f"HHS 处理完成: {len(df)} 行, {df['state'].nunique()} 个州")
    return df


def run_step2() -> pd.DataFrame:
    """执行 Step 2 完整流程。"""
    logger.info("=" * 60)
    logger.info("Step 2: HHS 住院数据处理")
    logger.info("=" * 60)

    df = load_hhs_data()
    df = process_hhs(df)

    # 保存
    output_path = PROJECT_ROOT / settings.paths.hhs_processed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"HHS 数据保存至: {output_path}")

    return df


if __name__ == "__main__":
    run_step2()
