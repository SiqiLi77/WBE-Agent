"""
Step 7: 数据质量报告生成。

对合并后的多源数据库进行全面质量检查，输出 HTML/CSV 报告。

检查项：
  - 各数据源的时间覆盖率
  - 各字段的缺失率
  - 站点间的数据一致性
  - 空间匹配质量（距离分布）
  - 异常值检测
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings, PROJECT_ROOT


def compute_coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每个站点各数据源的时间覆盖率。

    Returns
    -------
    pd.DataFrame : 每个站点一行，列包含各数据源的覆盖率百分比
    """
    # 定义各数据源的代表性列
    source_columns = {
        "nwss": "pcr_conc_lin",
        "noaa_precip": "precipitation_mm",
        "noaa_temp": "temp_avg_c",
        "usgs": "discharge_cfs",
        "hhs": "previous_day_admission_adult_covid_confirmed",
        "variants": "dominant_variant",
    }

    records = []
    for site_id, site_df in df.groupby("site_id"):
        n_days = len(site_df)
        record = {
            "site_id": site_id,
            "total_days": n_days,
            "date_min": site_df["date"].min(),
            "date_max": site_df["date"].max(),
        }
        for source_name, col in source_columns.items():
            if col in site_df.columns:
                n_valid = site_df[col].notna().sum()
                record[f"{source_name}_count"] = n_valid
                record[f"{source_name}_coverage_pct"] = round(n_valid / n_days * 100, 1) if n_days > 0 else 0
            else:
                record[f"{source_name}_count"] = 0
                record[f"{source_name}_coverage_pct"] = 0.0
        records.append(record)

    return pd.DataFrame(records)


def compute_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """计算每列的缺失率。"""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)

    report = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": pct.values,
        "dtype": [str(df[c].dtype) for c in missing.index],
    })
    return report.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def compute_spatial_match_quality(df: pd.DataFrame) -> pd.DataFrame:
    """评估空间匹配质量（NOAA/USGS 匹配距离分布）。"""
    records = []

    for dist_col, source in [
        ("noaa_match_distance_km", "NOAA"),
        ("usgs_match_distance_km", "USGS"),
    ]:
        if dist_col not in df.columns:
            continue
        distances = df.groupby("site_id")[dist_col].first().dropna()
        records.append({
            "source": source,
            "n_matched": len(distances),
            "mean_distance_km": distances.mean(),
            "median_distance_km": distances.median(),
            "max_distance_km": distances.max(),
            "min_distance_km": distances.min(),
        })

    return pd.DataFrame(records)


def compute_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """检测各数值列的极端值。"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    records = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        n_low = (series < q1 - 3 * iqr).sum()
        n_high = (series > q3 + 3 * iqr).sum()
        records.append({
            "column": col,
            "n_valid": len(series),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "q25": q1,
            "median": series.median(),
            "q75": q3,
            "max": series.max(),
            "n_extreme_low": n_low,
            "n_extreme_high": n_high,
        })

    return pd.DataFrame(records)


def generate_quality_report(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """生成完整的数据质量报告。"""
    report = {
        "coverage": compute_coverage_report(df),
        "missing": compute_missing_report(df),
        "spatial_match": compute_spatial_match_quality(df),
        "outliers": compute_outlier_summary(df),
    }
    return report


def run_step7() -> dict[str, pd.DataFrame]:
    """执行 Step 7 完整流程。"""
    logger.info("=" * 60)
    logger.info("Step 7: 数据质量报告")
    logger.info("=" * 60)

    merged_path = PROJECT_ROOT / settings.paths.merged_database
    if not merged_path.exists():
        raise FileNotFoundError(f"合并数据不存在: {merged_path}，请先运行 Step 6")

    df = pd.read_parquet(merged_path)
    logger.info(f"加载合并数据: {len(df)} 行, {len(df.columns)} 列")

    report = generate_quality_report(df)

    # 保存各报告
    report_dir = PROJECT_ROOT / settings.paths.outputs_dir / "quality_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    for name, report_df in report.items():
        path = report_dir / f"{name}.csv"
        report_df.to_csv(path, index=False)
        logger.info(f"  {name}: {path}")

    # 打印摘要
    logger.info("\n=== 数据覆盖率摘要 ===")
    logger.info(f"\n{report['coverage'].to_string()}")
    logger.info("\n=== 缺失率 Top 10 ===")
    logger.info(f"\n{report['missing'].head(10).to_string()}")
    logger.info("\n=== 空间匹配质量 ===")
    logger.info(f"\n{report['spatial_match'].to_string()}")

    return report


if __name__ == "__main__":
    run_step7()
