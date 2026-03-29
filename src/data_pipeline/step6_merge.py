"""
Step 6: 多源数据合并。

将 NWSS、HHS、NOAA、USGS、CDC 变异株数据合并为统一的站点-日期级数据库。

合并键：
  - NWSS ↔ NOAA/USGS: site_id + date
  - NWSS ↔ HHS: state + date
  - NWSS ↔ Variants: state (→ HHS Region) + date

输出：一个 parquet 文件，每行 = 一个站点的一天，包含所有数据源的字段。
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import settings, PROJECT_ROOT


def load_processed_data() -> dict[str, pd.DataFrame]:
    """加载所有已处理的数据源。"""
    datasets = {}

    paths = {
        "nwss": settings.paths.nwss_processed,
        "hhs": settings.paths.hhs_processed,
        "noaa": settings.paths.noaa_processed,
        "usgs": settings.paths.usgs_processed,
        "variants": settings.paths.variants_processed,
    }

    for name, rel_path in paths.items():
        full_path = PROJECT_ROOT / rel_path
        if full_path.exists():
            datasets[name] = pd.read_parquet(full_path)
            logger.info(f"加载 {name}: {len(datasets[name])} 行")
        else:
            logger.warning(f"{name} 数据不存在: {full_path}")

    return datasets


def merge_all(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    合并所有数据源。

    合并策略：以 NWSS 为主表，左连接其他数据源。
    """
    if "nwss" not in datasets:
        raise ValueError("NWSS 数据是必需的主表")

    merged = datasets["nwss"].copy()
    merged["date"] = pd.to_datetime(merged["date"])

    # 确保有 site_id 列（可能叫 key_plot_id）
    if "key_plot_id" in merged.columns and "site_id" not in merged.columns:
        merged["site_id"] = merged["key_plot_id"]

    # ── 合并 NOAA 气象数据 ──
    if "noaa" in datasets:
        noaa = datasets["noaa"].copy()
        noaa["date"] = pd.to_datetime(noaa["date"])
        # NOAA 数据的 site_id 对应 NWSS 的 site_id
        noaa_cols = ["site_id", "date", "PRCP", "TMAX", "TMIN", "TAVG",
                     "noaa_station_id", "match_distance_km"]
        noaa_cols = [c for c in noaa_cols if c in noaa.columns]
        noaa = noaa[noaa_cols]

        # 重命名避免冲突
        rename_map = {
            "PRCP": "precipitation_mm",
            "TMAX": "temp_max_c",
            "TMIN": "temp_min_c",
            "TAVG": "temp_avg_c",
            "match_distance_km": "noaa_match_distance_km",
        }
        noaa = noaa.rename(columns={k: v for k, v in rename_map.items() if k in noaa.columns})

        merged = merged.merge(noaa, on=["site_id", "date"], how="left")
        logger.info(f"合并 NOAA 后: {len(merged)} 行")

    # ── 合并 USGS 水文数据 ──
    if "usgs" in datasets:
        usgs = datasets["usgs"].copy()
        usgs["date"] = pd.to_datetime(usgs["date"])
        usgs_cols = ["site_id", "date", "discharge_cfs", "discharge_percentile",
                     "usgs_site_no", "match_distance_km"]
        usgs_cols = [c for c in usgs_cols if c in usgs.columns]
        usgs = usgs[usgs_cols]

        rename_map = {"match_distance_km": "usgs_match_distance_km"}
        usgs = usgs.rename(columns={k: v for k, v in rename_map.items() if k in usgs.columns})

        merged = merged.merge(usgs, on=["site_id", "date"], how="left")
        logger.info(f"合并 USGS 后: {len(merged)} 行")

    # ── 合并 HHS 住院数据 ──
    if "hhs" in datasets:
        hhs = datasets["hhs"].copy()
        hhs["date"] = pd.to_datetime(hhs["date"])

        # 需要 state 列来做匹配
        if "state" in merged.columns and "state" in hhs.columns:
            hhs_cols = [c for c in hhs.columns if c != "state" or c == "state"]
            # 避免重复列
            hhs_merge_cols = ["state", "date"] + [
                c for c in hhs.columns
                if c not in ["state", "date"] and c not in merged.columns
            ]
            hhs_subset = hhs[[c for c in hhs_merge_cols if c in hhs.columns]]

            merged = merged.merge(hhs_subset, on=["state", "date"], how="left")
            logger.info(f"合并 HHS 后: {len(merged)} 行")
        else:
            logger.warning("NWSS 或 HHS 缺少 state 列，跳过 HHS 合并")

    # ── 合并变异株数据 ──
    if "variants" in datasets:
        variants = datasets["variants"].copy()
        variants["date"] = pd.to_datetime(variants["date"])

        if "state" in variants.columns:
            var_cols = ["state", "date", "dominant_variant", "dominant_proportion",
                        "n_emerging_variants", "emerging_variants"]
            var_cols = [c for c in var_cols if c in variants.columns]
            variants_subset = variants[var_cols].drop_duplicates(subset=["state", "date"])

            merged = merged.merge(variants_subset, on=["state", "date"], how="left")
            logger.info(f"合并变异株后: {len(merged)} 行")

    # ── 排序 ──
    merged = merged.sort_values(["site_id", "date"]).reset_index(drop=True)

    logger.info(
        f"最终合并数据: {len(merged)} 行, "
        f"{merged['site_id'].nunique()} 个站点, "
        f"{len(merged.columns)} 列"
    )
    logger.info(f"列: {list(merged.columns)}")

    return merged


def run_step6() -> pd.DataFrame:
    """执行 Step 6 完整流程。"""
    logger.info("=" * 60)
    logger.info("Step 6: 多源数据合并")
    logger.info("=" * 60)

    datasets = load_processed_data()
    merged = merge_all(datasets)

    output_path = PROJECT_ROOT / settings.paths.merged_database
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info(f"合并数据保存至: {output_path}")

    return merged


if __name__ == "__main__":
    run_step6()
