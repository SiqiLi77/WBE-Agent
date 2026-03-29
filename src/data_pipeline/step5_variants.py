"""
Step 5: CDC 变异株数据获取与处理。

输入：CDC Variant Surveillance 数据
输出：按 HHS Region + 周 的变异株比例数据，对齐到日频

数据源：CDC COVID Data Tracker - Variant Proportions
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

from src.config import settings, PROJECT_ROOT


# CDC Variant Proportions API (SODA API on data.cdc.gov)
CDC_VARIANTS_URL = "https://data.cdc.gov/resource/jr58-6ysp.json"

# 州 → HHS Region 映射
STATE_TO_HHS_REGION: dict[str, int] = {
    "CT": 1, "ME": 1, "MA": 1, "NH": 1, "RI": 1, "VT": 1,
    "NJ": 2, "NY": 2,
    "DE": 3, "DC": 3, "MD": 3, "PA": 3, "VA": 3, "WV": 3,
    "AL": 4, "FL": 4, "GA": 4, "KY": 4, "MS": 4, "NC": 4, "SC": 4, "TN": 4,
    "IL": 5, "IN": 5, "MI": 5, "MN": 5, "OH": 5, "WI": 5,
    "AR": 6, "LA": 6, "NM": 6, "OK": 6, "TX": 6,
    "IA": 7, "KS": 7, "MO": 7, "NE": 7,
    "CO": 8, "MT": 8, "ND": 8, "SD": 8, "UT": 8, "WY": 8,
    "AZ": 9, "CA": 9, "HI": 9, "NV": 9,
    "AK": 10, "ID": 10, "OR": 10, "WA": 10,
}


def get_hhs_region(state: str) -> int | None:
    """州缩写 → HHS Region 编号。"""
    return STATE_TO_HHS_REGION.get(state.upper())


def download_variant_data(hhs_regions: list[int] | None = None, limit: int = 50000) -> pd.DataFrame:
    """
    从 CDC SODA API 下载变异株比例数据。

    只下载 share >= 0.05 (5%) 的变异株以减少数据量。

    Parameters
    ----------
    hhs_regions : 只下载指定 HHS Region 的数据
    limit : 每次 API 请求的记录数

    Returns
    -------
    pd.DataFrame : 原始变异株数据
    """
    cache_path = PROJECT_ROOT / settings.paths.variants_raw_csv
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"使用缓存的变异株数据: {cache_path}")
        return pd.read_csv(cache_path)

    logger.info("从 CDC API 下载变异株数据...")

    # 构建 where 子句：只下载 share >= 5% 的变异株 + 指定 regions
    where_parts = ["share >= 0.05"]
    if hhs_regions:
        region_list = ",".join(f"'{r}'" for r in hhs_regions)
        where_parts.append(f"usa_or_hhsregion in({region_list})")
        logger.info(f"过滤 HHS Regions: {hhs_regions}")

    where_clause = " AND ".join(where_parts)

    all_records = []
    offset = 0

    while True:
        # 手动构建 URL 以避免 $ 参数被 requests 库处理
        import urllib.parse
        query_params = {
            "$limit": str(limit),
            "$offset": str(offset),
            "$order": "week_ending",
            "$where": where_clause,
        }
        query_string = "&".join(
            f"%24{k[1:]}={urllib.parse.quote(v)}" for k, v in query_params.items()
        )
        full_url = f"{CDC_VARIANTS_URL}?{query_string}"

        try:
            resp = requests.get(full_url, timeout=120)
            resp.raise_for_status()
            batch = resp.json()
            if not batch or isinstance(batch, dict):
                if isinstance(batch, dict):
                    logger.error(f"CDC API 错误: {batch}")
                break
            all_records.extend(batch)
            offset += limit
            logger.info(f"  已下载 {len(all_records)} 条记录...")
            if len(batch) < limit:
                break
        except Exception as e:
            logger.error(f"CDC API 请求失败: {e}")
            break

    if not all_records:
        logger.error("未获取到变异株数据")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df.to_csv(cache_path, index=False)
    logger.info(f"变异株数据下载完成: {len(df)} 行")
    return df

    if not all_records:
        logger.error("未获取到变异株数据")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df.to_csv(cache_path, index=False)
    logger.info(f"变异株数据下载完成: {len(df)} 行")
    return df


def process_variant_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理变异株数据：标准化、识别主导株和新兴株。

    Returns
    -------
    pd.DataFrame : 处理后的数据
    """
    if df.empty:
        return df

    df = df.copy()

    # 标准化列名（CDC API 实际列名）
    col_map = {
        "usa_or_hhsregion": "hhs_region",
        "share": "proportion",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "week_ending" in df.columns:
        df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")

    if "proportion" in df.columns:
        df["proportion"] = pd.to_numeric(df["proportion"], errors="coerce")

    # 识别主导株（每个 region-week 中占比最高的）
    if all(c in df.columns for c in ["hhs_region", "week_ending", "proportion"]):
        idx = df.groupby(["hhs_region", "week_ending"])["proportion"].idxmax()
        df["is_dominant"] = False
        df.loc[idx, "is_dominant"] = True

        # 计算增长率（周环比）
        df = df.sort_values(["hhs_region", "variant", "week_ending"])
        df["growth_rate"] = df.groupby(["hhs_region", "variant"])["proportion"].pct_change(fill_method=None)

        # 标记新兴株（占比 <10% 但增长率 >50%）
        df["is_emerging"] = (df["proportion"] < 0.10) & (df["growth_rate"] > 0.50)

    logger.info(f"变异株数据处理完成: {len(df)} 行")
    return df


def align_variants_to_daily(
    df: pd.DataFrame,
    states: list[str],
) -> pd.DataFrame:
    """
    将周频变异株数据对齐到日频（同周内填充相同值）。

    Parameters
    ----------
    df : 处理后的变异株数据
    states : 需要覆盖的州列表

    Returns
    -------
    pd.DataFrame : 日频数据，列包含 state, date, dominant_variant,
                   dominant_proportion, n_emerging_variants
    """
    if df.empty or "week_ending" not in df.columns:
        return pd.DataFrame()

    # 按 HHS Region 聚合每周的摘要
    weekly_summaries = []
    for (region, week), group in df.groupby(["hhs_region", "week_ending"]):
        dominant = group[group["is_dominant"]].iloc[0] if group["is_dominant"].any() else None
        emerging = group[group["is_emerging"]] if "is_emerging" in group.columns else pd.DataFrame()

        weekly_summaries.append(
            {
                "hhs_region": region,
                "week_ending": week,
                "dominant_variant": dominant["variant"] if dominant is not None else None,
                "dominant_proportion": dominant["proportion"] if dominant is not None else None,
                "n_emerging_variants": len(emerging),
                "emerging_variants": ", ".join(emerging["variant"].tolist()) if len(emerging) > 0 else None,
            }
        )

    summary_df = pd.DataFrame(weekly_summaries)

    # 展开到日频：每个 week_ending 覆盖前 7 天
    daily_records = []
    for _, row in summary_df.iterrows():
        week_end = row["week_ending"]
        for day_offset in range(7):
            daily_row = row.copy()
            daily_row["date"] = week_end - pd.Timedelta(days=day_offset)
            daily_records.append(daily_row)

    daily_df = pd.DataFrame(daily_records)

    # 映射到州
    result_parts = []
    for state in states:
        region = get_hhs_region(state)
        if region is None:
            continue
        # 支持 int 和 str 类型的 region 比较
        state_data = daily_df[daily_df["hhs_region"] == region].copy()
        if len(state_data) == 0:
            state_data = daily_df[daily_df["hhs_region"] == str(region)].copy()
        state_data["state"] = state
        result_parts.append(state_data)

    if result_parts:
        return pd.concat(result_parts, ignore_index=True)
    return pd.DataFrame()


def run_step5(states: list[str] | None = None) -> pd.DataFrame:
    """执行 Step 5 完整流程。"""
    logger.info("=" * 60)
    logger.info("Step 5: CDC 变异株数据处理")
    logger.info("=" * 60)

    if states is None:
        states = ["MT", "OR", "NV", "WI", "IL", "MN", "GA", "NY",
                  "PA", "NC", "FL", "TX", "WY", "AZ", "WA", "NM"]

    # 确定需要的 HHS Regions
    needed_regions = list(set(
        get_hhs_region(s) for s in states if get_hhs_region(s) is not None
    ))
    logger.info(f"需要的 HHS Regions: {sorted(needed_regions)}")

    # 下载（只下载需要的 regions）
    raw_df = download_variant_data(hhs_regions=needed_regions)

    # 处理
    processed = process_variant_data(raw_df)

    # 对齐到日频
    daily = align_variants_to_daily(processed, states)

    if len(daily) > 0:
        output_path = PROJECT_ROOT / settings.paths.variants_processed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(output_path, index=False)
        logger.info(f"变异株数据保存至: {output_path}")

    return daily


if __name__ == "__main__":
    run_step5()
