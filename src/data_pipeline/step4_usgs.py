"""
Step 4: USGS 水文站匹配与数据获取。

输入：NWSS 站点坐标
输出：每个 NWSS 站点匹配的水文站日流量数据

策略：
  1. 通过 USGS NWIS API 查询站点目录
  2. 对每个 NWSS 站点找最近的活跃水文站
  3. 下载日均流量（discharge）数据
  4. 计算流量百分位数（相对于历史分布）
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from loguru import logger

from src.config import settings, PROJECT_ROOT
from src.utils.geo import haversine_km


# USGS NWIS API
USGS_SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
USGS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"


def query_usgs_sites_near(
    lat: float,
    lon: float,
    state: str,
    radius_km: float = 30.0,
    parameter_code: str = "00060",  # Discharge
) -> pd.DataFrame:
    """
    查询指定州内的 USGS 水文站，然后按距离过滤。

    Parameters
    ----------
    lat, lon : 目标坐标
    state : 州缩写（如 "MT"）
    radius_km : 搜索半径
    parameter_code : USGS 参数代码（00060=discharge, 00065=gage height）

    Returns
    -------
    pd.DataFrame : 附近的 USGS 站点列表
    """
    cache_dir = PROJECT_ROOT / settings.paths.usgs_cache_dir / "site_queries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"sites_{state}_{parameter_code}.csv"
    cache_file = cache_dir / cache_key

    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        params = {
            "format": "rdb",
            "stateCd": state,
            "parameterCd": parameter_code,
            "siteType": "ST",  # Stream
            "siteStatus": "active",
            "hasDataTypeCd": "dv",  # Daily values
        }

        try:
            resp = requests.get(USGS_SITE_URL, params=params, timeout=60)
            resp.raise_for_status()
            df = _parse_usgs_rdb(resp.text)
            if len(df) > 0:
                df.to_csv(cache_file, index=False)
        except Exception as e:
            logger.error(f"USGS 站点查询失败 (state={state}): {e}")
            return pd.DataFrame()

    if len(df) == 0:
        return df

    # 确保数值列
    for col in ["dec_lat_va", "dec_long_va"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["dec_lat_va", "dec_long_va"])

    # 计算距离并过滤
    df["distance_km"] = df.apply(
        lambda r: haversine_km(lat, lon, r["dec_lat_va"], r["dec_long_va"]),
        axis=1,
    )
    df = df[df["distance_km"] <= radius_km].sort_values("distance_km")
    return df


def _parse_usgs_rdb(text: str) -> pd.DataFrame:
    """解析 USGS RDB (tab-separated) 格式响应。"""
    lines = []
    header = None
    skip_next = False
    for line in text.split("\n"):
        if line.startswith("#"):
            continue
        if header is None:
            header = line.strip().split("\t")
            skip_next = True  # 下一行是格式描述行（如 5s 15s 等）
            continue
        if skip_next:
            skip_next = False
            continue
        if line.strip():
            lines.append(line.strip().split("\t"))

    if not header or not lines:
        return pd.DataFrame()

    df = pd.DataFrame(lines, columns=header[: len(lines[0])] if lines else header)

    # 转换数值列
    for col in ["dec_lat_va", "dec_long_va", "drain_area_va"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def download_usgs_daily_discharge(
    site_no: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    下载单个 USGS 站点的日均流量数据。

    Returns
    -------
    pd.DataFrame : 列包含 date, discharge_cfs
    """
    cache_dir = PROJECT_ROOT / settings.paths.usgs_cache_dir / "daily_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{site_no}_{start_date}_{end_date}.parquet"

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    params = {
        "format": "json",
        "sites": site_no,
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": "00060",  # Discharge
        "statCd": "00003",  # Mean
    }

    try:
        resp = requests.get(USGS_DV_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # 解析 JSON 响应
        time_series = data.get("value", {}).get("timeSeries", [])
        if not time_series:
            logger.warning(f"USGS {site_no}: 无数据")
            return pd.DataFrame(columns=["date", "discharge_cfs"])

        values = time_series[0].get("values", [{}])[0].get("value", [])
        records = []
        for v in values:
            records.append(
                {
                    "date": pd.Timestamp(v["dateTime"]),
                    "discharge_cfs": float(v["value"]) if v["value"] != "-999999" else np.nan,
                }
            )

        df = pd.DataFrame(records)
        if len(df) > 0:
            df.to_parquet(cache_file, index=False)
        return df

    except Exception as e:
        logger.error(f"USGS {site_no} 数据下载失败: {e}")
        return pd.DataFrame(columns=["date", "discharge_cfs"])


def match_and_download_usgs(
    nwss_sites: pd.DataFrame,
    site_lat_col: str = "latitude",
    site_lon_col: str = "longitude",
    site_id_col: str = "key_plot_id",
) -> pd.DataFrame:
    """对所有 NWSS 站点执行 USGS 匹配和数据下载。"""
    max_dist = settings.spatial_matching.usgs_max_distance_km
    all_hydro = []

    for _, site in nwss_sites.iterrows():
        site_id = site[site_id_col]
        lat, lon = site[site_lat_col], site[site_lon_col]
        state = site.get("state", "")

        if not state:
            logger.warning(f"站点 {site_id}: 缺少 state 信息，跳过")
            continue

        # 查询该州的 USGS 站点，按距离过滤
        nearby = query_usgs_sites_near(lat, lon, state=state, radius_km=max_dist)

        if len(nearby) == 0:
            logger.warning(f"站点 {site_id} ({state}): 无匹配 USGS 站 (半径 {max_dist}km)")
            continue

        # 选最近的
        best = nearby.iloc[0]
        usgs_site_no = str(best.get("site_no", ""))
        distance = best.get("distance_km", np.nan)

        logger.info(f"站点 {site_id} → USGS {usgs_site_no} (距离 {distance:.1f}km)")

        # 下载日流量
        start_date = str(
            site.get("start_date", site.get("date_min", "2020-01-01"))
        )[:10]
        end_date = str(
            site.get("end_date", site.get("date_max", "2024-12-31"))
        )[:10]
        discharge = download_usgs_daily_discharge(usgs_site_no, start_date, end_date)

        if len(discharge) > 0:
            discharge["site_id"] = site_id
            discharge["usgs_site_no"] = usgs_site_no
            discharge["match_distance_km"] = distance

            # 计算流量百分位
            discharge["discharge_percentile"] = discharge["discharge_cfs"].rank(pct=True)

            all_hydro.append(discharge)

    if all_hydro:
        result = pd.concat(all_hydro, ignore_index=True)
        logger.info(f"USGS 数据获取完成: {len(result)} 行, {result['site_id'].nunique()} 个站点")
        return result
    else:
        logger.error("未获取到任何 USGS 数据")
        return pd.DataFrame()


def run_step4(nwss_sites: pd.DataFrame | None = None) -> pd.DataFrame:
    """执行 Step 4 完整流程。"""
    logger.info("=" * 60)
    logger.info("Step 4: USGS 水文数据匹配与下载")
    logger.info("=" * 60)

    if nwss_sites is None:
        tier1_path = PROJECT_ROOT / settings.paths.tier1_sites
        if tier1_path.exists():
            nwss_sites = pd.read_csv(tier1_path)
        else:
            raise FileNotFoundError("需要提供 NWSS 站点元数据（含经纬度）")

    df = match_and_download_usgs(nwss_sites)

    if len(df) > 0:
        output_path = PROJECT_ROOT / settings.paths.usgs_processed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"USGS 数据保存至: {output_path}")

    return df


if __name__ == "__main__":
    run_step4()
