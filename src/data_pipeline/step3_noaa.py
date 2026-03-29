"""
Step 3: NOAA 气象站匹配与数据获取。

输入：NWSS 站点坐标 + GHCN-Daily 站点目录 + 站点清单（inventory）
输出：每个 NWSS 站点匹配的气象站日降水量和气温数据

策略：
  1. 下载 GHCN-Daily 站点目录 + inventory（含数据年份范围）
  2. 过滤出有 2020+ PRCP 数据的活跃站点
  3. 对每个 NWSS 站点，用 Haversine 找最近的活跃 GHCN 站点
  4. 下载匹配站点的日数据（PRCP, TMAX, TMIN）
  5. 清洗并对齐到 NWSS 时间网格
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import requests
from loguru import logger

from src.config import settings, PROJECT_ROOT
from src.utils.geo import batch_spatial_match, find_nearest_station


# GHCN-Daily URLs
GHCN_STATIONS_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
GHCN_INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
GHCN_DATA_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/{station_id}.dly"


def download_ghcn_inventory() -> pd.DataFrame:
    """
    下载 GHCN inventory（每个站点每个要素的年份范围）。

    格式：STATION_ID(11) LATITUDE(9) LONGITUDE(9) ELEMENT(4) FIRSTYEAR(5) LASTYEAR(5)
    """
    cache_path = PROJECT_ROOT / settings.paths.ghcn_daily_dir / "ghcnd-inventory.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"使用缓存的 GHCN inventory: {cache_path}")
        text = cache_path.read_text(encoding="utf-8")
    else:
        logger.info(f"下载 GHCN inventory: {GHCN_INVENTORY_URL}")
        resp = requests.get(GHCN_INVENTORY_URL, timeout=120)
        resp.raise_for_status()
        text = resp.text
        cache_path.write_text(text, encoding="utf-8")

    records = []
    for line in text.strip().split("\n"):
        if len(line) < 44:
            continue
        records.append({
            "station_id": line[0:11].strip(),
            "element": line[31:35].strip(),
            "first_year": int(line[36:40].strip()),
            "last_year": int(line[41:45].strip()),
        })

    return pd.DataFrame(records)


def download_ghcn_station_inventory(min_year: int = 2020) -> pd.DataFrame:
    """
    下载并解析 GHCN-Daily 站点目录，过滤出有近期数据的活跃站点。

    ghcnd-stations.txt 是固定宽度格式：
    ID            1-11   Character
    LATITUDE     13-20   Real
    LONGITUDE    22-30   Real
    ELEVATION    32-37   Real
    STATE        39-40   Character
    NAME         42-71   Character
    """
    cache_path = PROJECT_ROOT / settings.paths.ghcn_daily_dir / "ghcnd-stations.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"使用缓存的 GHCN 站点目录: {cache_path}")
        text = cache_path.read_text(encoding="utf-8")
    else:
        logger.info(f"下载 GHCN 站点目录: {GHCN_STATIONS_URL}")
        resp = requests.get(GHCN_STATIONS_URL, timeout=60)
        resp.raise_for_status()
        text = resp.text
        cache_path.write_text(text, encoding="utf-8")

    # 解析固定宽度格式
    records = []
    for line in text.strip().split("\n"):
        if len(line) < 40:
            continue
        records.append(
            {
                "station_id": line[0:11].strip(),
                "latitude": float(line[12:20].strip()),
                "longitude": float(line[21:30].strip()),
                "elevation": float(line[31:37].strip()) if line[31:37].strip() else np.nan,
                "state": line[38:40].strip(),
                "name": line[41:71].strip(),
            }
        )

    df = pd.DataFrame(records)
    # 只保留美国本土站点（以 US 开头）
    df = df[df["station_id"].str.startswith("US")]
    logger.info(f"GHCN 美国站点（全部）: {len(df)} 个")

    # 用 inventory 过滤：只保留有 PRCP 数据且 last_year >= min_year 的站点
    inv = download_ghcn_inventory()
    inv_us = inv[inv["station_id"].str.startswith("US")]
    active_prcp = inv_us[
        (inv_us["element"] == "PRCP") & (inv_us["last_year"] >= min_year)
    ]["station_id"].unique()

    df = df[df["station_id"].isin(active_prcp)]
    logger.info(f"GHCN 美国活跃站点（PRCP last_year >= {min_year}）: {len(df)} 个")

    return df


def download_ghcn_daily_data(
    station_id: str,
    start_date: str,
    end_date: str,
    elements: list[str] | None = None,
) -> pd.DataFrame:
    """
    下载单个 GHCN 站点的日数据。

    .dly 文件格式复杂（固定宽度，每行一个月），这里用 NOAA CDO API 替代。
    如果 API 不可用，fallback 到解析 .dly 文件。

    Parameters
    ----------
    station_id : GHCN 站点 ID
    start_date, end_date : 日期范围 (YYYY-MM-DD)
    elements : 要获取的气象要素，默认 ["PRCP", "TMAX", "TMIN"]

    Returns
    -------
    pd.DataFrame : 列包含 date, PRCP (mm), TMAX (°C), TMIN (°C)
    """
    if elements is None:
        elements = ["PRCP", "TMAX", "TMIN"]

    cache_dir = PROJECT_ROOT / settings.paths.ghcn_daily_dir / "station_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{station_id}_{start_date}_{end_date}.parquet"

    if cache_file.exists():
        logger.debug(f"使用缓存: {cache_file}")
        return pd.read_parquet(cache_file)

    # 尝试 NOAA CDO API (需要 token，可选)
    # Fallback: 下载 .dly 文件并解析
    dly_url = GHCN_DATA_URL.format(station_id=station_id)
    logger.info(f"下载 GHCN 日数据: {station_id}")

    try:
        resp = requests.get(dly_url, timeout=120)
        resp.raise_for_status()
        df = _parse_dly_file(resp.text, elements, start_date, end_date)
        df.to_parquet(cache_file, index=False)
        return df
    except Exception as e:
        logger.error(f"下载 {station_id} 失败: {e}")
        return pd.DataFrame(columns=["date"] + elements)


def _parse_dly_file(
    text: str,
    elements: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    解析 GHCN .dly 固定宽度文件。

    格式：每行 = 1 个站点 + 1 个月 + 1 个要素 + 31 天的值
    ID(11) YEAR(4) MONTH(2) ELEMENT(4) [VALUE(5)+MFLAG(1)+QFLAG(1)+SFLAG(1)] × 31
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    records: dict[str, dict] = {}  # date_str -> {element: value}

    for line in text.split("\n"):
        if len(line) < 269:
            continue

        element = line[17:21].strip()
        if element not in elements:
            continue

        year = int(line[11:15])
        month = int(line[15:17])

        for day in range(1, 32):
            offset = 21 + (day - 1) * 8
            value_str = line[offset : offset + 5].strip()
            qflag = line[offset + 6 : offset + 7].strip()

            if value_str == "-9999" or qflag != "":
                continue

            try:
                dt = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue  # 无效日期（如 2月30日）

            if dt < start or dt > end:
                continue

            date_key = dt.strftime("%Y-%m-%d")
            if date_key not in records:
                records[date_key] = {"date": dt}

            value = float(value_str)
            # GHCN 单位：PRCP=0.1mm, TMAX/TMIN=0.1°C
            if element == "PRCP":
                records[date_key][element] = value / 10.0  # → mm
            elif element in ("TMAX", "TMIN"):
                records[date_key][element] = value / 10.0  # → °C

    if not records:
        return pd.DataFrame(columns=["date"] + elements)

    df = pd.DataFrame(list(records.values()))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 计算日均温
    if "TMAX" in df.columns and "TMIN" in df.columns:
        df["TAVG"] = (df["TMAX"] + df["TMIN"]) / 2

    return df


def match_and_download_noaa(
    nwss_sites: pd.DataFrame,
    site_lat_col: str = "latitude",
    site_lon_col: str = "longitude",
    site_id_col: str = "key_plot_id",
) -> pd.DataFrame:
    """
    对所有 NWSS 站点执行 NOAA 匹配和数据下载。

    Parameters
    ----------
    nwss_sites : NWSS 站点元数据 DataFrame，需包含经纬度

    Returns
    -------
    pd.DataFrame : 所有站点的气象数据，列包含 site_id, date, PRCP, TMAX, TMIN, TAVG
    """
    # 1. 获取 GHCN 站点目录
    ghcn_stations = download_ghcn_station_inventory()

    # 2. 空间匹配
    matches = batch_spatial_match(
        nwss_sites,
        ghcn_stations,
        target_lat_col=site_lat_col,
        target_lon_col=site_lon_col,
        target_id_col=site_id_col,
        station_lat_col="latitude",
        station_lon_col="longitude",
        station_id_col="station_id",
        max_distance_km=settings.spatial_matching.noaa_max_distance_km,
    )

    logger.info(f"NOAA 匹配结果:\n{matches.to_string()}")
    unmatched = matches[matches["matched_station_id"].isna()]
    if len(unmatched) > 0:
        logger.warning(f"{len(unmatched)} 个站点无匹配 NOAA 站: {unmatched['target_id'].tolist()}")

    # 3. 逐站点下载数据
    all_weather = []
    for _, row in matches.dropna(subset=["matched_station_id"]).iterrows():
        site_id = row["target_id"]
        station_id = row["matched_station_id"]

        # 获取该站点的时间范围（从 NWSS 数据中）
        site_info = nwss_sites[nwss_sites[site_id_col] == site_id].iloc[0]
        # 支持 date_min/date_max 或 start_date/end_date 列名
        start_date = str(
            site_info.get("start_date",
            site_info.get("date_min", "2020-01-01"))
        )[:10]
        end_date = str(
            site_info.get("end_date",
            site_info.get("date_max", "2024-12-31"))
        )[:10]

        weather = download_ghcn_daily_data(station_id, start_date, end_date)
        if len(weather) > 0:
            weather["site_id"] = site_id
            weather["noaa_station_id"] = station_id
            weather["match_distance_km"] = row["distance_km"]
            all_weather.append(weather)

    if all_weather:
        result = pd.concat(all_weather, ignore_index=True)
        logger.info(f"NOAA 数据获取完成: {len(result)} 行, {result['site_id'].nunique()} 个站点")
        return result
    else:
        logger.error("未获取到任何 NOAA 数据")
        return pd.DataFrame()


def run_step3(nwss_sites: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    执行 Step 3 完整流程。

    Parameters
    ----------
    nwss_sites : NWSS 站点元数据（需包含经纬度）。
                 如果为 None，尝试从站点清单文件加载。
    """
    logger.info("=" * 60)
    logger.info("Step 3: NOAA 气象数据匹配与下载")
    logger.info("=" * 60)

    if nwss_sites is None:
        # 尝试从 tier1 文件加载
        tier1_path = PROJECT_ROOT / settings.paths.tier1_sites
        if tier1_path.exists():
            nwss_sites = pd.read_csv(tier1_path)
        else:
            raise FileNotFoundError("需要提供 NWSS 站点元数据（含经纬度）")

    df = match_and_download_noaa(nwss_sites)

    if len(df) > 0:
        output_path = PROJECT_ROOT / settings.paths.noaa_processed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"NOAA 数据保存至: {output_path}")

    return df


if __name__ == "__main__":
    run_step3()
