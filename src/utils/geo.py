"""
地理空间工具函数。

提供 Haversine 距离计算和最近站点匹配功能，
用于 NWSS→NOAA 和 NWSS→USGS 的空间匹配。
"""

import numpy as np
import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两个经纬度点之间的 Haversine 距离（公里）。

    Parameters
    ----------
    lat1, lon1 : 点1的纬度和经度（十进制度）
    lat2, lon2 : 点2的纬度和经度（十进制度）

    Returns
    -------
    float : 两点间的距离（公里）
    """
    R = 6371.0  # 地球平均半径 km

    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def find_nearest_station(
    target_lat: float,
    target_lon: float,
    stations_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: str = "station_id",
    max_distance_km: float = 50.0,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    从候选站点表中找到距离目标点最近的 N 个站点。

    Parameters
    ----------
    target_lat, target_lon : 目标点坐标
    stations_df : 候选站点 DataFrame，需包含经纬度列
    lat_col, lon_col, id_col : 列名
    max_distance_km : 最大搜索半径
    top_n : 返回最近的 N 个站点

    Returns
    -------
    pd.DataFrame : 最近的站点，附加 distance_km 列，按距离升序排列
    """
    distances = stations_df.apply(
        lambda row: haversine_km(
            target_lat, target_lon, row[lat_col], row[lon_col]
        ),
        axis=1,
    )
    result = stations_df.copy()
    result["distance_km"] = distances
    result = result[result["distance_km"] <= max_distance_km]
    result = result.nsmallest(top_n, "distance_km")
    return result.reset_index(drop=True)


def batch_spatial_match(
    targets_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    target_lat_col: str = "latitude",
    target_lon_col: str = "longitude",
    target_id_col: str = "site_id",
    station_lat_col: str = "latitude",
    station_lon_col: str = "longitude",
    station_id_col: str = "station_id",
    max_distance_km: float = 50.0,
) -> pd.DataFrame:
    """
    批量空间匹配：对每个目标站点找到最近的候选站点。

    Returns
    -------
    pd.DataFrame : 匹配结果表，列包含 target_id, matched_station_id, distance_km
    """
    matches = []
    for _, target in targets_df.iterrows():
        nearest = find_nearest_station(
            target[target_lat_col],
            target[target_lon_col],
            stations_df,
            lat_col=station_lat_col,
            lon_col=station_lon_col,
            id_col=station_id_col,
            max_distance_km=max_distance_km,
            top_n=1,
        )
        if len(nearest) > 0:
            matches.append(
                {
                    "target_id": target[target_id_col],
                    "matched_station_id": nearest.iloc[0][station_id_col],
                    "distance_km": nearest.iloc[0]["distance_km"],
                }
            )
        else:
            matches.append(
                {
                    "target_id": target[target_id_col],
                    "matched_station_id": None,
                    "distance_km": None,
                }
            )
    return pd.DataFrame(matches)
