"""
从 CDC NWSS 的另一个数据集获取站点坐标信息。
CDC 有一个 NWSS 站点元数据 API 可以查询。
如果不行，用州首府坐标作为近似值。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests
from loguru import logger

# 州首府坐标（作为 fallback）
STATE_COORDS = {
    "MT": (46.5958, -112.0270),  # Helena
    "OR": (44.9429, -123.0351),  # Salem
    "NV": (39.1638, -119.7674),  # Carson City
    "WI": (43.0747, -89.3841),   # Madison
    "IL": (39.7984, -89.6544),   # Springfield
    "MN": (44.9553, -93.1022),   # St. Paul
    "GA": (33.7490, -84.3880),   # Atlanta
    "NY": (42.6526, -73.7562),   # Albany
    "PA": (40.2732, -76.8867),   # Harrisburg
    "NC": (35.7796, -78.6382),   # Raleigh
    "FL": (30.4383, -84.2807),   # Tallahassee
    "TX": (30.2672, -97.7431),   # Austin
    "WY": (41.1400, -104.8202),  # Cheyenne
    "AZ": (33.4484, -112.0740),  # Phoenix
    "WA": (47.0379, -122.9007),  # Olympia
    "NM": (35.6870, -105.9378),  # Santa Fe
}

# 尝试从 CDC SODA API 获取站点坐标
# NWSS 有一个 wastewater monitoring site 数据集
CDC_SITES_URL = "https://data.cdc.gov/resource/2ew6-ywp6.json"


def try_cdc_api():
    """尝试从 CDC API 获取站点坐标。"""
    logger.info("尝试从 CDC API 获取站点坐标...")
    try:
        resp = requests.get(CDC_SITES_URL, params={"$limit": 5000}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data:
            df = pd.DataFrame(data)
            logger.info(f"CDC API 返回 {len(df)} 条记录")
            logger.info(f"列: {list(df.columns)}")
            logger.info(f"前 3 行:\n{df.head(3).to_string()}")
            return df
    except Exception as e:
        logger.warning(f"CDC API 失败: {e}")
    return None


def assign_coords_to_sites():
    """给站点清单添加坐标。"""
    sites_path = Path("outputs/tier1_core_final_sites.csv")
    if not sites_path.exists():
        logger.error("站点清单不存在")
        return

    sites = pd.read_csv(sites_path)
    logger.info(f"加载站点清单: {len(sites)} 个站点")

    # 先尝试 CDC API
    cdc_data = try_cdc_api()

    # 用州首府坐标作为近似
    if "state" in sites.columns:
        sites["latitude"] = sites["state"].map(lambda s: STATE_COORDS.get(s, (0, 0))[0])
        sites["longitude"] = sites["state"].map(lambda s: STATE_COORDS.get(s, (0, 0))[1])
        logger.info("使用州首府坐标作为近似值")

    # 保存更新后的站点清单
    sites.to_csv(sites_path, index=False)
    logger.info(f"更新站点清单: {sites_path}")
    logger.info(f"\n{sites[['key_plot_id', 'state', 'latitude', 'longitude']].to_string()}")

    # 同样更新 tier2 和 tier3
    for tier_file in ["tier2_extension_sites.csv", "tier3_challenge_sites.csv"]:
        tier_path = Path("outputs") / tier_file
        if tier_path.exists():
            tier_df = pd.read_csv(tier_path)
            if "state" in tier_df.columns:
                tier_df["latitude"] = tier_df["state"].map(lambda s: STATE_COORDS.get(s, (0, 0))[0])
                tier_df["longitude"] = tier_df["state"].map(lambda s: STATE_COORDS.get(s, (0, 0))[1])
                tier_df.to_csv(tier_path, index=False)
                logger.info(f"更新 {tier_file}")


if __name__ == "__main__":
    assign_coords_to_sites()
