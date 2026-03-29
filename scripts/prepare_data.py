"""
数据准备脚本：下载原始数据 + 探索 + 生成站点清单。

用法：
  python scripts/prepare_data.py --download          # 下载原始数据
  python scripts/prepare_data.py --explore            # 探索 NWSS 数据结构
  python scripts/prepare_data.py --select-sites       # 筛选站点并生成清单
  python scripts/prepare_data.py --all                # 全部执行
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import requests
import pandas as pd
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ── 数据源 URL ──
NWSS_URL = "https://data.cdc.gov/api/views/g653-rqe2/rows.csv?accessType=DOWNLOAD"
HHS_URL = "https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD"


# 目标州和气候带配置（来自 TASK_summary.md 的站点选择结果）
SITE_CONFIG = {
    # Tier 1 Core - Train (>700天重叠)
    "train": {
        "MT": "continental_north",
        "OR": "marine_west",
        "NV": "arid_semiarid",
        "WI": "continental_north",
        "IL": "continental_central",
        "MN": "continental_north",
        "GA": "subtropical_south",
        "NY": "continental_central",
    },
    # Tier 1 Core - Test (250-700天)
    "test": {
        "PA": "continental_central",
        "NC": "subtropical_south",
        "FL": "subtropical_south",
    },
    # Tier 2 Extension - Test
    "extension_test": {
        "TX": "subtropical_south",
        "WY": "continental_north",
        "AZ": "arid_semiarid",
        "WA": "marine_west",
    },
    # Tier 3 Challenge
    "phase2_only": {
        "NM": "arid_semiarid",
    },
}

ALL_STATES = []
for group in SITE_CONFIG.values():
    ALL_STATES.extend(group.keys())


def download_file(url: str, dest: Path, description: str) -> bool:
    """下载文件，支持大文件流式下载。"""
    if dest.exists():
        size_mb = dest.stat().st_size / 1024 / 1024
        logger.info(f"{description} 已存在: {dest} ({size_mb:.1f} MB)")
        return True

    logger.info(f"下载 {description}: {url}")
    logger.info("  这可能需要几分钟，请耐心等待...")

    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192 * 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (10 * 1024 * 1024) < 8192 * 16:
                    pct = downloaded / total * 100
                    logger.info(f"  {downloaded / 1024 / 1024:.0f} MB / {total / 1024 / 1024:.0f} MB ({pct:.0f}%)")

        size_mb = dest.stat().st_size / 1024 / 1024
        logger.info(f"  下载完成: {size_mb:.1f} MB")
        return True
    except Exception as e:
        logger.error(f"  下载失败: {e}")
        if dest.exists():
            dest.unlink()
        return False


def do_download():
    """下载所有原始数据。"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    nwss_path = RAW_DIR / "NWSS_Public_SARS-CoV-2_Concentration_in_Wastewater_Data.csv"
    download_file(NWSS_URL, nwss_path, "NWSS 污水数据")

    hhs_path = RAW_DIR / "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State.csv"
    download_file(HHS_URL, hhs_path, "HHS 住院数据")


def do_explore():
    """探索 NWSS 数据结构。"""
    nwss_path = RAW_DIR / "NWSS_Public_SARS-CoV-2_Concentration_in_Wastewater_Data.csv"
    if not nwss_path.exists():
        logger.error(f"NWSS 文件不存在: {nwss_path}，请先运行 --download")
        return

    logger.info("探索 NWSS 数据结构...")

    # 先读取列名和前几行
    sample = pd.read_csv(nwss_path, nrows=5)
    logger.info(f"列名: {list(sample.columns)}")
    logger.info(f"前 5 行:\n{sample.to_string()}")

    # 读取全量数据的基本统计（只读关键列）
    logger.info("加载全量数据（仅关键列）...")
    cols_to_try = ["key_plot_id", "date", "pcr_conc_lin", "normalization"]
    actual_cols = [c for c in cols_to_try if c in sample.columns]

    if not actual_cols:
        logger.warning(f"列名不匹配，实际列: {list(sample.columns)}")
        # 尝试读全部
        df = pd.read_csv(nwss_path, low_memory=False)
    else:
        df = pd.read_csv(nwss_path, usecols=actual_cols, low_memory=False)

    logger.info(f"总行数: {len(df)}")
    logger.info(f"总站点数: {df['key_plot_id'].nunique() if 'key_plot_id' in df.columns else 'N/A'}")

    if "normalization" in df.columns:
        logger.info(f"normalization 分布:\n{df['normalization'].value_counts()}")

    if "key_plot_id" in df.columns:
        # 提取州信息（key_plot_id 通常包含州缩写）
        df["state_guess"] = df["key_plot_id"].str.extract(r'_([A-Z]{2})_', expand=False)
        if df["state_guess"].notna().any():
            logger.info(f"按州分布 (前20):\n{df['state_guess'].value_counts().head(20)}")

    return df


def do_select_sites():
    """
    从 NWSS 数据中筛选站点并生成三层清单。

    筛选逻辑：
    1. 只保留 normalization == "flow-population" 的记录
    2. 对每个目标州，找到数据量最大、质量最好的站点
    3. 按 Tier 1/2/3 分组输出
    """
    nwss_path = RAW_DIR / "NWSS_Public_SARS-CoV-2_Concentration_in_Wastewater_Data.csv"
    if not nwss_path.exists():
        logger.error(f"NWSS 文件不存在，请先运行 --download")
        return

    logger.info("加载 NWSS 数据进行站点筛选...")
    df = pd.read_csv(nwss_path, low_memory=False)
    logger.info(f"原始数据: {len(df)} 行, 列: {list(df.columns)}")

    # 过滤 flow-population
    if "normalization" in df.columns:
        df = df[df["normalization"] == "flow-population"]
        logger.info(f"flow-population 过滤后: {len(df)} 行")

    # 日期解析
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 提取州信息
    # key_plot_id 格式:
    #   CDC_VERILY_xx_... → 州在 split("_")[2]
    #   NWSS_xx_...       → 州在 split("_")[1]
    if "key_plot_id" in df.columns:
        def extract_state(kpid: str) -> str:
            parts = kpid.split("_")
            if parts[0] == "CDC" and len(parts) > 2:
                return parts[2].upper()
            elif parts[0] == "NWSS" and len(parts) > 1:
                return parts[1].upper()
            return ""
        df["state"] = df["key_plot_id"].apply(extract_state)

    if "state" not in df.columns or df["state"].eq("").all():
        logger.error("无法从 key_plot_id 提取州信息")
        if "key_plot_id" in df.columns:
            logger.info(f"key_plot_id 样本:\n{df['key_plot_id'].head(20).tolist()}")
        return

    # 只保留目标州
    df = df[df["state"].isin(ALL_STATES)]
    logger.info(f"目标州过滤后: {len(df)} 行, {df['state'].nunique()} 个州")

    # 逐站点统计
    site_stats = df.groupby("key_plot_id").agg(
        state=("state", "first"),
        n_obs=("pcr_conc_lin", "count"),
        n_valid=("pcr_conc_lin", lambda x: x.notna().sum()),
        date_min=("date", "min"),
        date_max=("date", "max"),
        conc_mean=("pcr_conc_lin", "mean"),
        conc_std=("pcr_conc_lin", "std"),
        conc_median=("pcr_conc_lin", "median"),
    ).reset_index()

    site_stats["date_span_days"] = (site_stats["date_max"] - site_stats["date_min"]).dt.days
    site_stats["missing_pct"] = (1 - site_stats["n_valid"] / site_stats["n_obs"]) * 100
    site_stats["cv"] = site_stats["conc_std"] / site_stats["conc_mean"]

    logger.info(f"站点统计:\n{site_stats.sort_values('n_obs', ascending=False).head(30).to_string()}")

    # 对每个州选最佳站点（数据量最大 + 时间跨度最长）
    site_stats["quality_score"] = (
        site_stats["n_valid"].rank(pct=True) * 0.4
        + site_stats["date_span_days"].rank(pct=True) * 0.4
        + (1 - site_stats["missing_pct"].rank(pct=True)) * 0.2
    )

    best_per_state = site_stats.sort_values("quality_score", ascending=False).groupby("state").first().reset_index()
    logger.info(f"每州最佳站点:\n{best_per_state[['key_plot_id', 'state', 'n_obs', 'date_span_days', 'quality_score']].to_string()}")

    # 分配到 Tier
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    tier1_records = []
    tier2_records = []
    tier3_records = []

    for _, row in best_per_state.iterrows():
        state = row["state"]
        record = row.to_dict()

        # 确定气候带和角色
        for role, state_map in SITE_CONFIG.items():
            if state in state_map:
                record["climate_bucket"] = state_map[state]
                record["modeling_role"] = role
                break

        if state in SITE_CONFIG["train"] or state in SITE_CONFIG["test"]:
            tier1_records.append(record)
        elif state in SITE_CONFIG["extension_test"]:
            tier2_records.append(record)
        elif state in SITE_CONFIG["phase2_only"]:
            tier3_records.append(record)

    # 保存
    for name, records in [
        ("tier1_core_final_sites", tier1_records),
        ("tier2_extension_sites", tier2_records),
        ("tier3_challenge_sites", tier3_records),
    ]:
        if records:
            out_df = pd.DataFrame(records)
            path = OUTPUTS_DIR / f"{name}.csv"
            out_df.to_csv(path, index=False)
            logger.info(f"保存 {name}: {len(out_df)} 个站点 → {path}")

    logger.info(f"\n站点筛选完成: Tier1={len(tier1_records)}, Tier2={len(tier2_records)}, Tier3={len(tier3_records)}")


def do_prepare_hhs():
    """从 HHS 原始数据中提取目标州的数据。"""
    hhs_path = RAW_DIR / "COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State.csv"
    if not hhs_path.exists():
        logger.error(f"HHS 文件不存在，请先运行 --download")
        return

    logger.info("加载 HHS 数据...")
    df = pd.read_csv(hhs_path, low_memory=False)
    logger.info(f"原始 HHS: {len(df)} 行, 列: {list(df.columns)}")

    # 过滤目标州
    if "state" in df.columns:
        df_filtered = df[df["state"].isin(ALL_STATES)]
    elif "provider_state" in df.columns:
        df_filtered = df[df["provider_state"].isin(ALL_STATES)]
        df_filtered = df_filtered.rename(columns={"provider_state": "state"})
    else:
        logger.error(f"找不到州列，可用列: {list(df.columns)}")
        return

    logger.info(f"目标州过滤后: {len(df_filtered)} 行")

    # 保存
    out_path = OUTPUTS_DIR / "hhs_16states.csv"
    df_filtered.to_csv(out_path, index=False)
    logger.info(f"HHS 数据保存至: {out_path}")


@click.command()
@click.option("--download", is_flag=True, help="下载原始数据")
@click.option("--explore", is_flag=True, help="探索 NWSS 数据")
@click.option("--select-sites", is_flag=True, help="筛选站点")
@click.option("--prepare-hhs", is_flag=True, help="准备 HHS 数据")
@click.option("--all", "run_all", is_flag=True, help="全部执行")
def main(download, explore, select_sites, prepare_hhs, run_all):
    """数据准备脚本。"""
    if run_all:
        download = explore = select_sites = prepare_hhs = True

    if download:
        do_download()
    if explore:
        do_explore()
    if select_sites:
        do_select_sites()
    if prepare_hhs:
        do_prepare_hhs()

    if not any([download, explore, select_sites, prepare_hhs]):
        logger.info("请指定操作: --download, --explore, --select-sites, --prepare-hhs, --all")


if __name__ == "__main__":
    main()
