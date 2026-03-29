"""
异常检测 CLI 入口。

用法：
  python scripts/run_detection.py                          # 使用默认配置
  python scripts/run_detection.py --min-votes 2            # 设置最小投票数
  python scripts/run_detection.py --output events.csv      # 指定输出文件
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
from loguru import logger

from src.config import settings, PROJECT_ROOT, ensure_dirs
from src.anomaly_detection.ensemble import EnsembleDetector
from src.anomaly_detection.events import generate_event_catalog
from src.labeling.auto_label import auto_label_batch


@click.command()
@click.option("--min-votes", default=None, type=int, help="Ensemble 最小投票数")
@click.option("--output", default=None, help="事件目录输出路径")
def main(min_votes: int | None, output: str | None):
    """运行异常检测并生成事件目录。"""
    ensure_dirs()

    # 加载处理后的 NWSS 数据
    nwss_path = PROJECT_ROOT / settings.paths.nwss_processed
    if not nwss_path.exists():
        logger.error(f"NWSS 处理数据不存在: {nwss_path}，请先运行 pipeline step 1")
        return

    df = pd.read_parquet(nwss_path)
    logger.info(f"加载 NWSS 数据: {len(df)} 行, {df['key_plot_id'].nunique()} 个站点")

    # 确保列名一致
    if "key_plot_id" in df.columns and "site_id" not in df.columns:
        df["site_id"] = df["key_plot_id"]

    # 运行 Ensemble 检测
    votes = min_votes or settings.anomaly_detection.ensemble_min_votes
    detector = EnsembleDetector(min_votes=votes)

    value_col = "pcr_conc_lin_log1p"
    if value_col not in df.columns:
        logger.error(f"缺少列 {value_col}，请检查 NWSS 预处理是否完成")
        return

    detection_df = detector.detect_all_sites(df, value_col=value_col)

    # 生成事件目录
    output_path = output or str(
        PROJECT_ROOT / settings.paths.outputs_dir / "anomaly_event_catalog.csv"
    )
    catalog = generate_event_catalog(detection_df, output_path=output_path,
                                     zscore_col="rolling_zscore_7d")

    # Phase 2: 自动预标注
    if len(catalog) > 0:
        merged_path = PROJECT_ROOT / settings.paths.merged_database
        if merged_path.exists():
            merged_db = pd.read_parquet(merged_path)
            if "key_plot_id" in merged_db.columns and "site_id" not in merged_db.columns:
                merged_db["site_id"] = merged_db["key_plot_id"]
            labeled = auto_label_batch(catalog, merged_db)
            labeled_path = str(
                PROJECT_ROOT / settings.paths.labeled_data_dir / "auto_labeled_events.csv"
            )
            Path(labeled_path).parent.mkdir(parents=True, exist_ok=True)
            labeled.to_csv(labeled_path, index=False)
            logger.info(f"自动预标注保存至: {labeled_path}")
        else:
            logger.warning("合并数据库不存在，跳过自动预标注")

    logger.info(f"异常检测完成，共 {len(catalog)} 个事件")


if __name__ == "__main__":
    main()
