"""
Agent 调查 CLI 入口。

用法：
  python scripts/run_agent.py                              # 调查所有事件
  python scripts/run_agent.py --max-events 10              # 限制事件数
  python scripts/run_agent.py --event-id EVT-00001         # 调查单个事件
  python scripts/run_agent.py --events-file catalog.csv    # 指定事件文件
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import pandas as pd
from loguru import logger

from src.config import settings, PROJECT_ROOT, ensure_dirs
from src.agent.runner import InvestigationAgent


@click.command()
@click.option("--max-events", default=None, type=int, help="最大调查事件数")
@click.option("--event-id", default=None, help="调查单个事件 ID")
@click.option("--events-file", default=None, help="事件目录 CSV 路径")
@click.option("--model", default=None, help="LLM 模型名称")
@click.option("--api-base", default=None, help="自定义 API base URL")
@click.option("--api-key", default=None, help="自定义 API key")
@click.option("--output-file", default=None, help="输出文件路径")
def main(max_events: int | None, event_id: str | None, events_file: str | None, model: str | None, api_base: str | None, api_key: str | None, output_file: str | None):
    """运行 Agent 调查异常事件。"""
    ensure_dirs()

    # 加载合并数据库
    db_path = PROJECT_ROOT / settings.paths.merged_database
    if not db_path.exists():
        logger.error(f"合并数据库不存在: {db_path}，请先运行 pipeline step 1-6")
        return

    database = pd.read_parquet(db_path)
    database["date"] = pd.to_datetime(database["date"])
    logger.info(f"加载数据库: {len(database)} 行")

    # 加载站点元数据（合并所有 tier）
    tier_frames = []
    for tier_key in ["tier1_sites", "tier2_sites", "tier3_sites"]:
        tier_path = PROJECT_ROOT / getattr(settings.paths, tier_key)
        if tier_path.exists():
            df_tier = pd.read_csv(tier_path)
            tier_frames.append(df_tier)
            logger.info(f"加载 {tier_key}: {len(df_tier)} 个站点")
    site_meta = pd.concat(tier_frames, ignore_index=True) if tier_frames else pd.DataFrame()

    # 确保 site_id 列存在
    if "key_plot_id" in site_meta.columns and "site_id" not in site_meta.columns:
        site_meta["site_id"] = site_meta["key_plot_id"]
    if "key_plot_id" in database.columns and "site_id" not in database.columns:
        database["site_id"] = database["key_plot_id"]
    logger.info(f"站点元数据: {len(site_meta)} 个站点")

    # 初始化 Agent
    agent = InvestigationAgent(
        database=database,
        site_metadata=site_meta,
        model=model,
    )

    # Override API client if custom base/key provided
    if api_base or api_key:
        from openai import OpenAI
        agent.client = OpenAI(
            base_url=api_base or settings.agent.api_base,
            api_key=api_key or settings.agent.api_key,
            timeout=120.0,
            max_retries=2,
        )
        if model:
            agent.model = model
        logger.info(f"使用自定义 API: base={api_base}, model={agent.model}")

    # 加载事件目录
    if events_file:
        events_path = Path(events_file)
    else:
        events_path = PROJECT_ROOT / settings.paths.outputs_dir / "anomaly_event_catalog.csv"

    if not events_path.exists():
        logger.error(f"事件目录不存在: {events_path}，请先运行 run_detection.py")
        return

    events_df = pd.read_csv(events_path)
    logger.info(f"加载事件目录: {len(events_df)} 个事件")

    # 单事件调查
    if event_id:
        event = events_df[events_df["event_id"] == event_id]
        if event.empty:
            logger.error(f"事件 {event_id} 不存在")
            return
        events_df = event

    # 批量调查
    auto_labels = None
    auto_label_path = PROJECT_ROOT / settings.paths.labeled_data_dir / "auto_labeled_events.csv"
    if auto_label_path.exists():
        auto_labels = pd.read_csv(auto_label_path)
        logger.info(f"加载自动预标注: {len(auto_labels)} 个事件")

    silver_labels = None
    silver_label_path = PROJECT_ROOT / settings.paths.labeled_data_dir / "labeled_events.csv"
    if silver_label_path.exists():
        silver_labels = pd.read_csv(silver_label_path)
        logger.info(f"加载 silver labels: {len(silver_labels)} 个事件")

    output_path = Path(output_file) if output_file else None
    reports = agent.investigate_batch(
        events_df, max_events=max_events,
        auto_labels=auto_labels, silver_labels=silver_labels,
        output_path=output_path,
    )
    logger.info(f"调查完成: {len(reports)} 个报告")

    # 打印摘要
    for r in reports:
        logger.info(f"  {r.summary()}")


if __name__ == "__main__":
    main()
