"""
数据管线 CLI 入口。

用法：
  python scripts/run_pipeline.py --step 1        # 运行 Step 1 (NWSS)
  python scripts/run_pipeline.py --step 1-3      # 运行 Step 1 到 3
  python scripts/run_pipeline.py --step all       # 运行所有步骤
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
from loguru import logger

from src.config import ensure_dirs


@click.command()
@click.option("--step", default="all", help="要运行的步骤: 1, 2, ..., 7, 1-3, all")
def main(step: str):
    """运行数据管线。"""
    ensure_dirs()

    # 解析步骤范围
    if step == "all":
        steps = list(range(1, 8))
    elif "-" in step:
        start, end = step.split("-")
        steps = list(range(int(start), int(end) + 1))
    else:
        steps = [int(step)]

    logger.info(f"运行数据管线步骤: {steps}")

    for s in steps:
        try:
            if s == 1:
                from src.data_pipeline.step1_nwss import run_step1
                run_step1()
            elif s == 2:
                from src.data_pipeline.step2_hhs import run_step2
                run_step2()
            elif s == 3:
                from src.data_pipeline.step3_noaa import run_step3
                run_step3()
            elif s == 4:
                from src.data_pipeline.step4_usgs import run_step4
                run_step4()
            elif s == 5:
                from src.data_pipeline.step5_variants import run_step5
                run_step5()
            elif s == 6:
                from src.data_pipeline.step6_merge import run_step6
                run_step6()
            elif s == 7:
                from src.data_pipeline.step7_quality import run_step7
                run_step7()
            else:
                logger.warning(f"未知步骤: {s}")
        except Exception as e:
            logger.error(f"Step {s} 失败: {e}")
            raise


if __name__ == "__main__":
    main()
