"""
python lerobot_converter/scripts/aggregate_lerobot_datasets.py \
    --repo_ids "[miku112/piper_pick_banana_50,miku112/piper_pick_banana_100_new]" \
    --aggr_repo_id miku112/piper_pick_banana_100 \
    --roots "[datasets/lerobot/miku112/piper_pick_banana_50,datasets/lerobot/miku112/piper_pick_banana_100_new]" \
    --aggr_root datasets/lerobot/miku112/piper_pick_banana_100
"""
from dataclasses import dataclass, field
from pathlib import Path
import logging
import sys

import draccus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# 兼容两种使用方式：
# 1) 已通过 pip 安装 lerobot
# 2) 在同一工作区中直接使用 ../lerobot/src
WORKSPACE_ROOT = PROJECT_ROOT.parent
LEROBOT_SRC_DIR = WORKSPACE_ROOT / "lerobot" / "src"
if str(LEROBOT_SRC_DIR) not in sys.path:
    sys.path.append(str(LEROBOT_SRC_DIR))

from lerobot.datasets.aggregate import aggregate_datasets


@dataclass
class AggregateLeRobotDatasetsConfig:
    """将多个 LeRobot 数据集合并为一个数据集的配置。"""

    # 输入数据集 repo_id 列表，例如 ["user/ds_a", "user/ds_b"]
    repo_ids: list[str] = field(default_factory=list)

    # 合并后数据集的 repo_id，例如 "user/ds_merged"
    aggr_repo_id: str = ""

    # 每个源数据集本地根目录；为空时由 LeRobot 默认机制解析
    roots: list[str] = field(default_factory=list)

    # 合并数据集输出根目录；为空时由 LeRobot 默认机制解析
    aggr_root: str = ""

    # 可选：单个 data parquet 文件大小上限（MB）
    data_files_size_in_mb: float | None = None

    # 可选：单个 video 文件大小上限（MB）
    video_files_size_in_mb: float | None = None

    # 可选：每个 chunk 的最大文件数
    chunk_size: int | None = None

    # 日志级别：DEBUG/INFO/WARNING/ERROR
    log_level: str = "INFO"


@draccus.wrap()
def main(cfg: AggregateLeRobotDatasetsConfig):
    if not cfg.repo_ids:
        raise ValueError("请提供 --repo_ids，至少包含一个源数据集。")
    if not cfg.aggr_repo_id:
        raise ValueError("请提供 --aggr_repo_id。")

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    roots: list[Path] | None = None
    if cfg.roots:
        if len(cfg.roots) != len(cfg.repo_ids):
            raise ValueError(
                f"roots 数量({len(cfg.roots)})必须与 repo_ids 数量({len(cfg.repo_ids)})一致。"
            )
        roots = [Path(p) for p in cfg.roots]

    aggr_root = Path(cfg.aggr_root) if cfg.aggr_root else None

    aggregate_datasets(
        repo_ids=cfg.repo_ids,
        aggr_repo_id=cfg.aggr_repo_id,
        roots=roots,
        aggr_root=aggr_root,
        data_files_size_in_mb=cfg.data_files_size_in_mb,
        video_files_size_in_mb=cfg.video_files_size_in_mb,
        chunk_size=cfg.chunk_size,
    )


if __name__ == "__main__":
    main()
