from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class NormalizedFrame:
    """Normalized frame aligned to LeRobot `add_frame` payload semantics.

    `feature_values` keys should match LeRobot feature keys, such as:
    - `action`
    - `observation.state`
    - `observation.images.front`
    - `next.reward`
    - `next.done`
    """

    task: str | None = None
    feature_values: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None


@dataclass(slots=True, frozen=True)
class NormalizedEpisode:
    """Normalized episode made of ordered frames."""

    episode_id: int
    frames: tuple[NormalizedFrame, ...]


@dataclass(slots=True, frozen=True)
class ConversionOptions:
    """Runtime options shared by all converters."""

    dataset_name: str = "converted_dataset"
    fps: int = 30
    robot_type: str | None = None
    use_videos: bool = True
    features: dict[str, dict[str, Any]] | None = None
    default_task: str = "unknown"
    augment_task_instruction: bool = False
    task_key: str = "task"
    timestamp_key: str = "timestamp"


@dataclass
class DatasetsConverterConfig:
    """Standard runtime config for LeRobot converter scripts."""

    source: str = ""
    output_dir: str = ""
    dataset_name: str = ""
    fps: int = 30
    robot_type: str | None = None
    use_videos: bool = True
    features: dict[str, dict[str, Any]] | None = None
    default_task: str = "unknown"
    augment_task_instruction: bool = False
    task_key: str = "task"
    timestamp_key: str = "timestamp"

    @property
    def options(self) -> ConversionOptions:
        valid_keys = {item.name for item in fields(ConversionOptions)}
        kwargs = {key: getattr(self, key) for key in valid_keys if hasattr(self, key)}
        return ConversionOptions(**kwargs)



@dataclass(slots=True, frozen=True)
class ConversionReport:
    """Summary returned after conversion is completed."""

    output_dir: Path
    episode_count: int
    frame_count: int
    repo_id: str
