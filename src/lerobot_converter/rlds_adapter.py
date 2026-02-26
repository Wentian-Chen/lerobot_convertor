from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import json
from typing import Any

import draccus

from lerobot_converter.lerobot_target import LeRobotDatasetConverter
from lerobot_converter.models import ConversionOptions, NormalizedEpisode, NormalizedFrame


class RldsToLeRobotConverter(LeRobotDatasetConverter[Iterable[dict[str, Any]]]):
    """Thin RLDS adapter template.

    Users are expected to provide source episodes with this shape:
        {
            "episode_id": int,
            "steps": [
                {
                    "task": "optional-task",
                    "timestamp": 0.0,
                    "feature_values": { ... aligned with options.features ... }
                },
                ...
            ]
        }
    """

    def iter_source_episodes(
        self,
        source: Iterable[dict[str, Any]],
        _options: ConversionOptions,
    ) -> Iterable[dict[str, Any]]:
        for episode in source:
            yield episode

    def build_frame(
        self,
        source_frame: Any,
        options: ConversionOptions,
    ) -> NormalizedFrame:
        if not isinstance(source_frame, dict):
            raise TypeError("RLDS source frame must be a dictionary.")

        values = source_frame.get("feature_values")
        if not isinstance(values, dict):
            raise ValueError("RLDS source frame must include dict field 'feature_values'.")

        required_keys = set((options.features or {}).keys())
        missing_keys = required_keys - set(values.keys())
        if missing_keys:
            raise ValueError(f"Frame missing required feature keys: {sorted(missing_keys)}")

        task = source_frame.get("task", options.default_task)
        timestamp = source_frame.get("timestamp")
        return NormalizedFrame(task=str(task), feature_values=values, timestamp=timestamp)

    def build_episode(
        self,
        source_episode: Any,
        episode_index: int,
        options: ConversionOptions,
    ) -> NormalizedEpisode:
        if not isinstance(source_episode, dict):
            raise TypeError(f"Episode {episode_index} must be a dictionary.")
        steps = source_episode.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"Episode {episode_index} has invalid 'steps' value.")
        frames = tuple(self.build_frame(source_step, options) for source_step in steps)

        return NormalizedEpisode(
            episode_id=int(source_episode.get("episode_id", episode_index)),
            frames=frames,
        )


@dataclass
class RldsAdapterExampleConfig:
    source: str = ""
    output_dir: str = ""
    options: ConversionOptions = field(default_factory=ConversionOptions)


@draccus.wrap()
def run_rlds_adapter_example(cfg: RldsAdapterExampleConfig):
    adapter = RldsToLeRobotConverter()
    if not cfg.source:
        raise ValueError("Provide --source path to a RLDS JSON/JSONL file.")
    if not cfg.output_dir:
        raise ValueError("Provide --output_dir for converted dataset output.")
    source_path = cfg.source
    if source_path.endswith(".jsonl"):
        with open(source_path, encoding="utf-8") as file_obj:
            episodes = [json.loads(line) for line in file_obj if line.strip()]
    else:
        with open(source_path, encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if isinstance(payload, list):
            episodes = payload
        elif isinstance(payload, dict) and "episodes" in payload and isinstance(payload["episodes"], list):
            episodes = payload["episodes"]
        else:
            raise ValueError("RLDS example expects list payload or {'episodes': [...]} payload.")

    if cfg.options.features is None:
        raise ValueError("Provide --options.features and ensure each step contains 'feature_values'.")
    adapter.convert(episodes, cfg.output_dir, cfg.options)
    adapter.finalize_target()
