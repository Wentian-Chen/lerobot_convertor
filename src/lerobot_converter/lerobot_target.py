from __future__ import annotations

from pathlib import Path
from typing import Generic

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_converter.base import BaseDatasetConverter, TSource
from lerobot_converter.models import ConversionOptions, ConversionReport, NormalizedEpisode, NormalizedFrame


class LeRobotDatasetConverter(BaseDatasetConverter[TSource], Generic[TSource]):
    """Episode-oriented target writer backed by official LeRobotDataset APIs."""

    def __init__(self) -> None:
        self._dataset: LeRobotDataset | None = None
        self._output_dir: Path | None = None
        self._options: ConversionOptions | None = None
        self._feature_keys: set[str] = set()
        self._episode_count: int = 0
        self._frame_count: int = 0

    def initialize_target(
        self,
        output_dir: Path,
        options: ConversionOptions,
    ) -> None:
        if self._dataset is not None:
            raise RuntimeError("Target dataset has already been initialized.")
        if options.features is None:
            raise ValueError("ConversionOptions.features is required to initialize LeRobotDataset.")

        feature_spec = dict(options.features)

        self._dataset = LeRobotDataset.create(
            repo_id=options.dataset_name,
            root=output_dir,
            fps=options.fps,
            features=feature_spec,
            robot_type=options.robot_type,
            use_videos=options.use_videos,
        )
        self._output_dir = output_dir
        self._options = options
        self._feature_keys = set(feature_spec.keys())

    def convert(
        self,
        source: TSource,
        output_dir: str | Path,
        options: ConversionOptions | None = None,
    ) -> ConversionReport:
        runtime_options = options or ConversionOptions()
        target_path = Path(output_dir)

        if self._dataset is None:
            self.initialize_target(output_dir=target_path, options=runtime_options)

        source_episodes = self.iter_source_episodes(source, runtime_options)
        has_episode = False
        for source_episode_index, source_episode in enumerate(source_episodes):
            has_episode = True
            episode = self.build_episode(source_episode, source_episode_index, runtime_options)
            self.convert_episode(episode)

        if not has_episode:
            raise ValueError("No episode data found in source dataset.")

        return self._build_report()

    def convert_episode(self, episode: NormalizedEpisode) -> int:
        dataset = self._require_dataset()
        options = self._require_options()
        self._validate_episode(episode)

        dataset.episode_buffer = dataset.create_episode_buffer(episode_index=dataset.meta.total_episodes)
        for frame in episode.frames:
            self._validate_frame_keys(frame)
            dataset.add_frame(self._to_lerobot_frame(frame, options))
            self._frame_count += 1
        dataset.save_episode()
        self._episode_count += 1
        return len(episode.frames)

    def finalize_target(self) -> ConversionReport:
        dataset = self._require_dataset()
        output_dir = self._require_output_dir()
        options = self._require_options()

        dataset.finalize()
        report = ConversionReport(
            output_dir=output_dir,
            episode_count=self._episode_count,
            frame_count=self._frame_count,
            repo_id=options.dataset_name,
        )
        self._reset_runtime_state()
        return report

    def _validate_episode(self, episode: NormalizedEpisode) -> None:
        if episode.episode_id < 0:
            raise ValueError("episode_id must be greater than or equal to 0.")
        if not episode.frames:
            raise ValueError(f"Episode {episode.episode_id} does not contain any steps.")

    @staticmethod
    def _to_lerobot_frame(frame: NormalizedFrame, options: ConversionOptions) -> dict:
        values = dict(frame.feature_values)
        task = values.pop(options.task_key, frame.task or options.default_task)
        values.pop(options.timestamp_key, None)
        return {**values, "task": task}

    def _validate_frame_keys(self, frame: NormalizedFrame) -> None:
        unknown_keys = set(frame.feature_values.keys()) - self._feature_keys
        if unknown_keys:
            raise ValueError(f"Frame contains unknown feature keys: {sorted(unknown_keys)}")

    def _require_dataset(self) -> LeRobotDataset:
        if self._dataset is None:
            raise RuntimeError("Target dataset is not initialized. Call initialize_target() first.")
        return self._dataset

    def _require_output_dir(self) -> Path:
        if self._output_dir is None:
            raise RuntimeError("Target output directory is not initialized.")
        return self._output_dir

    def _require_options(self) -> ConversionOptions:
        if self._options is None:
            raise RuntimeError("Conversion options are not initialized.")
        return self._options

    def _build_report(self) -> ConversionReport:
        output_dir = self._require_output_dir()
        options = self._require_options()
        return ConversionReport(
            output_dir=output_dir,
            episode_count=self._episode_count,
            frame_count=self._frame_count,
            repo_id=options.dataset_name,
        )

    def _reset_runtime_state(self) -> None:
        self._dataset = None
        self._output_dir = None
        self._options = None
        self._feature_keys = set()
        self._episode_count = 0
        self._frame_count = 0


LeRobotDatasetConvertor = LeRobotDatasetConverter
