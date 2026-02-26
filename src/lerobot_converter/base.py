from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Generic, TypeVar

from lerobot_converter.models import ConversionOptions, ConversionReport, NormalizedEpisode, NormalizedFrame

TSource = TypeVar("TSource")


class BaseDatasetConverter(ABC, Generic[TSource]):
    """Base abstraction for source->target dataset conversion pipelines."""

    @abstractmethod
    def convert(
        self,
        source: TSource,
        output_dir: str | Path,
        options: ConversionOptions | None = None,
    ) -> ConversionReport:
        """Convert source dataset into target format and persist it.

        Args:
            source: Source dataset locator or descriptor.
            output_dir: Output directory for converted dataset.
            options: Optional conversion runtime options.

        Returns:
            Conversion report with output details.
        """
        raise NotImplementedError

    @abstractmethod
    def iter_source_episodes(
        self,
        source: TSource,
        options: ConversionOptions,
    ) -> Iterable[Any]:
        """Read source dataset and yield source episodes one by one."""

    @abstractmethod
    def build_frame(
        self,
        source_frame: Any,
        options: ConversionOptions,
    ) -> NormalizedFrame:
        """Build one normalized frame from source-specific frame data."""

    @abstractmethod
    def build_episode(
        self,
        source_episode: Any,
        episode_index: int,
        options: ConversionOptions,
    ) -> NormalizedEpisode:
        """Build one normalized episode from source-specific episode data."""

