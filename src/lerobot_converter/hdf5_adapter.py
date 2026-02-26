from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import draccus
from lerobot_converter.lerobot_target import LeRobotDatasetConverter
from lerobot_converter.models import ConversionOptions, NormalizedEpisode, NormalizedFrame


class Hdf5ToLeRobotConverter(LeRobotDatasetConverter[str | Path]):
    """Thin HDF5 adapter template.

    Design intent:
    - target writing lifecycle is handled by ``LeRobotDatasetConverter``;
    - source extraction and feature alignment are implemented by users in
      ``extract_episode_from_file``;
    - every step should be returned in an adapter-neutral frame schema:

      {
          "task": "optional-task",
          "timestamp": 0.0,  # optional
          "feature_values": {
              "action": ...,
              "observation.state": ...,
              ...
          }
      }
    """

    def iter_source_episodes(
        self,
        source: str | Path,
        options: ConversionOptions,
    ) -> Iterable[dict[str, Any]]:
        source_path = Path(source)
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for HDF5 conversion.") from exc

        file_paths = self._resolve_hdf5_files(source_path)
        if not file_paths:
            raise FileNotFoundError(f"No HDF5 files found under source path: {source_path}")

        def _generator() -> Iterable[dict[str, Any]]:
            for fallback_episode_id, file_path in enumerate(file_paths):
                with h5py.File(file_path, "r") as file_obj:
                    yield self.extract_episode_from_file(
                        file_obj=file_obj,
                        file_path=file_path,
                        episode_id=fallback_episode_id,
                        options=options,
                    )

        return _generator()

    def extract_episode_from_file(
        self,
        file_obj: Any,
        file_path: Path,
        episode_id: int,
        options: ConversionOptions,
    ) -> dict[str, Any]:
        """Extract one source episode from one opened HDF5 file.

        Users should override this to read custom HDF5 layouts and return:
            {"episode_id": int, "steps": list[dict]}

        Each step dict should at least contain ``feature_values`` aligned with
        ``options.features`` keys.
        """
        raise NotImplementedError(
            "Override extract_episode_from_file() in your custom adapter to read your HDF5 schema."
        )

    def build_frame(
        self,
        source_frame: Any,
        options: ConversionOptions,
    ) -> NormalizedFrame:
        if not isinstance(source_frame, dict):
            raise TypeError("HDF5 source frame must be a dictionary.")

        values = source_frame.get("feature_values")
        if not isinstance(values, dict):
            raise ValueError("HDF5 source frame must include dict field 'feature_values'.")

        required_keys = set((options.features or {}).keys())
        missing_keys = required_keys - set(values.keys())
        if missing_keys:
            raise ValueError(f"Frame missing required feature keys: {sorted(missing_keys)}")

        return NormalizedFrame(
            task=str(source_frame.get("task", options.default_task)),
            feature_values=values,
            timestamp=source_frame.get("timestamp"),
        )

    def build_episode(
        self,
        source_episode: Any,
        episode_index: int,
        options: ConversionOptions,
    ) -> NormalizedEpisode:
        if not isinstance(source_episode, dict):
            raise TypeError(f"Episode {episode_index} must be a dictionary.")
        source_steps = source_episode.get("steps")
        if not isinstance(source_steps, list):
            raise ValueError(f"Episode {episode_index} has invalid steps structure.")

        frames = tuple(self.build_frame(source_step, options) for source_step in source_steps)
        return NormalizedEpisode(
            episode_id=int(source_episode.get("episode_id", episode_index)),
            frames=frames,
        )

    @staticmethod
    def _resolve_hdf5_files(source_path: Path) -> list[Path]:
        if not source_path.exists():
            raise FileNotFoundError(f"HDF5 source path not found: {source_path}")

        if source_path.is_file():
            if source_path.suffix.lower() not in {".h5", ".hdf5"}:
                raise ValueError(f"Source file is not HDF5: {source_path}")
            return [source_path]

        if not source_path.is_dir():
            raise ValueError(f"Unsupported source path type: {source_path}")

        return sorted(source_path.glob("*.h5")) + sorted(source_path.glob("*.hdf5"))


@dataclass
class Hdf5AdapterExampleConfig:
    source: str = ""
    output_dir: str = ""
    options: ConversionOptions = field(default_factory=ConversionOptions)


@draccus.wrap()
def run_hdf5_adapter_example(cfg: Hdf5AdapterExampleConfig):
    if not cfg.source:
        raise ValueError("Provide --source path to a HDF5 file.")
    if not cfg.output_dir:
        raise ValueError("Provide --output_dir for converted dataset output.")
    if cfg.options.features is None:
        raise ValueError("Provide --options.features.")
    raise NotImplementedError(
        "Hdf5ToLeRobotConverter is a template. Subclass it and implement extract_episode_from_file()."
    )

