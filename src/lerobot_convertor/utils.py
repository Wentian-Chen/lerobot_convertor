from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any


@dataclass(slots=True, frozen=True)
class Hdf5NodeInfo:
    """Single HDF5 tree node description."""

    path: str
    node_type: str
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    attrs: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class RldsSampleInfo:
    """Single sampled RLDS episode/step structure description."""

    episode_index: int
    keys: list[str]
    step_count: int | None
    step_keys: list[str]
    extras: dict[str, Any]


def inspect_hdf5_structure(
    source: str | Path,
    max_depth: int | None = None,
    include_attrs: bool = True,
) -> list[Hdf5NodeInfo]:
    """Traverse an HDF5 file and return flattened tree metadata.

    Args:
        source: Path to a single `.h5` / `.hdf5` file.
        max_depth: Optional depth limit (root depth = 0).
        include_attrs: Whether to collect node attributes.

    Returns:
        Flattened node list sorted by traversal order.
    """

    source_path = Path(source)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {source_path}")

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("inspect_hdf5_structure requires h5py.") from exc

    result: list[Hdf5NodeInfo] = []

    def _attrs_to_dict(h5_obj: Any) -> dict[str, Any] | None:
        if not include_attrs:
            return None
        attrs: dict[str, Any] = {}
        for key, value in h5_obj.attrs.items():
            attrs[str(key)] = _to_python(value)
        return attrs

    def _visit(name: str, obj: Any) -> None:
        depth = 0 if name == "" else len(name.split("/"))
        if max_depth is not None and depth > max_depth:
            return

        if isinstance(obj, h5py.Group):
            result.append(
                Hdf5NodeInfo(
                    path=f"/{name}" if name else "/",
                    node_type="group",
                    attrs=_attrs_to_dict(obj),
                )
            )
            return

        if isinstance(obj, h5py.Dataset):
            result.append(
                Hdf5NodeInfo(
                    path=f"/{name}",
                    node_type="dataset",
                    dtype=str(obj.dtype),
                    shape=tuple(obj.shape),
                    attrs=_attrs_to_dict(obj),
                )
            )

    with h5py.File(source_path, "r") as file_obj:
        result.append(Hdf5NodeInfo(path="/", node_type="file", attrs=_attrs_to_dict(file_obj)))
        file_obj.visititems(_visit)

    return result


def print_hdf5_structure(
    source: str | Path,
    max_depth: int | None = None,
    include_attrs: bool = True,
) -> None:
    """Pretty-print HDF5 structure for quick adapter debugging."""

    nodes = inspect_hdf5_structure(source=source, max_depth=max_depth, include_attrs=include_attrs)
    for node in nodes:
        if node.node_type == "dataset":
            print(f"{node.path} [dataset] dtype={node.dtype} shape={node.shape}")
        else:
            print(f"{node.path} [{node.node_type}]")
        if include_attrs and node.attrs:
            print(f"  attrs={pformat(node.attrs, compact=True)}")


def inspect_rlds_structure(
    source: Iterable[Any],
    max_episodes: int = 3,
    max_steps_preview: int = 1,
) -> list[RldsSampleInfo]:
    """Inspect RLDS-like iterable structure with lightweight sampling.

    Supported source styles include:
    - iterable of dict episodes (common RLDS export form)
    - tensor-like values that provide `.numpy()` (best effort conversion)

    Args:
        source: RLDS-like iterable episode source.
        max_episodes: Max episodes to sample.
        max_steps_preview: Number of steps to inspect for key extraction.

    Returns:
        List of sampled structural summaries.
    """

    if max_episodes <= 0:
        raise ValueError("max_episodes must be > 0")
    if max_steps_preview <= 0:
        raise ValueError("max_steps_preview must be > 0")

    sampled: list[RldsSampleInfo] = []
    for episode_index, raw_episode in enumerate(source):
        if episode_index >= max_episodes:
            break

        episode = _to_python(raw_episode)
        if not isinstance(episode, Mapping):
            sampled.append(
                RldsSampleInfo(
                    episode_index=episode_index,
                    keys=[],
                    step_count=None,
                    step_keys=[],
                    extras={"episode_type": str(type(episode))},
                )
            )
            continue

        episode_keys = sorted(str(key) for key in episode.keys())
        raw_steps = episode.get("steps")
        step_count: int | None = None
        step_keys: list[str] = []

        if isinstance(raw_steps, Iterable) and not isinstance(raw_steps, (str, bytes, Mapping)):
            previewed_steps: list[Mapping[str, Any]] = []
            for step_index, raw_step in enumerate(raw_steps):
                if step_index >= max_steps_preview:
                    break
                step = _to_python(raw_step)
                if isinstance(step, Mapping):
                    previewed_steps.append(step)

            all_step_keys: set[str] = set()
            for step in previewed_steps:
                all_step_keys.update(str(key) for key in step.keys())
            step_keys = sorted(all_step_keys)

            if isinstance(raw_steps, list):
                step_count = len(raw_steps)

        extras = {
            key: _summarize_value(value)
            for key, value in episode.items()
            if key != "steps"
        }
        sampled.append(
            RldsSampleInfo(
                episode_index=episode_index,
                keys=episode_keys,
                step_count=step_count,
                step_keys=step_keys,
                extras=extras,
            )
        )

    return sampled


def print_rlds_structure(
    source: Iterable[Any],
    max_episodes: int = 3,
    max_steps_preview: int = 1,
) -> None:
    """Pretty-print sampled RLDS structure for adapter development."""

    samples = inspect_rlds_structure(
        source=source,
        max_episodes=max_episodes,
        max_steps_preview=max_steps_preview,
    )
    for sample in samples:
        print(f"episode[{sample.episode_index}] keys={sample.keys}")
        print(f"  steps.count={sample.step_count} steps.keys={sample.step_keys}")
        if sample.extras:
            print(f"  extras={pformat(sample.extras, compact=True)}")


def _to_python(value: Any) -> Any:
    """Best-effort conversion for tensor/array scalar containers."""

    if isinstance(value, Mapping):
        return {str(key): _to_python(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]

    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        try:
            arr = numpy_fn()
            tolist_fn = getattr(arr, "tolist", None)
            return tolist_fn() if callable(tolist_fn) else arr
        except Exception:
            return value

    tolist_fn = getattr(value, "tolist", None)
    if callable(tolist_fn):
        try:
            return tolist_fn()
        except Exception:
            return value

    item_fn = getattr(value, "item", None)
    if callable(item_fn):
        try:
            return item_fn()
        except Exception:
            return value

    return value


def _summarize_value(value: Any) -> Any:
    py_value = _to_python(value)
    if isinstance(py_value, Mapping):
        return {"type": "mapping", "keys": sorted(str(key) for key in py_value.keys())}
    if isinstance(py_value, list):
        return {"type": "list", "length": len(py_value)}
    return {"type": str(type(py_value)), "value": py_value}
