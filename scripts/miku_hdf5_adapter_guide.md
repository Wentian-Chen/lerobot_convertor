# Miku HDF5 -> LeRobot 转换流程说明

> 本脚本**复用** [src/lerobot_converter/hdf5_adapter.py](src/lerobot_converter/hdf5_adapter.py) 的标准流程。
> 你只需要在子类里实现 `extract_episode_from_file(...)`，其余 episode 遍历、frame 校验、save/finalize 都由基类处理。
> 运行参数规范定义在 [src/lerobot_converter/models.py](src/lerobot_converter/models.py) 的 `Hdf5AdapterRunConfig` 与 `ConversionOptions`。

## 1) 数据结构约定（输入）
每个 HDF5 文件代表一条 episode，关键字段：
- `/cam_head/color`: `(T, H, W, 3)`
- `/cam_wrist/color`: `(T, H, W, 3)`
- `/left_arm/joint`: `(T, 6)`
- `/left_arm/qpos`: `(T, 6)`
- `/left_arm/gripper`: `(T,)`
- 时间戳优先级：`/left_arm/timestamp` > `/cam_head/timestamp` > `/cam_wrist/timestamp`

## 2) Episode 读取（由基类处理）
- 读取入口在 `Hdf5ToLeRobotConverter.iter_source_episodes(...)`：
  - `source` 可以是单个 `.h5/.hdf5` 文件或目录；
  - 若是目录会自动遍历文件；
  - 每个文件会调用一次你覆写的 `extract_episode_from_file(...)`。
- 你无需再写 `iter_source_episodes`，只需实现提取函数。

## 3) 用户负责对齐阶段（extract_episode_from_file）
- 这是**用户自定义提取逻辑**的核心阶段：
  - 用户读取自己的 HDF5 字段；
  - 用户将每步数据组装为 `feature_values` 字典；
  - `feature_values` 的 key 必须与 `run_miku_hdf5_adapter` 中的 `feature` 完全一致。
- 当前示例实现包含：
  - `observation.state = concat(left_arm/qpos, left_arm/gripper)`（7 维，`float64`）
  - `action = concat(left_arm/joint, left_arm/gripper)`（7 维，`float64`）
  - `cam_head/color -> observation.images.image`（`HWC -> CHW`）
  - `cam_wrist/color -> observation.images.wrist_image`（`HWC -> CHW`）
- 每一步都会检查：`options.features` 中声明的 key 是否全部被填充；若缺失直接报错。

### 3.1 指令增强（augment_task_instruction）
- 在 `ConversionOptions` 中通过 `augment_task_instruction` 控制是否启用指令增强。
- 当 `augment_task_instruction=True`：
  - 在 `cfg.source` 根目录读取 `tasks_instruction.json`；
  - 每个 episode 随机选择一条 instruction，作为该 episode 所有帧的 `task`。
- 当 `augment_task_instruction=False`：
  - 使用 `default_task` 作为该 episode 的 `task`。

`tasks_instruction.json` 支持两种格式：
- 纯数组：`["instruction A", "instruction B"]`
- 对象：`{"tasks": [...]}` 或 `{"task_instructions": [...]}`

你应当修改的位置：
- 在 [lc_scripts/miku_hdf5_adapter.py](lc_scripts/miku_hdf5_adapter.py) 里改 `extract_episode_from_file(...)`，完成“读取源数据 -> 组装 step.feature_values”。
- 在同文件 `run_miku_hdf5_adapter(...)` 的 `feature` 字典里定义目标 schema。

## 4) 标准 frame 映射（build_frame）
- `build_frame` 直接读取 `source_frame["feature_values"]`，不再做 observation/action 二次归一化。
- `build_frame` 会检查当前 frame 是否覆盖 `options.features` 全部 key。
- 这保证了“用户在 `_extract_single_file_episode` 对齐一次，后续直接写入”。

## 5) feature 如何对齐（必须一一对应）
规则：`feature` 里每个 key，都必须在每个 step 的 `feature_values` 出现。

以本脚本内置 `feature` 为例：
- `observation.images.image` <- `/cam_head/color[idx]`，并执行 `HWC -> CHW`
- `observation.images.wrist_image` <- `/cam_wrist/color[idx]`，并执行 `HWC -> CHW`
- `observation.state` <- `concat(left_arm/qpos[idx], left_arm/gripper[idx])`
- `action` <- `concat(left_arm/joint[idx], left_arm/gripper[idx])`

若你新增 feature（例如 `next.reward`），必须在 `feature` 字典新增该 key，并在 `feature_values` 同步填充。

## 6) 写入 LeRobotDataset（save episode）
- `convert()` 中逐 episode 调用 `convert_episode()`。
- 每个 frame 经 key 校验后写入 `add_frame()`。
- 每个 episode 完成后调用 `save_episode()`。
- 全部结束后调用 `finalize_target()` 生成元数据与统计。

## 7) 与指定 feature 的对齐点
已对齐以下 feature 约束：
- `observation.images.image`: `image`, `shape=[3,480,640]`, `names=[channels,height,width]`
- `observation.images.wrist_image`: 同上
- `observation.state`: `float64`, `shape=[7]`
- `action`: `float64`, `shape=[7]`

## 8) 运行方式
```bash
python lc_scripts/miku_hdf5_adapter.py \
  --source ./datasets/miku112/piper_pick_banana_100 \
  --output_dir ./datasets/miku112/piper_pick_banana_100_v1
```

启用指令增强示例：
```bash
python lc_scripts/miku_hdf5_adapter.py \
  --source ./datasets/miku112/piper_pick_banana_100 \
  --output_dir ./datasets/miku112/piper_pick_banana_100_v1 \
  --augment_task_instruction true
```

> 若 `features` 未显式传入，脚本会使用 `run_miku_hdf5_adapter` 中内置的 `feature` 字典。
