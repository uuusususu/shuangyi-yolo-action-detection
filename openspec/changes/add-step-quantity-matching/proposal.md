## Why

当前作业顺序状态机只能判断某一帧是否出现配置类别，无法判断该类别在同一帧中出现了几个。现场需要把“脚垫、螺丝、标签”等类别配置为带数量的步骤，只有当前单帧检测数量等于目标数量时才让该步骤 PASS，并进入下一个类别。

## What Changes

- 为每个动作步骤增加目标数量配置，默认数量为 1，保持旧配置行为兼容。
- 步骤判定从“当前帧是否包含类别”升级为“当前帧该类别数量是否等于目标数量”。
- 数量只基于当前单帧检测结果计算，不在不同帧之间累计。
- 当前步骤数量匹配并满足稳定帧条件后 PASS，然后进入下一个已配置步骤。
- 主界面步骤卡片显示当前步骤的实时数量进度，例如 `脚垫 3/4`。
- 配置页在每个步骤类别后增加数量输入；空类别步骤忽略数量。
- 保留现有错序 NG、PASS/FAIL 反馈、生产统计和模型运行时配置能力。

## Capabilities

### New Capabilities

- `step-quantity-matching`: 定义动作顺序模式下每步骤类别数量配置、单帧数量匹配、进度显示和兼容旧配置的行为。

### Modified Capabilities

无。

## Impact

- `yolo_action_detection/src/config.py`：新增步骤数量配置、加载兼容和校验。
- `yolo_action_detection/src/ui/config_page.py`：为每个动作步骤增加数量输入并保存。
- `yolo_action_detection/src/step_sequence/step_sequence_engine.py`：用当前帧类别计数替代类别集合命中，驱动步骤 PASS。
- `yolo_action_detection/src/ui/main_window.py` 与 `src/ui/widgets/native_panels.py`：显示当前数量进度。
- `yolo_action_detection/tests`：增加配置兼容、单帧数量匹配、不跨帧累计、错序兼容和 UI 构造相关测试。
