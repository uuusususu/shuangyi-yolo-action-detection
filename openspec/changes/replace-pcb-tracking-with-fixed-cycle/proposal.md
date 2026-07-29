## Why

当前第一类别区域判定会为 PCB 生成并延续业务 `track_id`。旧 PCB 离开后，新 PCB 可能因位置相近而继承旧 ID 和已完成状态，导致新产品没有按现有“完成一轮间隔”开始全新判定。现场不需要产品身份追踪，只需要保留现有的“先检测第一类别 PCB，再判断其区域内其他类别”的轮次逻辑。

## What Changes

- **BREAKING**：第一类别区域判定不再生成、匹配、续接、显示或依赖 PCB 产品追踪 ID。
- 删除该模式中的合成父类 ID、空间轨迹匹配、轨迹过期和按 ID 保存跨轮次状态。
- 保留现有第一类别父区域逻辑：先识别配置中的第一类别，再只统计该父区域内后续配置类别及数量。
- 保留现有类别稳定帧、数量判定、PASS/NG、声音、证据、界面和完成一轮间隔行为。
- 一轮完成后继续按现有间隔保持结果；间隔到期时清空上一轮判定状态，当前画面中的父类别从空状态开始下一轮。
- 不新增 PCB 数量必须为 2 的门槛，不新增左右板状态、ROI、相机裁剪、相机分辨率处理、节拍配置或结果语义。

## Capabilities

### New Capabilities

- `first-category-region-check`：定义无产品追踪 ID 的第一类别父区域判定，同时保持现有父子类别、稳定帧、结果反馈和完成一轮间隔语义。

### Modified Capabilities

无。正式规格目录尚未包含第一类别区域判定能力；本提案取代未归档变更 `add-first-category-region-check` 中的父类 ID 追踪要求，其他已经确认的第一类别区域行为保持不变。

## Impact

- `yolo_action_detection/src/pcb_inspection/`：移除父类轨迹和以 `track_id` 为键的跨轮次状态，保留父区域内子类别归属及现有轮次判定。
- `yolo_action_detection/src/ui/main_window.py`：第一类别区域模式不再读取或显示产品追踪 ID，其他显示和反馈保持现状。
- `yolo_action_detection/src/config.py` 与配置页面：继续使用现有类别、数量、稳定帧和 `round_cooldown_seconds` 配置，不新增字段。
- `yolo_action_detection/tests/`：增加新 PCB 不继承上一轮状态、无追踪 ID、父子区域逻辑保持以及间隔重置覆盖。
- 中文项目文档：删除产品 ID 追踪说明，保留第一类别区域和完成一轮间隔操作说明。
