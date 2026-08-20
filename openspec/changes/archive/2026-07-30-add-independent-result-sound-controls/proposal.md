## Why

当前程序只有一个统一的声音反馈开关，PASS 与 FAIL 会被同时启用或同时关闭，无法满足现场只保留异常报警、按需开启通过提示的使用方式。同时，首类别模式中的同一父类 ID 可以进入多轮重试，声音必须按识别轮次去重，而不能按检测帧或异常内容永久去重。

## What Changes

- 将现有统一声音反馈配置拆分为 `PASS 提示音` 与 `FAIL 提示音` 两个独立开关，并在配置保存后立即生效。
- 默认关闭 PASS 提示音、开启 FAIL 提示音。
- PASS 开关开启时，每个完成的 PASS 结果只播放一次；关闭时所有 PASS 结果保持静音。
- FAIL 开关开启时，每个完成的 FAIL 轮次只播放一次；同一轮的重复帧、重复回调或重复识别不得重复播放。
- 同一父类 ID 在 FAIL 后重新入队并进入新的识别轮次时，新轮次再次 FAIL 仍播放一次，即使异常类别和数量与上一轮相同。
- 旧配置只包含 `sound_feedback_enabled` 时，将旧值迁移为 FAIL 提示音开关，PASS 提示音保持默认关闭；保存后持久化新的独立配置字段。
- 保留现有 PASS/FAIL 声音资源和安全失败行为，不修改迈德相机 SDK、识别模型或检测稳定帧规则。

## Capabilities

### New Capabilities

- `result-sound-feedback`: 定义 PASS/FAIL 独立声音配置、默认值、旧配置迁移、配置即时生效，以及普通顺序模式和首类别多 ID 模式下按轮次一次性播放的行为。

### Modified Capabilities

无。

## Impact

- `yolo_action_detection/src/config.py`：新增并校验 PASS/FAIL 独立声音配置，兼容旧配置字段。
- `yolo_action_detection/src/ui/config_page.py`：将单一声音反馈开关替换为两个独立开关并更新说明文案。
- `yolo_action_detection/src/ui/main_window.py`：分别控制 PASS/FAIL 播放，并按唯一轮次键去重 FAIL 声音。
- `yolo_action_detection/src/pcb_inspection/`：为可重试父类 ID 的每次识别轮次提供稳定的轮次身份或等价结果事件边界，避免按异常签名跨轮次吞掉声音事件。
- `yolo_action_detection/tests/`：覆盖默认值、保存与加载、旧配置迁移、PASS/FAIL 独立开关及每轮一次播放语义。
- `yolo_action_detection/README.md`：同步配置项和声音行为说明。
