## Why

当前运行时能够输出 YOLO OBB 检测和跟踪结果，但不能把同一帧中的元器件稳定归属到多块 PCB，也不能按 PCB 跟踪 ID 独立完成缺件判定和去重。产线需要在每帧只执行一次模型推理的前提下，同时检查多块保持间隔的 PCB，并避免对已经完成判定的产品重复计数。

## What Changes

- 使用同一个 YOLO OBB 模型在完整画面中一次检测 PCB 和四种固定位置的元器件。
- 新增可选的 PCB 检查模式；未启用时保留现有作业顺序判定行为。
- 仅将 PCB 的跟踪 ID 作为产品身份；元器件不做持续跟踪，而是根据中心点、与 PCB 多边形的重叠比例及标准化槽位位置归属到对应 PCB。
- 为每个 PCB 跟踪 ID 维护独立检查状态，四个配置类别均出现在各自槽位时判定 PASS；连续三个有效检查帧均不完整时判定 FAIL。
- PCB 完成判定后标记为已处理，在该跟踪生命周期内不再重复检查或计数；多块互不接触的 PCB 可以在同一推理结果中并行更新状态。
- 增加模式开关、PCB 类别、四个元器件类别及逻辑槽位、FAIL 稳定帧数和新一轮检查间隔的可配置项，并允许操作员在配置界面填写。
- 新一轮间隔只限制结果轮次的启动与输出，不停止相机采集、PCB 跟踪或未完成 PCB 的观测更新，避免流水线目标在间隔期间丢失。
- 复用现有 PASS/FAIL 反馈、FAIL 证据和生产统计链路记录每块 PCB 的最终结果。

## Capabilities

### New Capabilities

- `multi-pcb-component-inspection`: 定义单模型单帧检测、多 PCB 跟踪、元器件空间归属、槽位完整性判定、稳定帧规则、已处理去重和可配置轮次间隔。

### Modified Capabilities

无。

## Impact

- `yolo_action_detection/src/config.py` 与 `yolo_action_detection/config/config.example.json`：新增并校验 PCB 检查配置。
- `yolo_action_detection/src/ui/config_page.py`：提供 PCB 类别、四个元器件类别/槽位、稳定帧和轮次间隔配置入口。
- `yolo_action_detection/src/yolo_runtime`：继续复用单模型、单帧推理结果，不增加第二个模型或元器件跟踪器。
- 新增独立的多 PCB 检查与空间归属模块，并由 `yolo_action_detection/src/ui/main_window.py` 接入现有推理、反馈和统计流程。
- 测试将覆盖多 PCB 空间归属、连续 FAIL、PASS、已处理去重、ID 生命周期和轮次间隔。
