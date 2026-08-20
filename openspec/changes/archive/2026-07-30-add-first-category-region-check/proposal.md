## Why

当前“首类别区域检查”的核心问题不是检测框本身，而是判定对象必须绑定到唯一的父类 ID，并且 FAIL 不能被当成该 ID 的最终结束。

现场语义已经明确：开启首类别检测后，先稳定识别第一个类别作为父类实例，再只对当前轮次的这个父类 ID 匹配后续类别数量；完成一个 ID 后开始下一个 ID。某个父类 ID 如果本轮后续类别数量连续不对，应当红框报警并结束本轮，但该 ID 不能终局报废；它需要重新进入等待/重试状态，后续补齐配件后仍然可以 PASS。只有 PASS 才是该父类 ID 的最终完成状态。

因此本变更需要把“父类追踪”“轮次调度”“子类别数量判定”“FAIL 报警显示”拆开：所有父类可以同时追踪和显示，但同一时间只有一个 `current_round_id` 能推进步骤、累计 PASS/FAIL 稳定帧和更新右侧步骤卡；FAIL 后该 ID 红框显示并重新排队，不能阻塞其他 ID 的后续识别。

## What Changes

- 仅在启用 `first_category_region_check_enabled` 后启用本变更；关闭时继续使用普通动作顺序检测。
- 使用第一个有效配置类别作为父类，后续有效类别及其配置数量作为该父类内的子步骤。
- 为画面中的每个父类实例创建稳定业务 ID；模型没有原生 `track_id` 时由区域检查引擎按空间匹配分配合成 ID。
- 所有父类 ID 都持续追踪和绘制，但只允许一个当前轮次 ID 进入 `ACTIVE` 状态并统计子类别数量。
- 父类 ID 按“最先稳定出现”进入等待队列；同一帧稳定出现时按画面从左到右排序。
- 当前轮次 ID 的子步骤数量每帧按父类区域内归属结果计算；`0/配置数量`、少于配置数量、多于配置数量都属于数量不对，需要累计 FAIL 连续帧。
- 子步骤数量连续正确达到 `action_pass_stable_frames` 后锁存完成；完成后后续漏检或短暂误检不撤销。
- 未完成子步骤数量连续不等于配置值达到 `action_ng_stable_frames` 后，当前父类 ID 产生一次本轮 FAIL/NG 报警，父类框和异常步骤变红。
- FAIL/NG 不是终局：本轮结束后，该父类 ID 进入可重试等待状态并排到队尾；后续再次轮到它时继续判断未完成步骤，补齐后可以转为 PASS。
- PASS 才是终局：父类 ID 全部子步骤完成并输出 PASS 后，不再重新入队或重新判定。
- FAIL 后立即释放当前轮次并选择下一个等待 ID；被 FAIL 的旧 ID 不能阻塞新 ID。下一轮开始统计的时间仍由 `round_cooldown_seconds` 控制，间隔为 0 时立即开始。
- FAIL 红框只画当前可见或可靠恢复的父类框；当前帧识别不到时不画陈旧框。该 ID 后续 PASS 后从红框恢复为 PASS/正常完成颜色。
- FAIL 反馈需要分层去重：声音按识别轮次去重，每个完成的 FAIL 轮次最多播放一次；统计和证据仍按同一父类 ID、同一子步骤、同一实际数量/配置数量的异常签名去重，避免重复刷生产数据。
- UI 顶部状态、步骤卡、数量统计必须绑定同一个当前轮次 ID，禁止使用 `max(track_id)`、上一帧缓存或最新红框来选择详情对象。

## Capabilities

### New Capabilities

- `first-category-region-check`: 定义首类别开关、父类业务 ID、FIFO 轮次调度、单 `current_round_id` 子类别判定、0 数量 FAIL 累计、FAIL 可重试、PASS 终局、FAIL 事件去重、红框可恢复和按间隔启动下一轮。

### Modified Capabilities

无。

## Impact

- `yolo_action_detection/src/pcb_inspection/`：父类追踪、轮次队列、单活动 ID 状态机、子步骤锁存、可重试 FAIL 判定和 FAIL 事件去重。
- `yolo_action_detection/src/ui/main_window.py`：画面叠加层绘制所有父类 ID；FAIL_RETRY/异常父类当前可见时红框；顶部状态和步骤卡只绑定当前轮次 ID。
- `yolo_action_detection/src/ui/widgets/native_panels.py`：步骤卡显示当前轮次 ID 的数量、完成和 FAIL 状态，当前轮次异常步骤变红；旧 FAIL 红框不覆盖当前步骤卡。
- `yolo_action_detection/src/config.py`：复用并校验 `action_pass_stable_frames`、`action_ng_stable_frames`、`round_cooldown_seconds` 等稳定帧/间隔配置。
- `yolo_action_detection/README.md`：说明首类别模式、业务 ID、串行轮次、0/4 FAIL、FAIL 可重试、PASS 终局、红框可恢复和离场换 ID 语义。
- `yolo_action_detection/tests/`：覆盖多父类追踪但单 ID 判定、0/4 连续 FAIL、FAIL 后立即排下一个 ID、FAIL ID 重新入队、补齐后 PASS、旧红框不污染新 ID 步骤卡。
