## 1. 测试基线与失败用例

- [x] 1.1 梳理当前 `add-first-category-region-check` 已有实现和测试，确认终局 `NG_LATCHED`、同 ID 不可恢复等行为与新提案冲突，不回退无关用户修改。
- [x] 1.2 补充引擎失败测试：当前 ID 数量不对达到 `action_ng_stable_frames` 后只产生本轮 FAIL，并进入可重试等待状态。
- [x] 1.3 补充引擎失败测试：FAIL 后已完成子步骤保持完成，未完成步骤的错误连续帧清零，下一轮必须重新累计。
- [x] 1.4 补充引擎失败测试：同一父类 ID FAIL 后补齐配件，重新轮到该 ID 时可达到 `action_pass_stable_frames` 并转为 PASS。
- [x] 1.5 补充调度失败测试：FAIL 后立即释放当前轮次并选择下一个等待 ID，FAIL ID 进入队尾；没有其他 ID 时按 `round_cooldown_seconds` 重试自己。
- [x] 1.6 补充事件失败测试：同一 ID、同一步骤、同一实际数量/配置数量的 FAIL 签名不重复刷声音、统计或证据；异常签名变化时允许新事件。
- [x] 1.7 补充 UI 失败测试：FAIL_RETRY ID 当前可见时红框，当前帧识别不到时不画陈旧框；旧红框不覆盖当前轮次步骤卡；后续 PASS 后颜色恢复为 PASS/完成。

## 2. 父类追踪与轮次调度

- [x] 2.1 将父类业务 ID、可见性、离场宽限、最近 FAIL 摘要、PASS 终局状态集中到 `ParentTrack`/等价结构。
- [x] 2.2 将状态机从终局 `NG_LATCHED` 改为可重试 `FAIL_RETRY_WAITING` 或等价状态，保留 `PASS_LATCHED` 作为终局完成。
- [x] 2.3 实现 FAIL 后当前 ID 释放、重新入队到队尾、优先调度其他等待 ID；按 `round_cooldown_seconds` 控制下一轮开始统计时间。
- [x] 2.4 确保 `WAITING`、`COOLDOWN`、`FAIL_RETRY_WAITING` ID 只追踪和绘制，不累计子类别 PASS/FAIL。
- [x] 2.5 确保 `PASS_LATCHED` ID 不重新入队；离场退休后新实例获得新业务 ID，不复用旧生命周期 ID。

## 3. 子步骤数量判定与 FAIL 去重

- [x] 3.1 只对当前 `ACTIVE` 父类区域内唯一归属的子类别计数，非当前轮次父类区域内的子类别不推进判定。
- [x] 3.2 将 `observed_count != required_count` 统一作为错误数量，包括 `0/配置数量`、少于配置数量和多于配置数量。
- [x] 3.3 正确数量连续达到 `action_pass_stable_frames` 后锁存子步骤完成，后续漏检不撤销。
- [x] 3.4 错误数量连续达到 `action_ng_stable_frames` 后产生本轮 FAIL，记录类别、实际数量和配置数量，但不把该父类 ID 终局报废。
- [x] 3.5 FAIL 后保留已完成步骤，清零未完成步骤的本轮错误连续帧，允许后续重试补齐后 PASS。
- [x] 3.6 实现 FAIL 事件签名去重/节流，避免同一 ID 同一异常每轮重复触发外部 NG 反馈。

## 4. UI 状态绑定与绘制

- [x] 4.1 调整引擎输出，显式提供 `parents`、`current_round_id`、`current_step_states` 和按异常签名去重后的 PASS/FAIL 事件。
- [x] 4.2 主画面按父类 ID 绘制所有当前可见父类，FAIL_RETRY 或携带未清除 FAIL 摘要的 ID 当前可见时持续红框。
- [x] 4.3 当前帧未检测到父类时不绘制上一帧陈旧红框，短暂漏检恢复同 ID 后恢复对应显示状态。
- [x] 4.4 顶部状态、步骤卡和数量显示只绑定 `current_round_id`，移除按最大 ID、原始检测顺序或旧缓存选择详情对象的逻辑。
- [x] 4.5 步骤卡在当前轮次 FAIL 后将异常步骤显示为红色，并展示实际数量/配置数量；切到新 ID 后不得继续显示旧 ID 的步骤状态。
- [x] 4.6 同一 ID 后续 PASS 后，将父类框从红色恢复为 PASS/完成颜色。

## 5. 文档、配置与兼容

- [x] 5.1 校验首类别模式启用时必须有父类和至少一个子步骤，保持旧 `pcb_*` 字段加载兼容。
- [x] 5.2 更新配置页文案，说明“类别 1 为父类，后续类别在父类区域内按数量完成；FAIL 为本轮报警，补齐后可重试 PASS”。
- [x] 5.3 更新 `yolo_action_detection/README.md`，说明串行 ID 轮次、0/4 FAIL、FAIL 可重试、PASS 终局、红框恢复和间隔启动下一轮。
- [x] 5.4 不修改迈德相机 SDK 包。

## 6. 验证与交付

- [x] 6.1 运行区域检查引擎定向 pytest，覆盖 FAIL 可重试、补齐后 PASS、事件去重和调度队列。
- [x] 6.2 运行 UI 绑定和步骤卡定向 pytest。
- [x] 6.3 使用 `test.jpg`、生产配置和 `sanrepian2.onnx` 做静态图片验证，确认只对当前 ID 统计后续类别，FAIL 后可重试。
- [x] 6.4 运行完整 `yolo_action_detection/tests`，确认普通顺序模式和首类别模式都无回归。
- [x] 6.5 运行 `openspec validate add-first-category-region-check --strict`。
- [x] 6.6 运行 `packaging/build_yolo.bat` 和打包后 smoke test。
