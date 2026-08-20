## Context

首类别模式把 `category_names` 的第一个有效类别视为父类，把后续有效类别及 `category_counts` 视为父类内部需要完成的子步骤。每帧仍只执行一次完整画面推理，子类别通过空间关系归属到父类区域，不增加逐父类裁剪推理。

本次修正的核心边界是：父类可以多个同时在界面中，但“判定轮次”只能有一个当前 ID。父类追踪负责让每个产品有稳定身份；轮次调度负责决定当前检查哪个 ID；子步骤状态机只允许当前 `current_round_id` 推进数量、PASS 和 FAIL。FAIL 只代表当前轮次数量异常报警，不代表该父类 ID 永久结束；PASS 才代表该父类 ID 最终完成。

## Goals / Non-Goals

**Goals:**

- 仅在 `first_category_region_check_enabled` 开启时使用父类业务 ID 和轮次调度。
- 为所有可区分父类持续创建和维护稳定业务 ID。
- 按稳定出现顺序排队，同一帧稳定时按画面从左到右排序。
- 同一时间只允许一个父类 ID 统计子类别数量和更新步骤卡。
- `0/配置数量`、少于配置数量、多于配置数量都作为数量不对；连续达到 FAIL 稳定帧后触发本轮 FAIL 报警。
- FAIL 后立即结束当前轮，选择下一个等待 ID；FAIL 的父类 ID 重新排队，后续补齐后可以 PASS。
- PASS 后父类 ID 进入终局完成，不再重新识别或入队。
- FAIL 父类当前可见时持续红框，后续 PASS 后恢复为 PASS/完成颜色；旧红框不影响新活动 ID 的步骤和状态显示。
- 声音按识别轮次去重；统计和证据按同一异常签名去重/节流，避免同一 ID 同一数量错误每轮重复刷生产数据。

**Non-Goals:**

- 不改变首类别开关关闭时的普通动作顺序状态机。
- 不追踪子类别，也不使用子类别检测 ID 作为业务身份。
- 不新增第二次模型推理、父区域裁剪推理或模型训练能力。
- 不新增严格的子步骤先后顺序；配置顺序只用于展示和配置映射。
- 不修改迈德相机 SDK 包。

## Decisions

### 1. 状态分成父类追踪、轮次调度和子步骤判定

区域检查引擎维护三个层次：

- `ParentTrack`：每个父类 ID 的空间追踪、可见性、显示颜色和最近一次 FAIL 摘要。
- `RoundScheduler`：等待队列、当前轮次 ID、冷却截止时间、重试入队和 FAIL 事件去重。
- `ChildStepState`：属于某个父类 ID 的子步骤数量、连续帧和锁存状态。

这样可以同时满足“多个父类同时显示”和“完成/失败一轮后开始下一个”的要求。旧 FAIL 红框属于 `ParentTrack` 的显示状态，不属于当前活动步骤卡。

### 2. 父类生命周期使用可重试 FAIL、终局 PASS

每个父类 ID 使用以下生命周期：

`WAITING -> COOLDOWN -> ACTIVE -> PASS_LATCHED -> RETIRED`

`ACTIVE -> FAIL_RETRY_WAITING -> WAITING/COOLDOWN/ACTIVE`

任意未退休状态在连续离场超过宽限后进入 `RETIRED`。

- `WAITING`：父类已稳定出现并进入轮次队列，只追踪和绘制，不统计子类别。
- `COOLDOWN`：已被选为下一轮，但等待 `round_cooldown_seconds` 结束，期间不统计子类别。
- `ACTIVE`：当前唯一可统计子类别数量、更新步骤卡、触发 PASS/FAIL 的父类 ID。
- `FAIL_RETRY_WAITING`：该 ID 在上一轮数量异常并已报警，当前可见时红框；它不是终局，需要重新入队等待后续重试。
- `PASS_LATCHED`：该 ID 全部子步骤完成，PASS 只输出一次，不再重新入队。
- `RETIRED`：父类离场超过宽限，停止绘制和判定，业务 ID 永不复用。

父类检测带原生 `track_id` 时可作为匹配提示；没有原生 ID 时，使用 OBB IoU、中心距离和一对一约束做跨帧匹配。短暂漏检不立即退休，恢复后沿用原 ID、已完成步骤和最近 FAIL 显示状态。

### 3. 轮次队列只选择一个当前 ID

新父类稳定出现后按 `(stable_frame_index, center_x)` 加入 FIFO 等待队列。调度规则：

1. 没有当前轮次时，从等待队列取第一个仍可见或仍在漏检宽限内的 ID。
2. 若 `round_cooldown_seconds > 0`，该 ID 先进入 `COOLDOWN`；到期后仍未退休才进入 `ACTIVE`。
3. 若间隔为 0，该 ID 立即进入 `ACTIVE`。
4. 当前 ID PASS 后进入 `PASS_LATCHED`，释放当前轮次并立即预选下一个等待 ID。
5. 当前 ID FAIL 后进入 `FAIL_RETRY_WAITING`，释放当前轮次；如果该 ID 未退休且未 PASS，则排到等待队列尾部，优先让其他等待 ID 先执行。
6. 如果画面中只有这个 FAIL ID 可用，它可以在配置间隔后重新进入下一轮 ACTIVE。

反馈输出间隔只控制声音、证据、统计等外部反馈节奏；不得阻塞父类追踪、红框绘制、等待队列维护和下一 ID 预选。

### 4. 子类别只对 ACTIVE ID 计数

每帧只统计当前 `ACTIVE` 父类区域内唯一归属的子类别。其他 `WAITING`、`COOLDOWN`、`FAIL_RETRY_WAITING` 或 `PASS_LATCHED` ID 即使区域内出现子类别，也只用于画框，不推进 PASS/FAIL。

每个子步骤维护：

- `observed_count`：当前可靠帧内归属到 ACTIVE 父类的数量。
- `required_count`：配置数量。
- `match_streak`：连续正确数量帧数。
- `mismatch_streak`：连续错误数量帧数。
- `status`：`WAITING`、`MATCHING`、`COMPLETED`、`MISMATCHING`。

可靠帧更新规则：

1. 已 `COMPLETED` 的步骤不再因漏检或数量变化撤销，也不再触发 FAIL。
2. `observed_count == required_count`：`match_streak += 1`，`mismatch_streak = 0`；达到 `action_pass_stable_frames` 后锁存 `COMPLETED`。
3. `observed_count != required_count`：`mismatch_streak += 1`，`match_streak = 0`；这里包括 `0/4`、少于配置数量和多于配置数量。
4. `mismatch_streak >= action_ng_stable_frames`：该父类产生本轮 FAIL，记录异常类别、实际数量和配置数量；该未完成步骤的错误连续帧在重试前清零，避免下一轮刚开始立即沿用旧连续帧失败。
5. 归属歧义、父类不完整可见或当前 ID 不可靠时，该帧冻结连续计数。

全部有效子步骤均为 `COMPLETED` 时，当前父类进入 `PASS_LATCHED`。如果某 ID 曾经 FAIL，但后续重试中补齐并满足 PASS 稳定帧，仍然可以 PASS。

### 5. FAIL 事件去重与红框显示分离

FAIL 报警分为两层：

- 显示层：只要父类 ID 处于 `FAIL_RETRY_WAITING` 或携带未清除的 fail 摘要，当前帧可见时使用红色绘制父类框；当前帧不可见时不画陈旧框；后续 PASS 后改为 PASS/完成颜色。
- 事件层：声音按识别轮次去重；统计、证据保存、NG 记录按异常签名去重/节流。

建议异常签名为：

`(parent_id, step_key, observed_count, required_count)`

同一签名已上报过时，后续重试再次出现相同错误只保持红框和状态，不重复增加统计或保存证据；声音是否播放由结果声音反馈能力按识别轮次决定。若实际数量变化、异常步骤变化、父类重新进入新生命周期，或该 ID 曾经补齐后又发生新异常，则允许产生新的统计/证据事件。

### 6. 红框和步骤卡必须分离

画面叠加层可以同时绘制所有当前可见父类：

- `WAITING`/`COOLDOWN`/`ACTIVE` 使用正常观察色，除非该 ID 带有未清除 fail 摘要，此时可继续红框提示。
- `FAIL_RETRY_WAITING` 当前可见时使用红色。
- `PASS_LATCHED` 当前可见时使用 PASS 色。
- 当前帧未识别到的父类不绘制陈旧多边形；漏检宽限内恢复同 ID 时继续原显示状态。

右侧步骤卡和顶部状态只显示当前轮次 ID。旧 FAIL ID 仍可在画面中红框显示，但不能把步骤卡切回旧 ID，也不能把旧 ID 的 FAIL 状态覆盖新 ID 的数量判断。

### 7. UI 绑定规则必须显式化

引擎向 UI 输出：

- `parents`: 当前帧可绘制父类列表，包含业务 ID、OBB、生命周期状态和最近异常摘要。
- `current_round_id`: 当前 `COOLDOWN` 或 `ACTIVE` 的父类 ID，若没有则为空。
- `current_step_states`: 只属于 `current_round_id` 的步骤状态和数量。
- `event_queue`: 本帧新产生的 PASS/FAIL 轮次事件；声音使用轮次身份去重，统计和证据使用异常签名去重。

UI 禁止从原始检测列表自行选择详情 ID，尤其不能用 `max(track_id)`、上一帧 `last_slot_states` 或最新红框来决定步骤卡对象。

## Risks / Trade-offs

- [FAIL 后同 ID 马上重复 FAIL] -> FAIL 后清零本轮错误连续帧，并把 ID 放到队尾；下一轮必须重新累计 `action_ng_stable_frames`。
- [同一错误每轮重复刷 NG] -> 声音按轮次去重，统计和证据按异常签名去重/节流，红框持续但生产数据不重复。
- [0/4 太快误判] -> 通过 `action_ng_stable_frames` 控制，当前需求按配置帧数执行；歧义帧冻结计数。
- [旧 FAIL 红框误导当前步骤] -> 叠加层和详情区分离，右侧始终绑定 `current_round_id`。
- [父类短暂漏检导致换 ID] -> 使用漏检宽限和空间匹配恢复；退休后才允许新 ID。
- [FAIL 后下一轮节奏不一致] -> 调度器负责立即预选，`round_cooldown_seconds` 只控制新 ID 开始计数时间。

## Migration Plan

1. 先补充失败测试，固定 FAIL 可重试、FAIL 后排下一个 ID、FAIL ID 重新入队、补齐后 PASS、事件去重和旧红框不污染新步骤卡。
2. 重构区域检查引擎，把父类追踪、轮次调度、子步骤判定和 FAIL 显示状态拆成明确状态。
3. 调整 UI 数据合同，让顶部状态、步骤卡和数量只读 `current_round_id`。
4. 更新配置说明和 README，说明首类别模式的轮次、FAIL 可重试、PASS 终局、红框恢复和间隔语义。
5. 运行定向 pytest、完整测试、OpenSpec 严格校验、静态图片验证；如果进入打包阶段，再运行打包和 packaged smoke。

## Open Questions

无。当前提案按“首类别模式下串行判定父类 ID；连续配置帧数数量不对即本轮 FAIL；FAIL 后当前 ID 放回队列并立即调度下一个 ID；同 ID 后续补齐后可以 PASS；PASS 才终局”执行。
